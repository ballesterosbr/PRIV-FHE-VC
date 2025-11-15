import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from medmnist import INFO
import medmnist
from concrete.ml.torch.compile import compile_torch_model
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import warnings
import csv
import os
import gc

warnings.filterwarnings('ignore')

# ===========================
# GENERAL CONFIGURATION
# ===========================
DATA_FLAG = 'pathmnist'
SAVE_PATH = 'results_64x64_rgb_optimized.csv'
NUM_FHE_SAMPLES = 100
BATCH_SIZE = 16
SUBSET_SIZE = 13000
EPOCHS = 30
LR = 5e-4

# ===========================
# DATASET WITH PER-CHANNEL NORMALIZATION
# ===========================
info = INFO[DATA_FLAG]
DataClass = getattr(medmnist, info['python_class'])
num_classes = len(info['label'])
labels_map = info['label']

# Channel-specific normalization for RGB
# Values calculated from PathMNIST dataset or using medical standards
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64)),
    transforms.Lambda(lambda x: x.float()),
])

def load_pathmnist(batch_size=BATCH_SIZE, subset_size=SUBSET_SIZE):
    train_dataset = DataClass(split='train', transform=transform, download=True, size=64)
    val_dataset = DataClass(split='val', transform=transform, download=True, size=64)
    test_dataset = DataClass(split='test', transform=transform, download=True, size=64)
    
    train_dataset = Subset(train_dataset, range(min(subset_size, len(train_dataset))))
    val_dataset = Subset(val_dataset, range(min(subset_size // 4, len(val_dataset))))
    test_dataset = Subset(test_dataset, range(min(subset_size // 4, len(test_dataset))))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader

# ===========================
# RGB MODEL OPTIMIZED FOR FHE
# ===========================
class cnn64_rgb_optimized(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        
        self.features = nn.Sequential(
            # MORE filters for RGB (20 vs 12)
            nn.Conv2d(3, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            # 1x1 layer to merge RGB correlations
            nn.Conv2d(20, 20, kernel_size=1),
            nn.ReLU(),
            nn.AvgPool2d(2),  # -> 32x32
            
            nn.Conv2d(20, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),  # -> 16x16
            
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(4),  # -> 4x4
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 4 * 4, 96),
            nn.ReLU(),
            nn.Linear(96, num_classes),
        )    
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ===========================
# TRAINING AND EVALUATION
# ===========================
def train(model, train_loader, val_loader, epochs=EPOCHS, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.to(device)
    
    best_f1 = 0.0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_preds, train_labels = [], []
        
        for data, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data, targets = data.to(device), targets.squeeze().long().to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(targets.detach().cpu().numpy())
        
        val_acc, val_f1 = evaluate_model(model, val_loader, device)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        
        print(
            f"Epoch {epoch+1}/{epochs}: Loss: {total_loss/len(train_loader):.4f}, "
            f"Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f} | "
            f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
        )
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best model loaded with F1: {best_f1:.4f}")

def evaluate_model(model, loader, device='cuda'):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.squeeze().long().to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(targets.detach().cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    return accuracy, f1_macro

# ===========================
# CSV UTILITIES
# ===========================
def ensure_csv_header(save_path: str = SAVE_PATH):
    if not os.path.isfile(save_path):
        with open(save_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'n_bits', 'rounding_threshold', 'p_error',
                'image_idx', 'time_seconds', 'prediction', 'true_label'
            ])

def cfg_key(n_bits, rounding_threshold, p_error):
    return (str(int(n_bits)), str(int(rounding_threshold)), str(p_error))

def read_progress(save_path: str = SAVE_PATH):
    progress = {}
    if not os.path.isfile(save_path):
        return progress
    
    with open(save_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 7:
                continue
            n_bits_s, rt_s, pe_s, idx_s = row[0], row[1], row[2], row[3]
            key = (n_bits_s, rt_s, pe_s)
            
            if key not in progress:
                progress[key] = {'done_indices': set(), 'has_global': False}
            
            if idx_s == 'global':
                progress[key]['has_global'] = True
            else:
                try:
                    progress[key]['done_indices'].add(int(idx_s))
                except ValueError:
                    pass
    return progress

def compute_and_write_global_if_complete(n_bits, rounding_threshold, p_error, total_samples, save_path):
    key = cfg_key(n_bits, rounding_threshold, p_error)
    all_progress = read_progress(save_path)
    cfg_prog = all_progress.get(key, {'done_indices': set(), 'has_global': False})
    
    if cfg_prog.get('has_global', False):
        return False
    
    done_cnt = len(cfg_prog['done_indices'])
    if done_cnt < total_samples:
        return False
    
    preds_list, labels_list, times_list = [], [], []
    with open(save_path, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or len(row) < 7:
                continue
            nb_s, rt_s, pe_s, idx_s, time_s, pred_s, label_s = row
            if (nb_s, rt_s, pe_s) != key:
                continue
            if idx_s == 'global':
                continue
            try:
                preds_list.append(int(pred_s))
                labels_list.append(int(label_s))
                times_list.append(float(time_s))
            except ValueError:
                pass
    
    if len(preds_list) < total_samples:
        return False
    
    acc = accuracy_score(labels_list, preds_list)
    f1 = f1_score(labels_list, preds_list, average='macro')
    avg_time = np.mean(times_list)
    
    with open(save_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([n_bits, rounding_threshold, p_error, 'global', avg_time, f1, acc])
    
    print(f"[GLOBAL] cfg=({n_bits},{rounding_threshold},{p_error}) -> Acc={acc:.4f}, F1={f1:.4f}, T={avg_time:.2f}s")
    return True

# ===========================
# FHE COMPILATION
# ===========================
def compile_fhe_model(model, train_dataset, n_bits=7, rounding_threshold_bits=4, p_error=0.1):
    try:
        model.eval()
        model.cpu()
        
        num_samples = min(1000, len(train_dataset))
        x_calib = []
        for i in range(num_samples):
            sample, _ = train_dataset[i]
            x_calib.append(sample.numpy())
        x_calib = np.array(x_calib, dtype=np.float32)
        
        print(f"Compiling RGB with n_bits={n_bits}, rounding={rounding_threshold_bits}, p_error={p_error}")
        q_model = compile_torch_model(
            model,
            x_calib,
            n_bits=n_bits,
            rounding_threshold_bits=rounding_threshold_bits,
            p_error=p_error,
        )
        print('RGB model compiled successfully')
        return q_model
    except Exception as e:
        print(f"FHE compilation error: {e}")
        return None

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print('Memory cleared')

# ===========================
# FHE INFERENCE
# ===========================
def fhe_inference(
    q_model,
    test_dataset,
    n_bits,
    rounding_threshold,
    p_error,
    num_samples: int,
    max_seconds_per_sample: int = 86400,
    save_path: str = SAVE_PATH,
):
    ensure_csv_header(save_path)
    key = cfg_key(n_bits, rounding_threshold, p_error)
    
    progress = read_progress(save_path)
    done = progress.get(key, {'done_indices': set(), 'has_global': False})['done_indices']
    
    print(
        f"\nFHE Inference cfg=({n_bits},{rounding_threshold},{p_error}) | "
        f"completed: {len(done)}/{num_samples}"
    )
    
    for i in tqdm(range(num_samples), desc=f"FHE n_bits={n_bits} rt={rounding_threshold} p_error={p_error}"):
        if i in done:
            continue
        
        sample, label = test_dataset[i]
        float_input = sample.unsqueeze(0).numpy().astype(np.float32)
        
        start_time = time.time()
        try:
            result = q_model.forward(float_input, fhe='execute')
            pred = int(np.argmax(result))
            elapsed = time.time() - start_time
        except Exception as e:
            print(f"Error in sample {i}: {e}")
            pred = -1
            elapsed = float(max_seconds_per_sample)
        
        with open(save_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([n_bits, rounding_threshold, p_error, i, elapsed, pred, int(label.item())])
        
        if pred != -1:
            print(f"FHE pred: {pred} | true: {int(label.item())} | {elapsed:.2f}s")
        else:
            print(f"Failed idx={i} | {elapsed:.2f}s")
    
    compute_and_write_global_if_complete(n_bits, rounding_threshold, p_error, num_samples, save_path)

# ===========================
# MAIN
# ===========================
if __name__ == '__main__':
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device:', device)
        print('--- RGB OPTIMIZED CNN for PathMNIST + FHE ---')
        
        train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = load_pathmnist(
            batch_size=BATCH_SIZE, subset_size=SUBSET_SIZE
        )
        
        model = cnn64_rgb_optimized(num_classes=num_classes)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        train(model, train_loader, val_loader, epochs=EPOCHS, device=device)
        
        acc, f1 = evaluate_model(model, test_loader, device=device)
        print(f"\nStandard RGB evaluation | Acc: {acc:.4f} | F1: {f1:.4f}")
        
        # ADJUSTED CONFIGURATIONS FOR RGB + FHE
        # RGB needs more bits or less rounding to compensate for 3x inputs
        configurations = [
            # Conservative (more bits, less rounding)
            # (7, 4, 0.1),    # MORE bits than grayscale (6→7)
            # (7, 6, 0.05),   # RGB-FHE balance
            
            # Aggressive
            # (8, 4, 0.15),
            (8, 6, 0.008),
        ]
        
        ensure_csv_header(SAVE_PATH)
        all_progress = read_progress(SAVE_PATH)
        
        print(f"\n=== RGB FHE EXPERIMENTS ===")
        print(f"Configurations: {len(configurations)}")
        print(f"Samples: {NUM_FHE_SAMPLES}")
        
        for i, (n_bits, rounding_threshold, p_error) in enumerate(configurations, start=1):
            key = cfg_key(n_bits, rounding_threshold, p_error)
            cfg_prog = all_progress.get(key, {'done_indices': set(), 'has_global': False})
            
            if cfg_prog.get('has_global', False):
                print(f"\n=== EXP {i}/{len(configurations)} ===")
                print(f"cfg=({n_bits},{rounding_threshold},{p_error}) ALREADY COMPLETE")
                continue
            
            done_cnt = len(cfg_prog.get('done_indices', set()))
            if done_cnt >= NUM_FHE_SAMPLES:
                wrote = compute_and_write_global_if_complete(
                    n_bits, rounding_threshold, p_error, NUM_FHE_SAMPLES, SAVE_PATH
                )
                print(f"\n=== EXP {i}/{len(configurations)} ===")
                if wrote:
                    print(f"cfg=({n_bits},{rounding_threshold},{p_error}) COMPLETE")
                else:
                    print(f"cfg=({n_bits},{rounding_threshold},{p_error}) skipping")
                continue
            
            print(f"\n=== EXP {i}/{len(configurations)} ===")
            print(f"cfg: n_bits={n_bits}, rt={rounding_threshold}, p_error={p_error}")
            print(f"Resuming: {done_cnt}/{NUM_FHE_SAMPLES}")
            
            q_model = compile_fhe_model(
                model,
                train_dataset,
                n_bits=n_bits,
                rounding_threshold_bits=rounding_threshold,
                p_error=p_error,
            )
            
            if q_model is not None:
                fhe_inference(
                    q_model,
                    test_dataset,
                    n_bits,
                    rounding_threshold,
                    p_error,
                    num_samples=NUM_FHE_SAMPLES,
                    save_path=SAVE_PATH,
                )
            else:
                print(f"COMPILATION FAILED cfg=({n_bits},{rounding_threshold},{p_error})")
                with open(SAVE_PATH, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([n_bits, rounding_threshold, p_error, 'compilation_failed', 0, -1, -1])
            
            if 'q_model' in locals():
                del q_model
            clear_memory()
            print(f"Exp {i} completed\n")
        
        print("=== ALL EXPERIMENTS COMPLETED ===")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()