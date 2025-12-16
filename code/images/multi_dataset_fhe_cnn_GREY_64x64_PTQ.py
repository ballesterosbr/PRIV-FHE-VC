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
AVAILABLE_DATASETS = {
    'pathmnist': {'name': 'PathMNIST', 'classes': 9, 'epochs': 20},
    'dermamnist': {'name': 'DermaMNIST', 'classes': 7, 'epochs': 20},
    'retinamnist': {'name': 'RetinaMNIST', 'classes': 5, 'epochs': 15},
    'bloodmnist': {'name': 'BloodMNIST', 'classes': 8, 'epochs': 20},
}

IMG_SIZE = 64
BATCH_SIZE = 16
LR = 5e-4
NUM_FHE_SAMPLES = 50
P_ERROR = 0.008
EARLY_STOPPING_PATIENCE = 5

# Automatically generate FHE configurations
FHE_CONFIGURATIONS = []
for n_bits in range(5, 9):  # 5, 6, 7, 8
    for rounding in range(5, 7):  # 5, 6
        FHE_CONFIGURATIONS.append((n_bits, rounding, P_ERROR))

print(f"Generated FHE configurations: {len(FHE_CONFIGURATIONS)}")
for cfg in FHE_CONFIGURATIONS:
    print(f"  n_bits={cfg[0]}, rounding={cfg[1]}, p_error={cfg[2]}")

# ===========================
# CNN64 RGB MODEL (PTQ)
# ===========================
class cnn64_rgb_ptq(nn.Module):
    """
    CNN for 64x64 RGB images - Post Training Quantization
    Adapted to process 3 channels (RGB) instead of 1 (grayscale)
    """
    def __init__(self, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: RGB (3 channels) -> 16 features
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 64x64x3 -> 64x64x16
            nn.ReLU(),
            nn.AvgPool2d(2),  # -> 32x32x16
            
            # Block 2: 16 -> 32 features
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32x32x16 -> 32x32x32
            nn.ReLU(),
            nn.AvgPool2d(2),  # -> 16x16x32
            
            # Block 3: 32 -> 48 features (double layer)
            nn.Conv2d(32, 48, kernel_size=3, padding=1),  # 16x16x32 -> 16x16x48
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),  # 16x16x48 -> 16x16x48
            nn.ReLU(),
            nn.AvgPool2d(4),  # -> 4x4x48
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 4 * 4, 96),  # 768 -> 96
            nn.ReLU(),
            nn.Linear(96, num_classes),  # 96 -> num_classes
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ===========================
# DATASET
# ===========================
def get_dataset_info(data_flag):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    num_classes = len(info['label'])
    labels_map = info['label']
    
    print(f"\n{'='*60}")
    print(f"Dataset: {AVAILABLE_DATASETS[data_flag]['name']}")
    print(f"Number of classes: {num_classes}")
    print(f"Labels: {labels_map}")
    print(f"{'='*60}\n")
    
    return info, DataClass, num_classes, labels_map

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda x: x.float()),
])

def load_dataset(data_flag, batch_size=BATCH_SIZE):
    info, DataClass, num_classes, labels_map = get_dataset_info(data_flag)
    
    train_dataset = DataClass(split='train', transform=transform, download=True, size=IMG_SIZE)
    val_dataset = DataClass(split='val', transform=transform, download=True, size=IMG_SIZE)
    test_dataset = DataClass(split='test', transform=transform, download=True, size=IMG_SIZE)
    
    # Verify it's RGB (3 channels)
    sample_img, _ = train_dataset[0]
    if sample_img.shape[0] != 3:
        raise ValueError(f"ERROR: {data_flag} is not RGB! It has {sample_img.shape[0]} channels (expected 3)")
    
    print(f"Dataset verified: {sample_img.shape[0]} channels (RGB)")
    print(f"  Train: {len(train_dataset)} samples (100%)")
    print(f"  Val: {len(val_dataset)} samples (100%)")
    print(f"  Test: {len(test_dataset)} samples (100%)")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, num_classes

# ===========================
# TRAINING AND EVALUATION
# ===========================
def train(model, train_loader, val_loader, epochs, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.to(device)
    
    best_f1 = 0.0
    best_state = None
    epochs_without_improvement = 0
    
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
        
        # Early stopping check
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            print(f"New best model (F1: {best_f1:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement ({epochs_without_improvement}/{EARLY_STOPPING_PATIENCE})")
        
        # Early stopping
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping activated at epoch {epoch+1}")
            print(f"Best F1: {best_f1:.4f} ({epochs_without_improvement} epochs ago)")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best model loaded with F1: {best_f1:.4f}")
    
    return best_f1

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
# CSV UTILITIES / RESUME
# ===========================
def get_save_path(data_flag):
    return f'results_RGB_64x64_ptq_{data_flag}.csv'

def ensure_csv_header(save_path):
    if not os.path.isfile(save_path):
        with open(save_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'dataset', 'n_bits', 'rounding_threshold', 'p_error', 'image_idx', 'inference_time', 'prediction', 'real_tag'
            ])

def cfg_key(data_flag, n_bits, rounding_threshold, p_error):
    return (str(data_flag), str(int(n_bits)), str(int(rounding_threshold)), str(p_error))

def read_progress(save_path):
    progress = {}
    if not os.path.isfile(save_path):
        return progress
    
    with open(save_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 8:
                continue
            df_s, n_bits_s, rt_s, pe_s, idx_s = row[0], row[1], row[2], row[3], row[4]
            key = (df_s, n_bits_s, rt_s, pe_s)
            
            if key not in progress:
                progress[key] = {'done_indices': set()}
            
            if idx_s != 'compilation_failed':
                try:
                    progress[key]['done_indices'].add(int(idx_s))
                except ValueError:
                    pass
    
    return progress

# ===========================
# FHE COMPILATION
# ===========================
def compile_fhe_model(model, train_dataset, n_bits, rounding_threshold_bits, p_error, device='cuda'):
    inputset = torch.stack([train_dataset[i][0] for i in range(min(50, len(train_dataset)))])
    model.eval()
    model_cpu = model.cpu()
    inputset_cpu = inputset.cpu()
    
    print(
        f"Compiling FHE (n_bits={n_bits}, rounding={rounding_threshold_bits}, p_error={p_error}) ..."
    )
    
    try:
        q_model = compile_torch_model(
            model_cpu,
            inputset_cpu,
            n_bits=n_bits,
            rounding_threshold_bits={'n_bits': rounding_threshold_bits, "method": "approximate"},
            p_error=p_error,
        )
        
        if hasattr(q_model, 'fhe_circuit'):
            circuit = q_model.fhe_circuit
            print(f"Number of TLUs: {circuit.programmable_bootstrap_count}")
            print(f"Circuit statistics:\n{circuit.statistics}")
            
        print('FHE model compiled successfully')
        return q_model
    except Exception as e:
        print(f"FHE compilation error: {e}")
        import traceback
        traceback.print_exc()
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
    data_flag,
    n_bits,
    rounding_threshold,
    p_error,
    num_samples,
    max_seconds_per_sample=86400,
    save_path=None,
):
    ensure_csv_header(save_path)
    key = cfg_key(data_flag, n_bits, rounding_threshold, p_error)
    
    progress = read_progress(save_path)
    done = progress.get(key, {'done_indices': set()})['done_indices']
    
    print(
        f"\nFHE Inference {data_flag} cfg=({n_bits},{rounding_threshold},{p_error}) | "
        f"Completed: {len(done)}/{num_samples}"
    )
    
    for i in tqdm(range(num_samples), desc=f"FHE {data_flag}"):
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
            print(f"Error on sample {i}: {e}")
            pred = -1
            elapsed = float(max_seconds_per_sample)
        
        with open(save_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([data_flag, n_bits, rounding_threshold, p_error, i, elapsed, pred, int(label.item())])
        
        if pred != -1:
            print(f"Pred: {pred} | True: {int(label.item())} | Time: {elapsed:.2f}s")
        else:
            print(f"Failed idx={i}")
    
def save_clear_results(data_flag, acc, f1, save_path='results_RGB_64x64_ptq_clear.csv'):
    if not os.path.isfile(save_path):
        with open(save_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'accuracy', 'f1_macro'])
    
    with open(save_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([data_flag, acc, f1])

# ===========================
# MAIN
# ===========================
if __name__ == '__main__':
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'\n{"="*60}')
        print(f'Device: {device}')
        print(f'Image size: {IMG_SIZE}x{IMG_SIZE} RGB')
        print(f'Datasets to process: {len(AVAILABLE_DATASETS)} (RGB)')
        print(f'FHE configurations per dataset: {len(FHE_CONFIGURATIONS)}')
        print(f'FHE samples per configuration: {NUM_FHE_SAMPLES}')
        print(f'Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs')
        print(f'{"="*60}')
        
        # ITERATE OVER ALL RGB DATASETS
        for dataset_idx, (data_flag, dataset_info) in enumerate(AVAILABLE_DATASETS.items(), start=1):
            print(f'\n{"="*60}')
            print(f'DATASET {dataset_idx}/{len(AVAILABLE_DATASETS)}: {dataset_info["name"]} (RGB)')
            print(f'{"="*60}')
            
            save_path = get_save_path(data_flag)
            epochs = dataset_info['epochs']
            num_classes = dataset_info['classes']
            
            try:
                # Load RGB dataset
                train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, num_classes = load_dataset(
                    data_flag, batch_size=BATCH_SIZE
                )
                
                # Create RGB model
                model = cnn64_rgb_ptq(num_classes=num_classes)
                
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"\nTotal parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")
                
                # Training
                print("\n" + "="*60)
                print(f"PHASE 1: TRAINING ({epochs} epochs)")
                print("="*60)
                train(model, train_loader, val_loader, epochs=epochs, device=device)
                
                # Standard evaluation
                print("\n" + "="*60)
                print("PHASE 2: STANDARD EVALUATION")
                print("="*60)
                acc, f1 = evaluate_model(model, test_loader, device=device)
                save_clear_results(data_flag, acc, f1)
                print(f"Accuracy: {acc:.4f} | F1 macro: {f1:.4f}")
                
                # FHE experiments
                print("\n" + "="*60)
                print("PHASE 3: FHE EXPERIMENTS (PTQ)")
                print("="*60)
                
                ensure_csv_header(save_path)
                all_progress = read_progress(save_path)
                
                for cfg_idx, (n_bits, rounding_threshold, p_error) in enumerate(FHE_CONFIGURATIONS, start=1):
                    key = cfg_key(data_flag, n_bits, rounding_threshold, p_error)
                    cfg_prog = all_progress.get(key, {'done_indices': set()})
                                        
                    done_cnt = len(cfg_prog.get('done_indices', set()))
                    
                    print(f"\n=== EXPERIMENT {cfg_idx}/{len(FHE_CONFIGURATIONS)} ===")
                    print(f"cfg=({n_bits},{rounding_threshold},{p_error}) - Resuming {done_cnt}/{NUM_FHE_SAMPLES}")
                    
                    q_model = compile_fhe_model(
                        model, train_dataset, n_bits, rounding_threshold, p_error
                    )
                    
                    if q_model is not None:
                        fhe_inference(
                            q_model, test_dataset, data_flag, n_bits, rounding_threshold, p_error,
                            num_samples=NUM_FHE_SAMPLES, save_path=save_path
                        )
                    else:
                        print(f"Compilation failed for cfg=({n_bits},{rounding_threshold},{p_error})")
                        with open(save_path, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([data_flag, n_bits, rounding_threshold, p_error, 'compilation_failed', 0, -1, -1])
                    
                    if 'q_model' in locals():
                        del q_model
                    clear_memory()
                    print(f"Experiment {cfg_idx} completed\n")
                
                print(f"\nDataset {dataset_info['name']} completed")
                
            except Exception as e:
                print(f"Error in dataset {data_flag}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            finally:
                # Clear memory between datasets
                if 'model' in locals():
                    del model
                if 'train_loader' in locals():
                    del train_loader, val_loader, test_loader
                if 'train_dataset' in locals():
                    del train_dataset, val_dataset, test_dataset
                clear_memory()
        
        print("\n" + "="*60)
        print("ALL RGB DATASETS AND EXPERIMENTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"General error: {e}")
        import traceback
        traceback.print_exc()