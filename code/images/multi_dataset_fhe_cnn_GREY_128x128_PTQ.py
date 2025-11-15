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
# CONFIGURACIÓN GENERAL
# ===========================
# DATASETS DISPONIBLES EN ESCALA DE GRISES:
AVAILABLE_DATASETS = {
    'pneumoniamnist': {'name': 'PneumoniaMNIST', 'classes': 2, 'modality': 'Chest X-Ray'},
    'octmnist': {'name': 'OCTMNIST', 'classes': 4, 'modality': 'Retinal OCT'},
    'chestmnist': {'name': 'ChestMNIST', 'classes': 14, 'modality': 'Chest X-Ray'},  # Multi-label
    'breastmnist': {'name': 'BreastMNIST', 'classes': 2, 'modality': 'Breast Ultrasound'},
    'organamnist': {'name': 'OrganAMNIST', 'classes': 11, 'modality': 'Abdominal CT (Axial)'},
    'organcmnist': {'name': 'OrganCMNIST', 'classes': 11, 'modality': 'Abdominal CT (Coronal)'},
    'organsmnist': {'name': 'OrganSMNIST', 'classes': 11, 'modality': 'Abdominal CT (Sagittal)'},
}

# Selecciona el dataset a probar
DATA_FLAG = 'pneumoniamnist'  # Cambia esto para probar diferentes datasets

IMG_SIZE = 128

SAVE_PATH = f'results_{IMG_SIZE}x{IMG_SIZE}_{DATA_FLAG}.csv'
NUM_FHE_SAMPLES = 100  
BATCH_SIZE = 16
SUBSET_SIZE = 13000
EPOCHS = 10
LR = 5e-4

# ===========================
# DATASET
# ===========================
def get_dataset_info(data_flag):
    """Obtiene información del dataset"""
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    num_classes = len(info['label'])
    labels_map = info['label']
    
    print(f"\n{'='*60}")
    print(f"Dataset: {AVAILABLE_DATASETS[data_flag]['name']}")
    print(f"Modalidad: {AVAILABLE_DATASETS[data_flag]['modality']}")
    print(f"Número de clases: {num_classes}")
    print(f"Labels: {labels_map}")
    print(f"{'='*60}\n")
    
    return info, DataClass, num_classes, labels_map

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda x: x.float()),
])

def load_dataset(data_flag, batch_size=BATCH_SIZE, subset_size=SUBSET_SIZE):
    info, DataClass, num_classes, labels_map = get_dataset_info(data_flag)
    
    train_dataset = DataClass(split='train', transform=transform, download=True, size=IMG_SIZE)
    val_dataset = DataClass(split='val', transform=transform, download=True, size=IMG_SIZE)
    test_dataset = DataClass(split='test', transform=transform, download=True, size=IMG_SIZE)
    
    # Verificar si el dataset es grayscale (1 canal)
    sample_img, _ = train_dataset[0]
    if sample_img.shape[0] != 1:
        raise ValueError(f"ERROR: {data_flag} no es escala de grises! Tiene {sample_img.shape[0]} canales")
    
    print(f"✓ Dataset verificado: {sample_img.shape[0]} canal (escala de grises)")
    print(f"  Train: {len(train_dataset)} muestras")
    print(f"  Val: {len(val_dataset)} muestras")
    print(f"  Test: {len(test_dataset)} muestras")
    
    # Ajustar subset_size según el tamaño del dataset
    train_size = min(subset_size, len(train_dataset))
    val_size = min(subset_size // 4, len(val_dataset))
    test_size = min(subset_size // 4, len(test_dataset))
    
    train_dataset = Subset(train_dataset, range(train_size))
    val_dataset = Subset(val_dataset, range(val_size))
    test_dataset = Subset(test_dataset, range(test_size))
    
    print(f"  Usando - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, num_classes


# ===========================
# MODELO INTERMEDIO
# ===========================

class cnn128_intermediate(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),  # 128 -> 64

            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),  # 64 -> 32

            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(4),  # 32 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 8 * 8, 96),  # <-- antes era 48*4*4
            nn.ReLU(),
            nn.Linear(96, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ===========================
# ENTRENAMIENTO Y EVAL
# ===========================
def train(model, train_loader, val_loader, epochs=EPOCHS, device='cpu'):
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
        print(f"✓ Mejor modelo cargado con F1: {best_f1:.4f}")

def evaluate_model(model, loader, device='cpu'):
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
# UTILIDADES CSV / REANUDAR
# ===========================
def ensure_csv_header(save_path: str = SAVE_PATH):
    if not os.path.isfile(save_path):
        with open(save_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'dataset', 'n_bits', 'rounding_threshold', 'p_error',
                'imagen_idx', 'tiempo_segundos', 'prediccion', 'etiqueta_real'
            ])

def cfg_key(data_flag, n_bits, rounding_threshold, p_error):
    return (str(data_flag), str(int(n_bits)), str(int(rounding_threshold)), str(p_error))

def read_progress(save_path: str = SAVE_PATH):
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
                progress[key] = {'done_indices': set(), 'has_global': False}
            
            if idx_s == 'global':
                progress[key]['has_global'] = True
            else:
                try:
                    progress[key]['done_indices'].add(int(idx_s))
                except ValueError:
                    pass
    return progress

def compute_and_write_global_if_complete(data_flag, n_bits, rounding_threshold, p_error, num_samples: int, save_path: str = SAVE_PATH):
    key = cfg_key(data_flag, n_bits, rounding_threshold, p_error)
    if not os.path.isfile(save_path):
        return False
    
    preds, labels, times = [], [], []
    has_global = False
    
    with open(save_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 8:
                continue
            if (row[0], row[1], row[2], row[3]) != key:
                continue
            if row[4] == 'global':
                has_global = True
                break
            try:
                pred = int(row[6])
                lab = int(row[7])
                if pred != -1:
                    preds.append(pred)
                    labels.append(lab)
                times.append(float(row[5]))
            except Exception:
                pass
    
    if has_global:
        return False
    
    present_indices = set()
    with open(save_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 8:
                continue
            if (row[0], row[1], row[2], row[3]) != key:
                continue
            if row[4].isdigit():
                present_indices.add(int(row[4]))
    
    if len(present_indices) < num_samples:
        return False
    
    if preds and labels:
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
    else:
        acc = 0.0
        f1 = 0.0
    
    mean_t = float(np.mean(times)) if len(times) > 0 else 0.0
    
    with open(save_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([data_flag, n_bits, rounding_threshold, p_error, 'global', mean_t, acc, f1])
    
    print(f"\n[MÉTRICAS GLOBALES] {data_flag} cfg=({n_bits},{rounding_threshold},{p_error}) |")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | Tiempo medio: {mean_t:.2f}s")
    return True

# ===========================
# COMPILACIÓN FHE
# ===========================
def compile_fhe_model(model, train_dataset, n_bits, rounding_threshold_bits, p_error, device='cpu'):
    inputset = torch.stack([train_dataset[i][0] for i in range(min(50, len(train_dataset)))])
    model.eval()
    model_cpu = model.cpu()
    inputset_cpu = inputset.cpu()
    
    print(
        f"Compilando FHE (n_bits={n_bits}, rounding={rounding_threshold_bits}, p_error={p_error}) ..."
    )
    
    try:
        q_model = compile_torch_model(
            model_cpu,
            inputset_cpu,
            device='cpu',
            n_bits=n_bits,
            rounding_threshold_bits={'n_bits': rounding_threshold_bits, "method": "approximate"},
            p_error=p_error,
        )
        
        if hasattr(q_model, 'fhe_circuit'):
            circuit = q_model.fhe_circuit
            print(f"✓ Número de TLUs: {circuit.programmable_bootstrap_count}")
            print(f"✓ Estadísticas del circuito:\n{circuit.statistics}")
            
        print('✓ Modelo FHE compilado exitosamente')
        return q_model
    except Exception as e:
        print(f"✗ Error compilación FHE: {e}")
        return None

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print('✓ Memoria liberada')

# ===========================
# INFERENCIA FHE
# ===========================
def fhe_inference(
    q_model,
    test_dataset,
    data_flag,
    n_bits,
    rounding_threshold,
    p_error,
    num_samples: int,
    max_seconds_per_sample: int = 86400,
    save_path: str = SAVE_PATH,
):
    ensure_csv_header(save_path)
    key = cfg_key(data_flag, n_bits, rounding_threshold, p_error)
    
    progress = read_progress(save_path)
    done = progress.get(key, {'done_indices': set(), 'has_global': False})['done_indices']
    
    print(
        f"\nInferencia FHE {data_flag} cfg=({n_bits},{rounding_threshold},{p_error}) | "
        f"Completadas: {len(done)}/{num_samples}"
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
            print(f"✗ Error muestra {i}: {e}")
            pred = -1
            elapsed = float(max_seconds_per_sample)
        
        with open(save_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([data_flag, n_bits, rounding_threshold, p_error, i, elapsed, pred, int(label.item())])
        
        if pred != -1:
            print(f"✓ Pred: {pred} | Real: {int(label.item())} | Tiempo: {elapsed:.2f}s")
        else:
            print(f"✗ Fallo idx={i}")
    
    compute_and_write_global_if_complete(data_flag, n_bits, rounding_threshold, p_error, num_samples, save_path)

# ===========================
# MAIN
# ===========================
if __name__ == '__main__':
    try:
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        print(f'\n{"="*60}')
        print(f'Dispositivo: {device}')
        print(f'Dataset seleccionado: {DATA_FLAG}')
        print(f'{"="*60}')
        
        # Verificar que el dataset esté disponible
        if DATA_FLAG not in AVAILABLE_DATASETS:
            raise ValueError(f"Dataset '{DATA_FLAG}' no disponible. Opciones: {list(AVAILABLE_DATASETS.keys())}")
        
        # Cargar dataset
        train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, num_classes = load_dataset(
            DATA_FLAG, batch_size=BATCH_SIZE, subset_size=SUBSET_SIZE
        )
        
        # Crear modelo
        model = cnn128_intermediate(num_classes=num_classes)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nParámetros totales: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,}")
        
        # Entrenamiento
        print("\n" + "="*60)
        print("FASE 1: ENTRENAMIENTO")
        print("="*60)
        train(model, train_loader, val_loader, epochs=EPOCHS, device=device)
        
        # Evaluación estándar
        print("\n" + "="*60)
        print("FASE 2: EVALUACIÓN ESTÁNDAR")
        print("="*60)
        acc, f1 = evaluate_model(model, test_loader, device=device)
        print(f"Accuracy: {acc:.4f} | F1 macro: {f1:.4f}")
        
        # Configuraciones FHE
        print("\n" + "="*60)
        print("FASE 3: EXPERIMENTOS FHE")
        print("="*60)
        
        configurations = [
            # (7, 6, 0.1),
            (8, 6, 0.008),
        ]
        
        ensure_csv_header(SAVE_PATH)
        all_progress = read_progress(SAVE_PATH)
        
        print(f"Total configuraciones: {len(configurations)}")
        print(f"Muestras por configuración: {NUM_FHE_SAMPLES}")
        print(f"Archivo de resultados: {SAVE_PATH}")
        
        for i, (n_bits, rounding_threshold, p_error) in enumerate(configurations, start=1):
            key = cfg_key(DATA_FLAG, n_bits, rounding_threshold, p_error)
            cfg_prog = all_progress.get(key, {'done_indices': set(), 'has_global': False})
            
            if cfg_prog.get('has_global', False):
                print(f"\n=== EXPERIMENTO {i}/{len(configurations)} ===")
                print(f"✓ cfg=({n_bits},{rounding_threshold},{p_error}) ya completado")
                continue
            
            done_cnt = len(cfg_prog.get('done_indices', set()))
            if done_cnt >= NUM_FHE_SAMPLES:
                wrote = compute_and_write_global_if_complete(
                    DATA_FLAG, n_bits, rounding_threshold, p_error, NUM_FHE_SAMPLES, SAVE_PATH
                )
                print(f"\n=== EXPERIMENTO {i}/{len(configurations)} ===")
                print(f"✓ cfg=({n_bits},{rounding_threshold},{p_error}) marcado como completo")
                continue
            
            print(f"\n=== EXPERIMENTO {i}/{len(configurations)} ===")
            print(f"cfg=({n_bits},{rounding_threshold},{p_error}) - Reanudando {done_cnt}/{NUM_FHE_SAMPLES}")
            
            q_model = compile_fhe_model(
                model, train_dataset, n_bits, rounding_threshold, p_error
            )

            import re
            from collections import Counter

            mlir = q_model.fhe_circuit.mlir  # texto MLIR
            # cuenta ocurrencias y tamaños de todos los lookups
            pattern = re.compile(r'apply_mapped_lookup_table.*?tensor<([^>]+)>', re.DOTALL)
            shapes = []
            for m in pattern.finditer(mlir):
                shape = m.group(1)  # p.ej. "1x12x64x64x!FHE.eint<...>"
                # extrae dims numéricas y calcula elementos
                dims = [int(d) for d in re.findall(r'(\d+)', shape.split('x!FHE')[0])]
                nelems = 1
                for d in dims:
                    nelems *= d
                shapes.append((tuple(dims), nelems, shape))

            by_shape = Counter([s[0] for s in shapes])
            total_lookups = len(shapes)
            print("Total LUT ops:", total_lookups)
            print("Top formas (dims) y cuenta:")
            for dims, cnt in by_shape.most_common(10):
                print(dims, "→", cnt, "veces")

            # estima coste por forma (aprox: #ops * #elementos)
            from collections import defaultdict
            cost = defaultdict(int)
            for dims, nelems, _ in shapes:
                cost[dims] += nelems
            print("\nTop 'costes' aproximados (ops×elementos):")
            for dims, c in sorted(cost.items(), key=lambda x: -x[1])[:10]:
                print(dims, "→", c)


            
            if q_model is not None:
                fhe_inference(
                    q_model, test_dataset, DATA_FLAG, n_bits, rounding_threshold, p_error,
                    num_samples=NUM_FHE_SAMPLES, save_path=SAVE_PATH
                )
            else:
                print(f"✗ Compilación fallida para cfg=({n_bits},{rounding_threshold},{p_error})")
                with open(SAVE_PATH, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([DATA_FLAG, n_bits, rounding_threshold, p_error, 'compilation_failed', 0, -1, -1])
            
            if 'q_model' in locals():
                del q_model
            clear_memory()
            print(f"✓ Experimento {i} completado\n")
        
        print("\n" + "="*60)
        print("✓ TODOS LOS EXPERIMENTOS COMPLETADOS")
        print("="*60)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
