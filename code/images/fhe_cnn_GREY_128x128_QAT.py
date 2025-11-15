#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from medmnist import INFO
import medmnist
from concrete.ml.torch.compile import compile_brevitas_qat_model
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import warnings
import csv
import os
import gc

# Brevitas imports
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

warnings.filterwarnings('ignore')

# CONFIGURATION
DATA_FLAG           = 'organamnist'
SAVE_PATH           = 'results_128x128_qat_optimizado_3.csv'
NUM_FHE_SAMPLES     = 50
BATCH_SIZE          = 16
SUBSET_SIZE         = 8000
EPOCHS              = 10
LR                  = 5e-4

# DATASET 
info = INFO[DATA_FLAG]
DataClass = getattr(medmnist, info['python_class'])
num_classes = len(info['label'])
labels_map = info['label']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.Lambda(lambda x: x.float()),
])

def load_organamnist(batch_size=BATCH_SIZE, subset_size=SUBSET_SIZE):
    train_dataset = DataClass(split='train', transform=transform,
                              download=True, size=128)
    val_dataset   = DataClass(split='val',   transform=transform,
                              download=True, size=128)
    test_dataset  = DataClass(split='test',  transform=transform,
                              download=True, size=128)

    train_dataset = Subset(train_dataset,
                           range(min(subset_size, len(train_dataset))))
    val_dataset   = Subset(val_dataset,
                           range(min(subset_size // 4, len(val_dataset))))
    test_dataset  = Subset(test_dataset,
                           range(min(subset_size // 4, len(test_dataset))))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, drop_last=False)
    return train_dataset, val_dataset, test_dataset, \
           train_loader, val_loader, test_loader

# QAT MODEL WITH BREVITAS
class cnn128_qat(nn.Module):
    """
    CNN 128x128x1 with QAT using Brevitas.
    """
    def __init__(self, num_classes=11, bit_width=8):
        super().__init__()
        
        # Input quantization
        self.quant_inp = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        
        # Block 1: 128->128->64
        self.conv1 = QuantConv2d(
            1, 8, kernel_size=3, padding=1, bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=True
        )
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = QuantReLU(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        self.pool1 = nn.AvgPool2d(2)  # 128->64
        self.quant_pool1 = QuantIdentity(  
            bit_width=bit_width,
            return_quant_tensor=True
        )
        
        # Block 2: 64->64->32
        self.conv2 = QuantConv2d(
            8, 16, kernel_size=3, padding=1, bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=True
        )
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = QuantReLU(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        self.pool2 = nn.AvgPool2d(2)  # 64->32
        self.quant_pool2 = QuantIdentity(  
            bit_width=bit_width,
            return_quant_tensor=True
        )
        
        # Block 3: 32->32->8
        self.conv3 = QuantConv2d(
            16, 24, kernel_size=3, padding=1, bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=True
        )
        self.bn3 = nn.BatchNorm2d(24)
        self.relu3 = QuantReLU(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        self.pool3 = nn.AvgPool2d(4)  # 32->8
        self.quant_pool3 = QuantIdentity(  
            bit_width=bit_width,
            return_quant_tensor=True
        )
        
        # Additional pool: 8->4
        self.pool4 = nn.AvgPool2d(2)  # 8->4
        self.quant_pool4 = QuantIdentity(  
            bit_width=bit_width,
            return_quant_tensor=True
        )
        
        # Flatten
        self.flatten = nn.Flatten()
        self.quant_flatten = QuantIdentity(  
            bit_width=bit_width,
            return_quant_tensor=True
        )
        
        # Classifier: 24 * 4 * 4 = 384
        self.fc1 = QuantLinear(
            24 * 4 * 4, 64, bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=True
        )
        self.relu4 = QuantReLU(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        self.fc2 = QuantLinear(
            64, num_classes, bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )

    def forward(self, x):
        # Quantized input
        x = self.quant_inp(x)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.quant_pool1(x)  
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.quant_pool2(x)  
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.quant_pool3(x)  
        
        # Additional pool
        x = self.pool4(x)
        x = self.quant_pool4(x)  
        
        # Flatten
        x = self.flatten(x)
        x = self.quant_flatten(x) 
        
        # Classifier
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        
        return x


class cnn128_qat_optimizado(nn.Module):
    """
    CNN 128x128x1 
    """
    def __init__(self, num_classes=11, bit_width=8):
        super().__init__()
        
        # Input quantization
        self.quant_inp = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        
        # Block 1: 128->32 
        self.conv1 = QuantConv2d(
            1, 8, kernel_size=3, padding=1, bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=True
        )
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = QuantReLU(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        self.pool1 = nn.AvgPool2d(4)  # 128->32
        self.quant_pool1 = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        
        # Block 2: 32->8
        self.conv2 = QuantConv2d(
            8, 16, kernel_size=3, padding=1, bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=True
        )
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = QuantReLU(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        self.pool2 = nn.AvgPool2d(4)  # 32->8
        self.quant_pool2 = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        
        # Flatten
        self.flatten = nn.Flatten()
        self.quant_flatten = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        
        # Simplified classifier: 16 * 8 * 8 = 1024
        self.fc1 = QuantLinear(
            16 * 8 * 8, 32, bias=True,  
            weight_bit_width=bit_width,
            return_quant_tensor=True
        )
        self.relu_fc = QuantReLU(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        self.fc2 = QuantLinear(
            32, num_classes, bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )

    def forward(self, x):
        # Quantized input
        x = self.quant_inp(x)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.quant_pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.quant_pool2(x)
        
        # Flatten and classification
        x = self.flatten(x)
        x = self.quant_flatten(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.fc2(x)
        
        return x

class cnn128_qat_simplificado(nn.Module):
    """
    CNN 128x128x1.
    """
    def __init__(self, num_classes=11, bit_width=8):
        super().__init__()

        # Input quantization
        self.quant_inp = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Block 1: 128->64
        self.conv1 = QuantConv2d(1, 8, kernel_size=3, padding=1, bias=False, weight_bit_width=bit_width, return_quant_tensor=True)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.pool1 = nn.AvgPool2d(2)  # 128->64
        self.quant_pool1 = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Block 2: 64->32
        self.conv2 = QuantConv2d(8, 16, kernel_size=3, padding=1, bias=False, weight_bit_width=bit_width, return_quant_tensor=True)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.pool2 = nn.AvgPool2d(2)  # 64->32
        self.quant_pool2 = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Block 3: 32->8
        self.conv3 = QuantConv2d(16, 24, kernel_size=3, padding=1, bias=False, weight_bit_width=bit_width, return_quant_tensor=True)
        self.bn3 = nn.BatchNorm2d(24)
        self.relu3 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.pool3 = nn.AvgPool2d(4)  # 32->8
        self.quant_pool3 = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Flatten and dense layer
        self.flatten = nn.Flatten()
        self.quant_flatten = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Classifier
        self.fc1 = QuantLinear(24 * 8 * 8, 64, bias=True, weight_bit_width=bit_width, return_quant_tensor=True)
        self.relu_fc = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.fc2 = QuantLinear(64, num_classes, bias=True, weight_bit_width=bit_width, return_quant_tensor=False)

    def forward(self, x):
        x = self.quant_inp(x)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.quant_pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.quant_pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.quant_pool3(x)

        # Flatten and final layer
        x = self.flatten(x)
        x = self.quant_flatten(x)

        # Classification
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.fc2(x)

        return x

class cnn128_qat_ultraoptimizado(nn.Module):
    def __init__(self, num_classes=11, bit_width=8):
        super().__init__()
        
        # Input quantization
        self.quant_inp = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        
        # SINGLE consolidated block: 128->16
        self.conv1 = QuantConv2d(
            1, 12, kernel_size=5, padding=2, bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=True
        )
        self.bn1 = nn.BatchNorm2d(12)
        self.relu1 = QuantReLU(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        self.pool1 = nn.AvgPool2d(8)  # 128->16
        self.quant_pool1 = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        
        # Block 2: 16->8
        self.conv2 = QuantConv2d(
            12, 16, kernel_size=3, padding=1, bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=True
        )
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = QuantReLU(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        self.pool2 = nn.AvgPool2d(2)  # 16->8
        self.quant_pool2 = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        
        # Flatten
        self.flatten = nn.Flatten()
        self.quant_flatten = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        
        # 
        self.fc_out = QuantLinear(
            16 * 8 * 8, num_classes, bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )

    def forward(self, x):
        x = self.quant_inp(x)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.quant_pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.quant_pool2(x)
        
        # Direct classification
        x = self.flatten(x)
        x = self.quant_flatten(x)
        x = self.fc_out(x)
        
        return x


class cnn128_qat_basico(nn.Module):

    """
    CNN 128x128x1 OPTIMIZED for low p_error
    
    """
    def __init__(self, num_classes=11, bit_width=6):
        super().__init__()
        
        # Input quantization
        self.quant_inp = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        
        # Block 1: 128->32
        self.conv1 = QuantConv2d(
            1, 8, kernel_size=7, padding=3, stride=4, bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=True
        )
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = QuantReLU(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        self.quant1 = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        
        # Block 2: 32->8
        self.conv2 = QuantConv2d(
            8, 12, kernel_size=3, padding=1, stride=4, bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=True
        )
        self.bn2 = nn.BatchNorm2d(12)
        self.relu2 = QuantReLU(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        self.quant2 = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        
        # Flatten
        self.flatten = nn.Flatten()
        self.quant_flatten = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        
        self.fc1 = QuantLinear(
            12 * 8 * 8, 24, bias=True, 
            weight_bit_width=bit_width,
            return_quant_tensor=True
        )
        self.relu_fc = QuantReLU(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        self.fc_out = QuantLinear(
            24, num_classes, bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )

    def forward(self, x):
        x = self.quant_inp(x)
        
        # Block 1:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.quant1(x)
        
        # Block 2: 
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.quant2(x)
        
        # Classification
        x = self.flatten(x)
        x = self.quant_flatten(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.fc_out(x)
        
        return x
       


# TRAINING AND EVALUATION 
def train(model, train_loader, val_loader, epochs=EPOCHS, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    if device == 'mps':
        print("MPS not supported by Brevitas, using CPU for training")
        device = 'cpu'
    
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
        train_f1  = f1_score(train_labels, train_preds, average='macro')

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Loss={total_loss/len(train_loader):.4f} | "
              f"Train Acc={train_acc:.4f} | Train F1={train_f1:.4f} | "
              f"Val Acc={val_acc:.4f} | Val F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best model loaded with F1={best_f1:.4f}")

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

# CSV FUNCTIONS
def ensure_csv_header(save_path: str = SAVE_PATH):
    if not os.path.isfile(save_path):
        with open(save_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'n_bits', 'rounding_threshold', 'p_error',
                'imagen_idx', 'tiempo_segundos', 'prediccion', 'etiqueta_real'
            ])

def cfg_key(n_bits, rounding_threshold, p_error):
    return (str(int(n_bits)), str(int(rounding_threshold)), str(p_error))

def read_progress(save_path: str = SAVE_PATH):
    progress = {}
    if not os.path.isfile(save_path):
        return progress

    with open(save_path, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
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

def compute_and_write_global_if_complete(n_bits, rounding_threshold, p_error,
                                        num_samples: int, save_path: str = SAVE_PATH):
    key = cfg_key(n_bits, rounding_threshold, p_error)
    if not os.path.isfile(save_path):
        return False

    preds, labels, times = [], [], []
    has_global = False

    with open(save_path, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or len(row) < 7:
                continue
            if (row[0], row[1], row[2]) != key:
                continue
            if row[3] == 'global':
                has_global = True
                break
            try:
                pred = int(row[5])
                lab  = int(row[6])
                if pred != -1:
                    preds.append(pred)
                    labels.append(lab)
                    times.append(float(row[4]))
            except Exception:
                pass

    if has_global:
        return False

    present_indices = set()
    with open(save_path, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or len(row) < 7:
                continue
            if (row[0], row[1], row[2]) != key:
                continue
            if row[3].isdigit():
                present_indices.add(int(row[3]))

    if len(present_indices) < num_samples:
        return False

    if preds and labels:
        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, average='macro')
    else:
        acc = 0.0
        f1  = 0.0

    mean_t = float(np.mean(times)) if times else 0.0

    with open(save_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([n_bits, rounding_threshold, p_error,
                         'global', mean_t, acc, f1])

    print(f"\n[GLOBAL METRICS WRITTEN] cfg=({n_bits},{rounding_threshold},{p_error})")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | Mean time: {mean_t:.2f}s")
    return True

# FHE COMPILATION
def compile_fhe_model(model, train_dataset, n_bits, rounding_threshold_bits, p_error):
    inputset = torch.stack([train_dataset[i][0] for i in range(min(50, len(train_dataset)))])
    
    model.eval()
    model_cpu = model.cpu()
    inputset_cpu = inputset.cpu()

    print(f"Compiling QAT FHE model (n_bits={n_bits}, "
          f"rounding_threshold_bits={rounding_threshold_bits}, "
          f"p_error={p_error}) on CPU...")

    try:
        q_model = compile_brevitas_qat_model(
            model_cpu,
            inputset_cpu,
            #n_bits=n_bits,
            rounding_threshold_bits=rounding_threshold_bits,
            p_error=p_error,
        )
        print('QAT model compiled for FHE')
        return q_model
    except Exception as e:
        print(f"FHE QAT compilation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print('GPU and RAM memory cleared')

# FHE INFERENCE
def fhe_inference(q_model, test_dataset, n_bits, rounding_threshold, p_error,
                  num_samples: int, max_seconds_per_sample: int = 86400,
                  save_path: str = SAVE_PATH):
    ensure_csv_header(save_path)
    key = cfg_key(n_bits, rounding_threshold, p_error)
    progress = read_progress(save_path)
    done = progress.get(key, {'done_indices': set(), 'has_global': False})['done_indices']

    print(f"\nEncrypted inference (FHE-Execute) cfg=({n_bits},{rounding_threshold},{p_error}) | "
          f"completed: {len(done)}/{num_samples}")

    for i in tqdm(range(num_samples), desc=f"FHE n_bits={n_bits} threshold={rounding_threshold} p_error={p_error}"):
        if i in done:
            continue

        sample, label = test_dataset[i]
        float_input = sample.unsqueeze(0).numpy().astype(np.float32)

        start_time = time.time()
        try:
            result = q_model.forward(float_input, fhe='simulate')
            pred = int(np.argmax(result))
            elapsed = time.time() - start_time
        except Exception as e:
            print(f"Error in inference for sample {i}: {e}")
            pred = -1
            elapsed = float(max_seconds_per_sample)

        with open(save_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([n_bits, rounding_threshold, p_error,
                             i, elapsed, pred, int(label.item())])

        if pred != -1:
            print(f"FHE Prediction: {pred} | Real label: {int(label.item())} | Time: {elapsed:.2f}s | SAVED")
        else:
            print(f"Failure saved for idx={i} | Time={elapsed:.2f}s")

        compute_and_write_global_if_complete(n_bits, rounding_threshold,
                                            p_error, num_samples, save_path)

# MAIN
if __name__ == '__main__':
    try:
        device = 'cpu'
        
        print(f'Using device: {device}')
        print('--- QAT process start ---')

        # Load datasets 
        (train_dataset, val_dataset, test_dataset,
         train_loader, val_loader, test_loader) = load_organamnist(
            batch_size=BATCH_SIZE, subset_size=SUBSET_SIZE)

        # Configurations
        configurations = [
            (5,5, 0.001),
            (5,6, 0.001),
            (6, 6, 0.007),
            (6, 5, 0.007),
            (7, 7, 0.007),
            (7, 6, 0.007),
            (8, 8, 0.01),
            (8, 6, 0.01),
        ]

        ensure_csv_header(SAVE_PATH)
        all_progress = read_progress(SAVE_PATH)

        for cfg in configurations:
            bit_width, rt, pe = cfg
            key = cfg_key(bit_width, rt, pe)

            # Skip if already completed
            if all_progress.get(key, {}).get('has_global', False):
                print(f"\n=== Configuration {cfg} already completed, skipping ===")
                continue

            
            print(f"STARTING CONFIGURATION: bit_width={bit_width}, RT={rt}, p_error={pe}")
            

            # TRAIN MODEL 
            model = cnn128_qat_optimizado(num_classes=num_classes, bit_width=bit_width)
            
            print(f"\n--- Training model with bit_width={bit_width} ---")
            train(model, train_loader, val_loader, epochs=EPOCHS, device=device)

            # Evaluate accuracy of trained model
            acc, f1 = evaluate_model(model, test_loader, device=device)
            print(f"Standard QAT evaluation (bit_width={bit_width}) | "
                  f"Accuracy: {acc:.4f} | F1 macro: {f1:.4f}")

            
            q_model = compile_fhe_model(
                model=model, 
                train_dataset=train_dataset, 
                n_bits=None,  
                rounding_threshold_bits=rt, 
                p_error=pe
            )

            if q_model is None:
                print(f"\n=== Configuration {cfg} DOES NOT COMPILE, skipping ===")
                clear_memory()
                continue

            # FHE INFERENCE
            print(f"\n--- Starting FHE inference for configuration {cfg} ---")
            fhe_inference(
                q_model=q_model, 
                test_dataset=test_dataset,
                n_bits=bit_width,  # logging in CSV
                rounding_threshold=rt, 
                p_error=pe,
                num_samples=NUM_FHE_SAMPLES,
                max_seconds_per_sample=86400,
                save_path=SAVE_PATH
            )

            # Clear memory after each configuration
            clear_memory()
            
            print(f"\n✓ Configuration {cfg} completed\n")

    except KeyboardInterrupt:
        print("\n=== Aborted by user ===")
    except Exception as exc:
        print(f"\n=== Unexpected ERROR: {exc} ===")
        import traceback
        traceback.print_exc()
        raise