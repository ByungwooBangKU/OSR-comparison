# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split, ConcatDataset
# sklearn 관련 import 추가
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.manifold import TSNE # t-SNE 추가
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import random
import matplotlib
import gc
matplotlib.use('Agg') # 백엔드 설정

import matplotlib.pyplot as plt
plt.ioff() # 대화형 모드 비활성화
# Seaborn 임포트 (선택적 설치 필요: pip install seaborn)
try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False
    print("Warning: Seaborn not installed. Confusion matrix heatmap visualization will be skipped.")

import pandas as pd
import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union

from datasets import load_dataset, DatasetDict, concatenate_datasets
 
ID_DATA_PATH = 'log_all_critical.csv'
OE_MASKED_SYSLOG_PATH = 'masked_for_oe.csv'
OOD_DATA_PATH = 'log_all_critical.csv'
TEXT_COLUMN = 'text'
CLASS_COLUMN = 'class'
OE_MASKED_TEXT_COLUMN = 'masked_text_attention'
ID_EXCLUDE_CLASS = "unknown"
OOD_TARGET_CLASS = "unknown"
OE_SOURCES_TO_RUN = ['syslog_masked', 'snli', 'imdb']
RESULT_DIR = 'results_osr_syslog_comparison_all'
MODEL_DIR = 'models_osr_syslog_comparison_all'
DATA_DIR = 'data'
CACHE_DIR = os.path.join(DATA_DIR, "hf_cache")
MODEL_TYPE = 'roberta-base'
MAX_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 2e-5
OE_LAMBDA = 1.0
TEMPERATURE = 1.0
SEED = 42
SAVE_MODEL = True
EVAL_ONLY = False
STD_MODEL_PATH = None
OE_MODEL_PATH_PREFIX = 'roberta_oe_conf'
NO_PLOT = False
SKIP_STANDARD = False
SKIP_OE_ALL = False

# 시드 설정
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    print(f"Seed set to {seed}")

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# 데이터 클래스 (변경 없음)
class TextDataset(Dataset):
    # ... (이전 코드 복사) ...
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.labels = labels; print(f"Tokenizing {len(texts)} texts...")
        valid_texts = [str(t) if pd.notna(t) else "" for t in texts]
        self.encodings = tokenizer(valid_texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        print("Tokenization complete.")
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}; item['label'] = torch.tensor(self.labels[idx], dtype=torch.long); return item

# --- 데이터셋 준비 함수 (이전과 동일) ---
def prepare_syslog_data(tokenizer, max_length: int, id_data_path: str, text_col: str, class_col: str, exclude_class: str, seed: int = 42) -> Tuple[Dataset, Dataset, int, LabelEncoder, Dict, Dict]:
    # ... (이전 코드 복사) ...
    print(f"Preparing Syslog ID data from: {id_data_path}")
    try:
        df = pd.read_csv(id_data_path);
        if not all(c in df.columns for c in [text_col, class_col]): raise ValueError(f"ID Data CSV must contain '{text_col}' and '{class_col}' columns.")
        df = df.dropna(subset=[text_col, class_col]); df[class_col] = df[class_col].astype(str)
        df_known_initial = df[df[class_col] != exclude_class].copy()
        if df_known_initial.empty: raise ValueError(f"No data left after excluding class '{exclude_class}'.")
        print(f"  - Data size after excluding '{exclude_class}': {len(df_known_initial)}")
        initial_label_encoder = LabelEncoder(); initial_label_encoder.fit(df_known_initial[class_col])
        num_classes_initial = len(initial_label_encoder.classes_); print(f"  - Found {num_classes_initial} known classes initially.")
        print(f"  - Filtering known classes to have at least 2 samples for train/test split...")
        class_counts = df_known_initial[class_col].value_counts()
        classes_to_keep = class_counts[class_counts >= 2].index; classes_removed = class_counts[class_counts < 2].index
        if len(classes_removed) > 0: removed_class_names = [cls for cls in classes_removed]; print(f"  - Removing classes with less than 2 samples: {removed_class_names}"); df_known_final = df_known_initial[df_known_initial[class_col].isin(classes_to_keep)].copy()
        else: df_known_final = df_known_initial; print("  - All known classes have at least 2 samples.")
        if df_known_final.empty: raise ValueError("No data left after filtering classes with less than 2 samples.")
        final_classes = sorted(df_known_final[class_col].unique()); num_classes_final = len(final_classes)
        print(f"  - Final number of known classes for training/testing: {num_classes_final}"); print(f"  - Final known classes: {final_classes}")
        final_label_encoder = LabelEncoder(); final_label_encoder.fit(final_classes)
        final_label2id = {label: i for i, label in enumerate(final_label_encoder.classes_)}; final_id2label = {i: label for label, i in final_label2id.items()}
        print(f"  - Final Label to ID mapping: {final_label2id}")
        df_known_final['label'] = df_known_final[class_col].map(final_label2id)
        train_df, test_df = train_test_split(df_known_final, test_size=0.2, random_state=seed, stratify=df_known_final['label'])
        print(f"  - Split into Train: {len(train_df)}, Test: {len(test_df)}")
        train_dataset = TextDataset(train_df[text_col].tolist(), train_df['label'].tolist(), tokenizer, max_length)
        id_test_dataset = TextDataset(test_df[text_col].tolist(), test_df['label'].tolist(), tokenizer, max_length)
        return train_dataset, id_test_dataset, num_classes_final, final_label_encoder, final_label2id, final_id2label
    except Exception as e: print(f"Error preparing Syslog ID data: {e}"); raise

def prepare_syslog_masked_oe_data(tokenizer, max_length: int, oe_data_path: str, oe_text_col: str) -> Optional[Dataset]:
    # ... (이전 코드 복사) ...
    print(f"Preparing Syslog Masked OE data from: {oe_data_path}")
    try:
        df = pd.read_csv(oe_data_path)
        if oe_text_col not in df.columns: raise ValueError(f"OE Data CSV must contain '{oe_text_col}' column.")
        df = df.dropna(subset=[oe_text_col]); texts = df[oe_text_col].astype(str).tolist()
        if not texts: print(f"Warning: No valid OE texts found in '{oe_text_col}'."); return None
        oe_labels = np.full(len(texts), -1, dtype=int); oe_dataset = TextDataset(texts, oe_labels, tokenizer, max_length)
        print(f"  - Loaded {len(oe_dataset)} samples for OE training from Syslog Masked.")
        return oe_dataset
    except Exception as e: print(f"Error preparing Syslog Masked OE data: {e}"); return None

def prepare_external_oe_data(tokenizer, max_length: int, oe_source: str, data_dir: str = 'data', cache_dir: Optional[str] = None) -> Optional[Dataset]:
    # ... (이전 코드 복사) ...
    print(f"Preparing External OE data source: {oe_source}")
    if cache_dir is None: cache_dir = os.path.join(data_dir, "hf_cache"); os.makedirs(cache_dir, exist_ok=True)
    config = None; text_col = None; split = "train"
    if oe_source == "snli": config = {"name": "snli", "split": "train", "text_col": "hypothesis"}
    elif oe_source == "imdb": config = {"name": "imdb", "split": "train", "text_col": "text"}
    elif oe_source == "wikitext": config = {"name": "wikitext", "config": "wikitext-103-raw-v1", "split": "train", "text_col": "text"}
    else: print(f"Error: Unknown external OE source '{oe_source}'"); return None
    try:
        print(f"Loading {oe_source} ({config.get('config', 'default')} - {config['split']} split)...")
        ds = load_dataset(config["name"], config.get("config"), split=config["split"], cache_dir=cache_dir)
        if isinstance(ds, DatasetDict):
             if config['split'] in ds: ds = ds[config['split']]
             else: raise ValueError(f"Split '{config['split']}' not found for {oe_source}")
        texts = [item for item in ds[config['text_col']] if isinstance(item, str) and item.strip()]
        if oe_source == "wikitext": texts = [text for text in texts if not text.strip().startswith("=") and len(text.strip().split()) > 3]
        if not texts: print(f"Warning: No valid texts found for OE source {oe_source}."); return None
        oe_labels = np.full(len(texts), -1, dtype=int); oe_dataset = TextDataset(texts, oe_labels, tokenizer, max_length)
        print(f"  - Loaded {len(oe_dataset)} samples for OE training from {oe_source}.")
        return oe_dataset
    except Exception as e: print(f"Error loading external OE dataset {oe_source}: {e}"); return None

def prepare_syslog_ood_data(tokenizer, max_length: int, ood_data_path: str, text_col: str, class_col: str, ood_target_class: str) -> Optional[Dataset]:
    # ... (이전 코드 복사) ...
    print(f"Preparing Syslog OOD data (class: '{ood_target_class}') from: {ood_data_path}")
    try:
        df = pd.read_csv(ood_data_path);
        if not all(c in df.columns for c in [text_col, class_col]): raise ValueError(f"OOD Data CSV must contain '{text_col}' and '{class_col}' columns.")
        df = df.dropna(subset=[text_col, class_col]); df[class_col] = df[class_col].astype(str)
        df_ood = df[df[class_col] == ood_target_class].copy()
        if df_ood.empty: print(f"Warning: No data found for OOD class '{ood_target_class}'."); return None
        texts = df_ood[text_col].tolist(); ood_labels = np.full(len(texts), -1, dtype=int)
        ood_dataset = TextDataset(texts, ood_labels, tokenizer, max_length)
        print(f"  - Loaded {len(ood_dataset)} samples for OOD testing (class: '{ood_target_class}').")
        return ood_dataset
    except Exception as e: print(f"Error preparing Syslog OOD data: {e}"); return None

# 모델 정의 (수정됨 - 특징 추출 지원)
class RoBERTaOOD(nn.Module):
    def __init__(self, num_classes: int, model_name: str = 'roberta-base'):
        super(RoBERTaOOD, self).__init__()
        print(f"Initializing RoBERTa model ({model_name}) for {num_classes} classes.")
        config = RobertaConfig.from_pretrained(model_name)
        config.num_labels = num_classes
        config.torchscript = True; config.use_cache = True; config.return_dict = True
        config.output_hidden_states = True # <<<--- Hidden state 출력 활성화
        self.roberta = RobertaForSequenceClassification.from_pretrained(model_name, config=config)

    def forward(self, input_ids, attention_mask, output_features=False):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits
        if output_features:
            # 마지막 hidden state의 CLS 토큰 임베딩 사용 ([CLS]는 보통 첫번째 토큰)
            # outputs.hidden_states[-1] shape: (batch_size, seq_len, hidden_size)
            features = outputs.hidden_states[-1][:, 0, :] # CLS token embedding
            # 또는 Pooler output 사용 (선택적)
            # features = outputs.pooler_output # shape: (batch_size, hidden_size)
            return logits, features
        else:
            return logits

# 학습 함수 (OE - Confidence Loss 사용 - 변경 없음)
def train_with_oe_confidence_loss(model: nn.Module, train_loader: DataLoader, oe_loader: DataLoader, optimizer: optim.Optimizer, scheduler, device: torch.device, num_epochs: int, oe_lambda: float):
    # ... (이전 코드 복사) ...
    model.train(); use_amp = device.type == 'cuda'; scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print("Starting training with Outlier Exposure (Confidence Loss)... AMP enabled: ", use_amp)
    for epoch in range(num_epochs):
        oe_iter = iter(oe_loader); total_loss = 0; total_id_loss = 0; total_oe_loss = 0
        progress_bar = tqdm(train_loader, desc=f"OE Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device); labels = batch['label'].to(device)
            try: oe_batch = next(oe_iter)
            except StopIteration: oe_iter = iter(oe_loader); oe_batch = next(oe_iter)
            oe_input_ids = oe_batch['input_ids'].to(device); oe_attention_mask = oe_batch['attention_mask'].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=str(device).split(':')[0], enabled=use_amp):
                logits = model(input_ids, attention_mask); id_loss = F.cross_entropy(logits, labels)
                oe_logits = model(oe_input_ids, oe_attention_mask); oe_probs = F.softmax(oe_logits, dim=1)
                max_oe_probs, _ = oe_probs.max(dim=1); oe_loss = (1.0 - max_oe_probs).mean()
                total_batch_loss = id_loss + oe_lambda * oe_loss
            scaler.scale(total_batch_loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step()
            total_loss += total_batch_loss.item(); total_id_loss += id_loss.item(); total_oe_loss += oe_loss.item()
            progress_bar.set_postfix({'Total Loss': total_batch_loss.item(), 'ID Loss': id_loss.item(), 'OE Loss': oe_loss.item()})
        avg_loss = total_loss / len(train_loader); avg_id_loss = total_id_loss / len(train_loader); avg_oe_loss = total_oe_loss / len(train_loader)
        print(f"OE Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f} (ID: {avg_id_loss:.4f}, OE: {avg_oe_loss:.4f})")

# 학습 함수 (표준 - 변경 없음)
def train_standard(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, scheduler, device: torch.device, num_epochs: int):
    # ... (이전 코드 복사) ...
    model.train(); use_amp = device.type == 'cuda'; scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print("Starting standard training... AMP enabled: ", use_amp)
    for epoch in range(num_epochs):
        total_loss = 0; progress_bar = tqdm(train_loader, desc=f"Standard Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device); labels = batch['label'].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=str(device).split(':')[0], enabled=use_amp):
                logits = model(input_ids, attention_mask); loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += loss.item(); progress_bar.set_postfix({'loss': loss.item()})
        scheduler.step(); avg_loss = total_loss / len(train_loader)
        print(f"Standard Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# OSR 평가 함수 (수정됨 - 특징 벡터 반환 추가)
def evaluate_osr(model: nn.Module, id_loader: DataLoader, ood_loader: DataLoader, device: torch.device, temperature: float = 1.0, threshold: Optional[float] = None, return_data: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, np.ndarray]]]:
    """OSR 관점에서 모델 평가 (FPR@TPR90 포함, 특징 벡터 반환)"""
    model.eval()
    # 특징 벡터 저장을 위한 리스트 추가
    id_logits_all, id_scores_all, id_labels_true, id_labels_pred, id_features_all = [], [], [], [], []
    ood_logits_all, ood_scores_all, ood_features_all = [], [], []

    with torch.no_grad():
        for batch in tqdm(id_loader, desc="Evaluating ID for OSR"):
            input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device); labels = batch['label']
            # 특징 벡터 함께 받기
            logits, features = model(input_ids, attention_mask, output_features=True)
            softmax_probs = F.softmax(logits / temperature, dim=1); max_probs, preds = softmax_probs.max(dim=1)
            id_logits_all.append(logits.cpu()); id_scores_all.append(max_probs.cpu()); id_labels_true.extend(labels.numpy()); id_labels_pred.extend(preds.cpu().numpy())
            id_features_all.append(features.cpu()) # 특징 벡터 저장

    with torch.no_grad():
        for batch in tqdm(ood_loader, desc="Evaluating OOD for OSR"):
            input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device)
            # 특징 벡터 함께 받기
            logits, features = model(input_ids, attention_mask, output_features=True)
            softmax_probs = F.softmax(logits / temperature, dim=1); max_probs, _ = softmax_probs.max(dim=1)
            ood_logits_all.append(logits.cpu()); ood_scores_all.append(max_probs.cpu())
            ood_features_all.append(features.cpu()) # 특징 벡터 저장

    id_scores = torch.cat(id_scores_all).numpy() if id_scores_all else np.array([])
    ood_scores = torch.cat(ood_scores_all).numpy() if ood_scores_all else np.array([])
    # 특징 벡터 합치기
    id_features = torch.cat(id_features_all).numpy() if id_features_all else np.array([])
    ood_features = torch.cat(ood_features_all).numpy() if ood_features_all else np.array([])

    # ... (이전 메트릭 계산 로직 동일) ...
    results = { "Closed_Set_Accuracy": 0.0, "F1_Macro": 0.0, "AUROC": 0.0, "FPR@TPR90": 1.0, "Open_Set_Error": 1.0, "CCR": 0.0, "OSCR": 0.0, "F1_Open": 0.0, "Threshold": threshold }
    all_data = {"id_scores": id_scores, "ood_scores": ood_scores, "id_labels_true": np.array(id_labels_true), "id_labels_pred": np.array(id_labels_pred),
                "id_features": id_features, "ood_features": ood_features} # 특징 벡터 추가
    if len(id_labels_true) == 0: print("Warning: No ID samples for OSR eval."); return results, all_data if return_data else results
    closed_set_acc = accuracy_score(id_labels_true, id_labels_pred); f1_macro = f1_score(id_labels_true, id_labels_pred, average='macro', zero_division=0)
    results["Closed_Set_Accuracy"] = closed_set_acc; results["F1_Macro"] = f1_macro
    if len(ood_scores) == 0: print("Warning: No OOD samples for OSR eval."); return results, all_data if return_data else results
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)]); y_scores = np.concatenate([id_scores, ood_scores])
    valid_indices = ~np.isnan(y_scores)
    if np.sum(valid_indices) < len(y_true): print(f"Warning: Found NaN scores."); y_true = y_true[valid_indices]; y_scores = y_scores[valid_indices]
    if len(np.unique(y_true)) < 2: print("Warning: Only one class type (ID or OOD). AUROC/FPR skipped.")
    else:
        results["AUROC"] = roc_auc_score(y_true, y_scores); fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
        if np.any(tpr >= 0.90): results["FPR@TPR90"] = fpr[np.where(tpr >= 0.90)[0][0]]
        else: print("Warning: TPR >= 0.90 not reached."); results["FPR@TPR90"] = 1.0
    if threshold is None:
        if len(id_scores) > 0: threshold = np.percentile(id_scores, 5); results["Threshold"] = threshold; print(f"  - Auto threshold: {threshold:.4f}")
        else: threshold = 0.5
    id_correct_known = (id_scores >= threshold); ood_correct_unknown = (ood_scores < threshold)
    if len(ood_scores) > 0: open_set_error = 1.0 - np.mean(ood_correct_unknown); results["Open_Set_Error"] = open_set_error
    else: open_set_error = 0.0
    if np.sum(id_correct_known) > 0: ccr = accuracy_score(np.array(id_labels_true)[id_correct_known], np.array(id_labels_pred)[id_correct_known]); results["CCR"] = ccr
    else: ccr = 0.0
    oscr = ccr * (1.0 - open_set_error); results["OSCR"] = oscr
    y_true_binary = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)]); y_pred_binary = np.concatenate([id_scores < threshold, ood_scores < threshold]).astype(int)
    if len(np.unique(y_true_binary)) == 2: f1_open = f1_score(y_true_binary, y_pred_binary, pos_label=1, average='binary', zero_division=0); results["F1_Open"] = f1_open

    if return_data: return results, all_data
    return results

# 시각화 함수들 (t-SNE 추가)
def plot_confidence_histograms(id_scores: np.ndarray, ood_scores: np.ndarray, title: str, save_path: Optional[str] = None):
    # ... (이전 코드 복사) ...
    plt.figure(figsize=(10, 6)); id_scores_valid = id_scores[~np.isnan(id_scores)]; ood_scores_valid = ood_scores[~np.isnan(ood_scores)]
    if len(id_scores_valid) > 0: sns.histplot(id_scores_valid, bins=50, alpha=0.5, label='In-Distribution', color='blue', stat='density', kde=True)
    if len(ood_scores_valid) > 0: sns.histplot(ood_scores_valid, bins=50, alpha=0.5, label='Out-of-Distribution', color='red', stat='density', kde=True)
    plt.xlabel('Confidence Score'); plt.ylabel('Density'); plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    if save_path: plt.savefig(save_path); plt.close()
    else: plt.show(); plt.close()

def plot_roc_curve(id_scores: np.ndarray, ood_scores: np.ndarray, title: str, save_path: Optional[str] = None):
    # ... (이전 코드 복사) ...
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)]); y_scores = np.concatenate([id_scores, ood_scores])
    valid_indices = ~np.isnan(y_scores); y_true = y_true[valid_indices]; y_scores = y_scores[valid_indices]
    if len(np.unique(y_true)) < 2: print(f"Skipping ROC plot for '{title}'."); return
    fpr, tpr, _ = roc_curve(y_true, y_scores); auroc = roc_auc_score(y_true, y_scores)
    plt.figure(figsize=(8, 8)); plt.plot(fpr, tpr, lw=2, label=f'AUROC = {auroc:.4f}'); plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.01, 1.01]); plt.ylim([-0.01, 1.01]); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(title); plt.legend(loc='lower right'); plt.grid(alpha=0.3)
    if save_path: plt.savefig(save_path); plt.close()
    else: plt.show(); plt.close()

def plot_osr_comparison(results: Dict[str, Dict[str, float]], save_path: Optional[str] = None):
    # ... (이전 코드 복사, FPR@TPR90 포함) ...
    osr_results = {k: v for k, v in results.items() if '+' in k}
    if not osr_results: print("No OSR results to compare."); return
    datasets = sorted(list(set(k.split('+')[1] for k in osr_results)))
    methods = sorted(list(set(k.split('+')[0] for k in osr_results)))
    metrics = ['Closed_Set_Accuracy', 'F1_Macro', 'AUROC', 'FPR@TPR90', 'Open_Set_Error', 'CCR', 'OSCR', 'F1_Open']
    num_metrics = len(metrics); fig, axes = plt.subplots(num_metrics, 1, figsize=(max(10, len(datasets) * 1.5), 5 * num_metrics), squeeze=False); axes = axes.flatten()
    x = np.arange(len(datasets)); total_width = 0.8; width = total_width / len(methods)
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for j, method in enumerate(methods):
            values = [osr_results.get(f"{method}+{dataset}", {}).get(metric, np.nan) for dataset in datasets]
            if metric in ['Open_Set_Error', 'FPR@TPR90']: plot_values = [1.0 - v if pd.notna(v) else 0 for v in values]; ax.set_ylabel(f'1 - {metric}'); ax.set_title(f'1 - {metric} (Higher is Better)')
            else: plot_values = [v if pd.notna(v) else 0 for v in values]; ax.set_ylabel(metric); ax.set_title(f'{metric} (Higher is Better)')
            ax.set_ylim(0, 1.1); offset = width * j - total_width / 2 + width / 2; rects = ax.bar(x + offset, plot_values, width, label=method)
            ax.bar_label(rects, padding=3, fmt='%.3f', fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=15, ha='right'); ax.legend(loc='upper left', bbox_to_anchor=(1, 1)); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_path: plt.savefig(save_path); plt.close()
    else: plt.show(); plt.close()

def threshold_analysis(id_scores: np.ndarray, ood_scores: np.ndarray, title: str, save_path: Optional[str] = None) -> Optional[Tuple[float, float, float, float]]:
    # ... (이전 코드 복사) ...
    id_scores_valid = id_scores[~np.isnan(id_scores)]; ood_scores_valid = ood_scores[~np.isnan(ood_scores)]
    if len(id_scores_valid) == 0 or len(ood_scores_valid) == 0: print(f"Skipping threshold analysis for '{title}'."); return None
    all_scores = np.concatenate([id_scores_valid, ood_scores_valid]); min_score, max_score = np.min(all_scores), np.max(all_scores)
    thresholds = np.linspace(min_score - 1e-5, max_score + 1e-5, 200); tpr_values, fpr_values = [], []
    for threshold in thresholds:
        tpr = np.mean(id_scores_valid >= threshold) if len(id_scores_valid) > 0 else 0.0
        fpr = np.mean(ood_scores_valid >= threshold) if len(ood_scores_valid) > 0 else 0.0
        tpr_values.append(tpr); fpr_values.append(fpr)
    tpr_values = np.array(tpr_values); fpr_values = np.array(fpr_values)
    accuracies = (tpr_values * len(id_scores_valid) + (1 - fpr_values) * len(ood_scores_valid)) / (len(id_scores_valid) + len(ood_scores_valid))
    best_idx = np.argmax(accuracies); best_threshold = thresholds[best_idx]; best_accuracy = accuracies[best_idx]
    tpr90_idx = np.argmin(np.abs(tpr_values - 0.90)); tpr90_threshold = thresholds[tpr90_idx]; fpr90 = fpr_values[tpr90_idx]
    plt.figure(figsize=(10, 6)); plt.plot(thresholds, tpr_values, label='TPR (ID Recall)'); plt.plot(thresholds, fpr_values, label='FPR (OOD False Alarm)')
    plt.plot(thresholds, accuracies, label='Accuracy (OOD Detection)')
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold = {best_threshold:.3f} (Acc = {best_accuracy:.3f})')
    plt.axvline(x=tpr90_threshold, color='g', linestyle='--', label=f'Threshold for TPR≈0.90 (FPR = {fpr90:.3f})')
    plt.xlabel('Confidence Threshold'); plt.ylabel('Rate / Accuracy'); plt.title(title); plt.legend(); plt.grid(alpha=0.3); plt.ylim(-0.05, 1.05)
    if save_path: plt.savefig(save_path); plt.close()
    else: plt.show(); plt.close()
    return best_threshold, best_accuracy, tpr90_threshold, fpr90

# --- 추가된 시각화 함수 ---
def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, save_path: Optional[str] = None):
    """Confusion Matrix 시각화 및 저장"""
    if not SNS_AVAILABLE:
        print("Seaborn not available, skipping confusion matrix plot.")
        return
    if cm.shape[0] != len(class_names) or cm.shape[1] != len(class_names):
        print(f"Warning: Mismatch between confusion matrix shape {cm.shape} and class names length {len(class_names)}. Skipping plot.")
        return

    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(max(8, len(class_names)*0.5), max(6, len(class_names)*0.4))) # 클래스 수에 따라 크기 조절
    try:
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
        plt.title(title)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix plot saved to: {save_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
        plt.close() # 오류 발생 시에도 plot 닫기

def plot_tsne(id_features: np.ndarray, ood_features: np.ndarray, title: str, save_path: Optional[str] = None, seed: int = 42, perplexity: int = 30, n_iter: int = 1000):
    """t-SNE 시각화 (ID vs OOD)"""
    if len(id_features) == 0 and len(ood_features) == 0:
        print("No features to plot for t-SNE.")
        return
    if len(id_features) > 0 and len(ood_features) > 0:
        features = np.vstack((id_features, ood_features))
        labels = np.concatenate([np.ones(len(id_features)), np.zeros(len(ood_features))]) # 1: ID, 0: OOD
        legend_labels = {1: 'In-Distribution', 0: 'Out-of-Distribution'}
        colors = {1: 'blue', 0: 'red'}
    elif len(id_features) > 0:
        features = id_features
        labels = np.ones(len(id_features))
        legend_labels = {1: 'In-Distribution'}
        colors = {1: 'blue'}
    else: # Only OOD
        features = ood_features
        labels = np.zeros(len(ood_features))
        legend_labels = {0: 'Out-of-Distribution'}
        colors = {0: 'red'}

    print(f"Running t-SNE on {features.shape[0]} samples (perplexity={perplexity})...")
    try:
        tsne = TSNE(n_components=2, random_state=seed, perplexity=min(perplexity, features.shape[0] - 1), n_iter=n_iter, init='pca', learning_rate='auto') # perplexity 조정
        tsne_results = tsne.fit_transform(features)
    except Exception as e:
        print(f"Error running t-SNE: {e}. Skipping plot.")
        return

    plt.figure(figsize=(10, 8))
    for label_val, label_name in legend_labels.items():
        indices = labels == label_val
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], c=colors[label_val], label=label_name, alpha=0.6, s=10)

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        print(f"t-SNE plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()

def train_with_oe_uniform_loss(model: nn.Module, train_loader: DataLoader, oe_loader: DataLoader, optimizer: optim.Optimizer, scheduler, device: torch.device, num_epochs: int, oe_lambda: float):
    model.train()
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print("Starting training with Outlier Exposure (Uniform CE Loss)... AMP enabled: ", use_amp)

    for epoch in range(num_epochs):
        oe_iter = iter(oe_loader)
        total_loss = 0; total_id_loss = 0; total_oe_loss = 0
        progress_bar = tqdm(train_loader, desc=f"OE Uniform Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            # ... (ID 데이터 로드 및 처리) ...
            input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device); labels = batch['label'].to(device)
            # ... (OE 데이터 로드 및 처리) ...
            try: oe_batch = next(oe_iter)
            except StopIteration: oe_iter = iter(oe_loader); oe_batch = next(oe_iter)
            oe_input_ids = oe_batch['input_ids'].to(device); oe_attention_mask = oe_batch['attention_mask'].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=str(device).split(':')[0], enabled=use_amp):
                # ID 데이터 손실 (Cross Entropy)
                logits = model(input_ids, attention_mask)
                id_loss = F.cross_entropy(logits, labels)

                # OE 데이터 손실 (Uniform Cross-Entropy)
                oe_logits = model(oe_input_ids, oe_attention_mask)
                num_classes = oe_logits.size(1)
                # LogSoftmax 적용 후 KLDivLoss 사용 (CrossEntropy와 동일 효과)
                log_softmax_oe = F.log_softmax(oe_logits, dim=1)
                # 타겟 분포: Uniform Distribution (모든 클래스 확률이 1/k)
                # KLDivLoss는 log-probability 입력과 probability 타겟을 기대함
                uniform_target = torch.full_like(oe_logits, 1.0 / num_classes)
                # reduction='batchmean' : 배치 전체의 평균 손실 계산
                oe_loss = F.kl_div(log_softmax_oe, uniform_target, reduction='batchmean')

                total_batch_loss = id_loss + oe_lambda * oe_loss

            scaler.scale(total_batch_loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step()
            total_loss += total_batch_loss.item(); total_id_loss += id_loss.item(); total_oe_loss += oe_loss.item()
            progress_bar.set_postfix({'Total Loss': total_batch_loss.item(), 'ID Loss': id_loss.item(), 'OE Loss': oe_loss.item()})
        avg_loss = total_loss / len(train_loader); avg_id_loss = total_id_loss / len(train_loader); avg_oe_loss = total_oe_loss / len(train_loader)
        print(f"OE Uniform Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f} (ID: {avg_id_loss:.4f}, OE: {avg_oe_loss:.4f})")
        
# 메인 함수 (수정됨 - 시각화 호출 추가)
def main(args: Dict[str, Any]):
    set_seed(args['seed']); os.makedirs(args['result_dir'], exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(CACHE_DIR, exist_ok=True)
    if args['save_model']: os.makedirs(args['model_dir'], exist_ok=True)

    max_length = args['max_length']; batch_size = args['batch_size']; num_epochs = args['num_epochs']; learning_rate = args['learning_rate']
    oe_lambda = args['oe_lambda']; temperature = args['temperature']; model_type = args['model_type']; oe_sources = args['oe_sources']

    print(f"Loading RoBERTa tokenizer: {model_type}..."); tokenizer = RobertaTokenizer.from_pretrained(model_type)

    print("\n--- Preparing Datasets ---")
    # !!! 최종 클래스 이름 얻기 위해 final_id2label 사용 !!!
    train_dataset, id_test_dataset, num_classes, final_label_encoder, final_label2id, final_id2label = prepare_syslog_data(
        tokenizer, max_length, args['id_data_path'], args['text_col'], args['class_col'], args['id_exclude_class'], args['seed'])
    ood_test_dataset = prepare_syslog_ood_data(tokenizer, max_length, args['ood_data_path'], args['text_col'], args['class_col'], args['ood_target_class'])
    # 최종 Known 클래스 이름 리스트
    known_class_names = list(final_id2label.values())

    print("\n--- Creating Base DataLoaders ---")
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=num_workers>0, pin_memory=True)
    id_test_loader = DataLoader(id_test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    ood_test_loader = DataLoader(ood_test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if ood_test_dataset else None

    performance_metrics = {}; all_data = {}

    # --- 1. 표준 모델 (OE 없음) ---
    model_std = None
    std_model_filename = f'roberta_standard_{num_classes}cls_seed{args["seed"]}.pt'
    std_model_path = os.path.join(args['model_dir'], std_model_filename) if args['save_model'] else args['std_model_path']
    if not args['skip_standard']:
        print("\n--- Standard Model Training/Evaluation ---")
        model_std = RoBERTaOOD(num_classes, model_type).to(device)
        model_std.roberta.config.label2id = final_label2id; model_std.roberta.config.id2label = final_id2label
        if args['eval_only']:
            load_path = args['std_model_path'] if args['std_model_path'] else std_model_path
            if load_path and os.path.exists(load_path): print(f"Loading model from {load_path}..."); model_std.load_state_dict(torch.load(load_path, map_location=device))
            else: print(f"Error: Model path '{load_path}' not found."); model_std = None
        else:
            optimizer_std = AdamW(model_std.parameters(), lr=learning_rate)
            total_steps = len(train_loader) * num_epochs; scheduler_std = get_linear_schedule_with_warmup(optimizer_std, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
            train_standard(model_std, train_loader, optimizer_std, scheduler_std, device, num_epochs)
            if args['save_model']: torch.save(model_std.state_dict(), std_model_path); print(f"Model saved to {std_model_path}")

        if model_std and ood_test_loader:
            print("\nEvaluating Standard Model with OSR metrics...")
            results_std_osr, data_std_osr = evaluate_osr(model_std, id_test_loader, ood_test_loader, device, temperature=temperature, return_data=True)
            print(f"  OSR Results (vs {args['ood_target_class']}): {results_std_osr}")
            performance_metrics[f'Standard+{args["ood_target_class"]}'] = results_std_osr
            all_data[f'Standard+{args["ood_target_class"]}'] = data_std_osr
            if not args['no_plot']:
                # 표준 모델 시각화
                plot_confidence_histograms(data_std_osr['id_scores'], data_std_osr['ood_scores'], f'OSR Confidence - Standard vs {args["ood_target_class"]}', os.path.join(args['result_dir'], f'standard_osr_{args["ood_target_class"]}_hist.png'))
                plot_roc_curve(data_std_osr['id_scores'], data_std_osr['ood_scores'], f'OSR ROC - Standard vs {args["ood_target_class"]}', os.path.join(args['result_dir'], f'standard_osr_{args["ood_target_class"]}_roc.png'))
                threshold_analysis(data_std_osr['id_scores'], data_std_osr['ood_scores'], f'OSR Threshold Analysis - Standard vs {args["ood_target_class"]}', os.path.join(args['result_dir'], f'standard_osr_{args["ood_target_class"]}_threshold.png'))
                # Confusion Matrix 계산 및 시각화
                cm_std = confusion_matrix(data_std_osr['id_labels_true'], data_std_osr['id_labels_pred'], labels=np.arange(num_classes)) # 레이블 순서 고정
                plot_confusion_matrix(cm_std, known_class_names, f'Confusion Matrix - Standard (ID Test)', os.path.join(args['result_dir'], f'standard_confusion_matrix.png'))
                # t-SNE 시각화
                plot_tsne(data_std_osr['id_features'], data_std_osr['ood_features'], f't-SNE - Standard (ID vs OOD: {args["ood_target_class"]})', os.path.join(args['result_dir'], f'standard_tsne.png'), seed=args['seed'])

    # --- 2. OE 모델 (Confidence Loss) - 여러 소스에 대해 반복 ---
    if not args['skip_oe_all']:
        for oe_source in oe_sources:
            print(f"\n===== Running OE Experiment with Source: {oe_source} =====")
            # ... (OE 데이터 로드 로직) ...
            oe_train_dataset_current = None
            if oe_source == 'syslog_masked': oe_train_dataset_current = prepare_syslog_masked_oe_data(tokenizer, max_length, args['oe_masked_syslog_path'], args['oe_masked_text_col'])
            elif oe_source in ['snli', 'imdb', 'wikitext']: oe_train_dataset_current = prepare_external_oe_data(tokenizer, max_length, oe_source, data_dir=DATA_DIR, cache_dir=CACHE_DIR)
            else: print(f"Warning: Skipping invalid OE source '{oe_source}'"); continue
            if oe_train_dataset_current is None: print(f"Skipping OE training for source '{oe_source}'."); continue
            oe_train_loader_current = DataLoader(oe_train_dataset_current, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

            model_oe = None; oe_model_filename = f'roberta_oe_conf_{oe_source}_{num_classes}cls_seed{args["seed"]}.pt'
            oe_model_path = os.path.join(args['model_dir'], oe_model_filename)
            print(f"\n--- Outlier Exposure (Source: {oe_source}) Model Training/Evaluation ---")
            model_oe = RoBERTaOOD(num_classes, model_type).to(device)
            model_oe.roberta.config.label2id = final_label2id; model_oe.roberta.config.id2label = final_id2label
            if args['eval_only']:
                load_path = oe_model_path
                if load_path and os.path.exists(load_path): print(f"Loading model ({oe_source}) from {load_path}..."); model_oe.load_state_dict(torch.load(load_path, map_location=device))
                else: print(f"Error: Model path '{load_path}' not found."); model_oe = None
            else:
                optimizer_oe = AdamW(model_oe.parameters(), lr=learning_rate)
                total_steps = len(train_loader) * num_epochs; scheduler_oe = get_linear_schedule_with_warmup(optimizer_oe, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
                # train_with_oe_confidence_loss(model_oe, train_loader, oe_train_loader_current, optimizer_oe, scheduler_oe, device, num_epochs, oe_lambda)
                train_with_oe_uniform_loss(model_oe, train_loader, oe_train_loader_current, optimizer_oe, scheduler_oe, device, num_epochs, oe_lambda)
                if args['save_model']: torch.save(model_oe.state_dict(), oe_model_path); print(f"Model (Source: {oe_source}) saved to {oe_model_path}")

            if model_oe and ood_test_loader:
                print(f"\nEvaluating OE Model (Source: {oe_source}) with OSR metrics...")
                results_oe_osr, data_oe_osr = evaluate_osr(model_oe, id_test_loader, ood_test_loader, device, temperature=temperature, return_data=True)
                print(f"  OSR Results (vs {args['ood_target_class']}): {results_oe_osr}")
                performance_metrics[f'OE_Conf_{oe_source}+{args["ood_target_class"]}'] = results_oe_osr
                all_data[f'OE_Conf_{oe_source}+{args["ood_target_class"]}'] = data_oe_osr
                if not args['no_plot']:
                    # OE 모델 시각화
                    plot_confidence_histograms(data_oe_osr['id_scores'], data_oe_osr['ood_scores'], f'OSR Confidence - OE {oe_source} vs {args["ood_target_class"]}', os.path.join(args['result_dir'], f'oe_conf_{oe_source}_osr_{args["ood_target_class"]}_hist.png'))
                    plot_roc_curve(data_oe_osr['id_scores'], data_oe_osr['ood_scores'], f'OSR ROC - OE {oe_source} vs {args["ood_target_class"]}', os.path.join(args['result_dir'], f'oe_conf_{oe_source}_osr_{args["ood_target_class"]}_roc.png'))
                    threshold_analysis(data_oe_osr['id_scores'], data_oe_osr['ood_scores'], f'OSR Threshold Analysis - OE {oe_source} vs {args["ood_target_class"]}', os.path.join(args['result_dir'], f'oe_conf_{oe_source}_osr_{args["ood_target_class"]}_threshold.png'))
                    # Confusion Matrix 계산 및 시각화
                    cm_oe = confusion_matrix(data_oe_osr['id_labels_true'], data_oe_osr['id_labels_pred'], labels=np.arange(num_classes))
                    plot_confusion_matrix(cm_oe, known_class_names, f'Confusion Matrix - OE {oe_source} (ID Test)', os.path.join(args['result_dir'], f'oe_conf_{oe_source}_confusion_matrix.png'))
                    # t-SNE 시각화
                    plot_tsne(data_oe_osr['id_features'], data_oe_osr['ood_features'], f't-SNE - OE {oe_source} (ID vs OOD: {args["ood_target_class"]})', os.path.join(args['result_dir'], f'oe_conf_{oe_source}_tsne.png'), seed=args['seed'])
            # ... (이하 생략) ...
            del model_oe, oe_train_dataset_current, oe_train_loader_current
            if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    # --- 결과 요약 및 저장 ---
    print("\n--- Final OSR Results Summary ---")
    # ... (이전 결과 저장 및 plot_osr_comparison 호출 로직 동일) ...
    results_df = None
    if performance_metrics:
        results_df = pd.DataFrame(performance_metrics).T.sort_index()
        print("Performance Metrics DataFrame:"); print(results_df) # FPR@TPR90 포함됨
        base_filename = f'osr_syslog_comparison_summary_seed{args["seed"]}'
        csv_results_file = os.path.join(args['result_dir'], f'{base_filename}.csv'); txt_results_file = os.path.join(args['result_dir'], f'{base_filename}.txt'); json_results_file = os.path.join(args['result_dir'], f'{base_filename}.json')
        try: results_df.to_csv(csv_results_file, index=True); print(f"\nResults saved to CSV: {csv_results_file}")
        except Exception as e: print(f"\nError saving to CSV: {e}")
        try:
            with open(txt_results_file, 'w', encoding='utf-8') as f: f.write("--- Args ---\n"); f.write(json.dumps(args, indent=4)); f.write("\n\n--- Metrics ---\n"); f.write(results_df.to_string())
            print(f"Results saved to TXT: {txt_results_file}")
        except Exception as e: print(f"\nError saving to TXT: {e}")
        results_data = {'metrics': performance_metrics, 'arguments': args, 'timestamp': datetime.now().isoformat()}
        try:
            with open(json_results_file, 'w', encoding='utf-8') as f: json.dump(results_data, f, indent=4, default=str)
            print(f"Results saved to JSON: {json_results_file}")
        except Exception as e: print(f"\nError saving to JSON: {e}")

        if len(performance_metrics) > 0 and not args['no_plot']:
             print("\nGenerating OSR comparison plot...")
             plot_osr_comparison(performance_metrics, os.path.join(args['result_dir'], f'osr_methods_comparison_syslog_seed{args["seed"]}.png'))
    else: print("No performance metrics generated.")
    print("Experiment finished.")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RoBERTa OSR Comparison with Multiple OE Sources on Syslog Data')
    # ... (ArgumentParser 설정 - 이전과 동일, oe_sources 포함) ...
    parser.add_argument('--id_data_path', type=str, default=ID_DATA_PATH)
    parser.add_argument('--oe_masked_syslog_path', type=str, default=OE_MASKED_SYSLOG_PATH)
    parser.add_argument('--ood_data_path', type=str, default=OOD_DATA_PATH)
    parser.add_argument('--text_col', type=str, default=TEXT_COLUMN)
    parser.add_argument('--class_col', type=str, default=CLASS_COLUMN)
    parser.add_argument('--oe_masked_text_col', type=str, default=OE_MASKED_TEXT_COLUMN)
    parser.add_argument('--id_exclude_class', type=str, default=ID_EXCLUDE_CLASS)
    parser.add_argument('--ood_target_class', type=str, default=OOD_TARGET_CLASS)
    parser.add_argument('--max_length', type=int, default=MAX_LENGTH)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--oe_sources', nargs='+', default=OE_SOURCES_TO_RUN, choices=['syslog_masked', 'snli', 'imdb', 'wikitext'])
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--oe_lambda', type=float, default=OE_LAMBDA)
    parser.add_argument('--temperature', type=float, default=TEMPERATURE)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--model_type', type=str, default=MODEL_TYPE)
    parser.add_argument('--save_model', default=SAVE_MODEL, action=argparse.BooleanOptionalAction)
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR)
    parser.add_argument('--skip_standard', default=SKIP_STANDARD, action=argparse.BooleanOptionalAction)
    parser.add_argument('--skip_oe_all', default=SKIP_OE_ALL, action=argparse.BooleanOptionalAction, help='Skip all OE experiments')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--std_model_path', type=str, default=STD_MODEL_PATH)
    # oe_model_path 제거
    parser.add_argument('--result_dir', type=str, default=RESULT_DIR)
    parser.add_argument('--no_plot', default=NO_PLOT, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    args_dict = vars(args)

    if args_dict['eval_only'] and not args_dict['skip_oe_all']:
        print("Running in eval_only mode. Will look for pre-trained OE models based on naming convention.")

    main(args_dict)