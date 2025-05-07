# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split, ConcatDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.manifold import TSNE
# 20 Newsgroups 로딩용
from sklearn.datasets import fetch_20newsgroups
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import random
import matplotlib
import gc
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError: SNS_AVAILABLE = False; print("Warning: Seaborn not installed.")
import pandas as pd
import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union
from datasets import load_dataset, DatasetDict, concatenate_datasets
import re

# --- 설정값 (이전과 동일) ---
ID_SYSLOG_PATH = 'log_all_critical.csv'
OE_MASKED_SYSLOG_PATH = None # 명령줄에서 받아야 함
OOD_SYSLOG_UNKNOWN_PATH = 'log_unknown.csv'
TEXT_COLUMN = 'text'
CLASS_COLUMN = 'class'
OE_MASKED_TEXT_COLUMN = 'masked_text_attention'
ID_SYSLOG_EXCLUDE_CLASS = "unknown"
OOD_SYSLOG_TARGET_CLASS = "unknown"
NUM_20NG_ID_CLASSES = 10
# OE_SOURCES_TO_RUN = ['syslog_masked', 'snli', 'imdb'] # argparse에서 기본값 설정
RESULT_DIR_BASE = 'results_osr_comparison_multi_dataset'
MODEL_DIR_BASE = 'models_osr_comparison_multi_dataset'
DATA_DIR = 'data'
CACHE_DIR = os.path.join(DATA_DIR, "hf_cache")
MODEL_TYPE = 'roberta-base'
MAX_LENGTH = 128
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 2e-5
OE_LAMBDA = 1.0
TEMPERATURE = 1.0
SEED = 42
SAVE_MODEL = True
EVAL_ONLY = False
NO_PLOT = False
SKIP_STANDARD = False
SKIP_OE_ALL = False

# --- 함수 정의 (이전 코드와 동일, 변경 없음) ---
def set_seed(seed: int):
    # ... (코드 동일) ...
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    print(f"Seed set to {seed}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(Dataset):
    # ... (코드 동일) ...
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.labels = labels; # print(f"Tokenizing {len(texts)} texts...") # 로그 축소
        valid_texts = [str(t) if pd.notna(t) else "" for t in texts]
        self.encodings = tokenizer(valid_texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        # print("Tokenization complete.") # 로그 축소
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}; item['label'] = torch.tensor(self.labels[idx], dtype=torch.long); return item

def prepare_syslog_data(tokenizer, max_length: int, id_data_path: str, text_col: str, class_col: str, exclude_class: str, seed: int = 42) -> Tuple[Dataset, Dataset, int, LabelEncoder, Dict, Dict]:
    # ... (코드 동일) ...
    print(f"\n--- Preparing Syslog ID data from: {id_data_path} ---")
    try:
        df = pd.read_csv(id_data_path);
        if not all(c in df.columns for c in [text_col, class_col]): raise ValueError(f"CSV must contain '{text_col}' and '{class_col}' columns.")
        df = df.dropna(subset=[text_col, class_col]); df[class_col] = df[class_col].astype(str)
        df_known_initial = df[df[class_col] != exclude_class].copy()
        if df_known_initial.empty: raise ValueError(f"No data left after excluding class '{exclude_class}'.")
        print(f"  - Initial known data size: {len(df_known_initial)}")
        initial_label_encoder = LabelEncoder(); initial_label_encoder.fit(df_known_initial[class_col])
        print(f"  - Filtering known classes (min 2 samples)...")
        class_counts = df_known_initial[class_col].value_counts()
        classes_to_keep = class_counts[class_counts >= 2].index; classes_removed = class_counts[class_counts < 2].index
        if len(classes_removed) > 0: print(f"  - Removing classes: {classes_removed.tolist()}"); df_known_final = df_known_initial[df_known_initial[class_col].isin(classes_to_keep)].copy()
        else: df_known_final = df_known_initial; print("  - All classes have >= 2 samples.")
        if df_known_final.empty: raise ValueError("No data left after filtering.")
        final_classes = sorted(df_known_final[class_col].unique()); num_classes_final = len(final_classes)
        print(f"  - Final number of known classes: {num_classes_final}"); # print(f"  - Final known classes: {final_classes}") # 로그 축소
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

def prepare_20newsgroups_data(tokenizer, max_length: int, num_id_classes: int, seed: int = 42) -> Tuple[Dataset, Dataset, int, LabelEncoder, Dict, Dict, List[str]]:
    # ... (코드 동일) ...
    print(f"\n--- Preparing 20 Newsgroups ID data ({num_id_classes} classes) ---")
    try:
        all_categories = list(fetch_20newsgroups(subset='train').target_names)
        if num_id_classes > len(all_categories): num_id_classes = len(all_categories)
        np.random.seed(seed)
        id_category_indices = np.random.choice(len(all_categories), num_id_classes, replace=False)
        in_dist_categories = [all_categories[i] for i in id_category_indices]
        out_dist_categories = [cat for i, cat in enumerate(all_categories) if i not in id_category_indices]
        print(f"  - Selected {num_id_classes} ID categories: {in_dist_categories}")
        print(f"  - Remaining {len(out_dist_categories)} categories considered OOD within 20NG.")
        id_train_sk = fetch_20newsgroups(subset='train', categories=in_dist_categories, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=seed)
        id_test_sk = fetch_20newsgroups(subset='test', categories=in_dist_categories, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=seed)
        label_encoder = LabelEncoder()
        id_train_labels = label_encoder.fit_transform(id_train_sk.target)
        id_test_labels = label_encoder.transform(id_test_sk.target)
        final_label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
        final_id2label = {i: label for label, i in final_label2id.items()}
        num_classes_final = len(label_encoder.classes_)
        print(f"  - Final number of known classes: {num_classes_final}")
        print(f"  - Final Label to ID mapping: {final_label2id}")
        train_dataset = TextDataset(id_train_sk.data, id_train_labels, tokenizer, max_length)
        id_test_dataset = TextDataset(id_test_sk.data, id_test_labels, tokenizer, max_length)
        print(f"  - Split into Train: {len(train_dataset)}, Test: {len(id_test_dataset)}")
        return train_dataset, id_test_dataset, num_classes_final, label_encoder, final_label2id, final_id2label, out_dist_categories
    except Exception as e: print(f"Error preparing 20 Newsgroups ID data: {e}"); raise

def prepare_20newsgroups_ood_data(tokenizer, max_length: int, ood_categories: List[str], seed: int = 42) -> Optional[Dataset]:
    # ... (코드 동일) ...
    if not ood_categories: return None
    print(f"\n--- Preparing 20 Newsgroups OOD data (Categories: {ood_categories}) ---")
    try:
        ood_test_sk = fetch_20newsgroups(subset='test', categories=ood_categories, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=seed)
        if len(ood_test_sk.data) == 0: print(f"Warning: No data found for OOD categories: {ood_categories}"); return None
        texts = ood_test_sk.data
        ood_labels = np.full(len(texts), -1, dtype=int)
        ood_dataset = TextDataset(texts, ood_labels, tokenizer, max_length)
        print(f"  - Loaded {len(ood_dataset)} samples for OOD testing.")
        return ood_dataset
    except Exception as e: print(f"Error preparing 20 Newsgroups OOD data: {e}"); return None

def prepare_syslog_masked_oe_data(tokenizer, max_length: int, oe_data_path: str, oe_text_col: str) -> Optional[Dataset]:
    # ... (코드 동일) ...
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
    # ... (코드 동일) ...
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
    # ... (코드 동일) ...
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

class RoBERTaOOD(nn.Module):
    # ... (코드 동일) ...
    def __init__(self, num_classes: int, model_name: str = 'roberta-base'):
        super(RoBERTaOOD, self).__init__(); # print(f"Initializing RoBERTa model ({model_name}) for {num_classes} classes.") # 로그 축소
        config = RobertaConfig.from_pretrained(model_name); config.num_labels = num_classes
        config.torchscript = True; config.use_cache = True; config.return_dict = True; config.output_hidden_states = True
        self.roberta = RobertaForSequenceClassification.from_pretrained(model_name, config=config)
    def forward(self, input_ids, attention_mask, output_features=False):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits
        if output_features: features = outputs.hidden_states[-1][:, 0, :]; return logits, features
        else: return logits

def train_with_oe_uniform_loss(model: nn.Module, train_loader: DataLoader, oe_loader: DataLoader, optimizer: optim.Optimizer, scheduler, device: torch.device, num_epochs: int, oe_lambda: float):
    # ... (코드 동일) ...
    model.train(); use_amp = device.type == 'cuda'; scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print("Starting training with Outlier Exposure (Uniform CE Loss)... AMP enabled: ", use_amp)
    for epoch in range(num_epochs):
        oe_iter = iter(oe_loader); total_loss = 0; total_id_loss = 0; total_oe_loss = 0
        progress_bar = tqdm(train_loader, desc=f"OE Uniform Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device); labels = batch['label'].to(device)
            try: oe_batch = next(oe_iter)
            except StopIteration: oe_iter = iter(oe_loader); oe_batch = next(oe_iter)
            oe_input_ids = oe_batch['input_ids'].to(device); oe_attention_mask = oe_batch['attention_mask'].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=str(device).split(':')[0], enabled=use_amp):
                logits = model(input_ids, attention_mask); id_loss = F.cross_entropy(logits, labels)
                oe_logits = model(oe_input_ids, oe_attention_mask); num_classes = oe_logits.size(1)
                log_softmax_oe = F.log_softmax(oe_logits, dim=1)
                uniform_target = torch.full_like(oe_logits, 1.0 / num_classes)
                oe_loss = F.kl_div(log_softmax_oe, uniform_target, reduction='batchmean') # Uniform CE Loss
                total_batch_loss = id_loss + oe_lambda * oe_loss
            scaler.scale(total_batch_loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step()
            total_loss += total_batch_loss.item(); total_id_loss += id_loss.item(); total_oe_loss += oe_loss.item()
            progress_bar.set_postfix({'Total Loss': f"{total_batch_loss.item():.3f}", 'ID Loss': f"{id_loss.item():.3f}", 'OE Loss': f"{oe_loss.item():.3f}"})
        avg_loss = total_loss / len(train_loader); avg_id_loss = total_id_loss / len(train_loader); avg_oe_loss = total_oe_loss / len(train_loader)
        print(f"OE Uniform Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f} (ID: {avg_id_loss:.4f}, OE: {avg_oe_loss:.4f})")

def train_standard(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, scheduler, device: torch.device, num_epochs: int):
    # ... (코드 동일) ...
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
            total_loss += loss.item(); progress_bar.set_postfix({'loss': f"{loss.item():.3f}"})
        scheduler.step(); avg_loss = total_loss / len(train_loader)
        print(f"Standard Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

def evaluate_osr(model: nn.Module, id_loader: DataLoader, ood_loader: DataLoader, device: torch.device, temperature: float = 1.0, threshold: Optional[float] = None, return_data: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, np.ndarray]]]:
    # ... (코드 동일) ...
    model.eval(); id_logits_all, id_scores_all, id_labels_true, id_labels_pred, id_features_all = [], [], [], [], []; ood_logits_all, ood_scores_all, ood_features_all = [], [], []
    with torch.no_grad():
        for batch in tqdm(id_loader, desc="Evaluating ID for OSR"):
            input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device); labels = batch['label']
            logits, features = model(input_ids, attention_mask, output_features=True)
            softmax_probs = F.softmax(logits / temperature, dim=1); max_probs, preds = softmax_probs.max(dim=1)
            id_logits_all.append(logits.cpu()); id_scores_all.append(max_probs.cpu()); id_labels_true.extend(labels.numpy()); id_labels_pred.extend(preds.cpu().numpy()); id_features_all.append(features.cpu())
    with torch.no_grad():
        for batch in tqdm(ood_loader, desc="Evaluating OOD for OSR"):
            input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device)
            logits, features = model(input_ids, attention_mask, output_features=True)
            softmax_probs = F.softmax(logits / temperature, dim=1); max_probs, _ = softmax_probs.max(dim=1)
            ood_logits_all.append(logits.cpu()); ood_scores_all.append(max_probs.cpu()); ood_features_all.append(features.cpu())
    id_scores = torch.cat(id_scores_all).numpy() if id_scores_all else np.array([]); ood_scores = torch.cat(ood_scores_all).numpy() if ood_scores_all else np.array([])
    id_features = torch.cat(id_features_all).numpy() if id_features_all else np.array([]); ood_features = torch.cat(ood_features_all).numpy() if ood_features_all else np.array([])
    results = { "Closed_Set_Accuracy": 0.0, "F1_Macro": 0.0, "AUROC": 0.0, "FPR@TPR90": 1.0, "Open_Set_Error": 1.0, "CCR": 0.0, "OSCR": 0.0, "F1_Open": 0.0, "Threshold": threshold }
    all_data = {"id_scores": id_scores, "ood_scores": ood_scores, "id_labels_true": np.array(id_labels_true), "id_labels_pred": np.array(id_labels_pred), "id_features": id_features, "ood_features": ood_features}
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
        if len(id_scores) > 0: threshold = np.percentile(id_scores, 5); results["Threshold"] = threshold; # print(f"  - Auto threshold: {threshold:.4f}") # 로그 축소
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

def plot_confidence_histograms(id_scores: np.ndarray, ood_scores: np.ndarray, title: str, save_path: Optional[str] = None):
    # ... (코드 동일) ...
    if not SNS_AVAILABLE: return
    plt.figure(figsize=(10, 6)); id_scores_valid = id_scores[~np.isnan(id_scores)]; ood_scores_valid = ood_scores[~np.isnan(ood_scores)]
    if len(id_scores_valid) > 0: sns.histplot(id_scores_valid, bins=50, alpha=0.5, label='In-Distribution', color='blue', stat='density', kde=True)
    if len(ood_scores_valid) > 0: sns.histplot(ood_scores_valid, bins=50, alpha=0.5, label='Out-of-Distribution', color='red', stat='density', kde=True)
    plt.xlabel('Confidence Score'); plt.ylabel('Density'); plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    if save_path: plt.savefig(save_path); plt.close()
    else: plt.show(); plt.close()

def plot_roc_curve(id_scores: np.ndarray, ood_scores: np.ndarray, title: str, save_path: Optional[str] = None):
    # ... (코드 동일) ...
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
    # ... (코드 동일) ...
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
    # ... (코드 동일) ...
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

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, save_path: Optional[str] = None):
    # ... (코드 동일) ...
    if not SNS_AVAILABLE: print("Seaborn not available, skipping confusion matrix plot."); return
    if cm.shape[0] != len(class_names) or cm.shape[1] != len(class_names): print(f"Warning: Mismatch CM shape {cm.shape} and class names {len(class_names)}. Skipping plot."); return
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(max(8, len(class_names)*0.5), max(6, len(class_names)*0.4)))
    try:
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues"); plt.title(title); plt.ylabel('Actual Label'); plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
        if save_path: plt.savefig(save_path); print(f"Confusion matrix plot saved to: {save_path}")
        else: plt.show()
        plt.close()
    except Exception as e: print(f"Error plotting confusion matrix: {e}"); plt.close()

def plot_tsne(id_features: np.ndarray, ood_features: np.ndarray, title: str, save_path: Optional[str] = None, seed: int = 42, perplexity: int = 30, n_iter: int = 1000):
    # ... (코드 동일) ...
    if len(id_features) == 0 and len(ood_features) == 0: print("No features for t-SNE."); return
    if len(id_features) > 0 and len(ood_features) > 0:
        features = np.vstack((id_features, ood_features)); labels = np.concatenate([np.ones(len(id_features)), np.zeros(len(ood_features))])
        legend_labels = {1: 'In-Distribution', 0: 'Out-of-Distribution'}; colors = {1: 'blue', 0: 'red'}
    elif len(id_features) > 0: features = id_features; labels = np.ones(len(id_features)); legend_labels = {1: 'In-Distribution'}; colors = {1: 'blue'}
    else: features = ood_features; labels = np.zeros(len(ood_features)); legend_labels = {0: 'Out-of-Distribution'}; colors = {0: 'red'}
    print(f"Running t-SNE on {features.shape[0]} samples (perplexity={perplexity})...")
    try:
        effective_perplexity = min(perplexity, features.shape[0] - 1)
        if effective_perplexity < 5:
            print(f"Warning: Perplexity ({perplexity}) too high for number of samples ({features.shape[0]}). Using {effective_perplexity}.")
        if effective_perplexity <= 0:
             print("Error: Not enough samples for t-SNE. Skipping plot."); return
        tsne = TSNE(n_components=2, random_state=seed, perplexity=effective_perplexity, n_iter=n_iter, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(features)
    except Exception as e: print(f"Error running t-SNE: {e}. Skipping plot."); return
    plt.figure(figsize=(10, 8))
    for label_val, label_name in legend_labels.items():
        indices = labels == label_val; plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], c=colors[label_val], label=label_name, alpha=0.6, s=10)
    plt.title(title); plt.xlabel("t-SNE Dimension 1"); plt.ylabel("t-SNE Dimension 2"); plt.legend(); plt.grid(alpha=0.3)
    if save_path: plt.savefig(save_path); print(f"t-SNE plot saved to: {save_path}")
    else: plt.show()
    plt.close()

# --- run_experiment 함수 (내부 변경 없음, 전달받은 경로와 OE 소스 사용) ---
def run_experiment(args: Dict[str, Any], dataset_name: str):
    """주어진 ID 데이터셋에 대해 전체 실험 실행"""
    print(f"\n\n===== Starting Experiment for Dataset: {dataset_name} =====")
    current_result_dir = args['result_dir']
    current_model_dir = args['model_dir']
    print(f"Using Results Directory: {current_result_dir}")
    if args['save_model']: print(f"Using Models Directory: {current_model_dir}")

    print(f"Loading RoBERTa tokenizer: {args['model_type']}...")
    tokenizer = RobertaTokenizer.from_pretrained(args['model_type'])

    print("\n--- Preparing Datasets ---")
    ood_test_dataset = None
    known_class_names = []
    num_classes = 0
    final_label2id = {}
    final_id2label = {}
    ood_dataset_name = "ood"

    if dataset_name == 'syslog':
        # ... (데이터 로드 로직 동일) ...
        train_dataset, id_test_dataset, num_classes, _, final_label2id, final_id2label = prepare_syslog_data(
            tokenizer, args['max_length'], args['id_data_path'], args['text_col'], args['class_col'], args['id_exclude_class'], args['seed']
        )
        ood_test_dataset = prepare_syslog_ood_data(
            tokenizer, args['max_length'], args['ood_data_path'], args['text_col'], args['class_col'], args['ood_target_class']
        )
        known_class_names = list(final_id2label.values())
        ood_dataset_name = args['ood_target_class']
    elif dataset_name == '20newsgroups':
        # ... (데이터 로드 로직 동일) ...
        train_dataset, id_test_dataset, num_classes, _, final_label2id, final_id2label, ood_categories = prepare_20newsgroups_data(
            tokenizer, args['max_length'], args['num_20ng_id_classes'], args['seed']
        )
        ood_test_dataset = prepare_20newsgroups_ood_data(
            tokenizer, args['max_length'], ood_categories, args['seed']
        )
        known_class_names = list(final_id2label.values())
        ood_dataset_name = "20ng_ood"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    print("\n--- Creating Base DataLoaders ---")
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=num_workers, persistent_workers=num_workers>0, pin_memory=True)
    id_test_loader = DataLoader(id_test_dataset, batch_size=args['batch_size'], num_workers=num_workers, pin_memory=True)
    ood_test_loader = DataLoader(ood_test_dataset, batch_size=args['batch_size'], num_workers=num_workers, pin_memory=True) if ood_test_dataset else None

    performance_metrics = {}
    all_data = {}

    # --- 1. 표준 모델 (OE 없음) ---
    model_std = None
    std_model_filename = f'roberta_standard_{dataset_name}_{num_classes}cls_seed{args["seed"]}.pt'
    std_model_path = os.path.join(current_model_dir, std_model_filename)

    if not args['skip_standard']:
        print(f"\n--- Standard Model Training/Evaluation ({dataset_name}) ---")
        model_std = RoBERTaOOD(num_classes, args['model_type']).to(device)
        model_std.roberta.config.label2id = final_label2id; model_std.roberta.config.id2label = final_id2label
        if args['eval_only']:
            load_path = std_model_path
            if load_path and os.path.exists(load_path): print(f"Loading standard model from {load_path}..."); model_std.load_state_dict(torch.load(load_path, map_location=device))
            else: print(f"Error: Standard model path '{load_path}' not found for eval_only. Skipping standard evaluation."); model_std = None
        else:
            optimizer_std = AdamW(model_std.parameters(), lr=args['learning_rate'])
            total_steps = len(train_loader) * args['num_epochs']; scheduler_std = get_linear_schedule_with_warmup(optimizer_std, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
            train_standard(model_std, train_loader, optimizer_std, scheduler_std, device, args['num_epochs'])
            if args['save_model']: torch.save(model_std.state_dict(), std_model_path); print(f"Standard model saved to {std_model_path}")

        if model_std and ood_test_loader:
            print(f"\nEvaluating Standard Model ({dataset_name}) with OSR metrics...")
            results_std_osr, data_std_osr = evaluate_osr(model_std, id_test_loader, ood_test_loader, device, args['temperature'], return_data=True)
            print(f"  OSR Results (Standard vs {ood_dataset_name}): {results_std_osr}")
            performance_metrics[f'Standard+{ood_dataset_name}'] = results_std_osr # 키 형식 복원됨
            all_data[f'Standard+{ood_dataset_name}'] = data_std_osr
            if not args['no_plot']:
                # 플롯 저장 경로는 current_result_dir 사용 (정상)
                plot_confidence_histograms(data_std_osr['id_scores'], data_std_osr['ood_scores'], f'OSR Confidence - Standard ({dataset_name}) vs {ood_dataset_name}', os.path.join(current_result_dir, f'standard_osr_{ood_dataset_name}_hist.png'))
                plot_roc_curve(data_std_osr['id_scores'], data_std_osr['ood_scores'], f'OSR ROC - Standard ({dataset_name}) vs {ood_dataset_name}', os.path.join(current_result_dir, f'standard_osr_{ood_dataset_name}_roc.png'))
                threshold_analysis(data_std_osr['id_scores'], data_std_osr['ood_scores'], f'OSR Threshold Analysis - Standard ({dataset_name}) vs {ood_dataset_name}', os.path.join(current_result_dir, f'standard_osr_{ood_dataset_name}_threshold.png'))
                if len(data_std_osr['id_labels_true']) > 0:
                    cm_std = confusion_matrix(data_std_osr['id_labels_true'], data_std_osr['id_labels_pred'], labels=np.arange(num_classes))
                    plot_confusion_matrix(cm_std, known_class_names, f'Confusion Matrix - Standard ({dataset_name} ID Test)', os.path.join(current_result_dir, f'standard_confusion_matrix.png'))
                plot_tsne(data_std_osr['id_features'], data_std_osr['ood_features'], f't-SNE - Standard ({dataset_name} ID vs {ood_dataset_name})', os.path.join(current_result_dir, f'standard_tsne.png'), seed=args['seed'])
        del model_std
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    # --- 2. OE 모델 (Uniform Loss) - args['oe_sources']에 있는 모든 소스에 대해 반복 ---
    if not args['skip_oe_all']:
        # <<<--- 이 루프는 args['oe_sources']에 있는 모든 것을 처리함 --->>>
        for oe_source in args['oe_sources']:
            if dataset_name != 'syslog' and oe_source == 'syslog_masked':
                print(f"\nSkipping OE source 'syslog_masked' for non-syslog dataset '{dataset_name}'.")
                continue

            print(f"\n===== Running OE Experiment ({dataset_name}) with Source: {oe_source} =====")
            oe_train_dataset_current = None
            if oe_source == 'syslog_masked':
                # syslog_masked 소스일 경우, args에서 경로를 가져옴
                if not args['oe_masked_syslog_path']: # 경로가 제공되지 않았으면 경고 후 건너뜀
                     print(f"Warning: Skipping OE source 'syslog_masked' because --oe_masked_syslog_path was not provided.")
                     continue
                oe_train_dataset_current = prepare_syslog_masked_oe_data(
                    tokenizer, args['max_length'], args['oe_masked_syslog_path'], args['oe_masked_text_col']
                )
            elif oe_source in ['snli', 'imdb', 'wikitext']:
                # 다른 외부 소스 처리
                oe_train_dataset_current = prepare_external_oe_data(
                    tokenizer, args['max_length'], oe_source, data_dir=DATA_DIR, cache_dir=CACHE_DIR
                )
            else: print(f"Warning: Skipping invalid OE source '{oe_source}'"); continue

            if oe_train_dataset_current is None:
                print(f"Skipping OE training for source '{oe_source}' as no data was loaded.")
                continue

            oe_train_loader_current = DataLoader(oe_train_dataset_current, batch_size=args['batch_size'], shuffle=True, num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True)

            model_oe = None
            oe_model_filename = f'roberta_oe_uniform_{oe_source}_{dataset_name}_{num_classes}cls_seed{args["seed"]}.pt'
            oe_model_path = os.path.join(current_model_dir, oe_model_filename) # 저장 경로는 current_model_dir 사용
            print(f"\n--- Outlier Exposure (Source: {oe_source}, Loss: Uniform CE) Model Training/Evaluation ---")
            model_oe = RoBERTaOOD(num_classes, args['model_type']).to(device)
            model_oe.roberta.config.label2id = final_label2id; model_oe.roberta.config.id2label = final_id2label

            if args['eval_only']:
                load_path = oe_model_path
                if load_path and os.path.exists(load_path): print(f"Loading OE model ({oe_source}) from {load_path}..."); model_oe.load_state_dict(torch.load(load_path, map_location=device))
                else: print(f"Error: OE model path '{load_path}' not found for eval_only. Skipping OE {oe_source} evaluation."); model_oe = None
            else:
                optimizer_oe = AdamW(model_oe.parameters(), lr=args['learning_rate'])
                total_steps = len(train_loader) * args['num_epochs']; scheduler_oe = get_linear_schedule_with_warmup(optimizer_oe, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
                train_with_oe_uniform_loss(model_oe, train_loader, oe_train_loader_current, optimizer_oe, scheduler_oe, device, args['num_epochs'], args['oe_lambda'])
                if args['save_model']: torch.save(model_oe.state_dict(), oe_model_path); print(f"OE Model (Source: {oe_source}) saved to {oe_model_path}")

            if model_oe and ood_test_loader:
                print(f"\nEvaluating OE Model (Source: {oe_source}, Loss: Uniform CE) with OSR metrics...")
                results_oe_osr, data_oe_osr = evaluate_osr(model_oe, id_test_loader, ood_test_loader, device, args['temperature'], return_data=True)
                metric_key = f'OE_Uniform_{oe_source}+{ood_dataset_name}' # 키 형식 복원됨
                print(f"  OSR Results ({metric_key}): {results_oe_osr}")
                performance_metrics[metric_key] = results_oe_osr
                all_data[metric_key] = data_oe_osr
                if not args['no_plot']:
                    # 플롯 저장 경로는 current_result_dir 사용 (정상)
                    plot_confidence_histograms(data_oe_osr['id_scores'], data_oe_osr['ood_scores'], f'OSR Confidence - OE {oe_source} Uniform ({dataset_name}) vs {ood_dataset_name}', os.path.join(current_result_dir, f'oe_uniform_{oe_source}_osr_{ood_dataset_name}_hist.png'))
                    plot_roc_curve(data_oe_osr['id_scores'], data_oe_osr['ood_scores'], f'OSR ROC - OE {oe_source} Uniform ({dataset_name}) vs {ood_dataset_name}', os.path.join(current_result_dir, f'oe_uniform_{oe_source}_osr_{ood_dataset_name}_roc.png'))
                    threshold_analysis(data_oe_osr['id_scores'], data_oe_osr['ood_scores'], f'OSR Threshold Analysis - OE {oe_source} Uniform ({dataset_name}) vs {ood_dataset_name}', os.path.join(current_result_dir, f'oe_uniform_{oe_source}_osr_{ood_dataset_name}_threshold.png'))
                    if len(data_oe_osr['id_labels_true']) > 0:
                        cm_oe = confusion_matrix(data_oe_osr['id_labels_true'], data_oe_osr['id_labels_pred'], labels=np.arange(num_classes))
                        plot_confusion_matrix(cm_oe, known_class_names, f'Confusion Matrix - OE {oe_source} Uniform ({dataset_name} ID Test)', os.path.join(current_result_dir, f'oe_uniform_{oe_source}_confusion_matrix.png'))
                    plot_tsne(data_oe_osr['id_features'], data_oe_osr['ood_features'], f't-SNE - OE {oe_source} Uniform ({dataset_name} ID vs {ood_dataset_name})', os.path.join(current_result_dir, f'oe_uniform_{oe_source}_tsne.png'), seed=args['seed'])

            del model_oe, oe_train_dataset_current, oe_train_loader_current
            if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    # --- 결과 반환 ---
    return performance_metrics, all_data

# --- 메인 실행 로직 (수정됨) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RoBERTa OSR Comparison with Multiple OE Sources on Syslog or 20Newsgroups Data')
    parser.add_argument('--oe_masked_syslog_path', type=str, default=None, help='Path for masked syslog OE data (REQUIRED for syslog_masked OE source)')
    parser.add_argument('--dataset', type=str, default='syslog', choices=['syslog', '20newsgroups', 'all'], help='Which dataset to run experiment on.')
    parser.add_argument('--id_data_path', type=str, default='log_all_critical.csv', help='Path for syslog ID data')
    parser.add_argument('--ood_data_path', type=str, default='log_unknown.csv', help='Path for syslog OOD data')
    parser.add_argument('--result_dir_base', type=str, default='results_osr_comparison_multi_dataset', help='Base directory to save results')
    parser.add_argument('--model_dir_base', type=str, default='models_osr_comparison_multi_dataset', help='Base directory to save models')
    parser.add_argument('--text_col', type=str, default='text')
    parser.add_argument('--class_col', type=str, default='class')
    parser.add_argument('--oe_masked_text_col', type=str, default='masked_text_attention')
    parser.add_argument('--id_exclude_class', type=str, default='unknown', help='Class to exclude for syslog ID')
    parser.add_argument('--ood_target_class', type=str, default='unknown', help='OOD class for syslog')
    parser.add_argument('--num_20ng_id_classes', type=int, default=10, help='Number of ID classes for 20 Newsgroups')
    parser.add_argument('--model_type', type=str, default='roberta-base')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--oe_lambda', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    # <<<--- OE 소스 기본값 복원 및 선택지 유지 --->>>
    parser.add_argument('--oe_sources', nargs='+', default=['syslog_masked', 'snli', 'imdb'], choices=['syslog_masked', 'snli', 'imdb', 'wikitext'], help='List of OE sources to use.')
    parser.add_argument('--save_model', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--eval_only', action='store_true')
    # parser.add_argument('--std_model_path', type=str, default=None, help='Path to pre-trained standard model (if eval_only)') # 사용되지 않음
    parser.add_argument('--skip_standard', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--skip_oe_all', default=False, action=argparse.BooleanOptionalAction, help='Skip all OE experiments')
    parser.add_argument('--no_plot', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    args_dict = vars(args)

    set_seed(args.seed)

    oe_file_basename = None
    oe_path = args_dict.get('oe_masked_syslog_path')
    # syslog_masked 소스가 *선택된 경우에만* 경로 확인 및 이름 추출
    if 'syslog_masked' in args_dict.get('oe_sources', []):
        if oe_path and os.path.exists(oe_path):
            oe_file_name = os.path.basename(oe_path)
            oe_file_basename, _ = os.path.splitext(oe_file_name)
            oe_file_basename = re.sub(r'[^\w\-]+', '_', oe_file_basename)
            print(f"Extracted and sanitized OE file base name: '{oe_file_basename}'")
        elif not oe_path:
             print("Warning: 'syslog_masked' is in --oe_sources, but --oe_masked_syslog_path was not provided. This OE source will be skipped.")
             # 실제 실행 시 run_experiment 내부에서 경로 부재로 건너뛰므로 여기서 리스트 수정 불필요
        elif oe_path and not os.path.exists(oe_path):
             print(f"Warning: Provided --oe_masked_syslog_path '{oe_path}' does not exist. 'syslog_masked' OE source will be skipped.")
             # 실제 실행 시 run_experiment 내부에서 데이터 로드 실패로 건너뜀

    if args_dict['eval_only'] and not args_dict['skip_oe_all']:
        print("Running in eval_only mode. Will look for pre-trained models based on naming convention in specified directories.")

    datasets_to_run = []
    if args.dataset == 'all': datasets_to_run = ['syslog', '20newsgroups']
    elif args.dataset in ['syslog', '20newsgroups']: datasets_to_run = [args.dataset]
    else: raise ValueError(f"Invalid dataset choice: {args.dataset}")

    all_dataset_results = {}; all_dataset_data = {}

    for current_dataset_name in datasets_to_run:
        current_args = args_dict.copy()

        dataset_result_dir = os.path.join(args.result_dir_base, current_dataset_name)
        dataset_model_dir = os.path.join(args.model_dir_base, current_dataset_name)

        final_result_dir = dataset_result_dir
        final_model_dir = dataset_model_dir

        # syslog 데이터셋이고, syslog_masked OE 소스가 *포함되어 있고*, 유효한 파일 이름이 추출된 경우 경로 수정
        if oe_file_basename and current_dataset_name == 'syslog' and 'syslog_masked' in current_args.get('oe_sources', []):
            final_result_dir = os.path.join(dataset_result_dir, oe_file_basename)
            final_model_dir = os.path.join(dataset_model_dir, oe_file_basename)
            print(f"Output directories for this run (using OE file '{oe_file_basename}'):")
            print(f"  Results: {final_result_dir}")
            print(f"  Models: {final_model_dir}")
        else:
             print(f"Using standard output directories for dataset '{current_dataset_name}':")
             print(f"  Results: {final_result_dir}")
             print(f"  Models: {final_model_dir}")

        current_args['result_dir'] = final_result_dir
        current_args['model_dir'] = final_model_dir

        os.makedirs(final_result_dir, exist_ok=True)
        if current_args['save_model']:
            os.makedirs(final_model_dir, exist_ok=True)

        current_results, current_data = run_experiment(current_args, current_dataset_name)

        # <<<--- 결과 집계 시 키 접두사에서 oe_file_basename 제거 --->>>
        result_prefix = current_dataset_name # 예: 'syslog' 또는 '20newsgroups'

        if current_results:
            for key, value in current_results.items():
                # key 예시: Standard+unknown -> syslog_Standard+unknown
                # key 예시: OE_Uniform_imdb+unknown -> syslog_OE_Uniform_imdb+unknown
                # key 예시: OE_Uniform_syslog_masked+unknown -> syslog_OE_Uniform_syslog_masked+unknown
                all_dataset_results[f"{result_prefix}_{key}"] = value
        if current_data:
             for key, value in current_data.items():
                 all_dataset_data[f"{result_prefix}_{key}"] = value

    # --- 최종 전체 결과 요약 및 저장 ---
    print("\n\n===== Final Overall Results Summary =====")
    final_results_df = None
    if all_dataset_results:
        final_results_df = pd.DataFrame(all_dataset_results).T.sort_index()
        print("Overall Performance Metrics DataFrame:"); print(final_results_df)

        # 전체 요약 파일 이름에는 oe_file_basename 포함 (구분용)
        summary_suffix = f"_{oe_file_basename}" if oe_file_basename else ""
        overall_base_filename = f'osr_overall_summary_seed{args.seed}{summary_suffix}'

        overall_csv_file = os.path.join(args.result_dir_base, f'{overall_base_filename}.csv')
        overall_txt_file = os.path.join(args.result_dir_base, f'{overall_base_filename}.txt')
        overall_json_file = os.path.join(args.result_dir_base, f'{overall_base_filename}.json')

        try: final_results_df.to_csv(overall_csv_file, index=True); print(f"\nOverall results saved to CSV: {overall_csv_file}")
        except Exception as e: print(f"\nError saving overall results to CSV: {e}")
        try:
            with open(overall_txt_file, 'w', encoding='utf-8') as f:
                 f.write("--- Args ---\n"); f.write(json.dumps(args_dict, indent=4, default=str));
                 f.write("\n\n--- Overall Metrics ---\n"); f.write(final_results_df.to_string())
            print(f"Overall results saved to TXT: {overall_txt_file}")
        except Exception as e: print(f"\nError saving overall results to TXT: {e}")
        overall_results_data = {'metrics': all_dataset_results, 'arguments': args_dict, 'timestamp': datetime.now().isoformat()}
        try:
            with open(overall_json_file, 'w', encoding='utf-8') as f: json.dump(overall_results_data, f, indent=4, default=str)
            print(f"Overall results saved to JSON: {overall_json_file}")
        except Exception as e: print(f"\nError saving overall results to JSON: {e}")

        # OSR 비교 플롯은 동일한 OOD 데이터셋을 사용하는 결과들끼리 비교하는 것이 유용
        # 예: syslog+unknown 결과만 모아서 비교, 20ng+20ng_ood 결과만 모아서 비교
        # 키 구조가 복원되었으므로, 데이터셋별로 필터링하여 플롯 생성 가능
        if not args.no_plot:
            print("\nGenerating OSR comparison plots per OOD dataset...")
            ood_targets = set()
            for key in all_dataset_results.keys():
                if '+' in key:
                    ood_targets.add(key.split('+')[1]) # 예: 'unknown', '20ng_ood'

            for ood_target in ood_targets:
                # 해당 OOD 타겟을 사용하는 결과만 필터링
                filtered_results = {k: v for k, v in all_dataset_results.items() if k.endswith(f'+{ood_target}')}
                if filtered_results:
                    plot_title_suffix = f"_{ood_target}"
                    plot_save_path = os.path.join(args.result_dir_base, f'osr_comparison_seed{args.seed}{summary_suffix}{plot_title_suffix}.png')
                    print(f"  - Plotting comparison for OOD target '{ood_target}'...")
                    plot_osr_comparison(filtered_results, plot_save_path)
                else:
                    print(f"  - No results found for OOD target '{ood_target}' to plot.")

    else: print("No performance metrics were generated across datasets.")
    print("\nAll experiments finished.")