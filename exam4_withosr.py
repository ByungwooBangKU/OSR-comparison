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
# from sklearn.utils.class_weight import compute_class_weight # 현재 미사용
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_20newsgroups # 20 Newsgroups 로딩용
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import random
import matplotlib
import gc
import re

matplotlib.use('Agg') # 비-GUI 백엔드 설정
import matplotlib.pyplot as plt
plt.ioff() # 대화형 모드 끄기
try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False
    print("Warning: Seaborn not installed. Some plots might not be generated.")
import pandas as pd
import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union
# Hugging Face datasets 라이브러리 임포트
from datasets import load_dataset, DatasetDict


# --- 기본 설정값 (명령줄 인자로 덮어쓰기 가능) ---
ID_SYSLOG_PATH = 'log_all_critical.csv'
OOD_SYSLOG_UNKNOWN_PATH = 'log_unknown.csv'
TEXT_COLUMN = 'text'
CLASS_COLUMN = 'class'
OE_MASKED_TEXT_COLUMN = 'masked_text_attention'
ID_SYSLOG_EXCLUDE_CLASS = "unknown"
OOD_SYSLOG_TARGET_CLASS = "unknown" # 외부 OOD 파일 내 타겟 클래스
NUM_20NG_ID_CLASSES = 10
RESULT_DIR_BASE = '03_results_osr_multi_oe_experiments'
MODEL_DIR_BASE = '03_models_osr_multi_oe_experiments'
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
SAVE_MODEL_PER_EXPERIMENT = True
EVAL_ONLY = False
NO_PLOT_PER_EXPERIMENT = False
SKIP_STANDARD_MODEL = False
# SKIP_ALL_OE_EXPERIMENTS = False # 현재 메인 로직에서 사용 안 함

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed set to {seed}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.labels = labels
        valid_texts = [str(t) if pd.notna(t) else "" for t in texts]
        self.encodings = tokenizer(valid_texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def prepare_syslog_data(tokenizer, max_length: int, id_data_path: str, text_col: str, class_col: str, exclude_class: str, seed: int = 42) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset], int, Optional[LabelEncoder], Dict, Dict]:
    print(f"\n--- Preparing Syslog ID data from: {id_data_path} ---")
    print(f"    (Known classes will exclude '{exclude_class}', which will be treated as an OOD source from ID data)")
    try:
        df = pd.read_csv(id_data_path)
        if not all(c in df.columns for c in [text_col, class_col]):
            raise ValueError(f"ID Data CSV '{id_data_path}' must contain '{text_col}' and '{class_col}' columns.")
        df = df.dropna(subset=[text_col, class_col])
        df[class_col] = df[class_col].astype(str).str.lower()

        df_known_initial = df[df[class_col] != exclude_class.lower()].copy()
        df_id_unknown = df[df[class_col] == exclude_class.lower()].copy()

        id_unknown_dataset = None
        if not df_id_unknown.empty:
            print(f"  - Extracted {len(df_id_unknown)} samples for class '{exclude_class}' (to be treated as OOD from ID set).")
            unknown_texts = df_id_unknown[text_col].tolist()
            unknown_labels = np.full(len(unknown_texts), -1, dtype=int).tolist() # OOD는 -1 레이블
            id_unknown_dataset = TextDataset(unknown_texts, unknown_labels, tokenizer, max_length)
        else:
            print(f"  - No samples found for class '{exclude_class}' in '{id_data_path}' to be used as ID-internal OOD.")

        train_dataset, id_test_dataset, num_classes_final, final_label_encoder, final_label2id, final_id2label = None, None, 0, None, {}, {}

        if not df_known_initial.empty:
            class_counts = df_known_initial[class_col].value_counts()
            classes_to_keep = class_counts[class_counts >= 2].index
            if len(classes_to_keep) < len(class_counts):
                print(f"  - Filtering known classes with < 2 samples. Kept: {len(classes_to_keep)} classes.")
            df_known_final = df_known_initial[df_known_initial[class_col].isin(classes_to_keep)].copy()

            if not df_known_final.empty:
                final_classes = sorted(df_known_final[class_col].unique())
                num_classes_final = len(final_classes)
                if num_classes_final > 0:
                    final_label_encoder = LabelEncoder()
                    final_label_encoder.fit(final_classes)
                    final_label2id = {label: i for i, label in enumerate(final_label_encoder.classes_)}
                    final_id2label = {i: label for label, i in final_label2id.items()}
                    print(f"  - Final number of known classes: {num_classes_final}")
                    print(f"  - Final Label to ID mapping: {final_label2id}")

                    df_known_final['label'] = df_known_final[class_col].map(final_label2id)

                    min_class_count_for_split = df_known_final['label'].value_counts().min()
                    stratify_labels = df_known_final['label'] if min_class_count_for_split > 1 else None
                    
                    train_df, test_df = train_test_split(df_known_final, test_size=0.2, random_state=seed, stratify=stratify_labels)
                    print(f"  - Split known data into Train: {len(train_df)}, Test: {len(test_df)}")

                    train_dataset = TextDataset(train_df[text_col].tolist(), train_df['label'].tolist(), tokenizer, max_length)
                    id_test_dataset = TextDataset(test_df[text_col].tolist(), test_df['label'].tolist(), tokenizer, max_length)
                else: 
                    print(f"Warning: No classes with >= 2 samples identified for training in '{id_data_path}'. num_classes_final is 0.")
            else: 
                print(f"Warning: No known class data left after filtering classes with < 2 samples from '{id_data_path}'. num_classes_final is 0.")
        else: 
             print(f"Warning: No data left for known classes after excluding class '{exclude_class}' from '{id_data_path}'. num_classes_final is 0.")
        
        return train_dataset, id_test_dataset, id_unknown_dataset, num_classes_final, final_label_encoder, final_label2id, final_id2label
    except Exception as e:
        print(f"Error preparing Syslog ID data from '{id_data_path}': {e}")
        return None, None, None, 0, None, {}, {}

def prepare_20newsgroups_data(tokenizer, max_length: int, num_id_classes: int, seed: int = 42) -> Tuple[Optional[Dataset], Optional[Dataset], int, Optional[LabelEncoder], Dict, Dict, List[str]]:
    print(f"\n--- Preparing 20 Newsgroups ID data ({num_id_classes} classes) ---")
    try:
        all_categories = list(fetch_20newsgroups(subset='train').target_names)
        if num_id_classes <= 0 or num_id_classes > len(all_categories):
            print(f"Warning: Invalid num_id_classes ({num_id_classes}). Using all {len(all_categories)} categories.")
            num_id_classes = len(all_categories)

        np.random.seed(seed)
        id_category_indices = np.random.choice(len(all_categories), num_id_classes, replace=False)
        in_dist_categories = [all_categories[i] for i in id_category_indices]
        out_dist_categories = [cat for i, cat in enumerate(all_categories) if i not in id_category_indices]
        print(f"  - Selected {num_id_classes} ID categories: {in_dist_categories}")

        id_train_sk = fetch_20newsgroups(subset='train', categories=in_dist_categories, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=seed)
        id_test_sk = fetch_20newsgroups(subset='test', categories=in_dist_categories, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=seed)

        if not id_train_sk.data or not id_test_sk.data:
            print("Warning: No data found for selected 20 Newsgroups ID categories.")
            return None, None, 0, None, {}, {}, []

        label_encoder = LabelEncoder()
        id_train_labels = label_encoder.fit_transform(id_train_sk.target)
        id_test_labels = label_encoder.transform(id_test_sk.target)

        final_label2id = {label_encoder.classes_[i]: i for i in range(len(label_encoder.classes_))}
        final_id2label = {i: label_encoder.classes_[i] for i in range(len(label_encoder.classes_))}
        num_classes_final = len(label_encoder.classes_)
        print(f"  - Final number of known classes: {num_classes_final}")
        print(f"  - Final Label to ID mapping (showing target index to name): {final_id2label}")


        train_dataset = TextDataset(id_train_sk.data, id_train_labels.tolist(), tokenizer, max_length)
        id_test_dataset = TextDataset(id_test_sk.data, id_test_labels.tolist(), tokenizer, max_length)
        print(f"  - Split into Train: {len(train_dataset)}, Test: {len(id_test_dataset)}")
        return train_dataset, id_test_dataset, num_classes_final, label_encoder, final_label2id, final_id2label, out_dist_categories
    except Exception as e:
        print(f"Error preparing 20 Newsgroups ID data: {e}")
        return None, None, 0, None, {}, {}, []

def prepare_20newsgroups_ood_data(tokenizer, max_length: int, ood_categories: List[str], seed: int = 42) -> Optional[Dataset]:
    if not ood_categories:
        print("No OOD categories provided for 20 Newsgroups.")
        return None
    print(f"\n--- Preparing 20 Newsgroups OOD data (Categories: {ood_categories}) ---")
    try:
        ood_test_sk = fetch_20newsgroups(subset='test', categories=ood_categories, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=seed)
        if not ood_test_sk.data:
            print(f"Warning: No data found for OOD categories: {ood_categories}")
            return None
        texts = ood_test_sk.data
        ood_labels = np.full(len(texts), -1, dtype=int).tolist()
        ood_dataset = TextDataset(texts, ood_labels, tokenizer, max_length)
        print(f"  - Loaded {len(ood_dataset)} samples for OOD testing from 20 Newsgroups.")
        return ood_dataset
    except Exception as e:
        print(f"Error preparing 20 Newsgroups OOD data: {e}")
        return None

def prepare_syslog_masked_oe_data(tokenizer, max_length: int, oe_data_path: str, oe_text_col: str) -> Optional[Dataset]:
    print(f"\n--- Preparing Syslog Masked OE data from: {oe_data_path} ---")
    if not os.path.exists(oe_data_path):
        print(f"Error: OE data path not found: {oe_data_path}")
        return None
    try:
        df = pd.read_csv(oe_data_path)
        if oe_text_col not in df.columns:
            raise ValueError(f"OE Data CSV '{oe_data_path}' must contain '{oe_text_col}' column.")
        df = df.dropna(subset=[oe_text_col])
        texts = df[oe_text_col].astype(str).tolist()
        if not texts:
            print(f"Warning: No valid OE texts found in '{oe_text_col}' from '{oe_data_path}'.")
            return None
        oe_labels = np.full(len(texts), -1, dtype=int).tolist()
        oe_dataset = TextDataset(texts, oe_labels, tokenizer, max_length)
        print(f"  - Loaded {len(oe_dataset)} samples for OE training from Syslog Masked: {oe_data_path}")
        return oe_dataset
    except Exception as e:
        print(f"Error preparing Syslog Masked OE data from '{oe_data_path}': {e}")
        return None

def prepare_external_oe_data(tokenizer, max_length: int, oe_source_name: str, data_dir: str = 'data', cache_dir_hf: Optional[str] = None) -> Optional[Dataset]:
    print(f"\n--- Preparing External OE data source: {oe_source_name} ---")
    if cache_dir_hf is None:
        cache_dir_hf = os.path.join(data_dir, "hf_cache")
    os.makedirs(cache_dir_hf, exist_ok=True)

    dataset_config = None
    if oe_source_name == "snli":
        dataset_config = {"name": "snli", "split": "train", "text_col": "hypothesis"}
    elif oe_source_name == "imdb":
        dataset_config = {"name": "imdb", "split": "train", "text_col": "text"}
    elif oe_source_name == "wikitext":
        dataset_config = {"name": "wikitext", "config_name": "wikitext-103-raw-v1", "split": "train", "text_col": "text"}
    else:
        print(f"Error: Unknown external OE source name '{oe_source_name}'")
        return None

    try:
        print(f"  Loading {oe_source_name} (config: {dataset_config.get('config_name', 'default')}, split: {dataset_config['split']})...")
        ds = load_dataset(dataset_config["name"], name=dataset_config.get("config_name"), split=dataset_config["split"], cache_dir=cache_dir_hf)

        if isinstance(ds, DatasetDict):
            if dataset_config['split'] in ds:
                ds = ds[dataset_config['split']]
            else:
                raise ValueError(f"Split '{dataset_config['split']}' not found in DatasetDict for {oe_source_name}")

        texts = [item for item in ds[dataset_config['text_col']] if isinstance(item, str) and item.strip()]
        if oe_source_name == "wikitext":
            texts = [text for text in texts if not text.strip().startswith("=") and len(text.strip().split()) > 3]

        if not texts:
            print(f"Warning: No valid texts found for OE source {oe_source_name}.")
            return None

        oe_labels = np.full(len(texts), -1, dtype=int).tolist()
        oe_dataset = TextDataset(texts, oe_labels, tokenizer, max_length)
        print(f"  - Loaded {len(oe_dataset)} samples for OE training from {oe_source_name}.")
        return oe_dataset
    except Exception as e:
        print(f"Error loading external OE dataset {oe_source_name}: {e}")
        return None

def prepare_syslog_ood_data(tokenizer, max_length: int, ood_data_path: str, text_col: str, class_col: str, ood_target_class: str) -> Optional[Dataset]:
    print(f"\n--- Preparing Syslog OOD data (class: '{ood_target_class}') from: {ood_data_path} ---")
    if not os.path.exists(ood_data_path):
        print(f"Warning: OOD data path not found: {ood_data_path}. Skipping this OOD source.")
        return None
    try:
        df = pd.read_csv(ood_data_path)
        if not all(c in df.columns for c in [text_col, class_col]):
            raise ValueError(f"OOD Data CSV '{ood_data_path}' must contain '{text_col}' and '{class_col}' columns.")
        df = df.dropna(subset=[text_col, class_col])
        df[class_col] = df[class_col].astype(str).str.lower()
        df_ood = df[df[class_col] == ood_target_class.lower()].copy()

        if df_ood.empty:
            print(f"Warning: No data found for OOD class '{ood_target_class}' in '{ood_data_path}'.")
            return None

        texts = df_ood[text_col].tolist()
        ood_labels = np.full(len(texts), -1, dtype=int).tolist()
        ood_dataset = TextDataset(texts, ood_labels, tokenizer, max_length)
        print(f"  - Loaded {len(ood_dataset)} samples for OOD testing (class: '{ood_target_class}').")
        return ood_dataset
    except Exception as e:
        print(f"Error preparing Syslog OOD data from '{ood_data_path}': {e}")
        return None

class RoBERTaOOD(nn.Module):
    def __init__(self, num_classes: int, model_name: str = 'roberta-base'):
        super(RoBERTaOOD, self).__init__()
        self.actual_num_classes = num_classes 
        effective_num_labels = max(1, num_classes) 
        
        config = RobertaConfig.from_pretrained(model_name, num_labels=effective_num_labels)
        config.output_hidden_states = True
        self.roberta = RobertaForSequenceClassification.from_pretrained(model_name, config=config)

    def forward(self, input_ids, attention_mask, output_features=False):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if output_features:
            features = outputs.hidden_states[-1][:, 0, :]
            return logits, features
        else:
            return logits

def train_standard(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, scheduler, device: torch.device, num_epochs: int, current_experiment_name: str):
    model.train()
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Starting standard training for '{current_experiment_name}'... AMP enabled: {use_amp}")

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Std Epoch {epoch+1}/{num_epochs} ({current_experiment_name})")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=str(device).split(':')[0], enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.3f}"})
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Std Epoch {epoch+1}/{num_epochs} ({current_experiment_name}), Avg Loss: {avg_loss:.4f}")

def train_with_oe_uniform_loss(model: nn.Module, train_loader: DataLoader, oe_loader: DataLoader, optimizer: optim.Optimizer, scheduler, device: torch.device, num_epochs: int, oe_lambda: float, current_experiment_name: str):
    model.train()
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Starting OE (Uniform CE Loss) training for '{current_experiment_name}'... AMP enabled: {use_amp}")

    num_id_classes_for_oe = model.module.actual_num_classes if isinstance(model, nn.DataParallel) else model.actual_num_classes
    if num_id_classes_for_oe == 0:
        print("Warning: Training with OE but model's actual_num_classes is 0. OE loss might not be effective or meaningful.")

    for epoch in range(num_epochs):
        oe_iter = iter(oe_loader)
        total_loss = 0
        total_id_loss = 0
        total_oe_loss = 0
        progress_bar = tqdm(train_loader, desc=f"OE Epoch {epoch+1}/{num_epochs} ({current_experiment_name})")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            try:
                oe_batch = next(oe_iter)
            except StopIteration:
                oe_iter = iter(oe_loader)
                oe_batch = next(oe_iter)

            oe_input_ids = oe_batch['input_ids'].to(device)
            oe_attention_mask = oe_batch['attention_mask'].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=str(device).split(':')[0], enabled=use_amp):
                id_logits = model(input_ids, attention_mask)
                id_loss = F.cross_entropy(id_logits, labels)

                oe_logits = model(oe_input_ids, oe_attention_mask)
                log_softmax_oe = F.log_softmax(oe_logits, dim=1)
                
                effective_oe_num_classes = max(1, num_id_classes_for_oe)
                uniform_target_probs = torch.full_like(oe_logits, 1.0 / effective_oe_num_classes)
                
                if num_id_classes_for_oe > 0 : 
                    oe_loss = F.kl_div(log_softmax_oe, uniform_target_probs, reduction='batchmean', log_target=False)
                else: 
                    oe_loss = torch.tensor(0.0, device=device)

                total_batch_loss = id_loss + oe_lambda * oe_loss

            scaler.scale(total_batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += total_batch_loss.item()
            total_id_loss += id_loss.item()
            total_oe_loss += oe_loss.item()
            progress_bar.set_postfix({'Total Loss': f"{total_batch_loss.item():.3f}", 'ID Loss': f"{id_loss.item():.3f}", 'OE Loss': f"{oe_loss.item():.3f}"})
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        avg_id_loss = total_id_loss / len(train_loader)
        avg_oe_loss = total_oe_loss / len(train_loader)
        print(f"OE Epoch {epoch+1}/{num_epochs} ({current_experiment_name}), Avg Loss: {avg_loss:.4f} (ID: {avg_id_loss:.4f}, OE: {avg_oe_loss:.4f})")

def plot_enhanced_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, 
                                  save_path: Optional[str] = None, unknown_idx: int = -1,
                                  normalize: bool = True):
    """
    향상된 confusion matrix 플롯 함수 - OSR 성능을 보다 명확하게 시각화
    
    Args:
        cm: confusion matrix
        class_names: 클래스 이름 목록
        title: 그래프 제목
        save_path: 저장 경로
        unknown_idx: unknown 클래스의 인덱스 (default: 마지막 클래스)
        normalize: 행 기준 정규화 여부
    """
    if not SNS_AVAILABLE:
        print("Seaborn not available, skipping confusion matrix plot.")
        return
    if cm is None or len(class_names) == 0:
        print(f"Skipping plot for '{title}' due to missing data.")
        return
    
    plt.figure(figsize=(max(10, int(len(class_names)*0.9)), max(8, int(len(class_names)*0.8))))
    
    # Unknown 클래스 강조를 위한 색상맵 설정
    unknown_cmap = plt.cm.Blues
    class_colors = ['#e6f3ff' if i != unknown_idx else '#ffebeb' for i in range(len(class_names))]
    
    # Confusion matrix 정규화 (행 기준)
    if normalize:
        row_sums = cm.sum(axis=1)
        cm_normalized = np.zeros_like(cm, dtype=float)
        for i in range(cm.shape[0]):
            if row_sums[i] > 0:
                cm_normalized[i, :] = cm[i, :] / row_sums[i]
        # 두 버전 모두 저장
        cm_display = cm_normalized
        cm_text = cm.astype(int)  # 원본 값은 텍스트로 표시
    else:
        cm_display = cm
        cm_text = cm
    
    # 클래스별 정확도 계산
    class_accuracies = np.zeros(len(class_names))
    for i in range(len(class_names)):
        if cm[i].sum() > 0:
            class_accuracies[i] = cm[i, i] / cm[i].sum()
    
    # OSR 관련 지표 계산 (Unknown vs Known)
    if unknown_idx >= 0 and unknown_idx < len(class_names):
        # Unknown 클래스의 성능 지표
        tp_unknown = cm[unknown_idx, unknown_idx]
        fp_unknown = cm[:, unknown_idx].sum() - tp_unknown
        fn_unknown = cm[unknown_idx, :].sum() - tp_unknown
        
        precision = tp_unknown / (tp_unknown + fp_unknown) if (tp_unknown + fp_unknown) > 0 else 0
        recall = tp_unknown / (tp_unknown + fn_unknown) if (tp_unknown + fn_unknown) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Known 클래스의 정확도 (미스클래시피케이션 + Unknown 으로 분류된 것 제외)
        known_indices = [i for i in range(len(class_names)) if i != unknown_idx]
        known_cm = cm[np.ix_(known_indices, known_indices)]
        known_acc = np.trace(known_cm) / known_cm.sum() if known_cm.sum() > 0 else 0
        
        osr_metrics = {
            'Unknown Precision': precision,
            'Unknown Recall': recall,
            'Unknown F1': f1,
            'Known Accuracy': known_acc
        }
    else:
        osr_metrics = {}
    
    # 향상된 히트맵 그리기
    df_cm = pd.DataFrame(cm_display, index=class_names, columns=class_names)
    
    # 색상맵 설정 - Unknown 클래스 강조
    sns.heatmap(df_cm, annot=cm_text, fmt="d" if not normalize else ".2f", 
                cmap="Blues", cbar=True, 
                annot_kws={"size": 8 if len(class_names) > 10 else 10})
    
    plt.title(f"{title}\n" + "\n".join([f"{k}: {v:.4f}" for k, v in osr_metrics.items()]), 
              fontsize=14)
    
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Unknown 클래스 테두리 강조
    if unknown_idx >= 0:
        plt.axhline(y=unknown_idx, color='r', linewidth=2)
        plt.axhline(y=unknown_idx+1, color='r', linewidth=2)
        plt.axvline(x=unknown_idx, color='r', linewidth=2)
        plt.axvline(x=unknown_idx+1, color='r', linewidth=2)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # 클래스별 정확도 표시
    for i in range(len(class_names)):
        acc_text = f"{class_accuracies[i]:.2f}"
        plt.text(-0.5, i + 0.5, acc_text, 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=8, color='black', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced confusion matrix saved to: {save_path}")
    
    plt.close()

def plot_unknown_confusion_matrix(cm: np.ndarray, class_names: List[str], unknown_idx: int, 
                                 title: str, save_path: Optional[str] = None):
    """
    'unknown' 클래스에 초점을 맞춘 confusion matrix를 생성합니다.
    
    Args:
        cm: 전체 confusion matrix
        class_names: 클래스 이름 목록
        unknown_idx: unknown 클래스의 인덱스
        title: 그래프 제목
        save_path: 저장 경로
    """
    if not SNS_AVAILABLE:
        print("Seaborn not available, skipping confusion matrix plot.")
        return
        
    if unknown_idx < 0 or unknown_idx >= len(class_names):
        print(f"Invalid unknown_idx {unknown_idx} for unknown-focused confusion matrix. Skipping.")
        return
    
    # 원본 confusion matrix를 사용
    plt.figure(figsize=(max(10, int(len(class_names)*0.9)), max(8, int(len(class_names)*0.8))))
    
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # 히트맵 플로팅
    ax = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                    annot_kws={"size": 8 if len(class_names) > 10 else 10})
    
    # Unknown 클래스 강조 표시 (행과 열에 색상 추가)
    # 먼저 행(실제 레이블)에 대한 색상 지정
    for i in range(len(class_names)):
        if i == unknown_idx:
            ax.add_patch(plt.Rectangle((0, i), len(class_names), 1, fill=False, edgecolor='red', lw=2))
    
    # 열(예측 레이블)에 대한 색상 지정
    for j in range(len(class_names)):
        if j == unknown_idx:
            ax.add_patch(plt.Rectangle((j, 0), 1, len(class_names), fill=False, edgecolor='blue', lw=2))
    
    # Unknown 클래스 성능 지표 계산
    if unknown_idx >= 0 and unknown_idx < len(class_names):
        tp = cm[unknown_idx, unknown_idx]  # True Positive (실제 unknown, 예측 unknown)
        fp = cm[:, unknown_idx].sum() - tp  # False Positive (실제 known, 예측 unknown)
        fn = cm[unknown_idx, :].sum() - tp  # False Negative (실제 unknown, 예측 known)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 결과에 지표 추가
        metrics_text = f"Unknown Precision: {precision:.4f}\nUnknown Recall: {recall:.4f}\nUnknown F1: {f1:.4f}"
        plt.title(f"{title}\n{metrics_text}", fontsize=14)
    else:
        plt.title(title, fontsize=14)
    
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Unknown-focused confusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def create_binary_osr_confusion_matrix(cm: np.ndarray, class_names: List[str], unknown_idx: int):
    """
    OSR 이진 confusion matrix (Known vs Unknown)를 생성합니다.
    """
    if unknown_idx < 0 or unknown_idx >= len(class_names):
        return None, None
    
    # 이진 confusion matrix 생성 (Known vs Unknown)
    binary_cm = np.zeros((2, 2), dtype=int)
    
    # True Unknown, Predicted Unknown (TP for Unknown detection)
    binary_cm[1, 1] = cm[unknown_idx, unknown_idx]
    
    # True Known, Predicted Unknown (FP for Unknown detection)
    known_indices = [i for i in range(len(class_names)) if i != unknown_idx]
    binary_cm[0, 1] = np.sum(cm[known_indices, unknown_idx])
    
    # True Unknown, Predicted Known (FN for Unknown detection)
    binary_cm[1, 0] = np.sum(cm[unknown_idx, known_indices])
    
    # True Known, Predicted Known (TN for Unknown detection)
    for i in known_indices:
        for j in known_indices:
            binary_cm[0, 0] += cm[i, j]
    
    # 성능 지표 계산
    precision = binary_cm[1, 1] / (binary_cm[1, 1] + binary_cm[0, 1]) if (binary_cm[1, 1] + binary_cm[0, 1]) > 0 else 0
    recall = binary_cm[1, 1] / (binary_cm[1, 1] + binary_cm[1, 0]) if (binary_cm[1, 1] + binary_cm[1, 0]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (binary_cm[0, 0] + binary_cm[1, 1]) / binary_cm.sum() if binary_cm.sum() > 0 else 0
    
    metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Accuracy': accuracy
    }
    
    return binary_cm, metrics

def plot_binary_osr_confusion_matrix(binary_cm, metrics, title, save_path=None):
    """
    OSR 이진 confusion matrix를 시각화합니다.
    """
    if binary_cm is None:
        return
    
    binary_class_names = ['Known', 'Unknown']
    
    plt.figure(figsize=(8, 7))
    df_cm = pd.DataFrame(binary_cm, index=binary_class_names, columns=binary_class_names)
    
    # 히트맵 플로팅 (퍼센트와 실제 숫자 모두 표시)
    ax = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True, annot_kws={"size": 14})
    
    # 총 샘플 수 계산
    total_samples = np.sum(binary_cm)
    
    # 각 셀에 퍼센트 추가
    for i in range(2):
        for j in range(2):
            percentage = 100 * binary_cm[i, j] / total_samples if total_samples > 0 else 0
            ax.text(j + 0.5, i + 0.7, f"({percentage:.1f}%)", 
                    ha="center", va="center", fontsize=11)
    
    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    plt.title(f"{title}\n\n{metrics_text}", fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Binary OSR confusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
def plot_osr_binary_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                                     unknown_idx: int, title: str, save_path: Optional[str] = None):
    """
    OSR 이진 confusion matrix (Known vs Unknown)를 생성합니다.
    """
    if unknown_idx < 0 or unknown_idx >= len(class_names):
        print(f"Invalid unknown_idx {unknown_idx} for binary confusion matrix. Skipping.")
        return
    
    # 이진 confusion matrix 생성
    binary_cm = np.zeros((2, 2), dtype=int)
    
    # True Unknown, Predicted Unknown (TP for Unknown)
    binary_cm[1, 1] = cm[unknown_idx, unknown_idx]
    
    # True Known, Predicted Unknown (FP for Unknown)
    binary_cm[0, 1] = cm[:unknown_idx, unknown_idx].sum() + (cm[unknown_idx+1:, unknown_idx].sum() if unknown_idx < len(class_names)-1 else 0)
    
    # True Unknown, Predicted Known (FN for Unknown)
    binary_cm[1, 0] = cm[unknown_idx, :unknown_idx].sum() + (cm[unknown_idx, unknown_idx+1:].sum() if unknown_idx < len(class_names)-1 else 0)
    
    # True Known, Predicted Known (TN for Unknown)
    known_indices = list(range(unknown_idx)) + list(range(unknown_idx+1, len(class_names)))
    if known_indices:
        known_cm = cm[np.ix_(known_indices, known_indices)]
        binary_cm[0, 0] = known_cm.sum()
    
    # 성능 지표 계산
    precision = binary_cm[1, 1] / (binary_cm[1, 1] + binary_cm[0, 1]) if (binary_cm[1, 1] + binary_cm[0, 1]) > 0 else 0
    recall = binary_cm[1, 1] / (binary_cm[1, 1] + binary_cm[1, 0]) if (binary_cm[1, 1] + binary_cm[1, 0]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (binary_cm[0, 0] + binary_cm[1, 1]) / binary_cm.sum() if binary_cm.sum() > 0 else 0
    
    binary_class_names = ['Known', 'Unknown']
    
    plt.figure(figsize=(8, 6))
    df_cm = pd.DataFrame(binary_cm, index=binary_class_names, columns=binary_class_names)
    
    # 히트맵 플로팅
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    
    metrics_text = f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\nAccuracy: {accuracy:.4f}"
    
    plt.title(f"{title}\n{metrics_text}", fontsize=14)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Binary OSR confusion matrix saved to: {save_path}")
    
    plt.close()
    
def evaluate_osr(model: nn.Module, id_loader: Optional[DataLoader],
                 ood_loaders: Optional[Dict[str, DataLoader]],
                 device: torch.device, temperature: float = 1.0,
                 threshold_percentile: float = 5.0, return_data: bool = False,
                 num_known_classes: int = 0,
                 id_file_unknown_ood_name: Optional[str] = None,
                 known_class_names: Optional[List[str]] = None,
                 id_exclude_class_name: Optional[str] = "unknown",
                 analyze_thresholds: bool = True
                 ) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, Any]]]:
    model.eval()
    
    id_logits_all, id_scores_all, id_labels_true_list, id_preds_closed_set_list, id_features_all = [], [], [], [], []
    if id_loader and num_known_classes > 0:
        with torch.no_grad():
            for batch in tqdm(id_loader, desc="Evaluating ID (Knowns) for OSR", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label']
                logits, features = model(input_ids, attention_mask, output_features=True)
                
                if logits.size(1) != num_known_classes:
                     pass # 경고는 위에서 출력됨
                
                softmax_probs = F.softmax(logits / temperature, dim=1)
                max_probs, preds = softmax_probs.max(dim=1)
                id_logits_all.append(logits.cpu())
                id_scores_all.append(max_probs.cpu())
                id_labels_true_list.extend(labels.numpy())
                id_preds_closed_set_list.extend(preds.cpu().numpy())
                id_features_all.append(features.cpu())
    
    id_scores = torch.cat(id_scores_all).numpy() if id_scores_all else np.array([])
    id_features = torch.cat(id_features_all).numpy() if id_features_all and len(id_features_all[0]) > 0 else np.array([])
    id_labels_true_np = np.array(id_labels_true_list)
    id_preds_closed_set_np = np.array(id_preds_closed_set_list)

    all_results_metrics = {}
    all_data_for_plots = {
        "id_scores": id_scores, "id_labels_true": id_labels_true_np,
        "id_preds_closed_set": id_preds_closed_set_np, "id_features": id_features,
        "cm_kplus1": None, "kplus1_class_names": None
    }

    if len(id_labels_true_np) > 0 and num_known_classes > 0:
        all_results_metrics["Closed_Set_Accuracy"] = accuracy_score(id_labels_true_np, id_preds_closed_set_np)
        all_results_metrics["F1_Macro_Closed_Set"] = f1_score(id_labels_true_np, id_preds_closed_set_np, labels=np.arange(num_known_classes), average='macro', zero_division=0)
    else:
        all_results_metrics["Closed_Set_Accuracy"] = 0.0
        all_results_metrics["F1_Macro_Closed_Set"] = 0.0

    if len(id_scores) > 0:
        chosen_threshold = np.percentile(id_scores, threshold_percentile)
    else:
        chosen_threshold = 0.5 
        if num_known_classes == 0 and id_file_unknown_ood_name:
             print(f"Warning: No ID (Known) scores. Using default threshold: {chosen_threshold} for K+1 metrics involving '{id_file_unknown_ood_name}'.")
    all_results_metrics["Threshold_Used"] = chosen_threshold
    print(f"  Chosen MSP Threshold for OSR: {chosen_threshold:.4f} (based on {threshold_percentile}% of ID scores if available)")

    # 분석을 위한 다양한 임계값 생성
    threshold_range = None
    if analyze_thresholds and len(id_scores) > 0:
        # 다양한 MSP 임계값 생성 (1%부터 99%까지)
        threshold_range = [np.percentile(id_scores, p) for p in range(1, 100, 5)]
        all_data_for_plots["threshold_range"] = threshold_range
    
    unknown_data_collected = False
    unknown_scores = np.array([])
    
    if ood_loaders:
        for ood_name, ood_loader_single in ood_loaders.items():
            if ood_name == id_file_unknown_ood_name: continue 

            print(f"  Evaluating standard OSR metrics against OOD source: {ood_name}")
            current_ood_scores_all, current_ood_features_all = [], []
            if ood_loader_single is None: continue
            with torch.no_grad():
                for batch in tqdm(ood_loader_single, desc=f"Evaluating OOD ({ood_name})", leave=False):
                    input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                    logits, features = model(input_ids, attention_mask, output_features=True)
                    softmax_probs = F.softmax(logits / temperature, dim=1)
                    max_probs, _ = softmax_probs.max(dim=1)
                    current_ood_scores_all.append(max_probs.cpu())
                    current_ood_features_all.append(features.cpu())
            
            current_ood_scores = torch.cat(current_ood_scores_all).numpy() if current_ood_scores_all else np.array([])
            current_ood_features = torch.cat(current_ood_features_all).numpy() if current_ood_features_all and len(current_ood_features_all[0]) > 0 else np.array([])
            all_data_for_plots[f"ood_scores_{ood_name}"] = current_ood_scores
            all_data_for_plots[f"ood_features_{ood_name}"] = current_ood_features
            
            # "unknown" 클래스 스코어 저장 (threshold 분석용)
            if ood_name.lower() == "unknown" or id_exclude_class_name.lower() in ood_name.lower():
                unknown_scores = current_ood_scores
                unknown_data_collected = True

            if len(current_ood_scores) == 0 or len(id_scores) == 0:
                metrics_to_init = ["AUROC", "FPR@TPR90", "AUPR_In", "AUPR_Out", "DetectionAccuracy", "OSCR"]
                for m in metrics_to_init: all_results_metrics[f"{ood_name}_{m}"] = 0.0 if m != "FPR@TPR90" else 1.0
                continue
            
            y_true_osr = np.concatenate([np.ones_like(id_scores), np.zeros_like(current_ood_scores)])
            y_scores_osr = np.concatenate([id_scores, current_ood_scores])
            valid_indices = ~np.isnan(y_scores_osr)
            y_true_osr, y_scores_osr = y_true_osr[valid_indices], y_scores_osr[valid_indices]

            if len(np.unique(y_true_osr)) >= 2:
                all_results_metrics[f"{ood_name}_AUROC"] = roc_auc_score(y_true_osr, y_scores_osr)
                fpr, tpr, _ = roc_curve(y_true_osr, y_scores_osr)
                idx_tpr90 = np.where(tpr >= 0.90)[0]
                all_results_metrics[f"{ood_name}_FPR@TPR90"] = fpr[idx_tpr90[0]] if len(idx_tpr90) > 0 else 1.0
                precision_in, recall_in, _ = precision_recall_curve(y_true_osr, y_scores_osr, pos_label=1)
                all_results_metrics[f"{ood_name}_AUPR_In"] = auc(recall_in, precision_in)
                precision_out, recall_out, _ = precision_recall_curve(1 - y_true_osr, 1 - y_scores_osr, pos_label=1) 
                all_results_metrics[f"{ood_name}_AUPR_Out"] = auc(recall_out, precision_out)
                
                # 다양한 threshold에 대한 성능 지표 계산
                if analyze_thresholds and threshold_range and (ood_name.lower() == "unknown" or id_exclude_class_name.lower() in ood_name.lower()):
                    precision_at_threshold = []
                    recall_at_threshold = []
                    f1_at_threshold = []
                    detection_acc_at_threshold = []
                    
                    for thresh in threshold_range:
                        id_preds_binary = (id_scores >= thresh).astype(int)
                        ood_preds_binary = (current_ood_scores < thresh).astype(int)
                        
                        # True positives: 올바르게 unknown으로 식별된 OOD 샘플
                        tp = np.sum(ood_preds_binary)
                        # False positives: unknown으로 잘못 식별된 ID 샘플
                        fp = np.sum(1 - id_preds_binary)
                        # False negatives: ID로 잘못 식별된 OOD 샘플
                        fn = np.sum(1 - ood_preds_binary)
                        # True negatives: 올바르게 ID로 식별된 ID 샘플
                        tn = np.sum(id_preds_binary)
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        detection_acc = (tp + tn) / (tp + tn + fp + fn)
                        
                        precision_at_threshold.append(precision)
                        recall_at_threshold.append(recall)
                        f1_at_threshold.append(f1)
                        detection_acc_at_threshold.append(detection_acc)
                    
                    # 최적 F1 score 임계값 찾기
                    best_f1_idx = np.argmax(f1_at_threshold)
                    best_f1_threshold = threshold_range[best_f1_idx]
                    best_f1 = f1_at_threshold[best_f1_idx]
                    
                    # 결과 저장
                    all_data_for_plots[f"{ood_name}_precision_by_threshold"] = precision_at_threshold
                    all_data_for_plots[f"{ood_name}_recall_by_threshold"] = recall_at_threshold
                    all_data_for_plots[f"{ood_name}_f1_by_threshold"] = f1_at_threshold
                    all_data_for_plots[f"{ood_name}_detection_acc_by_threshold"] = detection_acc_at_threshold
                    
                    all_results_metrics[f"{ood_name}_Best_F1_Score"] = best_f1
                    all_results_metrics[f"{ood_name}_Best_F1_Threshold"] = best_f1_threshold
                    
                    # ROC 데이터 저장
                    all_data_for_plots[f"{ood_name}_roc_fpr"] = fpr
                    all_data_for_plots[f"{ood_name}_roc_tpr"] = tpr
                    
                    print(f"  Best F1 Score for {ood_name}: {best_f1:.4f} at threshold: {best_f1_threshold:.4f}")
            else: 
                all_results_metrics[f"{ood_name}_AUROC"] = 0.0; all_results_metrics[f"{ood_name}_FPR@TPR90"] = 1.0
                all_results_metrics[f"{ood_name}_AUPR_In"] = 0.0; all_results_metrics[f"{ood_name}_AUPR_Out"] = 0.0

            id_preds_binary = (id_scores >= chosen_threshold).astype(int)
            ood_preds_binary_current = (current_ood_scores < chosen_threshold).astype(int)
            all_results_metrics[f"{ood_name}_DetectionAccuracy"] = (np.sum(id_preds_binary) + np.sum(ood_preds_binary_current)) / (len(id_scores) + len(current_ood_scores))
            
            ccr = 0.0
            if num_known_classes > 0 and len(id_labels_true_np) > 0:
                known_mask = (id_scores >= chosen_threshold)
                if np.sum(known_mask) > 0:
                    ccr = accuracy_score(id_labels_true_np[known_mask], id_preds_closed_set_np[known_mask])
            oer = np.sum(current_ood_scores >= chosen_threshold) / len(current_ood_scores) if len(current_ood_scores) > 0 else 0.0
            all_results_metrics[f"{ood_name}_OSCR"] = ccr * (1.0 - oer)

    unknown_class_label_int = num_known_classes 
    y_true_kplus1_list, y_pred_kplus1_list = [], []

    if num_known_classes > 0 and len(id_labels_true_np) > 0: 
        y_true_kplus1_list.extend(id_labels_true_np)
        id_preds_for_kplus1 = np.where(id_scores < chosen_threshold, unknown_class_label_int, id_preds_closed_set_np)
        y_pred_kplus1_list.extend(id_preds_for_kplus1)

    id_unknown_samples_scores_for_kplus1 = [] 
    if id_file_unknown_ood_name and ood_loaders and id_file_unknown_ood_name in ood_loaders:
        unknown_loader = ood_loaders[id_file_unknown_ood_name]
        if unknown_loader:
            print(f"\n  Evaluating '{id_file_unknown_ood_name}' for K+1 classification metrics...")
            temp_unknown_scores, temp_unknown_closed_preds = [], []
            with torch.no_grad():
                for batch in tqdm(unknown_loader, desc=f"Processing {id_file_unknown_ood_name} for K+1", leave=False):
                    input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                    
                    logits = model(input_ids, attention_mask, output_features=False)
                    
                    if num_known_classes > 0 and logits.size(1) != num_known_classes:
                         pass # 경고는 위에서 출력됨
                    elif num_known_classes == 0 and logits.size(1) != 1: 
                         pass # 경고는 위에서 출력됨

                    softmax_probs = F.softmax(logits / temperature, dim=1)
                    max_probs, closed_preds_for_unknown = softmax_probs.max(dim=1)
                    temp_unknown_scores.append(max_probs.cpu())
                    temp_unknown_closed_preds.append(closed_preds_for_unknown.cpu())
                    y_true_kplus1_list.extend(np.full(len(input_ids), unknown_class_label_int)) 

            if temp_unknown_scores:
                id_unknown_samples_scores_for_kplus1 = torch.cat(temp_unknown_scores).numpy()
                all_data_for_plots[f"ood_scores_{id_file_unknown_ood_name}"] = id_unknown_samples_scores_for_kplus1
                
                # "unknown" 클래스 스코어 저장 (아직 저장되지 않은 경우)
                if not unknown_data_collected:
                    unknown_scores = id_unknown_samples_scores_for_kplus1
                    unknown_data_collected = True

                id_unknown_samples_closed_preds = torch.cat(temp_unknown_closed_preds).numpy()
                
                if num_known_classes > 0:
                    preds_for_unknown_kplus1 = np.where(id_unknown_samples_scores_for_kplus1 < chosen_threshold,
                                                        unknown_class_label_int,
                                                        id_unknown_samples_closed_preds)
                else: 
                    preds_for_unknown_kplus1 = np.full_like(id_unknown_samples_scores_for_kplus1, unknown_class_label_int, dtype=int)
                y_pred_kplus1_list.extend(preds_for_unknown_kplus1)
    
    # MSP 분포 분석을 위한 정보 저장
    if unknown_data_collected and analyze_thresholds:
        all_data_for_plots["unknown_scores"] = unknown_scores
        
        # 히스토그램 데이터를 bins로 계산
        if len(id_scores) > 0 and len(unknown_scores) > 0:
            bins = np.linspace(0, 1, 21)  # 0-1 사이를 20개 구간으로 나눔
            id_hist, _ = np.histogram(id_scores, bins=bins, density=True)
            unknown_hist, _ = np.histogram(unknown_scores, bins=bins, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            all_data_for_plots["id_hist"] = id_hist
            all_data_for_plots["unknown_hist"] = unknown_hist
            all_data_for_plots["hist_bins"] = bin_centers
    
    if y_true_kplus1_list and y_pred_kplus1_list:
        y_true_kplus1_np = np.array(y_true_kplus1_list)
        y_pred_kplus1_np = np.array(y_pred_kplus1_list)

        kplus1_labels_list = list(range(num_known_classes)) + [unknown_class_label_int]
        
        cm_kplus1_class_names = []
        if known_class_names and num_known_classes > 0:
            cm_kplus1_class_names.extend(known_class_names)
        cm_kplus1_class_names.append(f"{id_exclude_class_name}_as_class")
        all_data_for_plots["kplus1_class_names"] = cm_kplus1_class_names

        if len(np.unique(y_true_kplus1_np)) > 0:
            cm_kplus1 = confusion_matrix(y_true_kplus1_np, y_pred_kplus1_np, labels=kplus1_labels_list)
            all_data_for_plots["cm_kplus1"] = cm_kplus1

            f1_macro_kplus1 = f1_score(y_true_kplus1_np, y_pred_kplus1_np, labels=kplus1_labels_list, average='macro', zero_division=0)
            all_results_metrics[f"F1_Macro_Kplus1_incl_{id_exclude_class_name}"] = f1_macro_kplus1

            if unknown_class_label_int in kplus1_labels_list and cm_kplus1.shape[0] == len(kplus1_labels_list):
                try:
                    unknown_idx_in_cm = kplus1_labels_list.index(unknown_class_label_int)
                    tp_unknown = cm_kplus1[unknown_idx_in_cm, unknown_idx_in_cm]
                    fp_unknown = np.sum(cm_kplus1[:, unknown_idx_in_cm]) - tp_unknown
                    fn_unknown = np.sum(cm_kplus1[unknown_idx_in_cm, :]) - tp_unknown
                    precision_unknown = tp_unknown / (tp_unknown + fp_unknown) if (tp_unknown + fp_unknown) > 0 else 0.0
                    recall_unknown = tp_unknown / (tp_unknown + fn_unknown) if (tp_unknown + fn_unknown) > 0 else 0.0
                    f1_unknown = 2 * (precision_unknown * recall_unknown) / (precision_unknown + recall_unknown) if (precision_unknown + recall_unknown) > 0 else 0.0
                    all_results_metrics[f"{id_exclude_class_name}_Precision_Kplus1"] = precision_unknown
                    all_results_metrics[f"{id_exclude_class_name}_Recall_Kplus1 (DetectionRate)"] = recall_unknown
                    all_results_metrics[f"{id_exclude_class_name}_F1_Kplus1"] = f1_unknown
                except ValueError: pass 
        else: 
            all_results_metrics[f"F1_Macro_Kplus1_incl_{id_exclude_class_name}"] = 0.0
            all_results_metrics[f"{id_exclude_class_name}_Recall_Kplus1 (DetectionRate)"] = 0.0

    if return_data:
        return all_results_metrics, all_data_for_plots
    return all_results_metrics

def plot_threshold_performance(threshold_range, precision_values, recall_values, f1_values, detection_acc_values=None, title="", save_path=None):
    """다양한 임계값에 따른 성능 지표를 시각화합니다."""
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_range, precision_values, 'b-', label='Precision')
    plt.plot(threshold_range, recall_values, 'g-', label='Recall')
    plt.plot(threshold_range, f1_values, 'r-', label='F1 Score')
    if detection_acc_values is not None:
        plt.plot(threshold_range, detection_acc_values, 'y-', label='Detection Accuracy')
    
    # 최적 F1 임계값 찾기
    best_f1_idx = np.argmax(f1_values)
    best_f1_threshold = threshold_range[best_f1_idx]
    best_f1 = f1_values[best_f1_idx]
    
    plt.axvline(x=best_f1_threshold, color='k', linestyle='--', 
                label=f'Best F1: {best_f1:.4f} at {best_f1_threshold:.4f}')
    
    plt.xlabel('MSP Threshold')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_confidence_histograms(id_scores: np.ndarray, ood_scores: np.ndarray, title: str, save_path: Optional[str] = None):
    if not SNS_AVAILABLE: return
    plt.figure(figsize=(10, 6))
    id_scores_valid = id_scores[~np.isnan(id_scores)] if id_scores is not None and len(id_scores) > 0 else np.array([])
    ood_scores_valid = ood_scores[~np.isnan(ood_scores)] if ood_scores is not None and len(ood_scores) > 0 else np.array([])
    
    if len(id_scores_valid) == 0 and len(ood_scores_valid) == 0:
        print(f"Skipping histogram '{title}' as no valid ID or OOD scores provided.")
        plt.close()
        return

    if len(id_scores_valid) > 0:
        sns.histplot(id_scores_valid, bins=50, alpha=0.5, label='In-Distribution', color='blue', stat='density', kde=True)
    if len(ood_scores_valid) > 0:
        sns.histplot(ood_scores_valid, bins=50, alpha=0.5, label='Out-of-Distribution', color='red', stat='density', kde=True)
    
    plt.xlabel('Confidence Score (Max Softmax Probability)')
    plt.ylabel('Density')
    plt.title(title)
    if len(id_scores_valid) > 0 or len(ood_scores_valid) > 0 : plt.legend() 
    plt.grid(alpha=0.3)
    if save_path: plt.savefig(save_path); plt.close()
    else: plt.show(); plt.close()

def plot_roc_curve(id_scores: np.ndarray, ood_scores: np.ndarray, title: str, save_path: Optional[str] = None):
    if id_scores is None or ood_scores is None or len(id_scores) == 0 or len(ood_scores) == 0:
        print(f"Skipping ROC plot for '{title}' due to missing ID or OOD scores.")
        return
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_scores_concat = np.concatenate([id_scores, ood_scores])
    valid_indices = ~np.isnan(y_scores_concat)
    y_true = y_true[valid_indices]
    y_scores_concat = y_scores_concat[valid_indices]

    if len(np.unique(y_true)) < 2:
        print(f"Skipping ROC plot for '{title}' as only one class type is present in scores.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_scores_concat)
    auroc_val = roc_auc_score(y_true, y_scores_concat)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, lw=2, label=f'AUROC = {auroc_val:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.01, 1.01]); plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(title); plt.legend(loc='lower right'); plt.grid(alpha=0.3)
    if save_path: plt.savefig(save_path); plt.close()
    else: plt.show(); plt.close()

def plot_confusion_matrix_custom(cm: np.ndarray, class_names: List[str], title: str, save_path: Optional[str] = None):
    if not SNS_AVAILABLE:
        print("Seaborn not available, skipping confusion matrix plot.")
        return
    if cm is None:
        print(f"Skipping confusion matrix plot for '{title}' as CM data is None.")
        return
    if not class_names:
        print(f"Skipping confusion matrix plot for '{title}' as class_names is empty.")
        return
    if cm.shape[0] != len(class_names) or cm.shape[1] != len(class_names):
        print(f"Warning: Mismatch CM shape {cm.shape} and class names {len(class_names)}. Required: {len(class_names)}x{len(class_names)}. Skipping plot for '{title}'.")
        return

    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(max(8, int(len(class_names)*0.7)), max(6, int(len(class_names)*0.6))))
    try:
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True, annot_kws={"size": 8 if len(class_names) > 10 else 10})
        plt.title(title, fontsize=14); plt.ylabel('Actual Label', fontsize=12); plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10); plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        if save_path: plt.savefig(save_path); print(f"Confusion matrix plot saved to: {save_path}")
        else: plt.show()
        plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for '{title}': {e}"); plt.close()

def plot_tsne_custom(id_features: np.ndarray, ood_features: Optional[np.ndarray], title: str, save_path: Optional[str] = None, seed: int = 42, perplexity: int = 30, n_iter: int = 1000):
    if (id_features is None or len(id_features) == 0) and (ood_features is None or len(ood_features) == 0):
        print(f"No features for t-SNE plot '{title}'.")
        return

    features_list = []
    legend_elements = []

    if id_features is not None and len(id_features) > 0:
        features_list.append(id_features)
        legend_elements.append({'label': 'In-Distribution (ID)', 'color': 'blue'})

    if ood_features is not None and len(ood_features) > 0:
        features_list.append(ood_features)
        legend_elements.append({'label': 'Out-of-Distribution (OOD)', 'color': 'red'})
    elif id_features is not None and len(id_features) > 0 :
        print(f"Warning: No OOD features for t-SNE plot '{title}'. Plotting ID features only.")


    if not features_list:
        print(f"No valid features for t-SNE plot '{title}'.")
        return

    features_all = np.vstack(features_list)

    print(f"Running t-SNE on {features_all.shape[0]} samples for '{title}' (perplexity={perplexity})...")
    try:
        effective_perplexity = min(perplexity, features_all.shape[0] - 1)
        if effective_perplexity <= 1:
            print(f"Warning: Very few samples for t-SNE ({features_all.shape[0]}). Results might be unstable.")
            if features_all.shape[0] <=1 :
                print("Error: Cannot run t-SNE with 1 or fewer samples. Skipping plot.")
                return
        tsne = TSNE(n_components=2, random_state=seed, perplexity=effective_perplexity, n_iter=n_iter, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(features_all)
    except Exception as e:
        print(f"Error running t-SNE for '{title}': {e}. Skipping plot.")
        return

    plt.figure(figsize=(10, 8))
    current_offset = 0
    for i, el in enumerate(legend_elements):
        num_samples_for_label = len(features_list[i])
        indices_for_plot = np.arange(current_offset, current_offset + num_samples_for_label)

        plt.scatter(tsne_results[indices_for_plot, 0], tsne_results[indices_for_plot, 1],
                    c=el['color'], label=el['label'], alpha=0.5, s=15)
        current_offset += num_samples_for_label

    plt.title(title, fontsize=14)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    if legend_elements: plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"t-SNE plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def run_single_experiment(args_dict: Dict[str, Any], id_dataset_name: str, current_oe_source_name: Optional[str] = None, current_oe_data_path: Optional[str] = None):
    experiment_tag = id_dataset_name
    if current_oe_source_name:
        oe_name_part = os.path.basename(current_oe_source_name) if os.path.isfile(current_oe_source_name) else current_oe_source_name
        experiment_tag += f"_OE_{oe_name_part}"
    else:
        experiment_tag += "_Standard"

    print(f"\n\n===== Starting Single Experiment: {experiment_tag} =====")

    sanitized_oe_name = ""
    if current_oe_source_name:
        sanitized_oe_name = re.sub(r'[^\w\-.]+', '_', os.path.basename(current_oe_source_name) if os.path.isfile(current_oe_source_name) else current_oe_source_name)

    exp_result_subdir = os.path.join(id_dataset_name, f"OE_{sanitized_oe_name}" if sanitized_oe_name else "Standard")
    current_result_dir = os.path.join(args_dict['result_dir_base'], exp_result_subdir)
    current_model_dir = os.path.join(args_dict['model_dir_base'], exp_result_subdir)
    os.makedirs(current_result_dir, exist_ok=True)
    if args_dict['save_model_per_experiment']: os.makedirs(current_model_dir, exist_ok=True)

    print(f"Results dir: {current_result_dir}")
    if args_dict['save_model_per_experiment']: print(f"Models dir: {current_model_dir}")

    tokenizer = RobertaTokenizer.from_pretrained(args_dict['model_type'])

    print("\n--- Preparing Datasets ---")
    train_dataset, id_test_dataset, num_classes, label_encoder, id_label2id, id_id2label = None, None, 0, None, {}, {}
    id_unknown_dataset_from_id_file = None
    external_ood_test_dataset = None
    known_class_names_list = []
    
    id_unknown_loader_name_tag = f"{args_dict['id_exclude_class']}_from_ID_file"
    external_ood_name_tag = "external_ood" 

    if id_dataset_name == 'syslog':
        train_dataset, id_test_dataset, id_unknown_dataset_from_id_file, num_classes, label_encoder, id_label2id, id_id2label = prepare_syslog_data(
            tokenizer, args_dict['max_length'], args_dict['id_data_path'], args_dict['text_col'], args_dict['class_col'], args_dict['id_exclude_class'], args_dict['seed']
        )
        if not train_dataset and not id_unknown_dataset_from_id_file and num_classes == 0 :
            print(f"Critical error: No Syslog data (knowns or unknowns). Skipping: {experiment_tag}"); return {}, {}
        
        external_ood_test_dataset = prepare_syslog_ood_data(
            tokenizer, args_dict['max_length'], args_dict['ood_data_path'], args_dict['text_col'], args_dict['class_col'], args_dict['ood_target_class']
        )
        external_ood_name_tag = args_dict['ood_target_class'] if external_ood_test_dataset else "external_syslog_ood"
        if num_classes > 0 and id_id2label: known_class_names_list = list(id_id2label.values())

    elif id_dataset_name == '20newsgroups':
        train_dataset, id_test_dataset, num_classes, label_encoder, id_label2id, id_id2label, ood_categories_20ng = prepare_20newsgroups_data(
            tokenizer, args_dict['max_length'], args_dict['num_20ng_id_classes'], args_dict['seed']
        )
        if not train_dataset: print(f"Critical error: No 20NG ID data. Skipping: {experiment_tag}"); return {}, {}
        
        external_ood_test_dataset = prepare_20newsgroups_ood_data(
            tokenizer, args_dict['max_length'], ood_categories_20ng, args_dict['seed']
        )
        external_ood_name_tag = "20ng_ood" if external_ood_test_dataset else "external_20ng_ood"
        if num_classes > 0 and id_id2label: known_class_names_list = [id_id2label.get(i, f"Class_{i}") for i in range(num_classes)]
    else:
        raise ValueError(f"Unsupported ID dataset name: {id_dataset_name}")

    if num_classes == 0 and not id_unknown_dataset_from_id_file:
         print(f"No known classes and no unknown-from-ID data. Cannot proceed with experiment '{experiment_tag}'.")
         return {}, {}

    num_workers = args_dict.get('num_dataloader_workers', 0) # 기본값 0으로 설정
    train_loader = DataLoader(train_dataset, batch_size=args_dict['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0) if train_dataset else None
    id_test_loader = DataLoader(id_test_dataset, batch_size=args_dict['batch_size'], num_workers=num_workers, pin_memory=True) if id_test_dataset else None
    
    ood_loaders_for_eval = {}
    if external_ood_test_dataset:
        ood_loaders_for_eval[external_ood_name_tag] = DataLoader(external_ood_test_dataset, batch_size=args_dict['batch_size'], num_workers=num_workers, pin_memory=True)
    if id_unknown_dataset_from_id_file:
        ood_loaders_for_eval[id_unknown_loader_name_tag] = DataLoader(id_unknown_dataset_from_id_file, batch_size=args_dict['batch_size'], num_workers=num_workers, pin_memory=True)

    model = RoBERTaOOD(num_classes, args_dict['model_type']).to(device)
    if num_classes > 0 and id_label2id and id_id2label:
        model.roberta.config.label2id = id_label2id
        model.roberta.config.id2label = id_id2label
    elif num_classes == 0: 
        model.roberta.config.label2id = {"dummy_class": 0}
        model.roberta.config.id2label = {0: "dummy_class"}

    model_filename_base = f"roberta_{experiment_tag}_{num_classes}cls_seed{args_dict['seed']}.pt"
    model_save_path = os.path.join(current_model_dir, model_filename_base)
    experiment_results_dict, experiment_data_for_plots_dict = {}, {}

    if args_dict['eval_only']:
        if os.path.exists(model_save_path):
            print(f"Loading pre-trained model: {model_save_path}")
            model.load_state_dict(torch.load(model_save_path, map_location=device))
        else:
            print(f"Error: Model path '{model_save_path}' not found for eval_only. Skipping: {experiment_tag}"); return {}, {}
    else: 
        if not train_loader or num_classes == 0: 
            print(f"Warning: No training data or no known classes for '{experiment_tag}'. Model will not be trained.")
            if not os.path.exists(model_save_path) and args_dict['save_model_per_experiment']:
                 torch.save(model.state_dict(), model_save_path); print(f"  Saved un-trained model placeholder: {model_save_path}")
        else: 
            optimizer = AdamW(model.parameters(), lr=args_dict['learning_rate'])
            total_steps = len(train_loader) * args_dict['num_epochs']
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
            if current_oe_source_name: 
                oe_train_dataset_current = None
                if current_oe_data_path and os.path.exists(current_oe_data_path):
                    oe_train_dataset_current = prepare_syslog_masked_oe_data(tokenizer, args_dict['max_length'], current_oe_data_path, args_dict['oe_masked_text_col'])
                elif current_oe_source_name in ['snli', 'imdb', 'wikitext']:
                    oe_train_dataset_current = prepare_external_oe_data(tokenizer, args_dict['max_length'], current_oe_source_name, data_dir=args_dict['data_dir'], cache_dir_hf=args_dict['cache_dir'])
                
                if oe_train_dataset_current:
                    oe_loader = DataLoader(oe_train_dataset_current, batch_size=args_dict['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
                    train_with_oe_uniform_loss(model, train_loader, oe_loader, optimizer, scheduler, device, args_dict['num_epochs'], args_dict['oe_lambda'], experiment_tag)
                    del oe_loader, oe_train_dataset_current; gc.collect()
                else:
                    print(f"Failed to load OE data for '{current_oe_source_name}'. Training standard model instead.")
                    train_standard(model, train_loader, optimizer, scheduler, device, args_dict['num_epochs'], experiment_tag)
            else: 
                train_standard(model, train_loader, optimizer, scheduler, device, args_dict['num_epochs'], experiment_tag)
            
            if args_dict['save_model_per_experiment']:
                torch.save(model.state_dict(), model_save_path); print(f"Model saved: {model_save_path}")

    eval_id_loader = id_test_loader
    if not id_test_loader and num_classes == 0:
        print("Info: No ID (known) test data. Using dummy empty loader for ID part of evaluate_osr.")
        dummy_empty_dataset = TextDataset([], [], tokenizer, args_dict['max_length'])
        eval_id_loader = DataLoader(dummy_empty_dataset, batch_size=1)

    if not eval_id_loader and not ood_loaders_for_eval: 
        print(f"No data to evaluate for {experiment_tag}. Skipping evaluation.")
    else:
        results_osr, data_osr = evaluate_osr(
            model, eval_id_loader, ood_loaders_for_eval, device, args_dict['temperature'],
            threshold_percentile=args_dict.get('osr_threshold_percentile', 5.0), return_data=True,
            num_known_classes=num_classes,
            id_file_unknown_ood_name=id_unknown_loader_name_tag if id_unknown_dataset_from_id_file else None,
            known_class_names=known_class_names_list if num_classes > 0 else [],
            id_exclude_class_name=args_dict['id_exclude_class'],
            analyze_thresholds=True  # "unknown" 클래스 탐지에 대한 임계값 성능 분석
        )
        print(f"\n  OSSR Evaluation Results for Experiment '{experiment_tag}':")
        for k_res, v_res in results_osr.items(): print(f"    {k_res}: {v_res:.4f}" if isinstance(v_res, float) else f"    {k_res}: {v_res}")

        full_experiment_setup_key = re.sub(r'[^\w\-]+', '_', experiment_tag) 
        experiment_results_dict[full_experiment_setup_key] = results_osr
        experiment_data_for_plots_dict[full_experiment_setup_key] = data_osr

        if not args_dict['no_plot_per_experiment']:
            plot_filename_base_exp = full_experiment_setup_key 

            id_scores_plot = data_osr.get('id_scores')
            id_features_plot = data_osr.get('id_features')
            for ood_eval_name_key_plot in ood_loaders_for_eval.keys():
                ood_scores_plot = data_osr.get(f"ood_scores_{ood_eval_name_key_plot}")
                ood_features_plot = data_osr.get(f"ood_features_{ood_eval_name_key_plot}")
                plot_prefix_ood = f"{plot_filename_base_exp}_vs_{re.sub(r'[^\w-]+', '_', ood_eval_name_key_plot)}"
                
                if id_scores_plot is not None and len(id_scores_plot) > 0 and ood_scores_plot is not None and len(ood_scores_plot) > 0:
                    plot_confidence_histograms(id_scores_plot, ood_scores_plot, f'Confidence - {experiment_tag} vs {ood_eval_name_key_plot}', os.path.join(current_result_dir, f'{plot_prefix_ood}_hist.png'))
                    plot_roc_curve(id_scores_plot, ood_scores_plot, f'ROC - {experiment_tag} vs {ood_eval_name_key_plot}', os.path.join(current_result_dir, f'{plot_prefix_ood}_roc.png'))
                if id_features_plot is not None and len(id_features_plot) > 0 and ood_features_plot is not None and len(ood_features_plot) > 0:
                     plot_tsne_custom(id_features_plot, ood_features_plot, f't-SNE - {experiment_tag} (ID vs OOD: {ood_eval_name_key_plot})', os.path.join(current_result_dir, f'{plot_prefix_ood}_tsne.png'), seed=args_dict['seed'])
                     
                # 새로운 임계값 성능 시각화 추가
                threshold_range = data_osr.get("threshold_range")
                precision_values = data_osr.get(f"{ood_eval_name_key_plot}_precision_by_threshold")
                recall_values = data_osr.get(f"{ood_eval_name_key_plot}_recall_by_threshold")
                f1_values = data_osr.get(f"{ood_eval_name_key_plot}_f1_by_threshold")
                detection_acc_values = data_osr.get(f"{ood_eval_name_key_plot}_detection_acc_by_threshold")
                
                if threshold_range is not None and precision_values is not None and recall_values is not None and f1_values is not None:
                    plot_threshold_performance(
                        threshold_range, precision_values, recall_values, f1_values, detection_acc_values,
                        f'OSR Performance Metrics by Threshold - {experiment_tag} vs {ood_eval_name_key_plot}',
                        os.path.join(current_result_dir, f'{plot_prefix_ood}_threshold_perf.png')
                    )
                    
                    # ROC 곡선 시각화
                    fpr = data_osr.get(f"{ood_eval_name_key_plot}_roc_fpr")
                    tpr = data_osr.get(f"{ood_eval_name_key_plot}_roc_tpr")
                    if fpr is not None and tpr is not None:
                        plt.figure(figsize=(8, 8))
                        plt.plot(fpr, tpr, lw=2)
                        plt.plot([0, 1], [0, 1], 'k--', lw=2)
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.0])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title(f'ROC Curve - {experiment_tag} vs {ood_eval_name_key_plot}')
                        plt.grid(alpha=0.3)
                        plt.savefig(os.path.join(current_result_dir, f'{plot_prefix_ood}_roc_detailed.png'))
                        plt.close()
                
                # MSP 분포 히스토그램 시각화
                id_hist = data_osr.get("id_hist")
                unknown_hist = data_osr.get("unknown_hist")
                hist_bins = data_osr.get("hist_bins")
                
                if id_hist is not None and unknown_hist is not None and hist_bins is not None:
                    plt.figure(figsize=(10, 6))
                    plt.bar(hist_bins, id_hist, width=0.04, alpha=0.7, color='blue', label='ID')
                    plt.bar(hist_bins, unknown_hist, width=0.04, alpha=0.7, color='red', label='Unknown')
                    plt.xlabel('MSP Score')
                    plt.ylabel('Density')
                    plt.title(f'MSP Distribution - {experiment_tag}')
                    plt.legend()
                    plt.grid(alpha=0.3)
                    plt.savefig(os.path.join(current_result_dir, f'{plot_filename_base_exp}_msp_distribution.png'))
                    plt.close()
            
            if num_classes > 0 and data_osr.get('id_labels_true') is not None and len(data_osr['id_labels_true']) > 0:
                cm_closed_set = confusion_matrix(data_osr['id_labels_true'], data_osr['id_preds_closed_set'], labels=np.arange(num_classes))
                plot_confusion_matrix_custom(cm_closed_set, known_class_names_list, f'CM (Closed-Set {num_classes} classes) - {experiment_tag}', os.path.join(current_result_dir, f'{plot_filename_base_exp}_cm_closed_set.png'))

            if data_osr.get("cm_kplus1") is not None and data_osr.get("kplus1_class_names") is not None:
                cm_kplus1_data = data_osr["cm_kplus1"]
                kplus1_names_plot = data_osr["kplus1_class_names"]
                
                if cm_kplus1_data.shape[0] == len(kplus1_names_plot):
                    unknown_idx = len(kplus1_names_plot) - 1  # Unknown 클래스는 마지막에 있다고 가정
                    
                    # 기존 confusion matrix
                    plot_confusion_matrix_custom(
                        cm_kplus1_data, kplus1_names_plot,
                        f'CM (K+1 classes incl. {args_dict["id_exclude_class"]}) - {experiment_tag}',
                        os.path.join(current_result_dir, f'{plot_filename_base_exp}_cm_Kplus1_with_unknown.png')
                    )
                    
                    # Unknown 클래스에 초점을 맞춘 confusion matrix
                    plot_unknown_confusion_matrix(
                        cm_kplus1_data, kplus1_names_plot, unknown_idx,
                        f'Unknown-focused CM - {experiment_tag}',
                        os.path.join(current_result_dir, f'{plot_filename_base_exp}_cm_unknown_focused.png')
                    )
                    
                    # 이진 OSR confusion matrix (Known vs Unknown)
                    binary_cm, metrics = create_binary_osr_confusion_matrix(cm_kplus1_data, kplus1_names_plot, unknown_idx)
                    plot_binary_osr_confusion_matrix(
                        binary_cm, metrics,
                        f'OSR Binary CM (Known vs Unknown) - {experiment_tag}',
                        os.path.join(current_result_dir, f'{plot_filename_base_exp}_cm_binary_osr.png')
                    )
                else:
                    print(f"Warning: Skipping K+1 CM plot for {experiment_tag} due to shape/name mismatch.")
            else:
                print(f"Note: K+1 CM data not available for plotting for {experiment_tag}.")
    # 변수 정리 부분 수정
    del model
    
    # 변수의 참조 상태 먼저 확인
    should_delete_eval_id_loader = False
    if 'eval_id_loader' in locals() and 'id_test_loader' in locals():
        should_delete_eval_id_loader = eval_id_loader is not id_test_loader

    # 이제 DataLoader들을 삭제
    if 'train_loader' in locals() and train_loader: 
        del train_loader
    if 'id_test_loader' in locals() and id_test_loader: 
        del id_test_loader
    if should_delete_eval_id_loader and 'eval_id_loader' in locals():
        del eval_id_loader
    if 'dummy_empty_dataset' in locals(): 
        del dummy_empty_dataset
        
    for loader_key in list(ood_loaders_for_eval.keys()): # 키 리스트 복사 후 순회
        if ood_loaders_for_eval[loader_key]:
            del ood_loaders_for_eval[loader_key]
    del ood_loaders_for_eval
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    return experiment_results_dict, experiment_data_for_plots_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RoBERTa OSR Comparison')
    parser.add_argument('--id_dataset_type', type=str, default='syslog', choices=['syslog', '20newsgroups'])
    parser.add_argument('--id_data_path', type=str, default=ID_SYSLOG_PATH)
    parser.add_argument('--ood_data_path', type=str, default=OOD_SYSLOG_UNKNOWN_PATH)
    parser.add_argument('--result_dir_base', type=str, default=RESULT_DIR_BASE)
    parser.add_argument('--model_dir_base', type=str, default=MODEL_DIR_BASE)
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--cache_dir', type=str, default=CACHE_DIR)
    parser.add_argument('--oe_config_file', type=str, default=None)
    parser.add_argument('--oe_masked_syslog_path', type=str, default=None)
    parser.add_argument('--oe_sources_external', nargs='+', default=[])
    parser.add_argument('--oe_masked_text_col', type=str, default=OE_MASKED_TEXT_COLUMN)
    parser.add_argument('--model_type', type=str, default=MODEL_TYPE)
    parser.add_argument('--max_length', type=int, default=MAX_LENGTH)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--oe_lambda', type=float, default=OE_LAMBDA)
    parser.add_argument('--temperature', type=float, default=TEMPERATURE)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--osr_threshold_percentile', type=float, default=5.0)
    parser.add_argument('--text_col', type=str, default=TEXT_COLUMN)
    parser.add_argument('--class_col', type=str, default=CLASS_COLUMN)
    parser.add_argument('--id_exclude_class', type=str, default=ID_SYSLOG_EXCLUDE_CLASS)
    parser.add_argument('--ood_target_class', type=str, default=OOD_SYSLOG_TARGET_CLASS)
    parser.add_argument('--num_20ng_id_classes', type=int, default=NUM_20NG_ID_CLASSES)
    parser.add_argument('--save_model_per_experiment', default=SAVE_MODEL_PER_EXPERIMENT, action=argparse.BooleanOptionalAction)
    parser.add_argument('--eval_only', default=EVAL_ONLY, action='store_true')
    parser.add_argument('--no_plot_per_experiment', default=NO_PLOT_PER_EXPERIMENT, action=argparse.BooleanOptionalAction)
    parser.add_argument('--skip_standard_model', default=SKIP_STANDARD_MODEL, action=argparse.BooleanOptionalAction)
    parser.add_argument('--num_dataloader_workers', type=int, default=0)

    args = parser.parse_args()
    args_dict = vars(args)

    set_seed(args.seed)
    os.makedirs(args.result_dir_base, exist_ok=True)
    if args.save_model_per_experiment: os.makedirs(args.model_dir_base, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    all_experiments_results = {}
    
    if not args.skip_standard_model:
        print("\n--- Running Standard Model Experiment (No OE) ---")
        std_results, _ = run_single_experiment(args_dict, args.id_dataset_type, current_oe_source_name=None, current_oe_data_path=None)
        all_experiments_results.update(std_results)

    oe_tasks_to_run = []
    if args.oe_config_file and os.path.exists(args.oe_config_file):
        with open(args.oe_config_file, 'r') as f:
            for line in f:
                path = line.strip()
                if path and os.path.exists(path): oe_tasks_to_run.append({'name': os.path.basename(path), 'path': path})
                elif path: print(f"Warning: Path '{path}' from config not found.")
    elif args.oe_masked_syslog_path and os.path.exists(args.oe_masked_syslog_path):
        oe_tasks_to_run.append({'name': os.path.basename(args.oe_masked_syslog_path), 'path': args.oe_masked_syslog_path})
    elif args.oe_masked_syslog_path: 
        print(f"Warning: Provided --oe_masked_syslog_path '{args.oe_masked_syslog_path}' not found.")

    for ext_source_name in args.oe_sources_external:
        oe_tasks_to_run.append({'name': ext_source_name, 'path': None}) 

    if not oe_tasks_to_run and args.skip_standard_model:
        print("No OE datasets specified and standard model is skipped. Nothing to run.")
    else:
        for task_info in oe_tasks_to_run:
            print(f"\n--- Running OE Experiment with Source: {task_info['name']} ---")
            if task_info['path']: print(f"  Using OE data path: {task_info['path']}")
            oe_results, _ = run_single_experiment(args_dict, args.id_dataset_type, current_oe_source_name=task_info['name'], current_oe_data_path=task_info['path'])
            all_experiments_results.update(oe_results)

    print("\n\n===== Final Overall Results Summary =====")
    if all_experiments_results:
        final_results_df = pd.DataFrame.from_dict(all_experiments_results, orient='index').sort_index()
        print("Overall Performance Metrics DataFrame:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(final_results_df)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_base = f"osr_overall_summary_{args.id_dataset_type}_{ts}"
        final_results_df.to_csv(os.path.join(args.result_dir_base, f'{summary_base}.csv'))
        print(f"\nOverall results saved to CSV: {os.path.join(args.result_dir_base, f'{summary_base}.csv')}")
        
        with open(os.path.join(args.result_dir_base, f'{summary_base}.txt'), 'w', encoding='utf-8') as f:
            f.write("--- Experiment Arguments ---\n")
            f.write(json.dumps(args_dict, indent=4, default=str) + "\n\n")
            f.write("--- Overall Metrics ---\n")
            f.write(final_results_df.to_string())
        print(f"Overall results and arguments saved to TXT: {os.path.join(args.result_dir_base, f'{summary_base}.txt')}")

        json_summary = {'arguments': args_dict, 'timestamp': ts, 'results': all_experiments_results}
        with open(os.path.join(args.result_dir_base, f'{summary_base}.json'), 'w', encoding='utf-8') as f:
            json.dump(json_summary, f, indent=4, default=str)
        print(f"Overall results and arguments saved to JSON: {os.path.join(args.result_dir_base, f'{summary_base}.json')}")
    else:
        print("No performance metrics were generated.")
    print("\nAll specified experiments finished.")