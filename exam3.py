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
from sklearn.datasets import fetch_20newsgroups # 20 Newsgroups 로딩용
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import random
import matplotlib
import gc
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
from datasets import load_dataset, DatasetDict, concatenate_datasets # Hugging Face datasets
import re

# --- 기본 설정값 (명령줄 인자로 덮어쓰기 가능) ---
ID_SYSLOG_PATH = 'log_all_critical.csv'
OOD_SYSLOG_UNKNOWN_PATH = 'log_unknown.csv'
TEXT_COLUMN = 'text'
CLASS_COLUMN = 'class'
OE_MASKED_TEXT_COLUMN = 'masked_text_attention' # OE 파일에서 사용할 텍스트 컬럼
ID_SYSLOG_EXCLUDE_CLASS = "unknown"
OOD_SYSLOG_TARGET_CLASS = "unknown"
NUM_20NG_ID_CLASSES = 10
RESULT_DIR_BASE = '03_results_osr_multi_oe_experiments' # 기본 결과 저장 디렉토리
MODEL_DIR_BASE = '03_models_osr_multi_oe_experiments'   # 기본 모델 저장 디렉토리
DATA_DIR = 'data' # 외부 데이터셋 저장용
CACHE_DIR = os.path.join(DATA_DIR, "hf_cache") # Hugging Face 캐시
MODEL_TYPE = 'roberta-base'
MAX_LENGTH = 128
BATCH_SIZE = 64
NUM_EPOCHS = 20 # 각 실험별 에포크 수
LEARNING_RATE = 2e-5
OE_LAMBDA = 1.0
TEMPERATURE = 1.0
SEED = 42
SAVE_MODEL_PER_EXPERIMENT = True # 각 OE 실험별 모델 저장 여부
EVAL_ONLY = False
NO_PLOT_PER_EXPERIMENT = False # 각 OE 실험별 플롯 생성 여부
SKIP_STANDARD_MODEL = False
SKIP_ALL_OE_EXPERIMENTS = False # 이 플래그는 config 파일 사용 시 의미가 퇴색될 수 있음

# --- 함수 정의 (이전 코드와 대부분 동일, 필요시 약간의 로깅/경로 처리 수정) ---
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
        # print(f"Tokenizing {len(valid_texts)} texts...") # 로그 축소
        self.encodings = tokenizer(valid_texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def prepare_syslog_data(tokenizer, max_length: int, id_data_path: str, text_col: str, class_col: str, exclude_class: str, seed: int = 42) -> Tuple[Optional[Dataset], Optional[Dataset], int, Optional[LabelEncoder], Dict, Dict]:
    print(f"\n--- Preparing Syslog ID data from: {id_data_path} ---")
    try:
        df = pd.read_csv(id_data_path)
        if not all(c in df.columns for c in [text_col, class_col]):
            raise ValueError(f"ID Data CSV '{id_data_path}' must contain '{text_col}' and '{class_col}' columns.")
        df = df.dropna(subset=[text_col, class_col])
        df[class_col] = df[class_col].astype(str).str.lower() # 클래스명 소문자 통일
        df_known_initial = df[df[class_col] != exclude_class.lower()].copy()

        if df_known_initial.empty:
            print(f"Warning: No data left after excluding class '{exclude_class}' from '{id_data_path}'.")
            return None, None, 0, None, {}, {}

        # 클래스별 최소 샘플 수 필터링 (학습/테스트 분할 가능하도록)
        class_counts = df_known_initial[class_col].value_counts()
        classes_to_keep = class_counts[class_counts >= 2].index # 최소 2개는 있어야 train/test 분할 가능
        if len(classes_to_keep) < len(class_counts):
            print(f"  - Filtering classes with < 2 samples. Kept: {len(classes_to_keep)} classes.")
        df_known_final = df_known_initial[df_known_initial[class_col].isin(classes_to_keep)].copy()

        if df_known_final.empty:
            print(f"Warning: No data left after filtering classes with < 2 samples from '{id_data_path}'.")
            return None, None, 0, None, {}, {}

        final_classes = sorted(df_known_final[class_col].unique())
        num_classes_final = len(final_classes)
        if num_classes_final == 0:
            print(f"Warning: No classes identified for training in '{id_data_path}'.")
            return None, None, 0, None, {}, {}

        final_label_encoder = LabelEncoder()
        final_label_encoder.fit(final_classes) # 최종 사용될 클래스로만 인코더 학습
        final_label2id = {label: i for i, label in enumerate(final_label_encoder.classes_)}
        final_id2label = {i: label for label, i in final_label2id.items()}
        print(f"  - Final number of known classes: {num_classes_final}")
        print(f"  - Final Label to ID mapping: {final_label2id}")

        df_known_final['label'] = df_known_final[class_col].map(final_label2id)

        # Stratify 가능 여부 확인
        min_class_count_for_split = df_known_final['label'].value_counts().min()
        stratify_labels = df_known_final['label'] if min_class_count_for_split > 1 else None
        if stratify_labels is None:
            print("  - Warning: Not enough samples in some classes for stratified split. Using non-stratified split.")

        train_df, test_df = train_test_split(df_known_final, test_size=0.2, random_state=seed, stratify=stratify_labels)
        print(f"  - Split into Train: {len(train_df)}, Test: {len(test_df)}")

        train_dataset = TextDataset(train_df[text_col].tolist(), train_df['label'].tolist(), tokenizer, max_length)
        id_test_dataset = TextDataset(test_df[text_col].tolist(), test_df['label'].tolist(), tokenizer, max_length)
        return train_dataset, id_test_dataset, num_classes_final, final_label_encoder, final_label2id, final_id2label
    except Exception as e:
        print(f"Error preparing Syslog ID data from '{id_data_path}': {e}")
        return None, None, 0, None, {}, {}

def prepare_20newsgroups_data(tokenizer, max_length: int, num_id_classes: int, seed: int = 42) -> Tuple[Optional[Dataset], Optional[Dataset], int, Optional[LabelEncoder], Dict, Dict, List[str]]:
    print(f"\n--- Preparing 20 Newsgroups ID data ({num_id_classes} classes) ---")
    try:
        all_categories = list(fetch_20newsgroups(subset='train').target_names)
        if num_id_classes <= 0 or num_id_classes > len(all_categories):
            print(f"Warning: Invalid num_id_classes ({num_id_classes}). Using all {len(all_categories)} categories.")
            num_id_classes = len(all_categories)

        np.random.seed(seed) # 선택의 일관성 보장
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
        # OOD 데이터의 레이블은 일반적으로 -1 또는 ID 클래스 수보다 큰 값으로 설정
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
        # OE 데이터의 레이블은 -1 (또는 ID 클래스와 겹치지 않는 값)
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
    elif oe_source_name == "wikitext": # wikitext-103-raw-v1
        dataset_config = {"name": "wikitext", "config_name": "wikitext-103-raw-v1", "split": "train", "text_col": "text"}
    else:
        print(f"Error: Unknown external OE source name '{oe_source_name}'")
        return None

    try:
        print(f"  Loading {oe_source_name} (config: {dataset_config.get('config_name', 'default')}, split: {dataset_config['split']})...")
        # load_dataset은 config_name 인자를 사용
        ds = load_dataset(dataset_config["name"], name=dataset_config.get("config_name"), split=dataset_config["split"], cache_dir=cache_dir_hf)

        if isinstance(ds, DatasetDict): # 일부 데이터셋은 DatasetDict 형태로 반환
            if dataset_config['split'] in ds:
                ds = ds[dataset_config['split']]
            else:
                raise ValueError(f"Split '{dataset_config['split']}' not found in DatasetDict for {oe_source_name}")

        texts = [item for item in ds[dataset_config['text_col']] if isinstance(item, str) and item.strip()]
        if oe_source_name == "wikitext": # 위키텍스트는 제목 등 짧은 라인 제외
            texts = [text for text in texts if not text.strip().startswith("=") and len(text.strip().split()) > 3]

        if not texts:
            print(f"Warning: No valid texts found for OE source {oe_source_name}.")
            return None

        oe_labels = np.full(len(texts), -1, dtype=int).tolist() # OE 레이블은 -1
        oe_dataset = TextDataset(texts, oe_labels, tokenizer, max_length)
        print(f"  - Loaded {len(oe_dataset)} samples for OE training from {oe_source_name}.")
        return oe_dataset
    except Exception as e:
        print(f"Error loading external OE dataset {oe_source_name}: {e}")
        return None

def prepare_syslog_ood_data(tokenizer, max_length: int, ood_data_path: str, text_col: str, class_col: str, ood_target_class: str) -> Optional[Dataset]:
    print(f"\n--- Preparing Syslog OOD data (class: '{ood_target_class}') from: {ood_data_path} ---")
    if not os.path.exists(ood_data_path):
        print(f"Error: OOD data path not found: {ood_data_path}")
        return None
    try:
        df = pd.read_csv(ood_data_path)
        if not all(c in df.columns for c in [text_col, class_col]):
            raise ValueError(f"OOD Data CSV '{ood_data_path}' must contain '{text_col}' and '{class_col}' columns.")
        df = df.dropna(subset=[text_col, class_col])
        df[class_col] = df[class_col].astype(str).str.lower() # 클래스명 소문자 통일
        df_ood = df[df[class_col] == ood_target_class.lower()].copy()

        if df_ood.empty:
            print(f"Warning: No data found for OOD class '{ood_target_class}' in '{ood_data_path}'.")
            return None

        texts = df_ood[text_col].tolist()
        ood_labels = np.full(len(texts), -1, dtype=int).tolist() # OOD 레이블은 -1
        ood_dataset = TextDataset(texts, ood_labels, tokenizer, max_length)
        print(f"  - Loaded {len(ood_dataset)} samples for OOD testing (class: '{ood_target_class}').")
        return ood_dataset
    except Exception as e:
        print(f"Error preparing Syslog OOD data from '{ood_data_path}': {e}")
        return None

class RoBERTaOOD(nn.Module):
    def __init__(self, num_classes: int, model_name: str = 'roberta-base'):
        super(RoBERTaOOD, self).__init__()
        config = RobertaConfig.from_pretrained(model_name, num_labels=num_classes)
        # output_hidden_states=True를 config에 설정해야 hidden_states 반환
        config.output_hidden_states = True
        self.roberta = RobertaForSequenceClassification.from_pretrained(model_name, config=config)

    def forward(self, input_ids, attention_mask, output_features=False):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if output_features:
            # 마지막 레이어의 hidden states 가져오기, [CLS] 토큰의 representation 사용 (보통 첫번째 토큰)
            features = outputs.hidden_states[-1][:, 0, :]
            return logits, features
        else:
            return logits

def train_standard(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, scheduler, device: torch.device, num_epochs: int, current_experiment_name: str):
    model.train()
    use_amp = (device.type == 'cuda') # AMP 사용 여부
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
        scheduler.step() # 에포크 단위 스케줄러 스텝
        avg_loss = total_loss / len(train_loader)
        print(f"Std Epoch {epoch+1}/{num_epochs} ({current_experiment_name}), Avg Loss: {avg_loss:.4f}")

def train_with_oe_uniform_loss(model: nn.Module, train_loader: DataLoader, oe_loader: DataLoader, optimizer: optim.Optimizer, scheduler, device: torch.device, num_epochs: int, oe_lambda: float, current_experiment_name: str):
    model.train()
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Starting OE (Uniform CE Loss) training for '{current_experiment_name}'... AMP enabled: {use_amp}")

    for epoch in range(num_epochs):
        oe_iter = iter(oe_loader) # OE 로더 이터레이터
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
            except StopIteration: # OE 데이터가 ID 데이터보다 적을 경우, OE 로더 재시작
                oe_iter = iter(oe_loader)
                oe_batch = next(oe_iter)

            oe_input_ids = oe_batch['input_ids'].to(device)
            oe_attention_mask = oe_batch['attention_mask'].to(device)
            # OE 데이터의 레이블은 사용하지 않음 (uniform target 사용)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=str(device).split(':')[0], enabled=use_amp):
                # ID 데이터에 대한 손실
                id_logits = model(input_ids, attention_mask)
                id_loss = F.cross_entropy(id_logits, labels)

                # OE 데이터에 대한 손실 (Uniform Cross-Entropy)
                oe_logits = model(oe_input_ids, oe_attention_mask)
                num_classes = oe_logits.size(1)
                log_softmax_oe = F.log_softmax(oe_logits, dim=1)
                # KLDivLoss는 target이 확률 분포여야 함. uniform_target은 log_softmax 이전의 확률 분포.
                uniform_target_probs = torch.full_like(oe_logits, 1.0 / num_classes)
                oe_loss = F.kl_div(log_softmax_oe, uniform_target_probs, reduction='batchmean', log_target=False) # log_target=False가 중요

                total_batch_loss = id_loss + oe_lambda * oe_loss

            scaler.scale(total_batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += total_batch_loss.item()
            total_id_loss += id_loss.item()
            total_oe_loss += oe_loss.item()
            progress_bar.set_postfix({'Total Loss': f"{total_batch_loss.item():.3f}", 'ID Loss': f"{id_loss.item():.3f}", 'OE Loss': f"{oe_loss.item():.3f}"})
        scheduler.step() # 에포크 단위 스케줄러 스텝
        avg_loss = total_loss / len(train_loader)
        avg_id_loss = total_id_loss / len(train_loader)
        avg_oe_loss = total_oe_loss / len(train_loader)
        print(f"OE Epoch {epoch+1}/{num_epochs} ({current_experiment_name}), Avg Loss: {avg_loss:.4f} (ID: {avg_id_loss:.4f}, OE: {avg_oe_loss:.4f})")

def evaluate_osr(model: nn.Module, id_loader: DataLoader, ood_loader: Optional[DataLoader], device: torch.device, temperature: float = 1.0, threshold_percentile: float = 5.0, return_data: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, np.ndarray]]]:
    model.eval()
    id_logits_all, id_scores_all, id_labels_true, id_labels_pred, id_features_all = [], [], [], [], []
    ood_logits_all, ood_scores_all, ood_features_all = [], [], [] # OOD 로더가 None일 수 있음

    with torch.no_grad():
        for batch in tqdm(id_loader, desc="Evaluating ID for OSR", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'] # .numpy()는 나중에

            logits, features = model(input_ids, attention_mask, output_features=True)
            softmax_probs = F.softmax(logits / temperature, dim=1)
            max_probs, preds = softmax_probs.max(dim=1)

            id_logits_all.append(logits.cpu())
            id_scores_all.append(max_probs.cpu())
            id_labels_true.extend(labels.numpy()) # 여기서 numpy로 변환
            id_labels_pred.extend(preds.cpu().numpy())
            id_features_all.append(features.cpu())

    if ood_loader:
        with torch.no_grad():
            for batch in tqdm(ood_loader, desc="Evaluating OOD for OSR", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                # OOD 데이터는 레이블이 있지만, 여기서는 사용하지 않음

                logits, features = model(input_ids, attention_mask, output_features=True)
                softmax_probs = F.softmax(logits / temperature, dim=1)
                max_probs, _ = softmax_probs.max(dim=1) # OOD는 예측 클래스 불필요

                ood_logits_all.append(logits.cpu())
                ood_scores_all.append(max_probs.cpu())
                ood_features_all.append(features.cpu())

    id_scores = torch.cat(id_scores_all).numpy() if id_scores_all else np.array([])
    id_features = torch.cat(id_features_all).numpy() if id_features_all and len(id_features_all[0]) > 0 else np.array([])
    id_labels_true_np = np.array(id_labels_true)
    id_labels_pred_np = np.array(id_labels_pred)

    ood_scores = torch.cat(ood_scores_all).numpy() if ood_scores_all else np.array([])
    ood_features = torch.cat(ood_features_all).numpy() if ood_features_all and len(ood_features_all[0]) > 0 else np.array([])

    results = {
        "Closed_Set_Accuracy": 0.0, "F1_Macro": 0.0, "AUROC": 0.0,
        "FPR@TPR90": 1.0, "AUPR_In":0.0, "AUPR_Out":0.0, "DetectionAccuracy":0.0, "OSCR":0.0, # 추가 지표
        "Threshold_Used": 0.0 # 사용된 임계값 기록
    }
    all_data_dict = {
        "id_scores": id_scores, "ood_scores": ood_scores,
        "id_labels_true": id_labels_true_np, "id_labels_pred": id_labels_pred_np,
        "id_features": id_features, "ood_features": ood_features
    }

    if len(id_labels_true_np) == 0:
        print("Warning: No ID samples for OSR evaluation.")
        return results, all_data_dict if return_data else results

    closed_set_acc = accuracy_score(id_labels_true_np, id_labels_pred_np)
    f1_macro = f1_score(id_labels_true_np, id_labels_pred_np, average='macro', zero_division=0)
    results["Closed_Set_Accuracy"] = closed_set_acc
    results["F1_Macro"] = f1_macro

    if len(ood_scores) == 0: # OOD 데이터가 없으면 OSR 지표 계산 불가
        print("Warning: No OOD samples provided for OSR evaluation. AUROC, FPR etc. will be 0 or 1.")
        return results, all_data_dict if return_data else results

    # OSR 지표 계산
    y_true_osr = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)]) # ID=1, OOD=0
    y_scores_osr = np.concatenate([id_scores, ood_scores])

    # NaN 값 제거 (만약을 위해)
    valid_indices = ~np.isnan(y_scores_osr)
    if np.sum(valid_indices) < len(y_true_osr):
        print(f"Warning: Found {len(y_true_osr) - np.sum(valid_indices)} NaN scores. Removing them.")
        y_true_osr = y_true_osr[valid_indices]
        y_scores_osr = y_scores_osr[valid_indices]

    if len(np.unique(y_true_osr)) < 2: # ID 또는 OOD만 있는 경우
        print("Warning: Only one class type (ID or OOD) present after filtering. AUROC/FPR metrics might be uninformative.")
    else:
        results["AUROC"] = roc_auc_score(y_true_osr, y_scores_osr)
        fpr, tpr, thresholds_roc = roc_curve(y_true_osr, y_scores_osr)
        # FPR@TPR90: TPR이 0.90 이상인 지점 중 가장 낮은 FPR
        idx_tpr90 = np.where(tpr >= 0.90)[0]
        if len(idx_tpr90) > 0:
            results["FPR@TPR90"] = fpr[idx_tpr90[0]]
        else:
            print("Warning: TPR >= 0.90 not reached. FPR@TPR90 set to 1.0.")
            results["FPR@TPR90"] = 1.0

        # AUPR (Area Under Precision-Recall Curve)
        precision_in, recall_in, _ = precision_recall_curve(y_true_osr, y_scores_osr, pos_label=1) # ID가 Positive
        results["AUPR_In"] = auc(recall_in, precision_in)

        precision_out, recall_out, _ = precision_recall_curve(1 - y_true_osr, 1 - y_scores_osr, pos_label=1) # OOD가 Positive (점수는 1-score)
        results["AUPR_Out"] = auc(recall_out, precision_out)


    # 임계값 결정 (예: ID 점수의 특정 백분위수)
    # threshold_percentile은 ID 점수 중 몇 %를 OOD로 판단할지를 의미 (낮을수록 엄격)
    if len(id_scores) > 0 :
        # ID 점수가 낮은 쪽에서 %를 잘라 임계값으로 사용 (이 값보다 낮으면 OOD)
        chosen_threshold = np.percentile(id_scores, threshold_percentile)
    else: # ID 점수가 없으면 기본 임계값
        chosen_threshold = 0.5
    results["Threshold_Used"] = chosen_threshold

    # Detection Accuracy (임계값 기반)
    id_preds_binary = (id_scores >= chosen_threshold).astype(int)  # ID로 예측
    ood_preds_binary = (ood_scores < chosen_threshold).astype(int) # OOD로 예측

    correct_id_detection = np.sum(id_preds_binary)
    correct_ood_detection = np.sum(ood_preds_binary)
    total_samples = len(id_scores) + len(ood_scores)
    if total_samples > 0:
        results["DetectionAccuracy"] = (correct_id_detection + correct_ood_detection) / total_samples

    # OSCR (Open-Set Classification Rate)
    # CCR (Correct Classification Rate for knowns correctly rejected as knowns)
    # ID 샘플 중 임계값 이상 (Known으로 판단) & 실제 레이블과 예측 레이블 일치
    known_and_correctly_classified_as_known_mask = (id_scores >= chosen_threshold)
    if np.sum(known_and_correctly_classified_as_known_mask) > 0:
        ccr = accuracy_score(id_labels_true_np[known_and_correctly_classified_as_known_mask],
                               id_labels_pred_np[known_and_correctly_classified_as_known_mask])
    else:
        ccr = 0.0
    # OER (Open Set Error = FPR for unknowns)
    # OOD 샘플 중 임계값 이상 (Known으로 잘못 판단) 비율
    oer = np.sum(ood_scores >= chosen_threshold) / len(ood_scores) if len(ood_scores) > 0 else 0.0
    results["OSCR"] = ccr * (1.0 - oer)


    if return_data:
        return results, all_data_dict
    return results

# --- 시각화 함수들 (plot_confidence_histograms, plot_roc_curve, plot_osr_comparison, plot_confusion_matrix, plot_tsne) ---
# (이전 코드와 동일하게 사용 가능, 필요시 경로 인자 추가)
def plot_confidence_histograms(id_scores: np.ndarray, ood_scores: np.ndarray, title: str, save_path: Optional[str] = None):
    if not SNS_AVAILABLE: return
    plt.figure(figsize=(10, 6))
    id_scores_valid = id_scores[~np.isnan(id_scores)]
    ood_scores_valid = ood_scores[~np.isnan(ood_scores)]
    if len(id_scores_valid) > 0:
        sns.histplot(id_scores_valid, bins=50, alpha=0.5, label='In-Distribution', color='blue', stat='density', kde=True)
    if len(ood_scores_valid) > 0:
        sns.histplot(ood_scores_valid, bins=50, alpha=0.5, label='Out-of-Distribution', color='red', stat='density', kde=True)
    plt.xlabel('Confidence Score (Max Softmax Probability)')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path: plt.savefig(save_path); plt.close()
    else: plt.show(); plt.close()

def plot_roc_curve(id_scores: np.ndarray, ood_scores: np.ndarray, title: str, save_path: Optional[str] = None):
    if len(id_scores) == 0 or len(ood_scores) == 0:
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
    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Random guess line
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity/Recall)')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    if save_path: plt.savefig(save_path); plt.close()
    else: plt.show(); plt.close()

def plot_confusion_matrix_custom(cm: np.ndarray, class_names: List[str], title: str, save_path: Optional[str] = None):
    if not SNS_AVAILABLE:
        print("Seaborn not available, skipping confusion matrix plot.")
        return
    if cm.shape[0] != len(class_names) or cm.shape[1] != len(class_names):
        print(f"Warning: Mismatch CM shape {cm.shape} and class names {len(class_names)}. Skipping plot for '{title}'.")
        return

    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(max(8, int(len(class_names)*0.6)), max(6, int(len(class_names)*0.5))))
    try:
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True)
        plt.title(title, fontsize=14)
        plt.ylabel('Actual Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix plot saved to: {save_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for '{title}': {e}")
        plt.close()

def plot_tsne_custom(id_features: np.ndarray, ood_features: Optional[np.ndarray], title: str, save_path: Optional[str] = None, seed: int = 42, perplexity: int = 30, n_iter: int = 1000):
    if len(id_features) == 0 and (ood_features is None or len(ood_features) == 0):
        print(f"No features for t-SNE plot '{title}'.")
        return

    features_list = []
    labels_list = []
    legend_elements = []

    if len(id_features) > 0:
        features_list.append(id_features)
        labels_list.append(np.ones(len(id_features))) # ID = 1
        legend_elements.append({'label': 'In-Distribution (ID)', 'color': 'blue'})

    if ood_features is not None and len(ood_features) > 0:
        features_list.append(ood_features)
        labels_list.append(np.zeros(len(ood_features))) # OOD = 0
        legend_elements.append({'label': 'Out-of-Distribution (OOD)', 'color': 'red'})

    if not features_list: # 둘 다 없는 경우
        print(f"No valid features for t-SNE plot '{title}'.")
        return

    features_all = np.vstack(features_list)
    labels_all = np.concatenate(labels_list)

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
    for i, el in enumerate(legend_elements):
        indices = (labels_all == (1 if 'ID' in el['label'] else 0))
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                    c=el['color'], label=el['label'], alpha=0.5, s=15)

    plt.title(title, fontsize=14)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"t-SNE plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


# --- run_experiment 함수 (개별 실험 실행) ---
def run_single_experiment(args_dict: Dict[str, Any], id_dataset_name: str, current_oe_source_name: Optional[str] = None, current_oe_data_path: Optional[str] = None):
    """주어진 ID 데이터셋과 (선택적) OE 소스에 대해 단일 실험 실행"""
    experiment_tag = id_dataset_name
    if current_oe_source_name: # OE 실험인 경우
        experiment_tag += f"_OE_{current_oe_source_name}"
    else: # 표준 모델 실험인 경우
        experiment_tag += "_Standard"

    print(f"\n\n===== Starting Single Experiment: {experiment_tag} =====")

    # 결과 및 모델 저장 경로 설정 (실험별 하위 디렉토리)
    # current_oe_source_name이 파일 경로일 수 있으므로, 파일명만 추출하여 사용
    sanitized_oe_name = ""
    if current_oe_source_name:
        if os.path.isfile(current_oe_source_name): # 경로인 경우
             sanitized_oe_name = re.sub(r'[^\w\-.]+', '_', os.path.basename(current_oe_source_name))
        else: # snli, imdb 등 이름인 경우
             sanitized_oe_name = re.sub(r'[^\w\-]+', '_', current_oe_source_name)

    exp_result_subdir = id_dataset_name
    if sanitized_oe_name:
        exp_result_subdir = os.path.join(id_dataset_name, f"OE_{sanitized_oe_name}")
    else: # 표준 모델
        exp_result_subdir = os.path.join(id_dataset_name, "Standard")

    current_result_dir = os.path.join(args_dict['result_dir_base'], exp_result_subdir)
    current_model_dir = os.path.join(args_dict['model_dir_base'], exp_result_subdir)
    os.makedirs(current_result_dir, exist_ok=True)
    if args_dict['save_model_per_experiment']:
        os.makedirs(current_model_dir, exist_ok=True)

    print(f"Results for this experiment will be saved in: {current_result_dir}")
    if args_dict['save_model_per_experiment']:
        print(f"Models for this experiment will be saved in: {current_model_dir}")

    tokenizer = RobertaTokenizer.from_pretrained(args_dict['model_type'])

    # --- 1. 데이터 준비 ---
    print("\n--- Preparing Datasets for current experiment ---")
    train_dataset, id_test_dataset, num_classes, label_encoder, id_label2id, id_id2label = None, None, 0, None, {}, {}
    ood_test_dataset_for_eval = None # OSR 평가용 OOD 데이터
    known_class_names_list = []
    ood_dataset_eval_name_tag = "unknown_ood" # OOD 평가 대상 이름 (결과 키에 사용)

    if id_dataset_name == 'syslog':
        train_dataset, id_test_dataset, num_classes, label_encoder, id_label2id, id_id2label = prepare_syslog_data(
            tokenizer, args_dict['max_length'], args_dict['id_data_path'], args_dict['text_col'], args_dict['class_col'], args_dict['id_exclude_class'], args_dict['seed']
        )
        if train_dataset is None: # Syslog ID 데이터 준비 실패
            print(f"Critical error: Failed to prepare Syslog ID data. Skipping experiment: {experiment_tag}")
            return {}, {} # 빈 결과 반환
        ood_test_dataset_for_eval = prepare_syslog_ood_data(
            tokenizer, args_dict['max_length'], args_dict['ood_data_path'], args_dict['text_col'], args_dict['class_col'], args_dict['ood_target_class']
        )
        known_class_names_list = list(id_id2label.values()) if id_id2label else []
        ood_dataset_eval_name_tag = args_dict['ood_target_class']

    elif id_dataset_name == '20newsgroups':
        train_dataset, id_test_dataset, num_classes, label_encoder, id_label2id, id_id2label, ood_categories_20ng = prepare_20newsgroups_data(
            tokenizer, args_dict['max_length'], args_dict['num_20ng_id_classes'], args_dict['seed']
        )
        if train_dataset is None: # 20NG ID 데이터 준비 실패
            print(f"Critical error: Failed to prepare 20 Newsgroups ID data. Skipping experiment: {experiment_tag}")
            return {}, {}
        ood_test_dataset_for_eval = prepare_20newsgroups_ood_data(
            tokenizer, args_dict['max_length'], ood_categories_20ng, args_dict['seed']
        )
        known_class_names_list = [id_id2label.get(i, f"Class_{i}") for i in range(num_classes)] if id_id2label else []
        ood_dataset_eval_name_tag = "20ng_ood"
    else:
        raise ValueError(f"Unsupported ID dataset name: {id_dataset_name}")

    if num_classes == 0: # 클래스가 없는 경우 (데이터 로드 실패 등)
        print(f"No classes found for ID dataset '{id_dataset_name}'. Cannot proceed with experiment '{experiment_tag}'.")
        return {}, {}


    # 데이터 로더 생성
    num_workers = args_dict.get('num_dataloader_workers', 2) # 기본값 2
    train_loader = DataLoader(train_dataset, batch_size=args_dict['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    id_test_loader = DataLoader(id_test_dataset, batch_size=args_dict['batch_size'], num_workers=num_workers, pin_memory=True)
    ood_eval_loader = DataLoader(ood_test_dataset_for_eval, batch_size=args_dict['batch_size'], num_workers=num_workers, pin_memory=True) if ood_test_dataset_for_eval else None

    # --- 2. 모델 학습 및 평가 ---
    model = RoBERTaOOD(num_classes, args_dict['model_type']).to(device)
    # 모델 config에 label2id, id2label 설정 (Hugging Face 모델 저장/로드 시 필요)
    if id_label2id and id_id2label:
        model.roberta.config.label2id = id_label2id
        model.roberta.config.id2label = id_id2label

    model_filename_base = f"roberta_{experiment_tag}_{num_classes}cls_seed{args_dict['seed']}.pt"
    model_save_path = os.path.join(current_model_dir, model_filename_base)

    experiment_results = {}
    experiment_data_for_plots = {} # 이 실험의 tSNE, histogram용 데이터

    if args_dict['eval_only']:
        if os.path.exists(model_save_path):
            print(f"Loading pre-trained model for '{experiment_tag}' from {model_save_path}...")
            model.load_state_dict(torch.load(model_save_path, map_location=device))
        else:
            print(f"Error: Model path '{model_save_path}' not found for eval_only mode. Skipping evaluation for '{experiment_tag}'.")
            return {}, {} # 평가 불가
    else: # 학습 모드
        optimizer = AdamW(model.parameters(), lr=args_dict['learning_rate'])
        total_steps = len(train_loader) * args_dict['num_epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

        if current_oe_source_name: # OE 학습
            print(f"Preparing OE data for source: {current_oe_source_name}")
            oe_train_dataset_current = None
            if current_oe_data_path and os.path.exists(current_oe_data_path): # 제공된 경로 (syslog_masked 등)
                oe_train_dataset_current = prepare_syslog_masked_oe_data(
                    tokenizer, args_dict['max_length'], current_oe_data_path, args_dict['oe_masked_text_col']
                )
            elif current_oe_source_name in ['snli', 'imdb', 'wikitext']: # 외부 소스 이름
                oe_train_dataset_current = prepare_external_oe_data(
                    tokenizer, args_dict['max_length'], current_oe_source_name, data_dir=args_dict['data_dir'], cache_dir_hf=args_dict['cache_dir']
                )
            else:
                print(f"Warning: Could not determine how to load OE source '{current_oe_source_name}'. Skipping OE training for this source.")

            if oe_train_dataset_current:
                oe_train_loader_current = DataLoader(oe_train_dataset_current, batch_size=args_dict['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
                train_with_oe_uniform_loss(model, train_loader, oe_train_loader_current, optimizer, scheduler, device, args_dict['num_epochs'], args_dict['oe_lambda'], experiment_tag)
                del oe_train_dataset_current, oe_train_loader_current # 메모리 정리
                if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
            else:
                print(f"Failed to load OE data for '{current_oe_source_name}'. Training standard model instead for this run (equivalent to no OE).")
                train_standard(model, train_loader, optimizer, scheduler, device, args_dict['num_epochs'], experiment_tag)
        else: # 표준 학습
            train_standard(model, train_loader, optimizer, scheduler, device, args_dict['num_epochs'], experiment_tag)

        if args_dict['save_model_per_experiment']:
            torch.save(model.state_dict(), model_save_path)
            print(f"Model for '{experiment_tag}' saved to {model_save_path}")

    # 평가
    if ood_eval_loader is None: # OOD 평가 데이터가 없으면 OSR 지표 계산 불가
        print(f"Warning: No OOD evaluation data available for '{experiment_tag}'. Only closed-set metrics will be calculated.")

    # evaluate_osr 호출 시 ood_loader가 None일 수 있음을 명시
    results_osr, data_osr = evaluate_osr(model, id_test_loader, ood_eval_loader, device, args_dict['temperature'], threshold_percentile=args_dict.get('osr_threshold_percentile', 5.0), return_data=True)
    print(f"  OSSR Evaluation Results ({experiment_tag} vs {ood_dataset_eval_name_tag}): {results_osr}")

    # 결과 키 생성 시 ID 데이터셋 이름과 OE 소스 이름(또는 Standard) 및 OOD 평가 대상 이름을 포함
    # 예: "syslog_OE_oe_data_entropy_bottom25pct.csv+unknown"
    # 예: "20newsgroups_Standard+20ng_ood"
    metric_key_prefix = f"{id_dataset_name}_"
    if current_oe_source_name:
        # current_oe_source_name이 경로일 경우 파일명만 사용
        oe_name_for_key = os.path.basename(current_oe_source_name) if os.path.isfile(current_oe_source_name) else current_oe_source_name
        metric_key_prefix += f"OE_{oe_name_for_key}"
    else:
        metric_key_prefix += "Standard"
    full_metric_key = f"{metric_key_prefix}+{ood_dataset_eval_name_tag}"

    experiment_results[full_metric_key] = results_osr
    experiment_data_for_plots[full_metric_key] = data_osr # 플롯용 데이터 저장

    # 개별 실험 플롯 생성 (선택적)
    if not args_dict['no_plot_per_experiment']:
        plot_filename_prefix = re.sub(r'[^\w\-]+', '_', full_metric_key) # 파일명으로 사용 가능하게 정제

        if data_osr['id_scores'] is not None and data_osr['ood_scores'] is not None and len(data_osr['ood_scores']) > 0 : # OOD 점수가 있어야 의미 있음
            plot_confidence_histograms(data_osr['id_scores'], data_osr['ood_scores'],
                                       f'Confidence - {experiment_tag} vs {ood_dataset_eval_name_tag}',
                                       os.path.join(current_result_dir, f'{plot_filename_prefix}_hist.png'))
            plot_roc_curve(data_osr['id_scores'], data_osr['ood_scores'],
                           f'ROC - {experiment_tag} vs {ood_dataset_eval_name_tag}',
                           os.path.join(current_result_dir, f'{plot_filename_prefix}_roc.png'))
            # t-SNE 플롯 (ID 특징과 OOD 특징)
            plot_tsne_custom(data_osr['id_features'], data_osr['ood_features'],
                             f't-SNE - {experiment_tag} (ID vs OOD: {ood_dataset_eval_name_tag})',
                             os.path.join(current_result_dir, f'{plot_filename_prefix}_tsne.png'),
                             seed=args_dict['seed'])


        if data_osr['id_labels_true'] is not None and len(data_osr['id_labels_true']) > 0 and num_classes > 0:
            cm_std = confusion_matrix(data_osr['id_labels_true'], data_osr['id_labels_pred'], labels=np.arange(num_classes))
            plot_confusion_matrix_custom(cm_std, known_class_names_list,
                                   f'CM - {experiment_tag} (ID Test)',
                                   os.path.join(current_result_dir, f'{plot_filename_prefix}_cm.png'))

    del model, train_loader, id_test_loader, ood_eval_loader # 메모리 정리
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    return experiment_results, experiment_data_for_plots


# --- 메인 실행 로직 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RoBERTa OSR Comparison with Multiple OE Sources on Syslog or 20Newsgroups Data')
    # 주요 경로 및 설정 인자
    parser.add_argument('--id_dataset_type', type=str, default='syslog', choices=['syslog', '20newsgroups'], help='Which ID dataset to use.')
    parser.add_argument('--id_data_path', type=str, default=ID_SYSLOG_PATH, help='Path for Syslog ID data (used if id_dataset_type is syslog)')
    parser.add_argument('--ood_data_path', type=str, default=OOD_SYSLOG_UNKNOWN_PATH, help='Path for Syslog OOD data (used if id_dataset_type is syslog)')
    parser.add_argument('--result_dir_base', type=str, default=RESULT_DIR_BASE, help='Base directory to save all results')
    parser.add_argument('--model_dir_base', type=str, default=MODEL_DIR_BASE, help='Base directory to save all models')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Directory for external datasets like SNLI, IMDB')
    parser.add_argument('--cache_dir', type=str, default=CACHE_DIR, help='Cache directory for Hugging Face datasets')

    # OE 관련 인자
    parser.add_argument('--oe_config_file', type=str, default=None, help='Path to a text file listing OE dataset paths (masked syslog OE). One path per line.')
    parser.add_argument('--oe_masked_syslog_path', type=str, default=None, help='Path for a single masked syslog OE data (used if oe_config_file is not provided and oe_sources is not syslog_masked)')
    parser.add_argument('--oe_sources_external', nargs='+', default=[], choices=['snli', 'imdb', 'wikitext'], help='List of external OE sources to use (e.g., snli imdb). These run in addition to OE files from oe_config_file or oe_masked_syslog_path.')
    parser.add_argument('--oe_masked_text_col', type=str, default=OE_MASKED_TEXT_COLUMN, help='Text column name in masked OE CSV files.')

    # 모델 및 학습 하이퍼파라미터
    parser.add_argument('--model_type', type=str, default=MODEL_TYPE)
    parser.add_argument('--max_length', type=int, default=MAX_LENGTH)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--oe_lambda', type=float, default=OE_LAMBDA)
    parser.add_argument('--temperature', type=float, default=TEMPERATURE)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--osr_threshold_percentile', type=float, default=5.0, help='Percentile for ID scores to set OSR threshold (e.g., 5.0 means 5th percentile).')


    # Syslog 특정 인자
    parser.add_argument('--text_col', type=str, default=TEXT_COLUMN)
    parser.add_argument('--class_col', type=str, default=CLASS_COLUMN)
    parser.add_argument('--id_exclude_class', type=str, default=ID_SYSLOG_EXCLUDE_CLASS)
    parser.add_argument('--ood_target_class', type=str, default=OOD_SYSLOG_TARGET_CLASS)
    # 20Newsgroups 특정 인자
    parser.add_argument('--num_20ng_id_classes', type=int, default=NUM_20NG_ID_CLASSES)

    # 실행 제어 플래그
    parser.add_argument('--save_model_per_experiment', default=SAVE_MODEL_PER_EXPERIMENT, action=argparse.BooleanOptionalAction)
    parser.add_argument('--eval_only', default=EVAL_ONLY, action='store_true')
    parser.add_argument('--no_plot_per_experiment', default=NO_PLOT_PER_EXPERIMENT, action=argparse.BooleanOptionalAction)
    parser.add_argument('--skip_standard_model', default=SKIP_STANDARD_MODEL, action=argparse.BooleanOptionalAction)
    parser.add_argument('--num_dataloader_workers', type=int, default=2)


    args = parser.parse_args()
    args_dict = vars(args) # 인자를 딕셔너리로 변환하여 함수에 전달

    set_seed(args.seed)
    os.makedirs(args.result_dir_base, exist_ok=True)
    if args.save_model_per_experiment:
        os.makedirs(args.model_dir_base, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)


    all_experiments_results = {}
    all_experiments_plot_data = {} # 모든 실험의 플롯용 데이터 (필요 시)

    # 1. 표준 모델 실행 (OE 없음)
    if not args.skip_standard_model:
        print("\n--- Running Standard Model Experiment (No OE) ---")
        std_results, std_plot_data = run_single_experiment(args_dict, args.id_dataset_type, current_oe_source_name=None, current_oe_data_path=None)
        all_experiments_results.update(std_results)
        all_experiments_plot_data.update(std_plot_data)

    # 2. OE 데이터셋 기반 실험 실행
    oe_tasks_to_run = [] # (oe_source_name, oe_data_path) 튜플 리스트

    # 2a. oe_config_file에서 경로 읽기
    if args.oe_config_file:
        if os.path.exists(args.oe_config_file):
            print(f"\n--- Reading OE dataset paths from config file: {args.oe_config_file} ---")
            with open(args.oe_config_file, 'r') as f:
                for line in f:
                    path = line.strip()
                    if path and os.path.exists(path):
                        # OE 소스 이름은 파일명으로 (경로 구분자 제외)
                        source_name = os.path.basename(path)
                        oe_tasks_to_run.append({'name': source_name, 'path': path, 'type': 'masked_syslog'})
                        print(f"  Added OE task: {source_name} from path {path}")
                    elif path:
                        print(f"  Warning: Path '{path}' from config file not found. Skipping.")
        else:
            print(f"Warning: OE config file '{args.oe_config_file}' not found. Skipping OE datasets from config file.")

    # 2b. oe_masked_syslog_path (단일 경로) 추가 (config 파일이 우선)
    elif args.oe_masked_syslog_path: # config 파일이 없고, 단일 경로가 제공된 경우
        if os.path.exists(args.oe_masked_syslog_path):
            source_name = os.path.basename(args.oe_masked_syslog_path)
            oe_tasks_to_run.append({'name': source_name, 'path': args.oe_masked_syslog_path, 'type': 'masked_syslog'})
            print(f"Added single OE task: {source_name} from path {args.oe_masked_syslog_path}")
        else:
            print(f"Warning: Provided --oe_masked_syslog_path '{args.oe_masked_syslog_path}' not found. Skipping.")

    # 2c. 외부 OE 소스 추가
    if args.oe_sources_external:
        for ext_source_name in args.oe_sources_external:
            oe_tasks_to_run.append({'name': ext_source_name, 'path': None, 'type': 'external'}) # 경로는 없음
            print(f"Added external OE task: {ext_source_name}")


    if not oe_tasks_to_run and args.skip_standard_model : # 실행할 OE 작업도 없고 표준 모델도 스킵하면 종료
        print("No OE datasets specified and standard model is skipped. Nothing to run.")
    else:
        for task_info in oe_tasks_to_run:
            oe_source_name_for_run = task_info['name']
            oe_data_path_for_run = task_info['path'] # 외부 소스면 None

            print(f"\n--- Running OE Experiment with Source: {oe_source_name_for_run} ---")
            if oe_data_path_for_run:
                 print(f"  Using OE data path: {oe_data_path_for_run}")

            # run_single_experiment 호출 시 current_oe_source_name에 이름 또는 경로 전달
            # current_oe_data_path에는 실제 파일 경로 전달 (외부 소스면 None)
            oe_results, oe_plot_data = run_single_experiment(args_dict, args.id_dataset_type,
                                                             current_oe_source_name=oe_source_name_for_run,
                                                             current_oe_data_path=oe_data_path_for_run)
            all_experiments_results.update(oe_results)
            all_experiments_plot_data.update(oe_plot_data)


    # --- 최종 전체 결과 요약 및 저장 ---
    print("\n\n===== Final Overall Results Summary =====")
    if all_experiments_results:
        final_results_df = pd.DataFrame.from_dict(all_experiments_results, orient='index')
        final_results_df = final_results_df.sort_index() # 보기 좋게 정렬
        print("Overall Performance Metrics DataFrame:")
        print(final_results_df)

        # 전체 요약 파일명 (실행 시간 포함 가능)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename_base = f"osr_overall_summary_{args.id_dataset_type}_{timestamp_str}"

        # 저장 경로 (result_dir_base 최상위)
        overall_csv_path = os.path.join(args.result_dir_base, f'{summary_filename_base}.csv')
        overall_txt_path = os.path.join(args.result_dir_base, f'{summary_filename_base}.txt')
        overall_json_path = os.path.join(args.result_dir_base, f'{summary_filename_base}.json')

        try:
            final_results_df.to_csv(overall_csv_path, index=True)
            print(f"\nOverall results saved to CSV: {overall_csv_path}")
        except Exception as e: print(f"Error saving overall results to CSV: {e}")

        try:
            with open(overall_txt_path, 'w', encoding='utf-8') as f:
                f.write("--- Experiment Arguments ---\n")
                f.write(json.dumps(args_dict, indent=4, default=str)) # args_dict 저장
                f.write("\n\n--- Overall Metrics ---\n")
                f.write(final_results_df.to_string())
            print(f"Overall results and arguments saved to TXT: {overall_txt_path}")
        except Exception as e: print(f"Error saving overall results to TXT: {e}")

        # JSON에는 메트릭과 인자 모두 저장
        overall_summary_data_json = {
            'arguments': args_dict,
            'timestamp': timestamp_str,
            'results': all_experiments_results # 딕셔너리 형태 그대로 저장
        }
        try:
            with open(overall_json_path, 'w', encoding='utf-8') as f:
                json.dump(overall_summary_data_json, f, indent=4, default=str) # default=str 추가 (Numpy 객체 등 처리)
            print(f"Overall results and arguments saved to JSON: {overall_json_path}")
        except Exception as e: print(f"Error saving overall results to JSON: {e}")

        # OSR 비교 플롯 (모든 결과를 사용하여 생성)
        # plot_osr_comparison 함수가 필요 (이전 코드에서 가져오거나 새로 정의)
        # 이 함수는 all_experiments_results 딕셔너리를 받아 비교 플롯을 그림
        # 예시: plot_osr_comparison(all_experiments_results, os.path.join(args.result_dir_base, f'osr_overall_comparison_plot_{timestamp_str}.png'))
        # plot_osr_comparison 함수가 정의되어 있다고 가정하고 호출.
        # 이 함수는 metrics_dict (all_experiments_results)와 save_path를 인자로 받음.
        # plot_osr_comparison(all_experiments_results, os.path.join(args.result_dir_base, f'osr_comparison_plot_{args.id_dataset_type}_{timestamp_str}.png'))
        print(f"\nNote: Overall OSR comparison plot generation would require a 'plot_osr_comparison' function.")
        print(f"You can adapt such a function to take 'all_experiments_results' dictionary and save path.")

    else:
        print("No performance metrics were generated across all experiments.")

    print("\nAll specified experiments finished.")