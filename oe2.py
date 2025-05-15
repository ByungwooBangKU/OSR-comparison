"""
Unified OE (Out-of-Distribution) Extractor and OSR Experimentation
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Note: Renamed TextDataset from exam3.py to OSRTextDataset to avoid conflict
from torch.utils.data import DataLoader, Dataset as TorchDataset, Subset, random_split, ConcatDataset
from datasets import Dataset, DatasetDict, concatenate_datasets # Hugging Face datasets for oe.py part

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    RobertaTokenizer, # For OSR part
    RobertaForSequenceClassification, # For OSR part
    RobertaConfig, # For OSR part
    AdamW # For OSR part
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# 시각화 라이브러리
import matplotlib
matplotlib.use('Agg') # 비-GUI 백엔드 설정
import matplotlib.pyplot as plt
plt.ioff() # 대화형 모드 끄기
try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False
    print("Warning: Seaborn not available. Some plots might not be generated.")

# 텍스트 처리
import nltk
from nltk.tokenize import word_tokenize
import re
from collections import defaultdict
import json
import math
from typing import List, Dict, Tuple, Optional, Any, Union
from tqdm import tqdm # Use tqdm.auto for pl, tqdm for OSR part
import gc
from scipy.stats import entropy
import ast
from datetime import datetime
import random # For set_seed

# NLTK 데이터 다운로드
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# --- Configuration Class ---
class Config:
    """설정 클래스 - 모든 하이퍼파라미터와 경로 관리"""
    
    # === 기본 경로 및 파일 설정 (OE Extraction Part) ===
    ORIGINAL_DATA_PATH = 'log_all_critical.csv' # Source for ID data
    TEXT_COLUMN = 'text' # Text column in ORIGINAL_DATA_PATH and OE files
    CLASS_COLUMN = 'class' # Class column in ORIGINAL_DATA_PATH
    EXCLUDE_CLASS_FOR_TRAINING = "unknown" # This class will be OOD relative to the base classifier
    
    OUTPUT_DIR = 'unified_oe_osr_results' # Main output directory
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "base_classifier_model") # For model from Stage 1
    LOG_DIR = os.path.join(OUTPUT_DIR, "lightning_logs")
    CONFUSION_MATRIX_DIR = os.path.join(LOG_DIR, "confusion_matrices")
    VIS_DIR = os.path.join(OUTPUT_DIR, "oe_extraction_visualizations")
    OE_DATA_DIR = os.path.join(OUTPUT_DIR, "extracted_oe_datasets") # Where extracted OE CSVs are saved
    ATTENTION_DATA_DIR = os.path.join(OUTPUT_DIR, "attention_analysis")
    
    # === 모델 설정 (OE Extraction Part) ===
    MODEL_NAME = "roberta-base" # For base classifier in Stage 1
    MAX_LENGTH = 256 # For base classifier tokenization
    BATCH_SIZE = 64 # For base classifier training
    NUM_TRAIN_EPOCHS = 20 # For base classifier training
    LEARNING_RATE = 2e-5 # For base classifier training
    MIN_SAMPLES_PER_CLASS_FOR_TRAIN_VAL = 2
    
    # === 하드웨어 설정 (OE Extraction Part) ===
    ACCELERATOR = "auto"
    DEVICES = "auto"
    PRECISION = "16-mixed" if torch.cuda.is_available() else "32-true"
    NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1
    
    # === 학습 설정 (OE Extraction Part) ===
    LOG_EVERY_N_STEPS = 50
    GRADIENT_CLIP_VAL = 1.0
    USE_WEIGHTED_LOSS = True
    USE_LR_SCHEDULER = True
    RANDOM_STATE = 42 # Global seed
    
    # === 어텐션 설정 (OE Extraction Part) ===
    ATTENTION_TOP_PERCENT = 0.20
    MIN_TOP_WORDS = 2
    TOP_K_ATTENTION = 3 # For top_k_avg_attention metric
    ATTENTION_LAYER = -1  # 마지막 레이어
    
    # === OE 필터링 설정 (OE Extraction Part) ===
    METRIC_SETTINGS = {
        'attention_entropy': {'percentile': 80, 'mode': 'higher'},
        'top_k_avg_attention': {'percentile': 20, 'mode': 'lower'},
        'max_attention': {'percentile': 15, 'mode': 'lower'},
        'removed_avg_attention': {'percentile': 85, 'mode': 'higher'}
    }
    FILTERING_SEQUENCE = [
        ('removed_avg_attention', {'percentile': 85, 'mode': 'higher'}),
        ('attention_entropy', {'percentile': 75, 'mode': 'higher'})
    ]
    TEXT_COLUMN_IN_OE_FILES = 'masked_text_attention' # Column name for text in generated OE CSVs

    # === OSR Experiment Settings (Ported from exam3.py) ===
    OSR_EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, "osr_experiments") # Base for OSR results
    OSR_MODEL_DIR = os.path.join(OSR_EXPERIMENT_DIR, "models") # Where OSR models are saved
    OSR_RESULT_DIR = os.path.join(OSR_EXPERIMENT_DIR, "results") # Where OSR eval metrics/plots are saved
    
    OOD_SYSLOG_UNKNOWN_PATH_OSR = 'log_unknown.csv' # OOD data for evaluating OSR models
    # TEXT_COLUMN and CLASS_COLUMN are reused for OOD_SYSLOG_UNKNOWN_PATH_OSR
    OOD_TARGET_CLASS_OSR = "unknown" # The class to treat as OOD from OOD_SYSLOG_UNKNOWN_PATH_OSR

    OSR_MODEL_TYPE = 'roberta-base' # Model for OSR experiments
    OSR_MAX_LENGTH = 128
    OSR_BATCH_SIZE = 64
    OSR_NUM_EPOCHS = 30 # Epochs for each OSR experiment
    OSR_LEARNING_RATE = 2e-5
    OSR_OE_LAMBDA = 1.0
    OSR_TEMPERATURE = 1.0
    OSR_THRESHOLD_PERCENTILE = 5.0 # For OSR evaluation threshold
    OSR_NUM_DATALOADER_WORKERS = NUM_WORKERS # Reuse from OE part

    # Flags for OSR experiments
    OSR_SAVE_MODEL_PER_EXPERIMENT = True
    OSR_EVAL_ONLY = False
    OSR_NO_PLOT_PER_EXPERIMENT = False
    OSR_SKIP_STANDARD_MODEL = False # Whether to skip the OSR model trained without any OE data

    # Hugging Face cache for external datasets (if OSR part were to use them, not currently planned for this merge)
    DATA_DIR_EXTERNAL_HF = os.path.join(OUTPUT_DIR, 'data_external_hf') 
    CACHE_DIR_HF = os.path.join(DATA_DIR_EXTERNAL_HF, "hf_cache")

    # === 실행 단계 제어 ===
    STAGE_MODEL_TRAINING = True         # Train base classifier
    STAGE_ATTENTION_EXTRACTION = True   # Extract attention, create masked sentences
    STAGE_OE_EXTRACTION = True          # Extract OE datasets based on metrics
    STAGE_VISUALIZATION = True          # Visualize OE extraction metrics/tSNE
    STAGE_OSR_EXPERIMENTS = True        # Run OSR experiments using extracted OE data
        
    @classmethod
    def create_directories(cls):
        """필요한 디렉토리 생성"""
        dirs = [
            cls.OUTPUT_DIR, cls.MODEL_SAVE_DIR, cls.LOG_DIR,
            cls.CONFUSION_MATRIX_DIR, cls.VIS_DIR, cls.OE_DATA_DIR,
            cls.ATTENTION_DATA_DIR,
            # OSR Dirs
            cls.OSR_EXPERIMENT_DIR, cls.OSR_MODEL_DIR, cls.OSR_RESULT_DIR,
            cls.DATA_DIR_EXTERNAL_HF, cls.CACHE_DIR_HF
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def save_config(cls, filepath=None):
        """설정을 JSON 파일로 저장"""
        if filepath is None:
            filepath = os.path.join(cls.OUTPUT_DIR, 'config_unified.json')
        
        config_dict = {}
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                value = getattr(cls, attr)
                # Handle non-serializable types if any, though current ones are fine
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    config_dict[attr] = value
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str) # default=str for safety
        print(f"Configuration saved to {filepath}")

# === 헬퍼 함수들 (Global) ===
DEVICE_OSR = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    """시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True) # pl.seed_everything for PyTorch Lightning parts
    print(f"Seed set to {seed}")


def preprocess_text_for_roberta(text):
    """RoBERTa를 위한 텍스트 전처리"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_nltk(text):
    """NLTK를 사용한 토큰화"""
    if not text:
        return []
    try:
        return word_tokenize(text)
    except Exception:
        return text.split()


def create_masked_sentence(original_text, important_words):
    """중요 단어를 제거하여 마스킹된 문장 생성"""
    if not isinstance(original_text, str):
        return ""
    if not important_words:
        return original_text
    
    processed_text = preprocess_text_for_roberta(original_text)
    tokens = tokenize_nltk(processed_text)
    important_set_lower = {word.lower() for word in important_words}
    masked_tokens = [word for word in tokens if word.lower() not in important_set_lower]
    masked_sentence = ' '.join(masked_tokens)
    
    if not masked_sentence: # Handle cases where all words are removed
        return "__EMPTY_MASKED__" # Special token for empty masked sentences
    return masked_sentence


def safe_literal_eval(val):
    """문자열을 리스트로 안전하게 변환"""
    try:
        if isinstance(val, str) and val.strip().startswith('['):
            return ast.literal_eval(val)
        elif isinstance(val, list): # Already a list
            return val
        else:
            return [] # Default to empty list for other types or malformed strings
    except (ValueError, SyntaxError) as e:
        # print(f"Error parsing list string: '{str(val)[:50]}...' - {e}. Returning empty list.")
        return []


# === OSR Experiment Components (from exam3.py, adapted) ===

class OSRTextDataset(TorchDataset): # Renamed from TextDataset to avoid conflict
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

class RoBERTaOOD(nn.Module):
    def __init__(self, num_classes: int, model_name: str = 'roberta-base'):
        super(RoBERTaOOD, self).__init__()
        config = RobertaConfig.from_pretrained(model_name, num_labels=num_classes)
        config.output_hidden_states = True
        self.roberta = RobertaForSequenceClassification.from_pretrained(model_name, config=config)

    def forward(self, input_ids, attention_mask, output_features=False):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if output_features:
            features = outputs.hidden_states[-1][:, 0, :] # CLS token representation
            return logits, features
        else:
            return logits

def prepare_id_data_for_osr(datamodule: 'UnifiedDataModule', tokenizer, max_length: int) -> Tuple[Optional[OSRTextDataset], Optional[OSRTextDataset], int, Optional[LabelEncoder], Dict, Dict]:
    print(f"\n--- Preparing ID data for OSR from UnifiedDataModule ---")
    if datamodule.train_df_final is None or datamodule.val_df_final is None:
        print("Error: DataModule not set up or train/val split not performed.")
        return None, None, 0, None, {}, {}

    train_df = datamodule.train_df_final
    test_df = datamodule.val_df_final # Using validation set as test set for OSR ID

    # Recreate LabelEncoder based on the classes present in train_df and test_df combined
    # This ensures consistency if some classes were filtered out during datamodule setup
    # but are still relevant for OSR evaluation based on the original splits.
    # However, for simplicity and consistency with the base classifier, use datamodule's label info.
    
    num_classes = datamodule.num_labels
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array([datamodule.id2label[i] for i in range(num_classes)])
    
    id_label2id = datamodule.label2id
    id_id2label = datamodule.id2label

    print(f"  - Using {num_classes} known classes from UnifiedDataModule.")
    print(f"  - Label to ID mapping: {id_label2id}")

    # Ensure 'label' column is integer
    train_df['label'] = train_df['label'].astype(int)
    test_df['label'] = test_df['label'].astype(int)

    train_dataset = OSRTextDataset(train_df[Config.TEXT_COLUMN].tolist(), train_df['label'].tolist(), tokenizer, max_length)
    id_test_dataset = OSRTextDataset(test_df[Config.TEXT_COLUMN].tolist(), test_df['label'].tolist(), tokenizer, max_length)
    
    print(f"  - OSR Train: {len(train_dataset)}, OSR ID Test: {len(id_test_dataset)}")
    return train_dataset, id_test_dataset, num_classes, label_encoder, id_label2id, id_id2label


def prepare_syslog_ood_data_for_osr(tokenizer, max_length: int, ood_data_path: str, text_col: str, class_col: str, ood_target_class: str) -> Optional[OSRTextDataset]:
    print(f"\n--- Preparing Syslog OOD data (class: '{ood_target_class}') from: {ood_data_path} for OSR ---")
    if not os.path.exists(ood_data_path):
        print(f"Error: OOD data path not found: {ood_data_path}")
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
        ood_labels = np.full(len(texts), -1, dtype=int).tolist() # OOD label is -1
        ood_dataset = OSRTextDataset(texts, ood_labels, tokenizer, max_length)
        print(f"  - Loaded {len(ood_dataset)} samples for OOD testing (class: '{ood_target_class}').")
        return ood_dataset
    except Exception as e:
        print(f"Error preparing Syslog OOD data from '{ood_data_path}': {e}")
        return None

def prepare_generated_oe_data_for_osr(tokenizer, max_length: int, oe_data_path: str, oe_text_col: str) -> Optional[OSRTextDataset]:
    print(f"\n--- Preparing Generated OE data from: {oe_data_path} for OSR ---")
    if not os.path.exists(oe_data_path):
        print(f"Error: OE data path not found: {oe_data_path}")
        return None
    try:
        df = pd.read_csv(oe_data_path)
        if oe_text_col not in df.columns:
            # Try falling back to a default if the specific one isn't there
            # This can happen if an older OE file is used or config changes
            fallback_cols = ['masked_text_attention', 'text', Config.TEXT_COLUMN]
            found_col = False
            for col_attempt in fallback_cols:
                if col_attempt in df.columns:
                    oe_text_col_actual = col_attempt
                    print(f"  Warning: Specified OE text column '{oe_text_col}' not found. Using fallback '{oe_text_col_actual}'.")
                    found_col = True
                    break
            if not found_col:
                 raise ValueError(f"OE Data CSV '{oe_data_path}' must contain a valid text column (tried '{oe_text_col}' and fallbacks).")
        else:
            oe_text_col_actual = oe_text_col

        df = df.dropna(subset=[oe_text_col_actual])
        texts = df[oe_text_col_actual].astype(str).tolist()
        if not texts:
            print(f"Warning: No valid OE texts found in '{oe_text_col_actual}' from '{oe_data_path}'.")
            return None
        oe_labels = np.full(len(texts), -1, dtype=int).tolist() # OE label is -1
        oe_dataset = OSRTextDataset(texts, oe_labels, tokenizer, max_length)
        print(f"  - Loaded {len(oe_dataset)} samples for OE training from {oe_data_path} (using column '{oe_text_col_actual}').")
        return oe_dataset
    except Exception as e:
        print(f"Error preparing Generated OE data from '{oe_data_path}': {e}")
        return None

def train_standard_osr(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, scheduler, device: torch.device, num_epochs: int, current_experiment_name: str):
    model.train()
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Starting OSR standard training for '{current_experiment_name}'... AMP enabled: {use_amp}")

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Std OSR Epoch {epoch+1}/{num_epochs} ({current_experiment_name})", leave=False)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=str(device).split(':')[0], enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.3f}"})
        if scheduler: scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Std OSR Epoch {epoch+1}/{num_epochs} ({current_experiment_name}), Avg Loss: {avg_loss:.4f}")

def train_with_oe_uniform_loss_osr(model: nn.Module, train_loader: DataLoader, oe_loader: DataLoader, optimizer: optim.Optimizer, scheduler, device: torch.device, num_epochs: int, oe_lambda: float, current_experiment_name: str):
    model.train()
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Starting OSR OE (Uniform CE Loss) training for '{current_experiment_name}'... AMP enabled: {use_amp}")

    for epoch in range(num_epochs):
        oe_iter = iter(oe_loader)
        total_loss, total_id_loss, total_oe_loss = 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"OE OSR Epoch {epoch+1}/{num_epochs} ({current_experiment_name})", leave=False)

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

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=str(device).split(':')[0], enabled=use_amp):
                id_logits = model(input_ids, attention_mask)
                id_loss = F.cross_entropy(id_logits, labels)

                oe_logits = model(oe_input_ids, oe_attention_mask)
                num_classes = oe_logits.size(1)
                log_softmax_oe = F.log_softmax(oe_logits, dim=1)
                uniform_target_probs = torch.full_like(oe_logits, 1.0 / num_classes)
                oe_loss = F.kl_div(log_softmax_oe, uniform_target_probs, reduction='batchmean', log_target=False)
                
                total_batch_loss = id_loss + oe_lambda * oe_loss
            
            scaler.scale(total_batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += total_batch_loss.item()
            total_id_loss += id_loss.item()
            total_oe_loss += oe_loss.item()
            progress_bar.set_postfix({'Total': f"{total_batch_loss.item():.3f}", 'ID': f"{id_loss.item():.3f}", 'OE': f"{oe_loss.item():.3f}"})
        
        if scheduler: scheduler.step()
        avg_loss = total_loss / len(train_loader)
        avg_id_loss = total_id_loss / len(train_loader)
        avg_oe_loss = total_oe_loss / len(train_loader)
        print(f"OE OSR Epoch {epoch+1}/{num_epochs} ({current_experiment_name}), Avg Loss: {avg_loss:.4f} (ID: {avg_id_loss:.4f}, OE: {avg_oe_loss:.4f})")

def evaluate_osr(model: nn.Module, id_loader: DataLoader, ood_loader: Optional[DataLoader], device: torch.device, temperature: float = 1.0, threshold_percentile: float = 5.0, return_data: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, np.ndarray]]]:
    model.eval()
    id_logits_all, id_scores_all, id_labels_true, id_labels_pred, id_features_all = [], [], [], [], []
    ood_logits_all, ood_scores_all, ood_features_all = [], [], []

    with torch.no_grad():
        for batch in tqdm(id_loader, desc="Evaluating ID for OSR", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']

            logits, features = model(input_ids, attention_mask, output_features=True)
            softmax_probs = F.softmax(logits / temperature, dim=1)
            max_probs, preds = softmax_probs.max(dim=1)

            id_logits_all.append(logits.cpu())
            id_scores_all.append(max_probs.cpu())
            id_labels_true.extend(labels.numpy())
            id_labels_pred.extend(preds.cpu().numpy())
            id_features_all.append(features.cpu())

    if ood_loader:
        with torch.no_grad():
            for batch in tqdm(ood_loader, desc="Evaluating OOD for OSR", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                logits, features = model(input_ids, attention_mask, output_features=True)
                softmax_probs = F.softmax(logits / temperature, dim=1)
                max_probs, _ = softmax_probs.max(dim=1)
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
        "FPR@TPR90": 1.0, "AUPR_In":0.0, "AUPR_Out":0.0, "DetectionAccuracy":0.0, "OSCR":0.0,
        "Threshold_Used": 0.0
    }
    all_data_dict = {
        "id_scores": id_scores, "ood_scores": ood_scores,
        "id_labels_true": id_labels_true_np, "id_labels_pred": id_labels_pred_np,
        "id_features": id_features, "ood_features": ood_features
    }

    if len(id_labels_true_np) == 0:
        print("Warning: No ID samples for OSR evaluation.")
        return results, all_data_dict if return_data else results

    results["Closed_Set_Accuracy"] = accuracy_score(id_labels_true_np, id_labels_pred_np)
    results["F1_Macro"] = f1_score(id_labels_true_np, id_labels_pred_np, average='macro', zero_division=0)

    if len(ood_scores) == 0:
        print("Warning: No OOD samples for OSR evaluation. AUROC, FPR etc. will be 0 or 1.")
        return results, all_data_dict if return_data else results

    y_true_osr = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_scores_osr = np.concatenate([id_scores, ood_scores])
    
    valid_indices = ~np.isnan(y_scores_osr)
    y_true_osr, y_scores_osr = y_true_osr[valid_indices], y_scores_osr[valid_indices]

    if len(np.unique(y_true_osr)) < 2:
        print("Warning: Only one class type (ID or OOD) present after filtering.")
    else:
        results["AUROC"] = roc_auc_score(y_true_osr, y_scores_osr)
        fpr, tpr, thresholds_roc = roc_curve(y_true_osr, y_scores_osr)
        idx_tpr90 = np.where(tpr >= 0.90)[0]
        results["FPR@TPR90"] = fpr[idx_tpr90[0]] if len(idx_tpr90) > 0 else 1.0
        if len(idx_tpr90) == 0: print("Warning: TPR >= 0.90 not reached for FPR@TPR90.")

        precision_in, recall_in, _ = precision_recall_curve(y_true_osr, y_scores_osr, pos_label=1)
        results["AUPR_In"] = auc(recall_in, precision_in)
        precision_out, recall_out, _ = precision_recall_curve(1 - y_true_osr, 1 - y_scores_osr, pos_label=1)
        results["AUPR_Out"] = auc(recall_out, precision_out)

    chosen_threshold = np.percentile(id_scores, threshold_percentile) if len(id_scores) > 0 else 0.5
    results["Threshold_Used"] = chosen_threshold

    id_preds_binary = (id_scores >= chosen_threshold).astype(int)
    ood_preds_binary = (ood_scores < chosen_threshold).astype(int)
    total_samples = len(id_scores) + len(ood_scores)
    if total_samples > 0:
        results["DetectionAccuracy"] = (np.sum(id_preds_binary) + np.sum(ood_preds_binary)) / total_samples
    
    known_mask = (id_scores >= chosen_threshold)
    ccr = accuracy_score(id_labels_true_np[known_mask], id_labels_pred_np[known_mask]) if np.sum(known_mask) > 0 else 0.0
    oer = np.sum(ood_scores >= chosen_threshold) / len(ood_scores) if len(ood_scores) > 0 else 0.0
    results["OSCR"] = ccr * (1.0 - oer)

    return (results, all_data_dict) if return_data else results

# --- OSR Plotting Functions (from exam3.py) ---
def plot_confidence_histograms_osr(id_scores: np.ndarray, ood_scores: np.ndarray, title: str, save_path: Optional[str] = None):
    if not SNS_AVAILABLE: return
    plt.figure(figsize=(10, 6))
    if len(id_scores[~np.isnan(id_scores)]) > 0:
        sns.histplot(id_scores[~np.isnan(id_scores)], bins=50, alpha=0.5, label='In-Distribution', color='blue', stat='density', kde=True)
    if len(ood_scores[~np.isnan(ood_scores)]) > 0:
        sns.histplot(ood_scores[~np.isnan(ood_scores)], bins=50, alpha=0.5, label='Out-of-Distribution', color='red', stat='density', kde=True)
    plt.xlabel('Confidence Score (Max Softmax Probability)')
    plt.ylabel('Density'); plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    if save_path: plt.savefig(save_path); plt.close()
    else: plt.show(); plt.close()

def plot_roc_curve_osr(id_scores: np.ndarray, ood_scores: np.ndarray, title: str, save_path: Optional[str] = None):
    if len(id_scores) == 0 or len(ood_scores) == 0: return
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_scores_concat = np.concatenate([id_scores, ood_scores])
    valid_indices = ~np.isnan(y_scores_concat)
    y_true, y_scores_concat = y_true[valid_indices], y_scores_concat[valid_indices]
    if len(np.unique(y_true)) < 2: return

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

def plot_confusion_matrix_osr(cm: np.ndarray, class_names: List[str], title: str, save_path: Optional[str] = None):
    if not SNS_AVAILABLE or cm.shape[0] != len(class_names) or cm.shape[1] != len(class_names):
        if not SNS_AVAILABLE: print("Seaborn not available, skipping CM plot.")
        else: print(f"CM shape/class_names mismatch for '{title}'. Skipping.")
        return
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(max(8, int(len(class_names)*0.6)), max(6, int(len(class_names)*0.5))))
    try:
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True)
        plt.title(title, fontsize=14); plt.ylabel('Actual', fontsize=12); plt.xlabel('Predicted', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10); plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        if save_path: plt.savefig(save_path); print(f"CM plot saved: {save_path}")
        else: plt.show()
    except Exception as e: print(f"Error plotting CM for '{title}': {e}")
    finally: plt.close()

def plot_tsne_osr(id_features: np.ndarray, ood_features: Optional[np.ndarray], title: str, save_path: Optional[str] = None, seed: int = 42, perplexity: int = 30, n_iter: int = 1000):
    if len(id_features) == 0 and (ood_features is None or len(ood_features) == 0): return
    features_list, labels_list, legend_elements = [], [], []
    if len(id_features) > 0:
        features_list.append(id_features); labels_list.append(np.ones(len(id_features)))
        legend_elements.append({'label': 'In-Distribution (ID)', 'color': 'blue'})
    if ood_features is not None and len(ood_features) > 0:
        features_list.append(ood_features); labels_list.append(np.zeros(len(ood_features)))
        legend_elements.append({'label': 'Out-of-Distribution (OOD)', 'color': 'red'})
    if not features_list: return

    features_all = np.vstack(features_list); labels_all = np.concatenate(labels_list)
    print(f"Running t-SNE for OSR on {features_all.shape[0]} samples for '{title}'...")
    try:
        eff_perplexity = min(perplexity, features_all.shape[0] - 1)
        if eff_perplexity <=1: 
            print(f"Warning: t-SNE perplexity too low ({eff_perplexity}). Skipping plot.")
            return
        tsne = TSNE(n_components=2, random_state=seed, perplexity=eff_perplexity, n_iter=n_iter, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(features_all)
    except Exception as e: print(f"Error in t-SNE for '{title}': {e}. Skipping."); return
    
    plt.figure(figsize=(10, 8))
    for el in legend_elements:
        indices = (labels_all == (1 if 'ID' in el['label'] else 0))
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], c=el['color'], label=el['label'], alpha=0.5, s=15)
    plt.title(title, fontsize=14); plt.xlabel("t-SNE Dim 1"); plt.ylabel("t-SNE Dim 2")
    plt.legend(fontsize=10); plt.grid(alpha=0.3); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300); print(f"t-SNE plot saved: {save_path}")
    else: plt.show()
    plt.close()

# === PyTorch Lightning 컴포넌트 (OE Extraction Part) ===
class UnifiedDataModule(pl.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['config']) # Avoid saving full config object
        
        print(f"Initializing tokenizer for base classifier: {config.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        self.df_full = None
        self.df_known_for_train_val = None
        self.train_df_final = None
        self.val_df_final = None # This will serve as ID test set for OSR
        self.label2id = None
        self.id2label = None
        self.num_labels = None
        self.tokenized_train_val_datasets = None
        self.class_weights = None
            
    def prepare_data(self): pass # Downloads etc.
    
    def setup(self, stage=None):
        if self.df_full is not None: return # Already set up

        print(f"Loading data from {self.config.ORIGINAL_DATA_PATH}")
        self.df_full = pd.read_csv(self.config.ORIGINAL_DATA_PATH)
        
        required_cols = [self.config.TEXT_COLUMN, self.config.CLASS_COLUMN]
        if not all(col in self.df_full.columns for col in required_cols):
            raise ValueError(f"Missing columns in {self.config.ORIGINAL_DATA_PATH}: {required_cols}")
        
        self.df_full = self.df_full.dropna(subset=[self.config.TEXT_COLUMN, self.config.CLASS_COLUMN])
        self.df_full[self.config.CLASS_COLUMN] = self.df_full[self.config.CLASS_COLUMN].astype(str).str.lower()
        self.df_full[self.config.TEXT_COLUMN] = self.df_full[self.config.TEXT_COLUMN].astype(str)


        exclude_class_lower = self.config.EXCLUDE_CLASS_FOR_TRAINING.lower()
        df_known = self.df_full[self.df_full[self.config.CLASS_COLUMN] != exclude_class_lower].copy()
        print(f"Data size after excluding '{self.config.EXCLUDE_CLASS_FOR_TRAINING}': {len(df_known)}")
        
        known_classes_str = sorted(df_known[self.config.CLASS_COLUMN].unique())
        self.label2id = {label: i for i, label in enumerate(known_classes_str)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.num_labels = len(known_classes_str)

        if self.num_labels == 0:
            raise ValueError("No known classes found after excluding. Cannot proceed.")
        print(f"Label mapping complete: {self.num_labels} known classes for base classifier.")
        print(f"Label to ID mapping: {self.label2id}")
        
        df_known['label'] = df_known[self.config.CLASS_COLUMN].map(self.label2id)
        df_known = df_known.dropna(subset=['label']) # Remove rows where mapping failed (should not happen)
        df_known['label'] = df_known['label'].astype(int)
        
        print(f"Filtering classes with minimum {self.config.MIN_SAMPLES_PER_CLASS_FOR_TRAIN_VAL} samples...")
        label_counts = df_known['label'].value_counts()
        valid_labels = label_counts[label_counts >= self.config.MIN_SAMPLES_PER_CLASS_FOR_TRAIN_VAL].index
        self.df_known_for_train_val = df_known[df_known['label'].isin(valid_labels)].copy()
        
        # Update num_labels, label2id, id2label if classes were filtered
        if len(valid_labels) < self.num_labels:
            print(f"  Classes filtered due to insufficient samples. Original: {self.num_labels}, Final: {len(valid_labels)}")
            final_classes_str = sorted(self.df_known_for_train_val[self.config.CLASS_COLUMN].unique())
            self.label2id = {label: i for i, label in enumerate(final_classes_str)}
            self.id2label = {i: label for label, i in self.label2id.items()}
            self.num_labels = len(final_classes_str)
            self.df_known_for_train_val['label'] = self.df_known_for_train_val[self.config.CLASS_COLUMN].map(self.label2id)
            self.df_known_for_train_val['label'] = self.df_known_for_train_val['label'].astype(int)
            print(f"  Updated label mapping: {self.num_labels} classes. {self.label2id}")

        if len(self.df_known_for_train_val) == 0:
            raise ValueError("No data available after filtering for min samples per class.")
        
        print(f"Data for base classifier after filtering: {len(self.df_known_for_train_val)} samples")
        print("\n--- Class distribution in base classifier train/val data ---")
        print(self.df_known_for_train_val['label'].map(self.id2label).value_counts())
        
        if self.config.USE_WEIGHTED_LOSS: self._compute_class_weights()
        self._split_train_val()
        self._tokenize_datasets()

    def _compute_class_weights(self):
        labels_for_weights = self.df_known_for_train_val['label'].values
        unique_labels = np.unique(labels_for_weights)
        
        try:
            class_weights_array = compute_class_weight('balanced', classes=unique_labels, y=labels_for_weights)
            # Ensure class_weights tensor matches num_labels, accounting for filtered classes
            self.class_weights = torch.ones(self.num_labels) 
            temp_label2id_for_weights = {self.id2label[lbl_idx]: i for i, lbl_idx in enumerate(unique_labels)}

            for i, class_idx_in_unique_labels in enumerate(unique_labels): # class_idx_in_unique_labels is the actual label ID
                if class_idx_in_unique_labels < self.num_labels: # Ensure it's a valid final label ID
                     self.class_weights[class_idx_in_unique_labels] = class_weights_array[i]
            print(f"Computed class weights for base classifier: {self.class_weights}")
        except ValueError as e:
            print(f"Error computing class weights: {e}. Using uniform weights.")
            self.config.USE_WEIGHTED_LOSS = False; self.class_weights = None
            
    def _split_train_val(self):
        print("Splitting train/validation data for base classifier...")
        # Check if stratification is possible
        min_class_count = self.df_known_for_train_val['label'].value_counts().min()
        stratify_col = self.df_known_for_train_val['label'] if min_class_count > 1 else None
        if stratify_col is None:
            print("Warning: Not enough samples in some classes for stratified split. Using random split.")

        try:
            self.train_df_final, self.val_df_final = train_test_split(
                self.df_known_for_train_val, test_size=0.2,
                random_state=self.config.RANDOM_STATE, stratify=stratify_col
            )
        except ValueError: # Fallback if stratification fails for any other reason
            print("Warning: Stratified split failed unexpectedly. Using random split.")
            self.train_df_final, self.val_df_final = train_test_split(
                self.df_known_for_train_val, test_size=0.2, random_state=self.config.RANDOM_STATE
            )
        print(f"Base classifier split - Train: {len(self.train_df_final)}, Val (used as ID Test for OSR): {len(self.val_df_final)}")

    def _tokenize_datasets(self):
        raw_datasets = DatasetDict({
            'train': Dataset.from_pandas(self.train_df_final),
            'validation': Dataset.from_pandas(self.val_df_final)
        })
        
        def tokenize_func(examples):
            return self.tokenizer(
                [preprocess_text_for_roberta(text) for text in examples[self.config.TEXT_COLUMN]],
                truncation=True, padding=False, max_length=self.config.MAX_LENGTH
            )
        
        print("Tokenizing datasets for base classifier...")
        self.tokenized_train_val_datasets = raw_datasets.map(
            tokenize_func, batched=True,
            num_proc=max(1, self.config.NUM_WORKERS // 2),
            remove_columns=[col for col in raw_datasets['train'].column_names if col not in ['label', 'input_ids', 'attention_mask']]
        )
        self.tokenized_train_val_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        print("Tokenization for base classifier complete.")

    def train_dataloader(self):
        return DataLoader(
            self.tokenized_train_val_datasets['train'], batch_size=self.config.BATCH_SIZE,
            collate_fn=self.data_collator, num_workers=self.config.NUM_WORKERS,
            shuffle=True, pin_memory=True, persistent_workers=self.config.NUM_WORKERS > 0
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.tokenized_train_val_datasets['validation'], batch_size=self.config.BATCH_SIZE,
            collate_fn=self.data_collator, num_workers=self.config.NUM_WORKERS,
            pin_memory=True, persistent_workers=self.config.NUM_WORKERS > 0
        )

    def get_full_dataframe(self):
        if self.df_full is None: self.setup()
        return self.df_full

class UnifiedModel(pl.LightningModule): # For base classifier
    def __init__(self, config: Config, num_labels: int, label2id: dict, id2label: dict, class_weights=None):
        super().__init__()
        self.config_params = config # Store config directly
        self.save_hyperparameters(ignore=['config_params', 'class_weights']) # Avoid saving full config object

        print(f"Initializing base classifier model: {self.config_params.MODEL_NAME} for {num_labels} classes")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config_params.MODEL_NAME, num_labels=num_labels, label2id=label2id, id2label=id2label,
            ignore_mismatched_sizes=True, output_attentions=True, output_hidden_states=True
        )
        
        if self.config_params.USE_WEIGHTED_LOSS and class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights) # class_weights should be a tensor
            print("Base classifier using weighted CrossEntropyLoss")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            print("Base classifier using standard CrossEntropyLoss")
        
        metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=num_labels),
            'f1_weighted': torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='weighted'),
            'f1_macro': torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='macro')
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.val_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_labels)

    def setup(self, stage=None):
         if self.config_params.USE_WEIGHTED_LOSS and hasattr(self.loss_fn, 'weight') and self.loss_fn.weight is not None:
            self.loss_fn.weight = self.loss_fn.weight.to(self.device) # Ensure weights are on correct device
            print(f"Moved base classifier class weights to {self.device}")

    def forward(self, batch, output_features=False, output_attentions=False):
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        if input_ids is None or attention_mask is None:
            raise ValueError("Batch missing 'input_ids' or 'attention_mask'")
        
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=output_features, output_attentions=output_attentions
        )

    def _common_step(self, batch, batch_idx):
        if 'label' in batch: batch['labels'] = batch.pop('label')
        outputs = self.model(**batch)
        loss = outputs.loss # Use HF model's internal loss if labels are provided
        # If custom loss_fn is preferred and labels are not passed to model directly:
        # logits = outputs.logits
        # loss = self.loss_fn(logits, batch['labels'])
        preds = torch.argmax(outputs.logits, dim=1)
        return loss, preds, batch['labels']

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        self.log_dict(self.train_metrics(preds, labels), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        self.val_metrics.update(preds, labels)
        self.val_cm.update(preds, labels)
        self.log_dict(self.val_metrics, on_epoch=True, prog_bar=True) # Log collected metrics at epoch end
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        try:
            val_cm_computed = self.val_cm.compute()
            class_names = list(self.hparams.id2label.values()) # Access from hparams
            cm_df = pd.DataFrame(val_cm_computed.cpu().numpy(), index=class_names, columns=class_names)
            print("\nBase Classifier Validation Confusion Matrix (Epoch {}):".format(self.current_epoch))
            print(cm_df)
            cm_filename = os.path.join(self.config_params.CONFUSION_MATRIX_DIR, f"base_clf_val_cm_epoch_{self.current_epoch}.csv")
            cm_df.to_csv(cm_filename)
        except Exception as e: print(f"Error in base classifier validation CM: {e}")
        finally: self.val_cm.reset() # Reset for next epoch

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config_params.LEARNING_RATE)
        if self.config_params.USE_LR_SCHEDULER:
            if self.trainer and hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches > 0:
                num_training_steps = self.trainer.estimated_stepping_batches
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
            else: # Fallback if trainer info not available (e.g. not in Trainer context)
                print("Warning: Could not estimate training steps for base classifier scheduler. Using optimizer only.")
                return optimizer
        return optimizer

# === 어텐션 분석 클래스 (OE Extraction Part) ===
class AttentionAnalyzer:
    def __init__(self, config: Config, model_pl: UnifiedModel, tokenizer, device):
        self.config = config
        self.model_pl = model_pl.to(device) # Ensure model is on the correct device
        self.model_pl.eval()
        self.model_pl.freeze()
        self.tokenizer = tokenizer
        self.device = device # Store device

    @torch.no_grad()
    def get_word_attention_scores(self, texts: List[str], layer_idx: int = -1) -> List[Dict[str, float]]:
        batch_size = self.config.BATCH_SIZE # Use training batch size for consistency
        all_word_scores = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing word attention scores", leave=False):
            batch_texts = texts[i:i+batch_size]
            batch_scores = self._process_attention_batch(batch_texts, layer_idx)
            all_word_scores.extend(batch_scores)
        return all_word_scores

    def _process_attention_batch(self, batch_texts: List[str], layer_idx: int) -> List[Dict[str, float]]:
        if not batch_texts: return []
        processed_texts = [preprocess_text_for_roberta(text) for text in batch_texts]
        inputs = self.tokenizer(
            processed_texts, return_tensors='pt', truncation=True,
            max_length=self.config.MAX_LENGTH, padding=True, return_offsets_mapping=True
        )
        offset_mappings = inputs.pop('offset_mapping').cpu().numpy()
        input_ids_batch = inputs['input_ids'].cpu().numpy()
        
        # Move inputs to the same device as the model
        inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model_pl.model(**inputs_on_device, output_attentions=True)
            attentions_batch = outputs.attentions[layer_idx].cpu().numpy()

        batch_word_scores = [
            self._extract_word_scores_from_attention(
                attentions_batch[i], input_ids_batch[i], offset_mappings[i], processed_texts[i]
            ) for i in range(len(batch_texts))
        ]
        del inputs, inputs_on_device, outputs, attentions_batch; gc.collect()
        return batch_word_scores

    def _extract_word_scores_from_attention(self, attention_sample, input_ids, offset_mapping, original_text):
        attention_heads_mean = np.mean(attention_sample, axis=0)
        cls_attentions = attention_heads_mean[0, :]
        
        word_scores = defaultdict(list)
        current_word_indices = []
        last_word_end_offset = 0
        
        for j, (token_id, offset) in enumerate(zip(input_ids, offset_mapping)):
            if offset[0] == offset[1] or token_id in self.tokenizer.all_special_ids: continue
            
            is_continuation = (j > 0 and offset[0] == last_word_end_offset and token_id != self.tokenizer.unk_token_id) # Added unk_token check
            
            if not is_continuation and current_word_indices:
                start_offset = offset_mapping[current_word_indices[0]][0]
                end_offset = offset_mapping[current_word_indices[-1]][1]
                word = original_text[start_offset:end_offset]
                avg_score = np.mean(cls_attentions[current_word_indices])
                if word.strip(): word_scores[word.strip()].append(avg_score)
                current_word_indices = []
            
            current_word_indices.append(j)
            last_word_end_offset = offset[1]
        
        if current_word_indices: # Last word
            start_offset = offset_mapping[current_word_indices[0]][0]
            end_offset = offset_mapping[current_word_indices[-1]][1]
            word = original_text[start_offset:end_offset]
            avg_score = np.mean(cls_attentions[current_word_indices])
            if word.strip(): word_scores[word.strip()].append(avg_score)
            
        return {word: np.mean(scores) for word, scores in word_scores.items()}

    def extract_top_attention_words(self, word_scores_dict: Dict[str, float]) -> List[str]:
        if not word_scores_dict: return []
        sorted_words = sorted(word_scores_dict.items(), key=lambda x: x[1], reverse=True)
        num_words = len(sorted_words)
        n_top = max(self.config.MIN_TOP_WORDS, math.ceil(num_words * self.config.ATTENTION_TOP_PERCENT))
        
        # Simple stopword list (can be expanded)
        stopwords = {'__arg__', '__num__', 'a', 'an', 'the', 'is', 'was', 'to', 'of', 'for', 'on', 'in', 'at'}
        top_words_filtered = [word for word, score in sorted_words[:n_top] if word.lower() not in stopwords and len(word) > 1]
        
        return top_words_filtered if top_words_filtered else [word for word, score in sorted_words[:n_top]]


    def process_full_dataset(self, df: pd.DataFrame, exclude_class: str = None) -> pd.DataFrame:
        print("Processing dataset for attention analysis (base classifier)...")
        
        # exclude_class가 지정된 경우 해당 클래스를 제외하고 분석
        if exclude_class:
            exclude_class_lower = exclude_class.lower()
            df_for_analysis = df[df[self.config.CLASS_COLUMN].str.lower() != exclude_class_lower].copy()
            print(f"Excluding '{exclude_class}' class. Analyzing {len(df_for_analysis)}/{len(df)} samples.")
        else:
            df_for_analysis = df.copy()
            print(f"Analyzing all {len(df_for_analysis)} samples.")
        
        if df_for_analysis.empty:
            print("No data available for attention analysis after filtering.")
            return df.copy()  # 원본 데이터프레임 반환 (빈 컬럼들 추가됨)
        
        texts = df_for_analysis[self.config.TEXT_COLUMN].tolist()
        
        print("Computing word attention scores...")
        all_word_scores = self.get_word_attention_scores(texts, self.config.ATTENTION_LAYER)
        
        all_top_words, masked_texts = [], []
        print("Extracting top attention words and creating masked texts...")
        for i, (text, word_scores) in enumerate(zip(texts, all_word_scores)):
            top_words = self.extract_top_attention_words(word_scores)
            all_top_words.append(top_words)
            masked_texts.append(create_masked_sentence(text, top_words))
        
        # 결과를 원본 데이터프레임에 맞게 정렬
        result_df = df.copy()
        result_df['top_attention_words'] = None
        result_df[self.config.TEXT_COLUMN_IN_OE_FILES] = None
        
        # 분석된 데이터만 업데이트
        if exclude_class:
            analyze_mask = df[self.config.CLASS_COLUMN].str.lower() != exclude_class_lower
            result_df.loc[analyze_mask, 'top_attention_words'] = all_top_words
            result_df.loc[analyze_mask, self.config.TEXT_COLUMN_IN_OE_FILES] = masked_texts
            
            # exclude된 클래스는 빈 리스트와 원본 텍스트로 설정
            result_df.loc[~analyze_mask, 'top_attention_words'] = [[] for _ in range((~analyze_mask).sum())]
            result_df.loc[~analyze_mask, self.config.TEXT_COLUMN_IN_OE_FILES] = df.loc[~analyze_mask, self.config.TEXT_COLUMN]
        else:
            result_df['top_attention_words'] = all_top_words
            result_df[self.config.TEXT_COLUMN_IN_OE_FILES] = masked_texts
            
        return result_df

# === OE 추출 클래스 (OE Extraction Part) ===
# This TextDataset is for processing masked texts to get their attention metrics
class MaskedTextDatasetForMetrics(TorchDataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.texts = texts
        valid_texts = [str(t) if pd.notna(t) else "" for t in texts]
        self.encodings = tokenizer(
            valid_texts, max_length=max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        # Return only input_ids and attention_mask, no labels needed for this
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}


class OEExtractor:
    def __init__(self, config: Config, model_pl: UnifiedModel, tokenizer, device):
        self.config = config
        self.model_pl = model_pl.to(device) # Ensure model is on the correct device
        self.model_pl.eval()
        self.model_pl.freeze()
        self.tokenizer = tokenizer
        self.device = device # Store device

    @torch.no_grad()
    def extract_attention_metrics(self, dataloader: DataLoader, original_df: pd.DataFrame = None, 
                                exclude_class: str = None) -> Tuple[pd.DataFrame, List[np.ndarray]]:
        attention_metrics_list = []
        features_list = []
        print("Extracting attention metrics and features from masked texts...")
        
        batch_count = 0
        for batch_encodings in tqdm(dataloader, desc="Processing masked text batches", leave=False):
            # Move batch to device
            batch_on_device = {k: v.to(self.device) for k, v in batch_encodings.items()}
            
            outputs = self.model_pl.forward(batch_on_device, output_features=True, output_attentions=True)
            attentions_batch = outputs.attentions[-1].cpu().numpy()
            features_batch = outputs.hidden_states[-1][:, 0, :].cpu().numpy() # CLS token
            features_list.extend(list(features_batch))
            
            input_ids_batch = batch_encodings['input_ids'].cpu().numpy()
            for i in range(len(input_ids_batch)):
                metrics = self._compute_attention_metrics(attentions_batch[i], input_ids_batch[i])
                attention_metrics_list.append(metrics)
                batch_count += 1
        
        del outputs, attentions_batch, features_batch; gc.collect()
        
        # exclude_class 처리를 위한 로직 추가
        if original_df is not None and exclude_class and len(original_df) > len(attention_metrics_list):
            # 전체 데이터프레임과 매칭하기 위해 빈 메트릭 추가
            exclude_class_lower = exclude_class.lower()
            exclude_mask = original_df[self.config.CLASS_COLUMN].str.lower() == exclude_class_lower
            
            full_metrics_list = []
            full_features_list = []
            metrics_idx = 0
            
            default_metrics = {'max_attention': 0, 'top_k_avg_attention': 0, 'attention_entropy': 0}
            default_features = np.zeros_like(features_list[0]) if features_list else np.zeros(768)
            
            for idx, is_excluded in enumerate(exclude_mask):
                if is_excluded:
                    full_metrics_list.append(default_metrics.copy())
                    full_features_list.append(default_features.copy())
                else:
                    if metrics_idx < len(attention_metrics_list):
                        full_metrics_list.append(attention_metrics_list[metrics_idx])
                        full_features_list.append(features_list[metrics_idx])
                        metrics_idx += 1
                    else:
                        full_metrics_list.append(default_metrics.copy())
                        full_features_list.append(default_features.copy())
            
            return pd.DataFrame(full_metrics_list), full_features_list
        
        return pd.DataFrame(attention_metrics_list), features_list

    def _compute_attention_metrics(self, attention_sample, input_ids):
        valid_indices = np.where(
            (input_ids != self.tokenizer.pad_token_id) &
            (input_ids != self.tokenizer.cls_token_id) &
            (input_ids != self.tokenizer.sep_token_id)
        )[0]
        
        if len(valid_indices) == 0:
            return {'max_attention': 0, 'top_k_avg_attention': 0, 'attention_entropy': 0}
        
        cls_attentions = np.mean(attention_sample[:, 0, :], axis=0)[valid_indices] # Mean over heads, CLS token's view, valid tokens
        
        max_att = np.max(cls_attentions) if len(cls_attentions) > 0 else 0
        k = min(self.config.TOP_K_ATTENTION, len(cls_attentions))
        top_k_avg_att = np.mean(np.sort(cls_attentions)[-k:]) if k > 0 else 0
        
        att_probs = F.softmax(torch.tensor(cls_attentions), dim=0).numpy()
        att_entropy = entropy(att_probs) if len(att_probs) > 1 else 0 # Entropy needs >1 element
        
        return {'max_attention': max_att, 'top_k_avg_attention': top_k_avg_att, 'attention_entropy': att_entropy}

    def compute_removed_word_attention(self, df: pd.DataFrame, attention_analyzer: AttentionAnalyzer) -> pd.DataFrame:
        print("Computing removed word attention scores...")
        if 'top_attention_words' not in df.columns or self.config.TEXT_COLUMN not in df.columns:
            print("  Required columns ('top_attention_words', text column) not found. Skipping removed_avg_attention.")
            df['removed_avg_attention'] = 0.0
            return df
        
        texts = df[self.config.TEXT_COLUMN].tolist()
        word_attentions_list = attention_analyzer.get_word_attention_scores(texts) # Re-use analyzer
        
        removed_attentions = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing removed attention", leave=False):
            # Ensure top_attention_words is a list
            top_words_val = row['top_attention_words']
            top_words = safe_literal_eval(top_words_val) # Handles stringified lists and actual lists

            if top_words and idx < len(word_attentions_list):
                word_scores_dict = word_attentions_list[idx]
                # Match words case-insensitively if needed, or ensure consistency
                removed_scores = [word_scores_dict.get(word, 0) for word in top_words] # Exact match from dict
                removed_attentions.append(np.mean(removed_scores) if removed_scores else 0)
            else:
                removed_attentions.append(0)
        
        df['removed_avg_attention'] = removed_attentions
        print("Removed word attention computation complete.")
        return df

    def extract_oe_datasets(self, df: pd.DataFrame, exclude_class: str = None) -> None:
        print("Extracting OE datasets with different criteria...")
        
        # exclude_class가 있는 경우 해당 클래스 제외
        if exclude_class:
            exclude_class_lower = exclude_class.lower()
            df_for_oe = df[df[self.config.CLASS_COLUMN].str.lower() != exclude_class_lower].copy()
            print(f"OE extraction excluding '{exclude_class}' class: {len(df_for_oe)}/{len(df)} samples")
        else:
            df_for_oe = df.copy()
        
        if df_for_oe.empty:
            print("No data available for OE extraction after filtering.")
            return
            
        for metric, settings in self.config.METRIC_SETTINGS.items():
            if metric not in df_for_oe.columns:
                print(f"Skipping OE extraction for {metric} - column not found in DataFrame.")
                continue
            self._extract_single_metric_oe(df_for_oe, metric, settings)
        self._extract_sequential_filtering_oe(df_for_oe)

    def _extract_single_metric_oe(self, df: pd.DataFrame, metric: str, settings: dict):
        scores = np.nan_to_num(df[metric].values, nan=0.0)
        if settings['mode'] == 'higher':
            threshold = np.percentile(scores, 100 - settings['percentile'])
            selected_indices = np.where(scores >= threshold)[0]
        else: # 'lower'
            threshold = np.percentile(scores, settings['percentile'])
            selected_indices = np.where(scores <= threshold)[0]
        
        if len(selected_indices) > 0:
            # OE df needs the configured text column for OSR experiments
            oe_df_simple = df.iloc[selected_indices][[self.config.TEXT_COLUMN_IN_OE_FILES]].copy()
            # Rename to standard 'text' if needed by OSR loader, or handle in loader
            # For now, OSR loader will use TEXT_COLUMN_IN_OE_FILES from config
            
            # Extended version for analysis
            extended_cols = [self.config.TEXT_COLUMN_IN_OE_FILES, self.config.TEXT_COLUMN, 'top_attention_words', metric]
            extended_cols = [col for col in extended_cols if col in df.columns] # Only include existing columns
            oe_df_extended = df.iloc[selected_indices][extended_cols].copy()

            mode_desc = f"{settings['mode']}{settings['percentile']}pct"
            oe_filename = os.path.join(self.config.OE_DATA_DIR, f"oe_data_{metric}_{mode_desc}.csv")
            oe_extended_filename = os.path.join(self.config.OE_DATA_DIR, f"oe_data_{metric}_{mode_desc}_extended.csv")
            
            oe_df_simple.to_csv(oe_filename, index=False)
            oe_df_extended.to_csv(oe_extended_filename, index=False)
            print(f"Saved OE dataset ({len(oe_df_simple)} samples) for {metric} {mode_desc}: {oe_filename}")

    def _extract_sequential_filtering_oe(self, df: pd.DataFrame):
        print("Applying sequential filtering for OE extraction...")
        selected_mask = np.ones(len(df), dtype=bool)
        
        for step, (metric, settings) in enumerate(self.config.FILTERING_SEQUENCE):
            if metric not in df.columns:
                print(f"Skipping sequential filter step {step+1}: {metric} not found.")
                continue
            
            current_selection_df = df[selected_mask]
            if current_selection_df.empty:
                print(f"No samples left before applying filter: {metric}. Stopping sequential filtering.")
                selected_mask[:] = False # Mark all as not selected
                break

            scores = np.nan_to_num(current_selection_df[metric].values, nan=0.0)
            
            if settings['mode'] == 'higher':
                threshold = np.percentile(scores, 100 - settings['percentile'])
                step_mask = scores >= threshold
            else:
                threshold = np.percentile(scores, settings['percentile'])
                step_mask = scores <= threshold
            
            # Update global mask based on indices from current_selection_df
            current_indices = np.where(selected_mask)[0] # Indices in the original df
            indices_to_keep_from_current_step = current_indices[step_mask]
            
            selected_mask = np.zeros_like(selected_mask) # Reset global mask
            if len(indices_to_keep_from_current_step) > 0:
                selected_mask[indices_to_keep_from_current_step] = True # Set new selections
            
            print(f"Sequential Filter {step+1} ({metric} {settings['mode']} {settings['percentile']}%): {np.sum(selected_mask)} samples remaining")

        final_indices = np.where(selected_mask)[0]
        if len(final_indices) > 0:
            oe_df_simple = df.iloc[final_indices][[self.config.TEXT_COLUMN_IN_OE_FILES]].copy()
            
            extended_cols = [self.config.TEXT_COLUMN_IN_OE_FILES, self.config.TEXT_COLUMN, 'top_attention_words']
            extended_cols.extend([m for m, _ in self.config.FILTERING_SEQUENCE if m in df.columns])
            oe_df_extended = df.iloc[final_indices][extended_cols].copy()

            filter_desc = "_".join([f"{m}_{s['mode']}{s['percentile']}" for m, s in self.config.FILTERING_SEQUENCE])
            oe_filename = os.path.join(self.config.OE_DATA_DIR, f"oe_data_sequential_{filter_desc}.csv")
            oe_extended_filename = os.path.join(self.config.OE_DATA_DIR, f"oe_data_sequential_{filter_desc}_extended.csv")
            
            oe_df_simple.to_csv(oe_filename, index=False)
            oe_df_extended.to_csv(oe_extended_filename, index=False)
            print(f"Saved sequential OE dataset ({len(oe_df_simple)} samples): {oe_filename}")
        else:
            print("No samples selected by sequential filtering.")

# === 시각화 클래스 (OE Extraction Part) ===
class Visualizer:
    def __init__(self, config: Config):
        self.config = config
    
    def plot_metric_distribution(self, scores: np.ndarray, metric_name: str, title: str, save_path: str):
        if len(scores) == 0: return
        plt.figure(figsize=(10, 6))
        if SNS_AVAILABLE: sns.histplot(scores, bins=50, kde=True, stat='density')
        else: plt.hist(scores, bins=50, density=True, alpha=0.7)
        plt.title(title, fontsize=14); plt.xlabel(metric_name, fontsize=12); plt.ylabel('Density', fontsize=12)
        plt.grid(alpha=0.3)
        mean_val = np.mean(scores); plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        plt.legend(); plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
        print(f"Distribution plot saved: {save_path}")

    def plot_tsne(self, features: np.ndarray, labels: np.ndarray, title: str, save_path: str,
                  highlight_indices: Optional[np.ndarray] = None, highlight_label: str = 'OE Candidate',
                  class_names: Optional[Dict] = None, seed: int = 42):
        if len(features) == 0: return
        print(f"Running t-SNE for OE Viz on {features.shape[0]} samples...")
        try:
            perplexity = min(30, features.shape[0] - 1)
            if perplexity <= 1: print(f"t-SNE perplexity too low ({perplexity}). Skipping."); return
            tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, n_iter=1000, init='pca', learning_rate='auto')
            tsne_results = tsne.fit_transform(features)
        except Exception as e: print(f"Error in t-SNE for OE Viz: {e}"); return
        
        df_tsne = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
        df_tsne['label'] = labels
        df_tsne['is_highlighted'] = False
        if highlight_indices is not None: df_tsne.loc[highlight_indices, 'is_highlighted'] = True
        
        plt.figure(figsize=(14, 10))
        unique_labels = sorted(df_tsne['label'].unique())
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label_val in enumerate(unique_labels):
            subset = df_tsne[(df_tsne['label'] == label_val) & (~df_tsne['is_highlighted'])]
            if len(subset) > 0:
                c_name = class_names.get(label_val, f'Class {label_val}') if class_names else f'Class {label_val}'
                plt.scatter(subset['tsne1'], subset['tsne2'], color=colors[i], label=c_name, alpha=0.7, s=30)
        
        if highlight_indices is not None and len(df_tsne[df_tsne['is_highlighted']]) > 0:
            plt.scatter(df_tsne[df_tsne['is_highlighted']]['tsne1'], df_tsne[df_tsne['is_highlighted']]['tsne2'],
                        color='red', marker='x', s=100, label=highlight_label, alpha=0.9)
        
        plt.title(title, fontsize=16, pad=20); plt.xlabel("t-SNE Dim 1"); plt.ylabel("t-SNE Dim 2")
        plt.grid(alpha=0.3, linestyle='--')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=True, fancybox=True, shadow=True)
        plt.tight_layout(); plt.subplots_adjust(right=0.75)
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
        print(f"t-SNE plot for OE Viz saved: {save_path}")

    def visualize_all_metrics(self, df: pd.DataFrame):
        metric_columns = ['max_attention', 'top_k_avg_attention', 'attention_entropy', 'removed_avg_attention']
        for metric in metric_columns:
            if metric in df.columns and not df[metric].isnull().all():
                self.plot_metric_distribution(
                    df[metric].dropna().values, metric, f'Distribution of {metric}',
                    os.path.join(self.config.VIS_DIR, f'{metric}_distribution.png')
                )

    def visualize_oe_candidates(self, df: pd.DataFrame, features: List[np.ndarray], label2id: dict, id2label: dict):
        if not features or len(features) != len(df):
            print(f"Feature length mismatch or no features for OE t-SNE. Features: {len(features)}, DF: {len(df)}")
            return

        # Prepare labels for t-SNE: map original class column using provided label2id
        # Samples from EXCLUDE_CLASS_FOR_TRAINING are marked as 'Unknown' (-1)
        # Other classes are mapped or marked as 'Other/Filtered' (-2)
        tsne_labels = []
        unknown_class_lower = self.config.EXCLUDE_CLASS_FOR_TRAINING.lower()
        for cls_val in df[self.config.CLASS_COLUMN]: # Use original class column
            cls_str = str(cls_val).lower()
            if cls_str == unknown_class_lower:
                tsne_labels.append(-1)  # Unknown (OOD relative to base classifier)
            else:
                tsne_labels.append(label2id.get(cls_str, -2)) # Map known or mark as Other
        tsne_labels_np = np.array(tsne_labels)
        
        # Define class names for legend, including special labels
        class_names_viz = {**{k: str(v) for k,v in id2label.items()}, -1: 'Unknown (Excluded)', -2: 'Other/Filtered'}

        # Visualize for each metric setting
        for metric, settings in self.config.METRIC_SETTINGS.items():
            if metric not in df.columns: continue
            scores = np.nan_to_num(df[metric].values, nan=0.0)
            threshold = np.percentile(scores, 100 - settings['percentile'] if settings['mode'] == 'higher' else settings['percentile'])
            oe_indices = np.where(scores >= threshold if settings['mode'] == 'higher' else scores <= threshold)[0]
            mode_desc = f"{settings['mode']}{settings['percentile']}%"
            
            self.plot_tsne(
                np.array(features), tsne_labels_np,
                f't-SNE (OE Viz): Candidates by {metric} ({mode_desc})',
                os.path.join(self.config.VIS_DIR, f'tsne_oe_cand_{metric}_{mode_desc}.png'),
                highlight_indices=oe_indices, highlight_label=f'OE Candidate ({metric} {mode_desc})',
                class_names=class_names_viz, seed=self.config.RANDOM_STATE
            )
        
        # Visualize for sequential filtering (if applicable)
        if hasattr(self.config, 'FILTERING_SEQUENCE') and self.config.FILTERING_SEQUENCE:
            selected_mask = np.ones(len(df), dtype=bool)
            filter_steps_desc_list = []
            for step, (metric, settings) in enumerate(self.config.FILTERING_SEQUENCE):
                if metric not in df.columns: continue
                current_selection_df = df[selected_mask]
                if current_selection_df.empty: break
                scores = np.nan_to_num(current_selection_df[metric].values, nan=0.0)
                threshold = np.percentile(scores, 100 - settings['percentile'] if settings['mode'] == 'higher' else settings['percentile'])
                step_mask_on_subset = scores >= threshold if settings['mode'] == 'higher' else scores <= threshold
                
                current_indices = np.where(selected_mask)[0]
                indices_to_keep_from_step = current_indices[step_mask_on_subset]
                selected_mask = np.zeros_like(selected_mask)
                if len(indices_to_keep_from_step) > 0: selected_mask[indices_to_keep_from_step] = True
                
                mode_desc = f"{settings['mode']}{settings['percentile']}%"
                filter_steps_desc_list.append(f"{metric}({mode_desc})")

            final_indices_seq = np.where(selected_mask)[0]
            if len(final_indices_seq) > 0:
                seq_desc = " -> ".join(filter_steps_desc_list)
                self.plot_tsne(
                    np.array(features), tsne_labels_np,
                    f't-SNE (OE Viz): Sequential Filter Candidates\n{seq_desc} -> {len(final_indices_seq)} samples',
                    os.path.join(self.config.VIS_DIR, f'tsne_oe_cand_sequential_{"_".join(filter_steps_desc_list)}.png'),
                    highlight_indices=final_indices_seq, highlight_label=f'Sequential OE Candidate ({len(final_indices_seq)} samples)',
                    class_names=class_names_viz, seed=self.config.RANDOM_STATE
                )

# === 메인 파이프라인 클래스 ===
class UnifiedOEExtractor:
    def __init__(self, config: Config):
        self.config = config
        self.data_module: Optional[UnifiedDataModule] = None
        self.model: Optional[UnifiedModel] = None # Base classifier model
        self.attention_analyzer: Optional[AttentionAnalyzer] = None
        self.oe_extractor: Optional[OEExtractor] = None
        self.visualizer = Visualizer(config)
        
        config.create_directories()
        config.save_config()
        set_seed(config.RANDOM_STATE)
    
    def run_stage1_model_training(self):
        if not self.config.STAGE_MODEL_TRAINING:
            print("Skipping Stage 1: Base Model Training")
            # Try to load if skipping, ensure datamodule is set up for later stages
            if self._check_existing_model():
                self._load_existing_model()
            else:
                print("Error: STAGE_MODEL_TRAINING skipped, but no existing model found. Cannot proceed with other stages requiring the model.")
                sys.exit(1) # Critical if model needed later
            return

        print("\n" + "="*50 + "\nSTAGE 1: BASE MODEL TRAINING\n" + "="*50)
        self.data_module = UnifiedDataModule(self.config)
        self.data_module.setup() # This will print progress
        
        self.model = UnifiedModel(
            config=self.config, num_labels=self.data_module.num_labels,
            label2id=self.data_module.label2id, id2label=self.data_module.id2label,
            class_weights=self.data_module.class_weights
        )
        
        monitor_metric = 'val_f1_macro'
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.MODEL_SAVE_DIR,
            filename=f'base-clf-{{epoch:02d}}-{{{monitor_metric}:.4f}}',
            save_top_k=1, monitor=monitor_metric, mode='max'
        )
        early_stopping_callback = EarlyStopping(monitor=monitor_metric, patience=3, mode='max', verbose=True)
        csv_logger = CSVLogger(save_dir=self.config.LOG_DIR, name="base_model_training")
        
        if self._check_existing_model() and self.config.OSR_EVAL_ONLY: # If OSR_EVAL_ONLY, assume base model is also fixed
             print("Found existing base model and OSR_EVAL_ONLY is True. Skipping base model training.")
             self._load_existing_model(checkpoint_callback)
        else:
            trainer = pl.Trainer(
                max_epochs=self.config.NUM_TRAIN_EPOCHS, accelerator=self.config.ACCELERATOR,
                devices=self.config.DEVICES, precision=self.config.PRECISION, logger=csv_logger,
                callbacks=[checkpoint_callback, early_stopping_callback],
                deterministic=False, # Usually True with seed_everything, but can be False for speed if slightly different results are ok
                log_every_n_steps=self.config.LOG_EVERY_N_STEPS,
                gradient_clip_val=self.config.GRADIENT_CLIP_VAL
            )
            print("Starting base model training...")
            trainer.fit(self.model, datamodule=self.data_module)
            print("Base model training complete!")
            self._load_best_model(checkpoint_callback) # Load the best checkpoint

    def run_stage2_attention_extraction(self) -> Optional[pd.DataFrame]:
        if not self.config.STAGE_ATTENTION_EXTRACTION:
            print("Skipping Stage 2: Attention Extraction")
            if self.config.STAGE_OE_EXTRACTION or self.config.STAGE_VISUALIZATION:
                try: return self._load_attention_results()
                except FileNotFoundError: print("Attention results not found, cannot proceed with dependent stages if skipped."); return None
            return None

        print("\n" + "="*50 + "\nSTAGE 2: ATTENTION EXTRACTION\n" + "="*50)
        if self.model is None: self._load_existing_model()
        if self.data_module is None:
            self.data_module = UnifiedDataModule(self.config); self.data_module.setup()

        current_device = self.model.device if hasattr(self.model, 'device') else DEVICE_OSR

        self.attention_analyzer = AttentionAnalyzer(
            config=self.config, model_pl=self.model,
            tokenizer=self.data_module.tokenizer, device=current_device
        )
        
        full_df = self.data_module.get_full_dataframe()
        # EXCLUDE_CLASS_FOR_TRAINING 클래스를 제외하고 attention 분석
        processed_df = self.attention_analyzer.process_full_dataset(
            full_df, exclude_class=self.config.EXCLUDE_CLASS_FOR_TRAINING
        )
        
        output_path = os.path.join(self.config.ATTENTION_DATA_DIR, f"df_with_attention.csv")
        processed_df.to_csv(output_path, index=False)
        print(f"Attention analysis results saved: {output_path}")
        self._print_attention_samples(processed_df)
        return processed_df

    def run_stage3_oe_extraction(self, df_with_attention: Optional[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], Optional[List[np.ndarray]]]:
        if not self.config.STAGE_OE_EXTRACTION:
            print("Skipping Stage 3: OE Extraction")
            if self.config.STAGE_VISUALIZATION or self.config.STAGE_OSR_EXPERIMENTS:
                try: return self._load_final_metrics_and_features()
                except FileNotFoundError: print("Final metrics/features not found, cannot proceed if OE extraction skipped."); return None, None
            return None, None

        print("\n" + "="*50 + "\nSTAGE 3: OE EXTRACTION\n" + "="*50)
        if df_with_attention is None: df_with_attention = self._load_attention_results()
        if df_with_attention is None:
            print("Error: DataFrame with attention is not available. Cannot proceed with OE extraction.")
            return None, None

        if self.model is None: self._load_existing_model()
        if self.data_module is None: self.data_module = UnifiedDataModule(self.config); self.data_module.setup()
        
        current_device = self.model.device if hasattr(self.model, 'device') else DEVICE_OSR
        self.oe_extractor = OEExtractor(
            config=self.config, model_pl=self.model,
            tokenizer=self.data_module.tokenizer, device=current_device
        )
        
        masked_texts_col = self.config.TEXT_COLUMN_IN_OE_FILES
        if masked_texts_col not in df_with_attention.columns:
            print(f"Error: Column '{masked_texts_col}' not found in attention DataFrame. Cannot extract OE.")
            return df_with_attention, None

        # EXCLUDE_CLASS_FOR_TRAINING 클래스를 제외한 데이터만 처리
        exclude_class_lower = self.config.EXCLUDE_CLASS_FOR_TRAINING.lower()
        mask_for_analysis = df_with_attention[self.config.CLASS_COLUMN].str.lower() != exclude_class_lower
        df_for_metrics = df_with_attention[mask_for_analysis].copy()
        
        if df_for_metrics.empty:
            print("No data available for OE extraction after excluding unknown class.")
            return df_with_attention, None
        
        masked_texts = df_for_metrics[masked_texts_col].tolist()
        print(f"Processing {len(masked_texts)} samples for OE metrics (excluding '{self.config.EXCLUDE_CLASS_FOR_TRAINING}' class)")
        
        dataset = MaskedTextDatasetForMetrics(masked_texts, self.data_module.tokenizer, self.config.MAX_LENGTH)
        dataloader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE, num_workers=self.config.NUM_WORKERS, shuffle=False)
        
        attention_metrics_df, features = self.oe_extractor.extract_attention_metrics(
            dataloader, original_df=df_with_attention, exclude_class=self.config.EXCLUDE_CLASS_FOR_TRAINING
        )
        
        df_with_metrics = df_with_attention.reset_index(drop=True)
        if len(df_with_metrics) == len(attention_metrics_df):
            df_with_metrics = pd.concat([df_with_metrics, attention_metrics_df.reset_index(drop=True)], axis=1)
        else:
            print(f"Warning: Length mismatch between main DF ({len(df_with_metrics)}) and metrics DF ({len(attention_metrics_df)}). Metrics not merged.")
        
        if self.attention_analyzer:
            df_with_metrics = self.oe_extractor.compute_removed_word_attention(df_with_metrics, self.attention_analyzer)
        
        # OE 추출 시에도 exclude_class를 고려
        self.oe_extractor.extract_oe_datasets(df_with_metrics, exclude_class=self.config.EXCLUDE_CLASS_FOR_TRAINING)
        
        metrics_output_path = os.path.join(self.config.ATTENTION_DATA_DIR, f"df_with_all_metrics.csv")
        df_with_metrics.to_csv(metrics_output_path, index=False)
        print(f"DataFrame with all metrics saved: {metrics_output_path}")
        
        if features:
            features_path = os.path.join(self.config.ATTENTION_DATA_DIR, "extracted_features.npy")
            np.save(features_path, np.array(features))
            print(f"Extracted features saved: {features_path}")

        return df_with_metrics, features
    
    def run_stage4_visualization(self, df_with_metrics: Optional[pd.DataFrame], features: Optional[List[np.ndarray]]):
        if not self.config.STAGE_VISUALIZATION:
            print("Skipping Stage 4: OE Extraction Visualization")
            return

        print("\n" + "="*50 + "\nSTAGE 4: OE EXTRACTION VISUALIZATION\n" + "="*50)
        if df_with_metrics is None or features is None:
            df_with_metrics, features = self._load_final_metrics_and_features()
        
        if df_with_metrics is None:
            print("Error: DataFrame with metrics not available for visualization.")
            return

        self.visualizer.visualize_all_metrics(df_with_metrics)
        
        if features and self.data_module: # data_module needed for label mappings
            self.visualizer.visualize_oe_candidates(
                df_with_metrics, features,
                self.data_module.label2id, self.data_module.id2label
            )
        elif not features:
            print("No features available for t-SNE visualization of OE candidates.")
        elif not self.data_module:
            print("DataModule not available, cannot perform t-SNE visualization with class labels.")
            
        print("OE Extraction visualizations complete!")

    def _run_single_osr_experiment(self,
                                   osr_tokenizer,
                                   num_osr_classes: int,
                                   osr_id_label2id: Dict, osr_id_id2label: Dict,
                                   osr_known_class_names: List[str],
                                   id_train_loader_osr: DataLoader,
                                   id_test_loader_osr: DataLoader,
                                   ood_eval_loader_osr: Optional[DataLoader],
                                   current_oe_source_name: Optional[str], # Basename of OE file or "Standard"
                                   current_oe_data_path: Optional[str], # Full path to OE file, or None
                                   ood_dataset_eval_name_tag: str
                                   ) -> Tuple[Dict, Dict]:
        """ Helper to run one OSR trial (standard or with one OE dataset) """
        experiment_tag = "SyslogOSR" # Fixed ID dataset type for this integration
        if current_oe_source_name:
            experiment_tag += f"_OE_{current_oe_source_name}"
        else:
            experiment_tag += "_Standard"
        
        print(f"\n\n===== Starting Single OSR Experiment: {experiment_tag} =====")

        sanitized_oe_name = re.sub(r'[^\w\-.]+', '_', current_oe_source_name) if current_oe_source_name else "Standard"
        exp_result_subdir = os.path.join("SyslogOSR", f"OE_{sanitized_oe_name}" if current_oe_source_name else "Standard")
        
        current_result_dir = os.path.join(self.config.OSR_RESULT_DIR, exp_result_subdir)
        current_model_dir = os.path.join(self.config.OSR_MODEL_DIR, exp_result_subdir)
        os.makedirs(current_result_dir, exist_ok=True)
        if self.config.OSR_SAVE_MODEL_PER_EXPERIMENT:
            os.makedirs(current_model_dir, exist_ok=True)

        model_osr = RoBERTaOOD(num_osr_classes, self.config.OSR_MODEL_TYPE).to(DEVICE_OSR)
        if osr_id_label2id and osr_id_id2label: # For Hugging Face model hub compatibility
            model_osr.roberta.config.label2id = osr_id_label2id
            model_osr.roberta.config.id2label = osr_id_id2label
        
        model_filename_base = f"roberta_osr_{experiment_tag}_{num_osr_classes}cls_seed{self.config.RANDOM_STATE}.pt"
        model_save_path = os.path.join(current_model_dir, model_filename_base)

        experiment_results = {}
        experiment_data_for_plots = {}

        if self.config.OSR_EVAL_ONLY:
            if os.path.exists(model_save_path):
                print(f"Loading pre-trained OSR model for '{experiment_tag}' from {model_save_path}...")
                model_osr.load_state_dict(torch.load(model_save_path, map_location=DEVICE_OSR))
            else:
                print(f"Error: Model path '{model_save_path}' not found for OSR_EVAL_ONLY. Skipping.")
                return {}, {}
        else: # Training mode for OSR model
            optimizer = AdamW(model_osr.parameters(), lr=self.config.OSR_LEARNING_RATE)
            total_steps = len(id_train_loader_osr) * self.config.OSR_NUM_EPOCHS
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

            if current_oe_data_path: # OE-enhanced training
                print(f"Preparing OE data for OSR from: {current_oe_data_path}")
                oe_train_dataset_osr = prepare_generated_oe_data_for_osr(
                    osr_tokenizer, self.config.OSR_MAX_LENGTH,
                    current_oe_data_path, self.config.TEXT_COLUMN_IN_OE_FILES
                )
                if oe_train_dataset_osr:
                    oe_train_loader_osr = DataLoader(
                        oe_train_dataset_osr, batch_size=self.config.OSR_BATCH_SIZE, shuffle=True,
                        num_workers=self.config.OSR_NUM_DATALOADER_WORKERS, pin_memory=True,
                        persistent_workers=self.config.OSR_NUM_DATALOADER_WORKERS > 0
                    )
                    train_with_oe_uniform_loss_osr(
                        model_osr, id_train_loader_osr, oe_train_loader_osr, optimizer, scheduler,
                        DEVICE_OSR, self.config.OSR_NUM_EPOCHS, self.config.OSR_OE_LAMBDA, experiment_tag
                    )
                    del oe_train_dataset_osr, oe_train_loader_osr; gc.collect()
                else:
                    print(f"Failed to load OE data for OSR from '{current_oe_data_path}'. Training standard OSR model instead.")
                    train_standard_osr(model_osr, id_train_loader_osr, optimizer, scheduler, DEVICE_OSR, self.config.OSR_NUM_EPOCHS, experiment_tag)
            else: # Standard OSR training (no OE)
                train_standard_osr(model_osr, id_train_loader_osr, optimizer, scheduler, DEVICE_OSR, self.config.OSR_NUM_EPOCHS, experiment_tag)
            
            if self.config.OSR_SAVE_MODEL_PER_EXPERIMENT:
                torch.save(model_osr.state_dict(), model_save_path)
                print(f"OSR Model for '{experiment_tag}' saved to {model_save_path}")

        # Evaluation
        if ood_eval_loader_osr is None:
            print(f"Warning: No OOD evaluation data for OSR experiment '{experiment_tag}'. Only closed-set metrics.")
        
        results_osr, data_osr = evaluate_osr(
            model_osr, id_test_loader_osr, ood_eval_loader_osr, DEVICE_OSR,
            self.config.OSR_TEMPERATURE, self.config.OSR_THRESHOLD_PERCENTILE, return_data=True
        )
        print(f"  OSSR Evaluation Results ({experiment_tag} vs {ood_dataset_eval_name_tag}): {results_osr}")

        metric_key_prefix = f"SyslogOSR_"
        metric_key_prefix += f"OE_{current_oe_source_name}" if current_oe_source_name else "Standard"
        full_metric_key = f"{metric_key_prefix}+{ood_dataset_eval_name_tag}"
        
        experiment_results[full_metric_key] = results_osr
        experiment_data_for_plots[full_metric_key] = data_osr

        if not self.config.OSR_NO_PLOT_PER_EXPERIMENT:
            plot_filename_prefix = re.sub(r'[^\w\-]+', '_', full_metric_key)
            if data_osr['id_scores'] is not None and data_osr['ood_scores'] is not None and len(data_osr['ood_scores']) > 0:
                plot_confidence_histograms_osr(data_osr['id_scores'], data_osr['ood_scores'],
                                           f'Conf - {experiment_tag} vs {ood_dataset_eval_name_tag}',
                                           os.path.join(current_result_dir, f'{plot_filename_prefix}_hist.png'))
                plot_roc_curve_osr(data_osr['id_scores'], data_osr['ood_scores'],
                               f'ROC - {experiment_tag} vs {ood_dataset_eval_name_tag}',
                               os.path.join(current_result_dir, f'{plot_filename_prefix}_roc.png'))
                plot_tsne_osr(data_osr['id_features'], data_osr['ood_features'],
                             f't-SNE - {experiment_tag} (ID vs OOD: {ood_dataset_eval_name_tag})',
                             os.path.join(current_result_dir, f'{plot_filename_prefix}_tsne.png'),
                             seed=self.config.RANDOM_STATE)
            if data_osr['id_labels_true'] is not None and len(data_osr['id_labels_true']) > 0 and num_osr_classes > 0:
                cm_std = confusion_matrix(data_osr['id_labels_true'], data_osr['id_labels_pred'], labels=np.arange(num_osr_classes))
                plot_confusion_matrix_osr(cm_std, osr_known_class_names,
                                       f'CM - {experiment_tag} (ID Test)',
                                       os.path.join(current_result_dir, f'{plot_filename_prefix}_cm.png'))
        
        del model_osr; gc.collect(); torch.cuda.empty_cache()
        return experiment_results, experiment_data_for_plots


    def run_stage5_osr_experiments(self):
        if not self.config.STAGE_OSR_EXPERIMENTS:
            print("Skipping Stage 5: OSR Experiments")
            return

        print("\n" + "="*50 + "\nSTAGE 5: OSR EXPERIMENTS\n" + "="*50)

        if self.data_module is None or self.data_module.num_labels is None:
            print("Error: UnifiedDataModule not set up. Cannot proceed with OSR experiments.")
            # Attempt to set it up if model training was skipped but data is needed
            if self.config.STAGE_MODEL_TRAINING == False:
                print("Attempting to set up DataModule as base model training was skipped...")
                self.data_module = UnifiedDataModule(self.config)
                self.data_module.setup()
                if self.data_module.num_labels is None: # Still not set up
                    print("Critical Error: Failed to set up DataModule. Aborting OSR experiments.")
                    return
            else: # Should have been set up in Stage 1
                return


        osr_tokenizer = RobertaTokenizer.from_pretrained(self.config.OSR_MODEL_TYPE)

        # Prepare ID data (train and test) for OSR
        # Uses train_df_final and val_df_final from UnifiedDataModule
        id_train_dataset_osr, id_test_dataset_osr, num_osr_classes, \
        osr_label_encoder, osr_id_label2id, osr_id_id2label = prepare_id_data_for_osr(
            self.data_module, osr_tokenizer, self.config.OSR_MAX_LENGTH
        )

        if id_train_dataset_osr is None or num_osr_classes == 0:
            print("Error: Failed to prepare ID data for OSR. Aborting OSR experiments.")
            return
        
        osr_known_class_names = list(osr_id_id2label.values()) if osr_id_id2label else [f"Class_{i}" for i in range(num_osr_classes)]

        id_train_loader_osr = DataLoader(
            id_train_dataset_osr, batch_size=self.config.OSR_BATCH_SIZE, shuffle=True,
            num_workers=self.config.OSR_NUM_DATALOADER_WORKERS, pin_memory=True,
            persistent_workers=self.config.OSR_NUM_DATALOADER_WORKERS > 0
        )
        id_test_loader_osr = DataLoader(
            id_test_dataset_osr, batch_size=self.config.OSR_BATCH_SIZE,
            num_workers=self.config.OSR_NUM_DATALOADER_WORKERS, pin_memory=True
        )

        # Prepare OOD evaluation data for OSR
        ood_eval_dataset_osr = prepare_syslog_ood_data_for_osr(
            osr_tokenizer, self.config.OSR_MAX_LENGTH,
            self.config.OOD_SYSLOG_UNKNOWN_PATH_OSR,
            self.config.TEXT_COLUMN, self.config.CLASS_COLUMN, # Use general text/class columns
            self.config.OOD_TARGET_CLASS_OSR
        )
        ood_eval_loader_osr = DataLoader(
            ood_eval_dataset_osr, batch_size=self.config.OSR_BATCH_SIZE,
            num_workers=self.config.OSR_NUM_DATALOADER_WORKERS, pin_memory=True
        ) if ood_eval_dataset_osr else None
        
        ood_dataset_eval_name_tag = self.config.OOD_TARGET_CLASS_OSR # For result keys

        all_osr_experiments_results = {}
        # all_osr_experiments_plot_data = {} # If needed for combined plots later

        # 1. Standard OSR Model (No OE)
        if not self.config.OSR_SKIP_STANDARD_MODEL:
            print("\n--- Running Standard OSR Model Experiment (No OE) ---")
            std_results, _ = self._run_single_osr_experiment(
                osr_tokenizer, num_osr_classes, osr_id_label2id, osr_id_id2label, osr_known_class_names,
                id_train_loader_osr, id_test_loader_osr, ood_eval_loader_osr,
                current_oe_source_name=None, current_oe_data_path=None,
                ood_dataset_eval_name_tag=ood_dataset_eval_name_tag
            )
            all_osr_experiments_results.update(std_results)

        # 2. OSR Models with Extracted OE Data
        print(f"\n--- Running OSR Experiments with OE data from: {self.config.OE_DATA_DIR} ---")
        oe_files = [f for f in os.listdir(self.config.OE_DATA_DIR) if f.endswith('.csv') and 'extended' not in f]
        
        if not oe_files:
            print("No OE dataset files found in OE_DATA_DIR. Skipping OE-enhanced OSR experiments.")
        else:
            for oe_filename in oe_files:
                oe_data_path = os.path.join(self.config.OE_DATA_DIR, oe_filename)
                # Use basename without extension as source name
                oe_source_name = os.path.splitext(oe_filename)[0] 
                
                print(f"\n--- OSR Experiment with OE: {oe_source_name} ---")
                oe_results, _ = self._run_single_osr_experiment(
                    osr_tokenizer, num_osr_classes, osr_id_label2id, osr_id_id2label, osr_known_class_names,
                    id_train_loader_osr, id_test_loader_osr, ood_eval_loader_osr,
                    current_oe_source_name=oe_source_name, current_oe_data_path=oe_data_path,
                    ood_dataset_eval_name_tag=ood_dataset_eval_name_tag
                )
                all_osr_experiments_results.update(oe_results)

        # --- Final OSR Results Summary ---
        print("\n\n===== OSR Experiments Overall Results Summary =====")
        if all_osr_experiments_results:
            final_osr_results_df = pd.DataFrame.from_dict(all_osr_experiments_results, orient='index')
            final_osr_results_df = final_osr_results_df.sort_index()
            print("Overall OSR Performance Metrics DataFrame:")
            print(final_osr_results_df)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_filename_base = f"osr_overall_summary_SyslogOSR_{timestamp_str}"
            
            # Save in OSR_RESULT_DIR (main level)
            overall_csv_path = os.path.join(self.config.OSR_RESULT_DIR, f'{summary_filename_base}.csv')
            overall_txt_path = os.path.join(self.config.OSR_RESULT_DIR, f'{summary_filename_base}.txt')
            overall_json_path = os.path.join(self.config.OSR_RESULT_DIR, f'{summary_filename_base}.json')

            final_osr_results_df.to_csv(overall_csv_path, index=True)
            print(f"\nOverall OSR results saved to CSV: {overall_csv_path}")

            # Prepare args_dict for saving (relevant OSR config)
            osr_args_to_save = {k: v for k, v in self.config.__class__.__dict__.items() 
                                if k.startswith('OSR_') or k in ['RANDOM_STATE', 'TEXT_COLUMN', 'CLASS_COLUMN']}
            osr_args_to_save['ORIGINAL_DATA_PATH'] = self.config.ORIGINAL_DATA_PATH
            osr_args_to_save['OE_DATA_DIR_USED'] = self.config.OE_DATA_DIR
            
            with open(overall_txt_path, 'w', encoding='utf-8') as f:
                f.write("--- OSR Experiment Arguments (Subset of Config) ---\n")
                f.write(json.dumps(osr_args_to_save, indent=4, default=str))
                f.write("\n\n--- Overall OSR Metrics ---\n")
                f.write(final_osr_results_df.to_string())
            print(f"Overall OSR results and arguments saved to TXT: {overall_txt_path}")

            summary_json_data = {
                'arguments_osr': osr_args_to_save,
                'timestamp': timestamp_str,
                'results_osr': all_osr_experiments_results
            }
            with open(overall_json_path, 'w', encoding='utf-8') as f:
                json.dump(summary_json_data, f, indent=4, default=str)
            print(f"Overall OSR results and arguments saved to JSON: {overall_json_path}")
        else:
            print("No OSR performance metrics were generated.")
        print("\nOSR Experiments Finished.")


    def run_full_pipeline(self):
        print("Starting Unified OE Extraction & OSR Pipeline...")
        
        df_with_attention, df_with_metrics, features = None, None, None

        df_with_attention = self.run_stage1_model_training() # Stage 1
        df_with_attention = self.run_stage2_attention_extraction() # Stage 2
        df_with_metrics, features = self.run_stage3_oe_extraction(df_with_attention) # Stage 3
        self.run_stage4_visualization(df_with_metrics, features) # Stage 4
        self.run_stage5_osr_experiments() # Stage 5

        self._print_final_summary()
        print("\nUnified OE Extraction & OSR Pipeline Complete!")

    # === Helper methods for UnifiedOEExtractor ===
    def _check_existing_model(self) -> bool:
        return (os.path.exists(self.config.MODEL_SAVE_DIR) and 
                any(file.endswith('.ckpt') for file in os.listdir(self.config.MODEL_SAVE_DIR)))

    def _load_existing_model(self, checkpoint_callback=None):
        # Ensure data_module is set up before loading model, as model hparams might need it
        if self.data_module is None:
            self.data_module = UnifiedDataModule(self.config)
            self.data_module.setup() # This is crucial

        model_path = None
        if checkpoint_callback and hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path:
            model_path = checkpoint_callback.best_model_path
        else:
            checkpoint_files = [f for f in os.listdir(self.config.MODEL_SAVE_DIR) if f.endswith('.ckpt')]
            if checkpoint_files:
                model_path = os.path.join(self.config.MODEL_SAVE_DIR, sorted(checkpoint_files)[-1]) # Simplistic: take latest
        
        if model_path and os.path.exists(model_path):
            print(f"Loading base classifier model from: {model_path}")
            self.model = UnifiedModel.load_from_checkpoint(
                model_path,
                config=self.config, # Pass full config object
                # These hparams are expected by UnifiedModel if it was saved with them
                num_labels=self.data_module.num_labels, 
                label2id=self.data_module.label2id,
                id2label=self.data_module.id2label,
                class_weights=self.data_module.class_weights, # Pass weights if used
                # weights_only=True # If you only want weights, not optimizer states etc.
                                     # But for attention extraction, full model is fine.
            )
            print("Base classifier model loaded successfully!")
        else:
            print(f"Warning: No model checkpoint found at {self.config.MODEL_SAVE_DIR} or specified path. Cannot load model.")
            # Depending on workflow, this might be critical
            if self.config.STAGE_ATTENTION_EXTRACTION or self.config.STAGE_OE_EXTRACTION:
                 raise FileNotFoundError("Cannot proceed without a base model.")


    def _load_best_model(self, checkpoint_callback):
        if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
            print(f"Loading best base classifier model: {checkpoint_callback.best_model_path}")
            # Ensure data_module is set up for hparams
            if self.data_module is None: self.data_module = UnifiedDataModule(self.config); self.data_module.setup()

            self.model = UnifiedModel.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                config=self.config,
                num_labels=self.data_module.num_labels,
                label2id=self.data_module.label2id,
                id2label=self.data_module.id2label,
                class_weights=self.data_module.class_weights
            )
            print("Best base classifier model loaded successfully!")
        else:
            print("Warning: Best base model path not found. Using current model state.")

    def _load_attention_results(self) -> Optional[pd.DataFrame]:
        attention_file = os.path.join(self.config.ATTENTION_DATA_DIR, "df_with_attention.csv")
        if os.path.exists(attention_file):
            print(f"Loading attention results from: {attention_file}")
            df = pd.read_csv(attention_file)
            if 'top_attention_words' in df.columns:
                df['top_attention_words'] = df['top_attention_words'].apply(safe_literal_eval)
            return df
        print(f"Attention results file not found: {attention_file}")
        return None

    def _load_final_metrics_and_features(self) -> Tuple[Optional[pd.DataFrame], Optional[List[np.ndarray]]]:
        metrics_file = os.path.join(self.config.ATTENTION_DATA_DIR, "df_with_all_metrics.csv")
        features_file = os.path.join(self.config.ATTENTION_DATA_DIR, "extracted_features.npy")
        
        df_metrics, features_arr = None, None
        if os.path.exists(metrics_file):
            print(f"Loading final metrics DF from: {metrics_file}")
            df_metrics = pd.read_csv(metrics_file)
            if 'top_attention_words' in df_metrics.columns:
                df_metrics['top_attention_words'] = df_metrics['top_attention_words'].apply(safe_literal_eval)
        else:
            print(f"Metrics DF file not found: {metrics_file}")

        if os.path.exists(features_file):
            print(f"Loading extracted features from: {features_file}")
            features_arr = np.load(features_file, allow_pickle=True).tolist() # allow_pickle if saved as object array
        else:
            print(f"Extracted features file not found: {features_file}. Features will be None.")
        
        return df_metrics, features_arr


    def _print_attention_samples(self, df: pd.DataFrame, num_samples: int = 3):
        if df is None or df.empty: print("No data to sample for attention."); return
        print(f"\n--- Attention Analysis Samples (Max {num_samples}) ---")
        sample_df = df.sample(min(num_samples, len(df)))
        for i, row in sample_df.iterrows():
            print("-" * 30)
            print(f"Original: {str(row[self.config.TEXT_COLUMN])[:100]}...")
            print(f"Top Words: {row['top_attention_words']}")
            print(f"Masked: {str(row[self.config.TEXT_COLUMN_IN_OE_FILES])[:100]}...")

    def _print_final_summary(self):
        print("\n" + "="*50 + "\nPIPELINE SUMMARY\n" + "="*50)
        print(f"Output directory: {self.config.OUTPUT_DIR}")
        # ... (add more summary details if needed) ...
        print("\nGenerated Files (Examples):")
        max_files_to_list = 15
        count = 0
        for root, _, files in os.walk(self.config.OUTPUT_DIR):
            if count >= max_files_to_list: break
            for file in files:
                if count >= max_files_to_list: break
                if file.endswith(('.csv', '.png', '.json', '.txt', '.pt', '.ckpt')):
                    print(f"  - {os.path.join(root, file)}")
                    count +=1
            if count >= max_files_to_list and root == self.config.OUTPUT_DIR : # list at least top level
                print("  ... (many more files generated)")


# === 메인 함수 ===
def main():
    parser = argparse.ArgumentParser(description="Unified OE Extraction and OSR Experimentation Pipeline")
    
    # OE Extraction Args (subset, more are in Config)
    # Fix: Change type=str to type=float for attention_percent and type=int for top_words
    parser.add_argument('--attention_percent', type=float, default=Config.ATTENTION_TOP_PERCENT)
    parser.add_argument('--top_words', type=int, default=Config.MIN_TOP_WORDS)
    parser.add_argument('--data_path', type=str, default=Config.ORIGINAL_DATA_PATH)
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR)
    parser.add_argument('--oe_model_name', type=str, default=Config.MODEL_NAME, help="Model for base classifier (OE extraction)")
    parser.add_argument('--oe_epochs', type=int, default=Config.NUM_TRAIN_EPOCHS)
    
    # OSR Experiment Args (subset)
    parser.add_argument('--osr_model_type', type=str, default=Config.OSR_MODEL_TYPE, help="Model for OSR experiments")
    parser.add_argument('--osr_epochs', type=int, default=Config.OSR_NUM_EPOCHS)
    parser.add_argument('--ood_data_path_osr', type=str, default=Config.OOD_SYSLOG_UNKNOWN_PATH_OSR)

    # Stage control
    parser.add_argument('--skip_base_training', action='store_true')
    parser.add_argument('--skip_attention_extraction', action='store_true')
    parser.add_argument('--skip_oe_extraction', action='store_true')
    parser.add_argument('--skip_oe_visualization', action='store_true')
    parser.add_argument('--skip_osr_experiments', action='store_true')
    parser.add_argument('--osr_eval_only', action='store_true', help="Run OSR experiments in eval only mode")

    args = parser.parse_args()
    
    # Update Config from args
    Config.ATTENTION_TOP_PERCENT = args.attention_percent
    Config.MIN_TOP_WORDS = args.top_words
    Config.ORIGINAL_DATA_PATH = args.data_path
    Config.OUTPUT_DIR = args.output_dir
    Config.MODEL_NAME = args.oe_model_name
    Config.NUM_TRAIN_EPOCHS = args.oe_epochs
    
    Config.OSR_MODEL_TYPE = args.osr_model_type
    Config.OSR_NUM_EPOCHS = args.osr_epochs
    Config.OOD_SYSLOG_UNKNOWN_PATH_OSR = args.ood_data_path_osr
    Config.OSR_EVAL_ONLY = args.osr_eval_only # Set from args

    Config.STAGE_MODEL_TRAINING = not args.skip_base_training
    Config.STAGE_ATTENTION_EXTRACTION = not args.skip_attention_extraction
    Config.STAGE_OE_EXTRACTION = not args.skip_oe_extraction
    Config.STAGE_VISUALIZATION = not args.skip_oe_visualization
    Config.STAGE_OSR_EXPERIMENTS = not args.skip_osr_experiments
    
    # Update derived paths in Config based on new OUTPUT_DIR if changed
    Config.MODEL_SAVE_DIR = os.path.join(Config.OUTPUT_DIR, "base_classifier_model")
    Config.LOG_DIR = os.path.join(Config.OUTPUT_DIR, "lightning_logs")
    Config.CONFUSION_MATRIX_DIR = os.path.join(Config.LOG_DIR, "confusion_matrices")
    Config.VIS_DIR = os.path.join(Config.OUTPUT_DIR, "oe_extraction_visualizations")
    Config.OE_DATA_DIR = os.path.join(Config.OUTPUT_DIR, "extracted_oe_datasets")
    Config.ATTENTION_DATA_DIR = os.path.join(Config.OUTPUT_DIR, "attention_analysis")
    Config.OSR_EXPERIMENT_DIR = os.path.join(Config.OUTPUT_DIR, "osr_experiments")
    Config.OSR_MODEL_DIR = os.path.join(Config.OSR_EXPERIMENT_DIR, "models")
    Config.OSR_RESULT_DIR = os.path.join(Config.OSR_EXPERIMENT_DIR, "results")
    Config.DATA_DIR_EXTERNAL_HF = os.path.join(Config.OUTPUT_DIR, 'data_external_hf') 
    Config.CACHE_DIR_HF = os.path.join(Config.DATA_DIR_EXTERNAL_HF, "hf_cache")

    print(f"--- Unified OE/OSR Pipeline ---")
    print(f"Output Dir: {Config.OUTPUT_DIR}")
    
    pipeline = UnifiedOEExtractor(Config) # Config object passed
    pipeline.run_full_pipeline()
    
if __name__ == '__main__':
    main()
'''    
python oe2.py --attention_percent 0.01 --top_words 1 --output_dir unified_oe_osr_results0.01_1
python oe2.py --attention_percent 0.01 --top_words 2 --output_dir unified_oe_osr_results0.01_2
python oe2.py --attention_percent 0.01 --top_words 3 --output_dir unified_oe_osr_results0.01_3
python oe2.py --attention_percent 0.05 --top_words 1 --output_dir unified_oe_osr_results0.05_1
python oe2.py --attention_percent 0.05 --top_words 2 --output_dir unified_oe_osr_results0.05_2
python oe2.py --attention_percent 0.05 --top_words 3 --output_dir unified_oe_osr_results0.05_3
python oe2.py --attention_percent 0.1 --top_words 1 --output_dir unified_oe_osr_results0.1_1
python oe2.py --attention_percent 0.1 --top_words 2 --output_dir unified_oe_osr_results0.1_2
python oe2.py --attention_percent 0.1 --top_words 3 --output_dir unified_oe_osr_results0.1_3
python oe2.py --attention_percent 0.2 --top_words 1 --output_dir unified_oe_osr_results0.2_1
python oe2.py --attention_percent 0.2 --top_words 2 --output_dir unified_oe_osr_results0.2_2
python oe2.py --attention_percent 0.2 --top_words 3 --output_dir unified_oe_osr_results0.2_3
python oe2.py --attention_percent 0.3 --top_words 1 --output_dir unified_oe_osr_results0.3_1
python oe2.py --attention_percent 0.3 --top_words 2 --output_dir unified_oe_osr_results0.3_2
python oe2.py --attention_percent 0.3 --top_words 3 --output_dir unified_oe_osr_results0.3_3

'''