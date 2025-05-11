# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
# from sklearn.preprocessing import LabelEncoder # 사용되지 않음
# from sklearn.metrics import confusion_matrix # 사용되지 않음
from sklearn.manifold import TSNE
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping # TQDMProgressBar 제거
from pytorch_lightning.loggers import CSVLogger # TensorBoardLogger 제거

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False
    print("Warning: Seaborn not installed.")
import pandas as pd
# import json # 사용되지 않음
# import argparse # main 함수에서 직접 설정값 사용
# from datetime import datetime # 사용되지 않음
from typing import List, Dict, Tuple, Optional # Any, Union 제거
from tqdm.auto import tqdm
import gc
import math
from scipy.stats import entropy
import random
import re
import ast # TOP_WORDS_COLUMN 파싱용

# NLTK 임포트 추가
import nltk
from nltk.tokenize import word_tokenize

# NLTK 다운로드 (필요 시)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- 설정값 ---
# 이전 oe_extractor.py (02_train_id_model_and_mask.py 역할)에서 생성된 파일들
ORIGINAL_DATA_PATH_FOR_TRAINING = 'log_all_filtered.csv' # ID 모델 학습용 데이터
MASKED_DATA_INPUT_PATH = 'log_all_critical_filtered_attention_masked_for_oe.csv'  # oe_extractor.py의 출력 파일 (중요 단어 마스크됨)

# oe_extractor3.py의 주요 입력 및 출력 경로
TEXT_COLUMN = 'text'
CLASS_COLUMN = 'class' # ID 모델 학습 시 사용
MASKED_TEXT_COLUMN = 'masked_text_attention'
TOP_WORDS_COLUMN = 'top_attention_words'
EXCLUDE_CLASS_FOR_TRAINING = "unknown" # ID 모델 학습 시 제외

# OE 필터링 지표 및 설정
METRIC_SETTINGS = {
    'removed_avg_attention': {'percentile': 90, 'mode': 'higher'},
    'top_k_avg_attention': {'percentile': 80, 'mode': 'higher'},
    'max_attention': {'percentile': 85, 'mode': 'higher'},
    'attention_entropy': {'percentile': 25, 'mode': 'lower'}
}
SEQUENTIAL_FILTERING_SEQUENCE = [ # 순차 필터링 순서 및 설정
    ('removed_avg_attention', {'percentile': 90, 'mode': 'higher'}),
    ('attention_entropy', {'percentile': 30, 'mode': 'lower'})
]

# MSP 기반 필터링 설정
MSP_DIFFERENCE_THRESHOLD = 0.1

# 출력 경로 설정
OUTPUT_DIR_BASE = 'oe_extraction_results_v3' # 버전 명시
ID_MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR_BASE, "trained_id_model") # ID 모델 저장 경로
LOG_DIR_ID_MODEL = os.path.join(OUTPUT_DIR_BASE, "lightning_logs_id_model")
CONFUSION_MATRIX_DIR_ID_MODEL = os.path.join(LOG_DIR_ID_MODEL, "confusion_matrices")
VIS_DIR = os.path.join(OUTPUT_DIR_BASE, "visualizations")
OE_DATA_DIR = os.path.join(OUTPUT_DIR_BASE, "extracted_oe_datasets")
OE_DATA_MSP_FILTERED_DIR = os.path.join(OUTPUT_DIR_BASE, "extracted_oe_datasets_msp_filtered") # MSP 필터링 결과 저장

# 모델 및 학습 설정 (ID 모델 학습용)
MODEL_NAME = "roberta-base"
MAX_LENGTH = 256
BATCH_SIZE = 64
NUM_TRAIN_EPOCHS_ID_MODEL = 15 # ID 모델 학습 에포크
LEARNING_RATE_ID_MODEL = 2e-5
MIN_SAMPLES_PER_CLASS_FOR_TRAIN_VAL = 3
ACCELERATOR = "auto"
DEVICES = "auto"
PRECISION = "16-mixed" if torch.cuda.is_available() and ACCELERATOR != "cpu" else "32-true"
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1
LOG_EVERY_N_STEPS = 50
GRADIENT_CLIP_VAL = 1.0
USE_WEIGHTED_LOSS = True
USE_LR_SCHEDULER = True
RANDOM_STATE = 42
TOP_K_ATTENTION = 3 # top_k_avg_attention 계산 시 사용

# 디렉토리 생성
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
os.makedirs(ID_MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR_ID_MODEL, exist_ok=True)
os.makedirs(CONFUSION_MATRIX_DIR_ID_MODEL, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(OE_DATA_DIR, exist_ok=True)
os.makedirs(OE_DATA_MSP_FILTERED_DIR, exist_ok=True)

# --- 도우미 함수 ---
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed set to: {seed}")

def preprocess_text_for_roberta(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_nltk(text): # 사용되지 않지만 유틸리티로 남겨둠
    if not text:
        return []
    try:
        return word_tokenize(text)
    except Exception:
        return text.split()

def safe_literal_eval(val): # TOP_WORDS_COLUMN 파싱용
    try:
        if isinstance(val, str) and val.strip().startswith('['):
            return ast.literal_eval(val)
        elif isinstance(val, list): # 이미 리스트인 경우
            return val
        else:
            return []
    except Exception:
        # print(f"Warning: Could not parse value: {str(val)[:50]}... Returning empty list.")
        return []

# --- 데이터 클래스 (TextDatasetForInference: 레이블 없이 텍스트만 처리) ---
class TextDatasetForInference(TorchDataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.texts = [str(t) if pd.notna(t) else "" for t in texts]
        print(f"Tokenizing {len(self.texts)} texts for inference...")
        # padding='max_length' 대신 padding=True 사용하고 DataCollatorWithPadding 활용
        self.encodings = tokenizer(self.texts, max_length=max_length, padding=False, truncation=True)
        print("Tokenization complete.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

# --- PyTorch Lightning DataModule (ID 모델 학습용) ---
class LogDataModuleForKnownClasses(pl.LightningDataModule):
    def __init__(self, file_path, text_col, class_col, exclude_class, model_name, batch_size, max_length, min_samples_per_class=3, num_workers=1, random_state=42, use_weighted_loss=False):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.df_full = None
        self.df_known_for_train_val = None
        self.train_df_final = None
        self.val_df_final = None
        self.label2id = None
        self.id2label = None
        self.num_labels = None
        self.tokenized_train_val_datasets = None
        self.class_weights = None

    def prepare_data(self):
        pass # 데이터 다운로드 등 필요 시 사용

    def setup(self, stage=None):
        if self.df_full is None:
            print(f"DataModule: Loading data from {self.hparams.file_path}")
            self.df_full = pd.read_csv(self.hparams.file_path)
            required_cols = [self.hparams.text_col, self.hparams.class_col]
            if not all(col in self.df_full.columns for col in required_cols):
                raise ValueError(f"CSV missing required columns: {required_cols}")

            self.df_full = self.df_full.dropna(subset=[self.hparams.class_col])
            self.df_full[self.hparams.class_col] = self.df_full[self.hparams.class_col].astype(str).str.lower()
            exclude_class_lower = self.hparams.exclude_class.lower() if self.hparams.exclude_class else None

            print(f"DataModule: Excluding class '{self.hparams.exclude_class}' for training/validation.")
            df_known = self.df_full[self.df_full[self.hparams.class_col] != exclude_class_lower].copy()
            print(f"DataModule: Data size after excluding '{self.hparams.exclude_class}': {len(df_known)}")

            if df_known.empty:
                raise ValueError(f"No data left after excluding class '{self.hparams.exclude_class}'. Check your data and exclude_class setting.")

            known_classes_str = sorted(df_known[self.hparams.class_col].unique())
            self.label2id = {label: i for i, label in enumerate(known_classes_str)}
            self.id2label = {i: label for label, i in self.label2id.items()}
            self.num_labels = len(known_classes_str)
            print(f"\nDataModule - Label mapping complete (Known Classes): {self.num_labels} classes")
            print(f"Known Class Label to ID mapping: {self.label2id}")

            df_known['label'] = df_known[self.hparams.class_col].map(self.label2id)
            df_known = df_known.dropna(subset=['label']) # 매핑 실패한 경우 제거
            df_known['label'] = df_known['label'].astype(int)

            print(f"\nDataModule - Filtering known classes (min {self.hparams.min_samples_per_class} samples per class)...")
            label_counts_known = df_known['label'].value_counts()
            labels_to_keep = label_counts_known[label_counts_known >= self.hparams.min_samples_per_class].index
            self.df_known_for_train_val = df_known[df_known['label'].isin(labels_to_keep)].copy()
            print(f"DataModule - Data for training/validation after filtering: {len(self.df_known_for_train_val)} rows")

            if len(self.df_known_for_train_val) == 0:
                raise ValueError("No data left for training/validation after filtering by min_samples_per_class.")

            print("\n--- Class distribution for training/validation ---")
            print(self.df_known_for_train_val['label'].map(self.id2label).value_counts())

            if self.hparams.use_weighted_loss:
                labels_for_weights = self.df_known_for_train_val['label'].values
                unique_labels_in_train_val = np.unique(labels_for_weights)
                try:
                    class_weights_array = compute_class_weight('balanced', classes=unique_labels_in_train_val, y=labels_for_weights)
                    self.class_weights = torch.ones(self.num_labels) # 모든 클래스에 대해 초기화
                    for i, label_idx in enumerate(unique_labels_in_train_val):
                        if label_idx < self.num_labels: # 유효한 레이블 인덱스인지 확인
                            self.class_weights[label_idx] = class_weights_array[i]
                    print(f"\nDataModule - Calculated class weights: {self.class_weights}")
                except ValueError as e:
                    print(f"Warning: Error calculating class weights: {e}. Proceeding without weights.")
                    self.hparams.use_weighted_loss = False
                    self.class_weights = None

            print("\nDataModule - Splitting data into training and validation sets...")
            try:
                self.train_df_final, self.val_df_final = train_test_split(
                    self.df_known_for_train_val, test_size=0.2,
                    random_state=self.hparams.random_state, stratify=self.df_known_for_train_val['label']
                )
            except ValueError: # Stratify 실패 시 (예: 클래스별 샘플 부족)
                print("Warning: Stratified split failed. Falling back to non-stratified split.")
                self.train_df_final, self.val_df_final = train_test_split(
                    self.df_known_for_train_val, test_size=0.2, random_state=self.hparams.random_state
                )
            print(f"DataModule - Final training set: {len(self.train_df_final)}, Final validation set: {len(self.val_df_final)}")

            raw_train_val_datasets = DatasetDict({
                'train': Dataset.from_pandas(self.train_df_final),
                'validation': Dataset.from_pandas(self.val_df_final)
            })

            def tokenize_func(examples):
                return self.tokenizer(
                    [preprocess_text_for_roberta(text) for text in examples[self.hparams.text_col]],
                    truncation=True, padding=False, max_length=self.hparams.max_length # max_length 사용
                )

            print("\nDataModule - Tokenizing training/validation datasets...")
            self.tokenized_train_val_datasets = raw_train_val_datasets.map(
                tokenize_func, batched=True,
                num_proc=max(1, self.hparams.num_workers // 2),
                remove_columns=[col for col in raw_train_val_datasets['train'].column_names if col != 'label'] # label 컬럼만 남김
            )
            self.tokenized_train_val_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            print("DataModule - Tokenization complete.")

    def train_dataloader(self):
        if self.tokenized_train_val_datasets is None: self.setup()
        return DataLoader(
            self.tokenized_train_val_datasets['train'],
            batch_size=self.hparams.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0
        )

    def val_dataloader(self):
        if self.tokenized_train_val_datasets is None: self.setup()
        return DataLoader(
            self.tokenized_train_val_datasets['validation'],
            batch_size=self.hparams.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0
        )

    def get_full_dataframe_for_attention_processing(self):
        # 이 함수는 setup()에서 로드된 df_full을 반환하거나,
        # MASKED_DATA_INPUT_PATH에서 데이터를 로드하여 반환할 수 있습니다.
        # 여기서는 MASKED_DATA_INPUT_PATH를 직접 로드하는 것으로 가정합니다.
        print(f"Loading data for attention processing from: {MASKED_DATA_INPUT_PATH}")
        try:
            df = pd.read_csv(MASKED_DATA_INPUT_PATH)
            # TOP_WORDS_COLUMN 파싱
            if TOP_WORDS_COLUMN in df.columns:
                df[TOP_WORDS_COLUMN] = df[TOP_WORDS_COLUMN].apply(safe_literal_eval)
            else:
                print(f"Warning: '{TOP_WORDS_COLUMN}' not found in {MASKED_DATA_INPUT_PATH}. It will be created as empty lists.")
                df[TOP_WORDS_COLUMN] = [[] for _ in range(len(df))]

            # MASKED_TEXT_COLUMN이 없는 경우 (오류 또는 이전 단계 문제)
            if MASKED_TEXT_COLUMN not in df.columns:
                print(f"Warning: '{MASKED_TEXT_COLUMN}' not found in {MASKED_DATA_INPUT_PATH}. Using '{TEXT_COLUMN}' instead for masked text.")
                if TEXT_COLUMN in df.columns:
                    df[MASKED_TEXT_COLUMN] = df[TEXT_COLUMN]
                else:
                    raise ValueError(f"Neither '{MASKED_TEXT_COLUMN}' nor '{TEXT_COLUMN}' found in {MASKED_DATA_INPUT_PATH}")
            return df.reset_index(drop=True)
        except FileNotFoundError:
            print(f"Error: {MASKED_DATA_INPUT_PATH} not found. Cannot proceed with attention processing.")
            return pd.DataFrame()


# --- PyTorch Lightning Module (ID 모델 학습용) ---
class LogClassifierPL(pl.LightningModule):
    def __init__(self, model_name, num_labels, label2id, id2label, confusion_matrix_dir, learning_rate=2e-5,
                 use_weighted_loss=False, class_weights=None, use_lr_scheduler=False, warmup_steps=0):
        super().__init__()
        self.save_hyperparameters("model_name", "num_labels", "label2id", "id2label", "confusion_matrix_dir", "learning_rate",
                                  "use_weighted_loss", "class_weights", "use_lr_scheduler", "warmup_steps") # class_weights도 저장
        print(f"LightningModule: Initializing model {self.hparams.model_name} for {self.hparams.num_labels} known classes.")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hparams.model_name,
            num_labels=self.hparams.num_labels,
            label2id=self.hparams.label2id,
            id2label=self.hparams.id2label,
            ignore_mismatched_sizes=True, # Fine-tuning 시 유용
            output_attentions=True, # 어텐션 추출 활성화
            output_hidden_states=True # 특징 추출 활성화
        )

        if self.hparams.use_weighted_loss and self.hparams.class_weights is not None:
            # class_weights가 텐서가 아닐 경우 텐서로 변환
            weights = self.hparams.class_weights if isinstance(self.hparams.class_weights, torch.Tensor) else torch.tensor(self.hparams.class_weights, dtype=torch.float)
            self.loss_fn = nn.CrossEntropyLoss(weight=weights)
            print("LightningModule: Using Weighted CrossEntropyLoss")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            print("LightningModule: Using Standard CrossEntropyLoss")

        metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_labels),
            'f1_weighted': torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_labels, average='weighted'),
            'f1_macro': torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_labels, average='macro')
        })
        cm_metric = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_labels)
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.val_cm = cm_metric.clone()

    def setup(self, stage=None):
        if self.hparams.use_weighted_loss and isinstance(self.loss_fn, nn.CrossEntropyLoss) and self.loss_fn.weight is not None:
            self.loss_fn.weight = self.loss_fn.weight.to(self.device) # 가중치를 현재 디바이스로 이동
            print(f"LightningModule: Moved class weights to device {self.device}")

    def forward(self, batch_input_ids, batch_attention_mask, output_features=False, output_attentions_flag=False): # 인자 이름 변경
        # 모델의 forward를 직접 호출
        outputs = self.model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            output_hidden_states=output_features,
            output_attentions=output_attentions_flag # 인자 이름과 내부 플래그 이름 일치
        )
        return outputs # 전체 outputs 객체 반환

    def _common_step(self, batch, batch_idx):
        # DataModule에서 'label'로 전달되므로 'labels'로 변경
        # 또는 DataCollatorWithPadding이 'labels'로 변경했을 수 있음
        if 'labels' in batch:
            labels = batch.pop('labels') # 'labels' 키가 있으면 사용
        elif 'label' in batch:
            labels = batch.pop('label')  # 'label' 키가 있으면 사용
        else:
            raise KeyError("Batch does not contain 'label' or 'labels' key for ground truth.")

        # batch에는 input_ids, attention_mask만 남기거나, 모델이 알아서 처리하도록 함
        # 명시적으로 필요한 것만 전달하는 것이 안전
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        metrics_results = self.train_metrics(preds, labels)
        self.log_dict(metrics_results, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        metrics_results = self.val_metrics(preds, labels)
        self.val_cm.update(preds, labels)
        self.log_dict(metrics_results, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        try:
            val_cm_computed = self.val_cm.compute()
            print("\nValidation Confusion Matrix (Known Classes):")
            known_class_names = list(self.hparams.id2label.values())
            cm_df = pd.DataFrame(val_cm_computed.cpu().numpy(), index=known_class_names, columns=known_class_names)
            print(cm_df)
            cm_filename = os.path.join(self.hparams.confusion_matrix_dir, f"validation_cm_epoch_{self.current_epoch}.csv")
            cm_df.to_csv(cm_filename)
            print(f"Validation Confusion Matrix saved to: {cm_filename}")
        except Exception as e:
            print(f"Error computing/printing/saving validation confusion matrix: {e}")
        finally:
            self.val_cm.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.use_lr_scheduler:
            if self.trainer and hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches > 0:
                 num_training_steps = self.trainer.estimated_stepping_batches
                 print(f"LightningModule: Estimated training steps for LR scheduler: {num_training_steps}")
                 scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=num_training_steps)
                 return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
            else:
                print("Warning: Could not estimate training steps for LR scheduler. Using optimizer only.")
                return optimizer
        else:
            return optimizer

# --- 단어 어텐션 스코어 추출 함수 ---
@torch.no_grad()
def get_word_attention_scores_pl(texts: List[str], model_pl: LogClassifierPL, tokenizer, device, layer_idx=-1, head_idx=None, max_length_tokenizer=512, batch_size_attn=16):
    model_pl.eval() # 모델을 평가 모드로 설정
    # model_pl.to(device) # 이미 main 함수에서 device로 이동됨

    word_attention_scores_batch = [{} for _ in range(len(texts))] # 최종 결과 저장용

    valid_texts_with_indices = []
    for i, text in enumerate(texts):
        if isinstance(text, str) and text.strip():
            valid_texts_with_indices.append((i, text))

    if not valid_texts_with_indices:
        return word_attention_scores_batch

    for i in range(0, len(valid_texts_with_indices), batch_size_attn):
        current_batch_tuples = valid_texts_with_indices[i:i+batch_size_attn]
        batch_original_indices = [tpl[0] for tpl in current_batch_tuples]
        batch_texts_to_process = [tpl[1] for tpl in current_batch_tuples]

        inputs = tokenizer(
            batch_texts_to_process,
            return_tensors="pt",
            padding=True, # DataCollator 대신 직접 패딩
            truncation=True,
            max_length=max_length_tokenizer,
            return_offsets_mapping=True # 오프셋 매핑 사용
        )
        offset_mappings_batch = inputs.pop('offset_mapping').cpu().numpy()
        input_ids_batch_np = inputs['input_ids'].cpu().numpy() # NumPy 배열로 변환
        inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

        outputs = model_pl.forward(inputs_on_device['input_ids'], inputs_on_device['attention_mask'], output_attentions_flag=True)
        attentions_layer_batch = outputs.attentions[layer_idx].cpu().numpy() # (batch_size, num_heads, seq_len, seq_len)

        for batch_item_idx in range(len(batch_texts_to_process)):
            original_text_idx = batch_original_indices[batch_item_idx]
            original_text_sample = batch_texts_to_process[batch_item_idx] # 전처리 전 원본 텍스트
            attention_sample = attentions_layer_batch[batch_item_idx] # (num_heads, seq_len, seq_len)
            offset_mapping_sample = offset_mappings_batch[batch_item_idx] # (seq_len, 2)
            input_ids_sample = input_ids_batch_np[batch_item_idx] # (seq_len,)

            if head_idx is not None:
                head_attention = attention_sample[head_idx] # (seq_len, seq_len)
            else:
                head_attention = np.mean(attention_sample, axis=0) # (seq_len, seq_len), 헤드 평균

            # CLS 토큰(인덱스 0)이 다른 토큰에 주는 어텐션 사용
            token_attentions = head_attention[0, :] # (seq_len,)

            word_scores_for_sample = {}
            current_word_tokens_indices = []
            last_word_end_offset = 0

            for token_idx, (offset, token_id) in enumerate(zip(offset_mapping_sample, input_ids_sample)):
                if offset[0] == offset[1] or token_id in tokenizer.all_special_ids: # 특수 토큰 또는 패딩 무시
                    if current_word_tokens_indices: # 이전 단어 처리
                        start_char = offset_mapping_sample[current_word_tokens_indices[0]][0]
                        end_char = offset_mapping_sample[current_word_tokens_indices[-1]][1]
                        word = original_text_sample[start_char:end_char]
                        avg_score = np.mean(token_attentions[current_word_tokens_indices])
                        if word.strip():
                            # 동일 단어 여러 번 등장 시 높은 점수 유지 또는 평균 (여기서는 평균 사용)
                            if word.strip().lower() in word_scores_for_sample:
                                word_scores_for_sample[word.strip().lower()].append(avg_score)
                            else:
                                word_scores_for_sample[word.strip().lower()] = [avg_score]
                        current_word_tokens_indices = []
                    continue

                # RoBERTa는 Ġ (U+0120)로 단어 시작을 표시
                token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
                if token_str.startswith('Ġ') or not current_word_tokens_indices:
                    if current_word_tokens_indices: # 이전 단어 처리
                        start_char = offset_mapping_sample[current_word_tokens_indices[0]][0]
                        end_char = offset_mapping_sample[current_word_tokens_indices[-1]][1]
                        word = original_text_sample[start_char:end_char]
                        avg_score = np.mean(token_attentions[current_word_tokens_indices])
                        if word.strip():
                            if word.strip().lower() in word_scores_for_sample:
                                word_scores_for_sample[word.strip().lower()].append(avg_score)
                            else:
                                word_scores_for_sample[word.strip().lower()] = [avg_score]
                        current_word_tokens_indices = []
                    current_word_tokens_indices.append(token_idx)
                else: # 서브워드
                    current_word_tokens_indices.append(token_idx)

            # 마지막 단어 처리
            if current_word_tokens_indices:
                start_char = offset_mapping_sample[current_word_tokens_indices[0]][0]
                end_char = offset_mapping_sample[current_word_tokens_indices[-1]][1]
                word = original_text_sample[start_char:end_char]
                avg_score = np.mean(token_attentions[current_word_tokens_indices])
                if word.strip():
                    if word.strip().lower() in word_scores_for_sample:
                        word_scores_for_sample[word.strip().lower()].append(avg_score)
                    else:
                        word_scores_for_sample[word.strip().lower()] = [avg_score]

            # 단어별 최종 점수 (평균)
            final_word_scores = {word: np.mean(scores) for word, scores in word_scores_for_sample.items()}
            word_attention_scores_batch[original_text_idx] = final_word_scores

    return word_attention_scores_batch


# --- 어텐션/특징 추출 및 지표 계산 함수 ---
@torch.no_grad()
def extract_attention_and_features(model_pl: LogClassifierPL, dataloader: DataLoader, device: torch.device, tokenizer, top_k_val: int = 3) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    model_pl.eval()
    # model_pl.to(device) # 이미 main에서 처리

    attention_metrics_list = []
    features_list_all = []

    print("Extracting attention scores and features for metric calculation...")
    for batch in tqdm(dataloader, desc="Extracting Metrics"):
        batch_input_ids = batch['input_ids'].to(device)
        batch_attention_mask = batch['attention_mask'].to(device)

        outputs = model_pl.forward(batch_input_ids, batch_attention_mask, output_features=True, output_attentions_flag=True)

        attentions_batch = outputs.attentions[-1].cpu().numpy()  # (batch_size, num_heads, seq_len, seq_len)
        # CLS 토큰의 특징 벡터 사용 (RoBERTa의 경우 보통 첫 번째 토큰)
        features_batch = outputs.hidden_states[-1][:, 0, :].cpu().numpy() # (batch_size, hidden_size)
        features_list_all.extend(list(features_batch))

        for i in range(len(batch['input_ids'])):
            attn_sample_all_heads = attentions_batch[i]  # (num_heads, seq_len, seq_len)
            token_ids_sample = batch['input_ids'][i].cpu().numpy() # (seq_len,)

            valid_token_indices = np.where(
                (token_ids_sample != tokenizer.pad_token_id) &
                (token_ids_sample != tokenizer.cls_token_id) &
                (token_ids_sample != tokenizer.sep_token_id)
            )[0]

            if len(valid_token_indices) == 0:
                attention_metrics_list.append({'max_attention': 0, 'top_k_avg_attention': 0, 'attention_entropy': 0})
                continue

            # CLS 토큰(인덱스 0)이 다른 토큰에 주는 어텐션 (모든 헤드 평균)
            cls_attentions_to_others = np.mean(attn_sample_all_heads[:, 0, :], axis=0) # (seq_len,)
            valid_cls_attentions = cls_attentions_to_others[valid_token_indices]

            max_attention_val = np.max(valid_cls_attentions) if len(valid_cls_attentions) > 0 else 0

            k_for_avg = min(top_k_val, len(valid_cls_attentions))
            top_k_avg_attention_val = np.mean(np.sort(valid_cls_attentions)[-k_for_avg:]) if k_for_avg > 0 else 0

            # 엔트로피 계산 시 확률 분포로 정규화
            if len(valid_cls_attentions) > 0:
                attention_probs = F.softmax(torch.tensor(valid_cls_attentions, dtype=torch.float32), dim=0).numpy()
                attention_entropy_val = entropy(attention_probs)
            else:
                attention_entropy_val = 0

            attention_metrics_list.append({
                'max_attention': max_attention_val,
                'top_k_avg_attention': top_k_avg_attention_val,
                'attention_entropy': attention_entropy_val
            })

    df_results = pd.DataFrame(attention_metrics_list)
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
    return df_results, features_list_all


# --- MSP 계산 함수 ---
@torch.no_grad()
def calculate_msp_for_texts(texts: List[str], model_pl: LogClassifierPL, tokenizer, device, batch_size_msp=64, max_length_msp=MAX_LENGTH) -> List[float]:
    model_pl.eval()
    # model_pl.to(device) # 이미 main에서 처리

    msp_values = [0.0] * len(texts) # 결과 초기화

    valid_texts_with_indices = []
    for i, text in enumerate(texts):
        if isinstance(text, str) and text.strip() and text.lower() != "__empty_masked__": # 빈 마스크 텍스트 제외
            valid_texts_with_indices.append((i, text))

    if not valid_texts_with_indices:
        return msp_values # 모든 텍스트가 유효하지 않으면 0 리스트 반환

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    for i in tqdm(range(0, len(valid_texts_with_indices), batch_size_msp), desc="Calculating MSP"):
        current_batch_tuples = valid_texts_with_indices[i:i+batch_size_msp]
        batch_original_indices = [tpl[0] for tpl in current_batch_tuples]
        batch_texts_to_process = [tpl[1] for tpl in current_batch_tuples]

        # TextDatasetForInference와 DataLoader를 사용하여 배치 생성
        # TextDatasetForInference는 내부적으로 padding=False로 토큰화하므로, collate_fn에서 패딩
        temp_dataset = TextDatasetForInference(batch_texts_to_process, tokenizer, max_length=max_length_msp)
        temp_dataloader = DataLoader(temp_dataset, batch_size=len(batch_texts_to_process), collate_fn=data_collator)

        for batch_data in temp_dataloader: # 루프는 한 번만 실행됨
            batch_input_ids = batch_data['input_ids'].to(device)
            batch_attention_mask = batch_data['attention_mask'].to(device)

            outputs = model_pl.forward(batch_input_ids, batch_attention_mask)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            max_probs, _ = torch.max(probabilities, dim=1)
            max_probs_cpu = max_probs.cpu().numpy()

            for batch_item_idx, original_idx in enumerate(batch_original_indices):
                msp_values[original_idx] = max_probs_cpu[batch_item_idx]

    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
    return msp_values


# --- 시각화 함수 ---
def plot_metric_distribution(scores: np.ndarray, metric_name: str, title: str, save_path: Optional[str] = None):
    if len(scores) == 0:
        print(f"No scores provided for metric '{metric_name}' to plot distribution.")
        return
    plt.figure(figsize=(10, 6))
    if SNS_AVAILABLE:
        sns.histplot(scores, bins=50, kde=True, stat='density')
    else:
        plt.hist(scores, bins=50, density=True, alpha=0.7, label='Density')
        # KDE 직접 계산 (선택적, scipy 필요)
        # from scipy.stats import gaussian_kde
        # kde = gaussian_kde(scores)
        # x_range = np.linspace(min(scores), max(scores), 200)
        # plt.plot(x_range, kde(x_range), color='red', label='KDE')
        plt.legend()
    plt.title(title)
    plt.xlabel(metric_name)
    plt.ylabel('Density')
    plt.grid(alpha=0.3)
    if save_path: plt.savefig(save_path); print(f"{metric_name} distribution plot saved to: {save_path}"); plt.close()
    else: plt.show(); plt.close()

def plot_tsne(features: np.ndarray, labels: np.ndarray, title: str, save_path: Optional[str] = None,
              highlight_indices: Optional[np.ndarray] = None, highlight_label: str = 'OE Candidate',
              class_names: Optional[Dict[int, str]] = None, seed: int = 42, perplexity: int = 30, n_iter: int = 1000):
    if len(features) == 0: print("No features provided for t-SNE plot."); return

    print(f"Running t-SNE on {features.shape[0]} samples (perplexity={perplexity})...")
    try:
        # Perplexity는 샘플 수보다 작아야 함
        effective_perplexity = min(perplexity, features.shape[0] - 1)
        if effective_perplexity <= 1: # 매우 적은 샘플의 경우
            print(f"Warning: Too few samples ({features.shape[0]}) for reliable t-SNE with perplexity {perplexity}. Using perplexity={effective_perplexity}. Results might be unstable.")
            if features.shape[0] <=1:
                 print("Error: Cannot run t-SNE with 1 or fewer samples. Skipping plot.")
                 return

        tsne = TSNE(n_components=2, random_state=seed, perplexity=effective_perplexity, n_iter=n_iter, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(features)
    except Exception as e:
        print(f"Error running t-SNE: {e}. Skipping plot.")
        return

    df_tsne = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
    df_tsne['label'] = labels
    df_tsne['is_highlighted'] = False
    if highlight_indices is not None and len(highlight_indices) > 0:
        # highlight_indices가 df_tsne의 인덱스 범위 내에 있는지 확인
        valid_highlight_indices = [idx for idx in highlight_indices if idx < len(df_tsne)]
        if len(valid_highlight_indices) < len(highlight_indices):
            print(f"Warning: Some highlight_indices were out of bounds. Using {len(valid_highlight_indices)} valid indices.")
        if valid_highlight_indices:
            df_tsne.loc[valid_highlight_indices, 'is_highlighted'] = True


    plt.figure(figsize=(14, 10)) # 크기 조정
    unique_labels = sorted(df_tsne['label'].unique())
    # 색상 팔레트 개선 (tab20은 최대 20개, 더 많으면 tab20b, tab20c 등 또는 직접 생성)
    if len(unique_labels) <= 20:
        colors = plt.cm.get_cmap('tab20', len(unique_labels))
    else: # 더 많은 클래스에 대한 색상 생성 (예시)
        colors = plt.cm.get_cmap('viridis', len(unique_labels))


    for i, label_val in enumerate(unique_labels):
        subset = df_tsne[(df_tsne['label'] == label_val) & (~df_tsne['is_highlighted'])]
        if subset.empty: continue
        class_name_str = class_names.get(label_val, f'Class {label_val}') if class_names else f'Class {label_val}'
        plt.scatter(subset['tsne1'], subset['tsne2'], color=colors(i), label=class_name_str, alpha=0.6, s=25) # 마커 크기, 투명도 조정

    if highlight_indices is not None and df_tsne['is_highlighted'].any():
        highlight_subset = df_tsne[df_tsne['is_highlighted']]
        if not highlight_subset.empty:
            plt.scatter(highlight_subset['tsne1'], highlight_subset['tsne2'],
                        color='red', marker='x', s=80, label=highlight_label, alpha=0.9, zorder=5) # zorder로 위에 표시

    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    # 범례 위치 및 스타일 개선
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, fancybox=True, shadow=True, fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout(rect=[0, 0, 0.83, 1]) # 범례 공간 확보

    if save_path: plt.savefig(save_path, dpi=300); print(f"t-SNE plot saved to: {save_path}"); plt.close()
    else: plt.show(); plt.close()


# --- 메인 실행 로직 ---
def main():
    set_seed(RANDOM_STATE) # 전역 랜덤 시드 설정

    # --- 1. ID 분류 모델 학습 (필요 시) 또는 로드 ---
    print("--- 1. ID Classification Model Training/Loading ---")
    # DataModule은 ID 모델 학습에만 사용
    id_model_datamodule = LogDataModuleForKnownClasses(
        file_path=ORIGINAL_DATA_PATH_FOR_TRAINING, text_col=TEXT_COLUMN, class_col=CLASS_COLUMN,
        exclude_class=EXCLUDE_CLASS_FOR_TRAINING, model_name=MODEL_NAME, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
        min_samples_per_class=MIN_SAMPLES_PER_CLASS_FOR_TRAIN_VAL, num_workers=NUM_WORKERS,
        random_state=RANDOM_STATE, use_weighted_loss=USE_WEIGHTED_LOSS
    )
    id_model_datamodule.setup() # 데이터 로드, 전처리, 분할, 토큰화

    # 최적 모델 경로 탐색 (가장 최근 또는 best)
    best_id_model_path = None
    if os.path.exists(ID_MODEL_SAVE_DIR) and any(f.endswith(".ckpt") for f in os.listdir(ID_MODEL_SAVE_DIR)):
        checkpoint_files = [os.path.join(ID_MODEL_SAVE_DIR, f) for f in os.listdir(ID_MODEL_SAVE_DIR) if f.endswith(".ckpt")]
        # 'best-model' 포함 파일 우선, 없으면 수정 시간 기준 최신 파일
        best_checkpoints = [f for f in checkpoint_files if 'best-model' in os.path.basename(f)]
        if best_checkpoints:
            best_id_model_path = max(best_checkpoints, key=os.path.getmtime) # 여러 best 중 최신
            print(f"Found best ID model checkpoint: {best_id_model_path}")
        else:
            best_id_model_path = max(checkpoint_files, key=os.path.getmtime)
            print(f"Found latest ID model checkpoint (no 'best-model' prefix): {best_id_model_path}")

    trained_id_model_pl = None
    if best_id_model_path:
        print(f"Loading pre-trained ID model from: {best_id_model_path}")
        try:
            # load_from_checkpoint 시 class_weights는 hparams에서 자동으로 로드됨
            trained_id_model_pl = LogClassifierPL.load_from_checkpoint(
                best_id_model_path,
                # map_location=torch.device("cuda" if ACCELERATOR != "cpu" and torch.cuda.is_available() else "cpu") # 명시적 디바이스 매핑
                # class_weights는 hparams에 저장되어 있어야 함.
            )
            print("Pre-trained ID model loaded successfully.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Training a new model.")
            best_id_model_path = None # 로드 실패 시 새로 학습하도록 플래그 초기화

    if not trained_id_model_pl:
        print("No pre-trained ID model found or loading failed. Training a new ID model...")
        id_model_pl_module = LogClassifierPL(
            model_name=MODEL_NAME, num_labels=id_model_datamodule.num_labels,
            label2id=id_model_datamodule.label2id, id2label=id_model_datamodule.id2label,
            confusion_matrix_dir=CONFUSION_MATRIX_DIR_ID_MODEL,
            learning_rate=LEARNING_RATE_ID_MODEL, use_weighted_loss=USE_WEIGHTED_LOSS,
            class_weights=id_model_datamodule.class_weights, # DataModule에서 계산된 가중치 전달
            use_lr_scheduler=USE_LR_SCHEDULER, warmup_steps=0
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=ID_MODEL_SAVE_DIR, filename=f'best-model-{{epoch:02d}}-{{val_f1_macro:.4f}}',
            save_top_k=1, monitor='val_f1_macro', mode='max'
        )
        early_stopping_callback = EarlyStopping(monitor='val_f1_macro', patience=3, mode='max', verbose=True)
        csv_logger = CSVLogger(save_dir=LOG_DIR_ID_MODEL, name="id_model_training_logs")

        trainer = pl.Trainer(
            max_epochs=NUM_TRAIN_EPOCHS_ID_MODEL, accelerator=ACCELERATOR, devices=DEVICES,
            precision=PRECISION, logger=csv_logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
            log_every_n_steps=LOG_EVERY_N_STEPS, gradient_clip_val=GRADIENT_CLIP_VAL
        )
        trainer.fit(id_model_pl_module, datamodule=id_model_datamodule)
        print("ID model training complete.")
        best_id_model_path = checkpoint_callback.best_model_path
        if best_id_model_path and os.path.exists(best_id_model_path):
            trained_id_model_pl = LogClassifierPL.load_from_checkpoint(best_id_model_path)
            print(f"Newly trained ID model loaded from: {best_id_model_path}")
        else:
            print("Error: Could not load the newly trained model. Using the model from the last epoch.")
            trained_id_model_pl = id_model_pl_module
    
    current_device = torch.device("cuda" if ACCELERATOR != "cpu" and torch.cuda.is_available() else "cpu")
    trained_id_model_pl.to(current_device)
    trained_id_model_pl.eval()
    # trained_id_model_pl.freeze() # 추론 시에는 freeze 필요 없음, 어차피 no_grad 사용

    # --- 2. 어텐션 지표 계산용 데이터 로드 ---
    print(f"\n--- 2. Loading data for attention metric calculation from: {MASKED_DATA_INPUT_PATH} ---")
    # DataModule의 get_full_dataframe_for_attention_processing 사용
    data_for_attention_metrics = id_model_datamodule.get_full_dataframe_for_attention_processing()

    if data_for_attention_metrics.empty:
        print("No data loaded for attention metric calculation. Exiting.")
        return

    # TextDatasetForInference와 DataLoader 준비 (어텐션 지표 및 특징 추출용)
    # MASKED_TEXT_COLUMN의 텍스트를 사용
    texts_for_attn_extraction = data_for_attention_metrics[MASKED_TEXT_COLUMN].tolist()
    # TextDatasetForInference는 레이블이 필요 없음
    attention_metric_dataset = TextDatasetForInference(texts_for_attn_extraction, id_model_datamodule.tokenizer, max_length=MAX_LENGTH)
    attention_metric_dataloader = DataLoader(
        attention_metric_dataset, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=id_model_datamodule.tokenizer) # collate_fn 명시
    )

    # --- 3. 어텐션 기반 지표 및 특징 벡터 추출 ---
    print("\n--- 3. Extracting Attention-based Metrics and Features ---")
    attention_metrics_df, all_features_from_masked_text = extract_attention_and_features(
        trained_id_model_pl,
        attention_metric_dataloader,
        current_device,
        id_model_datamodule.tokenizer,
        top_k_val=TOP_K_ATTENTION
    )
    if len(data_for_attention_metrics) == len(attention_metrics_df):
        data_for_attention_metrics = pd.concat(
            [data_for_attention_metrics.reset_index(drop=True), attention_metrics_df.reset_index(drop=True)],
            axis=1
        )
        print("Attention metrics added to the dataframe.")
    else:
        print(f"Warning: Length mismatch between data ({len(data_for_attention_metrics)}) and attention metrics ({len(attention_metrics_df)}). Metrics not merged.")

    # --- 4. 제거된 단어의 평균 어텐션 계산 ---
    print("\n--- 4. Calculating Average Attention of Removed Words ---")
    if TOP_WORDS_COLUMN in data_for_attention_metrics.columns and TEXT_COLUMN in data_for_attention_metrics.columns:
        print("Calculating average attention scores for removed words...")
        original_texts_for_removed_attn = data_for_attention_metrics[TEXT_COLUMN].tolist()

        # get_word_attention_scores_pl 함수는 이미 배치 처리를 내장하고 있음
        word_attentions_list = get_word_attention_scores_pl(
            original_texts_for_removed_attn,
            trained_id_model_pl,
            id_model_datamodule.tokenizer,
            current_device,
            max_length_tokenizer=MAX_LENGTH # 함수 내부에서 사용할 max_length
        )

        removed_attentions_avg = []
        for idx, row in tqdm(data_for_attention_metrics.iterrows(), total=len(data_for_attention_metrics), desc="Processing Removed Avg Attn"):
            top_words_sample = row[TOP_WORDS_COLUMN] # 이미 파싱된 리스트
            if isinstance(top_words_sample, list) and top_words_sample and idx < len(word_attentions_list):
                sentence_word_attentions = word_attentions_list[idx] # 단어(소문자):점수 딕셔너리
                current_removed_scores = [sentence_word_attentions.get(word.lower(), 0.0) for word in top_words_sample if isinstance(word, str)]
                removed_attentions_avg.append(np.mean(current_removed_scores) if current_removed_scores else 0.0)
            else:
                removed_attentions_avg.append(0.0)
        data_for_attention_metrics['removed_avg_attention'] = removed_attentions_avg
        print("Average attention of removed words calculated.")
    else:
        print(f"Skipping calculation of removed_avg_attention: '{TOP_WORDS_COLUMN}' or '{TEXT_COLUMN}' not found.")
        data_for_attention_metrics['removed_avg_attention'] = 0.0 # 컬럼 생성 및 기본값 할당

    # --- 5. 어텐션 지표 분포 시각화 ---
    print("\n--- 5. Visualizing Attention Metric Distributions ---")
    metric_columns_for_vis = ['max_attention', 'top_k_avg_attention', 'attention_entropy', 'removed_avg_attention']
    for metric in metric_columns_for_vis:
        if metric in data_for_attention_metrics.columns and not data_for_attention_metrics[metric].isnull().all():
            plot_metric_distribution(
                data_for_attention_metrics[metric].dropna().values,
                metric,
                f'Distribution of {metric} (on Masked/Original Text)',
                os.path.join(VIS_DIR, f'{metric}_distribution.png')
            )
        else:
            print(f"Skipping visualization for '{metric}': column not found or all values are null.")

    # --- 6. 개별 지표 및 순차 필터링 기반 OE 데이터셋 추출 ---
    print("\n--- 6. Extracting OE Datasets based on Attention Metrics ---")
    # 이 부분은 기존 oe_extractor2.py의 로직과 유사하게 진행
    # 개별 지표 기반 OE 추출
    for metric_name, settings in METRIC_SETTINGS.items():
        if metric_name not in data_for_attention_metrics.columns:
            print(f"Metric '{metric_name}' not found in data. Skipping OE extraction for this metric.")
            continue
        if MASKED_TEXT_COLUMN not in data_for_attention_metrics.columns:
            print(f"'{MASKED_TEXT_COLUMN}' not found. Skipping OE extraction for metric '{metric_name}'.")
            continue

        scores = data_for_attention_metrics[metric_name].copy().fillna(0.0).values # NaN을 0으로 처리
        filter_percentile = settings['percentile']
        filter_mode = settings['mode']
        selected_indices = []

        if filter_mode == 'higher':
            threshold = np.percentile(scores, 100 - filter_percentile)
            selected_indices = np.where(scores >= threshold)[0]
            mode_desc = f"top{filter_percentile}pct"
        elif filter_mode == 'lower':
            threshold = np.percentile(scores, filter_percentile)
            selected_indices = np.where(scores <= threshold)[0]
            mode_desc = f"bottom{filter_percentile}pct"
        print(f"Filtering for '{metric_name}' ({mode_desc}): threshold={threshold:.4f}, selected {len(selected_indices)} samples.")

        if len(selected_indices) > 0:
            oe_df_filtered_metric = data_for_attention_metrics.iloc[selected_indices][[MASKED_TEXT_COLUMN]].copy()
            extended_cols_metric = [MASKED_TEXT_COLUMN, TEXT_COLUMN, metric_name, TOP_WORDS_COLUMN]
            extended_cols_metric = [col for col in extended_cols_metric if col in data_for_attention_metrics.columns]
            oe_df_extended_metric = data_for_attention_metrics.iloc[selected_indices][extended_cols_metric].copy()

            oe_filename_metric = os.path.join(OE_DATA_DIR, f"oe_data_{metric_name}_{mode_desc}.csv")
            oe_extended_filename_metric = os.path.join(OE_DATA_DIR, f"oe_data_{metric_name}_{mode_desc}_extended.csv")
            try:
                oe_df_filtered_metric.to_csv(oe_filename_metric, index=False)
                print(f"  Saved: {oe_filename_metric} ({len(oe_df_filtered_metric)} samples)")
                oe_df_extended_metric.to_csv(oe_extended_filename_metric, index=False)
                print(f"  Saved: {oe_extended_filename_metric}")
            except Exception as e: print(f"Error saving OE dataset for {metric_name}: {e}")
        else: print(f"No samples selected for OE dataset based on {metric_name} ({mode_desc}).")

    # 순차 필터링 기반 OE 추출
    print("\nApplying sequential filtering...")
    selected_mask_seq = np.ones(len(data_for_attention_metrics), dtype=bool)
    seq_filter_desc_list = []

    for filter_step, (metric_name_seq, settings_seq) in enumerate(SEQUENTIAL_FILTERING_SEQUENCE):
        if metric_name_seq not in data_for_attention_metrics.columns:
            print(f"Sequential filter: Metric '{metric_name_seq}' not found. Skipping this step.")
            continue
        if MASKED_TEXT_COLUMN not in data_for_attention_metrics.columns:
            print(f"Sequential filter: '{MASKED_TEXT_COLUMN}' not found. Stopping sequential filtering.")
            break

        current_selection_df = data_for_attention_metrics[selected_mask_seq]
        if current_selection_df.empty:
            print("Sequential filter: No samples left to filter. Stopping.")
            break
        scores_seq = current_selection_df[metric_name_seq].copy().fillna(0.0).values
        
        filter_percentile_seq = settings_seq['percentile']
        filter_mode_seq = settings_seq['mode']
        step_mask_on_current_selection = np.zeros(len(scores_seq), dtype=bool)

        if filter_mode_seq == 'higher':
            threshold_seq = np.percentile(scores_seq, 100 - filter_percentile_seq) if len(scores_seq) > 0 else 0
            step_mask_on_current_selection = scores_seq >= threshold_seq
            mode_desc_seq = f"top{filter_percentile_seq}pct"
        elif filter_mode_seq == 'lower':
            threshold_seq = np.percentile(scores_seq, filter_percentile_seq) if len(scores_seq) > 0 else 0
            step_mask_on_current_selection = scores_seq <= threshold_seq
            mode_desc_seq = f"bottom{filter_percentile_seq}pct"
        
        # 현재 선택된 샘플들 내에서의 인덱스를 전체 데이터프레임의 인덱스로 변환하여 selected_mask_seq 업데이트
        original_indices_of_current_selection = np.where(selected_mask_seq)[0]
        indices_to_keep_from_step = original_indices_of_current_selection[step_mask_on_current_selection]
        
        new_selected_mask_seq = np.zeros_like(selected_mask_seq)
        new_selected_mask_seq[indices_to_keep_from_step] = True
        selected_mask_seq = new_selected_mask_seq
        
        seq_filter_desc_list.append(f"{metric_name_seq}_{mode_desc_seq}")
        print(f"Sequential filter step {filter_step+1} ({metric_name_seq} {mode_desc_seq}): threshold={threshold_seq:.4f}, remaining {np.sum(selected_mask_seq)} samples.")

    final_selected_indices_seq = np.where(selected_mask_seq)[0]
    if len(final_selected_indices_seq) > 0:
        seq_filter_filename_tag = "_".join(seq_filter_desc_list)
        oe_df_filtered_seq = data_for_attention_metrics.iloc[final_selected_indices_seq][[MASKED_TEXT_COLUMN]].copy()
        
        extended_cols_seq_names = [m for m,_ in SEQUENTIAL_FILTERING_SEQUENCE]
        extended_cols_seq = [MASKED_TEXT_COLUMN, TEXT_COLUMN, TOP_WORDS_COLUMN] + [col for col in extended_cols_seq_names if col in data_for_attention_metrics.columns]
        oe_df_extended_seq = data_for_attention_metrics.iloc[final_selected_indices_seq][extended_cols_seq].copy()

        oe_filename_seq = os.path.join(OE_DATA_DIR, f"oe_data_sequential_{seq_filter_filename_tag}.csv")
        oe_extended_filename_seq = os.path.join(OE_DATA_DIR, f"oe_data_sequential_{seq_filter_filename_tag}_extended.csv")
        try:
            oe_df_filtered_seq.to_csv(oe_filename_seq, index=False)
            print(f"  Saved: {oe_filename_seq} ({len(oe_df_filtered_seq)} samples)")
            oe_df_extended_seq.to_csv(oe_extended_filename_seq, index=False)
            print(f"  Saved: {oe_extended_filename_seq}")
        except Exception as e: print(f"Error saving sequential OE dataset: {e}")
    else: print("No samples selected after sequential filtering.")


    # --- 7. MSP 차이 계산 및 OE 데이터셋 추가 생성 ---
    print(f"\n--- 7. Calculating MSP Difference and Generating MSP-Filtered OE Datasets ---")
    oe_extended_files = [f for f in os.listdir(OE_DATA_DIR) if f.endswith("_extended.csv")]

    for extended_filename in oe_extended_files:
        print(f"\nProcessing for MSP: {extended_filename}")
        extended_filepath = os.path.join(OE_DATA_DIR, extended_filename)
        try:
            df_extended = pd.read_csv(extended_filepath)
            if TEXT_COLUMN not in df_extended.columns or MASKED_TEXT_COLUMN not in df_extended.columns:
                print(f"  Skipping {extended_filename}: missing '{TEXT_COLUMN}' or '{MASKED_TEXT_COLUMN}'.")
                continue
            if df_extended.empty:
                print(f"  Skipping {extended_filename}: dataframe is empty.")
                continue

            original_texts = df_extended[TEXT_COLUMN].tolist()
            masked_texts = df_extended[MASKED_TEXT_COLUMN].tolist()

            msp_original = calculate_msp_for_texts(original_texts, trained_id_model_pl, id_model_datamodule.tokenizer, current_device)
            msp_masked = calculate_msp_for_texts(masked_texts, trained_id_model_pl, id_model_datamodule.tokenizer, current_device)

            df_extended['msp_original'] = msp_original
            df_extended['msp_masked'] = msp_masked
            df_extended['msp_difference'] = np.abs(np.array(msp_original) - np.array(msp_masked)) # 절대값 차이

            # 업데이트된 extended 파일 저장 (덮어쓰기 또는 새 이름)
            # 여기서는 덮어쓰기로 가정
            df_extended.to_csv(extended_filepath, index=False)
            print(f"  Updated {extended_filename} with MSP columns.")

            # MSP 차이 기반 필터링
            msp_filtered_df = df_extended[df_extended['msp_difference'] >= MSP_DIFFERENCE_THRESHOLD]
            print(f"  Found {len(msp_filtered_df)} samples with MSP difference >= {MSP_DIFFERENCE_THRESHOLD}.")

            if not msp_filtered_df.empty:
                # 저장할 OE 데이터는 마스크된 텍스트만 포함
                oe_msp_final_df = msp_filtered_df[[MASKED_TEXT_COLUMN]].copy()
                
                msp_filter_tag = f"msp_diff{str(MSP_DIFFERENCE_THRESHOLD).replace('.', 'p')}"
                base_name_no_ext = extended_filename.replace("_extended.csv", "")
                
                # 기본 OE 데이터 (마스크된 텍스트만)
                msp_oe_filename = os.path.join(OE_DATA_MSP_FILTERED_DIR, f"{base_name_no_ext}_{msp_filter_tag}.csv")
                oe_msp_final_df.to_csv(msp_oe_filename, index=False)
                print(f"  Saved MSP-filtered OE data: {msp_oe_filename} ({len(oe_msp_final_df)} samples)")

                # 확장 OE 데이터 (모든 정보 포함)
                msp_oe_extended_filename = os.path.join(OE_DATA_MSP_FILTERED_DIR, f"{base_name_no_ext}_{msp_filter_tag}_extended.csv")
                msp_filtered_df.to_csv(msp_oe_extended_filename, index=False) # 여기서는 msp_filtered_df 전체를 저장
                print(f"  Saved MSP-filtered extended OE data: {msp_oe_extended_filename}")

                # MSP 차이 분포 시각화 (선택적)
                plot_metric_distribution(
                    df_extended['msp_difference'].dropna().values,
                    f'MSP_Difference_{base_name_no_ext}',
                    f'Distribution of MSP Difference ({base_name_no_ext})',
                    os.path.join(VIS_DIR, f'msp_difference_dist_{base_name_no_ext}.png')
                )
            else:
                print(f"  No samples met MSP difference threshold for {extended_filename}.")

        except Exception as e:
            print(f"  Error processing {extended_filename} for MSP: {e}")


    # --- 8. t-SNE 시각화 (마스크된 텍스트 특징 기반) ---
    print("\n--- 8. t-SNE Visualization (based on features from masked text) ---")
    if all_features_from_masked_text and len(all_features_from_masked_text) == len(data_for_attention_metrics):
        # t-SNE 레이블 준비 (ID 모델 학습 시 사용된 레이블 정보 활용)
        tsne_labels_for_plot = []
        # ID 모델 학습 시 사용된 label2id와 exclude_class 활용
        known_label2id_map = id_model_datamodule.label2id if id_model_datamodule and hasattr(id_model_datamodule, 'label2id') else {}
        id_exclude_class_lower = EXCLUDE_CLASS_FOR_TRAINING.lower()

        if CLASS_COLUMN in data_for_attention_metrics.columns:
            for cls_val in data_for_attention_metrics[CLASS_COLUMN].astype(str).str.lower():
                if cls_val == id_exclude_class_lower:
                    tsne_labels_for_plot.append(-1)  # Unknown/Excluded class
                else:
                    tsne_labels_for_plot.append(known_label2id_map.get(cls_val, -2)) # Known or Other
        else: # CLASS_COLUMN이 없으면 모두 -2 (Other)로 처리
            print(f"Warning: '{CLASS_COLUMN}' not found in data_for_attention_metrics for t-SNE labeling. All points labeled as 'Other'.")
            tsne_labels_for_plot = [-2] * len(data_for_attention_metrics)
        tsne_labels_np = np.array(tsne_labels_for_plot)

        tsne_class_names_map = {-1: 'Unknown/Excluded', -2: 'Other/Unlabeled'}
        if id_model_datamodule and hasattr(id_model_datamodule, 'id2label'):
            tsne_class_names_map.update(id_model_datamodule.id2label)

        # 각 OE 추출 기준별 t-SNE 시각화
        # (1) 개별 어텐션 지표 기반 OE 후보
        for metric_name, settings in METRIC_SETTINGS.items():
            if metric_name not in data_for_attention_metrics.columns: continue
            scores_metric = data_for_attention_metrics[metric_name].copy().fillna(0.0).values
            percentile_metric = settings['percentile']
            mode_metric = settings['mode']
            indices_metric_oe = []
            if mode_metric == 'higher':
                threshold_metric = np.percentile(scores_metric, 100 - percentile_metric)
                indices_metric_oe = np.where(scores_metric >= threshold_metric)[0]
            else: # lower
                threshold_metric = np.percentile(scores_metric, percentile_metric)
                indices_metric_oe = np.where(scores_metric <= threshold_metric)[0]

            plot_tsne(
                features=np.array(all_features_from_masked_text), labels=tsne_labels_np,
                title=f't-SNE (OE by {metric_name} {mode_metric} {percentile_metric}%)',
                save_path=os.path.join(VIS_DIR, f'tsne_oe_{metric_name}_{mode_metric}{percentile_metric}pct.png'),
                highlight_indices=indices_metric_oe, highlight_label=f'OE ({metric_name})',
                class_names=tsne_class_names_map, seed=RANDOM_STATE
            )
        # (2) 순차 필터링 기반 OE 후보 (final_selected_indices_seq 사용)
        if 'final_selected_indices_seq' in locals() and len(final_selected_indices_seq) > 0 :
             seq_filter_tag_vis = "_".join(seq_filter_desc_list) if 'seq_filter_desc_list' in locals() else "sequential"
             plot_tsne(
                features=np.array(all_features_from_masked_text), labels=tsne_labels_np,
                title=f't-SNE (OE by Sequential Filtering: {seq_filter_tag_vis})',
                save_path=os.path.join(VIS_DIR, f'tsne_oe_sequential_{seq_filter_tag_vis}.png'),
                highlight_indices=final_selected_indices_seq, highlight_label=f'OE (Sequential)',
                class_names=tsne_class_names_map, seed=RANDOM_STATE
            )
        # (3) MSP 차이 기반 OE 후보 (각 extended 파일별로 생성된 msp_filtered_df 사용)
        for extended_filename in oe_extended_files: # 이전에 MSP 처리한 파일 목록 재사용
            base_name_no_ext = extended_filename.replace("_extended.csv", "")
            msp_oe_extended_filepath = os.path.join(OE_DATA_MSP_FILTERED_DIR, f"{base_name_no_ext}_msp_diff{str(MSP_DIFFERENCE_THRESHOLD).replace('.', 'p')}_extended.csv")
            if os.path.exists(msp_oe_extended_filepath):
                try:
                    df_msp_oe_for_tsne = pd.read_csv(msp_oe_extended_filepath)
                    # 이 DataFrame의 인덱스는 data_for_attention_metrics의 원래 인덱스와 일치해야 함
                    # 만약 그렇지 않다면, 매핑 방법 필요. 여기서는 iloc으로 가져온 것이므로 원래 인덱스를 유지한다고 가정.
                    # 하지만, 안전하게 하려면 data_for_attention_metrics에 unique_id를 추가하고 merge하는 것이 좋음.
                    # 여기서는 df_msp_oe_for_tsne가 data_for_attention_metrics의 부분집합이고, 인덱스가 유지된다고 가정.
                    # 실제로는 df_msp_oe_for_tsne의 인덱스는 0부터 시작하므로, 원래 인덱스를 복원해야 함.
                    # 가장 간단한 방법은 MSP 필터링 시 원본 인덱스를 저장해두는 것.
                    # 지금은 data_for_attention_metrics의 부분집합 인덱스를 찾아야 함.
                    # 예시: df_extended = pd.read_csv(os.path.join(OE_DATA_DIR, extended_filename))
                    # msp_filtered_df = df_extended[df_extended['msp_difference'] >= MSP_DIFFERENCE_THRESHOLD]
                    # highlight_indices_msp = msp_filtered_df.index.to_numpy() # df_extended 기준 인덱스
                    # 이 인덱스를 data_for_attention_metrics 기준으로 변환해야 함.
                    # data_for_attention_metrics에 unique_id가 있다면 쉽게 매칭 가능. 없다면 복잡.
                    # 여기서는 해당 파일의 모든 샘플을 하이라이트하는 것으로 단순화 (실제로는 필터링된 샘플만)
                    # 또는, MSP 필터링 시 원본 인덱스를 저장했다가 사용.
                    # 지금은 해당 extended 파일 전체를 대상으로 tSNE를 그리고, 그 중 MSP 조건 만족 샘플을 하이라이트 하는 방식은 아님.
                    # MSP로 필터링된 OE 후보군 자체를 tSNE에 표시하는 것이므로, 해당 OE 파일의 인덱스를 사용.
                    # 이 부분은 OE 파일 생성 로직과 연동하여 highlight_indices를 정확히 가져와야 함.
                    # 임시로, MSP 필터링된 OE 파일의 모든 샘플을 하이라이트 한다고 가정하고,
                    # 해당 OE 파일이 data_for_attention_metrics의 어떤 부분집합인지 알아야 함.
                    # 여기서는 시각화 통일성을 위해 MSP 필터링된 샘플의 인덱스를 찾아 하이라이트 하는 것으로 가정.
                    # (이 부분은 실제 데이터 흐름에 맞춰 정확한 인덱싱 필요)
                    # 예를 들어, MSP 필터링 시 `data_for_attention_metrics.iloc[msp_selected_indices]` 와 같이 선택했다면,
                    # `msp_selected_indices`를 그대로 `highlight_indices`로 사용 가능.

                    # MSP 필터링된 데이터의 원본 인덱스를 가져오는 로직 (가정)
                    # 실제 구현에서는 MSP 필터링 단계에서 원본 인덱스를 저장해두고 사용해야 함.
                    # 아래는 임시 방편으로, MSP extended 파일에 있는 모든 샘플을 하이라이트한다고 가정.
                    # 하지만, 실제로는 MSP_DIFFERENCE_THRESHOLD를 만족하는 샘플만 하이라이트 해야 함.
                    # 이 부분은 MSP 필터링 로직과 긴밀하게 연동되어야 합니다.
                    # 여기서는 해당 extended 파일에 대해 MSP 차이가 Threshold 이상인 샘플들의 인덱스를 다시 계산하여 사용.
                    df_current_extended = pd.read_csv(os.path.join(OE_DATA_DIR, extended_filename)) # MSP 추가 전 파일
                    if 'msp_difference' not in df_current_extended.columns and os.path.exists(msp_oe_extended_filepath): # MSP 정보가 추가된 파일이 있다면 그것을 사용
                        df_current_extended = pd.read_csv(msp_oe_extended_filepath)

                    if 'msp_difference' in df_current_extended.columns:
                        indices_msp_oe = df_current_extended[df_current_extended['msp_difference'] >= MSP_DIFFERENCE_THRESHOLD].index.to_numpy()
                        # 이 인덱스는 df_current_extended 기준. data_for_attention_metrics 기준으로 변환 필요.
                        # 만약 df_current_extended가 data_for_attention_metrics.iloc[selected_indices_for_this_extended_file] 로 만들어졌다면,
                        # highlight_indices = selected_indices_for_this_extended_file[indices_msp_oe]
                        # 이 부분은 복잡하므로, 여기서는 해당 extended 파일명으로만 tSNE 생성하고, 하이라이트는 생략하거나
                        # MSP 필터링된 OE 데이터셋 자체의 특징으로 tSNE를 그리는 것을 고려.
                        # 여기서는 해당 extended 파일의 모든 샘플을 플로팅하고, 그 중 MSP 조건 만족 샘플을 하이라이트 하는 것으로 가정.
                        # 단, all_features_from_masked_text는 전체 데이터에 대한 것이므로,
                        # df_current_extended의 인덱스를 all_features_from_masked_text에 맞게 조정해야 함.
                        # 가장 안전한 방법은 unique ID를 사용하는 것.
                        # 이 부분은 현재 코드 구조에서 정확한 인덱싱이 어려우므로,
                        # MSP 필터링된 OE 데이터셋 자체의 특징으로 tSNE를 그리는 것을 권장.
                        # 여기서는 일단 해당 extended 파일의 MSP 조건 만족 샘플을 하이라이트 시도.
                        # (주의: 아래 highlight_indices는 해당 extended 파일 내의 인덱스임)
                        if len(indices_msp_oe) > 0:
                             # 이 부분은 전체 features에 대한 부분집합의 인덱스를 정확히 알아야 함.
                             # 지금은 해당 extended 파일이 전체 데이터의 어떤 부분인지 알 수 없으므로,
                             # 이 tSNE는 해당 extended 파일의 feature만으로 그려야 함.
                             # 또는, MSP 필터링된 최종 OE 파일의 feature를 다시 추출해서 그려야 함.
                             # 여기서는 all_features_from_masked_text를 사용하므로,
                             # highlight_indices_msp가 전체 데이터 기준 인덱스여야 함.
                             # 이 부분은 재검토 필요.
                             # 임시로, 해당 extended 파일에 대한 tSNE는 생략하거나, 별도 feature 추출 필요.
                             print(f"  Skipping t-SNE for MSP-filtered data of {extended_filename} due to indexing complexity with all_features. Consider plotting t-SNE on the MSP-filtered OE dataset's own features.")
                    else:
                        print(f"  Skipping t-SNE for {extended_filename}: 'msp_difference' column not found after processing.")

                except FileNotFoundError:
                    print(f"  Skipping t-SNE for {extended_filename}: MSP extended file not found.")
                except Exception as e:
                    print(f"  Error during t-SNE for MSP-filtered data of {extended_filename}: {e}")
    else:
        print("Skipping t-SNE visualization: Features or data length mismatch, or no features extracted.")


    # --- 9. 결과 요약 ---
    print("\n--- 9. OE Data Extraction and Filtering Summary ---")
    print(f"ID Model training logs and checkpoints saved in: {ID_MODEL_SAVE_DIR} and {LOG_DIR_ID_MODEL}")
    print(f"Attention metric-based OE datasets saved in: {OE_DATA_DIR}")
    print(f"MSP-filtered OE datasets saved in: {OE_DATA_MSP_FILTERED_DIR}")
    print(f"Visualizations (distributions, t-SNE) saved in: {VIS_DIR}")

    print("\nKey output files and directories created:")
    for d in [ID_MODEL_SAVE_DIR, LOG_DIR_ID_MODEL, OE_DATA_DIR, OE_DATA_MSP_FILTERED_DIR, VIS_DIR]:
        if os.path.exists(d):
            print(f"  - Directory: {d}")
            # # 내부 파일 몇 개 예시 (너무 많으면 생략)
            # contents = os.listdir(d)[:5]
            # for item in contents:
            #     print(f"    - {item}")

    print("\nScript execution finished.")

if __name__ == '__main__':
    main()