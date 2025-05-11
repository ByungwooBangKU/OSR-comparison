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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

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
import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union
from tqdm.auto import tqdm
import gc
import math
from scipy.stats import entropy
import random
import re
# 추가: NLTK 임포트 추가
import nltk
from nltk.tokenize import word_tokenize

# NLTK 다운로드 (필요 시)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- 설정값 ---
ORIGINAL_DATA_PATH = 'log_all_critical.csv'
MASKED_DATA_PATH = 'log_all_critical_filtered_attention_masked_for_oe.csv'  # 파일명 수정
TEXT_COLUMN = 'text'
CLASS_COLUMN = 'class'
MASKED_TEXT_COLUMN = 'masked_text_attention'
TOP_WORDS_COLUMN = 'top_attention_words'
EXCLUDE_CLASS_FOR_TRAINING = "unknown"

OE_FILTER_METRIC = 'removed_avg_attention'
# 각 지표별 맞춤형 퍼센타일 및 모드 설정
METRIC_SETTINGS = {
    # 전략 1: 모델이 덜 집중하거나 혼란스러워하는 샘플을 OE로 간주
    'attention_entropy': {'percentile': 80, 'mode': 'higher'},  # 엔트로피가 높은 (어텐션 분산된) 상위 20%
    'top_k_avg_attention': {'percentile': 20, 'mode': 'lower'},    # 상위 K개 어텐션 평균이 낮은 하위 20%
    'max_attention': {'percentile': 20, 'mode': 'lower'},          # 최대 어텐션이 낮은 하위 20%

    # 전략 2: "핵심 정보"가 제거된 문장을 OE로 간주 (원본 문장 선택 기준)
    # 이 지표로 선택된 *원본 문장*을 마스킹하여 OE 데이터로 사용
    'removed_avg_attention': {'percentile': 80, 'mode': 'higher'}
}
# 기본값 유지 (backward compatibility)
OE_FILTER_PERCENTILE = 75
OE_FILTER_MODE = 'higher'

# 출력 경로 설정
OUTPUT_DIR = '02_oe_extraction_results'
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "trained_standard_model")
LOG_DIR = os.path.join(OUTPUT_DIR, "lightning_logs")
CONFUSION_MATRIX_DIR = os.path.join(LOG_DIR, "confusion_matrices")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
OE_DATA_DIR = os.path.join(OUTPUT_DIR, "extracted_oe_datasets")

# 모델 및 학습 설정
MODEL_NAME = "roberta-base"
MAX_LENGTH = 256
BATCH_SIZE = 64
NUM_TRAIN_EPOCHS = 15
LEARNING_RATE = 2e-5
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
TOP_K_ATTENTION = 3

# 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(OE_DATA_DIR, exist_ok=True)

# --- 도우미 함수 ---
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}")

def preprocess_text_for_roberta(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_nltk(text):
    if not text:
        return []
    try:
        return word_tokenize(text)
    except Exception:
        return text.split()

def create_masked_sentence_per_sentence(original_text, sentence_important_words):
    if not isinstance(original_text, str):
        return ""
    if not sentence_important_words:
        return original_text
    processed_text = preprocess_text_for_roberta(original_text)
    tokens = tokenize_nltk(processed_text)
    important_set_lower = {word.lower() for word in sentence_important_words}
    masked_tokens = [word for word in tokens if word.lower() not in important_set_lower]
    masked_sentence = ' '.join(masked_tokens)
    if not masked_sentence:
        return "__EMPTY_MASKED__"
    return masked_sentence

# --- 데이터 클래스 ---
class TextDataset(TorchDataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.labels = labels
        self.texts = texts  # 원본 텍스트 저장 (어텐션 분석에 필요)
        print(f"Tokenizing {len(texts)} texts...")
        valid_texts = [str(t) if pd.notna(t) else "" for t in texts]
        self.encodings = tokenizer(valid_texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        print("Tokenization complete.")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # key 이름 수정: label -> labels (transformers 모델 호환성)
        return item

# --- PyTorch Lightning DataModule ---
class LogDataModuleForKnownClasses(pl.LightningDataModule):
    def __init__(self, file_path, text_col, class_col, exclude_class, model_name, batch_size, min_samples_per_class=3, num_workers=1, random_state=42, use_weighted_loss=False):
        super().__init__()
        self.save_hyperparameters()
        print(f"DataModule: Initializing tokenizer for {self.hparams.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        # 인스턴스 변수 초기화
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
        pass

    def setup(self, stage=None):
        if self.df_full is None:
            print(f"DataModule: Loading data from {self.hparams.file_path}")
            self.df_full = pd.read_csv(self.hparams.file_path)
            required_cols = [self.hparams.text_col, self.hparams.class_col]
            if not all(col in self.df_full.columns for col in required_cols):
                raise ValueError(f"CSV 누락 컬럼: {required_cols}")
            self.df_full = self.df_full.dropna(subset=[self.hparams.class_col])
            self.df_full[self.hparams.class_col] = self.df_full[self.hparams.class_col].astype(str).str.lower()
            exclude_class_lower = self.hparams.exclude_class.lower() if self.hparams.exclude_class else None

            print(f"DataModule: Excluding class '{self.hparams.exclude_class}' for training/validation.")
            df_known = self.df_full[self.df_full[self.hparams.class_col] != exclude_class_lower].copy()
            print(f"DataModule: Data size after excluding '{self.hparams.exclude_class}': {len(df_known)}")

            known_classes_str = sorted(df_known[self.hparams.class_col].unique())
            self.label2id = {label: i for i, label in enumerate(known_classes_str)}
            self.id2label = {i: label for label, i in self.label2id.items()}
            self.num_labels = len(known_classes_str)
            print(f"\nDataModule - 레이블 매핑 완료 (Known Classes): {self.num_labels}개 클래스")
            print(f"Known Class Label to ID mapping: {self.label2id}")
            df_known['label'] = df_known[self.hparams.class_col].map(self.label2id)
            df_known = df_known.dropna(subset=['label'])
            df_known['label'] = df_known['label'].astype(int)

            print(f"\nDataModule - Known 클래스 데이터 필터링 (클래스당 최소 {self.hparams.min_samples_per_class}개)...")
            label_counts_known = df_known['label'].value_counts()
            labels_to_keep = label_counts_known[label_counts_known >= self.hparams.min_samples_per_class].index
            self.df_known_for_train_val = df_known[df_known['label'].isin(labels_to_keep)].copy()
            print(f"DataModule - 필터링 후 학습/검증 대상 데이터: {len(self.df_known_for_train_val)}")
            if len(self.df_known_for_train_val) == 0:
                raise ValueError("필터링 후 학습/검증 대상 데이터 없음.")
            print("\n--- 학습/검증 대상 데이터 클래스 분포 ---")
            print(self.df_known_for_train_val['label'].map(self.id2label).value_counts())

            if self.hparams.use_weighted_loss:
                labels_for_weights = self.df_known_for_train_val['label'].values
                unique_labels_in_train_val = np.unique(labels_for_weights)
                try:
                    class_weights_array = compute_class_weight('balanced', classes=unique_labels_in_train_val, y=labels_for_weights)
                    self.class_weights = torch.ones(self.num_labels)
                    for i, label_idx in enumerate(unique_labels_in_train_val):
                        if label_idx < self.num_labels:
                            self.class_weights[label_idx] = class_weights_array[i]
                    print(f"\nDataModule - 계산된 클래스 가중치: {self.class_weights}")
                except ValueError as e:
                    print(f"클래스 가중치 계산 오류: {e}. 가중치 없이 진행.")
                    self.hparams.use_weighted_loss = False
                    self.class_weights = None

            print("\nDataModule - 학습/검증 데이터 분할...")
            try:
                self.train_df_final, self.val_df_final = train_test_split(
                    self.df_known_for_train_val, test_size=0.2, 
                    random_state=self.hparams.random_state, stratify=self.df_known_for_train_val['label']
                )
            except ValueError:
                print("경고: Stratify 분할 실패.")
                self.train_df_final, self.val_df_final = train_test_split(
                    self.df_known_for_train_val, test_size=0.2, random_state=self.hparams.random_state
                )
            print(f"DataModule - 최종 학습셋: {len(self.train_df_final)}, 최종 검증셋: {len(self.val_df_final)}")

            raw_train_val_datasets = DatasetDict({
                'train': Dataset.from_pandas(self.train_df_final),
                'validation': Dataset.from_pandas(self.val_df_final)
            })
            
            def tokenize_func(examples):
                return self.tokenizer(
                    [preprocess_text_for_roberta(text) for text in examples[self.hparams.text_col]],
                    truncation=True, padding=False, max_length=self.tokenizer.model_max_length
                )
            
            print("\nDataModule - 학습/검증 데이터셋 토큰화 중...")
            self.tokenized_train_val_datasets = raw_train_val_datasets.map(
                tokenize_func, batched=True, 
                num_proc=max(1, self.hparams.num_workers // 2),
                remove_columns=[col for col in raw_train_val_datasets['train'].column_names if col != 'label']
            )
            self.tokenized_train_val_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            print("DataModule - 토큰화 완료.")

    def train_dataloader(self):
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
        return DataLoader(
            self.tokenized_train_val_datasets['validation'], 
            batch_size=self.hparams.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0
        )
    
    def get_full_dataframe(self):
        if self.df_full is None:
            self.setup()
        return self.df_full

# --- PyTorch Lightning Module ---
class LogClassifierPL(pl.LightningModule):
    def __init__(self, model_name, num_labels, label2id, id2label, confusion_matrix_dir, learning_rate=2e-5,
                 use_weighted_loss=False, class_weights=None, use_lr_scheduler=False, warmup_steps=0):
        super().__init__()
        self.save_hyperparameters()
        print(f"LightningModule: Initializing model {self.hparams.model_name} for {self.hparams.num_labels} known classes.")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hparams.model_name, 
            num_labels=self.hparams.num_labels, 
            label2id=self.hparams.label2id, 
            id2label=self.hparams.id2label, 
            ignore_mismatched_sizes=True, 
            output_attentions=True, 
            output_hidden_states=True
        )
        
        if self.hparams.use_weighted_loss and self.hparams.class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(self.hparams.class_weights, dtype=torch.float))
            print("Using Weighted CE Loss")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            print("Using Standard CE Loss")
        
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
            self.loss_fn.weight = self.loss_fn.weight.to(self.device)
            print(f"Moved weights to {self.device}")
    
    def forward(self, batch, output_features=False, output_attentions=False):
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        
        if input_ids is None or attention_mask is None:
            raise ValueError("Batch missing 'input_ids' or 'attention_mask'")
        
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=output_features, 
            output_attentions=output_attentions
        )
        
        return outputs
    
    def _common_step(self, batch, batch_idx):
        # 'label'에서 'labels'로 키 이름 변경
        if 'label' in batch:
            batch['labels'] = batch.pop('label')
        
        # 모델에 직접 전체 배치 전달 (labels 포함)
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        
        return loss, preds, batch['labels']
    
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
            print("\nValidation CM:")
            known_class_names = list(self.hparams.id2label.values())
            cm_df = pd.DataFrame(val_cm_computed.cpu().numpy(), index=known_class_names, columns=known_class_names)
            print(cm_df)
            cm_filename = os.path.join(self.hparams.confusion_matrix_dir, f"validation_cm_epoch_{self.current_epoch}.csv")
            cm_df.to_csv(cm_filename)
            print(f"Val CM saved: {cm_filename}")
        except Exception as e:
            print(f"Error Val CM: {e}")
        finally:
            self.val_cm.reset()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
        if self.hparams.use_lr_scheduler:
            if self.trainer and hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches > 0:
                num_training_steps = self.trainer.estimated_stepping_batches
                print(f"Est steps: {num_training_steps}")
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=self.hparams.warmup_steps, 
                    num_training_steps=num_training_steps
                )
                return {
                    "optimizer": optimizer, 
                    "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
                }
            else:
                print("Warn: No scheduler.")
                return optimizer
        else:
            return optimizer

# --- 단어 어텐션 스코어 추출 함수 (추가) ---
@torch.no_grad()
def get_word_attention_scores_pl(texts, model_pl, tokenizer, device, layer_idx=-1, head_idx=None, max_length=512):
    """
    각 텍스트의 단어별 어텐션 스코어를 계산합니다.
    
    Args:
        texts (List[str]): 어텐션 점수를 계산할 텍스트 목록
        model_pl (LogClassifierPL): 학습된 PyTorch Lightning 모델
        tokenizer: 모델 토크나이저
        device: 계산에 사용할 디바이스
        layer_idx (int): 사용할 어텐션 레이어 인덱스 (default: 마지막 레이어)
        head_idx (int): 사용할 어텐션 헤드 인덱스 (None일 경우 모든 헤드 평균)
        max_length (int): 최대 시퀀스 길이
    
    Returns:
        List[Dict[str, float]]: 각 텍스트의 단어별 어텐션 스코어 딕셔너리 목록
    """
    model_pl.eval()
    model_pl.to(device)
    
    # 결과 초기화 - 모든 텍스트에 대해 미리 빈 딕셔너리 생성
    word_attention_scores = [{} for _ in range(len(texts))]
    
    # 처리할 유효한 텍스트와 그 인덱스 찾기
    valid_indices = []
    valid_texts = []
    
    for i, text in enumerate(texts):
        if isinstance(text, str) and text.strip():
            valid_indices.append(i)
            valid_texts.append(text)
    
    # 유효한 텍스트가 없으면 빈 결과 반환
    if not valid_texts:
        return word_attention_scores
    
    # 배치 처리 (메모리 초과를 방지하기 위해)
    batch_size = 16  # 더 작은 배치 크기
    
    for batch_start in range(0, len(valid_texts), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_texts))
        batch_texts = valid_texts[batch_start:batch_end]
        batch_indices = valid_indices[batch_start:batch_end]
        
        try:
            # 토큰화 및 모델 입력 준비
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=max_length  # 최대 길이 적용
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 모델 추론 (어텐션 값 추출)
            outputs = model_pl.model(**inputs, output_attentions=True)
            attentions = outputs.attentions[layer_idx]  # 지정된 레이어의 어텐션 가져오기
            
            # 각 텍스트 처리
            for batch_idx, (text_idx, text) in enumerate(zip(batch_indices, batch_texts)):
                # 원본 토큰화 (최대 길이 제한 없이)
                all_tokens = tokenizer.tokenize(text)
                
                # 실제로 모델에 입력된 토큰 ID
                token_ids = inputs["input_ids"][batch_idx].cpu().numpy()
                
                # 어텐션 헤드 선택 (지정된 헤드 또는 모든 헤드 평균)
                if head_idx is not None:
                    att_matrix = attentions[batch_idx, head_idx].cpu().numpy()
                else:
                    att_matrix = attentions[batch_idx].mean(dim=0).cpu().numpy()  # 모든 헤드 평균
                
                # CLS 토큰(첫 번째 토큰)이 각 토큰에 주는 어텐션 점수
                cls_attentions = att_matrix[0]
                
                # 특수 토큰 ID 제외
                special_tokens_mask = np.logical_or.reduce([
                    token_ids == tokenizer.cls_token_id,
                    token_ids == tokenizer.sep_token_id,
                    token_ids == tokenizer.pad_token_id
                ])
                valid_token_indices = np.where(~special_tokens_mask)[0]
                
                if len(valid_token_indices) == 0:
                    continue
                
                # 토큰 -> 단어 매핑
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                token_to_word_map = {}
                word_scores = {}
                
                # RoBERTa 토크나이저 기반 단어 매핑
                current_word = ""
                current_word_idx = None
                
                for token_idx, token in enumerate(tokens):
                    if token == tokenizer.cls_token or token == tokenizer.sep_token or token == tokenizer.pad_token:
                        continue
                        
                    if token.startswith("Ġ"):  # 새 단어 시작
                        if current_word:  # 이전 단어 저장
                            for i in range(current_word_idx, token_idx):
                                token_to_word_map[i] = current_word
                        
                        current_word = token.replace("Ġ", "")
                        current_word_idx = token_idx
                    else:
                        if current_word_idx is None:  # 첫 토큰이 Ġ로 시작하지 않는 경우
                            current_word = token
                            current_word_idx = token_idx
                        else:
                            current_word += token
                
                # 마지막 단어 처리
                if current_word and current_word_idx is not None:
                    for i in range(current_word_idx, len(tokens)):
                        if i < len(tokens) and tokens[i] not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                            token_to_word_map[i] = current_word
                
                # 단어별 어텐션 점수 계산
                for token_idx in valid_token_indices:
                    if token_idx in token_to_word_map:
                        word = token_to_word_map[token_idx].lower()
                        score = cls_attentions[token_idx]
                        
                        if word in word_scores:
                            word_scores[word] = max(word_scores[word], score)
                        else:
                            word_scores[word] = score
                
                # 결과 저장
                word_attention_scores[text_idx] = word_scores
                
        except Exception as e:
            print(f"배치 처리 중 오류: {e}")
            continue
    
    return word_attention_scores

# --- 어텐션/특징 추출 및 지표 계산 함수 (개선) ---
@torch.no_grad()
def extract_attention_and_features(model_pl: LogClassifierPL, dataloader: DataLoader, device: torch.device, tokenizer, top_k: int = 3) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    """데이터로더를 순회하며 어텐션 기반 지표와 특징 벡터 추출"""
    model_pl.eval()
    model_pl.to(device)

    attention_metrics = []
    features_list = []
    tokenized_texts = []  # 토큰화된 텍스트 저장 (토큰 어텐션 시각화 등을 위해)
    
    print("Extracting attention scores and features...")
    for batch in tqdm(dataloader, desc="Extracting"):
        # 배치 데이터 준비
        batch_for_model = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device)
        }
        
        # 어텐션 및 특징 동시 추출
        outputs = model_pl.forward(batch_for_model, output_features=True, output_attentions=True)
        
        # 결과 처리
        attentions_batch = outputs.attentions[-1].cpu().numpy()  # 마지막 레이어 어텐션
        features_batch = outputs.hidden_states[-1][:, 0, :].cpu().numpy()  # 마지막 hidden state의 CLS 토큰
        features_list.extend(list(features_batch))
        
        # 배치 내 각 샘플 처리
        for i in range(len(batch['input_ids'])):
            attn_sample = attentions_batch[i]  # (num_heads, seq_len, seq_len)
            token_ids = batch['input_ids'][i].cpu().numpy()
            
            # CLS, SEP, PAD를 제외한 유효 토큰 인덱스 찾기
            valid_token_indices = np.where(
                (token_ids != tokenizer.pad_token_id) &
                (token_ids != tokenizer.cls_token_id) &
                (token_ids != tokenizer.sep_token_id)
            )[0]
            
            # 토큰화된 텍스트 저장
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            tokenized_texts.append(tokens)
            
            if len(valid_token_indices) == 0:
                attention_metrics.append({
                    'max_attention': 0,
                    'top_k_avg_attention': 0,
                    'attention_entropy': 0
                })
                continue
            
            # 모든 헤드의 평균 어텐션 (CLS 토큰이 다른 토큰에 주는 어텐션)
            cls_attentions = np.mean(attn_sample[:, 0, :], axis=0)  # (seq_len,)
            valid_cls_attentions = cls_attentions[valid_token_indices]
            
            # 어텐션 지표 계산
            max_attention = np.max(valid_cls_attentions) if len(valid_cls_attentions) > 0 else 0
            
            # 상위 K개 어텐션 평균
            k = min(top_k, len(valid_cls_attentions))
            top_k_avg_attention = np.mean(np.sort(valid_cls_attentions)[-k:]) if k > 0 else 0
            
            # 어텐션 엔트로피 계산 (정규화된 확률 분포 사용)
            attention_probs = F.softmax(torch.tensor(valid_cls_attentions), dim=0).numpy()
            attention_entropy = entropy(attention_probs) if len(attention_probs) > 0 else 0
            
            attention_metrics.append({
                'max_attention': max_attention,
                'top_k_avg_attention': top_k_avg_attention,
                'attention_entropy': attention_entropy
            })
    
    df_results = pd.DataFrame(attention_metrics)
    
    # 메모리 정리
    del outputs, attentions_batch, features_batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    return df_results, features_list

# --- 시각화 함수 ---
def plot_metric_distribution(scores: np.ndarray, metric_name: str, title: str, save_path: Optional[str] = None):
    if len(scores) == 0:
        print(f"No scores for {metric_name}.")
        return
    
    plt.figure(figsize=(10, 6))
    if SNS_AVAILABLE:
        sns.histplot(scores, bins=50, kde=True, stat='density')
    else:
        plt.hist(scores, bins=50, density=True)
        
    plt.title(title)
    plt.xlabel(metric_name)
    plt.ylabel('Density')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"{metric_name} dist plot saved: {save_path}")
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_tsne(features, labels, title, save_path=None, 
                  highlight_indices=None, highlight_label='OE Candidate', 
                  class_names=None, seed=42, perplexity=30, n_iter=1000):
    """
    An improved version of the t-SNE plotting function with better formatting
    and layout handling for proper image rendering.
    
    Args:
        features: numpy array of feature vectors
        labels: numpy array of class labels
        title: plot title
        save_path: path to save the figure to
        highlight_indices: indices of samples to highlight
        highlight_label: label for highlighted samples
        class_names: dictionary mapping label indices to class names
        seed: random seed for reproducibility
        perplexity: t-SNE perplexity parameter
        n_iter: number of iterations for t-SNE
    """
    if len(features) == 0:
        print("No features for t-SNE.")
        return
    
    print(f"Running t-SNE on {features.shape[0]} samples...")
    try:
        # Adjust perplexity to avoid errors with small datasets
        adjusted_perplexity = min(perplexity, features.shape[0] - 1)
        tsne = TSNE(
            n_components=2, 
            random_state=seed, 
            perplexity=adjusted_perplexity, 
            n_iter=n_iter, 
            init='pca', 
            learning_rate='auto'
        )
        tsne_results = tsne.fit_transform(features)
    except Exception as e:
        print(f"Error running t-SNE: {e}. Skipping plot.")
        return
    
    df_tsne = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
    df_tsne['label'] = labels
    df_tsne['is_highlighted'] = False
    
    if highlight_indices is not None:
        df_tsne.loc[highlight_indices, 'is_highlighted'] = True
    
    # Create a larger figure with better proportions
    plt.figure(figsize=(14, 10))
    
    # Use a different colormap with better contrast
    unique_labels = sorted(df_tsne['label'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each class separately
    for i, label_val in enumerate(unique_labels):
        subset = df_tsne[(df_tsne['label'] == label_val) & (~df_tsne['is_highlighted'])]
        if len(subset) > 0:
            class_name = class_names.get(label_val, f'Class {label_val}') if class_names else f'Class {label_val}'
            plt.scatter(subset['tsne1'], subset['tsne2'], color=colors[i], label=class_name, alpha=0.7, s=30)
    
    # Highlight the OE candidates with a different marker and color
    if highlight_indices is not None:
        highlight_subset = df_tsne[df_tsne['is_highlighted']]
        if len(highlight_subset) > 0:
            plt.scatter(highlight_subset['tsne1'], highlight_subset['tsne2'], 
                      color='red', marker='x', s=100, label=highlight_label, alpha=0.9)
    
    # Improved formatting
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    
    # Add a grid for better readability
    plt.grid(alpha=0.3, linestyle='--')
    
    # Improve legend positioning and formatting
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                      fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Add some padding around the plot
    plt.tight_layout()
    
    # Ensure the legend doesn't get cut off
    plt.subplots_adjust(right=0.75)
    
    if save_path:
        # Ensure high DPI for better quality image
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
def plot_tsne_old(features: np.ndarray, labels: np.ndarray, title: str, save_path: Optional[str] = None, 
              highlight_indices: Optional[np.ndarray] = None, highlight_label: str = 'OE Candidate', 
              class_names: Optional[Dict[int, str]] = None, seed: int = 42, perplexity: int = 30, n_iter: int = 1000):
    if len(features) == 0:
        print("No features for t-SNE.")
        return
    
    print(f"Running t-SNE on {features.shape[0]} samples...")
    try:
        tsne = TSNE(
            n_components=2, 
            random_state=seed, 
            perplexity=min(perplexity, features.shape[0] - 1), 
            n_iter=n_iter, 
            init='pca', 
            learning_rate='auto'
        )
        tsne_results = tsne.fit_transform(features)
    except Exception as e:
        print(f"Error running t-SNE: {e}. Skipping plot.")
        return
    
    df_tsne = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
    df_tsne['label'] = labels
    df_tsne['is_highlighted'] = False
    
    if highlight_indices is not None:
        df_tsne.loc[highlight_indices, 'is_highlighted'] = True
    
    plt.figure(figsize=(12, 10))
    unique_labels = sorted(df_tsne['label'].unique())
    colors = plt.cm.get_cmap('tab20', len(unique_labels))
    
    for i, label_val in enumerate(unique_labels):
        subset = df_tsne[(df_tsne['label'] == label_val) & (~df_tsne['is_highlighted'])]
        class_name = class_names.get(label_val, f'Class {label_val}') if class_names else f'Class {label_val}'
        plt.scatter(subset['tsne1'], subset['tsne2'], color=colors(i), label=class_name, alpha=0.5, s=10)
    
    if highlight_indices is not None:
        highlight_subset = df_tsne[df_tsne['is_highlighted']]
        plt.scatter(highlight_subset['tsne1'], highlight_subset['tsne2'], color='red', marker='x', s=50, label=highlight_label, alpha=0.8)
    
    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    if save_path:
        plt.savefig(save_path)
        print(f"t-SNE plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

# --- 메인 실행 로직 ---
def main():
    pl.seed_everything(RANDOM_STATE)

    # 1. 데이터 모듈 초기화 및 설정
    print("--- 1. 데이터 모듈 설정 ---")
    log_data_module = LogDataModuleForKnownClasses(
        file_path=ORIGINAL_DATA_PATH, text_col=TEXT_COLUMN, class_col=CLASS_COLUMN,
        exclude_class=EXCLUDE_CLASS_FOR_TRAINING, model_name=MODEL_NAME, batch_size=BATCH_SIZE,
        min_samples_per_class=MIN_SAMPLES_PER_CLASS_FOR_TRAIN_VAL, num_workers=NUM_WORKERS,
        random_state=RANDOM_STATE, use_weighted_loss=USE_WEIGHTED_LOSS
    )
    log_data_module.setup()

    # 2. 모델 모듈 초기화
    print("\n--- 2. 모델 모듈 초기화 ---")
    model_module = LogClassifierPL(
        model_name=MODEL_NAME, num_labels=log_data_module.num_labels,
        label2id=log_data_module.label2id, id2label=log_data_module.id2label,
        confusion_matrix_dir=CONFUSION_MATRIX_DIR,
        learning_rate=LEARNING_RATE, use_weighted_loss=USE_WEIGHTED_LOSS,
        class_weights=log_data_module.class_weights, use_lr_scheduler=USE_LR_SCHEDULER, warmup_steps=0
    )

    # 3. 콜백 및 로거 설정
    print("\n--- 3. 콜백 및 로거 설정 ---")
    monitor_metric = 'val_f1_macro'
    monitor_mode = 'max'
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_SAVE_DIR, 
        filename=f'best-model-{{epoch:02d}}-{{{monitor_metric}:.4f}}', 
        save_top_k=1, 
        monitor=monitor_metric, 
        mode=monitor_mode
    )
    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric, 
        patience=3, 
        mode=monitor_mode, 
        verbose=True
    )
    csv_logger = CSVLogger(save_dir=LOG_DIR, name="standard_model_training")

    # 4. 트레이너 설정 및 학습
    print("\n--- 4. 트레이너 설정 및 학습 시작 ---")
    trainer = pl.Trainer(
        max_epochs=NUM_TRAIN_EPOCHS, 
        accelerator=ACCELERATOR, 
        devices=DEVICES,
        precision=PRECISION, 
        logger=csv_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        deterministic=False, 
        log_every_n_steps=LOG_EVERY_N_STEPS, 
        gradient_clip_val=GRADIENT_CLIP_VAL
    )
    
    # 이미 학습된 모델이 있는지 확인
    if os.path.exists(MODEL_SAVE_DIR) and any(file.endswith('.ckpt') for file in os.listdir(MODEL_SAVE_DIR)):
        print("기존 학습된 모델 파일 발견. 학습 단계 건너뛰기...")
    else:
        trainer.fit(model_module, datamodule=log_data_module)
        print("--- 모델 학습 완료 ---")

    # 5. 최적 모델 로드
    print("\n--- 5. 최적 모델 로드 ---")
    best_model_path = checkpoint_callback.best_model_path if hasattr(checkpoint_callback, 'best_model_path') else None
    
    if not best_model_path or not os.path.exists(best_model_path):
        print("경고: 최적 모델을 찾을 수 없습니다. 가장 최근 모델을 검색합니다.")
        # 모델 디렉토리에서 가장 최근 체크포인트 찾기
        checkpoint_files = [f for f in os.listdir(MODEL_SAVE_DIR) if f.endswith('.ckpt')]
        if checkpoint_files:
            best_model_path = os.path.join(MODEL_SAVE_DIR, sorted(checkpoint_files)[-1])
            print(f"최근 모델 발견: {best_model_path}")
        else:
            print("기존 모델 없음. 현재 모델 사용.")
            best_model_path = None
        
    if best_model_path:
        trained_model_pl = LogClassifierPL.load_from_checkpoint(
            best_model_path, 
            confusion_matrix_dir=CONFUSION_MATRIX_DIR,
            # 아래 라인 추가
            weights_only=True
        )
        print("모델 로드 완료.")
    else:
        trained_model_pl = model_module
        print("현재 모델 사용.")
    
    current_device = torch.device("cuda" if torch.cuda.is_available() and ACCELERATOR != "cpu" else "cpu")
    trained_model_pl.to(current_device)
    trained_model_pl.eval()
    trained_model_pl.freeze()

    # 6. 마스크된 데이터 파일 로드
    print("\n--- 6. 마스크된 데이터 파일 로드 ---")
    try:
        masked_df = pd.read_csv(MASKED_DATA_PATH)
        print(f"마스크된 데이터 파일 로드 완료: {len(masked_df)} 샘플")
        
        # top_words_column 확인 및 변환
        if TOP_WORDS_COLUMN in masked_df.columns:
            import ast
            
            def safe_literal_eval(val):
                try:
                    if isinstance(val, str) and val.strip().startswith('['):
                        return ast.literal_eval(val)
                    else:
                        return []
                except Exception as e:
                    print(f"Error parsing: {val[:30]}... - {e}")
                    return []
            
            masked_df[TOP_WORDS_COLUMN] = masked_df[TOP_WORDS_COLUMN].apply(safe_literal_eval)
            print(f"상위 어텐션 단어 컬럼 변환 완료")
        else:
            print(f"경고: '{TOP_WORDS_COLUMN}' 컬럼을 찾을 수 없습니다.")
            masked_df[TOP_WORDS_COLUMN] = [[]] * len(masked_df)
    
    except FileNotFoundError:
        print(f"마스크된 데이터 파일을 찾을 수 없습니다: {MASKED_DATA_PATH}")
        print("빈 DataFrame 생성")
        masked_df = pd.DataFrame({
            TEXT_COLUMN: [],
            MASKED_TEXT_COLUMN: [],
            TOP_WORDS_COLUMN: []
        })
    except Exception as e:
        print(f"마스크된 데이터 로드 중 오류: {e}")
        print("빈 DataFrame 생성")
        masked_df = pd.DataFrame({
            TEXT_COLUMN: [],
            MASKED_TEXT_COLUMN: [],
            TOP_WORDS_COLUMN: []
        })

    # 7. 전체 데이터 또는 마스크된 데이터에 대한 어텐션 지표 계산
    print("\n--- 7. 어텐션 지표 계산 ---")
    
    # 처리할 데이터 결정 (마스크된 데이터 우선)
    if len(masked_df) > 0 and MASKED_TEXT_COLUMN in masked_df.columns:
        print(f"마스크된 데이터 {len(masked_df)}개를 사용하여 어텐션 지표 계산")
        data_for_attention = masked_df
        text_column_to_use = MASKED_TEXT_COLUMN
    else:
        print(f"원본 데이터를 사용하여 어텐션 지표 계산")
        data_for_attention = log_data_module.get_full_dataframe().reset_index(drop=True)
        text_column_to_use = TEXT_COLUMN
    
    # 레이블 및 데이터셋 준비
    dummy_labels = [-1] * len(data_for_attention)
    attention_dataset = TextDataset(
        data_for_attention[text_column_to_use].tolist(), 
        dummy_labels, 
        log_data_module.tokenizer, 
        MAX_LENGTH
    )
    attention_dataloader = DataLoader(
        attention_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        shuffle=False  # 순서 유지
    )
    
    # 어텐션 지표 및 특징 벡터 추출
    attention_metrics_df, all_features = extract_attention_and_features(
        trained_model_pl, 
        attention_dataloader, 
        current_device,
        log_data_module.tokenizer,  # 토크나이저 직접 전달
        top_k=TOP_K_ATTENTION
    )
    
    if len(data_for_attention) == len(attention_metrics_df):
        data_for_attention = pd.concat(
            [data_for_attention.reset_index(drop=True), attention_metrics_df.reset_index(drop=True)], 
            axis=1
        )
        print("어텐션 지표가 데이터프레임에 추가되었습니다.")
    else:
        print(f"경고: 길이 불일치. 데이터: {len(data_for_attention)}, 지표: {len(attention_metrics_df)}")
    
    # 8. 제거된 단어의 어텐션 계산
    print("\n--- 8. 제거된 단어의 어텐션 점수 계산 ---")
    if TOP_WORDS_COLUMN in data_for_attention.columns and TEXT_COLUMN in data_for_attention.columns:
        print("제거된 단어의 어텐션 점수 계산 중...")
        
        # 원본 텍스트에 대한 단어별 어텐션 점수 계산
        texts = data_for_attention[TEXT_COLUMN].tolist()
        
        # 배치 처리로 처리량 개선
        batch_size = 64
        word_attentions_list = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Calculating Word Attentions"):
            batch_texts = texts[i:i+batch_size]
            batch_scores = get_word_attention_scores_pl(
                batch_texts, 
                trained_model_pl, 
                log_data_module.tokenizer, 
                current_device
            )
            word_attentions_list.extend(batch_scores)
        
        # 제거된 단어의 어텐션 점수 계산
        removed_attentions = []
        
        for idx, row in tqdm(data_for_attention.iterrows(), total=len(data_for_attention), desc="Calculating Removed Avg Attn"):
            top_words = row[TOP_WORDS_COLUMN] if TOP_WORDS_COLUMN in row else []
            
            if isinstance(top_words, list) and top_words and idx < len(word_attentions_list):
                sentence_word_attentions = word_attentions_list[idx]
                # 제거된 각 단어의 어텐션 점수 가져오기
                removed_scores = [sentence_word_attentions.get(word.lower(), 0) for word in top_words]
                # 평균 계산
                removed_attentions.append(np.mean(removed_scores) if removed_scores else 0)
            else:
                removed_attentions.append(0)
        
        data_for_attention['removed_avg_attention'] = removed_attentions
        print("제거된 단어의 평균 어텐션 점수 계산 완료.")
    else:
        print(f"필요한 컬럼이 없어 제거된 단어 어텐션 계산을 건너뜁니다.")
        data_for_attention['removed_avg_attention'] = 0.0
    
    # 9. 어텐션 지표 분포 시각화
    print("\n--- 9. 어텐션 지표 분포 시각화 ---")
    metric_columns = ['max_attention', 'top_k_avg_attention', 'attention_entropy', 'removed_avg_attention']
    
    for metric in metric_columns:
        if metric in data_for_attention.columns and not data_for_attention[metric].isnull().all():
            print(f"{metric} 분포 시각화 중...")
            plot_metric_distribution(
                data_for_attention[metric].dropna().values,
                metric,
                f'Distribution of {metric}',
                os.path.join(VIS_DIR, f'{metric}_distribution.png')
            )
        else:
            print(f"{metric} 시각화 건너뛰기 (없거나 모두 null).")
        

    # 10. OE 데이터셋 추출 부분 수정
    print("\n--- 10. OE 데이터셋 추출 (순차적 필터링 적용) ---")

    # 순차적 필터링 설정
    # FILTERING_SEQUENCE = [
    #     ('removed_avg_attention', {'percentile': 90, 'mode': 'higher'}),
    #     ('attention_entropy', {'percentile': 30, 'mode': 'lower'})
    # ]
    FILTERING_SEQUENCE = [
        # 1단계: 모델이 원래 중요하다고 판단했던 단어들의 중요도가 높은 원본 문장 선택
        ('removed_avg_attention', {'percentile': 80, 'mode': 'higher'}), # 상위 20% 원본 문장 선택
        # 2단계: 위에서 선택된 문장들 중에서, (마스킹 후) 어텐션 엔트로피가 높은 (모델이 혼란스러워하는) 샘플 선택
        ('attention_entropy', {'percentile': 75, 'mode': 'higher'})     # 1단계 결과 중 상위 25% 선택
    ]
    # 필터링 결과를 저장할 마스크 초기화 (처음에는 모든 샘플이 대상)
    selected_mask = np.ones(len(data_for_attention), dtype=bool)

    # 각 필터 순차적으로 적용
    for filter_step, (metric, settings) in enumerate(FILTERING_SEQUENCE):
        if metric not in data_for_attention.columns or MASKED_TEXT_COLUMN not in data_for_attention.columns:
            print(f"{metric} 또는 {MASKED_TEXT_COLUMN} 컬럼이 없어 건너뜁니다.")
            continue
        
        # 현재 선택된 샘플에서만 점수 추출
        current_selection = data_for_attention[selected_mask]
        scores = current_selection[metric].values
        scores = np.nan_to_num(scores, nan=0.0)
        
        # 현재 선택된 샘플에 대한 백분위수 계산
        if settings['mode'] == 'higher':
            threshold = np.percentile(scores, 100 - settings['percentile'])
            step_mask = scores >= threshold
            print(f"필터 {filter_step+1}: {metric} >= {threshold:.4f} (상위 {settings['percentile']}%)")
        else:  # 'lower'
            threshold = np.percentile(scores, settings['percentile'])
            step_mask = scores <= threshold
            print(f"필터 {filter_step+1}: {metric} <= {threshold:.4f} (하위 {settings['percentile']}%)")
        
        # 현재 선택 내에서 필터링된 인덱스 구하기
        filtered_indices = np.where(selected_mask)[0][step_mask]
        
        # 전체 마스크 업데이트
        selected_mask = np.zeros_like(selected_mask)
        selected_mask[filtered_indices] = True
        
        print(f"필터 {filter_step+1} 후 남은 샘플: {np.sum(selected_mask)} / {len(data_for_attention)}")

    # 최종 선택된 샘플 인덱스
    final_selected_indices = np.where(selected_mask)[0]

    if len(final_selected_indices) > 0:
        # OE 데이터셋에 마스크된 텍스트 저장 (기본 버전)
        oe_df_filtered = data_for_attention.iloc[final_selected_indices][[MASKED_TEXT_COLUMN]].copy()
        
        # 추가 정보를 포함한 확장 버전
        extended_columns = [MASKED_TEXT_COLUMN, TEXT_COLUMN, TOP_WORDS_COLUMN] + [m for m, _ in FILTERING_SEQUENCE]
        extended_columns = [col for col in extended_columns if col in data_for_attention.columns]
        oe_df_extended = data_for_attention.iloc[final_selected_indices][extended_columns].copy()
        
        # 파일명 생성
        filter_desc = "_".join([f"{m}_{s['mode']}_{s['percentile']}" for m, s in FILTERING_SEQUENCE])
        oe_filename = os.path.join(OE_DATA_DIR, f"oe_data_sequential_{filter_desc}.csv")
        oe_extended_filename = os.path.join(OE_DATA_DIR, f"oe_data_sequential_{filter_desc}_extended.csv")
        
        try:
            oe_df_filtered.to_csv(oe_filename, index=False)
            print(f"순차적 필터링 OE 데이터셋 ({len(oe_df_filtered)} 샘플) 저장: {oe_filename}")
            
            oe_df_extended.to_csv(oe_extended_filename, index=False)
            print(f"확장 OE 데이터셋 저장: {oe_extended_filename}")
            
            # 샘플 통계 정보 계산
            for metric in [m for m, _ in FILTERING_SEQUENCE]:
                if metric in oe_df_extended.columns:
                    values = oe_df_extended[metric].values
                    values = np.nan_to_num(values, nan=0.0)
                    print(f"선택된 샘플의 {metric} - 평균: {np.mean(values):.4f}, 중앙값: {np.median(values):.4f}, 최소: {np.min(values):.4f}, 최대: {np.max(values):.4f}")
        
        except Exception as e:
            print(f"OE 데이터셋 저장 중 오류: {e}")
    else:
        print("순차적 필터링 후 선택된 샘플이 없습니다.")

    for metric in metric_columns:
        if metric not in data_for_attention.columns or MASKED_TEXT_COLUMN not in data_for_attention.columns:
            print(f"{metric} 또는 {MASKED_TEXT_COLUMN} 컬럼이 없어 건너뜁니다.")
            continue
        
        scores = data_for_attention[metric].values
        scores = np.nan_to_num(scores, nan=0.0)
        
        # 해당 메트릭에 맞는 최적 설정 적용
        if metric in METRIC_SETTINGS:
            filter_percentile = METRIC_SETTINGS[metric]['percentile']
            filter_mode = METRIC_SETTINGS[metric]['mode']
        else:
            # 설정이 없는 경우 기본값 사용
            filter_percentile = OE_FILTER_PERCENTILE  
            filter_mode = OE_FILTER_MODE
        
        # 모드에 따라 임계값과 선택 로직 적용
        if filter_mode == 'higher':
            threshold = np.percentile(scores, 100 - filter_percentile)
            selected_indices = np.where(scores >= threshold)[0]
            mode_desc = f"top{filter_percentile}pct"
            print(f"{metric} >= {threshold:.4f} ({mode_desc}) 기준으로 OE 필터링...")
        elif filter_mode == 'lower':
            threshold = np.percentile(scores, filter_percentile)
            selected_indices = np.where(scores <= threshold)[0]
            mode_desc = f"bottom{filter_percentile}pct"
            print(f"{metric} <= {threshold:.4f} ({mode_desc}) 기준으로 OE 필터링...")
        else:
            selected_indices = data_for_attention.index
            mode_desc = "all"
            print(f"{metric}에 기반한 필터링을 적용하지 않습니다.")
        
        if len(selected_indices) > 0:
            # OE 데이터셋에 마스킹된 텍스트 저장
            oe_df_filtered = data_for_attention.iloc[selected_indices][[MASKED_TEXT_COLUMN]].copy()
            
            # 추가 정보를 포함한 확장 버전도 저장
            extended_columns = [MASKED_TEXT_COLUMN, TEXT_COLUMN, metric, TOP_WORDS_COLUMN]
            extended_columns = [col for col in extended_columns if col in data_for_attention.columns]
            oe_df_extended = data_for_attention.iloc[selected_indices][extended_columns].copy()
            
            # 기본 버전 저장
            oe_filename = os.path.join(OE_DATA_DIR, f"oe_data_{metric}_{mode_desc}.csv")
            # 확장 버전 저장
            oe_extended_filename = os.path.join(OE_DATA_DIR, f"oe_data_{metric}_{mode_desc}_extended.csv")
            
            try:
                oe_df_filtered.to_csv(oe_filename, index=False)
                print(f"OE 데이터셋 ({len(oe_df_filtered)} 샘플) 저장: {oe_filename}")
                
                oe_df_extended.to_csv(oe_extended_filename, index=False)
                print(f"확장 OE 데이터셋 저장: {oe_extended_filename}")
            except Exception as e:
                print(f"{metric} OE 데이터셋 저장 중 오류: {e}")
        else:
            print(f"{metric} {mode_desc} 기준으로 OE 샘플이 선택되지 않았습니다.")

    # t-SNE 시각화 부분 수정 (모든 지표 저장)
    print("\n--- 11. t-SNE 시각화 ---")
    if len(all_features) == len(data_for_attention):
        # 각 지표별 시각화 처리
        for metric in metric_columns:
            if metric not in data_for_attention.columns:
                print(f"{metric} 컬럼이 없어 t-SNE 시각화를 건너뜁니다.")
                continue
                
            # 각 개별 지표에 대한 OE 후보 인덱스 계산
            scores = data_for_attention[metric].values
            scores = np.nan_to_num(scores, nan=0.0)
            
            # 해당 메트릭에 맞는 최적 설정 적용
            if metric in METRIC_SETTINGS:
                filter_percentile = METRIC_SETTINGS[metric]['percentile']
                filter_mode = METRIC_SETTINGS[metric]['mode']
            else:
                filter_percentile = OE_FILTER_PERCENTILE  
                filter_mode = OE_FILTER_MODE
            
            if filter_mode == 'higher':
                threshold = np.percentile(scores, 100 - filter_percentile)
                oe_candidate_indices = np.where(scores >= threshold)[0]
                mode_desc = f"{filter_mode} {filter_percentile}%"
            else:  # 'lower'
                threshold = np.percentile(scores, filter_percentile)
                oe_candidate_indices = np.where(scores <= threshold)[0]
                mode_desc = f"{filter_mode} {filter_percentile}%"
            
            # t-SNE 레이블 준비
            tsne_labels = []
            known_label2id = log_data_module.label2id if log_data_module else {}
            unknown_class_lower = EXCLUDE_CLASS_FOR_TRAINING.lower()
            
            # 클래스 레이블 할당
            if CLASS_COLUMN in data_for_attention.columns:
                for cls in data_for_attention[CLASS_COLUMN]:
                    if isinstance(cls, str) and cls.lower() == unknown_class_lower:
                        tsne_labels.append(-1)  # Unknown 클래스
                    else:
                        tsne_labels.append(known_label2id.get(cls, -2) if isinstance(cls, str) else -2)  # Known 클래스 또는 기타
            else:
                # 클래스 컬럼이 없다면 모두 동일 레이블로 처리
                tsne_labels = [0] * len(data_for_attention)
            
            tsne_labels = np.array(tsne_labels)
            
            # 클래스 이름 매핑
            if log_data_module and hasattr(log_data_module, 'id2label'):
                tsne_class_names = {**log_data_module.id2label, -1: 'Unknown', -2: 'Other/Filtered'}
            else:
                tsne_class_names = {-1: 'Unknown', -2: 'Other/Filtered'}
            
            # 개별 지표에 대한 t-SNE 시각화
            plot_tsne(
                features=np.array(all_features),
                labels=tsne_labels,
                title=f't-SNE (Known vs Unknown vs OE Candidates by {metric} {mode_desc})',
                save_path=os.path.join(VIS_DIR, f'tsne_visualization_{metric}_{filter_mode}_{filter_percentile}pct.png'),
                highlight_indices=oe_candidate_indices,
                highlight_label=f'OE Candidate ({metric} {mode_desc})',
                class_names=tsne_class_names,
                seed=RANDOM_STATE
            )
            
            print(f"{metric} {mode_desc} 지표 t-SNE 시각화 저장 완료")
        
        # 순차적 필터링 결과에 대한 t-SNE 시각화 추가
        sequential_oe_indices = final_selected_indices
        
        if len(sequential_oe_indices) > 0:
            # 순차적 필터링에 대한 설명 문자열 생성
            seq_filter_desc = " + ".join([f"{m} {s['mode']} {s['percentile']}%" for m, s in FILTERING_SEQUENCE])
            
            # t-SNE 시각화 실행 (순차적 필터링)
            plot_tsne(
                features=np.array(all_features),
                labels=tsne_labels,
                title=f't-SNE (Known vs Unknown vs OE Candidates by Sequential Filtering)',
                save_path=os.path.join(VIS_DIR, f'tsne_visualization_sequential_filtering.png'),
                highlight_indices=sequential_oe_indices,
                highlight_label=f'OE Candidate (Sequential: {seq_filter_desc})',
                class_names=tsne_class_names,
                seed=RANDOM_STATE
            )
            
            print(f"순차적 필터링 결과 t-SNE 시각화 저장 완료")
        else:
            print("순차적 필터링 후 선택된 샘플이 없어 t-SNE 시각화를 건너뜁니다.")
    else:
        print(f"t-SNE 시각화 건너뛰기: 특징 벡터({len(all_features)})와 데이터({len(data_for_attention)}) 길이 불일치")

    # 12. 결과 요약 및 저장
    print("\n--- 12. 결과 요약 ---")
    # 상위 OE 데이터 샘플 확인
    if OE_FILTER_METRIC in data_for_attention.columns and MASKED_TEXT_COLUMN in data_for_attention.columns:
        scores = data_for_attention[OE_FILTER_METRIC].values
        sorted_indices = np.argsort(scores)
        
        if OE_FILTER_MODE == 'higher':
            sorted_indices = sorted_indices[::-1]  # 내림차순 (높은 값이 더 중요)
        
        print(f"\n상위 10개 OE 후보 샘플 ({OE_FILTER_METRIC} 기준):")
        for i, idx in enumerate(sorted_indices[:10]):
            score = data_for_attention.iloc[idx][OE_FILTER_METRIC]
            masked_text = data_for_attention.iloc[idx][MASKED_TEXT_COLUMN]
            original_text = data_for_attention.iloc[idx][TEXT_COLUMN] if TEXT_COLUMN in data_for_attention.columns else "N/A"
            top_words = data_for_attention.iloc[idx][TOP_WORDS_COLUMN] if TOP_WORDS_COLUMN in data_for_attention.columns else []
            
            print(f"\n{i+1}. {OE_FILTER_METRIC}: {score:.4f}")
            print(f"   원본: {original_text[:100]}..." if len(original_text) > 100 else f"   원본: {original_text}")
            print(f"   마스크됨: {masked_text[:100]}..." if len(masked_text) > 100 else f"   마스크됨: {masked_text}")
            print(f"   제거된 주요 단어: {', '.join(top_words[:5])}" + ("..." if len(top_words) > 5 else ""))
    
    # 결과 파일 경로 표시
    print("\n--- OE 데이터셋 추출 완료 ---")
    print(f"생성된 OE 데이터셋 저장 경로: {OE_DATA_DIR}")
    print(f"시각화 결과 저장 경로: {VIS_DIR}")
    print("\n저장된 주요 파일:")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            if file.endswith('.csv') or file.endswith('.png'):
                print(f"  - {os.path.join(root, file)}")

if __name__ == '__main__':
    main()