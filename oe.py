# -*- coding: utf-8 -*-
"""
Unified OE (Out-of-Distribution) Extractor
통합된 OE 추출기 - 모델 학습부터 OE 데이터셋 추출까지 전체 파이프라인
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
from sklearn.manifold import TSNE
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# 시각화 라이브러리
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False
    print("Warning: Seaborn not available")

# 텍스트 처리
import nltk
from nltk.tokenize import word_tokenize
import re
from collections import defaultdict
import json
import math
from typing import List, Dict, Tuple, Optional, Any, Union
from tqdm.auto import tqdm
import gc
from scipy.stats import entropy
import ast
from datetime import datetime

# NLTK 데이터 다운로드
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class Config:
    """설정 클래스 - 모든 하이퍼파라미터와 경로 관리"""
    
    # === 파일 경로 ===
    ORIGINAL_DATA_PATH = 'log_all_critical.csv'
    TEXT_COLUMN = 'text'
    CLASS_COLUMN = 'class'
    TRAIN_TEST_COLUMN = 'train/test'
    EXCLUDE_CLASS_FOR_TRAINING = "unknown"
    
    # === 출력 디렉토리 ===
    OUTPUT_DIR = 'unified_oe_extraction_results'
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "trained_model")
    LOG_DIR = os.path.join(OUTPUT_DIR, "lightning_logs")
    CONFUSION_MATRIX_DIR = os.path.join(LOG_DIR, "confusion_matrices")
    VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
    OE_DATA_DIR = os.path.join(OUTPUT_DIR, "extracted_oe_datasets")
    ATTENTION_DATA_DIR = os.path.join(OUTPUT_DIR, "attention_analysis")
    
    # === 모델 설정 ===
    MODEL_NAME = "roberta-base"
    MAX_LENGTH = 256
    BATCH_SIZE = 64
    NUM_TRAIN_EPOCHS = 20
    LEARNING_RATE = 2e-5
    MIN_SAMPLES_PER_CLASS_FOR_TRAIN_VAL = 2
    
    # === 하드웨어 설정 ===
    ACCELERATOR = "auto"
    DEVICES = "auto"
    PRECISION = "16-mixed" if torch.cuda.is_available() else "32-true"
    NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1
    
    # === 학습 설정 ===
    LOG_EVERY_N_STEPS = 50
    GRADIENT_CLIP_VAL = 1.0
    USE_WEIGHTED_LOSS = True
    USE_LR_SCHEDULER = True
    RANDOM_STATE = 42
    
    # === 어텐션 설정 ===
    ATTENTION_TOP_PERCENT = 0.10
    MIN_TOP_WORDS = 1
    TOP_K_ATTENTION = 3
    ATTENTION_LAYER = -1  # 마지막 레이어
    
    # === OE 필터링 설정 ===
    OE_FILTER_METRIC = 'removed_avg_attention'
    # 각 지표별 최적화된 설정
    # METRIC_SETTINGS = {
    #     'attention_entropy': {'percentile': 80, 'mode': 'higher'},
    #     'top_k_avg_attention': {'percentile': 20, 'mode': 'lower'},
    #     'max_attention': {'percentile': 20, 'mode': 'lower'},
    #     'removed_avg_attention': {'percentile': 80, 'mode': 'higher'}
    # }
    
    # 순차적 필터링 설정
    # FILTERING_SEQUENCE = [
    #     ('removed_avg_attention', {'percentile': 80, 'mode': 'higher'}),
    #     ('attention_entropy', {'percentile': 75, 'mode': 'higher'})
    # ]

    METRIC_SETTINGS = {
        'attention_entropy': {'percentile': 80, 'mode': 'higher'},        # 유지
        'top_k_avg_attention': {'percentile': 20, 'mode': 'lower'},       # 유지
        'max_attention': {'percentile': 15, 'mode': 'lower'},             # 20→15: 더 엄격하게
        'removed_avg_attention': {'percentile': 85, 'mode': 'higher'}     # 80→85: 분포가 0에 집중되어 있어서
    }

    # 순차적 필터링도 조정 고려
    FILTERING_SEQUENCE = [
        ('removed_avg_attention', {'percentile': 85, 'mode': 'higher'}),  # 80→85
        ('attention_entropy', {'percentile': 75, 'mode': 'higher'})       # 유지
    ]    
    # === 실행 단계 제어 ===
    STAGE_MODEL_TRAINING = True
    STAGE_ATTENTION_EXTRACTION = True
    STAGE_OE_EXTRACTION = True
    STAGE_VISUALIZATION = True
    
    @classmethod
    def create_directories(cls):
        """필요한 디렉토리 생성"""
        dirs = [
            cls.OUTPUT_DIR, cls.MODEL_SAVE_DIR, cls.LOG_DIR,
            cls.CONFUSION_MATRIX_DIR, cls.VIS_DIR, cls.OE_DATA_DIR,
            cls.ATTENTION_DATA_DIR
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def save_config(cls):
        """설정을 JSON 파일로 저장"""
        config_dict = {}
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                value = getattr(cls, attr)
                if isinstance(value, (str, int, float, bool, list, dict)):
                    config_dict[attr] = value
        
        with open(os.path.join(cls.OUTPUT_DIR, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {cls.OUTPUT_DIR}/config.json")


# === 헬퍼 함수들 ===
def set_seed(seed: int):
    """시드 설정"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)
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
    
    if not masked_sentence:
        return "__EMPTY_MASKED__"
    return masked_sentence


def safe_literal_eval(val):
    """문자열을 리스트로 안전하게 변환"""
    try:
        if isinstance(val, str) and val.strip().startswith('['):
            return ast.literal_eval(val)
        else:
            return []
    except Exception as e:
        print(f"Error parsing: {str(val)[:30]}... - {e}")
        return []


# === PyTorch Lightning 컴포넌트 ===
class UnifiedDataModule(pl.LightningDataModule):
    """통합된 데이터 모듈"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # 토크나이저 초기화
        print(f"Initializing tokenizer: {config.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # 데이터 저장을 위한 속성들
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
        """데이터 설정 및 전처리"""
        if self.df_full is None:
            print(f"Loading data from {self.config.ORIGINAL_DATA_PATH}")
            self.df_full = pd.read_csv(self.config.ORIGINAL_DATA_PATH)
            
            # 필수 컬럼 확인
            required_cols = [self.config.TEXT_COLUMN, self.config.CLASS_COLUMN]
            if not all(col in self.df_full.columns for col in required_cols):
                raise ValueError(f"Missing columns: {required_cols}")
            
            # 결측치 제거 및 클래스 정규화
            self.df_full = self.df_full.dropna(subset=[self.config.CLASS_COLUMN])
            self.df_full[self.config.CLASS_COLUMN] = self.df_full[self.config.CLASS_COLUMN].astype(str).str.lower()
            
            # 학습/검증용 데이터 준비 (exclude 클래스 제외)
            exclude_class_lower = self.config.EXCLUDE_CLASS_FOR_TRAINING.lower()
            df_known = self.df_full[self.df_full[self.config.CLASS_COLUMN] != exclude_class_lower].copy()
            print(f"Data size after excluding '{self.config.EXCLUDE_CLASS_FOR_TRAINING}': {len(df_known)}")
            
            # 레이블 매핑 생성
            known_classes_str = sorted(df_known[self.config.CLASS_COLUMN].unique())
            self.label2id = {label: i for i, label in enumerate(known_classes_str)}
            self.id2label = {i: label for label, i in self.label2id.items()}
            self.num_labels = len(known_classes_str)
            print(f"Label mapping complete: {self.num_labels} known classes")
            print(f"Label to ID mapping: {self.label2id}")
            
            # 수치 레이블 생성
            df_known['label'] = df_known[self.config.CLASS_COLUMN].map(self.label2id)
            df_known = df_known.dropna(subset=['label'])
            df_known['label'] = df_known['label'].astype(int)
            
            # 최소 샘플 수 필터링
            print(f"Filtering classes with minimum {self.config.MIN_SAMPLES_PER_CLASS_FOR_TRAIN_VAL} samples...")
            label_counts = df_known['label'].value_counts()
            valid_labels = label_counts[label_counts >= self.config.MIN_SAMPLES_PER_CLASS_FOR_TRAIN_VAL].index
            self.df_known_for_train_val = df_known[df_known['label'].isin(valid_labels)].copy()
            print(f"Data after filtering: {len(self.df_known_for_train_val)} samples")
            
            if len(self.df_known_for_train_val) == 0:
                raise ValueError("No data available after filtering")
            
            # 클래스 분포 출력
            print("\n--- Class distribution in train/val data ---")
            print(self.df_known_for_train_val['label'].map(self.id2label).value_counts())
            
            # 클래스 가중치 계산
            if self.config.USE_WEIGHTED_LOSS:
                self._compute_class_weights()
            
            # 학습/검증 분할
            self._split_train_val()
            
            # 토큰화
            self._tokenize_datasets()
    
    def _compute_class_weights(self):
        """클래스 가중치 계산"""
        labels_for_weights = self.df_known_for_train_val['label'].values
        unique_labels = np.unique(labels_for_weights)
        
        try:
            class_weights_array = compute_class_weight(
                'balanced', classes=unique_labels, y=labels_for_weights
            )
            self.class_weights = torch.ones(self.num_labels)
            for i, label_idx in enumerate(unique_labels):
                if label_idx < self.num_labels:
                    self.class_weights[label_idx] = class_weights_array[i]
            print(f"Computed class weights: {self.class_weights}")
        except ValueError as e:
            print(f"Error computing class weights: {e}. Using uniform weights.")
            self.config.USE_WEIGHTED_LOSS = False
            self.class_weights = None
    
    def _split_train_val(self):
        """학습/검증 데이터 분할"""
        print("Splitting train/validation data...")
        try:
            self.train_df_final, self.val_df_final = train_test_split(
                self.df_known_for_train_val, 
                test_size=0.2,
                random_state=self.config.RANDOM_STATE,
                stratify=self.df_known_for_train_val['label']
            )
        except ValueError:
            print("Warning: Stratified split failed. Using random split.")
            self.train_df_final, self.val_df_final = train_test_split(
                self.df_known_for_train_val,
                test_size=0.2,
                random_state=self.config.RANDOM_STATE
            )
        print(f"Final split - Train: {len(self.train_df_final)}, Val: {len(self.val_df_final)}")
    
    def _tokenize_datasets(self):
        """데이터셋 토큰화"""
        raw_datasets = DatasetDict({
            'train': Dataset.from_pandas(self.train_df_final),
            'validation': Dataset.from_pandas(self.val_df_final)
        })
        
        def tokenize_func(examples):
            return self.tokenizer(
                [preprocess_text_for_roberta(text) for text in examples[self.config.TEXT_COLUMN]],
                truncation=True,
                padding=False,
                max_length=self.config.MAX_LENGTH
            )
        
        print("Tokenizing datasets...")
        self.tokenized_train_val_datasets = raw_datasets.map(
            tokenize_func,
            batched=True,
            num_proc=max(1, self.config.NUM_WORKERS // 2),
            remove_columns=[col for col in raw_datasets['train'].column_names if col != 'label']
        )
        self.tokenized_train_val_datasets.set_format(
            type='torch', columns=['input_ids', 'attention_mask', 'label']
        )
        print("Tokenization complete.")
    
    def train_dataloader(self):
        return DataLoader(
            self.tokenized_train_val_datasets['train'],
            batch_size=self.config.BATCH_SIZE,
            collate_fn=self.data_collator,
            num_workers=self.config.NUM_WORKERS,
            shuffle=True,
            pin_memory=True,
            persistent_workers=self.config.NUM_WORKERS > 0
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.tokenized_train_val_datasets['validation'],
            batch_size=self.config.BATCH_SIZE,
            collate_fn=self.data_collator,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=self.config.NUM_WORKERS > 0
        )
    
    def get_full_dataframe(self):
        """전체 데이터프레임 반환"""
        if self.df_full is None:
            self.setup()
        return self.df_full


class UnifiedModel(pl.LightningModule):
    """통합된 PyTorch Lightning 모델"""
    
    def __init__(self, config: Config, num_labels: int, label2id: dict, id2label: dict, class_weights=None):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # 모델 초기화
        print(f"Initializing model: {config.MODEL_NAME} for {num_labels} classes")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
            output_attentions=True,
            output_hidden_states=True
        )
        
        # 손실 함수 설정
        if config.USE_WEIGHTED_LOSS and class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float))
            print("Using weighted CrossEntropyLoss")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            print("Using standard CrossEntropyLoss")
        
        # 메트릭 설정
        metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=num_labels),
            'f1_weighted': torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='weighted'),
            'f1_macro': torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='macro')
        })
        
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.val_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_labels)
    
    def setup(self, stage=None):
        if self.config.USE_WEIGHTED_LOSS and hasattr(self.loss_fn, 'weight') and self.loss_fn.weight is not None:
            self.loss_fn.weight = self.loss_fn.weight.to(self.device)
            print(f"Moved class weights to {self.device}")
    
    def forward(self, batch, output_features=False, output_attentions=False):
        """Forward pass with optional feature and attention extraction"""
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
        # 레이블 키 이름 통일
        if 'label' in batch:
            batch['labels'] = batch.pop('label')
        
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        
        return loss, preds, batch['labels']
    
    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        metrics_results = self.train_metrics(preds, labels)
        self.log_dict(metrics_results, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        metrics_results = self.val_metrics(preds, labels)
        self.val_cm.update(preds, labels)
        self.log_dict(metrics_results, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        try:
            val_cm_computed = self.val_cm.compute()
            print("\nValidation Confusion Matrix:")
            class_names = list(self.hparams.id2label.values())
            cm_df = pd.DataFrame(
                val_cm_computed.cpu().numpy(),
                index=class_names,
                columns=class_names
            )
            print(cm_df)
            
            # 혼동 행렬 저장
            cm_filename = os.path.join(
                self.config.CONFUSION_MATRIX_DIR,
                f"validation_cm_epoch_{self.current_epoch}.csv"
            )
            cm_df.to_csv(cm_filename)
            print(f"Confusion matrix saved: {cm_filename}")
        except Exception as e:
            print(f"Error computing validation confusion matrix: {e}")
        finally:
            self.val_cm.reset()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.LEARNING_RATE)
        
        if self.config.USE_LR_SCHEDULER:
            if (self.trainer and hasattr(self.trainer, 'estimated_stepping_batches') and 
                self.trainer.estimated_stepping_batches > 0):
                num_training_steps = self.trainer.estimated_stepping_batches
                print(f"Estimated training steps: {num_training_steps}")
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0,
                    num_training_steps=num_training_steps
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
                }
            else:
                print("Warning: Could not estimate training steps for scheduler")
                return optimizer
        else:
            return optimizer


# === 어텐션 분석 클래스 ===
class AttentionAnalyzer:
    """어텐션 분석 및 마스킹 담당 클래스"""
    
    def __init__(self, config: Config, model_pl: UnifiedModel, tokenizer, device):
        self.config = config
        self.model_pl = model_pl
        self.tokenizer = tokenizer
        self.device = device
        self.model_pl.to(device)
        self.model_pl.eval()
        self.model_pl.freeze()
    
    @torch.no_grad()
    def get_word_attention_scores(self, texts: List[str], layer_idx: int = -1) -> List[Dict[str, float]]:
        """배치 텍스트에 대한 단어별 어텐션 스코어 계산"""
        batch_size = self.config.BATCH_SIZE
        all_word_scores = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing attention scores"):
            batch_texts = texts[i:i+batch_size]
            batch_scores = self._process_attention_batch(batch_texts, layer_idx)
            all_word_scores.extend(batch_scores)
        
        return all_word_scores
    
    def _process_attention_batch(self, batch_texts: List[str], layer_idx: int) -> List[Dict[str, float]]:
        """단일 배치에 대한 어텐션 처리"""
        if not batch_texts:
            return []
        
        # 텍스트 전처리
        processed_texts = [preprocess_text_for_roberta(text) for text in batch_texts]
        
        # 토큰화 및 모델 입력 준비
        inputs = self.tokenizer(
            processed_texts,
            return_tensors='pt',
            truncation=True,
            max_length=self.config.MAX_LENGTH,
            padding=True,
            return_offsets_mapping=True
        )
        
        offset_mappings = inputs.pop('offset_mapping').cpu().numpy()
        input_ids_batch = inputs['input_ids'].cpu().numpy()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 모델 실행
        with torch.no_grad():
            outputs = self.model_pl.model(**inputs, output_attentions=True)
            attentions_batch = outputs.attentions[layer_idx].cpu().numpy()
        
        # 배치 내 각 샘플 처리
        batch_word_scores = []
        for i in range(len(batch_texts)):
            word_scores = self._extract_word_scores_from_attention(
                attentions_batch[i],
                input_ids_batch[i],
                offset_mappings[i],
                processed_texts[i]
            )
            batch_word_scores.append(word_scores)
        
        # 메모리 정리
        del inputs, outputs, attentions_batch
        return batch_word_scores
    
    def _extract_word_scores_from_attention(self, attention_sample, input_ids, offset_mapping, original_text):
        """단일 샘플에서 단어별 어텐션 스코어 추출"""
        # 평균 어텐션 계산 (모든 헤드의 평균)
        attention_heads_mean = np.mean(attention_sample, axis=0)
        cls_attentions = attention_heads_mean[0, :]  # CLS 토큰의 어텐션
        
        # 토큰을 단어로 매핑
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        word_scores = defaultdict(list)
        current_word_indices = []
        last_word_end_offset = 0
        
        for j, (token_id, offset) in enumerate(zip(input_ids, offset_mapping)):
            # 특수 토큰 제외
            if offset[0] == offset[1] or token_id in self.tokenizer.all_special_ids:
                continue
            
            # 새 단어인지 확인
            is_continuation = (j > 0 and offset[0] == last_word_end_offset)
            
            if not is_continuation and current_word_indices:
                # 이전 단어 처리
                start = offset_mapping[current_word_indices[0]][0]
                end = offset_mapping[current_word_indices[-1]][1]
                try:
                    word = original_text[start:end]
                    avg_score = np.mean(cls_attentions[current_word_indices])
                    if word.strip():
                        word_scores[word.strip()].append(avg_score)
                except IndexError:
                    pass
                current_word_indices = []
            
            current_word_indices.append(j)
            last_word_end_offset = offset[1]
        
        # 마지막 단어 처리
        if current_word_indices:
            start = offset_mapping[current_word_indices[0]][0]
            end = offset_mapping[current_word_indices[-1]][1]
            try:
                word = original_text[start:end]
                avg_score = np.mean(cls_attentions[current_word_indices])
                if word.strip():
                    word_scores[word.strip()].append(avg_score)
            except IndexError:
                pass
        
        # 단어별 평균 스코어 계산
        final_word_scores = {word: np.mean(scores) for word, scores in word_scores.items()}
        return final_word_scores
    
    def extract_top_attention_words(self, word_scores_dict: Dict[str, float]) -> List[str]:
        """어텐션 스코어에서 상위 단어 추출"""
        if not word_scores_dict:
            return []
        
        sorted_words = sorted(word_scores_dict.items(), key=lambda x: x[1], reverse=True)
        num_words = len(sorted_words)
        n_top = max(self.config.MIN_TOP_WORDS, math.ceil(num_words * self.config.ATTENTION_TOP_PERCENT))
        
        # 불용어 제외 (옵션)
        stopwords = {
            '__arg__', '__num__', '__id__', '__addr__', '__path__', '__netif__', '__version__', '__user__',
            'a', 'an', 'the', 'is', 'was', 'on', 'in', 'at', 'to', 'of', 'for', 'and', 'or', 'but',
            'error', 'failed', 'failure', 'critical', 'warning', 'device', 'system', 'detected',
            'has', 'been', 'not', 'are', 'with', 'due', 'because', 'than', 'its', 'from', 'this',
            'that', 'will', 'be'
        }
        
        top_words = [word for word, score in sorted_words[:n_top] 
                    if word.lower() not in stopwords]
        
        # 불용어 제거 후 단어가 없으면 원래 상위 단어 사용
        if not top_words and sorted_words:
            top_words = [word for word, score in sorted_words[:n_top]]
        
        return top_words
    
    def process_full_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """전체 데이터셋에 대한 어텐션 분석 및 마스킹"""
        print("Processing full dataset for attention analysis...")
        
        # 텍스트 추출
        texts = df[self.config.TEXT_COLUMN].tolist()
        
        # 어텐션 스코어 계산
        print("Computing word attention scores...")
        all_word_scores = self.get_word_attention_scores(texts, self.config.ATTENTION_LAYER)
        
        # 상위 어텐션 단어 추출
        print("Extracting top attention words...")
        all_top_words = []
        for word_scores in all_word_scores:
            top_words = self.extract_top_attention_words(word_scores)
            all_top_words.append(top_words)
        
        # 마스킹된 텍스트 생성
        print("Creating masked texts...")
        masked_texts = []
        for i, (text, top_words) in enumerate(zip(texts, all_top_words)):
            masked_text = create_masked_sentence(text, top_words)
            masked_texts.append(masked_text)
        
        # 결과를 데이터프레임에 추가
        result_df = df.copy()
        result_df['top_attention_words'] = all_top_words
        result_df['masked_text_attention'] = masked_texts
        
        return result_df


# === OE 추출 클래스 ===
class OEExtractor:
    """OE 데이터 추출 및 분석 담당 클래스"""
    
    def __init__(self, config: Config, model_pl: UnifiedModel, tokenizer, device):
        self.config = config
        self.model_pl = model_pl
        self.tokenizer = tokenizer
        self.device = device
        self.model_pl.to(device)
        self.model_pl.eval()
        self.model_pl.freeze()
    
    @torch.no_grad()
    def extract_attention_metrics(self, dataloader: DataLoader) -> Tuple[pd.DataFrame, List[np.ndarray]]:
        """어텐션 기반 지표 및 특징 벡터 추출"""
        attention_metrics = []
        features_list = []
        
        print("Extracting attention metrics and features...")
        for batch in tqdm(dataloader, desc="Processing batches"):
            # 배치 준비
            batch_for_model = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device)
            }
            
            # 모델 실행
            outputs = self.model_pl.forward(batch_for_model, output_features=True, output_attentions=True)
            
            # 어텐션 및 특징 추출
            attentions_batch = outputs.attentions[-1].cpu().numpy()  # 마지막 레이어
            features_batch = outputs.hidden_states[-1][:, 0, :].cpu().numpy()  # CLS 토큰
            features_list.extend(list(features_batch))
            
            # 배치 내 각 샘플 처리
            for i in range(len(batch['input_ids'])):
                metrics = self._compute_attention_metrics(
                    attentions_batch[i],
                    batch['input_ids'][i].cpu().numpy()
                )
                attention_metrics.append(metrics)
        
        df_metrics = pd.DataFrame(attention_metrics)
        
        # 메모리 정리
        del outputs, attentions_batch, features_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        return df_metrics, features_list
    
    def _compute_attention_metrics(self, attention_sample, input_ids):
        """단일 샘플에 대한 어텐션 지표 계산"""
        # 유효 토큰 인덱스 찾기 (특수 토큰 제외)
        valid_indices = np.where(
            (input_ids != self.tokenizer.pad_token_id) &
            (input_ids != self.tokenizer.cls_token_id) &
            (input_ids != self.tokenizer.sep_token_id)
        )[0]
        
        if len(valid_indices) == 0:
            return {
                'max_attention': 0,
                'top_k_avg_attention': 0,
                'attention_entropy': 0
            }
        
        # 평균 어텐션 계산 (모든 헤드)
        cls_attentions = np.mean(attention_sample[:, 0, :], axis=0)
        valid_attentions = cls_attentions[valid_indices]
        
        # 지표 계산
        max_attention = np.max(valid_attentions)
        
        # 상위 K개 평균
        k = min(self.config.TOP_K_ATTENTION, len(valid_attentions))
        top_k_avg_attention = np.mean(np.sort(valid_attentions)[-k:]) if k > 0 else 0
        
        # 어텐션 엔트로피
        attention_probs = F.softmax(torch.tensor(valid_attentions), dim=0).numpy()
        attention_entropy = entropy(attention_probs) if len(attention_probs) > 0 else 0
        
        return {
            'max_attention': max_attention,
            'top_k_avg_attention': top_k_avg_attention,
            'attention_entropy': attention_entropy
        }
    
    def compute_removed_word_attention(self, df: pd.DataFrame, attention_analyzer: AttentionAnalyzer) -> pd.DataFrame:
        """제거된 단어의 어텐션 스코어 계산"""
        print("Computing removed word attention scores...")
        
        if 'top_attention_words' not in df.columns or self.config.TEXT_COLUMN not in df.columns:
            print("Required columns not found for removed word attention calculation")
            df['removed_avg_attention'] = 0.0
            return df
        
        # 원본 텍스트에 대한 어텐션 스코어 계산
        texts = df[self.config.TEXT_COLUMN].tolist()
        word_attentions_list = attention_analyzer.get_word_attention_scores(texts)
        
        # 제거된 단어의 평균 어텐션 계산
        removed_attentions = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing removed attention"):
            top_words = row['top_attention_words'] if isinstance(row['top_attention_words'], list) else []
            
            if top_words and idx < len(word_attentions_list):
                word_scores = word_attentions_list[idx]
                removed_scores = [word_scores.get(word.lower(), 0) for word in top_words]
                removed_attentions.append(np.mean(removed_scores) if removed_scores else 0)
            else:
                removed_attentions.append(0)
        
        df['removed_avg_attention'] = removed_attentions
        print("Removed word attention computation complete.")
        return df
    
    def extract_oe_datasets(self, df: pd.DataFrame) -> None:
        """다양한 기준으로 OE 데이터셋 추출"""
        print("Extracting OE datasets with different criteria...")
        
        # 개별 지표별 OE 추출
        for metric, settings in self.config.METRIC_SETTINGS.items():
            if metric not in df.columns:
                print(f"Skipping {metric} - column not found")
                continue
            
            self._extract_single_metric_oe(df, metric, settings)
        
        # 순차적 필터링
        self._extract_sequential_filtering_oe(df)
    
    def _extract_single_metric_oe(self, df: pd.DataFrame, metric: str, settings: dict) -> None:
        """단일 지표 기반 OE 추출"""
        scores = np.nan_to_num(df[metric].values, nan=0.0)
        
        if settings['mode'] == 'higher':
            threshold = np.percentile(scores, 100 - settings['percentile'])
            selected_indices = np.where(scores >= threshold)[0]
            mode_desc = f"top{settings['percentile']}pct"
        else:  # 'lower'
            threshold = np.percentile(scores, settings['percentile'])
            selected_indices = np.where(scores <= threshold)[0]
            mode_desc = f"bottom{settings['percentile']}pct"
        
        if len(selected_indices) > 0:
            # 기본 OE 데이터셋 (마스킹된 텍스트만)
            oe_df = df.iloc[selected_indices][['masked_text_attention']].copy()
            
            # 확장 버전 (추가 정보 포함)
            extended_columns = ['masked_text_attention', self.config.TEXT_COLUMN, 'top_attention_words', metric]
            extended_columns = [col for col in extended_columns if col in df.columns]
            oe_df_extended = df.iloc[selected_indices][extended_columns].copy()
            
            # 저장
            oe_filename = os.path.join(self.config.OE_DATA_DIR, f"oe_data_{metric}_{mode_desc}.csv")
            oe_extended_filename = os.path.join(self.config.OE_DATA_DIR, f"oe_data_{metric}_{mode_desc}_extended.csv")
            
            oe_df.to_csv(oe_filename, index=False)
            oe_df_extended.to_csv(oe_extended_filename, index=False)
            
            print(f"Saved OE dataset ({len(oe_df)} samples): {oe_filename}")
    
    def _extract_sequential_filtering_oe(self, df: pd.DataFrame) -> None:
        """순차적 필터링 기반 OE 추출"""
        print("Applying sequential filtering...")
        
        selected_mask = np.ones(len(df), dtype=bool)
        
        for step, (metric, settings) in enumerate(self.config.FILTERING_SEQUENCE):
            if metric not in df.columns:
                print(f"Skipping filter step {step+1}: {metric} not found")
                continue
            
            # 현재 선택된 샘플에서 점수 추출
            current_selection = df[selected_mask]
            scores = np.nan_to_num(current_selection[metric].values, nan=0.0)
            
            # 필터 적용
            if settings['mode'] == 'higher':
                threshold = np.percentile(scores, 100 - settings['percentile'])
                step_mask = scores >= threshold
            else:
                threshold = np.percentile(scores, settings['percentile'])
                step_mask = scores <= threshold
            
            # 마스크 업데이트
            filtered_indices = np.where(selected_mask)[0][step_mask]
            selected_mask = np.zeros_like(selected_mask)
            selected_mask[filtered_indices] = True
            
            print(f"Filter {step+1}: {metric} {settings['mode']} {settings['percentile']}% -> {np.sum(selected_mask)} samples")
        
        # 결과 저장
        final_indices = np.where(selected_mask)[0]
        if len(final_indices) > 0:
            oe_df = df.iloc[final_indices][['masked_text_attention']].copy()
            
            # 확장 버전
            extended_columns = ['masked_text_attention', self.config.TEXT_COLUMN, 'top_attention_words']
            extended_columns.extend([m for m, _ in self.config.FILTERING_SEQUENCE if m in df.columns])
            oe_df_extended = df.iloc[final_indices][extended_columns].copy()
            
            # 파일명 생성
            filter_desc = "_".join([f"{m}_{s['mode']}_{s['percentile']}" 
                                  for m, s in self.config.FILTERING_SEQUENCE])
            
            oe_filename = os.path.join(self.config.OE_DATA_DIR, f"oe_data_sequential_{filter_desc}.csv")
            oe_extended_filename = os.path.join(self.config.OE_DATA_DIR, f"oe_data_sequential_{filter_desc}_extended.csv")
            
            oe_df.to_csv(oe_filename, index=False)
            oe_df_extended.to_csv(oe_extended_filename, index=False)
            
            print(f"Saved sequential OE dataset ({len(oe_df)} samples): {oe_filename}")


# === 시각화 클래스 ===
class Visualizer:
    """시각화 담당 클래스"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def plot_metric_distribution(self, scores: np.ndarray, metric_name: str, title: str, save_path: str):
        """지표 분포 시각화"""
        if len(scores) == 0:
            print(f"No scores for {metric_name}")
            return
        
        plt.figure(figsize=(10, 6))
        if SNS_AVAILABLE:
            sns.histplot(scores, bins=50, kde=True, stat='density')
        else:
            plt.hist(scores, bins=50, density=True, alpha=0.7)
        
        plt.title(title, fontsize=14)
        plt.xlabel(metric_name, fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(alpha=0.3)
        
        # 통계 정보 추가
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Distribution plot saved: {save_path}")

# === 시각화 클래스 ===
class Visualizer:
    """시각화 담당 클래스"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def plot_metric_distribution(self, scores: np.ndarray, metric_name: str, title: str, save_path: str):
        """지표 분포 시각화"""
        if len(scores) == 0:
            print(f"No scores for {metric_name}")
            return
        
        plt.figure(figsize=(10, 6))
        if SNS_AVAILABLE:
            sns.histplot(scores, bins=50, kde=True, stat='density')
        else:
            plt.hist(scores, bins=50, density=True, alpha=0.7)
        
        plt.title(title, fontsize=14)
        plt.xlabel(metric_name, fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(alpha=0.3)
        
        # 통계 정보 추가
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Distribution plot saved: {save_path}")
    
    def plot_tsne(self, features: np.ndarray, labels: np.ndarray, title: str, save_path: str,
                  highlight_indices: Optional[np.ndarray] = None, highlight_label: str = 'OE Candidate',
                  class_names: Optional[Dict] = None, seed: int = 42):
        """t-SNE 시각화"""
        if len(features) == 0:
            print("No features for t-SNE")
            return
        
        print(f"Running t-SNE on {features.shape[0]} samples...")
        try:
            # perplexity 조정
            perplexity = min(30, features.shape[0] - 1)
            tsne = TSNE(
                n_components=2,
                random_state=seed,
                perplexity=perplexity,
                n_iter=1000,
                init='pca',
                learning_rate='auto'
            )
            tsne_results = tsne.fit_transform(features)
        except Exception as e:
            print(f"Error running t-SNE: {e}")
            return
        
        # 데이터프레임 생성
        df_tsne = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
        df_tsne['label'] = labels
        df_tsne['is_highlighted'] = False
        
        if highlight_indices is not None:
            df_tsne.loc[highlight_indices, 'is_highlighted'] = True
        
        # 시각화
        plt.figure(figsize=(14, 10))
        
        # 클래스별 색상 설정
        unique_labels = sorted(df_tsne['label'].unique())
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        # 각 클래스 플롯
        for i, label_val in enumerate(unique_labels):
            subset = df_tsne[(df_tsne['label'] == label_val) & (~df_tsne['is_highlighted'])]
            if len(subset) > 0:
                class_name = class_names.get(label_val, f'Class {label_val}') if class_names else f'Class {label_val}'
                plt.scatter(subset['tsne1'], subset['tsne2'], color=colors[i], 
                          label=class_name, alpha=0.7, s=30)
        
        # 하이라이트된 포인트
        if highlight_indices is not None:
            highlight_subset = df_tsne[df_tsne['is_highlighted']]
            if len(highlight_subset) > 0:
                plt.scatter(highlight_subset['tsne1'], highlight_subset['tsne2'],
                          color='red', marker='x', s=100, label=highlight_label, alpha=0.9)
        
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel("t-SNE Dimension 1", fontsize=14)
        plt.ylabel("t-SNE Dimension 2", fontsize=14)
        plt.grid(alpha=0.3, linestyle='--')
        
        # 범례 설정
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                  fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"t-SNE plot saved: {save_path}")
    
    def visualize_all_metrics(self, df: pd.DataFrame):
        """모든 어텐션 지표 분포 시각화"""
        metric_columns = ['max_attention', 'top_k_avg_attention', 'attention_entropy', 'removed_avg_attention']
        
        for metric in metric_columns:
            if metric in df.columns and not df[metric].isnull().all():
                print(f"Visualizing {metric} distribution...")
                self.plot_metric_distribution(
                    df[metric].dropna().values,
                    metric,
                    f'Distribution of {metric}',
                    os.path.join(self.config.VIS_DIR, f'{metric}_distribution.png')
                )
            else:
                print(f"Skipping {metric} visualization")
    
    def visualize_oe_candidates(self, df: pd.DataFrame, features: List[np.ndarray], label2id: dict, id2label: dict):
        """OE 후보들을 t-SNE로 시각화"""
        if len(features) != len(df):
            print(f"Feature length mismatch: {len(features)} vs {len(df)}")
            return
        
        # 레이블 준비
        tsne_labels = []
        unknown_class_lower = self.config.EXCLUDE_CLASS_FOR_TRAINING.lower()
        
        for cls in df[self.config.CLASS_COLUMN]:
            if isinstance(cls, str) and cls.lower() == unknown_class_lower:
                tsne_labels.append(-1)  # Unknown
            else:
                tsne_labels.append(label2id.get(cls, -2) if isinstance(cls, str) else -2)
        
        tsne_labels = np.array(tsne_labels)
        class_names = {**id2label, -1: 'Unknown', -2: 'Other/Filtered'}
        
        # 각 지표별 시각화
        for metric, settings in self.config.METRIC_SETTINGS.items():
            if metric not in df.columns:
                continue
            
            scores = np.nan_to_num(df[metric].values, nan=0.0)
            
            if settings['mode'] == 'higher':
                threshold = np.percentile(scores, 100 - settings['percentile'])
                oe_indices = np.where(scores >= threshold)[0]
                mode_desc = f"top{settings['percentile']}%"
            else:
                threshold = np.percentile(scores, settings['percentile'])
                oe_indices = np.where(scores <= threshold)[0]
                mode_desc = f"bottom{settings['percentile']}%"
            
            self.plot_tsne(
                features=np.array(features),
                labels=tsne_labels,
                title=f't-SNE: OE Candidates by {metric} ({mode_desc})',
                save_path=os.path.join(self.config.VIS_DIR, f'tsne_{metric}_{settings["mode"]}_{settings["percentile"]}pct.png'),
                highlight_indices=oe_indices,
                highlight_label=f'OE Candidate ({metric} {mode_desc})',
                class_names=class_names,
                seed=self.config.RANDOM_STATE
            )
        
        # 순차적 필터링 시각화 추가
        if hasattr(self.config, 'FILTERING_SEQUENCE') and self.config.FILTERING_SEQUENCE:
            print("Creating sequential filtering visualization...")
            
            # 순차적 필터링 로직 재구현
            selected_mask = np.ones(len(df), dtype=bool)
            filter_steps = []
            
            for step, (metric, settings) in enumerate(self.config.FILTERING_SEQUENCE):
                if metric not in df.columns:
                    print(f"Skipping filter step {step+1}: {metric} not found")
                    continue
                
                # 현재 선택된 샘플에서 점수 추출
                current_selection = df[selected_mask]
                scores = np.nan_to_num(current_selection[metric].values, nan=0.0)
                
                # 필터 적용
                if settings['mode'] == 'higher':
                    threshold = np.percentile(scores, 100 - settings['percentile'])
                    step_mask = scores >= threshold
                    mode_desc = f"top{settings['percentile']}%"
                else:
                    threshold = np.percentile(scores, settings['percentile'])
                    step_mask = scores <= threshold
                    mode_desc = f"bottom{settings['percentile']}%"
                
                # 마스크 업데이트
                filtered_indices = np.where(selected_mask)[0][step_mask]
                selected_mask = np.zeros_like(selected_mask)
                selected_mask[filtered_indices] = True
                
                filter_steps.append((metric, settings, mode_desc, np.sum(selected_mask)))
            
            # 최종 선택된 인덱스
            final_indices = np.where(selected_mask)[0]
            
            if len(final_indices) > 0:
                # 파일명 생성
                filter_desc = "_".join([f"{m}_{s['mode']}_{s['percentile']}" 
                                      for m, s in self.config.FILTERING_SEQUENCE])
                
                # 필터링 단계별 설명 생성
                steps_desc = " -> ".join([f"{m}({mode_desc})" for m, s, mode_desc, count in filter_steps])
                total_samples = len(final_indices)
                
                # t-SNE 시각화
                self.plot_tsne(
                    features=np.array(features),
                    labels=tsne_labels,
                    title=f't-SNE: Sequential Filtering OE Candidates\n{steps_desc} -> {total_samples} samples',
                    save_path=os.path.join(self.config.VIS_DIR, f'tsne_sequential_{filter_desc}.png'),
                    highlight_indices=final_indices,
                    highlight_label=f'Sequential OE Candidate ({total_samples} samples)',
                    class_names=class_names,
                    seed=self.config.RANDOM_STATE
                )
                
                print(f"Sequential filtering visualization complete: {len(final_indices)} samples highlighted")
            else:
                print("Warning: No samples selected by sequential filtering - skipping visualization")
                

    def plot_tsne(self, features: np.ndarray, labels: np.ndarray, title: str, save_path: str,
                  highlight_indices: Optional[np.ndarray] = None, highlight_label: str = 'OE Candidate',
                  class_names: Optional[Dict] = None, seed: int = 42):
        """t-SNE 시각화"""
        if len(features) == 0:
            print("No features for t-SNE")
            return
        
        print(f"Running t-SNE on {features.shape[0]} samples...")
        try:
            # perplexity 조정
            perplexity = min(30, features.shape[0] - 1)
            tsne = TSNE(
                n_components=2,
                random_state=seed,
                perplexity=perplexity,
                n_iter=1000,
                init='pca',
                learning_rate='auto'
            )
            tsne_results = tsne.fit_transform(features)
        except Exception as e:
            print(f"Error running t-SNE: {e}")
            return
        
        # 데이터프레임 생성
        df_tsne = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
        df_tsne['label'] = labels
        df_tsne['is_highlighted'] = False
        
        if highlight_indices is not None:
            df_tsne.loc[highlight_indices, 'is_highlighted'] = True
        
        # 시각화
        plt.figure(figsize=(14, 10))
        
        # 클래스별 색상 설정
        unique_labels = sorted(df_tsne['label'].unique())
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        # 각 클래스 플롯
        for i, label_val in enumerate(unique_labels):
            subset = df_tsne[(df_tsne['label'] == label_val) & (~df_tsne['is_highlighted'])]
            if len(subset) > 0:
                class_name = class_names.get(label_val, f'Class {label_val}') if class_names else f'Class {label_val}'
                plt.scatter(subset['tsne1'], subset['tsne2'], color=colors[i], 
                          label=class_name, alpha=0.7, s=30)
        
        # 하이라이트된 포인트
        if highlight_indices is not None:
            highlight_subset = df_tsne[df_tsne['is_highlighted']]
            if len(highlight_subset) > 0:
                plt.scatter(highlight_subset['tsne1'], highlight_subset['tsne2'],
                          color='red', marker='x', s=100, label=highlight_label, alpha=0.9)
        
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel("t-SNE Dimension 1", fontsize=14)
        plt.ylabel("t-SNE Dimension 2", fontsize=14)
        plt.grid(alpha=0.3, linestyle='--')
        
        # 범례 설정
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                  fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"t-SNE plot saved: {save_path}")
    
    def visualize_all_metrics(self, df: pd.DataFrame):
        """모든 어텐션 지표 분포 시각화"""
        metric_columns = ['max_attention', 'top_k_avg_attention', 'attention_entropy', 'removed_avg_attention']
        
        for metric in metric_columns:
            if metric in df.columns and not df[metric].isnull().all():
                print(f"Visualizing {metric} distribution...")
                self.plot_metric_distribution(
                    df[metric].dropna().values,
                    metric,
                    f'Distribution of {metric}',
                    os.path.join(self.config.VIS_DIR, f'{metric}_distribution.png')
                )
            else:
                print(f"Skipping {metric} visualization")
    

# === 데이터셋 클래스 ===
class TextDataset(TorchDataset):
    """텍스트 데이터셋 클래스"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        
        print(f"Tokenizing {len(texts)} texts...")
        valid_texts = [str(t) if pd.notna(t) else "" for t in texts]
        self.encodings = tokenizer(
            valid_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        print("Tokenization complete.")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# === 메인 파이프라인 클래스 ===
class UnifiedOEExtractor:
    """통합된 OE 추출 파이프라인"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_module = None
        self.model = None
        self.attention_analyzer = None
        self.oe_extractor = None
        self.visualizer = Visualizer(config)
        
        # 디렉토리 생성
        config.create_directories()
        config.save_config()
        
        # 시드 설정
        set_seed(config.RANDOM_STATE)
    
    def run_stage1_model_training(self):
        """Stage 1: 모델 학습"""
        if not self.config.STAGE_MODEL_TRAINING:
            print("Skipping model training stage")
            return
        
        print("\n" + "="*50)
        print("STAGE 1: MODEL TRAINING")
        print("="*50)
        
        # 데이터 모듈 초기화
        print("Initializing data module...")
        self.data_module = UnifiedDataModule(self.config)
        self.data_module.setup()
        
        # 모델 초기화
        print("Initializing model...")
        self.model = UnifiedModel(
            config=self.config,
            num_labels=self.data_module.num_labels,
            label2id=self.data_module.label2id,
            id2label=self.data_module.id2label,
            class_weights=self.data_module.class_weights
        )
        
        # 트레이너 설정
        monitor_metric = 'val_f1_macro'
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.MODEL_SAVE_DIR,
            filename=f'best-model-{{epoch:02d}}-{{{monitor_metric}:.4f}}',
            save_top_k=1,
            monitor=monitor_metric,
            mode='max'
        )
        
        early_stopping_callback = EarlyStopping(
            monitor=monitor_metric,
            patience=3,
            mode='max',
            verbose=True
        )
        
        csv_logger = CSVLogger(save_dir=self.config.LOG_DIR, name="model_training")
        
        # 기존 모델 확인
        if self._check_existing_model():
            print("Found existing trained model. Skipping training...")
            self._load_existing_model(checkpoint_callback)
        else:
            # 학습 실행
            trainer = pl.Trainer(
                max_epochs=self.config.NUM_TRAIN_EPOCHS,
                accelerator=self.config.ACCELERATOR,
                devices=self.config.DEVICES,
                precision=self.config.PRECISION,
                logger=csv_logger,
                callbacks=[checkpoint_callback, early_stopping_callback],
                deterministic=False,
                log_every_n_steps=self.config.LOG_EVERY_N_STEPS,
                gradient_clip_val=self.config.GRADIENT_CLIP_VAL
            )
            
            print("Starting model training...")
            trainer.fit(self.model, datamodule=self.data_module)
            print("Model training complete!")
            
            # 최적 모델 로드
            self._load_best_model(checkpoint_callback)
    
    def run_stage2_attention_extraction(self):
        """Stage 2: 어텐션 추출 및 마스킹"""
        if not self.config.STAGE_ATTENTION_EXTRACTION:
            print("Skipping attention extraction stage")
            return
        
        print("\n" + "="*50)
        print("STAGE 2: ATTENTION EXTRACTION")
        print("="*50)
        
        # 모델 로드 (Stage 1을 건너뛴 경우)
        if self.model is None:
            self._load_existing_model()
        
        # 어텐션 분석기 초기화
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_analyzer = AttentionAnalyzer(
            config=self.config,
            model_pl=self.model,
            tokenizer=self.data_module.tokenizer,
            device=device
        )
        
        # 전체 데이터셋에 대한 어텐션 분석
        print("Processing full dataset for attention analysis...")
        full_df = self.data_module.get_full_dataframe()
        processed_df = self.attention_analyzer.process_full_dataset(full_df)
        
        # 결과 저장
        output_path = os.path.join(
            self.config.ATTENTION_DATA_DIR,
            f"{os.path.splitext(os.path.basename(self.config.ORIGINAL_DATA_PATH))[0]}_with_attention.csv"
        )
        processed_df.to_csv(output_path, index=False)
        print(f"Attention analysis results saved: {output_path}")
        
        # 샘플 출력
        self._print_attention_samples(processed_df)
        
        return processed_df
    
    def run_stage3_oe_extraction(self, df_with_attention=None):
        """Stage 3: OE 데이터 추출"""
        if not self.config.STAGE_OE_EXTRACTION:
            print("Skipping OE extraction stage")
            return
        
        print("\n" + "="*50)
        print("STAGE 3: OE EXTRACTION")
        print("="*50)
        
        # 어텐션 분석 결과 로드 (Stage 2를 건너뛴 경우)
        if df_with_attention is None:
            df_with_attention = self._load_attention_results()
        
        # OE 추출기 초기화
        if self.oe_extractor is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.oe_extractor = OEExtractor(
                config=self.config,
                model_pl=self.model,
                tokenizer=self.data_module.tokenizer,
                device=device
            )
        
        # 마스킹된 텍스트에 대한 어텐션 지표 계산
        print("Computing attention metrics for masked texts...")
        masked_texts = df_with_attention['masked_text_attention'].tolist()
        dummy_labels = [-1] * len(masked_texts)
        
        dataset = TextDataset(
            masked_texts,
            dummy_labels,
            self.data_module.tokenizer,
            self.config.MAX_LENGTH
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            shuffle=False
        )
        
        attention_metrics_df, features = self.oe_extractor.extract_attention_metrics(dataloader)
        
        # 어텐션 지표를 데이터프레임에 결합
        if len(df_with_attention) == len(attention_metrics_df):
            df_with_metrics = pd.concat(
                [df_with_attention.reset_index(drop=True), attention_metrics_df.reset_index(drop=True)],
                axis=1
            )
        else:
            print(f"Warning: Length mismatch. Using original dataframe.")
            df_with_metrics = df_with_attention.copy()
        
        # 제거된 단어의 어텐션 계산
        if self.attention_analyzer is not None:
            df_with_metrics = self.oe_extractor.compute_removed_word_attention(
                df_with_metrics, self.attention_analyzer
            )
        
        # OE 데이터셋 추출
        self.oe_extractor.extract_oe_datasets(df_with_metrics)
        
        # 결과 저장
        metrics_output_path = os.path.join(
            self.config.ATTENTION_DATA_DIR,
            f"{os.path.splitext(os.path.basename(self.config.ORIGINAL_DATA_PATH))[0]}_with_all_metrics.csv"
        )
        df_with_metrics.to_csv(metrics_output_path, index=False)
        print(f"Complete metrics saved: {metrics_output_path}")
        
        return df_with_metrics, features
    
    def run_stage4_visualization(self, df_with_metrics=None, features=None):
        """Stage 4: 시각화"""
        if not self.config.STAGE_VISUALIZATION:
            print("Skipping visualization stage")
            return
        
        print("\n" + "="*50)
        print("STAGE 4: VISUALIZATION")
        print("="*50)
        
        # 데이터 로드 (이전 단계를 건너뛴 경우)
        if df_with_metrics is None or features is None:
            df_with_metrics, features = self._load_final_results()
        
        # 지표 분포 시각화
        print("Creating metric distribution plots...")
        self.visualizer.visualize_all_metrics(df_with_metrics)
        
        # t-SNE 시각화
        if features is not None and len(features) > 0:
            print("Creating t-SNE visualizations...")
            self.visualizer.visualize_oe_candidates(
                df_with_metrics,
                features,
                self.data_module.label2id,
                self.data_module.id2label
            )
        
        print("All visualizations complete!")
    
    def run_full_pipeline(self):
        """전체 파이프라인 실행"""
        print("Starting Unified OE Extraction Pipeline...")
        print(f"Configuration: {self.config.__dict__}")
        
        # Stage 1: 모델 학습
        self.run_stage1_model_training()
        
        # Stage 2: 어텐션 추출
        df_with_attention = self.run_stage2_attention_extraction()
        
        # Stage 3: OE 추출
        df_with_metrics, features = self.run_stage3_oe_extraction(df_with_attention)
        
        # Stage 4: 시각화
        self.run_stage4_visualization(df_with_metrics, features)
        
        # 최종 요약
        self._print_final_summary()
        
        print("\nUnified OE Extraction Pipeline Complete!")
    
    # === 헬퍼 메서드들 ===
    def _check_existing_model(self) -> bool:
        """기존 모델 확인"""
        return (os.path.exists(self.config.MODEL_SAVE_DIR) and 
                any(file.endswith('.ckpt') for file in os.listdir(self.config.MODEL_SAVE_DIR)))
    
    def _load_existing_model(self, checkpoint_callback=None):
        """기존 모델 로드"""
        if checkpoint_callback and hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path:
            model_path = checkpoint_callback.best_model_path
        else:
            # 가장 최근 체크포인트 찾기
            checkpoint_files = [f for f in os.listdir(self.config.MODEL_SAVE_DIR) if f.endswith('.ckpt')]
            if not checkpoint_files:
                raise FileNotFoundError("No checkpoint files found")
            model_path = os.path.join(self.config.MODEL_SAVE_DIR, sorted(checkpoint_files)[-1])
        
        print(f"Loading model from: {model_path}")
        
        # 데이터 모듈이 없으면 초기화
        if self.data_module is None:
            self.data_module = UnifiedDataModule(self.config)
            self.data_module.setup()
        
        # 모델 로드
        self.model = UnifiedModel.load_from_checkpoint(
            model_path,
            config=self.config,
            confusion_matrix_dir=self.config.CONFUSION_MATRIX_DIR,
            weights_only=True
        )
        print("Model loaded successfully!")
    
    def _load_best_model(self, checkpoint_callback):
        """최적 모델 로드"""
        if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
            print(f"Loading best model: {checkpoint_callback.best_model_path}")
            self.model = UnifiedModel.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                config=self.config,
                confusion_matrix_dir=self.config.CONFUSION_MATRIX_DIR,
                weights_only=True
            )
            print("Best model loaded successfully!")
        else:
            print("Warning: Best model path not found. Using current model state.")
    
    def _load_attention_results(self) -> pd.DataFrame:
        """어텐션 분석 결과 로드"""
        attention_file = os.path.join(
            self.config.ATTENTION_DATA_DIR,
            f"{os.path.splitext(os.path.basename(self.config.ORIGINAL_DATA_PATH))[0]}_with_attention.csv"
        )
        
        if os.path.exists(attention_file):
            print(f"Loading attention results from: {attention_file}")
            df = pd.read_csv(attention_file)
            # top_attention_words 컬럼 변환
            if 'top_attention_words' in df.columns:
                df['top_attention_words'] = df['top_attention_words'].apply(safe_literal_eval)
            return df
        else:
            raise FileNotFoundError(f"Attention results file not found: {attention_file}")
    
    def _load_final_results(self) -> Tuple[pd.DataFrame, List[np.ndarray]]:
        """최종 결과 로드"""
        metrics_file = os.path.join(
            self.config.ATTENTION_DATA_DIR,
            f"{os.path.splitext(os.path.basename(self.config.ORIGINAL_DATA_PATH))[0]}_with_all_metrics.csv"
        )
        
        if os.path.exists(metrics_file):
            print(f"Loading final results from: {metrics_file}")
            df = pd.read_csv(metrics_file)
            # top_attention_words 컬럼 변환
            if 'top_attention_words' in df.columns:
                df['top_attention_words'] = df['top_attention_words'].apply(safe_literal_eval)
            
            # features는 다시 계산해야 함 (저장되지 않음)
            print("Recomputing features for visualization...")
            if self.model is None:
                self._load_existing_model()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            oe_extractor = OEExtractor(
                config=self.config,
                model_pl=self.model,
                tokenizer=self.data_module.tokenizer,
                device=device
            )
            
            # 마스킹된 텍스트에 대한 특징 추출
            masked_texts = df['masked_text_attention'].tolist()
            dummy_labels = [-1] * len(masked_texts)
            
            dataset = TextDataset(
                masked_texts,
                dummy_labels,
                self.data_module.tokenizer,
                self.config.MAX_LENGTH
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                num_workers=self.config.NUM_WORKERS,
                shuffle=False
            )
            
            _, features = oe_extractor.extract_attention_metrics(dataloader)
            
            return df, features
        else:
            raise FileNotFoundError(f"Final results file not found: {metrics_file}")
    
    def _print_attention_samples(self, df: pd.DataFrame, num_samples: int = 5):
        """어텐션 분석 샘플 출력"""
        print(f"\n--- Attention Analysis Samples (Top {num_samples}) ---")
        sample_indices = df.sample(min(num_samples, len(df))).index
        
        for i in sample_indices:
            print("-" * 50)
            print(f"Original: {df.loc[i, self.config.TEXT_COLUMN]}")
            print(f"Top Words: {df.loc[i, 'top_attention_words']}")
            print(f"Masked: {df.loc[i, 'masked_text_attention']}")
    
    def _print_final_summary(self):
        """최종 요약 출력"""
        print("\n" + "="*50)
        print("PIPELINE SUMMARY")
        print("="*50)
        print(f"Output directory: {self.config.OUTPUT_DIR}")
        print(f"Model saved in: {self.config.MODEL_SAVE_DIR}")
        print(f"OE datasets saved in: {self.config.OE_DATA_DIR}")
        print(f"Visualizations saved in: {self.config.VIS_DIR}")
        print(f"Attention analysis saved in: {self.config.ATTENTION_DATA_DIR}")
        
        # 생성된 파일 목록
        print("\nGenerated Files:")
        for root, dirs, files in os.walk(self.config.OUTPUT_DIR):
            for file in files:
                if file.endswith(('.csv', '.png', '.json')):
                    print(f"  - {os.path.join(root, file)}")


# === 메인 함수 ===
def main():
    """메인 함수 - 명령행 인자 처리 및 파이프라인 실행"""
    parser = argparse.ArgumentParser(description="Unified OE Extraction Pipeline")
    
    # 파일 경로 인자
    parser.add_argument('--data_path', type=str, default='log_all_critical.csv',
                        help='Path to the input CSV file')
    parser.add_argument('--output_dir', type=str, default='unified_oe_extraction_results',
                        help='Output directory for all results')
    
    # 실행 단계 제어
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training stage')
    parser.add_argument('--skip-attention', action='store_true',
                        help='Skip attention extraction stage')
    parser.add_argument('--skip-oe', action='store_true',
                        help='Skip OE extraction stage')
    parser.add_argument('--skip-viz', action='store_true',
                        help='Skip visualization stage')
    
    # 모델 설정
    parser.add_argument('--model-name', type=str, default='roberta-base',
                        help='Pretrained model name')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training and inference')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate')
    
    # 어텐션 설정
    parser.add_argument('--attention-percent', type=float, default=0.10,
                        help='Percentage of top attention words to use')
    parser.add_argument('--min-top-words', type=int, default=1,
                        help='Minimum number of top words to extract')
    
    args = parser.parse_args()
    
    # 설정 업데이트
    Config.ORIGINAL_DATA_PATH = args.data_path
    Config.OUTPUT_DIR = args.output_dir
    Config.MODEL_NAME = args.model_name
    Config.BATCH_SIZE = args.batch_size
    Config.NUM_TRAIN_EPOCHS = args.epochs
    Config.LEARNING_RATE = args.learning_rate
    Config.ATTENTION_TOP_PERCENT = args.attention_percent
    Config.MIN_TOP_WORDS = args.min_top_words
    
    # 단계 제어
    Config.STAGE_MODEL_TRAINING = not args.skip_training
    Config.STAGE_ATTENTION_EXTRACTION = not args.skip_attention
    Config.STAGE_OE_EXTRACTION = not args.skip_oe
    Config.STAGE_VISUALIZATION = not args.skip_viz
    
    # 경로 업데이트
    Config.MODEL_SAVE_DIR = os.path.join(Config.OUTPUT_DIR, "trained_model")
    Config.LOG_DIR = os.path.join(Config.OUTPUT_DIR, "lightning_logs")
    Config.CONFUSION_MATRIX_DIR = os.path.join(Config.LOG_DIR, "confusion_matrices")
    Config.VIS_DIR = os.path.join(Config.OUTPUT_DIR, "visualizations")
    Config.OE_DATA_DIR = os.path.join(Config.OUTPUT_DIR, "extracted_oe_datasets")
    Config.ATTENTION_DATA_DIR = os.path.join(Config.OUTPUT_DIR, "attention_analysis")
    
    print("Starting Unified OE Extraction Pipeline...")
    print(f"Input file: {Config.ORIGINAL_DATA_PATH}")
    print(f"Output directory: {Config.OUTPUT_DIR}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.NUM_TRAIN_EPOCHS}")
    
    # 파이프라인 실행
    extractor = UnifiedOEExtractor(Config)
    extractor.run_full_pipeline()


if __name__ == '__main__':
    main()