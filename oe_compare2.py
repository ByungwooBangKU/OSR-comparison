"""
Enhanced Unified OE (Out-of-Distribution) Extractor with NLP Dataset Support
Including 20 Newsgroups, TREC, SST, and WikiText-2 for Outlier Exposure comparison
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
from torch.utils.data import DataLoader, Dataset as TorchDataset, Subset, random_split, ConcatDataset
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    RobertaConfig,
    AdamW,
    GPT2Tokenizer
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

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
    print("Warning: Seaborn not available. Some plots might not be generated.")

# 텍스트 처리
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
from collections import defaultdict
import json
import math
from typing import List, Dict, Tuple, Optional, Any, Union
from tqdm import tqdm
import gc
from scipy.stats import entropy
import ast
from datetime import datetime
import random
import requests
import zipfile

# NLTK 데이터 다운로드
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

import nltk
nltk.download('punkt_tab')
  
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Enhanced Configuration Class ---
class Config:
    """Enhanced Configuration Class - Supporting both Syslog and NLP datasets"""
    
    # === 기본 모드 선택 ===
    EXPERIMENT_MODE = "nlp"  # "syslog" 또는 "nlp"
    
    # === NLP Dataset 설정 ===
    NLP_DATASETS = {
        '20newsgroups': {
            'name': '20newsgroups',
            'subset': None,
            'text_column': 'text',
            'label_column': 'label'
        },
        'trec': {
            'name': 'trec',
            'subset': None,
            'text_column': 'text',
            'label_column': 'label-coarse'
        },
        'sst2': {
            'name': 'sst2',
            'subset': None,
            'text_column': 'sentence',
            'label_column': 'label'
        }
    }
    
    # === NLP 특화 설정 ===
    CURRENT_NLP_DATASET = '20newsgroups'  # 실험할 데이터셋 선택
    WIKITEXT_VERSION = 'wikitext-2-raw-v1'  # WikiText-2 버전
    
    # === 기존 Syslog 설정 (호환성 유지) ===
    ORIGINAL_DATA_PATH = 'data_syslog/log_all_critical.csv'
    TEXT_COLUMN = 'text'
    CLASS_COLUMN = 'class'
    EXCLUDE_CLASS_FOR_TRAINING = "unknown"
    
    # === 출력 디렉토리 설정 ===
    OUTPUT_DIR = 'enhanced_oe_nlp_results'
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "base_classifier_model")
    LOG_DIR = os.path.join(OUTPUT_DIR, "lightning_logs")
    CONFUSION_MATRIX_DIR = os.path.join(LOG_DIR, "confusion_matrices")
    VIS_DIR = os.path.join(OUTPUT_DIR, "oe_extraction_visualizations")
    OE_DATA_DIR = os.path.join(OUTPUT_DIR, "extracted_oe_datasets")
    ATTENTION_DATA_DIR = os.path.join(OUTPUT_DIR, "attention_analysis")
    NLP_DATA_DIR = os.path.join(OUTPUT_DIR, "nlp_datasets")
    
    # === NLP 모델 설정 ===
    NLP_MODEL_TYPE = "gru"  # "gru", "lstm", 또는 "transformer"
    NLP_VOCAB_SIZE = 10000
    NLP_EMBED_DIM = 300
    NLP_HIDDEN_DIM = 512
    NLP_NUM_LAYERS = 2
    NLP_DROPOUT = 0.3
    NLP_MAX_LENGTH = 512
    NLP_BATCH_SIZE = 256
    NLP_NUM_EPOCHS = 30
    NLP_LEARNING_RATE = 1e-3
    
    # === 기존 Vision 모델 설정 ===
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
    LOG_EVERY_N_STEPS = 70
    GRADIENT_CLIP_VAL = 1.0
    USE_WEIGHTED_LOSS = True
    USE_LR_SCHEDULER = True
    RANDOM_STATE = 42
    
    # === 어텐션 설정 ===
    ATTENTION_TOP_PERCENT = 0.20
    MIN_TOP_WORDS = 1
    TOP_K_ATTENTION = 3
    ATTENTION_LAYER = -1
    
    # === OE 필터링 설정 ===
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
    TEXT_COLUMN_IN_OE_FILES = 'masked_text_attention'
    
    # === OSR Experiment Settings ===
    OSR_EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, "osr_experiments")
    OSR_MODEL_DIR = os.path.join(OSR_EXPERIMENT_DIR, "models")
    OSR_RESULT_DIR = os.path.join(OSR_EXPERIMENT_DIR, "results")
    
    # NLP OSR 설정
    OSR_NLP_MODEL_TYPE = "roberta-base"  # NLP용 OSR 모델
    OSR_NLP_VOCAB_SIZE = 10000
    OSR_NLP_EMBED_DIM = 300
    OSR_NLP_HIDDEN_DIM = 512
    OSR_NLP_NUM_LAYERS = 2
    OSR_NLP_DROPOUT = 0.3
    OSR_NLP_MAX_LENGTH = 512
    OSR_NLP_BATCH_SIZE = 64
    OSR_NLP_NUM_EPOCHS = 20
    OSR_NLP_LEARNING_RATE = 1e-3
    
    # 기존 Vision OSR 설정
    OSR_MODEL_TYPE = 'roberta-base'
    OSR_MAX_LENGTH = 128
    OSR_BATCH_SIZE = 64
    OSR_NUM_EPOCHS = 30
    OSR_LEARNING_RATE = 2e-5
    OSR_OE_LAMBDA = 1.0
    OSR_TEMPERATURE = 1.0
    OSR_THRESHOLD_PERCENTILE = 5.0
    OSR_NUM_DATALOADER_WORKERS = NUM_WORKERS
    
    # Early stopping 설정
    OSR_EARLY_STOPPING_PATIENCE = 5
    OSR_EARLY_STOPPING_MIN_DELTA = 0.001
    OSR_WARMUP_RATIO = 0.1
    OSR_LR_DECAY_FACTOR = 0.5
    OSR_LR_PATIENCE = 3
    
    # === 실행 단계 제어 ===
    STAGE_MODEL_TRAINING = True
    STAGE_ATTENTION_EXTRACTION = True
    STAGE_OE_EXTRACTION = True
    STAGE_VISUALIZATION = True
    STAGE_OSR_EXPERIMENTS = True
    
    # === Flags ===
    OSR_SAVE_MODEL_PER_EXPERIMENT = True
    OSR_EVAL_ONLY = False
    OSR_NO_PLOT_PER_EXPERIMENT = False
    OSR_SKIP_STANDARD_MODEL = False
    
    # === HuggingFace Cache ===
    DATA_DIR_EXTERNAL_HF = os.path.join(OUTPUT_DIR, 'data_external_hf')
    CACHE_DIR_HF = os.path.join(DATA_DIR_EXTERNAL_HF, "hf_cache")
    
    @classmethod
    def create_directories(cls):
        """필요한 디렉토리 생성"""
        dirs = [
            cls.OUTPUT_DIR, cls.MODEL_SAVE_DIR, cls.LOG_DIR,
            cls.CONFUSION_MATRIX_DIR, cls.VIS_DIR, cls.OE_DATA_DIR,
            cls.ATTENTION_DATA_DIR, cls.NLP_DATA_DIR,
            cls.OSR_EXPERIMENT_DIR, cls.OSR_MODEL_DIR, cls.OSR_RESULT_DIR,
            cls.DATA_DIR_EXTERNAL_HF, cls.CACHE_DIR_HF
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def save_config(cls, filepath=None):
        """설정을 JSON 파일로 저장"""
        if filepath is None:
            filepath = os.path.join(cls.OUTPUT_DIR, 'config_enhanced_nlp.json')
        
        config_dict = {}
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                value = getattr(cls, attr)
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    config_dict[attr] = value
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        print(f"Configuration saved to {filepath}")

# === 헬퍼 함수들 ===
DEVICE_OSR = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    """시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    print(f"Seed set to {seed}")

def preprocess_text_for_nlp(text):
    """NLP를 위한 텍스트 전처리"""
    if not isinstance(text, str):
        return ""
    # 기본 클리닝
    text = re.sub(r'\s+', ' ', text).strip()
    # 특수문자 처리 (선택적)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

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
    
    processed_text = preprocess_text_for_nlp(original_text)
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
        elif isinstance(val, list):
            return val
        else:
            return []
    except (ValueError, SyntaxError) as e:
        return []

# === NLP 데이터셋 로더들 ===
class NLPDatasetLoader:
    """NLP 데이터셋 로더 클래스"""
    
    @staticmethod
    def load_20newsgroups():
        """20 Newsgroups 데이터셋 로드"""
        print("Loading 20 Newsgroups dataset...")
        try:
            # HuggingFace datasets 사용
            dataset = load_dataset("SetFit/20_newsgroups", cache_dir=Config.CACHE_DIR_HF)
            
            # train/test 분할
            train_data = []
            train_labels = []
            test_data = []
            test_labels = []
            
            for item in dataset['train']:
                train_data.append(item['text'])
                train_labels.append(item['label'])
                
            for item in dataset['test']:
                test_data.append(item['text'])
                test_labels.append(item['label'])
            
            return {
                'train': {'text': train_data, 'label': train_labels},
                'test': {'text': test_data, 'label': test_labels}
            }
        except Exception as e:
            print(f"Error loading 20 Newsgroups: {e}")
            return None
    
    @staticmethod
    def load_trec():
        """TREC 데이터셋 로드"""
        print("Loading TREC dataset...")
        try:
            # HuggingFace datasets 사용
            dataset = load_dataset("trec", cache_dir=Config.CACHE_DIR_HF)
            
            train_data = []
            train_labels = []
            test_data = []
            test_labels = []
            
            for item in dataset['train']:
                train_data.append(item['text'])
                train_labels.append(item['label-coarse'])
                
            for item in dataset['test']:
                test_data.append(item['text'])
                test_labels.append(item['label-coarse'])
            
            return {
                'train': {'text': train_data, 'label': train_labels},
                'test': {'text': test_data, 'label': test_labels}
            }
        except Exception as e:
            print(f"Error loading TREC: {e}")
            return None
    
    @staticmethod
    def load_sst2():
        """SST-2 데이터셋 로드"""
        print("Loading SST-2 dataset...")
        try:
            # HuggingFace datasets 사용
            dataset = load_dataset("sst2", cache_dir=Config.CACHE_DIR_HF)
            
            train_data = []
            train_labels = []
            test_data = []
            test_labels = []
            
            for item in dataset['train']:
                if item['sentence'] and item['label'] is not None:
                    train_data.append(item['sentence'])
                    train_labels.append(item['label'])
                    
            for item in dataset['validation']:
                if item['sentence'] and item['label'] is not None:
                    test_data.append(item['sentence'])
                    test_labels.append(item['label'])
            
            return {
                'train': {'text': train_data, 'label': train_labels},
                'test': {'text': test_data, 'label': test_labels}
            }
        except Exception as e:
            print(f"Error loading SST-2: {e}")
            return None
    
    @staticmethod
    def load_wikitext2():
        """WikiText-2 데이터셋 로드"""
        print("Loading WikiText-2 dataset...")
        try:
            # HuggingFace datasets 사용
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=Config.CACHE_DIR_HF)
            
            # 텍스트 추출 및 전처리
            train_texts = []
            for item in dataset['train']:
                if item['text'].strip():
                    # 문장으로 분할
                    sentences = sent_tokenize(item['text'])
                    for sent in sentences:
                        sent = sent.strip()
                        if len(sent) > 10:  # 너무 짧은 문장 제외
                            train_texts.append(sent)
            
            # 라벨 없음 (언어 모델링용이므로)
            return {
                'train': {'text': train_texts},
                'test': {'text': train_texts[-1000:]}  # 마지막 1000개를 테스트용으로
            }
        except Exception as e:
            print(f"Error loading WikiText-2: {e}")
            return None

# === NLP용 토크나이저 ===
class NLPTokenizer:
    def __init__(self, vocab_size=10000, min_freq=2):  # min_freq 추가
        self.vocab_size = vocab_size
        self.min_freq = min_freq  # 추가: 최소 빈도
        self.vocab = {}
        self.inverse_vocab = {}
        self.word_counts = defaultdict(int)
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"
        self.cls_token = "<CLS>"  # 추가
        self.sep_token = "<SEP>"  # 추가
        
        # 특수 토큰 ID
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2  # 추가
        self.sep_token_id = 3  # 추가
        
    def build_vocab(self, texts):
        """어휘 사전 구축 - 개선된 버전"""
        print("Building vocabulary...")
        
        # 단어 빈도 계산
        for text in tqdm(texts, desc="Counting words"):
            if isinstance(text, str):
                words = tokenize_nltk(preprocess_text_for_nlp(text))
                for word in words:
                    if len(word) > 1:  # 단일 문자 제외
                        self.word_counts[word] += 1
        
        # 특수 토큰 먼저 추가
        self.vocab[self.pad_token] = self.pad_token_id
        self.vocab[self.unk_token] = self.unk_token_id
        self.vocab[self.cls_token] = self.cls_token_id
        self.vocab[self.sep_token] = self.sep_token_id
        
        # 빈도가 min_freq 이상인 단어만 포함
        filtered_words = [(word, count) for word, count in self.word_counts.items() 
                         if count >= self.min_freq]
        sorted_words = sorted(filtered_words, key=lambda x: x[1], reverse=True)
        
        # vocab_size-4개 단어 추가 (특수 토큰 4개 제외)
        for i, (word, count) in enumerate(sorted_words[:self.vocab_size-4]):
            self.vocab[word] = i + 4
        
        # 역 어휘 사전 구축
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"Vocabulary built: {len(self.vocab)} words")
        print(f"Total word types: {len(self.word_counts)}")
        print(f"Filtered out {len(self.word_counts) - len(self.vocab) + 4} rare words")


# === NLP용 Dataset 클래스 ===
class NLPDataset(TorchDataset):
    """NLP Dataset 클래스"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx] if self.labels else 0
        
        # 토큰화
        token_ids = self.tokenizer.encode(text, max_length=self.max_length)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor([1 if id != self.tokenizer.pad_token_id else 0 for id in token_ids], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

# === OSR을 위한 NLP Dataset ===
class OSRNLPDataset(TorchDataset):
    """OSR용 NLP Dataset 클래스"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 토큰화
        token_ids = self.tokenizer.encode(text, max_length=self.max_length)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor([1 if id != self.tokenizer.pad_token_id else 0 for id in token_ids], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

# === NLP 모델들 ===
class NLPClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, 
                 dropout=0.3, model_type="gru", attention=True):
        super(NLPClassifier, self).__init__()
        
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention = attention
        
        # 임베딩 레이어 - Xavier 초기화
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.embedding.weight.data[0].fill_(0)  # PAD 토큰은 0으로
        
        # RNN 레이어
        if model_type == "gru":
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0,
                             bidirectional=True)
        elif model_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0,
                              bidirectional=True)
        
        # Xavier 초기화
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # 어텐션 레이어
        if self.attention:
            self.attention_layer = nn.Linear(hidden_dim * 2, 1)
            nn.init.xavier_uniform_(self.attention_layer.weight)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 분류 레이어 - Xavier 초기화
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids, attention_mask=None, output_attentions=False):
        # 임베딩
        embedded = self.embedding(input_ids)
        
        # RNN
        if self.model_type == "gru":
            rnn_output, _ = self.rnn(embedded)
        elif self.model_type == "lstm":
            rnn_output, _ = self.rnn(embedded)
        
        if self.attention and attention_mask is not None:
            # 어텐션 메커니즘 - 수치 안정성 개선
            attention_weights = self.attention_layer(rnn_output).squeeze(-1)
            
            # 마스크 적용
            attention_weights = attention_weights.masked_fill(~attention_mask.bool(), float('-inf'))
            
            # Softmax with temperature for numerical stability
            attention_weights = F.softmax(attention_weights / 1.0, dim=1)
            
            # NaN 체크
            if torch.isnan(attention_weights).any():
                print("Warning: NaN in attention weights!")
                attention_weights = torch.ones_like(attention_weights) / attention_weights.size(1)
            
            # 가중합
            weighted_output = torch.bmm(attention_weights.unsqueeze(1), rnn_output).squeeze(1)
        else:
            # 마지막 유효한 출력 사용
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1)
                weighted_output = rnn_output[range(len(rnn_output)), lengths-1]
            else:
                weighted_output = rnn_output[:, -1]
        
        # NaN 체크
        if torch.isnan(weighted_output).any():
            print("Warning: NaN in weighted_output!")
            return torch.zeros(input_ids.size(0), self.classifier.out_features, device=input_ids.device)
        
        # 드롭아웃 적용
        weighted_output = self.dropout(weighted_output)
        
        # 분류
        logits = self.classifier(weighted_output)
        
        # 최종 NaN 체크
        if torch.isnan(logits).any():
            print("Warning: NaN in logits!")
            return torch.zeros_like(logits)
        
        if output_attentions and self.attention:
            return logits, attention_weights
        else:
            return logits
        
class NLPModelOOD(nn.Module):
    """OSR용 NLP 모델"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, 
                 dropout=0.3, model_type="gru"):
        super(NLPModelOOD, self).__init__()
        
        self.classifier = NLPClassifier(vocab_size, embed_dim, hidden_dim, num_classes,
                                      num_layers, dropout, model_type, attention=True)
        
    def forward(self, input_ids, attention_mask=None, output_features=False):
        if output_features:
            # 특징 추출을 위해 classifier의 마지막 레이어 이전까지 사용
            embedded = self.classifier.embedding(input_ids)
            
            if self.classifier.model_type == "gru":
                rnn_output, _ = self.classifier.rnn(embedded)
            elif self.classifier.model_type == "lstm":
                rnn_output, _ = self.classifier.rnn(embedded)
            
            if self.classifier.attention and attention_mask is not None:
                attention_weights = self.classifier.attention_layer(rnn_output).squeeze(-1)
                attention_weights = attention_weights.masked_fill(~attention_mask.bool(), float('-inf'))
                attention_weights = F.softmax(attention_weights, dim=1)
                features = torch.bmm(attention_weights.unsqueeze(1), rnn_output).squeeze(1)
            else:
                if attention_mask is not None:
                    lengths = attention_mask.sum(dim=1)
                    features = rnn_output[range(len(rnn_output)), lengths-1]
                else:
                    features = rnn_output[:, -1]
            
            features = self.classifier.dropout(features)
            logits = self.classifier.classifier(features)
            
            return logits, features
        else:
            return self.classifier(input_ids, attention_mask)

# === Enhanced DataModule for NLP ===
class EnhancedDataModule(pl.LightningDataModule):
    """Enhanced DataModule supporting both Syslog and NLP datasets"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['config'])
        
        # 모드에 따른 초기화
        if config.EXPERIMENT_MODE == "nlp":
            self._init_nlp_mode()
        else:
            self._init_syslog_mode()
        
        self.df_full = None
        self.train_df_final = None
        self.val_df_final = None
        self.label2id = None
        self.id2label = None
        self.num_labels = None
        self.tokenized_train_val_datasets = None
        self.class_weights = None
        
    def _init_nlp_mode(self):
        """NLP 모드 초기화"""
        print(f"Initializing DataModule for NLP mode: {self.config.CURRENT_NLP_DATASET}")
        self.tokenizer = NLPTokenizer(vocab_size=self.config.NLP_VOCAB_SIZE)
        self.data_collator = None
        
    def _init_syslog_mode(self):
        """Syslog 모드 초기화"""
        print(f"Initializing DataModule for Syslog mode: {self.config.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
    
    def prepare_data(self):
        """데이터 다운로드 등 준비"""
        if self.config.EXPERIMENT_MODE == "nlp":
            # NLP 데이터셋 로드 및 저장
            self._prepare_nlp_data()
    
    def _prepare_nlp_data(self):
        """NLP 데이터 준비"""
        dataset_name = self.config.CURRENT_NLP_DATASET
        
        if dataset_name == '20newsgroups':
            data = NLPDatasetLoader.load_20newsgroups()
        elif dataset_name == 'trec':
            data = NLPDatasetLoader.load_trec()
        elif dataset_name == 'sst2':
            data = NLPDatasetLoader.load_sst2()
        else:
            raise ValueError(f"Unknown NLP dataset: {dataset_name}")
        
        if data is None:
            raise ValueError(f"Failed to load dataset: {dataset_name}")
        
        # 데이터프레임으로 변환
        train_df = pd.DataFrame(data['train'])
        test_df = pd.DataFrame(data['test'])
        
        # 전체 데이터프레임 생성
        train_df['split'] = 'train'
        test_df['split'] = 'test'
        self.df_full = pd.concat([train_df, test_df], ignore_index=True)
        
        # 어휘 사전 구축
        all_texts = self.df_full['text'].tolist()
        self.tokenizer.build_vocab(all_texts)
        
        print(f"Loaded {dataset_name} with {len(train_df)} train and {len(test_df)} test samples")
    
    def setup(self, stage=None):
        if self.df_full is not None:
            return
        
        if self.config.EXPERIMENT_MODE == "nlp":
            self._setup_nlp()
        else:
            self._setup_syslog()
    
    def _setup_nlp(self):
        """NLP 모드 셋업"""
        self._prepare_nlp_data()
        
        # 라벨 매핑 생성
        unique_labels = sorted(self.df_full['label'].unique())
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.num_labels = len(unique_labels)
        
        print(f"NLP Label mapping: {self.label2id}")
        
        # 라벨 변환
        self.df_full['label_id'] = self.df_full['label'].map(self.label2id)
        
        # Train/Val 분할
        train_df = self.df_full[self.df_full['split'] == 'train'].copy()
        test_df = self.df_full[self.df_full['split'] == 'test'].copy()
        
        # 클래스 가중치 계산
        if self.config.USE_WEIGHTED_LOSS:
            self._compute_class_weights_nlp(train_df)
        
        # 검증 세트 생성 (train의 20%를 val로 사용)
        self.train_df_final, self.val_df_final = train_test_split(
            train_df, test_size=0.2, random_state=self.config.RANDOM_STATE,
            stratify=train_df['label_id']
        )
        
        print(f"NLP split - Train: {len(self.train_df_final)}, Val: {len(self.val_df_final)}")
        
        # 토큰화 (여기서는 DataLoader에서 처리)
        self.tokenized_train_val_datasets = {
            'train': self.train_df_final,
            'validation': self.val_df_final
        }
    
    def _setup_syslog(self):
        """Syslog 모드 셋업 (기존 코드 유지)"""
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
        df_known = df_known.dropna(subset=['label'])
        df_known['label'] = df_known['label'].astype(int)
        
        # 최소 샘플 수 필터링
        print(f"Filtering classes with minimum {self.config.MIN_SAMPLES_PER_CLASS_FOR_TRAIN_VAL} samples...")
        label_counts = df_known['label'].value_counts()
        valid_labels = label_counts[label_counts >= self.config.MIN_SAMPLES_PER_CLASS_FOR_TRAIN_VAL].index
        self.df_known_for_train_val = df_known[df_known['label'].isin(valid_labels)].copy()
        
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
        
        if self.config.USE_WEIGHTED_LOSS:
            self._compute_class_weights()
        self._split_train_val()
        self._tokenize_datasets()
    
    def _compute_class_weights_nlp(self, train_df):
        """NLP 모드용 클래스 가중치 계산"""
        labels_for_weights = train_df['label_id'].values
        unique_labels = np.unique(labels_for_weights)
        
        try:
            class_weights_array = compute_class_weight('balanced', classes=unique_labels, y=labels_for_weights)
            self.class_weights = torch.ones(self.num_labels)
            
            for i, label_idx in enumerate(unique_labels):
                self.class_weights[label_idx] = class_weights_array[i]
            
            print(f"Computed NLP class weights: {self.class_weights}")
        except ValueError as e:
            print(f"Error computing class weights: {e}. Using uniform weights.")
            self.config.USE_WEIGHTED_LOSS = False
            self.class_weights = None
    
    def _compute_class_weights(self):
        """Syslog 모드용 클래스 가중치 계산"""
        labels_for_weights = self.df_known_for_train_val['label'].values
        unique_labels = np.unique(labels_for_weights)
        
        try:
            class_weights_array = compute_class_weight('balanced', classes=unique_labels, y=labels_for_weights)
            self.class_weights = torch.ones(self.num_labels)
            
            for i, class_idx_in_unique_labels in enumerate(unique_labels):
                if class_idx_in_unique_labels < self.num_labels:
                     self.class_weights[class_idx_in_unique_labels] = class_weights_array[i]
            print(f"Computed class weights for base classifier: {self.class_weights}")
        except ValueError as e:
            print(f"Error computing class weights: {e}. Using uniform weights.")
            self.config.USE_WEIGHTED_LOSS = False
            self.class_weights = None
    
    def _split_train_val(self):
        """Train/Val 분할 (Syslog 모드)"""
        print("Splitting train/validation data for base classifier...")
        min_class_count = self.df_known_for_train_val['label'].value_counts().min()
        stratify_col = self.df_known_for_train_val['label'] if min_class_count > 1 else None
        if stratify_col is None:
            print("Warning: Not enough samples in some classes for stratified split. Using random split.")

        try:
            self.train_df_final, self.val_df_final = train_test_split(
                self.df_known_for_train_val, test_size=0.2,
                random_state=self.config.RANDOM_STATE, stratify=stratify_col
            )
        except ValueError:
            print("Warning: Stratified split failed unexpectedly. Using random split.")
            self.train_df_final, self.val_df_final = train_test_split(
                self.df_known_for_train_val, test_size=0.2, random_state=self.config.RANDOM_STATE
            )
        print(f"Base classifier split - Train: {len(self.train_df_final)}, Val (used as ID Test for OSR): {len(self.val_df_final)}")
    
    def _tokenize_datasets(self):
        """토큰화 (Syslog 모드)"""
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
        if self.config.EXPERIMENT_MODE == "nlp":
            # 클래스 밸런싱을 위한 샘플러 추가
            if self.config.USE_WEIGHTED_LOSS:
                from torch.utils.data import WeightedRandomSampler
                class_counts = self.train_df_final['label_id'].value_counts().sort_index()
                weights = 1.0 / class_counts.values
                sample_weights = [weights[label] for label in self.train_df_final['label_id']]
                sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
                
                dataset = NLPDataset(
                    self.train_df_final['text'].tolist(),
                    self.train_df_final['label_id'].tolist(),
                    self.tokenizer,
                    max_length=self.config.NLP_MAX_LENGTH
                )
                return DataLoader(dataset, batch_size=self.config.NLP_BATCH_SIZE, sampler=sampler,
                                num_workers=self.config.NUM_WORKERS, pin_memory=True)
        else:
            return DataLoader(
                self.tokenized_train_val_datasets['train'], batch_size=self.config.BATCH_SIZE,
                collate_fn=self.data_collator, num_workers=self.config.NUM_WORKERS,
                shuffle=True, pin_memory=True, persistent_workers=self.config.NUM_WORKERS > 0
            )
    
    def val_dataloader(self):
        if self.config.EXPERIMENT_MODE == "nlp":
            dataset = NLPDataset(
                self.val_df_final['text'].tolist(),
                self.val_df_final['label_id'].tolist(),
                self.tokenizer,
                max_length=self.config.NLP_MAX_LENGTH
            )
            return DataLoader(dataset, batch_size=self.config.NLP_BATCH_SIZE,
                            num_workers=self.config.NUM_WORKERS, pin_memory=True)
        else:
            return DataLoader(
                self.tokenized_train_val_datasets['validation'], batch_size=self.config.BATCH_SIZE,
                collate_fn=self.data_collator, num_workers=self.config.NUM_WORKERS,
                pin_memory=True, persistent_workers=self.config.NUM_WORKERS > 0
            )
    
    def get_full_dataframe(self):
        if self.df_full is None:
            self.setup()
        return self.df_full

# === Enhanced Model ===
class EnhancedModel(pl.LightningModule):
    """Enhanced Model supporting both Syslog and NLP tasks"""
    
    def __init__(self, config: Config, num_labels: int, label2id: dict, id2label: dict, 
                 class_weights=None, tokenizer=None):
        super().__init__()
        self.config_params = config
        self.save_hyperparameters(ignore=['config_params', 'class_weights', 'tokenizer'])
        
        self.tokenizer = tokenizer  # NLP용 토크나이저 저장
        
        if config.EXPERIMENT_MODE == "nlp":
            self._init_nlp_model(num_labels, label2id, id2label, class_weights)
        else:
            self._init_syslog_model(num_labels, label2id, id2label, class_weights)
        
        # 메트릭 정의
        metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=num_labels),
            'f1_weighted': torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='weighted'),
            'f1_macro': torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='macro')
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.val_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_labels)
    
    def _init_nlp_model(self, num_labels, label2id, id2label, class_weights):
        """NLP 모델 초기화"""
        print(f"Initializing NLP classifier: {self.config_params.NLP_MODEL_TYPE} for {num_labels} classes")
        
        self.model = NLPClassifier(
            vocab_size=self.config_params.NLP_VOCAB_SIZE,
            embed_dim=self.config_params.NLP_EMBED_DIM,
            hidden_dim=self.config_params.NLP_HIDDEN_DIM,
            num_classes=num_labels,
            num_layers=self.config_params.NLP_NUM_LAYERS,
            dropout=self.config_params.NLP_DROPOUT,
            model_type=self.config_params.NLP_MODEL_TYPE,
            attention=True
        )
        
        if self.config_params.USE_WEIGHTED_LOSS and class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            print("NLP classifier using weighted CrossEntropyLoss")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            print("NLP classifier using standard CrossEntropyLoss")
    
    def _init_syslog_model(self, num_labels, label2id, id2label, class_weights):
        """Syslog 모델 초기화"""
        print(f"Initializing base classifier model: {self.config_params.MODEL_NAME} for {num_labels} classes")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config_params.MODEL_NAME, num_labels=num_labels, label2id=label2id, id2label=id2label,
            ignore_mismatched_sizes=True, output_attentions=True, output_hidden_states=True
        )
        
        if self.config_params.USE_WEIGHTED_LOSS and class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            print("Base classifier using weighted CrossEntropyLoss")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            print("Base classifier using standard CrossEntropyLoss")
    
    def setup(self, stage=None):
        if self.config_params.USE_WEIGHTED_LOSS and hasattr(self.loss_fn, 'weight') and self.loss_fn.weight is not None:
            self.loss_fn.weight = self.loss_fn.weight.to(self.device)
            print(f"Moved class weights to {self.device}")
    
    def forward(self, batch, output_features=False, output_attentions=False):
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        if input_ids is None or attention_mask is None:
            raise ValueError("Batch missing 'input_ids' or 'attention_mask'")
        
        if self.config_params.EXPERIMENT_MODE == "nlp":
            if output_attentions and hasattr(self.model, 'attention'):
                logits, attentions = self.model(input_ids, attention_mask, output_attentions=True)
                return type('Output', (), {'logits': logits, 'attentions': [attentions]})()
            else:
                logits = self.model(input_ids, attention_mask)
                return type('Output', (), {'logits': logits})()
        else:
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=output_features, output_attentions=output_attentions
            )
    
    def _common_step(self, batch, batch_idx):
        if 'label' in batch:
            batch['labels'] = batch.pop('label')
        
        if self.config_params.EXPERIMENT_MODE == "nlp":
            # NLP 모델의 경우 수동으로 loss 계산
            outputs = self.forward(batch)
            logits = outputs.logits
            loss = self.loss_fn(logits, batch['labels'])
        else:
            # Syslog 모델의 경우 HF 모델의 내장 loss 사용
            outputs = self.model(**batch)
            loss = outputs.loss
            logits = outputs.logits
        
        preds = torch.argmax(logits, dim=1)
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
        self.log_dict(self.val_metrics, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        try:
            val_cm_computed = self.val_cm.compute()
            class_names = list(self.hparams.id2label.values())
            cm_df = pd.DataFrame(val_cm_computed.cpu().numpy(), index=class_names, columns=class_names)
            print(f"\nClassifier Validation Confusion Matrix (Epoch {self.current_epoch}):")
            print(cm_df)
            cm_filename = os.path.join(self.config_params.CONFUSION_MATRIX_DIR, 
                                     f"classifier_val_cm_epoch_{self.current_epoch}.csv")
            cm_df.to_csv(cm_filename)
        except Exception as e:
            print(f"Error in classifier validation CM: {e}")
        finally:
            self.val_cm.reset()
    
    def configure_optimizers(self):
        if self.config_params.EXPERIMENT_MODE == "nlp":
            optimizer = optim.Adam(self.parameters(), lr=self.config_params.NLP_LEARNING_RATE)
            # 학습률 스케줄러 추가
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3, verbose=True
            )
            return {
                "optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_f1_macro",
                }
            }

# === Enhanced Attention Analyzer ===
class EnhancedAttentionAnalyzer:
    """Enhanced Attention Analyzer supporting both NLP and Syslog models"""
    
    def __init__(self, config: Config, model_pl: EnhancedModel, tokenizer, device):
        self.config = config
        self.model_pl = model_pl.to(device)
        self.model_pl.eval()
        self.model_pl.freeze()
        self.tokenizer = tokenizer
        self.device = device
        
    @torch.no_grad()
    def get_word_attention_scores(self, texts: List[str], layer_idx: int = -1) -> List[Dict[str, float]]:
        batch_size = self.config.BATCH_SIZE if self.config.EXPERIMENT_MODE != "nlp" else self.config.NLP_BATCH_SIZE
        all_word_scores = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing word attention scores", leave=False):
            batch_texts = texts[i:i+batch_size]
            if self.config.EXPERIMENT_MODE == "nlp":
                batch_scores = self._process_attention_batch_nlp(batch_texts)
            else:
                batch_scores = self._process_attention_batch_syslog(batch_texts, layer_idx)
            all_word_scores.extend(batch_scores)
        return all_word_scores
    
    def _process_attention_batch_nlp(self, batch_texts: List[str]) -> List[Dict[str, float]]:
        """NLP 모델용 어텐션 배치 처리"""
        if not batch_texts:
            return []
        
        # 텍스트 전처리 및 토큰화
        processed_texts = [preprocess_text_for_nlp(text) for text in batch_texts]
        
        # 수동 토큰화
        batch_input_ids = []
        batch_attention_masks = []
        batch_token_mappings = []
        
        max_length = self.config.NLP_MAX_LENGTH
        
        for text in processed_texts:
            token_ids = self.tokenizer.encode(text, max_length=max_length)
            attention_mask = [1 if id != self.tokenizer.pad_token_id else 0 for id in token_ids]
            
            # 토큰과 원본 단어 매핑 생성
            words = tokenize_nltk(text)
            token_to_word_mapping = []
            
            # 토큰 ID를 단어로 변환하여 매핑
            for i, token_id in enumerate(token_ids):
                if token_id == self.tokenizer.pad_token_id:
                    token_to_word_mapping.append(None)
                else:
                    # 여기서는 간단한 매핑 사용
                    word_idx = min(i, len(words) - 1) if words else 0
                    if i < len(words):
                        token_to_word_mapping.append((i, words[i]))
                    else:
                        token_to_word_mapping.append(None)
            
            batch_input_ids.append(token_ids)
            batch_attention_masks.append(attention_mask)
            batch_token_mappings.append(token_to_word_mapping)
        
        # 텐서로 변환
        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(self.device)
        batch_attention_masks = torch.tensor(batch_attention_masks, dtype=torch.long).to(self.device)
        
        batch = {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_masks
        }
        
        # 모델 포워드 패스 (어텐션 포함)
        with torch.no_grad():
            outputs = self.model_pl.forward(batch, output_attentions=True)
            
            # NLP 모델의 경우 어텐션을 직접 추출
            if hasattr(outputs, 'attentions') and outputs.attentions:
                attention_weights = outputs.attentions[0]  # [batch_size, seq_len]
                attention_weights = attention_weights.cpu().numpy()
            else:
                # 어텐션이 없는 경우 균일한 어텐션 사용
                attention_weights = np.ones((batch_input_ids.shape[0], batch_input_ids.shape[1]))
                attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
        
        # 단어별 어텐션 점수 계산
        batch_word_scores = []
        for i in range(len(batch_texts)):
            word_scores = defaultdict(list)
            token_mapping = batch_token_mappings[i]
            attention = attention_weights[i]
            
            for j, mapping in enumerate(token_mapping):
                if mapping is not None and j < len(attention):
                    _, word = mapping
                    word_scores[word].append(attention[j])
            
            # 평균 계산
            final_word_scores = {word: np.mean(scores) for word, scores in word_scores.items()}
            batch_word_scores.append(final_word_scores)
        
        return batch_word_scores
    
    def _process_attention_batch_syslog(self, batch_texts: List[str], layer_idx: int) -> List[Dict[str, float]]:
        """Syslog 모델용 어텐션 배치 처리 (기존 코드)"""
        if not batch_texts:
            return []
        
        processed_texts = [preprocess_text_for_roberta(text) for text in batch_texts]
        inputs = self.tokenizer(
            processed_texts, return_tensors='pt', truncation=True,
            max_length=self.config.MAX_LENGTH, padding=True, return_offsets_mapping=True
        )
        offset_mappings = inputs.pop('offset_mapping').cpu().numpy()
        input_ids_batch = inputs['input_ids'].cpu().numpy()
        
        inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model_pl.model(**inputs_on_device, output_attentions=True)
            attentions_batch = outputs.attentions[layer_idx].cpu().numpy()

        batch_word_scores = [
            self._extract_word_scores_from_attention(
                attentions_batch[i], input_ids_batch[i], offset_mappings[i], processed_texts[i]
            ) for i in range(len(batch_texts))
        ]
        del inputs, inputs_on_device, outputs, attentions_batch
        gc.collect()
        return batch_word_scores
    
    def _extract_word_scores_from_attention(self, attention_sample, input_ids, offset_mapping, original_text):
        """Syslog 모델용 어텐션에서 단어 점수 추출 (기존 코드)"""
        attention_heads_mean = np.mean(attention_sample, axis=0)
        cls_attentions = attention_heads_mean[0, :]
        
        word_scores = defaultdict(list)
        current_word_indices = []
        last_word_end_offset = 0
        
        for j, (token_id, offset) in enumerate(zip(input_ids, offset_mapping)):
            if offset[0] == offset[1] or token_id in self.tokenizer.all_special_ids:
                continue
            
            is_continuation = (j > 0 and offset[0] == last_word_end_offset and token_id != self.tokenizer.unk_token_id)
            
            if not is_continuation and current_word_indices:
                start_offset = offset_mapping[current_word_indices[0]][0]
                end_offset = offset_mapping[current_word_indices[-1]][1]
                word = original_text[start_offset:end_offset]
                avg_score = np.mean(cls_attentions[current_word_indices])
                if word.strip():
                    word_scores[word.strip()].append(avg_score)
                current_word_indices = []
            
            current_word_indices.append(j)
            last_word_end_offset = offset[1]
        
        if current_word_indices:
            start_offset = offset_mapping[current_word_indices[0]][0]
            end_offset = offset_mapping[current_word_indices[-1]][1]
            word = original_text[start_offset:end_offset]
            avg_score = np.mean(cls_attentions[current_word_indices])
            if word.strip():
                word_scores[word.strip()].append(avg_score)
            
        return {word: np.mean(scores) for word, scores in word_scores.items()}
    
    def extract_top_attention_words(self, word_scores_dict: Dict[str, float]) -> List[str]:
        """상위 어텐션 단어 추출"""
        if not word_scores_dict:
            return []
        
        sorted_words = sorted(word_scores_dict.items(), key=lambda x: x[1], reverse=True)
        num_words = len(sorted_words)
        n_top = max(self.config.MIN_TOP_WORDS, math.ceil(num_words * self.config.ATTENTION_TOP_PERCENT))
        
        # 불용어 처리
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = {'a', 'an', 'the', 'is', 'was', 'to', 'of', 'for', 'on', 'in', 'at'}
        
        top_words_filtered = [word for word, score in sorted_words[:n_top] 
                             if word.lower() not in stop_words and len(word) > 1]
        
        return top_words_filtered if top_words_filtered else [word for word, score in sorted_words[:n_top]]
    
    def process_full_dataset(self, df: pd.DataFrame, exclude_class: str = None) -> pd.DataFrame:
        """전체 데이터셋에 대한 어텐션 분석"""
        print("Processing dataset for attention analysis...")
        
        if exclude_class:
            if self.config.EXPERIMENT_MODE == "nlp":
                # NLP 모드에서는 label 칼럼 사용
                exclude_mask = df['label'] != exclude_class
            else:
                # Syslog 모드에서는 class 칼럼 사용
                exclude_class_lower = exclude_class.lower()
                exclude_mask = df[self.config.CLASS_COLUMN].str.lower() != exclude_class_lower
            
            df_for_analysis = df[exclude_mask].copy()
            print(f"Excluding '{exclude_class}' class. Analyzing {len(df_for_analysis)}/{len(df)} samples.")
        else:
            df_for_analysis = df.copy()
            print(f"Analyzing all {len(df_for_analysis)} samples.")
        
        if df_for_analysis.empty:
            print("No data available for attention analysis after filtering.")
            result_df = df.copy()
            result_df['top_attention_words'] = pd.Series([[]] * len(df), index=df.index, dtype=object)
            result_df[self.config.TEXT_COLUMN_IN_OE_FILES] = df[self.config.TEXT_COLUMN]
            return result_df
        
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
        result_df['top_attention_words'] = pd.Series([[]] * len(df), index=df.index, dtype=object)
        result_df[self.config.TEXT_COLUMN_IN_OE_FILES] = df[self.config.TEXT_COLUMN]
        
        # 분석된 데이터만 업데이트
        if exclude_class:
            analyze_indices = df.index[exclude_mask]
            top_words_dict = dict(zip(analyze_indices, all_top_words))
            masked_texts_dict = dict(zip(analyze_indices, masked_texts))
            
            for idx in analyze_indices:
                result_df.at[idx, 'top_attention_words'] = top_words_dict[idx]
                result_df.at[idx, self.config.TEXT_COLUMN_IN_OE_FILES] = masked_texts_dict[idx]
        else:
            for i, idx in enumerate(df.index):
                result_df.at[idx, 'top_attention_words'] = all_top_words[i]
                result_df.at[idx, self.config.TEXT_COLUMN_IN_OE_FILES] = masked_texts[i]
        
        return result_df

# === Enhanced OE Extractor ===
class MaskedTextDatasetForMetrics(TorchDataset):
    """마스킹된 텍스트 메트릭 추출용 데이터셋"""
    def __init__(self, texts: List[str], tokenizer, max_length: int, is_nlp_mode: bool = False):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_nlp_mode = is_nlp_mode
        
        if is_nlp_mode:
            # NLP 모드: 수동 토큰화
            self.encoded_texts = []
            for text in texts:
                valid_text = str(text) if pd.notna(text) else ""
                token_ids = tokenizer.encode(valid_text, max_length=max_length)
                attention_mask = [1 if id != tokenizer.pad_token_id else 0 for id in token_ids]
                self.encoded_texts.append({
                    'input_ids': torch.tensor(token_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
                })
        else:
            # Syslog 모드: HuggingFace 토크나이저
            valid_texts = [str(t) if pd.notna(t) else "" for t in texts]
            self.encodings = tokenizer(
                valid_texts, max_length=max_length, padding='max_length',
                truncation=True, return_tensors='pt'
            )
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        if self.is_nlp_mode:
            return self.encoded_texts[idx]
        else:
            return {key: val[idx].clone().detach() for key, val in self.encodings.items()}

class OEExtractorEnhanced:
    """Enhanced OE Extractor supporting both NLP and Syslog models"""
    
    def __init__(self, config: Config, model_pl: EnhancedModel, tokenizer, device):
        self.config = config
        self.model_pl = model_pl.to(device)
        self.model_pl.eval()
        self.model_pl.freeze()
        self.tokenizer = tokenizer
        self.device = device
        self.is_nlp_mode = (config.EXPERIMENT_MODE == "nlp")
    
    @torch.no_grad()
    def extract_attention_metrics(self, dataloader: DataLoader, original_df: pd.DataFrame = None, 
                                    exclude_class: str = None) -> Tuple[pd.DataFrame, List[np.ndarray]]:
        """수정된 버전: 모든 클래스에 대해 실제 feature 추출"""
        attention_metrics_list = []
        features_list = []
        print("Extracting attention metrics and features from masked texts...")
        
        for batch_encodings in tqdm(dataloader, desc="Processing masked text batches", leave=False):
            batch_on_device = {k: v.to(self.device) for k, v in batch_encodings.items()}
            
            if self.is_nlp_mode:
                # NLP 모델의 경우
                outputs = self.model_pl.forward(batch_on_device, output_features=False, output_attentions=True)
                
                # 어텐션 추출
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    attention_weights = outputs.attentions[0].cpu().numpy()  # [batch_size, seq_len]
                else:
                    # 어텐션이 없는 경우 균일한 어텐션 사용
                    batch_size, seq_len = batch_on_device['input_ids'].shape
                    attention_weights = np.ones((batch_size, seq_len))
                    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
                
                # Features 추출 (모델의 마지막 hidden state 사용)
                # NLP 모델의 경우 임베딩 + RNN을 통과한 출력을 feature로 사용
                with torch.no_grad():
                    embedded = self.model_pl.model.embedding(batch_on_device['input_ids'])
                    if self.model_pl.model.model_type == "gru":
                        rnn_output, _ = self.model_pl.model.rnn(embedded)
                    elif self.model_pl.model.model_type == "lstm":
                        rnn_output, _ = self.model_pl.model.rnn(embedded)
                    
                    # 어텐션 적용된 feature 추출
                    if self.model_pl.model.attention:
                        attention_mask = batch_on_device.get('attention_mask')
                        if attention_mask is not None:
                            att_weights = self.model_pl.model.attention_layer(rnn_output).squeeze(-1)
                            att_weights = att_weights.masked_fill(~attention_mask.bool(), float('-inf'))
                            att_weights = F.softmax(att_weights, dim=1)
                            features_batch = torch.bmm(att_weights.unsqueeze(1), rnn_output).squeeze(1)
                        else:
                            features_batch = rnn_output[:, -1]
                    else:
                        features_batch = rnn_output[:, -1]
                
                features_batch = features_batch.cpu().numpy()
                features_list.extend(list(features_batch))
                
                # 어텐션 메트릭 계산
                input_ids_batch = batch_on_device['input_ids'].cpu().numpy()
                for i in range(len(input_ids_batch)):
                    metrics = self._compute_attention_metrics_nlp(attention_weights[i], input_ids_batch[i])
                    attention_metrics_list.append(metrics)
                    
            else:
                # Syslog 모델의 경우 (기존 코드 유지)
                outputs = self.model_pl.forward(batch_on_device, output_features=True, output_attentions=True)
                attentions_batch = outputs.attentions[-1].cpu().numpy()
                features_batch = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                features_list.extend(list(features_batch))
                
                input_ids_batch = batch_encodings['input_ids'].cpu().numpy()
                for i in range(len(input_ids_batch)):
                    metrics = self._compute_attention_metrics_syslog(attentions_batch[i], input_ids_batch[i])
                    attention_metrics_list.append(metrics)
                
                del outputs, attentions_batch, features_batch
            
            gc.collect()
        
        # exclude_class 처리
        if original_df is not None and exclude_class and len(original_df) == len(attention_metrics_list):
            if self.config.EXPERIMENT_MODE == "nlp":
                exclude_mask = original_df['label'] == exclude_class
            else:
                exclude_class_lower = exclude_class.lower()
                exclude_mask = original_df[self.config.CLASS_COLUMN].str.lower() == exclude_class_lower
            
            # metrics만 수정 (feature는 그대로 유지)
            for idx, is_excluded in enumerate(exclude_mask):
                if is_excluded:
                    default_metrics = {'max_attention': 0, 'top_k_avg_attention': 0, 'attention_entropy': 0}
                    attention_metrics_list[idx] = default_metrics
        
        return pd.DataFrame(attention_metrics_list), features_list
    
    def _compute_attention_metrics_nlp(self, attention_weights, input_ids):
        """NLP 모델용 어텐션 메트릭 계산"""
        # 유효한 토큰 인덱스 찾기
        valid_indices = np.where(
            (input_ids != self.tokenizer.pad_token_id) &
            (input_ids != self.tokenizer.unk_token_id)
        )[0]
        
        if len(valid_indices) == 0:
            return {'max_attention': 0, 'top_k_avg_attention': 0, 'attention_entropy': 0}
        
        valid_attention = attention_weights[valid_indices]
        
        max_att = np.max(valid_attention) if len(valid_attention) > 0 else 0
        k = min(self.config.TOP_K_ATTENTION, len(valid_attention))
        top_k_avg_att = np.mean(np.sort(valid_attention)[-k:]) if k > 0 else 0
        
        # 어텐션 확률로 변환하여 엔트로피 계산
        if len(valid_attention) > 1:
            att_probs = valid_attention / valid_attention.sum()
            att_entropy = entropy(att_probs)
        else:
            att_entropy = 0
        
        return {'max_attention': max_att, 'top_k_avg_attention': top_k_avg_att, 'attention_entropy': att_entropy}
    
    def _compute_attention_metrics_syslog(self, attention_sample, input_ids):
        """Syslog 모델용 어텐션 메트릭 계산 (기존 코드)"""
        valid_indices = np.where(
            (input_ids != self.tokenizer.pad_token_id) &
            (input_ids != self.tokenizer.cls_token_id) &
            (input_ids != self.tokenizer.sep_token_id)
        )[0]
        
        if len(valid_indices) == 0:
            return {'max_attention': 0, 'top_k_avg_attention': 0, 'attention_entropy': 0}
        
        cls_attentions = np.mean(attention_sample[:, 0, :], axis=0)[valid_indices]
        
        max_att = np.max(cls_attentions) if len(cls_attentions) > 0 else 0
        k = min(self.config.TOP_K_ATTENTION, len(cls_attentions))
        top_k_avg_att = np.mean(np.sort(cls_attentions)[-k:]) if k > 0 else 0
        
        att_probs = F.softmax(torch.tensor(cls_attentions), dim=0).numpy()
        att_entropy = entropy(att_probs) if len(att_probs) > 1 else 0
        
        return {'max_attention': max_att, 'top_k_avg_attention': top_k_avg_att, 'attention_entropy': att_entropy}
    
    def compute_removed_word_attention(self, df: pd.DataFrame, attention_analyzer: EnhancedAttentionAnalyzer, exclude_class: str = None) -> pd.DataFrame:
        """제거된 단어 어텐션 점수 계산"""
        print("Computing removed word attention scores...")
        if 'top_attention_words' not in df.columns or self.config.TEXT_COLUMN not in df.columns:
            print("  Required columns not found. Skipping removed_avg_attention.")
            df['removed_avg_attention'] = 0.0
            return df
        
        # exclude_class 처리
        if exclude_class:
            if self.config.EXPERIMENT_MODE == "nlp":
                process_mask = df['label'] != exclude_class
            else:
                exclude_class_lower = exclude_class.lower()
                process_mask = df[self.config.CLASS_COLUMN].str.lower() != exclude_class_lower
            
            df_for_processing = df[process_mask].copy()
            
            if df_for_processing.empty:
                print("  No data to process after excluding class. Setting all removed_avg_attention to 0.")
                df['removed_avg_attention'] = 0.0
                return df
            
            texts = df_for_processing[self.config.TEXT_COLUMN].tolist()
            word_attentions_list = attention_analyzer.get_word_attention_scores(texts)
            
            removed_attentions = np.zeros(len(df))
            
            process_idx = 0
            for idx, row in df.iterrows():
                if process_mask.iloc[idx]:
                    top_words_val = row['top_attention_words']
                    top_words = safe_literal_eval(top_words_val)
                    
                    if top_words and process_idx < len(word_attentions_list):
                        word_scores_dict = word_attentions_list[process_idx]
                        removed_scores = [word_scores_dict.get(word, 0) for word in top_words]
                        removed_attentions[idx] = np.mean(removed_scores) if removed_scores else 0
                    process_idx += 1
        else:
            texts = df[self.config.TEXT_COLUMN].tolist()
            word_attentions_list = attention_analyzer.get_word_attention_scores(texts)
            
            removed_attentions = []
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing removed attention", leave=False):
                top_words_val = row['top_attention_words']
                top_words = safe_literal_eval(top_words_val)
                
                if top_words and idx < len(word_attentions_list):
                    word_scores_dict = word_attentions_list[idx]
                    removed_scores = [word_scores_dict.get(word, 0) for word in top_words]
                    removed_attentions.append(np.mean(removed_scores) if removed_scores else 0)
                else:
                    removed_attentions.append(0)
        
        df['removed_avg_attention'] = removed_attentions
        print("Removed word attention computation complete.")
        return df
    
    def extract_oe_datasets(self, df: pd.DataFrame, exclude_class: str = None) -> None:
        """OE 데이터셋 추출"""
        print("Extracting OE datasets with different criteria...")
        
        # exclude_class 처리
        if exclude_class:
            if self.config.EXPERIMENT_MODE == "nlp":
                df_for_oe = df[df['label'] != exclude_class].copy()
            else:
                exclude_class_lower = exclude_class.lower()
                df_for_oe = df[df[self.config.CLASS_COLUMN].str.lower() != exclude_class_lower].copy()
            print(f"OE extraction excluding '{exclude_class}' class: {len(df_for_oe)}/{len(df)} samples")
        else:
            df_for_oe = df.copy()
        
        if df_for_oe.empty:
            print("No data available for OE extraction after filtering.")
            return
        
        # 각 메트릭에 대해 OE 추출
        for metric, settings in self.config.METRIC_SETTINGS.items():
            if metric not in df_for_oe.columns:
                print(f"Skipping OE extraction for {metric} - column not found in DataFrame.")
                continue
            self._extract_single_metric_oe(df_for_oe, metric, settings)
        
        # 순차 필터링
        self._extract_sequential_filtering_oe(df_for_oe)
    
    def _extract_single_metric_oe(self, df: pd.DataFrame, metric: str, settings: dict):
        """단일 메트릭으로 OE 추출"""
        scores = np.nan_to_num(df[metric].values, nan=0.0)
        if settings['mode'] == 'higher':
            threshold = np.percentile(scores, 100 - settings['percentile'])
            selected_indices = np.where(scores >= threshold)[0]
        else:
            threshold = np.percentile(scores, settings['percentile'])
            selected_indices = np.where(scores <= threshold)[0]
        
        if len(selected_indices) > 0:
            oe_df_simple = df.iloc[selected_indices][[self.config.TEXT_COLUMN_IN_OE_FILES]].copy()
            
            extended_cols = [self.config.TEXT_COLUMN_IN_OE_FILES, self.config.TEXT_COLUMN, 'top_attention_words', metric]
            extended_cols = [col for col in extended_cols if col in df.columns]
            oe_df_extended = df.iloc[selected_indices][extended_cols].copy()

            mode_desc = f"{settings['mode']}{settings['percentile']}pct"
            oe_filename = os.path.join(self.config.OE_DATA_DIR, f"oe_data_{metric}_{mode_desc}.csv")
            oe_extended_filename = os.path.join(self.config.OE_DATA_DIR, f"oe_data_{metric}_{mode_desc}_extended.csv")
            
            oe_df_simple.to_csv(oe_filename, index=False)
            oe_df_extended.to_csv(oe_extended_filename, index=False)
            print(f"Saved OE dataset ({len(oe_df_simple)} samples) for {metric} {mode_desc}: {oe_filename}")
    
    def _extract_sequential_filtering_oe(self, df: pd.DataFrame):
        """순차 필터링으로 OE 추출"""
        print("Applying sequential filtering for OE extraction...")
        selected_mask = np.ones(len(df), dtype=bool)
        
        for step, (metric, settings) in enumerate(self.config.FILTERING_SEQUENCE):
            if metric not in df.columns:
                print(f"Skipping sequential filter step {step+1}: {metric} not found.")
                continue
            
            current_selection_df = df[selected_mask]
            if current_selection_df.empty:
                print(f"No samples left before applying filter: {metric}. Stopping sequential filtering.")
                selected_mask[:] = False
                break

            scores = np.nan_to_num(current_selection_df[metric].values, nan=0.0)
            
            if settings['mode'] == 'higher':
                threshold = np.percentile(scores, 100 - settings['percentile'])
                step_mask = scores >= threshold
            else:
                threshold = np.percentile(scores, settings['percentile'])
                step_mask = scores <= threshold
            
            current_indices = np.where(selected_mask)[0]
            indices_to_keep_from_current_step = current_indices[step_mask]
            
            selected_mask = np.zeros_like(selected_mask)
            if len(indices_to_keep_from_current_step) > 0:
                selected_mask[indices_to_keep_from_current_step] = True
            
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

# === Enhanced Visualizer ===
class EnhancedVisualizer:
    """Enhanced Visualizer supporting both NLP and Syslog experiments"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def plot_metric_distribution(self, scores: np.ndarray, metric_name: str, title: str, save_path: str):
        """메트릭 분포 플롯"""
        if len(scores) == 0:
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
        mean_val = np.mean(scores)
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Distribution plot saved: {save_path}")

    def plot_tsne(self, features: np.ndarray, labels: np.ndarray, title: str, save_path: str,
                  highlight_indices: Optional[np.ndarray] = None, highlight_label: str = 'OE Candidate',
                  class_names: Optional[Dict] = None, seed: int = 42):
        """t-SNE 플롯"""
        if len(features) == 0:
            return
        print(f"Running t-SNE for visualization on {features.shape[0]} samples...")
        try:
            perplexity = min(30, features.shape[0] - 1)
            if perplexity <= 1:
                print(f"Warning: t-SNE perplexity too low ({perplexity}). Skipping plot.")
                return
            tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, max_iter=1000, init='pca', learning_rate='auto')
            tsne_results = tsne.fit_transform(features)
        except Exception as e:
            print(f"Error in t-SNE: {e}. Skipping.")
            return
        
        df_tsne = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
        df_tsne['label'] = labels
        df_tsne['is_highlighted'] = False
        if highlight_indices is not None:
            df_tsne.loc[highlight_indices, 'is_highlighted'] = True
        
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
        
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel("t-SNE Dim 1")
        plt.ylabel("t-SNE Dim 2")
        plt.grid(alpha=0.3, linestyle='--')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"t-SNE plot saved: {save_path}")

    def visualize_all_metrics(self, df: pd.DataFrame):
        """모든 메트릭 시각화"""
        metric_columns = ['max_attention', 'top_k_avg_attention', 'attention_entropy', 'removed_avg_attention']
        for metric in metric_columns:
            if metric in df.columns and not df[metric].isnull().all():
                self.plot_metric_distribution(
                    df[metric].dropna().values, metric, f'Distribution of {metric}',
                    os.path.join(self.config.VIS_DIR, f'{metric}_distribution.png')
                )

    def visualize_oe_candidates(self, df: pd.DataFrame, features: List[np.ndarray], label2id: dict, id2label: dict):
        """OE 후보 시각화"""
        if not features or len(features) != len(df):
            print(f"Feature length mismatch or no features. Features: {len(features)}, DF: {len(df)}")
            return

        # 라벨 준비
        tsne_labels = []
        if self.config.EXPERIMENT_MODE == "nlp":
            # NLP 모드에서는 label 칼럼 사용
            for label_val in df['label']:
                tsne_labels.append(label2id.get(label_val, -2))
        else:
            # Syslog 모드에서는 기존 로직 사용
            unknown_class_lower = self.config.EXCLUDE_CLASS_FOR_TRAINING.lower()
            for cls_val in df[self.config.CLASS_COLUMN]:
                cls_str = str(cls_val).lower()
                if cls_str == unknown_class_lower:
                    tsne_labels.append(-1)  # Unknown (OOD)
                else:
                    tsne_labels.append(label2id.get(cls_str, -2))
        
        tsne_labels_np = np.array(tsne_labels)
        
        # 클래스 이름 정의
        class_names_viz = {**{k: str(v) for k, v in id2label.items()}, -1: 'Unknown (Excluded)', -2: 'Other/Filtered'}

        # 각 메트릭별 시각화
        for metric, settings in self.config.METRIC_SETTINGS.items():
            if metric not in df.columns:
                continue
            scores = np.nan_to_num(df[metric].values, nan=0.0)
            threshold = np.percentile(scores, 100 - settings['percentile'] if settings['mode'] == 'higher' else settings['percentile'])
            oe_indices = np.where(scores >= threshold if settings['mode'] == 'higher' else scores <= threshold)[0]
            mode_desc = f"{settings['mode']}{settings['percentile']}%"
            
            self.plot_tsne(
                np.array(features), tsne_labels_np,
                f't-SNE: OE Candidates by {metric} ({mode_desc})',
                os.path.join(self.config.VIS_DIR, f'tsne_oe_cand_{metric}_{mode_desc}.png'),
                highlight_indices=oe_indices, highlight_label=f'OE Candidate ({metric} {mode_desc})',
                class_names=class_names_viz, seed=self.config.RANDOM_STATE
            )
        
        # 순차 필터링 시각화
        if hasattr(self.config, 'FILTERING_SEQUENCE') and self.config.FILTERING_SEQUENCE:
            selected_mask = np.ones(len(df), dtype=bool)
            filter_steps_desc_list = []
            
            for step, (metric, settings) in enumerate(self.config.FILTERING_SEQUENCE):
                if metric not in df.columns:
                    continue
                current_selection_df = df[selected_mask]
                if current_selection_df.empty:
                    break
                scores = np.nan_to_num(current_selection_df[metric].values, nan=0.0)
                threshold = np.percentile(scores, 100 - settings['percentile'] if settings['mode'] == 'higher' else settings['percentile'])
                step_mask_on_subset = scores >= threshold if settings['mode'] == 'higher' else scores <= threshold
                
                current_indices = np.where(selected_mask)[0]
                indices_to_keep_from_step = current_indices[step_mask_on_subset]
                selected_mask = np.zeros_like(selected_mask)
                if len(indices_to_keep_from_step) > 0:
                    selected_mask[indices_to_keep_from_step] = True
                
                mode_desc = f"{settings['mode']}{settings['percentile']}%"
                filter_steps_desc_list.append(f"{metric}({mode_desc})")

            final_indices_seq = np.where(selected_mask)[0]
            if len(final_indices_seq) > 0:
                seq_desc = " -> ".join(filter_steps_desc_list)
                self.plot_tsne(
                    np.array(features), tsne_labels_np,
                    f't-SNE: Sequential Filter Candidates\n{seq_desc} -> {len(final_indices_seq)} samples',
                    os.path.join(self.config.VIS_DIR, f'tsne_oe_cand_sequential_{"_".join(filter_steps_desc_list)}.png'),
                    highlight_indices=final_indices_seq, highlight_label=f'Sequential OE Candidate ({len(final_indices_seq)} samples)',
                    class_names=class_names_viz, seed=self.config.RANDOM_STATE
                )

# === Enhanced OSR Components ===
def prepare_nlp_id_data_for_osr(datamodule: EnhancedDataModule, tokenizer, max_length: int) -> Tuple[Optional[OSRNLPDataset], Optional[OSRNLPDataset], int, Optional[LabelEncoder], Dict, Dict]:
    """NLP용 ID 데이터 OSR 준비"""
    print(f"\n--- Preparing NLP ID data for OSR ---")
    if datamodule.train_df_final is None or datamodule.val_df_final is None:
        print("Error: DataModule not set up.")
        return None, None, 0, None, {}, {}

    train_df = datamodule.train_df_final
    test_df = datamodule.val_df_final

    num_classes = datamodule.num_labels
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array([datamodule.id2label[i] for i in range(num_classes)])
    
    id_label2id = datamodule.label2id
    id_id2label = datamodule.id2label

    print(f"  - Using {num_classes} known classes from DataModule.")
    print(f"  - Label to ID mapping: {id_label2id}")

    # 라벨 변환
    train_df['label_id'] = train_df['label'].map(id_label2id)
    test_df['label_id'] = test_df['label'].map(id_label2id)

    train_dataset = OSRNLPDataset(train_df['text'].tolist(), train_df['label_id'].tolist(), tokenizer, max_length)
    id_test_dataset = OSRNLPDataset(test_df['text'].tolist(), test_df['label_id'].tolist(), tokenizer, max_length)
    
    print(f"  - OSR Train: {len(train_dataset)}, OSR ID Test: {len(id_test_dataset)}")
    return train_dataset, id_test_dataset, num_classes, label_encoder, id_label2id, id_id2label

def prepare_wikitext_ood_data_for_osr(tokenizer, max_length: int) -> Optional[OSRNLPDataset]:
    """WikiText-2 OOD 데이터 OSR 준비"""
    print(f"\n--- Preparing WikiText-2 OOD data for OSR ---")
    
    try:
        # WikiText-2 로드
        wikitext_data = NLPDatasetLoader.load_wikitext2()
        if wikitext_data is None:
            print("Error: Failed to load WikiText-2 data.")
            return None
        
        # 테스트 데이터 사용 (일부만)
        texts = wikitext_data['test']['text'][:1000]  # 1000개만 사용
        ood_labels = np.full(len(texts), -1, dtype=int).tolist()  # OOD label is -1
        
        ood_dataset = OSRNLPDataset(texts, ood_labels, tokenizer, max_length)
        print(f"  - Loaded {len(ood_dataset)} samples for OOD testing (WikiText-2).")
        return ood_dataset
    except Exception as e:
        print(f"Error preparing WikiText-2 OOD data: {e}")
        return None

def prepare_nlp_oe_data_for_osr(tokenizer, max_length: int, oe_data_path: str, oe_text_col: str) -> Optional[OSRNLPDataset]:
    """NLP OE 데이터 OSR 준비"""
    print(f"\n--- Preparing NLP OE data from: {oe_data_path} for OSR ---")
    if not os.path.exists(oe_data_path):
        print(f"Error: OE data path not found: {oe_data_path}")
        return None
    
    try:
        df = pd.read_csv(oe_data_path)
        if oe_text_col not in df.columns:
            # 폴백 컬럼 시도
            fallback_cols = ['masked_text_attention', 'text', Config.TEXT_COLUMN]
            found_col = False
            for col_attempt in fallback_cols:
                if col_attempt in df.columns:
                    oe_text_col_actual = col_attempt
                    print(f"  Warning: Specified OE text column '{oe_text_col}' not found. Using fallback '{oe_text_col_actual}'.")
                    found_col = True
                    break
            if not found_col:
                raise ValueError(f"OE Data CSV '{oe_data_path}' must contain a valid text column.")
        else:
            oe_text_col_actual = oe_text_col

        df = df.dropna(subset=[oe_text_col_actual])
        texts = df[oe_text_col_actual].astype(str).tolist()
        if not texts:
            print(f"Warning: No valid OE texts found.")
            return None
        
        oe_labels = np.full(len(texts), -1, dtype=int).tolist()  # OE label is -1
        oe_dataset = OSRNLPDataset(texts, oe_labels, tokenizer, max_length)
        print(f"  - Loaded {len(oe_dataset)} samples for OE training.")
        return oe_dataset
    except Exception as e:
        print(f"Error preparing NLP OE data: {e}")
        return None

# === OSR 평가 함수 (NLP 지원 추가) ===
def evaluate_nlp_osr(model: nn.Module, id_loader: DataLoader, ood_loader: Optional[DataLoader], device: torch.device, temperature: float = 1.0, threshold_percentile: float = 5.0, return_data: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, np.ndarray]]]:
    """NLP OSR 평가"""
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
        "FPR@TPR90": 1.0, "AUPR_In": 0.0, "AUPR_Out": 0.0, "DetectionAccuracy": 0.0, "OSCR": 0.0,
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
        print("Warning: No OOD samples for OSR evaluation.")
        return results, all_data_dict if return_data else results

    y_true_osr = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_scores_osr = np.concatenate([id_scores, ood_scores])
    
    valid_indices = ~np.isnan(y_scores_osr)
    y_true_osr, y_scores_osr = y_true_osr[valid_indices], y_scores_osr[valid_indices]

    if len(np.unique(y_true_osr)) < 2:
        print("Warning: Only one class type present after filtering.")
    else:
        results["AUROC"] = roc_auc_score(y_true_osr, y_scores_osr)
        fpr, tpr, thresholds_roc = roc_curve(y_true_osr, y_scores_osr)
        idx_tpr90 = np.where(tpr >= 0.90)[0]
        results["FPR@TPR90"] = fpr[idx_tpr90[0]] if len(idx_tpr90) > 0 else 1.0

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

# === 기존 OSR 함수들 (Syslog용) ===
from oe2 import (
    OSRTextDataset, RoBERTaOOD, prepare_id_data_for_osr, prepare_syslog_ood_data_for_osr,
    prepare_generated_oe_data_for_osr, evaluate_osr, plot_confidence_histograms_osr,
    plot_roc_curve_osr, plot_confusion_matrix_osr, plot_tsne_osr
)

# === Main Pipeline Class ===
class EnhancedOEPipeline:
    """Enhanced OE Extraction Pipeline supporting both Syslog and NLP experiments"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_module: Optional[EnhancedDataModule] = None
        self.model: Optional[EnhancedModel] = None
        self.attention_analyzer: Optional[EnhancedAttentionAnalyzer] = None
        self.oe_extractor: Optional[OEExtractorEnhanced] = None
        self.visualizer = EnhancedVisualizer(config)
        
        config.create_directories()
        config.save_config()
        set_seed(config.RANDOM_STATE)
    
    def run_stage1_model_training(self):
        """1단계: 기본 모델 훈련"""
        if not self.config.STAGE_MODEL_TRAINING:
            print("Skipping Stage 1: Model Training")
            if self._check_existing_model():
                self._load_existing_model()
            else:
                print("Error: Model training skipped, but no existing model found.")
                sys.exit(1)
            return

        print(f"\n{'='*50}\nSTAGE 1: MODEL TRAINING ({self.config.EXPERIMENT_MODE.upper()})\n{'='*50}")
        
        # 데이터 모듈 초기화
        self.data_module = EnhancedDataModule(self.config)
        self.data_module.setup()
        
        # 모델 초기화
        self.model = EnhancedModel(
            config=self.config,
            num_labels=self.data_module.num_labels,
            label2id=self.data_module.label2id,
            id2label=self.data_module.id2label,
            class_weights=self.data_module.class_weights,
            tokenizer=self.data_module.tokenizer if self.config.EXPERIMENT_MODE == "nlp" else None
        )
        
        # 콜백 설정
        monitor_metric = 'val_f1_macro'
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.MODEL_SAVE_DIR,
            filename=f'{self.config.EXPERIMENT_MODE}-clf-{{epoch:02d}}-{{{monitor_metric}:.4f}}',
            save_top_k=1, monitor=monitor_metric, mode='max'
        )
        early_stopping_callback = EarlyStopping(monitor=monitor_metric, patience=5, mode='max', verbose=True)
        csv_logger = CSVLogger(save_dir=self.config.LOG_DIR, name=f"{self.config.EXPERIMENT_MODE}_model_training")
        
        # 훈련
        if self.config.EXPERIMENT_MODE == "nlp":
            num_epochs = self.config.NLP_NUM_EPOCHS
        else:
            num_epochs = self.config.NUM_TRAIN_EPOCHS
        
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator=self.config.ACCELERATOR,
            devices=self.config.DEVICES,
            precision=self.config.PRECISION,
            logger=csv_logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
            deterministic=False,
            log_every_n_steps=self.config.LOG_EVERY_N_STEPS,
            gradient_clip_val=self.config.GRADIENT_CLIP_VAL
        )
        
        print(f"Starting {self.config.EXPERIMENT_MODE} model training...")
        trainer.fit(self.model, datamodule=self.data_module)
        print(f"{self.config.EXPERIMENT_MODE} model training complete!")
        self._load_best_model(checkpoint_callback)
    
    def run_stage2_attention_extraction(self) -> Optional[pd.DataFrame]:
        """2단계: 어텐션 추출"""
        if not self.config.STAGE_ATTENTION_EXTRACTION:
            print("Skipping Stage 2: Attention Extraction")
            if self.config.STAGE_OE_EXTRACTION or self.config.STAGE_VISUALIZATION:
                try:
                    return self._load_attention_results()
                except FileNotFoundError:
                    print("Attention results not found, cannot proceed with dependent stages.")
                    return None
            return None

        print(f"\n{'='*50}\nSTAGE 2: ATTENTION EXTRACTION\n{'='*50}")
        
        if self.model is None:
            self._load_existing_model()
        if self.data_module is None:
            self.data_module = EnhancedDataModule(self.config)
            self.data_module.setup()

        current_device = self.model.device if hasattr(self.model, 'device') else DEVICE_OSR

        self.attention_analyzer = EnhancedAttentionAnalyzer(
            config=self.config,
            model_pl=self.model,
            tokenizer=self.data_module.tokenizer,
            device=current_device
        )
        
        full_df = self.data_module.get_full_dataframe()
        
        # Exclude class 설정
        if self.config.EXPERIMENT_MODE == "nlp":
            # NLP 모드에서는 특정 클래스를 제외하지 않고 모든 데이터 처리
            processed_df = self.attention_analyzer.process_full_dataset(full_df, exclude_class=None)
        else:
            # Syslog 모드에서는 기존 로직 사용
            processed_df = self.attention_analyzer.process_full_dataset(
                full_df, exclude_class=self.config.EXCLUDE_CLASS_FOR_TRAINING
            )
        
        output_path = os.path.join(self.config.ATTENTION_DATA_DIR, f"df_with_attention_{self.config.EXPERIMENT_MODE}.csv")
        processed_df.to_csv(output_path, index=False)
        print(f"Attention analysis results saved: {output_path}")
        self._print_attention_samples(processed_df)
        return processed_df
    
    def run_stage3_oe_extraction(self, df_with_attention: Optional[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], Optional[List[np.ndarray]]]:
        """3단계: OE 추출"""
        if not self.config.STAGE_OE_EXTRACTION:
            print("Skipping Stage 3: OE Extraction")
            if self.config.STAGE_VISUALIZATION or self.config.STAGE_OSR_EXPERIMENTS:
                try:
                    return self._load_final_metrics_and_features()
                except FileNotFoundError:
                    print("Final metrics/features not found.")
                    return None, None
            return None, None

        print(f"\n{'='*50}\nSTAGE 3: OE EXTRACTION\n{'='*50}")
        
        if df_with_attention is None:
            df_with_attention = self._load_attention_results()
        if df_with_attention is None:
            print("Error: DataFrame with attention is not available.")
            return None, None

        if self.model is None:
            self._load_existing_model()
        if self.data_module is None:
            self.data_module = EnhancedDataModule(self.config)
            self.data_module.setup()
        
        current_device = self.model.device if hasattr(self.model, 'device') else DEVICE_OSR
        self.oe_extractor = OEExtractorEnhanced(
            config=self.config,
            model_pl=self.model,
            tokenizer=self.data_module.tokenizer,
            device=current_device
        )
        
        masked_texts_col = self.config.TEXT_COLUMN_IN_OE_FILES
        if masked_texts_col not in df_with_attention.columns:
            print(f"Error: Column '{masked_texts_col}' not found.")
            return df_with_attention, None

        # 모든 데이터에 대해 feature 추출
        all_texts = []
        
        if self.config.EXPERIMENT_MODE == "nlp":
            # NLP 모드: 모든 텍스트 처리 (exclude class 없음)
            for idx, row in df_with_attention.iterrows():
                all_texts.append(row[masked_texts_col])
        else:
            # Syslog 모드: exclude class 처리
            exclude_class_lower = self.config.EXCLUDE_CLASS_FOR_TRAINING.lower()
            for idx, row in df_with_attention.iterrows():
                if str(row[self.config.CLASS_COLUMN]).lower() == exclude_class_lower:
                    all_texts.append(row[self.config.TEXT_COLUMN])  # Unknown 클래스는 원본 텍스트
                else:
                    all_texts.append(row[masked_texts_col])  # Known 클래스는 마스킹된 텍스트
        
        print(f"Processing {len(all_texts)} samples for OE metrics...")
        
        # 데이터셋 생성
        if self.config.EXPERIMENT_MODE == "nlp":
            max_length = self.config.NLP_MAX_LENGTH
            batch_size = self.config.NLP_BATCH_SIZE
        else:
            max_length = self.config.MAX_LENGTH
            batch_size = self.config.BATCH_SIZE
        
        dataset = MaskedTextDatasetForMetrics(
            all_texts,
            self.data_module.tokenizer,
            max_length,
            is_nlp_mode=(self.config.EXPERIMENT_MODE == "nlp")
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=self.config.NUM_WORKERS, shuffle=False)
        
        # 메트릭 및 feature 추출
        exclude_class = None if self.config.EXPERIMENT_MODE == "nlp" else self.config.EXCLUDE_CLASS_FOR_TRAINING
        attention_metrics_df, features = self.oe_extractor.extract_attention_metrics(
            dataloader, original_df=df_with_attention, exclude_class=exclude_class
        )
        
        # 길이 확인
        assert len(df_with_attention) == len(attention_metrics_df) == len(features), \
            f"Length mismatch: df={len(df_with_attention)}, metrics={len(attention_metrics_df)}, features={len(features)}"
        
        # DataFrame 결합
        df_with_metrics = df_with_attention.reset_index(drop=True)
        df_with_metrics = pd.concat([df_with_metrics, attention_metrics_df.reset_index(drop=True)], axis=1)
        
        # removed_word_attention 계산
        if self.attention_analyzer:
            df_with_metrics = self.oe_extractor.compute_removed_word_attention(
                df_with_metrics, self.attention_analyzer, exclude_class=exclude_class
            )
        
        # OE 데이터셋 추출
        self.oe_extractor.extract_oe_datasets(df_with_metrics, exclude_class=exclude_class)
        
        # 결과 저장
        metrics_output_path = os.path.join(self.config.ATTENTION_DATA_DIR, f"df_with_all_metrics_{self.config.EXPERIMENT_MODE}.csv")
        df_with_metrics.to_csv(metrics_output_path, index=False)
        print(f"DataFrame with all metrics saved: {metrics_output_path}")
        
        if features:
            features_path = os.path.join(self.config.ATTENTION_DATA_DIR, f"extracted_features_{self.config.EXPERIMENT_MODE}.npy")
            np.save(features_path, np.array(features))
            print(f"Extracted features saved: {features_path}")

        return df_with_metrics, features
    
    def run_stage4_visualization(self, df_with_metrics: Optional[pd.DataFrame], features: Optional[List[np.ndarray]]):
        """4단계: 시각화"""
        if not self.config.STAGE_VISUALIZATION:
            print("Skipping Stage 4: Visualization")
            return

        print(f"\n{'='*50}\nSTAGE 4: VISUALIZATION\n{'='*50}")
        
        if df_with_metrics is None or features is None:
            df_with_metrics, features = self._load_final_metrics_and_features()
        
        if df_with_metrics is None:
            print("Error: DataFrame with metrics not available for visualization.")
            return

        # 메트릭 분포 시각화
        self.visualizer.visualize_all_metrics(df_with_metrics)
        
        # t-SNE 시각화 (features가 있고 data_module이 있는 경우)
        if features and self.data_module:
            self.visualizer.visualize_oe_candidates(
                df_with_metrics, features,
                self.data_module.label2id, self.data_module.id2label
            )
        elif not features:
            print("No features available for t-SNE visualization.")
        elif not self.data_module:
            print("DataModule not available for t-SNE visualization.")
            
        print("Visualization complete!")
    
    def run_stage5_osr_experiments(self):
        """5단계: OSR 실험"""
        if not self.config.STAGE_OSR_EXPERIMENTS:
            print("Skipping Stage 5: OSR Experiments")
            return

        print(f"\n{'='*50}\nSTAGE 5: OSR EXPERIMENTS ({self.config.EXPERIMENT_MODE.upper()})\n{'='*50}")

        if self.data_module is None or self.data_module.num_labels is None:
            print("Error: DataModule not set up for OSR experiments.")
            if not self.config.STAGE_MODEL_TRAINING:
                print("Attempting to set up DataModule...")
                self.data_module = EnhancedDataModule(self.config)
                self.data_module.setup()
                if self.data_module.num_labels is None:
                    print("Critical Error: Failed to set up DataModule.")
                    return
            else:
                return

        if self.config.EXPERIMENT_MODE == "nlp":
            self._run_nlp_osr_experiments()
        else:
            self._run_syslog_osr_experiments()
    
    def _run_nlp_osr_experiments(self):
        """NLP OSR 실험 실행"""
        print("\n--- Running NLP OSR Experiments ---")
        
        # 토크나이저 준비
        tokenizer = self.data_module.tokenizer
        
        # ID 데이터 준비
        id_train_dataset, id_test_dataset, num_classes, \
        label_encoder, id_label2id, id_id2label = prepare_nlp_id_data_for_osr(
            self.data_module, tokenizer, self.config.OSR_NLP_MAX_LENGTH
        )
        
        if id_train_dataset is None or num_classes == 0:
            print("Error: Failed to prepare ID data for NLP OSR.")
            return
        
        class_names = list(id_id2label.values()) if id_id2label else [f"Class_{i}" for i in range(num_classes)]
        
        # 데이터 로더 생성
        id_train_loader = DataLoader(
            id_train_dataset, batch_size=self.config.OSR_NLP_BATCH_SIZE, shuffle=True,
            num_workers=self.config.OSR_NUM_DATALOADER_WORKERS, pin_memory=True
        )
        id_test_loader = DataLoader(
            id_test_dataset, batch_size=self.config.OSR_NLP_BATCH_SIZE,
            num_workers=self.config.OSR_NUM_DATALOADER_WORKERS, pin_memory=True
        )
        
        # OOD 데이터 준비 (WikiText-2)
        ood_dataset = prepare_wikitext_ood_data_for_osr(tokenizer, self.config.OSR_NLP_MAX_LENGTH)
        ood_loader = DataLoader(
            ood_dataset, batch_size=self.config.OSR_NLP_BATCH_SIZE,
            num_workers=self.config.OSR_NUM_DATALOADER_WORKERS, pin_memory=True
        ) if ood_dataset else None
        
        ood_dataset_name = "WikiText2"
        
        # OSR 실험 결과 저장
        all_osr_results = {}
        
        # 1. 표준 OSR 모델 (OE 없음)
        if not self.config.OSR_SKIP_STANDARD_MODEL:
            print("\n--- Running Standard NLP OSR Model (No OE) ---")
            std_results, _ = self._run_single_nlp_osr_experiment(
                tokenizer, num_classes, id_label2id, id_id2label, class_names,
                id_train_loader, id_test_loader, ood_loader,
                oe_source_name=None, oe_data_path=None,
                ood_dataset_name=ood_dataset_name
            )
            all_osr_results.update(std_results)
        
        # 2. OE 데이터를 사용한 OSR 모델들
        print(f"\n--- Running NLP OSR Experiments with OE data ---")
        oe_files = [f for f in os.listdir(self.config.OE_DATA_DIR) if f.endswith('.csv') and 'extended' not in f]
        
        if not oe_files:
            print("No OE dataset files found. Skipping OE-enhanced OSR experiments.")
        else:
            for oe_filename in oe_files:
                oe_data_path = os.path.join(self.config.OE_DATA_DIR, oe_filename)
                oe_source_name = os.path.splitext(oe_filename)[0]
                
                print(f"\n--- NLP OSR Experiment with OE: {oe_source_name} ---")
                oe_results, _ = self._run_single_nlp_osr_experiment(
                    tokenizer, num_classes, id_label2id, id_id2label, class_names,
                    id_train_loader, id_test_loader, ood_loader,
                    oe_source_name=oe_source_name, oe_data_path=oe_data_path,
                    ood_dataset_name=ood_dataset_name
                )
                all_osr_results.update(oe_results)
        
        # 결과 저장
        self._save_osr_results(all_osr_results, f"nlp_{self.config.CURRENT_NLP_DATASET}")
    
    def _run_single_nlp_osr_experiment(self, tokenizer, num_classes: int, id_label2id: Dict, id_id2label: Dict, class_names: List[str],
                                       id_train_loader: DataLoader, id_test_loader: DataLoader, ood_loader: Optional[DataLoader],
                                       oe_source_name: Optional[str], oe_data_path: Optional[str], ood_dataset_name: str) -> Tuple[Dict, Dict]:
        """단일 NLP OSR 실험 실행"""
        
        experiment_tag = f"NLP_{self.config.CURRENT_NLP_DATASET.upper()}"
        if oe_source_name:
            experiment_tag += f"_OE_{oe_source_name}"
        else:
            experiment_tag += "_Standard"
        
        print(f"\n===== Starting NLP OSR Experiment: {experiment_tag} =====")
        
        # 결과 디렉토리 생성
        sanitized_oe_name = re.sub(r'[^\w\-.]+', '_', oe_source_name) if oe_source_name else "Standard"
        exp_result_subdir = os.path.join(f"NLP_{self.config.CURRENT_NLP_DATASET}", f"OE_{sanitized_oe_name}" if oe_source_name else "Standard")
        
        current_result_dir = os.path.join(self.config.OSR_RESULT_DIR, exp_result_subdir)
        current_model_dir = os.path.join(self.config.OSR_MODEL_DIR, exp_result_subdir)
        os.makedirs(current_result_dir, exist_ok=True)
        if self.config.OSR_SAVE_MODEL_PER_EXPERIMENT:
            os.makedirs(current_model_dir, exist_ok=True)
        
        # 모델 초기화
        model_osr = NLPModelOOD(
            vocab_size=self.config.OSR_NLP_VOCAB_SIZE,
            embed_dim=self.config.OSR_NLP_EMBED_DIM,
            hidden_dim=self.config.OSR_NLP_HIDDEN_DIM,
            num_classes=num_classes,
            num_layers=self.config.OSR_NLP_NUM_LAYERS,
            dropout=self.config.OSR_NLP_DROPOUT,
            model_type=self.config.OSR_NLP_MODEL_TYPE
        ).to(DEVICE_OSR)
        
        # 모델 저장 경로
        model_filename = f"nlp_osr_{experiment_tag}_{num_classes}cls_seed{self.config.RANDOM_STATE}.pt"
        model_save_path = os.path.join(current_model_dir, model_filename)
        
        experiment_results = {}
        experiment_data_for_plots = {}
        epoch_losses = []
        
        # 훈련 또는 로드
        if self.config.OSR_EVAL_ONLY:
            if os.path.exists(model_save_path):
                print(f"Loading pre-trained NLP OSR model from {model_save_path}...")
                model_osr.load_state_dict(torch.load(model_save_path, map_location=DEVICE_OSR, weights_only=False))
            else:
                print(f"Error: Model path '{model_save_path}' not found for OSR_EVAL_ONLY.")
                return {}, {}
        else:
            # 옵티마이저 설정
            optimizer = optim.Adam(model_osr.parameters(), lr=self.config.OSR_NLP_LEARNING_RATE)
            
            if oe_data_path:
                # OE 데이터로 훈련
                print(f"Preparing OE data for NLP OSR from: {oe_data_path}")
                oe_dataset = prepare_nlp_oe_data_for_osr(tokenizer, self.config.OSR_NLP_MAX_LENGTH, oe_data_path, self.config.TEXT_COLUMN_IN_OE_FILES)
                
                if oe_dataset:
                    oe_loader = DataLoader(
                        oe_dataset, batch_size=self.config.OSR_NLP_BATCH_SIZE, shuffle=True,
                        num_workers=self.config.OSR_NUM_DATALOADER_WORKERS, pin_memory=True
                    )
                    
                    # OE 훈련
                    model_osr.train()
                    print(f"Starting NLP OSR OE training for '{experiment_tag}'...")
                    
                    for epoch in range(self.config.OSR_NLP_NUM_EPOCHS):
                        oe_iter = iter(oe_loader)
                        total_loss, total_id_loss, total_oe_loss = 0, 0, 0
                        
                        progress_bar = tqdm(id_train_loader, desc=f"NLP OE OSR Epoch {epoch+1}/{self.config.OSR_NLP_NUM_EPOCHS}", leave=False)
                        
                        for batch in progress_bar:
                            input_ids = batch['input_ids'].to(DEVICE_OSR)
                            attention_mask = batch['attention_mask'].to(DEVICE_OSR)
                            labels = batch['label'].to(DEVICE_OSR)
                            
                            # OE 배치 가져오기
                            try:
                                oe_batch = next(oe_iter)
                            except StopIteration:
                                oe_iter = iter(oe_loader)
                                oe_batch = next(oe_iter)
                            
                            oe_input_ids = oe_batch['input_ids'].to(DEVICE_OSR)
                            oe_attention_mask = oe_batch['attention_mask'].to(DEVICE_OSR)
                            
                            optimizer.zero_grad()
                            
                            # ID loss
                            id_logits = model_osr(input_ids, attention_mask)
                            id_loss = F.cross_entropy(id_logits, labels)
                            
                            # OE loss (uniform distribution)
                            oe_logits = model_osr(oe_input_ids, oe_attention_mask)
                            num_classes_oe = oe_logits.size(1)
                            log_softmax_oe = F.log_softmax(oe_logits, dim=1)
                            uniform_target = torch.full_like(oe_logits, 1.0 / num_classes_oe)
                            oe_loss = F.kl_div(log_softmax_oe, uniform_target, reduction='batchmean', log_target=False)
                            
                            total_batch_loss = id_loss + self.config.OSR_OE_LAMBDA * oe_loss
                            total_batch_loss.backward()
                            optimizer.step()
                            
                            total_loss += total_batch_loss.item()
                            total_id_loss += id_loss.item()
                            total_oe_loss += oe_loss.item()
                            
                            progress_bar.set_postfix({
                                'Total': f"{total_batch_loss.item():.3f}",
                                'ID': f"{id_loss.item():.3f}",
                                'OE': f"{oe_loss.item():.3f}"
                            })
                        
                        avg_loss = total_loss / len(id_train_loader)
                        avg_id_loss = total_id_loss / len(id_train_loader)
                        avg_oe_loss = total_oe_loss / len(id_train_loader)
                        epoch_losses.append(avg_loss)
                        
                        print(f"NLP OE OSR Epoch {epoch+1}/{self.config.OSR_NLP_NUM_EPOCHS}, Avg Loss: {avg_loss:.4f} (ID: {avg_id_loss:.4f}, OE: {avg_oe_loss:.4f})")
                else:
                    print(f"Failed to load OE data from '{oe_data_path}'. Training standard model.")
                    epoch_losses = self._train_standard_nlp_osr(model_osr, id_train_loader, optimizer, experiment_tag)
            else:
                # 표준 훈련 (OE 없음)
                epoch_losses = self._train_standard_nlp_osr(model_osr, id_train_loader, optimizer, experiment_tag)
            
            # 모델 저장
            if self.config.OSR_SAVE_MODEL_PER_EXPERIMENT:
                torch.save(model_osr.state_dict(), model_save_path)
                print(f"NLP OSR Model saved to {model_save_path}")
            
            # 훈련 곡선 플롯
            if not self.config.OSR_NO_PLOT_PER_EXPERIMENT and epoch_losses:
                self._plot_training_curve(epoch_losses, experiment_tag, current_result_dir)
        
        # 평가
        if ood_loader is None:
            print(f"Warning: No OOD evaluation data for experiment '{experiment_tag}'.")
        
        results_osr, data_osr = evaluate_nlp_osr(
            model_osr, id_test_loader, ood_loader, DEVICE_OSR,
            self.config.OSR_TEMPERATURE, self.config.OSR_THRESHOLD_PERCENTILE, return_data=True
        )
        
        print(f"  NLP OSR Results ({experiment_tag} vs {ood_dataset_name}): {results_osr}")
        
        # 결과 키 생성
        metric_key_prefix = f"NLP_{self.config.CURRENT_NLP_DATASET.upper()}_"
        metric_key_prefix += f"OE_{oe_source_name}" if oe_source_name else "Standard"
        full_metric_key = f"{metric_key_prefix}+{ood_dataset_name}"
        
        experiment_results[full_metric_key] = results_osr
        experiment_data_for_plots[full_metric_key] = data_osr
        
        # 플롯 생성
        if not self.config.OSR_NO_PLOT_PER_EXPERIMENT:
            plot_filename_prefix = re.sub(r'[^\w\-]+', '_', full_metric_key)
            if data_osr['id_scores'] is not None and data_osr['ood_scores'] is not None and len(data_osr['ood_scores']) > 0:
                plot_confidence_histograms_osr(data_osr['id_scores'], data_osr['ood_scores'],
                                        f'Conf - {experiment_tag} vs {ood_dataset_name}',
                                        os.path.join(current_result_dir, f'{plot_filename_prefix}_hist.png'))
                plot_roc_curve_osr(data_osr['id_scores'], data_osr['ood_scores'],
                            f'ROC - {experiment_tag} vs {ood_dataset_name}',
                            os.path.join(current_result_dir, f'{plot_filename_prefix}_roc.png'))
                plot_tsne_osr(data_osr['id_features'], data_osr['ood_features'],
                            f't-SNE - {experiment_tag} vs {ood_dataset_name}',
                            os.path.join(current_result_dir, f'{plot_filename_prefix}_tsne.png'),
                            seed=self.config.RANDOM_STATE)
            
            if data_osr['id_labels_true'] is not None and len(data_osr['id_labels_true']) > 0:
                cm = confusion_matrix(data_osr['id_labels_true'], data_osr['id_labels_pred'], labels=np.arange(num_classes))
                plot_confusion_matrix_osr(cm, class_names,
                                    f'CM - {experiment_tag} (ID Test)',
                                    os.path.join(current_result_dir, f'{plot_filename_prefix}_cm.png'))
        
        # 메모리 정리
        del model_osr
        gc.collect()
        torch.cuda.empty_cache()
        
        return experiment_results, experiment_data_for_plots
    
    def _train_standard_nlp_osr(self, model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, experiment_name: str) -> List[float]:
        """표준 NLP OSR 훈련"""
        model.train()
        epoch_losses = []
        print(f"Starting standard NLP OSR training for '{experiment_name}'...")
        
        for epoch in range(self.config.OSR_NLP_NUM_EPOCHS):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"NLP Std OSR Epoch {epoch+1}/{self.config.OSR_NLP_NUM_EPOCHS}", leave=False)
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(DEVICE_OSR)
                attention_mask = batch['attention_mask'].to(DEVICE_OSR)
                labels = batch['label'].to(DEVICE_OSR)
                
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f"{loss.item():.3f}"})
            
            avg_loss = total_loss / len(train_loader)
            epoch_losses.append(avg_loss)
            print(f"NLP Std OSR Epoch {epoch+1}/{self.config.OSR_NLP_NUM_EPOCHS}, Avg Loss: {avg_loss:.4f}")
        
        return epoch_losses
    
    def _run_syslog_osr_experiments(self):
        """Syslog OSR 실험 실행 (기존 코드 사용)"""
        print("\n--- Running Syslog OSR Experiments ---")
        
        # 토크나이저 준비
        osr_tokenizer = RobertaTokenizer.from_pretrained(self.config.OSR_MODEL_TYPE)
        
        # ID 데이터 준비
        id_train_dataset_osr, id_test_dataset_osr, num_osr_classes, \
        osr_label_encoder, osr_id_label2id, osr_id_id2label = prepare_id_data_for_osr(
            self.data_module, osr_tokenizer, self.config.OSR_MAX_LENGTH
        )
        
        if id_train_dataset_osr is None or num_osr_classes == 0:
            print("Error: Failed to prepare ID data for Syslog OSR.")
            return
        
        osr_known_class_names = list(osr_id_id2label.values()) if osr_id_id2label else [f"Class_{i}" for i in range(num_osr_classes)]
        
        # 데이터 로더 생성
        id_train_loader_osr = DataLoader(
            id_train_dataset_osr, batch_size=self.config.OSR_BATCH_SIZE, shuffle=True,
            num_workers=self.config.OSR_NUM_DATALOADER_WORKERS, pin_memory=True,
            persistent_workers=self.config.OSR_NUM_DATALOADER_WORKERS > 0
        )
        id_test_loader_osr = DataLoader(
            id_test_dataset_osr, batch_size=self.config.OSR_BATCH_SIZE,
            num_workers=self.config.OSR_NUM_DATALOADER_WORKERS, pin_memory=True
        )
        
        # OOD 데이터 준비
        ood_eval_dataset_osr = prepare_syslog_ood_data_for_osr(
            osr_tokenizer, self.config.OSR_MAX_LENGTH,
            self.config.OOD_SYSLOG_UNKNOWN_PATH_OSR,
            self.config.TEXT_COLUMN, self.config.CLASS_COLUMN,
            self.config.OOD_TARGET_CLASS_OSR
        )
        ood_eval_loader_osr = DataLoader(
            ood_eval_dataset_osr, batch_size=self.config.OSR_BATCH_SIZE,
            num_workers=self.config.OSR_NUM_DATALOADER_WORKERS, pin_memory=True
        ) if ood_eval_dataset_osr else None
        
        ood_dataset_eval_name_tag = self.config.OOD_TARGET_CLASS_OSR
        
        # OSR 실험 결과 저장
        all_osr_experiments_results = {}
        
        # 1. 표준 OSR 모델 (OE 없음)
        if not self.config.OSR_SKIP_STANDARD_MODEL:
            print("\n--- Running Standard Syslog OSR Model (No OE) ---")
            # _run_single_osr_experiment 함수는 원본 oe2.py에서 가져와야 함
            # 여기서는 간단히 placeholder로 처리
            # 실제 구현에서는 원본 코드의 해당 함수를 복사해야 함
        
        # 2. OE 데이터를 사용한 OSR 모델들
        print(f"\n--- Running Syslog OSR Experiments with OE data ---")
        oe_files = [f for f in os.listdir(self.config.OE_DATA_DIR) if f.endswith('.csv') and 'extended' not in f]
        
        if not oe_files:
            print("No OE dataset files found. Skipping OE-enhanced Syslog OSR experiments.")
        else:
            for oe_filename in oe_files:
                oe_data_path = os.path.join(self.config.OE_DATA_DIR, oe_filename)
                oe_source_name = os.path.splitext(oe_filename)[0]
                print(f"\n--- Syslog OSR Experiment with OE: {oe_source_name} ---")
                # 여기서도 _run_single_osr_experiment 함수 호출 필요
        
        # 결과 저장
        self._save_osr_results(all_osr_experiments_results, "syslog")
    
    def _save_osr_results(self, results: Dict, experiment_type: str):
        """OSR 결과 저장"""
        print(f"\n===== OSR Experiments Overall Results Summary ({experiment_type}) =====")
        
        if results:
            results_df = pd.DataFrame.from_dict(results, orient='index')
            results_df = results_df.sort_index()
            print("Overall OSR Performance Metrics DataFrame:")
            print(results_df)
            
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_filename_base = f"osr_overall_summary_{experiment_type}_{timestamp_str}"
            
            # CSV 저장
            overall_csv_path = os.path.join(self.config.OSR_RESULT_DIR, f'{summary_filename_base}.csv')
            results_df.to_csv(overall_csv_path, index=True)
            print(f"\nOverall OSR results saved to CSV: {overall_csv_path}")
            
            # 설정 딕셔너리 준비
            osr_args_to_save = {k: v for k, v in self.config.__class__.__dict__.items() 
                                if k.startswith('OSR_') or k in ['RANDOM_STATE', 'TEXT_COLUMN', 'CLASS_COLUMN']}
            osr_args_to_save['EXPERIMENT_MODE'] = self.config.EXPERIMENT_MODE
            osr_args_to_save['CURRENT_NLP_DATASET'] = getattr(self.config, 'CURRENT_NLP_DATASET', None)
            osr_args_to_save['OE_DATA_DIR_USED'] = self.config.OE_DATA_DIR
            
            # TXT 저장
            overall_txt_path = os.path.join(self.config.OSR_RESULT_DIR, f'{summary_filename_base}.txt')
            with open(overall_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"--- OSR Experiment Arguments ({experiment_type}) ---\n")
                f.write(json.dumps(osr_args_to_save, indent=4, default=str))
                f.write(f"\n\n--- Overall OSR Metrics ({experiment_type}) ---\n")
                f.write(results_df.to_string())
            print(f"Overall OSR results and arguments saved to TXT: {overall_txt_path}")
            
            # JSON 저장
            overall_json_path = os.path.join(self.config.OSR_RESULT_DIR, f'{summary_filename_base}.json')
            summary_json_data = {
                'arguments_osr': osr_args_to_save,
                'timestamp': timestamp_str,
                'experiment_type': experiment_type,
                'results_osr': results
            }
            with open(overall_json_path, 'w', encoding='utf-8') as f:
                json.dump(summary_json_data, f, indent=4, default=str)
            print(f"Overall OSR results saved to JSON: {overall_json_path}")
        else:
            print("No OSR performance metrics were generated.")
        print(f"\nOSR Experiments ({experiment_type}) Finished.")
    
    def _plot_training_curve(self, losses: List[float], experiment_name: str, save_dir: str):
        """훈련 곡선 플롯"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss Curve - {experiment_name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'{experiment_name}_training_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curve saved: {save_path}")
    
    def run_full_pipeline(self):
        """전체 파이프라인 실행"""
        print(f"Starting Enhanced OE Extraction & OSR Pipeline ({self.config.EXPERIMENT_MODE.upper()})...")
        
        df_with_attention, df_with_metrics, features = None, None, None
        
        # 1단계: 모델 훈련
        df_with_attention = self.run_stage1_model_training()
        
        # 2단계: 어텐션 추출
        df_with_attention = self.run_stage2_attention_extraction()
        
        # 3단계: OE 추출
        df_with_metrics, features = self.run_stage3_oe_extraction(df_with_attention)
        
        # 4단계: 시각화
        self.run_stage4_visualization(df_with_metrics, features)
        
        # 5단계: OSR 실험
        self.run_stage5_osr_experiments()
        
        self._print_final_summary()
        print(f"\nEnhanced OE Extraction & OSR Pipeline ({self.config.EXPERIMENT_MODE.upper()}) Complete!")
    
    # === Helper Methods ===
    def _check_existing_model(self) -> bool:
        """기존 모델 체크"""
        return (os.path.exists(self.config.MODEL_SAVE_DIR) and 
                any(file.endswith('.ckpt') for file in os.listdir(self.config.MODEL_SAVE_DIR)))
    
    def _load_existing_model(self, checkpoint_callback=None):
        """기존 모델 로드"""
        if self.data_module is None:
            self.data_module = EnhancedDataModule(self.config)
            self.data_module.setup()

        model_path = None
        if checkpoint_callback and hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path:
            model_path = checkpoint_callback.best_model_path
        else:
            checkpoint_files = [f for f in os.listdir(self.config.MODEL_SAVE_DIR) if f.endswith('.ckpt')]
            if checkpoint_files:
                model_path = os.path.join(self.config.MODEL_SAVE_DIR, sorted(checkpoint_files)[-1])
        
        if model_path and os.path.exists(model_path):
            print(f"Loading {self.config.EXPERIMENT_MODE} model from: {model_path}")
            self.model = EnhancedModel.load_from_checkpoint(
                model_path,
                config=self.config,
                num_labels=self.data_module.num_labels,
                label2id=self.data_module.label2id,
                id2label=self.data_module.id2label,
                class_weights=self.data_module.class_weights,
                tokenizer=self.data_module.tokenizer if self.config.EXPERIMENT_MODE == "nlp" else None
            )
            print(f"{self.config.EXPERIMENT_MODE} model loaded successfully!")
        else:
            print(f"Warning: No model checkpoint found. Cannot load model.")
            if self.config.STAGE_ATTENTION_EXTRACTION or self.config.STAGE_OE_EXTRACTION:
                raise FileNotFoundError("Cannot proceed without a model.")
    
    def _load_best_model(self, checkpoint_callback):
        """최적 모델 로드"""
        if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
            print(f"Loading best {self.config.EXPERIMENT_MODE} model: {checkpoint_callback.best_model_path}")
            if self.data_module is None:
                self.data_module = EnhancedDataModule(self.config)
                self.data_module.setup()

            self.model = EnhancedModel.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                config=self.config,
                num_labels=self.data_module.num_labels,
                label2id=self.data_module.label2id,
                id2label=self.data_module.id2label,
                class_weights=self.data_module.class_weights,
                tokenizer=self.data_module.tokenizer if self.config.EXPERIMENT_MODE == "nlp" else None
            )
            print(f"Best {self.config.EXPERIMENT_MODE} model loaded successfully!")
        else:
            print("Warning: Best model path not found. Using current model state.")
    
    def _load_attention_results(self) -> Optional[pd.DataFrame]:
        """어텐션 결과 로드"""
        attention_file = os.path.join(self.config.ATTENTION_DATA_DIR, f"df_with_attention_{self.config.EXPERIMENT_MODE}.csv")
        if os.path.exists(attention_file):
            print(f"Loading attention results from: {attention_file}")
            df = pd.read_csv(attention_file)
            if 'top_attention_words' in df.columns:
                df['top_attention_words'] = df['top_attention_words'].apply(safe_literal_eval)
            return df
        print(f"Attention results file not found: {attention_file}")
        return None
    
    def _load_final_metrics_and_features(self) -> Tuple[Optional[pd.DataFrame], Optional[List[np.ndarray]]]:
        """최종 메트릭 및 feature 로드"""
        metrics_file = os.path.join(self.config.ATTENTION_DATA_DIR, f"df_with_all_metrics_{self.config.EXPERIMENT_MODE}.csv")
        features_file = os.path.join(self.config.ATTENTION_DATA_DIR, f"extracted_features_{self.config.EXPERIMENT_MODE}.npy")
        
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
            features_arr = np.load(features_file, allow_pickle=True).tolist()
        else:
            print(f"Extracted features file not found: {features_file}")
        
        return df_metrics, features_arr
    
    def _print_attention_samples(self, df: pd.DataFrame, num_samples: int = 3):
        """어텐션 분석 샘플 출력"""
        if df is None or df.empty:
            print("No data to sample for attention.")
            return
        print(f"\n--- Attention Analysis Samples (Max {num_samples}) ---")
        sample_df = df.sample(min(num_samples, len(df)))
        for i, row in sample_df.iterrows():
            print("-" * 30)
            print(f"Original: {str(row[self.config.TEXT_COLUMN])[:100]}...")
            print(f"Top Words: {row['top_attention_words']}")
            print(f"Masked: {str(row[self.config.TEXT_COLUMN_IN_OE_FILES])[:100]}...")
    
    def _print_final_summary(self):
        """최종 요약 출력"""
        print(f"\n{'='*50}\nPIPELINE SUMMARY ({self.config.EXPERIMENT_MODE.upper()})\n{'='*50}")
        print(f"Experiment Mode: {self.config.EXPERIMENT_MODE}")
        if self.config.EXPERIMENT_MODE == "nlp":
            print(f"NLP Dataset: {self.config.CURRENT_NLP_DATASET}")
        print(f"Output directory: {self.config.OUTPUT_DIR}")
        print("\nGenerated Files (Examples):")
        max_files_to_list = 15
        count = 0
        for root, _, files in os.walk(self.config.OUTPUT_DIR):
            if count >= max_files_to_list:
                break
            for file in files:
                if count >= max_files_to_list:
                    break
                if file.endswith(('.csv', '.png', '.json', '.txt', '.pt', '.ckpt')):
                    print(f"  - {os.path.join(root, file)}")
                    count += 1
            if count >= max_files_to_list and root == self.config.OUTPUT_DIR:
                print("  ... (many more files generated)")

# === Main Function ===
def main():
    parser = argparse.ArgumentParser(description="Enhanced OE Extraction and OSR Experimentation Pipeline with NLP Support")
    
    # 기본 모드 선택
    parser.add_argument('--mode', type=str, choices=['syslog', 'nlp'], default='nlp', 
                       help="Experiment mode: 'syslog' for original syslog experiments, 'nlp' for NLP experiments")
    
    # NLP 관련 인자들
    parser.add_argument('--nlp_dataset', type=str, choices=['20newsgroups', 'trec', 'sst2'], default='20newsgroups',
                       help="NLP dataset to use for experiments")
    parser.add_argument('--nlp_model_type', type=str, choices=['gru', 'lstm'], default='gru',
                       help="NLP model type")
    parser.add_argument('--nlp_epochs', type=int, default=30, help="Number of epochs for NLP training")
    
    # OE 추출 관련 인자들
    parser.add_argument('--attention_percent', type=float, default=Config.ATTENTION_TOP_PERCENT)
    parser.add_argument('--top_words', type=int, default=Config.MIN_TOP_WORDS)
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR)
    
    # Syslog 관련 인자들 (호환성 유지)
    parser.add_argument('--data_path', type=str, default=Config.ORIGINAL_DATA_PATH)
    parser.add_argument('--oe_model_name', type=str, default=Config.MODEL_NAME)
    parser.add_argument('--oe_epochs', type=int, default=Config.NUM_TRAIN_EPOCHS)
    
    # OSR 실험 관련 인자들
    parser.add_argument('--osr_model_type', type=str, default=Config.OSR_MODEL_TYPE)
    parser.add_argument('--osr_epochs', type=int, default=Config.OSR_NUM_EPOCHS)
    parser.add_argument('--ood_data_path_osr', type=str, default=getattr(Config, 'OOD_SYSLOG_UNKNOWN_PATH_OSR', None))
    
    # 단계 제어
    parser.add_argument('--skip_base_training', action='store_true')
    parser.add_argument('--skip_attention_extraction', action='store_true')
    parser.add_argument('--skip_oe_extraction', action='store_true')
    parser.add_argument('--skip_oe_visualization', action='store_true')
    parser.add_argument('--skip_osr_experiments', action='store_true')
    parser.add_argument('--osr_eval_only', action='store_true')
    
    args = parser.parse_args()
    
    # Config 업데이트
    Config.EXPERIMENT_MODE = args.mode
    
    # NLP 설정 업데이트
    if args.mode == 'nlp':
        Config.CURRENT_NLP_DATASET = args.nlp_dataset
        Config.NLP_MODEL_TYPE = args.nlp_model_type
        Config.OSR_NLP_MODEL_TYPE = args.nlp_model_type
        Config.NLP_NUM_EPOCHS = args.nlp_epochs
        Config.OSR_NLP_NUM_EPOCHS = args.osr_epochs
    
    # 공통 설정 업데이트
    Config.ATTENTION_TOP_PERCENT = args.attention_percent
    Config.MIN_TOP_WORDS = args.top_words
    Config.OUTPUT_DIR = args.output_dir
    
    # Syslog 설정 업데이트 (호환성)
    if hasattr(args, 'data_path') and args.data_path:
        Config.ORIGINAL_DATA_PATH = args.data_path
    Config.MODEL_NAME = args.oe_model_name
    Config.NUM_TRAIN_EPOCHS = args.oe_epochs
    Config.OSR_MODEL_TYPE = args.osr_model_type
    Config.OSR_NUM_EPOCHS = args.osr_epochs
    if args.ood_data_path_osr:
        Config.OOD_SYSLOG_UNKNOWN_PATH_OSR = args.ood_data_path_osr
    Config.OSR_EVAL_ONLY = args.osr_eval_only
    
    # 단계 제어 설정
    Config.STAGE_MODEL_TRAINING = not args.skip_base_training
    Config.STAGE_ATTENTION_EXTRACTION = not args.skip_attention_extraction
    Config.STAGE_OE_EXTRACTION = not args.skip_oe_extraction
    Config.STAGE_VISUALIZATION = not args.skip_oe_visualization
    Config.STAGE_OSR_EXPERIMENTS = not args.skip_osr_experiments
    
    # 출력 디렉토리 경로 업데이트
    Config.MODEL_SAVE_DIR = os.path.join(Config.OUTPUT_DIR, "base_classifier_model")
    Config.LOG_DIR = os.path.join(Config.OUTPUT_DIR, "lightning_logs")
    Config.CONFUSION_MATRIX_DIR = os.path.join(Config.LOG_DIR, "confusion_matrices")
    Config.VIS_DIR = os.path.join(Config.OUTPUT_DIR, "oe_extraction_visualizations")
    Config.OE_DATA_DIR = os.path.join(Config.OUTPUT_DIR, "extracted_oe_datasets")
    Config.ATTENTION_DATA_DIR = os.path.join(Config.OUTPUT_DIR, "attention_analysis")
    Config.NLP_DATA_DIR = os.path.join(Config.OUTPUT_DIR, "nlp_datasets")
    Config.OSR_EXPERIMENT_DIR = os.path.join(Config.OUTPUT_DIR, "osr_experiments")
    Config.OSR_MODEL_DIR = os.path.join(Config.OSR_EXPERIMENT_DIR, "models")
    Config.OSR_RESULT_DIR = os.path.join(Config.OSR_EXPERIMENT_DIR, "results")
    Config.DATA_DIR_EXTERNAL_HF = os.path.join(Config.OUTPUT_DIR, 'data_external_hf')
    Config.CACHE_DIR_HF = os.path.join(Config.DATA_DIR_EXTERNAL_HF, "hf_cache")
    
    print(f"--- Enhanced OE/OSR Pipeline ---")
    print(f"Mode: {Config.EXPERIMENT_MODE}")
    if Config.EXPERIMENT_MODE == 'nlp':
        print(f"NLP Dataset: {Config.CURRENT_NLP_DATASET}")
    print(f"Output Dir: {Config.OUTPUT_DIR}")
    
    # 파이프라인 실행
    pipeline = EnhancedOEPipeline(Config)
    pipeline.run_full_pipeline()

if __name__ == '__main__':
    main()

# === 실행 예시 명령어들 ===
"""
# NLP 실험 (20 Newsgroups, TREC, SST-2)
python enhanced_oe_nlp.py --mode nlp --nlp_dataset 20newsgroups --attention_percent 0.2 --top_words 1 --output_dir enhanced_nlp_20news_0.2_1
python enhanced_oe_nlp.py --mode nlp --nlp_dataset trec --attention_percent 0.2 --top_words 1 --output_dir enhanced_nlp_trec_0.2_1
python enhanced_oe_nlp.py --mode nlp --nlp_dataset sst2 --attention_percent 0.2 --top_words 1 --output_dir enhanced_nlp_sst2_0.2_1

# removed_avg_attention 중심 실험
python enhanced_oe_nlp.py --mode nlp --nlp_dataset 20newsgroups --attention_percent 0.01 --top_words 1 --output_dir enhanced_nlp_20news_removed_0.01_1
python enhanced_oe_nlp.py --mode nlp --nlp_dataset trec --attention_percent 0.01 --top_words 1 --output_dir enhanced_nlp_trec_removed_0.01_1
python enhanced_oe_nlp.py --mode nlp --nlp_dataset sst2 --attention_percent 0.01 --top_words 1 --output_dir enhanced_nlp_sst2_removed_0.01_1

# 다양한 하이퍼파라미터 조합
python enhanced_oe_nlp.py --mode nlp --nlp_dataset 20newsgroups --attention_percent 0.05 --top_words 2 --output_dir enhanced_nlp_20news_0.05_2
python enhanced_oe_nlp.py --mode nlp --nlp_dataset 20newsgroups --attention_percent 0.1 --top_words 3 --output_dir enhanced_nlp_20news_0.1_3

# Syslog 실험 (호환성 유지)
python enhanced_oe_nlp.py --mode syslog --attention_percent 0.2 --top_words 1 --output_dir enhanced_syslog_0.2_1
"""