"""
Enhanced Unified OE (Out-of-Distribution) Extractor with NLP Dataset Support
구조화된 Outlier Exposure 실험 - In-Distribution, OE, Test-Out 구분
RoBERTa-base 모델 지원 추가
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
    AutoModel,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    RobertaTokenizer,
    RobertaModel,
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

# NLTK 초기화
NLTK_DATA_PATH = os.path.expanduser('~/AppData/Roaming/nltk_data')
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_PATH)

# 전역 플래그로 한 번만 다운로드하도록 제어
_NLTK_DOWNLOADS_DONE = False

def ensure_nltk_data():
    """NLTK 데이터가 있는지 확인하고, 필요시 다운로드"""
    global _NLTK_DOWNLOADS_DONE
    
    if _NLTK_DOWNLOADS_DONE:
        return
    
    downloads_needed = []
    
    # 필요한 데이터 확인
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        downloads_needed.append('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        downloads_needed.append('stopwords')
    
    # 필요한 것만 다운로드
    if downloads_needed:
        print(f"Downloading required NLTK data: {downloads_needed}")
        for item in downloads_needed:
            nltk.download(item, quiet=True, download_dir=NLTK_DATA_PATH)
    
    _NLTK_DOWNLOADS_DONE = True

# 초기 다운로드 실행
ensure_nltk_data()

# --- Configuration Class ---
class Config:
    """Enhanced Configuration Class - 구조화된 Outlier Exposure 실험"""
    
    # === 일반 설정 ===
    EXPERIMENT_MODE = "nlp"
    RANDOM_STATE = 42
    
    # === 데이터셋 설정 ===
    # In-Distribution 데이터셋 설정
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
    CURRENT_NLP_DATASET = '20newsgroups'
    
    # Outlier Exposure 데이터셋 설정
    WIKITEXT_VERSION = 'wikitext-2-raw-v1'
    OE_SAMPLE_SIZE = 5000  # OE 데이터셋 샘플링 크기
    
    # Test-Out 데이터셋 설정
    TEST_OUT_DATASET = 'wmt16'  # WMT16을 test-out 데이터셋으로 사용
    TEST_OUT_SAMPLE_SIZE = 1000  # 테스트용 샘플링 크기
    
    # === 출력 디렉토리 설정 ===
    OUTPUT_DIR = 'structured_oe_nlp_results'
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "base_classifier_model")
    LOG_DIR = os.path.join(OUTPUT_DIR, "lightning_logs")
    CONFUSION_MATRIX_DIR = os.path.join(LOG_DIR, "confusion_matrices")
    VIS_DIR = os.path.join(OUTPUT_DIR, "oe_extraction_visualizations")
    OE_DATA_DIR = os.path.join(OUTPUT_DIR, "extracted_oe_datasets")
    ATTENTION_DATA_DIR = os.path.join(OUTPUT_DIR, "attention_analysis")
    NLP_DATA_DIR = os.path.join(OUTPUT_DIR, "nlp_datasets")
    
    # === 모델 설정 ===
    NLP_MODEL_TYPE = "gru"  # "gru", "lstm" 또는 "roberta"
    
    # RNN 모델 설정
    NLP_VOCAB_SIZE = 10000
    NLP_EMBED_DIM = 300
    NLP_HIDDEN_DIM = 512
    NLP_NUM_LAYERS = 2
    NLP_DROPOUT = 0.3
    NLP_MAX_LENGTH = 512
    NLP_BATCH_SIZE = 64
    NLP_NUM_EPOCHS = 20
    NLP_LEARNING_RATE = 1e-3
    
    # RoBERTa 모델 설정
    TRANSFORMER_MODEL_NAME = "roberta-base"
    TRANSFORMER_LEARNING_RATE = 2e-5
    TRANSFORMER_BATCH_SIZE = 32  # 메모리 고려
    TRANSFORMER_MAX_LENGTH = 256
    TRANSFORMER_NUM_EPOCHS = 5
    
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
    
    # === 어텐션 설정 ===
    ATTENTION_TOP_PERCENT = 0.20
    MIN_TOP_WORDS = 1
    TOP_K_ATTENTION = 3
    ATTENTION_LAYER = -1
    
    # === OE 필터링 설정 ===
    METRIC_SETTINGS = {
        'attention_entropy': {'percentile': 75, 'mode': 'higher'},
        'max_attention': {'percentile': 15, 'mode': 'lower'},
        'removed_avg_attention': {'percentile': 85, 'mode': 'higher'},
        'top_k_avg_attention': {'percentile': 25, 'mode': 'lower'}
    }
    FILTERING_SEQUENCE = [
        ('removed_avg_attention', {'percentile': 85, 'mode': 'higher'}),
        ('attention_entropy', {'percentile': 75, 'mode': 'higher'}),
        ('max_attention', {'percentile': 15, 'mode': 'lower'})
    ]
    TEXT_COLUMN_IN_OE_FILES = 'masked_text_attention'
    
    # === OSR 실험 설정 ===
    OSR_EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, "osr_experiments")
    OSR_MODEL_DIR = os.path.join(OSR_EXPERIMENT_DIR, "models")
    OSR_RESULT_DIR = os.path.join(OSR_EXPERIMENT_DIR, "results")
    
    # OSR 설정
    OSR_NLP_MODEL_TYPE = "gru"  # NLP 모델 타입과 동일하게 설정됨
    OSR_NLP_VOCAB_SIZE = 10000
    OSR_NLP_EMBED_DIM = 300
    OSR_NLP_HIDDEN_DIM = 512
    OSR_NLP_NUM_LAYERS = 2
    OSR_NLP_DROPOUT = 0.3
    OSR_NLP_MAX_LENGTH = 512
    OSR_NLP_BATCH_SIZE = 64
    OSR_NLP_NUM_EPOCHS = 15
    OSR_NLP_LEARNING_RATE = 1e-3
    OSR_OE_LAMBDA = 1.0  # OE 손실 가중치
    OSR_TEMPERATURE = 1.0
    OSR_THRESHOLD_PERCENTILE = 5.0
    
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
            filepath = os.path.join(cls.OUTPUT_DIR, 'config_structured_nlp.json')
        
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def preprocess_text_for_transformer(text):
    """Transformer 모델을 위한 텍스트 전처리"""
    if not isinstance(text, str):
        return ""
    # 기본 클리닝 (특수문자 유지 - transformer 토크나이저가 처리)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_nltk(text):
    """NLTK를 사용한 토큰화"""
    if not text:
        return []
    
    # 글로벌 플래그 확인
    global _NLTK_DOWNLOADS_DONE
    if not _NLTK_DOWNLOADS_DONE:
        ensure_nltk_data()
    
    try:
        result = word_tokenize(text)
        return result if result is not None else []
    except Exception as e:
        print(f"NLTK tokenization failed: {e}. Using simple split.")
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

# === 데이터셋 로더들 ===
class NLPDatasetLoader:
    """NLP 데이터셋 로더 클래스 - 구조화된 Outlier Exposure 실험용"""
    
    @staticmethod
    def load_in_distribution_dataset(dataset_name):
        """In-Distribution 데이터셋 로드"""
        print(f"Loading In-Distribution dataset: {dataset_name}")
        
        if dataset_name == '20newsgroups':
            return NLPDatasetLoader._load_20newsgroups()
        elif dataset_name == 'trec':
            return NLPDatasetLoader._load_trec()
        elif dataset_name == 'sst2':
            return NLPDatasetLoader._load_sst2()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    @staticmethod
    def _load_20newsgroups():
        """20 Newsgroups 데이터셋 로드"""
        try:
            dataset = load_dataset("SetFit/20_newsgroups", cache_dir=Config.CACHE_DIR_HF)
            
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
    def _load_trec():
        """TREC 데이터셋 로드"""
        try:
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
    def _load_sst2():
        """SST-2 데이터셋 로드"""
        try:
            dataset = load_dataset("sst2", cache_dir=Config.CACHE_DIR_HF)
            
            train_data = []
            train_labels = []
            test_data = []
            test_labels = []
            
            for item in dataset['train']:
                if item['sentence'] and item['label'] is not None:
                    train_data.append(item['sentence'])
                    train_labels.append(item['label'])
                    
            for item in dataset['validation']:  # SST-2는 'validation' 사용
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
    def load_outlier_exposure_dataset():
        """Outlier Exposure 데이터셋 로드 (WikiText-2)"""
        print("Loading Outlier Exposure dataset (WikiText-2)")
        try:
            dataset = load_dataset("wikitext", Config.WIKITEXT_VERSION, cache_dir=Config.CACHE_DIR_HF)
            
            # 텍스트 추출 및 전처리
            oe_texts = []
            for item in dataset['train']:
                if item['text'].strip():
                    # 문장으로 분할
                    sentences = sent_tokenize(item['text'])
                    for sent in sentences:
                        sent = sent.strip()
                        if len(sent) > 10:  # 너무 짧은 문장 제외
                            oe_texts.append(sent)
            
            # 샘플링
            if len(oe_texts) > Config.OE_SAMPLE_SIZE:
                oe_texts = random.sample(oe_texts, Config.OE_SAMPLE_SIZE)
            
            print(f"Loaded {len(oe_texts)} samples for Outlier Exposure")
            return {'text': oe_texts}
        except Exception as e:
            print(f"Error loading WikiText-2: {e}")
            return None
    
    @staticmethod
    def load_test_out_dataset():
        """Test-Out 데이터셋 로드 (WMT16)"""
        print(f"Loading Test-Out dataset ({Config.TEST_OUT_DATASET})")
        
        if Config.TEST_OUT_DATASET == 'wmt16':
            try:
                # WMT16 데이터셋 로드
                dataset = load_dataset("wmt16", "de-en", cache_dir=Config.CACHE_DIR_HF)
                
                # 영어 텍스트만 추출
                test_out_texts = []
                
                for item in dataset['test']:
                    if 'en' in item['translation'] and item['translation']['en'].strip():
                        test_out_texts.append(item['translation']['en'])
                
                # 샘플링
                if len(test_out_texts) > Config.TEST_OUT_SAMPLE_SIZE:
                    test_out_texts = random.sample(test_out_texts, Config.TEST_OUT_SAMPLE_SIZE)
                
                print(f"Loaded {len(test_out_texts)} samples for Test-Out evaluation")
                return {'text': test_out_texts}
            except Exception as e:
                print(f"Error loading WMT16: {e}")
                return None
        else:
            print(f"Unknown Test-Out dataset: {Config.TEST_OUT_DATASET}")
            return None

# === NLP용 토크나이저 ===
class NLPTokenizer:
    def __init__(self, vocab_size=10000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.vocab = {}
        self.inverse_vocab = {}
        self.word_counts = defaultdict(int)
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"
        self.cls_token = "<CLS>"
        self.sep_token = "<SEP>"
        
        # 특수 토큰 ID
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        
    def build_vocab(self, texts):
        """어휘 사전 구축"""
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
    
    def encode(self, text, max_length=512):
        """텍스트를 토큰 ID 시퀀스로 인코딩"""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # 텍스트 전처리
        text = preprocess_text_for_nlp(text)
        words = tokenize_nltk(text)
        
        # 토큰 ID로 변환
        token_ids = [self.cls_token_id]  # CLS 토큰으로 시작
        
        for word in words:
            if word in self.vocab:
                token_ids.append(self.vocab[word])
            else:
                token_ids.append(self.unk_token_id)
        
        # SEP 토큰 추가
        token_ids.append(self.sep_token_id)
        
        # max_length 적용 (패딩 또는 자르기)
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length-1] + [self.sep_token_id]
        else:
            # 패딩
            token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
        
        return token_ids
    
    def decode(self, token_ids):
        """토큰 ID 시퀀스를 텍스트로 디코딩"""
        words = []
        for token_id in token_ids:
            if token_id == self.pad_token_id:
                continue
            elif token_id in self.inverse_vocab:
                word = self.inverse_vocab[token_id]
                if word not in [self.cls_token, self.sep_token]:
                    words.append(word)
        return ' '.join(words)

# === Dataset 클래스들 ===
class NLPDataset(TorchDataset):
    """NLP Dataset 클래스 - GRU/LSTM 모델용"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels if labels is not None else [0] * len(texts)
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

class TransformerDataset(TorchDataset):
    """Transformer 모델용 Dataset 클래스"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels if labels is not None else [-1] * len(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Transformer 토크나이저 사용
        encoding = self.tokenizer(
            preprocess_text_for_transformer(text),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 배치 차원 제거
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding['label'] = torch.tensor(label, dtype=torch.long)
        
        return encoding

class UnlabeledNLPDataset(TorchDataset):
    """레이블이 없는 NLP Dataset 클래스 (OE 및 Test-Out용) - GRU/LSTM 모델용"""
    
    def __init__(self, texts, tokenizer, max_length=512, label=-1):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label = label  # OE 및 Test-Out은 모두 -1로 표시
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 토큰화
        token_ids = self.tokenizer.encode(text, max_length=self.max_length)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor([1 if id != self.tokenizer.pad_token_id else 0 for id in token_ids], dtype=torch.long),
            'label': torch.tensor(self.label, dtype=torch.long)
        }

class UnlabeledTransformerDataset(TorchDataset):
    """레이블이 없는 Transformer Dataset 클래스 (OE 및 Test-Out용)"""
    
    def __init__(self, texts, tokenizer, max_length=512, label=-1):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label = label  # OE 및 Test-Out은 모두 -1로 표시
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Transformer 토크나이저 사용
        encoding = self.tokenizer(
            preprocess_text_for_transformer(text),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 배치 차원 제거
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding['label'] = torch.tensor(self.label, dtype=torch.long)
        
        return encoding

# === NLP 모델들 ===
class NLPClassifier(nn.Module):
    """GRU/LSTM 기반 NLP 분류기"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, 
                 dropout=0.3, model_type="gru", attention=True):
        super(NLPClassifier, self).__init__()
        
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention = attention
        
        # 임베딩 레이어
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
        
        # 분류 레이어
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids, attention_mask=None, output_attentions=False, output_features=False):
        # 임베딩
        embedded = self.embedding(input_ids)
        
        # RNN
        if self.model_type == "gru":
            rnn_output, _ = self.rnn(embedded)
        elif self.model_type == "lstm":
            rnn_output, _ = self.rnn(embedded)
        
        # 어텐션 메커니즘
        if self.attention and attention_mask is not None:
            attention_weights = self.attention_layer(rnn_output).squeeze(-1)
            attention_weights = attention_weights.masked_fill(~attention_mask.bool(), float('-inf'))
            attention_weights = F.softmax(attention_weights, dim=1)
            weighted_output = torch.bmm(attention_weights.unsqueeze(1), rnn_output).squeeze(1)
        else:
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1)
                weighted_output = rnn_output[range(len(rnn_output)), lengths-1]
            else:
                weighted_output = rnn_output[:, -1]
        
        # 드롭아웃 적용
        features = self.dropout(weighted_output)
        
        # 분류
        logits = self.classifier(features)
        
        if output_features and output_attentions:
            if self.attention:
                return logits, features, attention_weights
            else:
                return logits, features, None
        elif output_features:
            return logits, features
        elif output_attentions:
            if self.attention:
                return logits, attention_weights
            else:
                return logits, None
        else:
            return logits

class RoBERTaClassifier(nn.Module):
    """RoBERTa 기반 분류기"""
    
    def __init__(self, model_name, num_classes, dropout=0.1):
        super(RoBERTaClassifier, self).__init__()
        self.model_name = model_name
        
        # RoBERTa 모델 로드
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, output_attentions=False, output_features=False):
        # Transformer 출력
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
            output_attentions=output_attentions,
            output_hidden_states=output_features
        )
        
        # [CLS] 토큰의 표현 사용
        pooled_output = outputs.last_hidden_state[:, 0, :]
        features = self.dropout(pooled_output)
        logits = self.classifier(features)
        
        if output_features and output_attentions:
            return logits, features, outputs.attentions[-1] if hasattr(outputs, 'attentions') else None
        elif output_features:
            return logits, features
        elif output_attentions:
            return logits, outputs.attentions[-1] if hasattr(outputs, 'attentions') else None
        else:
            return logits

# === EnhancedDataModule ===
class StructuredOEDataModule(pl.LightningDataModule):
    """구조화된 Outlier Exposure 실험용 DataModule"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['config'])
        
        # 모델 타입에 따라 토크나이저 초기화
        if config.NLP_MODEL_TYPE in ['gru', 'lstm']:
            self.tokenizer = NLPTokenizer(vocab_size=self.config.NLP_VOCAB_SIZE)
            self.is_transformer = False
        elif config.NLP_MODEL_TYPE == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME)
            self.is_transformer = True
        else:
            raise ValueError(f"Unknown model type: {config.NLP_MODEL_TYPE}")
        
        # 데이터셋 관련
        self.in_dist_data = None
        self.outlier_exposure_data = None
        self.test_out_data = None
        
        self.all_train_df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
        self.label2id = None
        self.id2label = None
        self.num_labels = None
        self.class_weights = None
    
    def prepare_data(self):
        """데이터셋 다운로드"""
        # 데이터셋 로드
        self.in_dist_data = NLPDatasetLoader.load_in_distribution_dataset(self.config.CURRENT_NLP_DATASET)
        self.outlier_exposure_data = NLPDatasetLoader.load_outlier_exposure_dataset()
        self.test_out_data = NLPDatasetLoader.load_test_out_dataset()
        
        if self.in_dist_data is None:
            raise ValueError(f"Failed to load In-Distribution dataset: {self.config.CURRENT_NLP_DATASET}")
        
        if self.outlier_exposure_data is None:
            print("Warning: Failed to load Outlier Exposure dataset. OE experiments will be skipped.")
        
        if self.test_out_data is None:
            print("Warning: Failed to load Test-Out dataset. Evaluation will be limited.")
    
    def setup(self, stage=None):
        """데이터셋 준비"""
        if self.in_dist_data is None:
            self.prepare_data()
        
        # 라벨 매핑 생성
        unique_labels = sorted(set(self.in_dist_data['train']['label']))
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.num_labels = len(unique_labels)
        
        print(f"In-Distribution dataset {self.config.CURRENT_NLP_DATASET}: {self.num_labels} classes")
        print(f"Label mapping: {self.label2id}")
        
        # 데이터프레임 생성
        self.all_train_df = pd.DataFrame({
            'text': self.in_dist_data['train']['text'],
            'label': [self.label2id[label] for label in self.in_dist_data['train']['label']],
            'split': 'train'
        })
        
        # 훈련/검증 분할
        train_idx, val_idx = train_test_split(
            range(len(self.all_train_df)), test_size=0.2, 
            random_state=self.config.RANDOM_STATE,
            stratify=self.all_train_df['label']
        )
        
        # 올바르게 분할 적용
        self.train_df = self.all_train_df.iloc[train_idx].reset_index(drop=True)
        self.val_df = self.all_train_df.iloc[val_idx].reset_index(drop=True)
        self.val_df['split'] = 'val'
        
        # 테스트 데이터
        self.test_df = pd.DataFrame({
            'text': self.in_dist_data['test']['text'],
            'label': [self.label2id[label] for label in self.in_dist_data['test']['label']],
            'split': 'test'
        })
        
        print(f"Data split - Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")
        
        # 클래스 가중치 계산
        if self.config.USE_WEIGHTED_LOSS:
            self._compute_class_weights()
        
        # 토크나이저가 NLPTokenizer인 경우에만 어휘 구축
        if not self.is_transformer:
            all_texts = self.train_df['text'].tolist()
            all_texts.extend(self.val_df['text'].tolist())
            all_texts.extend(self.test_df['text'].tolist())
            
            self.tokenizer.build_vocab(all_texts)
    
    def _compute_class_weights(self):
        """클래스 가중치 계산"""
        labels = self.train_df['label'].values
        unique_labels = np.unique(labels)
        
        try:
            class_weights_array = compute_class_weight('balanced', classes=unique_labels, y=labels)
            self.class_weights = torch.tensor(class_weights_array, dtype=torch.float)
            print(f"Computed class weights: {self.class_weights}")
        except ValueError as e:
            print(f"Error computing class weights: {e}. Using uniform weights.")
            self.config.USE_WEIGHTED_LOSS = False
            self.class_weights = None
    
    def train_dataloader(self):
        """훈련 데이터로더"""
        if self.is_transformer:
            # Transformer 모델용 데이터셋
            train_dataset = TransformerDataset(
                self.train_df['text'].tolist(),
                self.train_df['label'].tolist(),
                self.tokenizer,
                max_length=self.config.TRANSFORMER_MAX_LENGTH
            )
            
            batch_size = self.config.TRANSFORMER_BATCH_SIZE
        else:
            # GRU/LSTM 모델용 데이터셋
            train_dataset = NLPDataset(
                self.train_df['text'].tolist(),
                self.train_df['label'].tolist(),
                self.tokenizer,
                max_length=self.config.NLP_MAX_LENGTH
            )
            
            batch_size = self.config.NLP_BATCH_SIZE
        
        return DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """검증 데이터로더"""
        if self.is_transformer:
            # Transformer 모델용 데이터셋
            val_dataset = TransformerDataset(
                self.val_df['text'].tolist(),
                self.val_df['label'].tolist(),
                self.tokenizer,
                max_length=self.config.TRANSFORMER_MAX_LENGTH
            )
            
            batch_size = self.config.TRANSFORMER_BATCH_SIZE
        else:
            # GRU/LSTM 모델용 데이터셋
            val_dataset = NLPDataset(
                self.val_df['text'].tolist(),
                self.val_df['label'].tolist(),
                self.tokenizer,
                max_length=self.config.NLP_MAX_LENGTH
            )
            
            batch_size = self.config.NLP_BATCH_SIZE
        
        return DataLoader(
            val_dataset, 
            batch_size=batch_size,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """테스트 데이터로더"""
        if self.is_transformer:
            # Transformer 모델용 데이터셋
            test_dataset = TransformerDataset(
                self.test_df['text'].tolist(),
                self.test_df['label'].tolist(),
                self.tokenizer,
                max_length=self.config.TRANSFORMER_MAX_LENGTH
            )
            
            batch_size = self.config.TRANSFORMER_BATCH_SIZE
        else:
            # GRU/LSTM 모델용 데이터셋
            test_dataset = NLPDataset(
                self.test_df['text'].tolist(),
                self.test_df['label'].tolist(),
                self.tokenizer,
                max_length=self.config.NLP_MAX_LENGTH
            )
            
            batch_size = self.config.NLP_BATCH_SIZE
        
        return DataLoader(
            test_dataset, 
            batch_size=batch_size,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True
        )
    
    def get_outlier_exposure_dataloader(self):
        """Outlier Exposure 데이터로더"""
        if self.outlier_exposure_data is None:
            return None
        
        if self.is_transformer:
            # Transformer 모델용 데이터셋
            oe_dataset = UnlabeledTransformerDataset(
                self.outlier_exposure_data['text'],
                self.tokenizer,
                max_length=self.config.TRANSFORMER_MAX_LENGTH,
                label=-1  # OE 샘플은 -1로 표시
            )
            
            batch_size = self.config.TRANSFORMER_BATCH_SIZE
        else:
            # GRU/LSTM 모델용 데이터셋
            oe_dataset = UnlabeledNLPDataset(
                self.outlier_exposure_data['text'],
                self.tokenizer,
                max_length=self.config.NLP_MAX_LENGTH,
                label=-1  # OE 샘플은 -1로 표시
            )
            
            batch_size = self.config.NLP_BATCH_SIZE
        
        return DataLoader(
            oe_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True
        )
    
    def get_test_out_dataloader(self):
        """Test-Out 데이터로더"""
        if self.test_out_data is None:
            return None
        
        if self.is_transformer:
            # Transformer 모델용 데이터셋
            test_out_dataset = UnlabeledTransformerDataset(
                self.test_out_data['text'],
                self.tokenizer,
                max_length=self.config.TRANSFORMER_MAX_LENGTH,
                label=-1  # Test-Out 샘플은 -1로 표시
            )
            
            batch_size = self.config.TRANSFORMER_BATCH_SIZE
        else:
            # GRU/LSTM 모델용 데이터셋
            test_out_dataset = UnlabeledNLPDataset(
                self.test_out_data['text'],
                self.tokenizer,
                max_length=self.config.NLP_MAX_LENGTH,
                label=-1  # Test-Out 샘플은 -1로 표시
            )
            
            batch_size = self.config.NLP_BATCH_SIZE
        
        return DataLoader(
            test_out_dataset, 
            batch_size=batch_size,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True
        )

# === NLPModel (Lightning 모듈) ===
class NLPModel(pl.LightningModule):
    """NLP 모델 - Lightning 모듈"""
    
    def __init__(self, config, num_labels, label2id=None, id2label=None, class_weights=None, cm_interval=5):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['config', 'class_weights'])
        self.cm_interval = cm_interval  # Confusion matrix 계산 간격

        # 모델 초기화
        if config.NLP_MODEL_TYPE in ['gru', 'lstm']:
            # GRU/LSTM 모델 사용
            self.model = NLPClassifier(
                vocab_size=config.NLP_VOCAB_SIZE,
                embed_dim=config.NLP_EMBED_DIM,
                hidden_dim=config.NLP_HIDDEN_DIM,
                num_classes=num_labels,
                num_layers=config.NLP_NUM_LAYERS,
                dropout=config.NLP_DROPOUT,
                model_type=config.NLP_MODEL_TYPE,
                attention=True
            )
            self.is_transformer = False
        elif config.NLP_MODEL_TYPE == 'roberta':
            # RoBERTa 모델 사용
            self.model = RoBERTaClassifier(
                model_name=config.TRANSFORMER_MODEL_NAME,
                num_classes=num_labels,
                dropout=config.NLP_DROPOUT
            )
            self.is_transformer = True
        else:
            raise ValueError(f"Unknown model type: {config.NLP_MODEL_TYPE}")
        
        # 손실 함수
        if config.USE_WEIGHTED_LOSS and class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            print("Using weighted CrossEntropyLoss")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            print("Using standard CrossEntropyLoss")
        
        # 메트릭 정의
        metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=num_labels),
            'f1_weighted': torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='weighted'),
            'f1_macro': torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='macro')
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.val_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_labels)
    
    def forward(self, batch, output_attentions=False, output_features=False):
        # 필요한 입력 추출
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # 모델에 따라 다른 입력 처리
        if self.is_transformer:
            token_type_ids = batch.get('token_type_ids')
            
            if output_features:
                logits, features = self.model(
                    input_ids, 
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_features=True
                )
                return logits, features
            elif output_attentions:
                logits, attentions = self.model(
                    input_ids, 
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_attentions=True
                )
                return logits, attentions
            else:
                return self.model(
                    input_ids, 
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
        else:
            # GRU/LSTM 모델
            if output_features:
                return self.model(input_ids, attention_mask, output_features=True)
            elif output_attentions:
                return self.model(input_ids, attention_mask, output_attentions=True)
            else:
                return self.model(input_ids, attention_mask)
    
    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.loss_fn(logits, batch['label'])
        
        preds = torch.argmax(logits, dim=1)
        self.train_metrics.update(preds, batch['label'])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.loss_fn(logits, batch['label'])
        
        preds = torch.argmax(logits, dim=1)
        self.val_metrics.update(preds, batch['label'])
        self.val_cm.update(preds, batch['label'])
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.loss_fn(logits, batch['label'])
        
        preds = torch.argmax(logits, dim=1)
        self.test_metrics.update(preds, batch['label'])
        self.log('test_loss', loss, on_epoch=True)
        
        return loss
    
    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), on_epoch=True)
        self.train_metrics.reset()
    
    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), on_epoch=True)
        
        if self.current_epoch % self.cm_interval == 0:
            try:
                cm = self.val_cm.compute().cpu().numpy()
                cm_df = pd.DataFrame(cm, 
                                index=[f'True_{i}' for i in range(self.hparams.num_labels)],
                                columns=[f'Pred_{i}' for i in range(self.hparams.num_labels)])
                
                cm_path = os.path.join(self.config.CONFUSION_MATRIX_DIR, 
                                    f'cm_epoch_{self.current_epoch}.csv')
                cm_df.to_csv(cm_path)
                print(f"Confusion matrix saved to {cm_path}")
            except Exception as e:
                print(f"Error saving confusion matrix: {e}")
            
            self.val_metrics.reset()
            self.val_cm.reset()
    
    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), on_epoch=True)
        self.test_metrics.reset()
    
    def configure_optimizers(self):
        if self.is_transformer:
            # Transformer 모델용 옵티마이저
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.TRANSFORMER_LEARNING_RATE)
            
            if self.config.USE_LR_SCHEDULER:
                # 웜업이 있는 LinearScheduler 사용
                total_steps = self.trainer.estimated_stepping_batches
                warmup_steps = int(total_steps * self.config.OSR_WARMUP_RATIO)
                
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
                
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step"
                    }
                }
            else:
                return optimizer
        else:
            # GRU/LSTM 모델용 옵티마이저
            optimizer = optim.Adam(self.parameters(), lr=self.config.NLP_LEARNING_RATE)
            
            if self.config.USE_LR_SCHEDULER:
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
            else:
                return optimizer

# === OEModel (Outlier Exposure 모델) ===
class OEModel(pl.LightningModule):
    """Outlier Exposure 모델 - Lightning 모듈"""
    
    def __init__(self, config, num_labels, label2id=None, id2label=None, class_weights=None):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['config', 'class_weights'])
        
        # 모델 초기화
        if config.NLP_MODEL_TYPE in ['gru', 'lstm']:
            # GRU/LSTM 모델 사용
            self.model = NLPClassifier(
                vocab_size=config.NLP_VOCAB_SIZE,
                embed_dim=config.NLP_EMBED_DIM,
                hidden_dim=config.NLP_HIDDEN_DIM,
                num_classes=num_labels,
                num_layers=config.NLP_NUM_LAYERS,
                dropout=config.NLP_DROPOUT,
                model_type=config.NLP_MODEL_TYPE,
                attention=True
            )
            self.is_transformer = False
        elif config.NLP_MODEL_TYPE == 'roberta':
            # RoBERTa 모델 사용
            self.model = RoBERTaClassifier(
                model_name=config.TRANSFORMER_MODEL_NAME,
                num_classes=num_labels,
                dropout=config.NLP_DROPOUT
            )
            self.is_transformer = True
        else:
            raise ValueError(f"Unknown model type: {config.NLP_MODEL_TYPE}")
        
        # 손실 함수
        if config.USE_WEIGHTED_LOSS and class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            print("Using weighted CrossEntropyLoss")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            print("Using standard CrossEntropyLoss")
        
        # OE 람다
        self.oe_lambda = config.OSR_OE_LAMBDA
        
        # 메트릭 정의
        metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=num_labels),
            'f1_weighted': torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='weighted'),
            'f1_macro': torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='macro')
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
    
    def forward(self, batch, output_attentions=False, output_features=False):
        # 필요한 입력 추출
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # 모델에 따라 다른 입력 처리
        if self.is_transformer:
            token_type_ids = batch.get('token_type_ids')
            
            if output_features:
                logits, features = self.model(
                    input_ids, 
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_features=True
                )
                return logits, features
            elif output_attentions:
                logits, attentions = self.model(
                    input_ids, 
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_attentions=True
                )
                return logits, attentions
            else:
                return self.model(
                    input_ids, 
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
        else:
            # GRU/LSTM 모델
            if output_features:
                return self.model(input_ids, attention_mask, output_features=True)
            elif output_attentions:
                return self.model(input_ids, attention_mask, output_attentions=True)
            else:
                return self.model(input_ids, attention_mask)
    
    def training_step(self, batch, batch_idx):
        # ID 배치
        id_batch, oe_batch = batch
        
        # ID 손실
        id_logits = self.forward(id_batch)
        id_loss = self.loss_fn(id_logits, id_batch['label'])
        
        # OE 손실
        oe_logits = self.forward(oe_batch)
        log_softmax_oe = F.log_softmax(oe_logits, dim=1)
        uniform_target = torch.full_like(oe_logits, 1.0 / self.hparams.num_labels)
        oe_loss = F.kl_div(log_softmax_oe, uniform_target, reduction='batchmean', log_target=False)
        
        # 전체 손실
        total_loss = id_loss + self.oe_lambda * oe_loss
        
        # 측정
        id_preds = torch.argmax(id_logits, dim=1)
        self.train_metrics.update(id_preds, id_batch['label'])
        
        # 로깅
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_id_loss', id_loss, on_step=True, on_epoch=True)
        self.log('train_oe_loss', oe_loss, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.loss_fn(logits, batch['label'])
        
        preds = torch.argmax(logits, dim=1)
        self.val_metrics.update(preds, batch['label'])
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.loss_fn(logits, batch['label'])
        
        preds = torch.argmax(logits, dim=1)
        self.test_metrics.update(preds, batch['label'])
        self.log('test_loss', loss, on_epoch=True)
        
        return loss
    
    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), on_epoch=True)
        self.train_metrics.reset()
    
    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), on_epoch=True)
        self.val_metrics.reset()
    
    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), on_epoch=True)
        self.test_metrics.reset()
    
    def configure_optimizers(self):
        if self.is_transformer:
            # Transformer 모델용 옵티마이저
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.TRANSFORMER_LEARNING_RATE)
            
            if self.config.USE_LR_SCHEDULER:
                # 웜업이 있는 LinearScheduler 사용
                total_steps = self.trainer.estimated_stepping_batches
                warmup_steps = int(total_steps * self.config.OSR_WARMUP_RATIO)
                
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
                
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step"
                    }
                }
            else:
                return optimizer
        else:
            # GRU/LSTM 모델용 옵티마이저
            optimizer = optim.Adam(self.parameters(), lr=self.config.OSR_NLP_LEARNING_RATE)
            
            if self.config.USE_LR_SCHEDULER:
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
            else:
                return optimizer

# === OOD 평가 함수 ===
def evaluate_ood_detection(model, id_loader, ood_loader, device="cuda", temperature=1.0, score_type="msp"):
    """OOD 검출 성능 평가"""
    model.eval()
    
    id_scores = []
    ood_scores = []
    
    # ID 샘플에 대한 점수 계산
    with torch.no_grad():
        for batch in tqdm(id_loader, desc="Evaluating ID samples"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            
            if score_type == "msp":
                # Maximum Softmax Probability
                softmax_probs = F.softmax(outputs / temperature, dim=1)
                max_probs, _ = torch.max(softmax_probs, dim=1)
                id_scores.extend(max_probs.cpu().numpy())
            elif score_type == "entropy":
                # Entropy (높을수록 불확실)
                softmax_probs = F.softmax(outputs / temperature, dim=1)
                entropy_vals = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-10), dim=1)
                id_scores.extend((-entropy_vals).cpu().numpy())  # 낮을수록 OOD로 분류하기 위해 부호 변경
    
    # OOD 샘플에 대한 점수 계산
    with torch.no_grad():
        for batch in tqdm(ood_loader, desc="Evaluating OOD samples"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            
            if score_type == "msp":
                # Maximum Softmax Probability
                softmax_probs = F.softmax(outputs / temperature, dim=1)
                max_probs, _ = torch.max(softmax_probs, dim=1)
                ood_scores.extend(max_probs.cpu().numpy())
            elif score_type == "entropy":
                # Entropy (높을수록 불확실)
                softmax_probs = F.softmax(outputs / temperature, dim=1)
                entropy_vals = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-10), dim=1)
                ood_scores.extend((-entropy_vals).cpu().numpy())  # 낮을수록 OOD로 분류하기 위해 부호 변경
    
    # 결과 계산
    id_scores_np = np.array(id_scores)
    ood_scores_np = np.array(ood_scores)
    
    # AUROC 계산
    y_true = np.concatenate([np.ones_like(id_scores_np), np.zeros_like(ood_scores_np)])
    y_score = np.concatenate([id_scores_np, ood_scores_np])
    auroc = roc_auc_score(y_true, y_score)
    
    # FPR@95% TPR 계산
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_at_tpr_95 = fpr[idx_tpr_95]
    
    # AUPR 계산
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    aupr = auc(recall, precision)
    
    return {
        "AUROC": auroc * 100.0,  # 백분율로 변환
        "FPR@TPR95": fpr_at_tpr_95 * 100.0,  # 백분율로 변환
        "AUPR": aupr * 100.0,  # 백분율로 변환
        "id_scores": id_scores_np,
        "ood_scores": ood_scores_np
    }

# === 시각화 함수 ===
def plot_score_distributions(id_scores, ood_scores, title, save_path):
    """점수 분포 시각화"""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(id_scores, label='In-Distribution', shade=True)
    sns.kdeplot(ood_scores, label='Out-of-Distribution', shade=True)
    plt.title(title, fontsize=14)
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(id_scores, ood_scores, title, save_path):
    """ROC 곡선 시각화"""
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# === 메인 파이프라인 ===
class OutlierExposurePipeline:
    """Outlier Exposure 파이프라인"""
    
    def __init__(self, config):
        self.config = config
        self.datamodule = None
        self.base_model = None
        self.oe_model = None
        
        # 디렉터리 생성
        config.create_directories()
        config.save_config()
        
        # 시드 설정
        set_seed(config.RANDOM_STATE)
    
    def run(self):
        """전체 파이프라인 실행"""
        print(f"\n{'='*50}\nOutlier Exposure Pipeline - {self.config.CURRENT_NLP_DATASET}\n{'='*50}")
        
        # 데이터 모듈 초기화
        self.datamodule = StructuredOEDataModule(self.config)
        self.datamodule.prepare_data()
        self.datamodule.setup()
        
        # 기본 모델 훈련 (In-Distribution only)
        self._train_base_model()
        
        # OE 모델 훈련 (In-Distribution + Outlier Exposure)
        self._train_oe_model()
        
        # OOD 검출 평가
        self._evaluate_ood_detection()
    
    def _train_base_model(self):
        """기본 모델 훈련 (In-Distribution only)"""
        print(f"\n{'='*30}\nTraining Base Model\n{'='*30}")
        
        # 모델 초기화
        self.base_model = NLPModel(
            self.config,
            num_labels=self.datamodule.num_labels,
            label2id=self.datamodule.label2id,
            id2label=self.datamodule.id2label,
            class_weights=self.datamodule.class_weights
        )
        
        # 에포크 수 설정
        if self.config.NLP_MODEL_TYPE == 'roberta':
            num_epochs = self.config.TRANSFORMER_NUM_EPOCHS
        else:
            num_epochs = self.config.NLP_NUM_EPOCHS
        
        # 콜백 설정
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.MODEL_SAVE_DIR,
            filename='base-model-{epoch:02d}-{val_f1_macro:.4f}',
            save_top_k=1,
            monitor='val_f1_macro',
            mode='max'
        )
        
        early_stopping_callback = EarlyStopping(
            monitor='val_f1_macro',
            patience=self.config.OSR_EARLY_STOPPING_PATIENCE,
            mode='max',
            verbose=True
        )
        
        # 로거 설정
        logger = CSVLogger(save_dir=self.config.LOG_DIR, name="base_model")
        
        # 트레이너 초기화
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator=self.config.ACCELERATOR,
            devices=self.config.DEVICES,
            precision=self.config.PRECISION,
            callbacks=[checkpoint_callback, early_stopping_callback],
            logger=logger,
            log_every_n_steps=self.config.LOG_EVERY_N_STEPS
        )
        
        # 모델 훈련
        trainer.fit(self.base_model, self.datamodule)
        
        # 최고 모델 로드
        if checkpoint_callback.best_model_path:
            print(f"Loading best model from: {checkpoint_callback.best_model_path}")
            self.base_model = NLPModel.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                config=self.config,
                num_labels=self.datamodule.num_labels,
                label2id=self.datamodule.label2id,
                id2label=self.datamodule.id2label,
                class_weights=self.datamodule.class_weights
            )
        
        # 테스트 평가
        trainer.test(self.base_model, self.datamodule)
    
    def _train_oe_model(self):
        """OE 모델 훈련 (In-Distribution + Outlier Exposure)"""
        print(f"\n{'='*30}\nTraining Outlier Exposure Model\n{'='*30}")
        
        # Outlier Exposure 데이터로더 확인
        oe_dataloader = self.datamodule.get_outlier_exposure_dataloader()
        if oe_dataloader is None:
            print("No Outlier Exposure data available. Skipping OE model training.")
            return
        
        # 사전 훈련된 가중치 복사
        self.oe_model = OEModel(
            self.config,
            num_labels=self.datamodule.num_labels,
            label2id=self.datamodule.label2id,
            id2label=self.datamodule.id2label,
            class_weights=self.datamodule.class_weights
        )
        
        # 기본 모델의 가중치 복사
        if self.config.NLP_MODEL_TYPE in ['gru', 'lstm']:
            self.oe_model.model.load_state_dict(self.base_model.model.state_dict())
        elif self.config.NLP_MODEL_TYPE == 'roberta':
            # 트랜스포머 모델은 조금 다르게 처리
            for target_param, source_param in zip(self.oe_model.model.parameters(), self.base_model.model.parameters()):
                target_param.data.copy_(source_param.data)
        
        # 에포크 수 설정
        if self.config.NLP_MODEL_TYPE == 'roberta':
            num_epochs = self.config.TRANSFORMER_NUM_EPOCHS
        else:
            num_epochs = self.config.OSR_NLP_NUM_EPOCHS
        
        # 콜백 설정
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.MODEL_SAVE_DIR,
            filename='oe-model-{epoch:02d}-{val_f1_macro:.4f}',
            save_top_k=1,
            monitor='val_f1_macro',
            mode='max'
        )
        
        early_stopping_callback = EarlyStopping(
            monitor='val_f1_macro',
            patience=self.config.OSR_EARLY_STOPPING_PATIENCE,
            mode='max',
            verbose=True
        )
        
        # 로거 설정
        logger = CSVLogger(save_dir=self.config.LOG_DIR, name="oe_model")
        
        # OE 훈련을 위한 데이터로더 조합
        class CombinedDataLoader(pl.LightningDataModule):
            def __init__(self, id_dataloader, oe_dataloader):
                super().__init__()
                self.id_dataloader = id_dataloader
                self.oe_dataloader = oe_dataloader
            
            def train_dataloader(self):
                return self._zip_dataloaders(self.id_dataloader, self.oe_dataloader)
            
            def _zip_dataloaders(self, dataloader1, dataloader2):
                while True:
                    loader1_iter = iter(dataloader1)
                    loader2_iter = iter(dataloader2)
                    
                    try:
                        while True:
                            yield next(loader1_iter), next(loader2_iter)
                    except StopIteration:
                        pass
        
        combined_datamodule = CombinedDataLoader(
            self.datamodule.train_dataloader(), 
            oe_dataloader
        )
        
        # 트레이너 초기화
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator=self.config.ACCELERATOR,
            devices=self.config.DEVICES,
            precision=self.config.PRECISION,
            callbacks=[checkpoint_callback, early_stopping_callback],
            logger=logger,
            log_every_n_steps=self.config.LOG_EVERY_N_STEPS
        )
        
        # OE 모델 훈련
        trainer.fit(self.oe_model, combined_datamodule, val_dataloaders=self.datamodule.val_dataloader())
        
        # 최고 모델 로드
        if checkpoint_callback.best_model_path:
            print(f"Loading best OE model from: {checkpoint_callback.best_model_path}")
            self.oe_model = OEModel.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                config=self.config,
                num_labels=self.datamodule.num_labels,
                label2id=self.datamodule.label2id,
                id2label=self.datamodule.id2label,
                class_weights=self.datamodule.class_weights
            )
        
        # 테스트 평가
        trainer.test(self.oe_model, self.datamodule.test_dataloader())
    
    def _evaluate_ood_detection(self):
        """OOD 검출 평가"""
        print(f"\n{'='*30}\nEvaluating OOD Detection\n{'='*30}")
        
        # Test-Out 데이터로더 확인
        test_out_dataloader = self.datamodule.get_test_out_dataloader()
        if test_out_dataloader is None:
            print("No Test-Out data available. Skipping OOD detection evaluation.")
            return
        
        if self.base_model is None:
            print("Base model not available. Skipping OOD detection evaluation.")
            return
        
        # 평가 장치 확인
        device = next(self.base_model.parameters()).device
        
        # 출력 디렉터리 확인
        results_dir = self.config.OSR_RESULT_DIR
        os.makedirs(results_dir, exist_ok=True)
        
        # 기본 모델 평가
        print("\nEvaluating Base Model for OOD Detection...")
        base_results = evaluate_ood_detection(
            self.base_model, 
            self.datamodule.test_dataloader(), 
            test_out_dataloader,
            device=device,
            temperature=self.config.OSR_TEMPERATURE,
            score_type="msp"
        )
        
        # 결과 출력
        print(f"Base Model OOD Detection Results:")
        print(f"  AUROC: {base_results['AUROC']:.2f}%")
        print(f"  FPR@TPR95: {base_results['FPR@TPR95']:.2f}%")
        print(f"  AUPR: {base_results['AUPR']:.2f}%")
        
        # 시각화
        plot_score_distributions(
            base_results['id_scores'],
            base_results['ood_scores'],
            f"Score Distribution: Base Model vs {self.config.TEST_OUT_DATASET}",
            os.path.join(results_dir, f"base_model_score_dist.png")
        )
        
        plot_roc_curve(
            base_results['id_scores'],
            base_results['ood_scores'],
            f"ROC Curve: Base Model vs {self.config.TEST_OUT_DATASET}",
            os.path.join(results_dir, f"base_model_roc.png")
        )
        
        # OE 모델 평가
        if self.oe_model is not None:
            print("\nEvaluating OE Model for OOD Detection...")
            oe_results = evaluate_ood_detection(
                self.oe_model, 
                self.datamodule.test_dataloader(), 
                test_out_dataloader,
                device=device,
                temperature=self.config.OSR_TEMPERATURE,
                score_type="msp"
            )
            
            # 결과 출력
            print(f"OE Model OOD Detection Results:")
            print(f"  AUROC: {oe_results['AUROC']:.2f}%")
            print(f"  FPR@TPR95: {oe_results['FPR@TPR95']:.2f}%")
            print(f"  AUPR: {oe_results['AUPR']:.2f}%")
            
            # 시각화
            plot_score_distributions(
                oe_results['id_scores'],
                oe_results['ood_scores'],
                f"Score Distribution: OE Model vs {self.config.TEST_OUT_DATASET}",
                os.path.join(results_dir, f"oe_model_score_dist.png")
            )
            
            plot_roc_curve(
                oe_results['id_scores'],
                oe_results['ood_scores'],
                f"ROC Curve: OE Model vs {self.config.TEST_OUT_DATASET}",
                os.path.join(results_dir, f"oe_model_roc.png")
            )
            
            # 비교 결과
            improvement_auroc = oe_results['AUROC'] - base_results['AUROC']
            improvement_fpr = base_results['FPR@TPR95'] - oe_results['FPR@TPR95']
            improvement_aupr = oe_results['AUPR'] - base_results['AUPR']
            
            print(f"\nPerformance Improvement with Outlier Exposure:")
            print(f"  AUROC: +{improvement_auroc:.2f}%")
            print(f"  FPR@TPR95: -{improvement_fpr:.2f}%")
            print(f"  AUPR: +{improvement_aupr:.2f}%")
            
            # 결과 저장
            results = {
                'dataset': self.config.CURRENT_NLP_DATASET,
                'model_type': self.config.NLP_MODEL_TYPE,
                'oe_dataset': 'WikiText-2',
                'test_out_dataset': self.config.TEST_OUT_DATASET,
                'base_auroc': base_results['AUROC'],
                'base_fpr_tpr95': base_results['FPR@TPR95'],
                'base_aupr': base_results['AUPR'],
                'oe_auroc': oe_results['AUROC'],
                'oe_fpr_tpr95': oe_results['FPR@TPR95'],
                'oe_aupr': oe_results['AUPR'],
                'improvement_auroc': improvement_auroc,
                'improvement_fpr': improvement_fpr,
                'improvement_aupr': improvement_aupr
            }
            
            results_file = os.path.join(
                results_dir, 
                f"results_{self.config.CURRENT_NLP_DATASET}_{self.config.NLP_MODEL_TYPE}.json"
            )
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            print(f"Results saved to: {results_file}")

# === Main Function ===
def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Structured Outlier Exposure Experiments")
    
    # 데이터셋 선택
    parser.add_argument('--dataset', type=str, choices=['20newsgroups', 'trec', 'sst2'],
                        default='20newsgroups', help="In-distribution dataset")
    
    # 모델 설정
    parser.add_argument('--model_type', type=str, choices=['gru', 'lstm', 'roberta'],
                        default='gru', help="NLP model type")
    parser.add_argument('--epochs', type=int, default=None,
                        help="Number of epochs for training")
    parser.add_argument('--oe_epochs', type=int, default=None,
                        help="Number of epochs for OE training")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Batch size")
    
    # OE 설정
    parser.add_argument('--oe_lambda', type=float, default=1.0,
                        help="Weight for OE loss")
    parser.add_argument('--oe_samples', type=int, default=5000,
                        help="Number of OE samples to use")
    
    # 출력 설정
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Config 설정 업데이트
    Config.CURRENT_NLP_DATASET = args.dataset
    Config.NLP_MODEL_TYPE = args.model_type
    Config.OSR_NLP_MODEL_TYPE = args.model_type
    Config.OSR_OE_LAMBDA = args.oe_lambda
    Config.OE_SAMPLE_SIZE = args.oe_samples
    
    # 모델별 설정 업데이트
    if args.model_type == 'roberta':
        # RoBERTa 설정 적용
        if args.epochs is not None:
            Config.TRANSFORMER_NUM_EPOCHS = args.epochs
        if args.oe_epochs is not None:
            Config.OSR_NLP_NUM_EPOCHS = args.oe_epochs
        if args.batch_size is not None:
            Config.TRANSFORMER_BATCH_SIZE = args.batch_size
    else:
        # GRU/LSTM 설정 적용
        if args.epochs is not None:
            Config.NLP_NUM_EPOCHS = args.epochs
        if args.oe_epochs is not None:
            Config.OSR_NLP_NUM_EPOCHS = args.oe_epochs
        if args.batch_size is not None:
            Config.NLP_BATCH_SIZE = args.batch_size
    
    # 출력 디렉터리 설정
    if args.output_dir:
        Config.OUTPUT_DIR = args.output_dir
    else:
        # 기본 출력 디렉터리 생성
        model_name = Config.NLP_MODEL_TYPE
        Config.OUTPUT_DIR = f"oe_results_{args.dataset}_{model_name}"
    
    # 출력 디렉터리 경로 업데이트
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
    
    # 파이프라인 실행
    pipeline = OutlierExposurePipeline(Config)
    pipeline.run()

if __name__ == "__main__":
    main()