"""
향상된 통합 OE(Out-of-Distribution) Extractor와 NLP 데이터셋 지원
20 Newsgroups, TREC, SST, WikiText-2를 이용한 Outlier Exposure 비교 실험
OOD 탐지 결과:
데이터셋            메트릭              기본모델       OE적용           개선
------------------------------------------------------------
SNLI            FPR90           59.08       6.89        52.19
SNLI            AUROC           70.76      96.47        25.71
SNLI            AUPR            78.82      98.17        19.35
IMDB            FPR90           48.83      51.61        -2.78
IMDB            AUROC           75.00      72.04        -2.96
IMDB            AUPR            80.70      78.05        -2.65
WMT16           FPR90           56.52      16.13        40.39
WMT16           AUROC           70.88      93.62        22.74
WMT16           AUPR            59.58      92.37        32.79
Yelp            FPR90           52.54      46.13         6.41
Yelp            AUROC           74.46      78.07         3.61
Yelp            AUPR            80.96      84.04         3.08
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

# --- 설정 클래스 ---
class Config:
    """향상된 설정 클래스 - Syslog 및 NLP 데이터셋 지원"""
    
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
    NLP_MODEL_TYPE = "roberta-base"  # "gru", "lstm", 또는 "transformer"
    NLP_VOCAB_SIZE = 10000
    NLP_EMBED_DIM = 300
    NLP_HIDDEN_DIM = 512
    NLP_NUM_LAYERS = 2
    NLP_DROPOUT = 0.3
    NLP_MAX_LENGTH = 512
    NLP_BATCH_SIZE = 256
    NLP_NUM_EPOCHS = 20
    NLP_LEARNING_RATE = 1e-3
    
    # === OE 특정 설정 ===
    NLP_OE_EPOCHS = 5    # OE 미세조정을 위한 에폭 수
    OE_LAMBDA = 1.0      # OE 손실 가중치
    
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
        'attention_entropy': {'percentile': 75, 'mode': 'higher'},      # 상위 25%
        'max_attention': {'percentile': 15, 'mode': 'lower'},          # 하위 15%
        'removed_avg_attention': {'percentile': 85, 'mode': 'higher'}, # 상위 15%
        'top_k_avg_attention': {'percentile': 25, 'mode': 'lower'}     # 하위 25%
    }

    # 순차 필터링 (가장 효과적인 순서로)
    FILTERING_SEQUENCE = [
        ('removed_avg_attention', {'percentile': 85, 'mode': 'higher'}),  # 1단계: 가장 선별력 높음
        ('attention_entropy', {'percentile': 75, 'mode': 'higher'}),       # 2단계: 엔트로피 필터링
        ('max_attention', {'percentile': 15, 'mode': 'lower'})             # 3단계: 최종 정제
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
    OSR_NLP_BATCH_SIZE = 128
    OSR_NLP_NUM_EPOCHS = 20
    OSR_NLP_LEARNING_RATE = 1e-3
    
    # 기존 Vision OSR 설정
    OSR_MODEL_TYPE = 'roberta-base'
    OSR_MAX_LENGTH = 128
    OSR_BATCH_SIZE = 128
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
    """NLTK를 사용한 토큰화 - 다운로드 방지 버전"""
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
        # NLTK가 실패하면 간단한 분할 사용
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
    def load_wikitext2_for_oe():
        """WikiText-2 데이터셋 (Outlier Exposure용)"""
        print("Loading WikiText-2 for Outlier Exposure...")
        try:
            # HuggingFace datasets 사용
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=Config.CACHE_DIR_HF)
            
            # 텍스트 추출 및 전처리
            oe_texts = []
            for item in dataset['train']:
                if item['text'].strip():
                    # 문장으로 분할
                    sentences = sent_tokenize(item['text'])
                    for sent in sentences:
                        sent = sent.strip()
                        if len(sent) > 10 and len(sent) < 500:  # 적절한 길이의 문장만 사용
                            oe_texts.append(sent)
            
            # 크기 제한 (선택적)
            max_oe_samples = 50000  # 필요에 따라 조정
            if len(oe_texts) > max_oe_samples:
                random.seed(Config.RANDOM_STATE)
                oe_texts = random.sample(oe_texts, max_oe_samples)
            
            # OOD 라벨 생성 (-1로 표시)
            oe_labels = [-1] * len(oe_texts)
            
            print(f"준비된 WikiText-2 OE 문장 수: {len(oe_texts)}")
            return {'text': oe_texts, 'label': oe_labels}
        except Exception as e:
            print(f"Error loading WikiText-2: {e}")
            return None
    
    @staticmethod
    def load_test_ood_datasets():
        """테스트용 OOD 데이터셋 로드"""
        ood_datasets = {}
        
        # 1. SNLI
        try:
            print("SNLI 테스트 OOD 데이터셋 로드 중...")
            dataset = load_dataset("snli", cache_dir=Config.CACHE_DIR_HF)
            texts = [item['hypothesis'] for item in dataset['test'] if item['hypothesis']]
            texts = texts[:5000]  # 크기 제한
            ood_datasets['SNLI'] = {'text': texts, 'label': [-1] * len(texts)}
        except Exception as e:
            print(f"SNLI 로드 오류: {e}")
        
        # 2. IMDB
        try:
            print("IMDB 테스트 OOD 데이터셋 로드 중...")
            dataset = load_dataset("imdb", cache_dir=Config.CACHE_DIR_HF)
            texts = [item['text'] for item in dataset['test'] if item['text']]
            texts = texts[:5000]  # 크기 제한
            ood_datasets['IMDB'] = {'text': texts, 'label': [-1] * len(texts)}
        except Exception as e:
            print(f"IMDB 로드 오류: {e}")
        
        # 3. WMT16 (영어 부분)
        try:
            print("WMT16 테스트 OOD 데이터셋 로드 중...")
            dataset = load_dataset("wmt16", "ro-en", cache_dir=Config.CACHE_DIR_HF)
            texts = [item['translation']['en'] for item in dataset['test'] if 'translation' in item]
            texts = texts[:5000]  # 크기 제한
            ood_datasets['WMT16'] = {'text': texts, 'label': [-1] * len(texts)}
        except Exception as e:
            print(f"WMT16 로드 오류: {e}")
        
        # 4. Multi30K (영어 설명)
        try:
            print("Multi30K 테스트 OOD 데이터셋 로드 중...")
            dataset = load_dataset("multi30k", "de-en", cache_dir=Config.CACHE_DIR_HF)
            texts = [item['en'] for item in dataset['test'] if 'en' in item]
            ood_datasets['Multi30K'] = {'text': texts, 'label': [-1] * len(texts)}
        except Exception as e:
            print(f"Multi30K 로드 오류: {e}")
        
        # 5. Yelp
        try:
            print("Yelp 테스트 OOD 데이터셋 로드 중...")
            dataset = load_dataset("yelp_review_full", cache_dir=Config.CACHE_DIR_HF)
            texts = [item['text'] for item in dataset['test'] if 'text' in item]
            texts = texts[:5000]  # 크기 제한
            ood_datasets['Yelp'] = {'text': texts, 'label': [-1] * len(texts)}
        except Exception as e:
            print(f"Yelp 로드 오류: {e}")
            
        # 더 많은 OOD 데이터셋을 추가할 수 있습니다
        
        return ood_datasets

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
        
        print(f"어휘 사전 구축 완료: {len(self.vocab)} 단어")
        print(f"총 단어 유형 수: {len(self.word_counts)}")
        print(f"빈도 필터링으로 제외된 단어 수: {len(self.word_counts) - len(self.vocab) + 4}")
    
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
            while len(token_ids) < max_length:
                token_ids.append(self.pad_token_id)
        
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

# === OE 실험을 위한 데이터셋 클래스 ===
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

# OE 배치 데이터셋 클래스 (ID와 OE 샘플을 함께 로드)
class OEBatchDataset(TorchDataset):
    """Outlier Exposure용 배치 데이터셋 - ID와 OE 샘플을 함께 처리"""
    
    def __init__(self, id_dataset, oe_dataset, oe_ratio=0.5):
        self.id_dataset = id_dataset
        self.oe_dataset = oe_dataset
        self.oe_ratio = oe_ratio
        
        # 더 작은 데이터셋에 맞춰 샘플링
        self.id_indices = list(range(len(id_dataset)))
        self.oe_indices = list(range(len(oe_dataset)))
        
        # oe_ratio에 따라 배치당 OE 샘플 수 결정
        self.length = len(id_dataset)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # ID 샘플 가져오기
        id_sample = self.id_dataset[self.id_indices[idx % len(self.id_indices)]]
        
        # OE 샘플 가져오기 (랜덤 선택)
        oe_idx = random.choice(self.oe_indices)
        oe_sample = self.oe_dataset[oe_idx]
        
        return {
            'id': id_sample,
            'oe': oe_sample
        }

# OE 배치 생성기
class OEBatchSampler:
    """Outlier Exposure용 배치 샘플러"""
    
    def __init__(self, id_dataset_size, oe_dataset_size, batch_size, oe_ratio=0.5, shuffle=True):
        self.id_dataset_size = id_dataset_size
        self.oe_dataset_size = oe_dataset_size
        self.batch_size = batch_size
        self.oe_ratio = oe_ratio
        self.shuffle = shuffle
        
        # 배치당 ID와 OE 샘플 수 계산
        self.id_per_batch = max(1, int(batch_size * (1 - oe_ratio)))
        self.oe_per_batch = max(1, batch_size - self.id_per_batch)
        
        # 전체 배치 수 계산
        self.num_batches = id_dataset_size // self.id_per_batch
        
    def __iter__(self):
        # ID 및 OE 인덱스 생성
        id_indices = list(range(self.id_dataset_size))
        oe_indices = list(range(self.oe_dataset_size))
        
        if self.shuffle:
            random.shuffle(id_indices)
            random.shuffle(oe_indices)
        
        # 각 배치에 대한 ID 및 OE 인덱스 생성
        for i in range(self.num_batches):
            batch_id_indices = id_indices[i * self.id_per_batch:(i + 1) * self.id_per_batch]
            
            # OE 인덱스 선택 (필요한 경우 반복)
            batch_oe_indices = []
            for j in range(self.oe_per_batch):
                oe_idx = oe_indices[j % self.oe_dataset_size]
                batch_oe_indices.append(self.id_dataset_size + oe_idx)  # OE 인덱스는 ID 크기 이후부터 시작
            
            yield batch_id_indices + batch_oe_indices
    
    def __len__(self):
        return self.num_batches

# OE 콜레이터 (ID와 OE 샘플을 배치에 함께 포함)
class OECollator:
    """ID와 OE 샘플을 함께 처리하는 콜레이터"""
    
    def __call__(self, batch):
        id_batch = {
            'input_ids': torch.stack([item['id']['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['id']['attention_mask'] for item in batch]),
            'label': torch.stack([item['id']['label'] for item in batch])
        }
        
        oe_batch = {
            'input_ids': torch.stack([item['oe']['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['oe']['attention_mask'] for item in batch]),
            'label': torch.stack([item['oe']['label'] for item in batch])
        }
        
        return {'id': id_batch, 'oe': oe_batch}

# === Enhanced DataModule ===
class EnhancedDataModule(pl.LightningDataModule):
    """향상된 데이터 모듈 - Outlier Exposure 지원"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['config'])
        
        # 모드에 따른 초기화
        if config.EXPERIMENT_MODE == "nlp":
            self._init_nlp_mode()
        else:
            self._init_syslog_mode()
        
        # 데이터 컨테이너 초기화
        self.in_distribution_data = None
        self.oe_data = None
        self.test_ood_datasets = None
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
        """데이터 준비 단계"""
        if self.config.EXPERIMENT_MODE == "nlp":
            # In-distribution 데이터 로드
            self._prepare_in_distribution_data()
            
            # Outlier Exposure 데이터 로드
            if self.config.STAGE_OE_EXTRACTION:
                self._prepare_oe_data()
            
            # 테스트 OOD 데이터셋 로드 
            if self.config.STAGE_OSR_EXPERIMENTS:
                self._prepare_test_ood_data()
    
    def _prepare_in_distribution_data(self):
        """In-distribution NLP 데이터 준비"""
        dataset_name = self.config.CURRENT_NLP_DATASET
        
        if dataset_name == '20newsgroups':
            self.in_distribution_data = NLPDatasetLoader.load_20newsgroups()
        elif dataset_name == 'trec':
            self.in_distribution_data = NLPDatasetLoader.load_trec()
        elif dataset_name == 'sst2':
            self.in_distribution_data = NLPDatasetLoader.load_sst2()
        else:
            raise ValueError(f"알 수 없는 NLP 데이터셋: {dataset_name}")
        
        if self.in_distribution_data is None:
            raise ValueError(f"데이터셋 로드 실패: {dataset_name}")
        
        print(f"{dataset_name} in-distribution 데이터셋 로드 완료")
    
    def _prepare_oe_data(self):
        """Outlier Exposure 데이터 준비 (WikiText-2)"""
        self.oe_data = NLPDatasetLoader.load_wikitext2_for_oe()
        if self.oe_data is None:
            print("경고: Outlier Exposure용 WikiText-2 로드 실패")
    
    def _prepare_test_ood_data(self):
        """테스트 OOD 데이터셋 준비"""
        self.test_ood_datasets = NLPDatasetLoader.load_test_ood_datasets()
        if not self.test_ood_datasets:
            print("경고: 테스트 OOD 데이터셋 로드 실패")
    
    def setup(self, stage=None):
        """학습/검증/테스트 데이터셋 설정"""
        if self.in_distribution_data is None:
            self._prepare_in_distribution_data()
        
        # 데이터프레임으로 변환
        train_df = pd.DataFrame(self.in_distribution_data['train'])
        test_df = pd.DataFrame(self.in_distribution_data['test'])
        
        # 분할 표시자 추가
        train_df['split'] = 'train'
        test_df['split'] = 'test'
        
        # 전체 데이터셋 생성
        self.df_full = pd.concat([train_df, test_df], ignore_index=True)
        
        # 라벨 매핑 설정
        unique_labels = sorted(self.df_full['label'].unique())
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.num_labels = len(unique_labels)
        
        print(f"NLP 라벨 매핑: {self.label2id}")
        
        # 라벨을 ID로 변환
        self.df_full['label_id'] = self.df_full['label'].map(self.label2id)
        
        # 훈련 데이터를 train/val로 분할
        train_data = self.df_full[self.df_full['split'] == 'train'].copy()
        
        # 가중치 손실을 위한 클래스 가중치
        if self.config.USE_WEIGHTED_LOSS:
            self._compute_class_weights(train_data)
        
        # train/val 분할 생성
        self.train_df_final, self.val_df_final = train_test_split(
            train_data, test_size=0.2, random_state=self.config.RANDOM_STATE,
            stratify=train_data['label_id']
        )
        
        print(f"NLP 분할 - Train: {len(self.train_df_final)}, Val: {len(self.val_df_final)}")
        
        # 어휘 사전 구축
        all_texts = self.df_full['text'].tolist()
        self.tokenizer.build_vocab(all_texts)
        
        # DataLoader용 데이터셋 저장
        self.tokenized_train_val_datasets = {
            'train': self.train_df_final,
            'validation': self.val_df_final
        }
    
    def _compute_class_weights(self, train_df):
        """가중치 손실을 위한 클래스 가중치 계산"""
        labels_for_weights = train_df['label_id'].values
        unique_labels = np.unique(labels_for_weights)
        
        try:
            class_weights_array = compute_class_weight('balanced', classes=unique_labels, y=labels_for_weights)
            self.class_weights = torch.ones(self.num_labels)
            
            for i, label_idx in enumerate(unique_labels):
                self.class_weights[label_idx] = class_weights_array[i]
            
            print(f"NLP 클래스 가중치 계산: {self.class_weights}")
        except ValueError as e:
            print(f"클래스 가중치 계산 오류: {e}. 균일 가중치 사용.")
            self.config.USE_WEIGHTED_LOSS = False
            self.class_weights = None
    
    def train_dataloader(self):
        """훈련용 DataLoader - OE 데이터를 포함"""
        # In-distribution 데이터셋 생성
        id_dataset = NLPDataset(
            self.train_df_final['text'].tolist(),
            self.train_df_final['label_id'].tolist(),
            self.tokenizer,
            max_length=self.config.NLP_MAX_LENGTH
        )
        
        # OE를 사용하는 경우, 결합된 데이터셋 생성
        if self.config.STAGE_OE_EXTRACTION and self.oe_data is not None:
            # OE 데이터셋 생성
            oe_dataset = NLPDataset(
                self.oe_data['text'],
                self.oe_data['label'],  # -1값
                self.tokenizer,
                max_length=self.config.NLP_MAX_LENGTH
            )
            
            # ID와 OE 데이터를 결합한 데이터셋 생성
            combined_dataset = OEBatchDataset(id_dataset, oe_dataset)
            
            return DataLoader(
                combined_dataset,
                batch_size=self.config.NLP_BATCH_SIZE,
                shuffle=True,
                num_workers=self.config.NUM_WORKERS,
                collate_fn=OECollator(),
                pin_memory=True,
                persistent_workers=self.config.NUM_WORKERS > 0
            )
        else:
            # ID 데이터만 사용하는 DataLoader 반환
            return DataLoader(
                id_dataset,
                batch_size=self.config.NLP_BATCH_SIZE,
                shuffle=True,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=True,
                persistent_workers=self.config.NUM_WORKERS > 0
            )
    
    def val_dataloader(self):
        """검증용 DataLoader (In-distribution 데이터만)"""
        dataset = NLPDataset(
            self.val_df_final['text'].tolist(),
            self.val_df_final['label_id'].tolist(),
            self.tokenizer,
            max_length=self.config.NLP_MAX_LENGTH
        )
        return DataLoader(
            dataset,
            batch_size=self.config.NLP_BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=self.config.NUM_WORKERS > 0
        )
    
    def get_ood_dataloaders(self):
        """OOD 테스트 데이터셋용 DataLoader"""
        if not self.test_ood_datasets:
            return {}
        
        ood_loaders = {}
        for name, dataset_dict in self.test_ood_datasets.items():
            dataset = NLPDataset(
                dataset_dict['text'],
                dataset_dict['label'],  # 모두 -1
                self.tokenizer,
                max_length=self.config.NLP_MAX_LENGTH
            )
            ood_loaders[name] = DataLoader(
                dataset,
                batch_size=self.config.NLP_BATCH_SIZE,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=True,
                persistent_workers=self.config.NUM_WORKERS > 0
            )
        
        return ood_loaders

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

# === Enhanced Model ===
class EnhancedModel(pl.LightningModule):
    """향상된 모델 - Outlier Exposure 지원"""
    
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
        # 배치가 입력 ID와 어텐션 마스크를 포함하는 딕셔너리인 경우 처리
        if isinstance(batch, dict) and 'input_ids' in batch and 'attention_mask' in batch:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
        # 배치가 텐서인 경우
        elif isinstance(batch, tuple) or isinstance(batch, list):
            input_ids, attention_mask = batch[0], batch[1]
        else:
            # 그 외의 경우 에러
            raise ValueError("Invalid batch format. Expected dictionary with 'input_ids' and 'attention_mask' or tuple/list.")
            
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
    
    def training_step(self, batch, batch_idx):
        # 배치가 ID와 OE 데이터를 포함하는지 확인 (OE 훈련 여부)
        if isinstance(batch, dict) and 'id' in batch and 'oe' in batch:
            # ID와 OE 샘플을 모두 포함하는 배치
            id_batch = batch['id']
            oe_batch = batch['oe']
            
            # ID 샘플에 대한 포워드 패스
            id_outputs = self.forward(id_batch)
            id_logits = id_outputs.logits
            id_loss = self.loss_fn(id_logits, id_batch['label'])
            
            # OE 샘플에 대한 포워드 패스
            oe_outputs = self.forward(oe_batch)
            oe_logits = oe_outputs.logits
            
            # OE 손실: OE 샘플에 대해 균일 분포(높은 엔트로피) 장려
            # 균일 타겟에 대한 교차 엔트로피로 구현
            num_classes = oe_logits.size(1)
            uniform_target = torch.ones_like(oe_logits) / num_classes
            oe_loss = F.kl_div(
                F.log_softmax(oe_logits, dim=1),
                uniform_target,
                reduction='batchmean'
            )
            
            # 손실 결합
            loss = id_loss + self.config_params.OE_LAMBDA * oe_loss
            
            # 메트릭 기록
            self.log('train_id_loss', id_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_oe_loss', oe_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            
            # 메트릭 업데이트
            id_preds = torch.argmax(id_logits, dim=1)
            self.log_dict(self.train_metrics(id_preds, id_batch['label']), on_step=False, on_epoch=True, prog_bar=True)
            
            return loss
        else:
            # 기본 ID 훈련 (OE 없음)
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
            self.log_dict(self.train_metrics(preds, batch['labels']), on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
    
    def validation_step(self, batch, batch_idx):
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
        self.val_metrics.update(preds, batch['labels'])
        self.val_cm.update(preds, batch['labels'])
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
        else:
            # Syslog 모드 옵티마이저 설정
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.config_params.LEARNING_RATE)
            
            # 학습률 스케줄러 설정
            if self.config_params.USE_LR_SCHEDULER:
                total_steps = self.trainer.estimated_stepping_batches
                warmup_steps = int(total_steps * 0.1)
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
                )
                return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
            else:
                return optimizer

# === OOD 탐지 평가 함수 ===
def evaluate_ood_detection(model, data_module):
    """OOD 탐지 성능 평가"""
    model.eval()
    results = {}
    
    # ID 검증 데이터 가져오기
    val_loader = data_module.val_dataloader()
    
    # OOD 테스트 DataLoader 가져오기
    ood_loaders = data_module.get_ood_dataloaders()
    
    # 각 OOD 데이터셋에 대해
    for ood_name, ood_loader in ood_loaders.items():
        # OOD 탐지 메트릭 계산 (AUROC, AUPR, FPR90)
        id_scores = []
        ood_scores = []
        
        # ID 점수 계산
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items()}
                outputs = model(inputs)
                logits = outputs.logits
                # 최대 소프트맥스 확률을 OOD 점수로 사용
                probs = F.softmax(logits, dim=1)
                scores = 1.0 - torch.max(probs, dim=1)[0]  # OOD 점수는 1-MSP
                id_scores.extend(scores.cpu().numpy())
        
        # OOD 점수 계산
        with torch.no_grad():
            for batch in ood_loader:
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items()}
                outputs = model(inputs)
                logits = outputs.logits
                # 최대 소프트맥스 확률을 OOD 점수로 사용
                probs = F.softmax(logits, dim=1)
                scores = 1.0 - torch.max(probs, dim=1)[0]  # OOD 점수는 1-MSP
                ood_scores.extend(scores.cpu().numpy())
        
        # 메트릭 계산
        # AUROC 계산
        auroc = roc_auc_score(
            [0] * len(id_scores) + [1] * len(ood_scores),
            id_scores + ood_scores
        )
        
        # AUPR 계산
        precision, recall, _ = precision_recall_curve(
            [0] * len(id_scores) + [1] * len(ood_scores),
            id_scores + ood_scores
        )
        aupr = auc(recall, precision)
        
        # FPR90 계산 (90% True Positive Rate에서의 False Positive Rate)
        fpr, tpr, thresholds = roc_curve(
            [0] * len(id_scores) + [1] * len(ood_scores),
            id_scores + ood_scores
        )
        idx = np.argmin(np.abs(tpr - 0.9))
        fpr90 = fpr[idx]
        
        results[ood_name] = {
            'AUROC': auroc * 100,  # 백분율로 변환
            'AUPR': aupr * 100,    # 백분율로 변환
            'FPR90': fpr90 * 100   # 백분율로 변환
        }
    
    return results

# === NLP Outlier Exposure 실험 실행 함수 ===
def run_nlp_oe_experiment(config):
    """NLP Outlier Exposure 실험 실행"""
    print(f"\n{'='*50}\nNLP OUTLIER EXPOSURE 실험\n{'='*50}")
    print(f"In-distribution 데이터셋: {config.CURRENT_NLP_DATASET}")
    
    # 재현성을 위한 시드 설정
    set_seed(config.RANDOM_STATE)
    
    # 데이터 모듈 초기화
    data_module = EnhancedDataModule(config)
    data_module.prepare_data()
    data_module.setup()
    
    # OE 없는 기본 모델 초기화
    print("Outlier Exposure 없는 기본 모델 훈련...")
    # OE 단계 비활성화
    config.STAGE_OE_EXTRACTION = False
    
    baseline_model = EnhancedModel(
        config=config,
        num_labels=data_module.num_labels,
        label2id=data_module.label2id,
        id2label=data_module.id2label,
        class_weights=data_module.class_weights,
        tokenizer=data_module.tokenizer
    )
    
    # 기본 모델 훈련
    baseline_trainer = pl.Trainer(
        max_epochs=config.NLP_NUM_EPOCHS,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        precision=config.PRECISION,
        deterministic=False,
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
        gradient_clip_val=config.GRADIENT_CLIP_VAL
    )
    baseline_trainer.fit(baseline_model, datamodule=data_module)
    
    # OOD 탐지에 대한 기본 모델 평가
    baseline_results = evaluate_ood_detection(baseline_model, data_module)
    
    # Outlier Exposure로 훈련
    print("\nOutlier Exposure로 모델 훈련...")
    # OE 단계 활성화
    config.STAGE_OE_EXTRACTION = True
    
    # OE 데이터 준비
    data_module_oe = EnhancedDataModule(config)
    data_module_oe.prepare_data()
    data_module_oe.setup()
    
    # 기본 모델에서 가중치 복사하고 OE 활성화
    oe_model = EnhancedModel(
        config=config,
        num_labels=data_module_oe.num_labels,
        label2id=data_module_oe.label2id,
        id2label=data_module_oe.id2label,
        class_weights=data_module_oe.class_weights,
        tokenizer=data_module_oe.tokenizer
    )
    oe_model.load_state_dict(baseline_model.state_dict())
    
    # OE로 미세조정
    oe_trainer = pl.Trainer(
        max_epochs=config.NLP_OE_EPOCHS,  # 미세조정엔 더 적은 에폭
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        precision=config.PRECISION,
        deterministic=False,
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
        gradient_clip_val=config.GRADIENT_CLIP_VAL
    )
    oe_trainer.fit(oe_model, datamodule=data_module_oe)
    
    # OE 모델의 OOD 탐지 평가
    oe_results = evaluate_ood_detection(oe_model, data_module_oe)
    
    # 결과 비교
    print("\nOOD 탐지 결과:")
    print(f"{'데이터셋':<15} {'메트릭':<10} {'기본모델':>10} {'OE적용':>10} {'개선':>12}")
    print("-" * 60)
    
    for dataset_name in oe_results:
        for metric in ['FPR90', 'AUROC', 'AUPR']:
            baseline_value = baseline_results[dataset_name][metric]
            oe_value = oe_results[dataset_name][metric]
            improvement = oe_value - baseline_value if metric in ['AUROC', 'AUPR'] else baseline_value - oe_value
            print(f"{dataset_name:<15} {metric:<10} {baseline_value:>10.2f} {oe_value:>10.2f} {improvement:>12.2f}")
    
    return baseline_results, oe_results

# === 메인 함수 ===
def main():
    parser = argparse.ArgumentParser(description="향상된 NLP Outlier Exposure 실험")
    
    # 기본 모드 선택
    parser.add_argument('--mode', type=str, choices=['syslog', 'nlp'], default='nlp', 
                       help="실험 모드: 'syslog' 또는 'nlp'")
    
    # NLP 관련 인자들
    parser.add_argument('--nlp_dataset', type=str, choices=['20newsgroups', 'trec', 'sst2'], default='20newsgroups',
                       help="실험에 사용할 NLP 데이터셋")
    parser.add_argument('--nlp_model_type', type=str, choices=['gru', 'lstm'], default='gru',
                       help="NLP 모델 유형")
    parser.add_argument('--nlp_epochs', type=int, default=30, help="NLP 훈련 에폭 수")
    parser.add_argument('--oe_epochs', type=int, default=5, help="OE 미세조정 에폭 수")
    parser.add_argument('--oe_lambda', type=float, default=1.0, help="OE 손실 가중치")
    
    # 출력 디렉토리
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR, help="결과 저장 디렉토리")
    
    # 단계 제어
    parser.add_argument('--skip_base_training', action='store_true', help="기본 모델 훈련 스킵")
    parser.add_argument('--skip_oe_extraction', action='store_true', help="OE 추출 단계 스킵")
    
    args = parser.parse_args()
    
    # Config 업데이트
    Config.EXPERIMENT_MODE = args.mode
    
    # NLP 설정 업데이트
    if args.mode == 'nlp':
        Config.CURRENT_NLP_DATASET = args.nlp_dataset
        Config.NLP_MODEL_TYPE = args.nlp_model_type
        Config.NLP_NUM_EPOCHS = args.nlp_epochs
        Config.NLP_OE_EPOCHS = args.oe_epochs
        Config.OE_LAMBDA = args.oe_lambda
    
    # 출력 디렉토리 설정
    Config.OUTPUT_DIR = args.output_dir
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
    
    # 단계 제어 업데이트
    Config.STAGE_MODEL_TRAINING = not args.skip_base_training
    Config.STAGE_OE_EXTRACTION = not args.skip_oe_extraction
    
    # 필요한 디렉토리 생성
    Config.create_directories()
    Config.save_config()
    
    print(f"--- 향상된 NLP Outlier Exposure 실험 ---")
    print(f"모드: {Config.EXPERIMENT_MODE}")
    print(f"NLP 데이터셋: {Config.CURRENT_NLP_DATASET}")
    print(f"출력 디렉토리: {Config.OUTPUT_DIR}")
    
    # 시드 설정
    set_seed(Config.RANDOM_STATE)
    
    # NLP Outlier Exposure 실험 실행
    run_nlp_oe_experiment(Config)

if __name__ == '__main__':
    main()