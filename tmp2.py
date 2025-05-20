# Okay, I've analyzed your request and the provided code. This is a significant refactoring to tailor the script specifically for your NLP Outlier Exposure experiments.

# Here's a summary of the planned changes:

# 1.  **Configuration (`Config` class):**
#     *   Remove all Syslog-specific configurations.
#     *   Set `EXPERIMENT_MODE` to "nlp" permanently.
#     *   Add configurations for OOD Test datasets (SNLI, IMDB, Multi30k, WMT16, Yelp).
#     *   Add `DEFAULT_OOD_TEST_DATASET` (e.g., 'wmt16').
#     *   Modify `NLP_MODEL_TYPE` and `OSR_NLP_MODEL_TYPE` to accept "roberta-base" in addition to "gru", "lstm".
#     *   The `CURRENT_NLP_DATASET` will dictate the In-Distribution dataset.

# 2.  **Dataset Loaders (`NLPDatasetLoader`):**
#     *   Add loaders for SNLI, IMDB, Multi30k, WMT16, Yelp. These will primarily be used for OOD testing and will be loaded with a placeholder label (e.g., -1) as their actual class doesn't matter for OOD detection scoring.

# 3.  **Data Module (`EnhancedDataModule`):**
#     *   Heavily simplify by removing all Syslog-related logic.
#     *   When `config.NLP_MODEL_TYPE` is "roberta-base", it will use `AutoTokenizer`. Otherwise, it uses the custom `NLPTokenizer`.
#     *   Its primary responsibility will be to prepare the In-Distribution dataset (specified by `CURRENT_NLP_DATASET`) for training the base classifier.

# 4.  **Model (`EnhancedModel`):**
#     *   When `config.NLP_MODEL_TYPE` is "roberta-base", it will instantiate `AutoModelForSequenceClassification`.
#     *   The `forward` method will handle outputs from both custom NLP models and RoBERTa.

# 5.  **Attention Analyzer & OE Extractor:**
#     *   These will need to correctly handle attention/hidden states from RoBERTa if it's chosen as the base NLP model. The logic for `_process_attention_batch_syslog` (which uses HuggingFace models) can be adapted.

# 6.  **OSR Components:**
#     *   `NLPModelOOD`: Will be updated to instantiate `RobertaForSequenceClassification` if `config.OSR_NLP_MODEL_TYPE` is "roberta-base". It will use the appropriate tokenizer.
#     *   `prepare_wikitext_ood_data_for_osr` will be generalized to `prepare_nlp_external_data_for_osr` to load any named dataset (like WMT16 for OOD test, or WikiText-2 for OE).
#     *   The main OSR experiment loop (`_run_nlp_osr_experiments`) will be restructured:
#         *   Load the specified In-Distribution dataset (e.g., 20 Newsgroups).
#         *   Load the specified OOD Test dataset (e.g., WMT16).
#         *   Run experiments:
#             1.  **Standard Model:** Train OSR model on ID data only. Evaluate against OOD Test.
#             2.  **WikiText-2 OE:** Train OSR model on ID + WikiText-2 as OE. Evaluate against OOD Test.
#             3.  **Attention-Derived OE:** For each attention-based OE file generated from the ID dataset (e.g., `oe_data_20newsgroups_attention_entropy_...`), train an OSR model using ID + this OE data. Evaluate against OOD Test.

# 7.  **Main Script & Argparse:**
#     *   Remove Syslog-related arguments.
#     *   Add `--ood_test_dataset` argument.
#     *   `--nlp_model_type` will now accept "roberta-base".
#     *   The script will run in "nlp" mode by default.

# This will result in a cleaner script focused on your NLP experiments.

# **Important Considerations before I generate the code:**

# *   **Tokenizer for OSR with RoBERTa:** If the base model is GRU (using `NLPTokenizer`) but the OSR model is RoBERTa, the OSR data loading will need to use `AutoTokenizer`. The current structure might need slight adjustments in `_run_single_nlp_osr_experiment` to select the correct tokenizer based on `Config.OSR_NLP_MODEL_TYPE`. For simplicity, it's often best if the base classifier and OSR classifier use compatible tokenizer types if one is custom and the other pretrained. I will assume `Config.NLP_MODEL_TYPE` and `Config.OSR_NLP_MODEL_TYPE` will be set consistently (e.g., both "gru" or both "roberta-base") for a given pipeline run.
# *   **Computational Cost:** Running OSR experiments for every combination of ID dataset, OE source (WikiText-2 + multiple attention-derived files), and OOD test dataset can be very time-consuming. The script will perform this as requested.

# Let me proceed with generating the modified code. It will be a substantial update.

# ```python
"""
Enhanced Unified OE (Out-of-Distribution) Extractor with NLP Dataset Support
Focusing on NLP experiments: 20 Newsgroups, TREC, SST-2 for In-Distribution,
WikiText-2 and Attention-derived data for Outlier Exposure,
and SNLI, IMDB, Multi30k, WMT16, Yelp for OOD Testing.
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
from datasets import Dataset as HFDataset, DatasetDict, concatenate_datasets, load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    RobertaTokenizer, # Kept for potential direct use if needed, but AutoTokenizer is preferred
    RobertaForSequenceClassification, # Kept for OSR model
    AdamW,
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
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # 병렬 처리 유지하되 경고 방지

# NLTK 초기화
NLTK_DATA_PATH = os.path.expanduser('~/AppData/Roaming/nltk_data')
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_PATH)

_NLTK_DOWNLOADS_DONE = False

def ensure_nltk_data():
    global _NLTK_DOWNLOADS_DONE
    if _NLTK_DOWNLOADS_DONE:
        return
    downloads_needed = []
    try: nltk.data.find('tokenizers/punkt')
    except LookupError: downloads_needed.append('punkt')
    try: nltk.data.find('corpora/stopwords')
    except LookupError: downloads_needed.append('stopwords')
    
    if downloads_needed:
        print(f"Downloading required NLTK data: {downloads_needed}")
        for item in downloads_needed:
            nltk.download(item, quiet=True, download_dir=NLTK_DATA_PATH)
    _NLTK_DOWNLOADS_DONE = True

ensure_nltk_data()

# --- Enhanced Configuration Class ---
class Config:
    EXPERIMENT_MODE = "nlp"  # Fixed to NLP

    # === NLP Dataset 설정 ===
    NLP_ID_DATASETS = { # In-Distribution Datasets
        '20newsgroups': {
            'name': 'SetFit/20_newsgroups', # Using SetFit version for consistency
            'text_column': 'text',
            'label_column': 'label'
        },
        'trec': {
            'name': 'trec',
            'text_column': 'text',
            'label_column': 'label-coarse' # or 'label-fine'
        },
        'sst2': {
            'name': 'sst2',
            'text_column': 'sentence',
            'label_column': 'label'
        }
    }
    NLP_OE_DATASETS = { # Standard OE Datasets
        'wikitext2': {
            'name': 'wikitext',
            'subset': 'wikitext-2-raw-v1',
            'text_column': 'text',
        }
    }
    NLP_OOD_TEST_DATASETS = { # For OOD evaluation
        'wmt16': {'name': 'wmt16', 'subset': 'ro-en', 'text_column': 'translation.en'}, # Example, pick a language pair
        'snli': {'name': 'snli', 'text_column': 'hypothesis'}, # or premise
        'imdb': {'name': 'imdb', 'text_column': 'text'},
        'multi30k': {'name': 'multi30k', 'subset': 'en', 'text_column': 'text'}, # English part
        'yelp_polarity': {'name': 'yelp_polarity', 'text_column': 'text'},
    }
 
    CURRENT_NLP_ID_DATASET = '20newsgroups'  # Experiment할 ID 데이터셋 선택
    DEFAULT_OE_DATASET = 'wikitext2' # Standard OE dataset
    DEFAULT_OOD_TEST_DATASET = 'wmt16' # OOD Test dataset

    # === 출력 디렉토리 설정 ===
    OUTPUT_DIR = 'enhanced_oe_nlp_results2' # Base output directory
    # Subdirectories will be derived from this
    
    # === NLP 모델 설정 (Base Classifier & OSR Model if GRU/LSTM) ===
    NLP_MODEL_TYPE = "roberta-base"  # "gru", "lstm", or "roberta-base"
    NLP_VOCAB_SIZE = 20000 # For custom tokenizer if NLP_MODEL_TYPE is GRU/LSTM
    NLP_EMBED_DIM = 300
    NLP_HIDDEN_DIM = 512
    NLP_NUM_LAYERS = 2
    NLP_DROPOUT = 0.3
    NLP_MAX_LENGTH = 256 # Max length for tokenizer
    NLP_BATCH_SIZE = 64 # Adjusted for potentially larger models
    NLP_NUM_EPOCHS = 30 # Base classifier epochs
    NLP_LEARNING_RATE = 2e-5 if NLP_MODEL_TYPE == "roberta-base" else 1e-3

    # === 하드웨어 설정 ===
    ACCELERATOR = "auto"
    DEVICES = "auto"
    PRECISION = "16-mixed" if torch.cuda.is_available() else "32-true"
    NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1
    
    # === 학습 설정 (Base Classifier) ===
    LOG_EVERY_N_STEPS = 50
    GRADIENT_CLIP_VAL = 1.0
    USE_WEIGHTED_LOSS = True # For base classifier
    RANDOM_STATE = 42
    
    # === 어텐션 설정 (for deriving OE data from ID) ===
    ATTENTION_TOP_PERCENT = 0.20
    MIN_TOP_WORDS = 10
    TOP_K_ATTENTION = 3 # For attention metrics on masked text
    ATTENTION_LAYER = -1 # For RoBERTa-like models, last layer
    
    # === OE 필터링 설정 (for deriving OE data from ID) ===
    METRIC_SETTINGS = {
        'attention_entropy': {'percentile': 70, 'mode': 'higher'},
        'max_attention': {'percentile': 10, 'mode': 'lower'},
        'removed_avg_attention': {'percentile': 80, 'mode': 'higher'},
        'top_k_avg_attention': {'percentile': 20, 'mode': 'lower'}
    }
    FILTERING_SEQUENCE = [
        ('removed_avg_attention', {'percentile': 80, 'mode': 'higher'}),
        ('attention_entropy', {'percentile': 70, 'mode': 'higher'}),
        ('max_attention', {'percentile': 10, 'mode': 'lower'})
    ]
    TEXT_COLUMN_IN_OE_FILES = 'masked_text_attention' # Column name in generated OE csv files
    
    # === OSR Experiment Settings ===
    OSR_NLP_MODEL_TYPE = "roberta-base" # OSR model: "gru", "lstm", or "roberta-base"
    # OSR_NLP_VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT use NLP_ counterparts if GRU/LSTM
    OSR_NLP_MAX_LENGTH = 256 # Max length for OSR model inputs
    OSR_NLP_BATCH_SIZE = 32
    OSR_NLP_NUM_EPOCHS = 15 # Epochs for OSR model training
    OSR_NLP_LEARNING_RATE = 2e-5 if OSR_NLP_MODEL_TYPE == "roberta-base" else 1e-3
    
    OSR_OE_LAMBDA = 1.0 # Weight for OE loss term
    OSR_TEMPERATURE = 1.0 # For softmax scaling in OSR evaluation
    OSR_THRESHOLD_PERCENTILE = 5.0 # For determining OOD threshold from ID scores
    OSR_NUM_DATALOADER_WORKERS = NUM_WORKERS
    
    # === 실행 단계 제어 ===
    STAGE_MODEL_TRAINING = True
    STAGE_ATTENTION_EXTRACTION = True
    STAGE_OE_EXTRACTION = True # Extracting OE data from ID using attention
    STAGE_VISUALIZATION = True
    STAGE_OSR_EXPERIMENTS = True
    
    # === Flags ===
    OSR_SAVE_MODEL_PER_EXPERIMENT = True
    OSR_EVAL_ONLY = True # If true, loads existing OSR models instead of training
    OSR_NO_PLOT_PER_EXPERIMENT = False
    OSR_SKIP_STANDARD_MODEL = False # If true, skips OSR without any OE
    
    # === HuggingFace Cache ===
    # DATA_DIR_EXTERNAL_HF, CACHE_DIR_HF will be derived from OUTPUT_DIR

    # Derived paths (will be set in create_directories)
    MODEL_SAVE_DIR: str
    LOG_DIR: str
    CONFUSION_MATRIX_DIR: str
    VIS_DIR: str
    OE_DATA_DIR: str         # For attention-derived OE datasets
    ATTENTION_DATA_DIR: str
    NLP_DATA_DIR: str        # For downloaded raw NLP datasets (not currently used for saving, HF handles cache)
    OSR_EXPERIMENT_DIR: str
    OSR_MODEL_DIR: str
    OSR_RESULT_DIR: str
    DATA_DIR_EXTERNAL_HF: str
    CACHE_DIR_HF: str


    @classmethod
    def update_derived_paths(cls):
        cls.MODEL_SAVE_DIR = os.path.join(cls.OUTPUT_DIR, "base_classifier_model")
        cls.LOG_DIR = os.path.join(cls.OUTPUT_DIR, "lightning_logs")
        cls.CONFUSION_MATRIX_DIR = os.path.join(cls.LOG_DIR, "confusion_matrices")
        cls.VIS_DIR = os.path.join(cls.OUTPUT_DIR, "oe_extraction_visualizations")
        cls.OE_DATA_DIR = os.path.join(cls.OUTPUT_DIR, "derived_oe_datasets_from_id") # Store attention-derived OE
        cls.ATTENTION_DATA_DIR = os.path.join(cls.OUTPUT_DIR, "attention_analysis")
        cls.NLP_DATA_DIR = os.path.join(cls.OUTPUT_DIR, "nlp_datasets_cache") # More of a cache concept
        
        cls.OSR_EXPERIMENT_DIR = os.path.join(cls.OUTPUT_DIR, "osr_experiments")
        cls.OSR_MODEL_DIR = os.path.join(cls.OSR_EXPERIMENT_DIR, "models")
        cls.OSR_RESULT_DIR = os.path.join(cls.OSR_EXPERIMENT_DIR, "results")
        
        cls.DATA_DIR_EXTERNAL_HF = os.path.join(cls.OUTPUT_DIR, 'data_external_hf') # Main cache for HuggingFace
        cls.CACHE_DIR_HF = os.path.join(cls.DATA_DIR_EXTERNAL_HF, "hf_cache")


    @classmethod
    def create_directories(cls):
        cls.update_derived_paths()
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
        cls.update_derived_paths()
        if filepath is None:
            filepath = os.path.join(cls.OUTPUT_DIR, f'config_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        config_dict = {}
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                value = getattr(cls, attr)
                # Ensure derived paths are strings for JSON
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                     if isinstance(value, str) and os.path.isabs(value) and cls.OUTPUT_DIR in value: # Heuristic for path
                        value = os.path.relpath(value, os.path.dirname(cls.OUTPUT_DIR)) # Store relative path if possible
                     config_dict[attr] = value
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        print(f"Configuration saved to {filepath}")

# === 헬퍼 함수들 ===
DEVICE_OSR = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True) # PyTorch Lightning specific
    print(f"Seed set to {seed}")

def preprocess_text_for_custom_nlp(text): # For GRU/LSTM with custom tokenizer
    if not isinstance(text, str): return ""
    text = re.sub(r'\s+', ' ', text).strip().lower()
    text = re.sub(r'[^\w\s]', ' ', text) # Remove punctuation for simpler tokenization
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text_for_hf(text): # For HuggingFace tokenizers (RoBERTa etc.)
    if not isinstance(text, str): return ""
    return re.sub(r'\s+', ' ', text).strip() # Minimal preprocessing

def tokenize_nltk(text):
    if not text: return []
    global _NLTK_DOWNLOADS_DONE
    if not _NLTK_DOWNLOADS_DONE: ensure_nltk_data()
    try:
        return word_tokenize(text)
    except Exception as e:
        print(f"NLTK word_tokenize failed: {e}. Using simple split.")
        return text.split()

def create_masked_sentence(original_text, important_words):
    if not isinstance(original_text, str): return ""
    if not important_words: return original_text
    
    processed_text = preprocess_text_for_custom_nlp(original_text) # Use consistent preprocessing
    tokens = tokenize_nltk(processed_text)
    important_set_lower = {word.lower() for word in important_words}
    masked_tokens = [word for word in tokens if word.lower() not in important_set_lower]
    masked_sentence = ' '.join(masked_tokens)
    
    return "__EMPTY_MASKED__" if not masked_sentence else masked_sentence

def safe_literal_eval(val):
    try:
        if isinstance(val, str) and val.strip().startswith('['): return ast.literal_eval(val)
        elif isinstance(val, list): return val
        return []
    except: return []

def get_nested_value(item, field_path):
    """
    중첩된 딕셔너리에서 점 표기법을 사용하여 값을 추출합니다.
    예: get_nested_value(item, 'translation.en')은 item['translation']['en']을 반환합니다.
    """
    if '.' not in field_path:
        return item.get(field_path)
    
    parts = field_path.split('.')
    value = item
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value
# === NLP 데이터셋 로더들 ===
class NLPDatasetLoader:

    @staticmethod
    def _load_hf_dataset(dataset_config: Dict, split='train', text_only=False):
        print(f"Loading {dataset_config['name']} ({dataset_config.get('subset', 'default')}) - split: {split}")
        try:
            # 전체 데이터셋을 로드한 다음 스플릿 선택
            # 이는 비효율적이지만 다양한 HF 데이터셋 구조에 더 견고합니다
            if dataset_config.get('subset'):
                dataset_full = load_dataset(dataset_config['name'], dataset_config['subset'], cache_dir=Config.CACHE_DIR_HF)
            else:
                dataset_full = load_dataset(dataset_config['name'], cache_dir=Config.CACHE_DIR_HF)

            # 요청된 스플릿 가져오기, 필요시 대체 스플릿 사용
            if split in dataset_full:
                data_split = dataset_full[split]
            elif split == 'test' and 'validation' in dataset_full: # test 대체용 fallback
                print(f"'{split}' split not found, using 'validation' split instead for {dataset_config['name']}.")
                data_split = dataset_full['validation']
            elif split == 'validation' and 'test' in dataset_full: # validation 대체용 fallback
                print(f"'{split}' split not found, using 'test' split instead for {dataset_config['name']}.")
                data_split = dataset_full['test']
            else: # 특정 스플릿이 없으면 'train' 또는 첫 번째 사용 가능한 스플릿 사용
                available_splits = list(dataset_full.keys())
                chosen_split_name = 'train' if 'train' in available_splits else available_splits[0]
                print(f"'{split}' split not found, using '{chosen_split_name}' split instead for {dataset_config['name']}.")
                data_split = dataset_full[chosen_split_name]

            texts = []
            labels = [] # text_only 또는 label_column이 없으면 비어 있음

            for item in tqdm(data_split, desc=f"Processing {dataset_config['name']} [{split}]"):
                text = get_nested_value(item, dataset_config['text_column'])
                if text is not None and isinstance(text, str) and text.strip():
                    texts.append(text.strip())
                    if not text_only and dataset_config.get('label_column'):
                        label = item.get(dataset_config['label_column'])
                        if label is not None:
                            labels.append(label)
                        else: # 레이블이 누락된 경우에도 대응 유지
                            labels.append(-1) # 누락된 레이블용 플레이스홀더
                
            if not text_only and dataset_config.get('label_column'):
                # 레이블이 필요한 경우 누락된 레이블이 있는 샘플 필터링
                valid_indices = [i for i, lbl in enumerate(labels) if lbl != -1]
                texts = [texts[i] for i in valid_indices]
                labels = [labels[i] for i in valid_indices]
                return {'text': texts, 'label': labels}
            return {'text': texts} # 레이블 없음 또는 text_only
        
        except Exception as e:
            print(f"Error loading dataset {dataset_config['name']} ({dataset_config.get('subset', 'default')}): {e}")
            return None
    @staticmethod
    def load_id_dataset(dataset_name: str): # For In-Distribution
        if dataset_name not in Config.NLP_ID_DATASETS:
            raise ValueError(f"Unknown ID dataset: {dataset_name}")
        config = Config.NLP_ID_DATASETS[dataset_name]
        
        # Most ID datasets have standard train/test splits.
        # We'll combine them then re-split in DataModule.
        train_data = NLPDatasetLoader._load_hf_dataset(config, split='train')
        # Try 'test', then 'validation' for the test part of ID data
        test_data = NLPDatasetLoader._load_hf_dataset(config, split='test')
        if test_data is None or not test_data['text']: # Fallback
            test_data = NLPDatasetLoader._load_hf_dataset(config, split='validation')

        if train_data and test_data and train_data['text'] and test_data['text']:
            return {
                'train': train_data,
                'test': test_data
            }
        print(f"Failed to load sufficient train/test data for ID dataset {dataset_name}")
        return None

    @staticmethod
    def load_oe_dataset(dataset_name: str = Config.DEFAULT_OE_DATASET): # For standard OE (e.g. WikiText2)
        if dataset_name not in Config.NLP_OE_DATASETS:
            raise ValueError(f"Unknown OE dataset: {dataset_name}")
        config = Config.NLP_OE_DATASETS[dataset_name]
        # OE data is usually just text from the 'train' split or main body
        data = NLPDatasetLoader._load_hf_dataset(config, split='train', text_only=True)
        if data and data['text']:
            # For WikiText, often need sentence splitting
            if dataset_name == 'wikitext2':
                all_sentences = []
                for doc in tqdm(data['text'], desc="Sentence tokenizing WikiText2"):
                    sentences = sent_tokenize(doc)
                    all_sentences.extend([s.strip() for s in sentences if len(s.strip()) > 10])
                return {'text': all_sentences}
            return {'text': data['text']}
        return None

    @staticmethod
    def load_ood_test_dataset(dataset_name: str = Config.DEFAULT_OOD_TEST_DATASET): # For OOD evaluation
        if dataset_name not in Config.NLP_OOD_TEST_DATASETS:
            raise ValueError(f"Unknown OOD Test dataset: {dataset_name}")
        config = Config.NLP_OOD_TEST_DATASETS[dataset_name]
        # OOD test data is usually from 'test' or 'validation' split, text only
        data = NLPDatasetLoader._load_hf_dataset(config, split='test', text_only=True)
        if data is None or not data['text']: # Fallback
             data = NLPDatasetLoader._load_hf_dataset(config, split='validation', text_only=True)
        
        if data and data['text']:
            return {'text': data['text']}
        return None

# === NLP용 토크나이저 (Custom GRU/LSTM) ===
class NLPTokenizer:
    def __init__(self, vocab_size=10000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.vocab = {}
        self.inverse_vocab = {}
        self.word_counts = defaultdict(int)
        self.pad_token = "<PAD>"; self.unk_token = "<UNK>";
        self.cls_token = "<CLS>"; self.sep_token = "<SEP>";
        self.pad_token_id = 0; self.unk_token_id = 1;
        self.cls_token_id = 2; self.sep_token_id = 3;
        
    def build_vocab(self, texts: List[str]):
        print("Building custom vocabulary...")
        for text in tqdm(texts, desc="Counting words for vocab"):
            words = tokenize_nltk(preprocess_text_for_custom_nlp(text))
            for word in words:
                if len(word) > 1: self.word_counts[word] += 1
        
        self.vocab = {
            self.pad_token: self.pad_token_id, self.unk_token: self.unk_token_id,
            self.cls_token: self.cls_token_id, self.sep_token: self.sep_token_id
        }
        
        filtered_words = [(w, c) for w, c in self.word_counts.items() if c >= self.min_freq]
        sorted_words = sorted(filtered_words, key=lambda x: x[1], reverse=True)
        
        for i, (word, _) in enumerate(sorted_words[:self.vocab_size - len(self.vocab)]):
            self.vocab[word] = i + len(self.vocab) # Start IDs after special tokens
        
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Custom vocabulary built: {len(self.vocab)} words (target size {self.vocab_size})")

    def encode(self, text: str, max_length=512) -> List[int]:
        if not isinstance(text, str): text = str(text) if text is not None else ""
        words = tokenize_nltk(preprocess_text_for_custom_nlp(text))
        
        token_ids = [self.cls_token_id]
        for word in words:
            token_ids.append(self.vocab.get(word, self.unk_token_id))
        token_ids.append(self.sep_token_id)
        
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length-1] + [self.sep_token_id]
        else:
            token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        words = [self.inverse_vocab[token_id] for token_id in token_ids 
                 if token_id != self.pad_token_id and \
                    self.inverse_vocab[token_id] not in [self.cls_token, self.sep_token]]
        return ' '.join(words)

    # Add properties for HuggingFace compatibility if needed by DataCollator
    @property
    def model_max_length(self):
        return Config.NLP_MAX_LENGTH # Or a more specific max_length if set

    def __call__(self, texts: Union[str, List[str]], **kwargs): # Mimic HF tokenizer call
        max_length = kwargs.get('max_length', Config.NLP_MAX_LENGTH)
        padding = kwargs.get('padding', 'max_length') # 'max_length' is typical
        truncation = kwargs.get('truncation', True)

        if isinstance(texts, str): texts = [texts]
        
        all_input_ids = []
        all_attention_masks = []

        for text in texts:
            token_ids = self.encode(text, max_length=max_length) # Relies on encode to handle padding
            attention_mask = [1 if id != self.pad_token_id else 0 for id in token_ids]
            all_input_ids.append(token_ids)
            all_attention_masks.append(attention_mask)

        # For return_tensors='pt'
        if kwargs.get('return_tensors') == 'pt':
            return {
                'input_ids': torch.tensor(all_input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(all_attention_masks, dtype=torch.long)
            }
        return {'input_ids': all_input_ids, 'attention_mask': all_attention_masks}


# === NLP용 Dataset 클래스 ===
class NLPTorchDataset(TorchDataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]], 
                 tokenizer: Union[NLPTokenizer, AutoTokenizer], max_length: int, 
                 is_hf_tokenizer: bool):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_hf_tokenizer = is_hf_tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]) # Ensure text is string
        label = self.labels[idx] if self.labels else 0 # Default label if none

        if self.is_hf_tokenizer:
            # Preprocess text for HuggingFace tokenizer
            processed_text = preprocess_text_for_hf(text)
            encoding = self.tokenizer(
                processed_text,
                max_length=self.max_length,
                padding='max_length', # Pad to max_length
                truncation=True,
                return_tensors='pt' 
            )
            input_ids = encoding['input_ids'].squeeze(0) # Remove batch dim
            attention_mask = encoding['attention_mask'].squeeze(0)
        else: # Custom NLPTokenizer
            # Preprocess text for custom tokenizer
            processed_text = preprocess_text_for_custom_nlp(text)
            token_ids = self.tokenizer.encode(processed_text, max_length=self.max_length)
            input_ids = torch.tensor(token_ids, dtype=torch.long)
            attention_mask = torch.tensor([1 if id != self.tokenizer.pad_token_id else 0 for id in token_ids], dtype=torch.long)
            

            
        item = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if self.labels: # This condition is important
            item['label'] = torch.tensor(label, dtype=torch.long)
        return item

# OSRNLPDataset can reuse NLPTorchDataset structure
OSRNNLPTorchDataset = NLPTorchDataset


# === NLP 모델들 ===
class CustomNLPClassifier(nn.Module): # For GRU/LSTM
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, 
                 dropout=0.3, model_type="gru", attention=True):
        super().__init__()
        self.model_type = model_type
        self.attention_enabled = attention # Renamed from self.attention to avoid conflict
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.embedding.weight.data[0].fill_(0) 
        
        rnn_module = nn.GRU if model_type == "gru" else nn.LSTM
        self.rnn = rnn_module(embed_dim, hidden_dim, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0,
                              bidirectional=True)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name: nn.init.xavier_uniform_(param)
            elif 'bias' in name: nn.init.zeros_(param)
        
        if self.attention_enabled:
            self.attention_layer = nn.Linear(hidden_dim * 2, 1)
            nn.init.xavier_uniform_(self.attention_layer.weight)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids, attention_mask=None, output_attentions=False, output_features=False):
        embedded = self.embedding(input_ids)
        rnn_output, _ = self.rnn(embedded) # LSTM returns (h_n, c_n) as second val
        
        features = None # To store features before classifier
        att_weights = None # To store attention weights

        if self.attention_enabled and attention_mask is not None:
            att_logits = self.attention_layer(rnn_output).squeeze(-1)
            att_logits = att_logits.masked_fill(~attention_mask.bool(), float('-inf'))
            att_weights = F.softmax(att_logits, dim=1)
            if torch.isnan(att_weights).any():
                att_weights = torch.ones_like(att_weights) / att_weights.size(1)
            weighted_output = torch.bmm(att_weights.unsqueeze(1), rnn_output).squeeze(1)
        else: # Use last hidden state if no attention or mask
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1)
                # Ensure lengths are valid indices
                valid_lengths = torch.clamp(lengths - 1, min=0)
                weighted_output = rnn_output[torch.arange(len(rnn_output)), valid_lengths]

            else:
                weighted_output = rnn_output[:, -1]
        
        features = weighted_output # Feature is the RNN output (attentional or last state)
        
        if torch.isnan(weighted_output).any():
            return torch.zeros(input_ids.size(0), self.classifier.out_features, device=input_ids.device)
        
        dropped_output = self.dropout(weighted_output)
        logits = self.classifier(dropped_output)
        
        if torch.isnan(logits).any(): return torch.zeros_like(logits)

        # Construct output similar to HuggingFace models for consistency
        class Output:
            def __init__(self, logits=None, attentions=None, hidden_states=None):
                self.logits = logits
                self.attentions = attentions # List of attention tensors, here just one
                self.hidden_states = hidden_states # List of hidden_states, here just one (features)
        
        return_att = [att_weights] if output_attentions and att_weights is not None else None
        return_feat = [features] if output_features and features is not None else None # HF expects list of hidden_states
        
        return Output(logits=logits, attentions=return_att, hidden_states=return_feat)


class NLPModelOOD(nn.Module): # OSR Model Wrapper
    def __init__(self, config: Config, num_classes: int, tokenizer_vocab_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.is_hf_model = config.OSR_NLP_MODEL_TYPE == "roberta-base"

        if self.is_hf_model:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config.OSR_NLP_MODEL_TYPE,
                num_labels=num_classes,
                output_attentions=False, # Not typically used for OSR loss, but can be enabled for analysis
                output_hidden_states=True # Need last hidden state for features
            )
        else: # GRU/LSTM
            if tokenizer_vocab_size is None:
                 raise ValueError("tokenizer_vocab_size must be provided for custom NLP models in OSR")
            self.model = CustomNLPClassifier(
                vocab_size=tokenizer_vocab_size,
                embed_dim=config.NLP_EMBED_DIM,
                hidden_dim=config.NLP_HIDDEN_DIM,
                num_classes=num_classes,
                num_layers=config.NLP_NUM_LAYERS,
                dropout=config.NLP_DROPOUT,
                model_type=config.OSR_NLP_MODEL_TYPE, # gru or lstm
                attention=True # Enable attention for feature representation
            )

    def forward(self, input_ids, attention_mask=None, output_features=False):
        # Standardize output: return (logits, features) if output_features is True
        # Else, return just logits
        if self.is_hf_model:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            if output_features:
                # Use the CLS token's representation from the last layer
                features = outputs.hidden_states[-1][:, 0, :] # [batch_size, hidden_size]
                return logits, features
            return logits
        else: # CustomNLPClassifier
            outputs = self.model(input_ids, attention_mask, output_features=True) # Custom model already can give features
            logits = outputs.logits
            if output_features:
                features = outputs.hidden_states[0] if outputs.hidden_states else None # It's already the desired feature
                return logits, features
            return logits

# === Enhanced DataModule for NLP ===
class EnhancedDataModule(pl.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # self.save_hyperparameters(ignore=['config']) # Causes issues with non-serializable tokenizer

        self.is_hf_tokenizer = self.config.NLP_MODEL_TYPE == "roberta-base"
        if self.is_hf_tokenizer:
            print(f"Using HuggingFace tokenizer for: {self.config.NLP_MODEL_TYPE}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.NLP_MODEL_TYPE, cache_dir=self.config.CACHE_DIR_HF)
        else:
            print(f"Using custom NLPTokenizer for: {self.config.NLP_MODEL_TYPE}")
            self.tokenizer = NLPTokenizer(vocab_size=self.config.NLP_VOCAB_SIZE)
        
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer) if self.is_hf_tokenizer else None

        self.df_full = None
        self.train_df_final = None
        self.val_df_final = None
        self.label2id = None
        self.id2label = None
        self.num_labels = None
        self.class_weights = None
        
    def prepare_data(self): # Download etc.
        print(f"Preparing ID data: {self.config.CURRENT_NLP_ID_DATASET}")
        NLPDatasetLoader.load_id_dataset(self.config.CURRENT_NLP_ID_DATASET) # Just to ensure download
        if not self.is_hf_tokenizer: # Build vocab if custom tokenizer
            # Need to load texts to build vocab
            raw_data = NLPDatasetLoader.load_id_dataset(self.config.CURRENT_NLP_ID_DATASET)
            if raw_data:
                all_texts = raw_data['train']['text'] + raw_data['test']['text']
                self.tokenizer.build_vocab(all_texts)
            else:
                raise RuntimeError(f"Failed to load data for vocab building: {self.config.CURRENT_NLP_ID_DATASET}")

    # In EnhancedDataModule.setup()
    def setup(self, stage=None):
        if self.df_full is not None: return

        data = NLPDatasetLoader.load_id_dataset(self.config.CURRENT_NLP_ID_DATASET)
        if data is None: raise ValueError(f"Failed to load ID dataset: {self.config.CURRENT_NLP_ID_DATASET}")

        train_df = pd.DataFrame(data['train'])
        test_df = pd.DataFrame(data['test'])
        train_df['split'] = 'train'; test_df['split'] = 'test'
        self.df_full = pd.concat([train_df, test_df], ignore_index=True).dropna(subset=['text', 'label'])
        self.df_full['text'] = self.df_full['text'].astype(str)

        # Ensure labels are standard Python types
        unique_labels_raw = sorted(self.df_full['label'].unique())
        unique_labels = [item.item() if hasattr(item, 'item') else item for item in unique_labels_raw] # Convert numpy types to python types
        # Also ensure the integer IDs are python ints
        self.label2id = {label: int(i) for i, label in enumerate(unique_labels)}
        self.id2label = {int(i): label for label, i in self.label2id.items()} # Keys are already int from enumerate
        
        self.num_labels = len(unique_labels)
        print(f"ID Dataset '{self.config.CURRENT_NLP_ID_DATASET}' Label mapping: {self.label2id}")
        
        # Ensure 'label_id' is also standard int if it's used later for stratification or indexing
        self.df_full['label_id'] = self.df_full['label'].map(self.label2id).astype(int)


        train_val_df = self.df_full[self.df_full['split'] == 'train'].copy()
        self.val_df_final = self.df_full[self.df_full['split'] == 'test'].copy() 

        # Stratify column must be clean
        stratify_labels = train_val_df['label_id']
        can_stratify = len(stratify_labels.unique()) > 1 and stratify_labels.value_counts().min() > 1

        self.train_df_final, self.val_df_for_train = train_test_split(
            train_val_df, test_size=0.1, random_state=self.config.RANDOM_STATE,
            stratify=stratify_labels if can_stratify else None
        )
        print(f"ID Data splits: Train: {len(self.train_df_final)}, Base Classifier Val: {len(self.val_df_for_train)}, OSR ID Test (from original test split): {len(self.val_df_final)}")

        if self.config.USE_WEIGHTED_LOSS:
            labels_for_weights_raw = self.train_df_final['label_id'].values
            # Ensure labels_for_weights are python ints for compute_class_weight
            labels_for_weights = np.array([int(l) for l in labels_for_weights_raw])
            
            unique_w_labels_raw = np.unique(labels_for_weights)
            unique_w_labels = np.array([int(l) for l in unique_w_labels_raw])

            try:
                class_weights_array = compute_class_weight('balanced', classes=unique_w_labels, y=labels_for_weights)
                self.class_weights = torch.zeros(self.num_labels) 
                for i, label_idx_val in enumerate(unique_w_labels): # label_idx_val is already int
                    if int(label_idx_val) < self.num_labels: 
                         self.class_weights[int(label_idx_val)] = class_weights_array[i]
                print(f"Computed class weights for base classifier: {self.class_weights}")
            except ValueError as e:
                print(f"Error computing class weights: {e}. Using uniform weights.")
                self.class_weights = None
     
    def _create_dataloader(self, df: pd.DataFrame, shuffle=False, sampler=None):
        dataset = NLPTorchDataset(
            df['text'].tolist(),
            df['label_id'].tolist(),
            self.tokenizer,
            max_length=self.config.NLP_MAX_LENGTH,
            is_hf_tokenizer=self.is_hf_tokenizer
        )
        return DataLoader(
            dataset, 
            batch_size=self.config.NLP_BATCH_SIZE, 
            shuffle=shuffle, 
            sampler=sampler,
            num_workers=self.config.NUM_WORKERS, 
            multiprocessing_context='spawn' if self.config.NUM_WORKERS > 0 else None,  # 추가된 부분
            pin_memory=True,
            collate_fn=self.data_collator,
            persistent_workers=self.config.NUM_WORKERS > 0
        )

    def train_dataloader(self):
        sampler = None
        shuffle = True
        if self.config.USE_WEIGHTED_LOSS and self.class_weights is not None: # sampler for weighted loss
            from torch.utils.data import WeightedRandomSampler
            class_counts = self.train_df_final['label_id'].value_counts().sort_index()
            
            # Ensure class_counts align with num_labels, handle missing classes
            weights_per_class = torch.zeros(self.num_labels)
            for class_idx, count in class_counts.items():
                if count > 0: # Check to prevent division by zero
                    weights_per_class[class_idx] = 1.0 / count
                else: # Assign a very small weight if a class somehow has 0 samples after split
                    weights_per_class[class_idx] = 1e-6 # Or handle as error

            sample_weights = [weights_per_class[label] for label in self.train_df_final['label_id']]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            shuffle = False # Sampler handles shuffling
        return self._create_dataloader(self.train_df_final, shuffle=shuffle, sampler=sampler)

    def val_dataloader(self):
        return self._create_dataloader(self.val_df_for_train) # Using the split-off validation set

    def test_dataloader(self): # This will be the OSR ID test set
        return self._create_dataloader(self.val_df_final)

    def get_full_dataframe(self):
        if self.df_full is None: self.setup()
        return self.df_full

# === Enhanced Model (Base Classifier) ===
class EnhancedModel(pl.LightningModule):
    def __init__(self, config: Config, num_labels: int, label2id: dict, id2label: dict, 
                 class_weights=None, tokenizer_for_custom_model: Optional[NLPTokenizer]=None): # Pass custom tokenizer if needed
        super().__init__()
        self.config_params = config 
        self.num_labels = num_labels # Manually assign if not saving via save_hyperparameters
        self.label2id = label2id
        self.id2label = id2label
        self.class_weights_tensor = class_weights
        
        # Explicitly save what you need, Lightning will make them self.hparams.X
        self.save_hyperparameters(
            {'num_labels': num_labels, 'label2id': label2id, 'id2label': id2label}
        )
        # self.save_hyperparameters(ignore=['config_params', 'class_weights_tensor', 'tokenizer_for_custom_model'])

        self.is_hf_model = self.config_params.NLP_MODEL_TYPE == "roberta-base"
        
        if self.is_hf_model:
            print(f"Initializing HF base classifier: {self.config_params.NLP_MODEL_TYPE} for {num_labels} classes")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config_params.NLP_MODEL_TYPE, num_labels=num_labels, label2id=label2id, id2label=id2label,
                output_attentions=True, output_hidden_states=True, cache_dir=self.config_params.CACHE_DIR_HF
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config_params.NLP_MODEL_TYPE, cache_dir=self.config_params.CACHE_DIR_HF)

        else: # GRU / LSTM
            print(f"Initializing Custom NLP classifier: {self.config_params.NLP_MODEL_TYPE} for {num_labels} classes")
            if tokenizer_for_custom_model is None:
                raise ValueError("tokenizer_for_custom_model is required for GRU/LSTM models")
            self.tokenizer = tokenizer_for_custom_model # This is an NLPTokenizer instance
            self.model = CustomNLPClassifier(
                vocab_size=len(self.tokenizer.vocab), # Use actual vocab size
                embed_dim=self.config_params.NLP_EMBED_DIM,
                hidden_dim=self.config_params.NLP_HIDDEN_DIM,
                num_classes=num_labels,
                num_layers=self.config_params.NLP_NUM_LAYERS,
                dropout=self.config_params.NLP_DROPOUT,
                model_type=self.config_params.NLP_MODEL_TYPE,
                attention=True 
            )

        if self.config_params.USE_WEIGHTED_LOSS and self.class_weights_tensor is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
            print("Base classifier using weighted CrossEntropyLoss")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            print("Base classifier using standard CrossEntropyLoss")
        
        metrics_collection = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=num_labels),
            'f1_macro': torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='macro')
        })
        self.train_metrics = metrics_collection.clone(prefix='train_')
        self.val_metrics = metrics_collection.clone(prefix='val_')
        self.val_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_labels)

    def setup(self, stage=None):
        if self.config_params.USE_WEIGHTED_LOSS and hasattr(self.loss_fn, 'weight') and self.loss_fn.weight is not None:
            self.loss_fn.weight = self.loss_fn.weight.to(self.device)
            print(f"Moved class weights to {self.device}")

    def forward(self, batch, output_attentions=False, output_features=False): # For inference/analysis
        # output_features for CustomNLPClassifier corresponds to output_hidden_states for HF
        return self.model(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'],
            output_attentions=output_attentions,
            # For HF models, output_hidden_states is set at init. For custom, pass flag.
            **(dict(output_features=output_features) if not self.is_hf_model else {}) 
        )

    def _common_step(self, batch, batch_idx):
        # Try 'labels' first (common for HF collators), then 'label'
        if 'labels' in batch:
            labels_tensor = batch.pop('labels')
        elif 'label' in batch:
            labels_tensor = batch.pop('label')
        else:
            # This indicates a more fundamental issue with data preparation or collation
            print("Problematic batch keys:", batch.keys())
            raise KeyError("Batch from DataLoader does not contain 'label' or 'labels' key.")
        
        # For HF model, it computes loss internally if labels are passed.
        # For custom model, we compute loss manually.
        if self.is_hf_model:
            # Pass the original batch (which still has input_ids, attention_mask) 
            # and the extracted labels_tensor separately to the model call.
            outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=labels_tensor)
            loss = outputs.loss
            logits = outputs.logits
        else: # Custom model
            outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits
            loss = self.loss_fn(logits, labels_tensor)
        
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels_tensor # return the tensor of labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        self.log_dict(self.train_metrics(preds, labels), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        self.val_metrics.update(preds, labels)
        self.val_cm.update(preds, labels)
        self.log('val_f1_macro', self.val_metrics['f1_macro'], on_epoch=True, prog_bar=True) # Log specifically for checkpoint
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        # Compute and log metrics
        metrics_output = self.val_metrics.compute()
        self.log_dict(metrics_output, prog_bar=True)
        self.val_metrics.reset()

        # Confusion Matrix
        try:
            val_cm_computed = self.val_cm.compute()
            class_names = [str(self.id2label.get(i, f"Class {i}")) for i in range(self.num_labels)]
            cm_df = pd.DataFrame(val_cm_computed.cpu().numpy(), index=class_names, columns=class_names)
            # print(f"\nClassifier Validation Confusion Matrix (Epoch {self.current_epoch}):\n{cm_df}")
            cm_filename = os.path.join(self.config_params.CONFUSION_MATRIX_DIR, 
                                     f"clf_val_cm_epoch_{self.current_epoch}.csv")
            cm_df.to_csv(cm_filename)
        except Exception as e: print(f"Error in classifier validation CM: {e}")
        finally: self.val_cm.reset()

    def configure_optimizers(self):
        if self.is_hf_model:
            # For HuggingFace models, transformers.AdamW is often used,
            # but torch.optim.AdamW is now preferred.
            # Ensure you are using one consistently or as intended.
            # If you imported `from transformers import AdamW`, this uses that.
            # Otherwise, if you meant torch.optim.AdamW, ensure it's clear.
            # For this example, assuming torch.optim.AdamW or a compatible AdamW.
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.config_params.NLP_LEARNING_RATE)
        else: # GRU/LSTM
            optimizer = optim.Adam(self.parameters(), lr=self.config_params.NLP_LEARNING_RATE)
        
        # Check if trainer information is available to calculate steps
        if self.trainer and hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches > 0:
            num_training_steps = self.trainer.estimated_stepping_batches
            # 예를 들어 전체 스텝의 6%를 워밍업으로 사용 (이 비율은 조절 가능)
            num_warmup_steps = int(num_training_steps * 0.06)
            print(f"Base Classifier Scheduler: Total steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        else:
            # Fallback if trainer info is not available (e.g., when model is initialized outside of Trainer context)
            # or if estimated_stepping_batches is 0 (e.g. sanity check with very few batches might lead to this)
            print("Warning: Could not estimate training steps for base classifier scheduler. Using optimizer only, or scheduler with fixed steps if preferred.")
            # Optionally, you could define a default scheduler here if needed,
            # or just return the optimizer. For simplicity, returning optimizer only for now.
            # If a scheduler is critical even in fallback, define total_steps based on epochs and a guess of batches_per_epoch.
            # For example:
            #  if self.trainer and self.trainer.datamodule and hasattr(self.trainer.datamodule, 'train_dataloader'):
            #      try:
            #          num_batches_per_epoch = len(self.trainer.datamodule.train_dataloader())
            #          num_training_steps = num_batches_per_epoch * self.config_params.NLP_NUM_EPOCHS
            #          num_warmup_steps = int(num_training_steps * 0.06)
            #          scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
            #          return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
            #      except Exception:
            #          pass # Fall through to optimizer only

            return optimizer
                
        if self.trainer and hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches > 0:
            num_training_steps = self.trainer.estimated_stepping_batches
            # 예를 들어 전체 스텝의 6%를 워밍업으로 사용
            num_warmup_steps = int(num_training_steps * 0.06)
            print(f"Scheduler: Total steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

# === Enhanced Attention Analyzer ===
# (This class will be largely similar, but adapted to use the EnhancedModel and its tokenizer)
# ... Implementation of EnhancedAttentionAnalyzer, OEExtractorEnhanced, EnhancedVisualizer
# ... These will need to be careful about whether self.model is HF or Custom, and use the appropriate tokenizer
# ... and preprocessing.

class EnhancedAttentionAnalyzer:
    def __init__(self, config: Config, model_pl: EnhancedModel, device):
        self.config = config
        self.model_pl = model_pl.to(device)
        self.model_pl.eval()
        self.model_pl.freeze()
        self.tokenizer = model_pl.tokenizer # Get tokenizer from the model
        self.is_hf_model = model_pl.is_hf_model
        self.device = device
        
    @torch.no_grad()
    def get_word_attention_scores(self, texts: List[str]) -> List[Dict[str, float]]:
        batch_size = self.config.NLP_BATCH_SIZE
        all_word_scores = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing word attention scores", leave=False):
            batch_texts = texts[i:i+batch_size]
            if self.is_hf_model:
                batch_scores = self._process_attention_batch_hf(batch_texts, self.config.ATTENTION_LAYER)
            else:
                batch_scores = self._process_attention_batch_custom_nlp(batch_texts)
            all_word_scores.extend(batch_scores)
        return all_word_scores

    def _process_attention_batch_custom_nlp(self, batch_texts: List[str]) -> List[Dict[str, float]]:
        if not batch_texts: return []
        
        # Manual tokenization for custom NLP model
        batch_input_ids = []
        batch_attention_masks = [] # Mask for RNN attention, not padding mask from tokenizer
        original_tokenized_texts = []

        for text_str in batch_texts:
            # Preprocessing for custom tokenizer
            processed_text = preprocess_text_for_custom_nlp(text_str)
            token_ids = self.tokenizer.encode(processed_text, max_length=self.config.NLP_MAX_LENGTH)
            # RNN attention mask is based on non-pad tokens, up to where SEP is.
            # CLS and SEP are actual tokens for the RNN.
            try:
                # Find first PAD token to determine actual sequence length for attention
                # This assumes encode pads to max_length
                actual_len = token_ids.index(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id in token_ids else len(token_ids)
            except ValueError: # Should not happen if padded
                actual_len = len(token_ids)

            mask = [1] * actual_len + [0] * (self.config.NLP_MAX_LENGTH - actual_len)
            
            batch_input_ids.append(token_ids)
            batch_attention_masks.append(mask)
            original_tokenized_texts.append(tokenize_nltk(processed_text)) # For mapping scores back to words

        inputs_on_device = {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long).to(self.device),
            'attention_mask': torch.tensor(batch_attention_masks, dtype=torch.long).to(self.device)
        }
        
        outputs = self.model_pl.forward(inputs_on_device, output_attentions=True)
        
        # CustomNLPClassifier stores its single attention layer output in outputs.attentions[0]
        # Shape: [batch_size, seq_len (att_weights)]
        attention_batch = outputs.attentions[0].cpu().numpy() if outputs.attentions and outputs.attentions[0] is not None else None
        if attention_batch is None: # Fallback if no attention (e.g. attention disabled in custom model)
            attention_batch = np.ones((len(batch_texts), self.config.NLP_MAX_LENGTH)) / self.config.NLP_MAX_LENGTH

        batch_word_scores = []
        for i in range(len(batch_texts)):
            words_in_sample = original_tokenized_texts[i]
            # Attention scores correspond to input tokens (excluding PAD, but including CLS/SEP for RNN)
            # The attention weights from CustomNLPClassifier are over the sequence fed to the attention_layer.
            # This is typically the full sequence up to SEP (or max_length).
            sample_att_scores = attention_batch[i, :len(words_in_sample)+2] # +2 for CLS, SEP
            
            word_scores = defaultdict(list)
            # Map token-level attention to word-level.
            # For GRU/LSTM with its own attention, the scores are directly for each token position.
            # We need to align these with the original words.
            # The `att_weights` from `CustomNLPClassifier` are of shape (batch_size, seq_len)
            # where seq_len is the length of input to the attention mechanism.
            # CLS token (idx 0), word tokens, SEP token (idx len(words_in_sample)+1)
            
            # Simple mapping: assign score of token to the word. If a word is split by tokenizer (not by nltk here), this is tricky.
            # NLTK tokenize_nltk gives words. self.tokenizer.encode gives sub-words/tokens for HF, or word indices for custom.
            # For custom NLPTokenizer, tokens from encode are mostly 1-to-1 with words from tokenize_nltk(preprocess_text_for_custom_nlp(text))
            # We need to ignore CLS [0] and SEP [last_valid_idx] for word scores.
            
            # The att_weights are for the sequence *after* embedding, before weighted sum.
            # Their length should match the non-padded input to RNN.
            # batch_attention_masks[i] is the mask for the RNN's attention layer.
            
            # The att_weights from CustomNLPClassifier are of size (batch, seq_len)
            # where seq_len is the actual length of the input sequence to the attention layer.
            # This length is `inputs_on_device['attention_mask'].sum(dim=1)`
            
            current_attention_scores = attention_batch[i] # These are for the input_ids tokens
            
            # We need to map input_ids back to words carefully
            # For custom tokenizer, input_ids[1:-1] (excluding CLS/SEP) should map to words_in_sample
            # The attention scores (current_attention_scores) are for these input_ids positions
            
            # Assuming one score per token from custom attention
            # And custom tokenizer tokenizes into words (mostly)
            # The attention scores are for positions including CLS and SEP
            
            # Let's use the length of `words_in_sample` to guide
            # `current_attention_scores` has scores for CLS, word_1, ..., word_N, SEP, PAD...
            # We care about scores for word_1 to word_N
            
            # The `att_weights` from `CustomNLPClassifier` is of shape (batch_size, seq_len)
            # seq_len is the length of sequence passed to the attention_layer
            # This length is determined by the `attention_mask` (non-padded elements)
            
            # The `words_in_sample` are from `tokenize_nltk(preprocess_text_for_custom_nlp(text_str))`
            # The `token_ids` from `self.tokenizer.encode` are `[CLS, word_id1, word_id2, ..., SEP, PAD...]`
            # The attention scores `current_attention_scores` are for these `token_ids` positions
            # (specifically, for the hidden states corresponding to these tokens)

            # We want to assign scores to `words_in_sample`.
            # The scores for `words_in_sample[j]` should come from `current_attention_scores[j+1]` (due to CLS at start)
            
            final_scores_dict = {}
            for word_idx, word in enumerate(words_in_sample):
                # score_idx corresponds to the position of the word's token in the input_ids list
                score_idx = word_idx + 1 # +1 to skip CLS token's score
                if score_idx < len(current_attention_scores):
                     final_scores_dict[word] = float(current_attention_scores[score_idx]) # Take the score for that token position
                else: # Should not happen if lengths are correct
                     final_scores_dict[word] = 0.0
            batch_word_scores.append(final_scores_dict)
            
        return batch_word_scores

    def _process_attention_batch_hf(self, batch_texts: List[str], layer_idx: int) -> List[Dict[str, float]]:
        if not batch_texts: return []
        
        processed_texts_for_hf = [preprocess_text_for_hf(text) for text in batch_texts]
        inputs = self.tokenizer(
            processed_texts_for_hf, return_tensors='pt', truncation=True,
            max_length=self.config.NLP_MAX_LENGTH, padding=True, return_offsets_mapping=True
        )
        offset_mappings_np = inputs.pop('offset_mapping').cpu().numpy()
        input_ids_np = inputs['input_ids'].cpu().numpy()
        
        inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model_pl.model(**inputs_on_device) # model_pl.model is the HF model
        # HF model attentions: tuple of (batch_size, num_heads, seq_len, seq_len)
        # We need CLS token's attention to other tokens.
        attentions_batch_all_layers = outputs.attentions # This is a tuple of all layers' attentions
        
        # attentions_batch for the specified layer
        # Shape: (batch_size, num_heads, sequence_length, sequence_length)
        attentions_batch_layer = attentions_batch_all_layers[layer_idx].cpu().numpy()

        batch_word_scores = []
        for i in range(len(batch_texts)):
            # Mean over heads, then take CLS token's attention distribution
            # CLS token is at index 0
            # cls_token_attentions_to_sequence = mean over heads of attentions_batch_layer[i, :, 0, :]
            # Shape of cls_token_attentions_to_sequence: (sequence_length,)
            cls_token_attentions_to_sequence = np.mean(attentions_batch_layer[i, :, 0, :], axis=0)
            
            batch_word_scores.append(
                self._extract_word_scores_from_cls_attention_hf(
                    cls_token_attentions_to_sequence,
                    input_ids_np[i],
                    offset_mappings_np[i],
                    processed_texts_for_hf[i] # Original text fed to tokenizer
                )
            )
        del inputs, inputs_on_device, outputs, attentions_batch_all_layers
        gc.collect()
        return batch_word_scores

    def _extract_word_scores_from_cls_attention_hf(self, cls_attentions_np, # score for each token from CLS
                                                 input_ids_sample, offset_mapping_sample, original_text_sample):
        word_scores = defaultdict(list)
        current_word_tokens_indices = [] # Store indices of tokens belonging to current word
        last_word_end_offset = 0

        for token_idx, (token_id, offset) in enumerate(zip(input_ids_sample, offset_mapping_sample)):
            # Skip special tokens (CLS, SEP, PAD) and tokens with no span (offset[0]==offset[1])
            if token_id in self.tokenizer.all_special_ids or offset[0] == offset[1]:
                # If we were accumulating tokens for a word, process it now if this is a separator
                if current_word_tokens_indices:
                    start_char = offset_mapping_sample[current_word_tokens_indices[0]][0]
                    end_char = offset_mapping_sample[current_word_tokens_indices[-1]][1]
                    word_text = original_text_sample[start_char:end_char]
                    avg_score_for_word = np.mean(cls_attentions_np[current_word_tokens_indices])
                    if word_text.strip():
                        word_scores[word_text.strip()].append(avg_score_for_word)
                    current_word_tokens_indices = []
                last_word_end_offset = offset[1] # even for special tokens, update this
                continue

            # Check if current token starts a new word or continues previous one
            # A new word starts if offset[0] is not equal to last_word_end_offset
            # (i.e., there's a space or it's the first word)
            if offset[0] != last_word_end_offset and current_word_tokens_indices:
                # Process previously accumulated word
                start_char = offset_mapping_sample[current_word_tokens_indices[0]][0]
                end_char = offset_mapping_sample[current_word_tokens_indices[-1]][1]
                word_text = original_text_sample[start_char:end_char]
                avg_score_for_word = np.mean(cls_attentions_np[current_word_tokens_indices])
                if word_text.strip():
                    word_scores[word_text.strip()].append(avg_score_for_word)
                current_word_tokens_indices = [] # Reset for new word
            
            current_word_tokens_indices.append(token_idx)
            last_word_end_offset = offset[1]

        # Process any remaining word at the end of sequence (if not ended by special token)
        if current_word_tokens_indices:
            start_char = offset_mapping_sample[current_word_tokens_indices[0]][0]
            end_char = offset_mapping_sample[current_word_tokens_indices[-1]][1]
            word_text = original_text_sample[start_char:end_char]
            avg_score_for_word = np.mean(cls_attentions_np[current_word_tokens_indices])
            if word_text.strip():
                word_scores[word_text.strip()].append(avg_score_for_word)
        
        # If a word appeared multiple times, average its scores
        return {word: np.mean(scores) for word, scores in word_scores.items()}


    def extract_top_attention_words(self, word_scores_dict: Dict[str, float]) -> List[str]:
        if not word_scores_dict: return []
        
        sorted_words = sorted(word_scores_dict.items(), key=lambda x: x[1], reverse=True)
        num_words = len(sorted_words)
        n_top = max(self.config.MIN_TOP_WORDS, math.ceil(num_words * self.config.ATTENTION_TOP_PERCENT))
        
        try: stop_words = set(stopwords.words('english'))
        except: stop_words = {'a', 'an', 'the', 'is', 'was', 'to', 'of', 'for', 'on', 'in', 'at', 'and', 'it', 'this', 'that'} # Basic fallback
        
        top_words_filtered = [word for word, score in sorted_words[:n_top] 
                             if word.lower() not in stop_words and len(word) > 1 and word.isalnum()] # Only alphanumeric
        
        return top_words_filtered if top_words_filtered else [word for word, score in sorted_words[:n_top] if len(word)>1 and word.isalnum()][:self.config.MIN_TOP_WORDS] # Fallback if all are stopwords

    def process_full_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Processing ID dataset for attention analysis (to derive OE data)...")
        # For NLP mode, we always process the full ID dataset to get attention words for OE derivation.
        # The 'exclude_class' logic was for Syslog's 'unknown' class. Not relevant here.
        df_for_analysis = df.copy()
        texts = df_for_analysis[self.config.NLP_ID_DATASETS[self.config.CURRENT_NLP_ID_DATASET]['text_column']].tolist()
        
        all_word_scores = self.get_word_attention_scores(texts)
        
        all_top_words, masked_texts_list = [], []
        text_col_name = self.config.NLP_ID_DATASETS[self.config.CURRENT_NLP_ID_DATASET]['text_column']

        for i, (text, word_scores) in tqdm(enumerate(zip(texts, all_word_scores)), total=len(texts), desc="Extracting top words & masking"):
            top_words = self.extract_top_attention_words(word_scores)
            all_top_words.append(top_words)
            # Use the original text from dataframe for masking
            masked_texts_list.append(create_masked_sentence(df_for_analysis.iloc[i][text_col_name], top_words))
        
        result_df = df_for_analysis.copy() # Start with the df that was analyzed
        result_df['top_attention_words'] = all_top_words
        result_df[self.config.TEXT_COLUMN_IN_OE_FILES] = masked_texts_list # Store masked text
        return result_df

# (OEExtractorEnhanced, Visualizer, OSR Components, Main Pipeline will follow a similar pattern of adaptation)
# This is a very large refactoring. I'll pause here to ensure this direction is correct.
# The next steps would be to implement OEExtractorEnhanced, EnhancedVisualizer, the OSR specific dataloaders/models,
# and finally the main pipeline logic, all tailored for the NLP-only mode.

# Placeholder for the rest of the classes to make the script runnable for syntax checks
class MaskedTextDatasetForMetrics(TorchDataset): # Simplified
    def __init__(self, texts: List[str], tokenizer, max_length: int, is_hf_tokenizer: bool):
        self.texts = [str(t) if pd.notna(t) else "__EMPTY__" for t in texts] # Handle NaN/None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_hf_tokenizer = is_hf_tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.is_hf_tokenizer:
            processed_text = preprocess_text_for_hf(text)
            encoding = self.tokenizer(processed_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            return {'input_ids': encoding['input_ids'].squeeze(0), 'attention_mask': encoding['attention_mask'].squeeze(0)}
        else:
            processed_text = preprocess_text_for_custom_nlp(text)
            token_ids = self.tokenizer.encode(processed_text, max_length=self.max_length)
            attention_mask = [1 if tid != self.tokenizer.pad_token_id else 0 for tid in token_ids]
            return {'input_ids': torch.tensor(token_ids, dtype=torch.long), 
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long)}

class OEExtractorEnhanced: # Simplified structure
    def __init__(self, config: Config, model_pl: EnhancedModel, device):
        self.config = config
        self.model_pl = model_pl.to(device); self.model_pl.eval(); self.model_pl.freeze()
        self.tokenizer = model_pl.tokenizer
        self.is_hf_model = model_pl.is_hf_model
        self.device = device

    @torch.no_grad()
    def extract_attention_metrics(self, dataloader: DataLoader, original_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[np.ndarray]]:
        attention_metrics_list = []
        features_list = [] # For CLS token or equivalent
        
        for batch_encodings in tqdm(dataloader, desc="Extracting OE metrics from masked", leave=False):
            batch_on_device = {k: v.to(self.device) for k, v in batch_encodings.items()}
            
            # Forward pass to get attentions (from masked text) and features
            # output_attentions=True for HF, output_features=True for custom
            if self.is_hf_model:
                outputs = self.model_pl.model(
                    input_ids=batch_on_device['input_ids'],
                    attention_mask=batch_on_device['attention_mask'],
                    output_attentions=True, # Need attentions from the masked text processing
                    output_hidden_states=True # Need features for t-SNE
                )
                # For HF: attentions is a tuple of (batch, heads, seq, seq) per layer
                # We use last layer's CLS token attention to sequence
                att_batch_layer = outputs.attentions[self.config.ATTENTION_LAYER].cpu().numpy()
                # Features: last hidden state, CLS token: (batch, hidden_size)
                features_batch = outputs.hidden_states[-1][:, 0, :].cpu().numpy() 
            else: # Custom model
                outputs = self.model_pl.model( # model_pl.model is CustomNLPClassifier
                    input_ids=batch_on_device['input_ids'],
                    attention_mask=batch_on_device['attention_mask'], # This mask is for RNN attention
                    output_attentions=True,
                    output_features=True
                )
                # Custom: attentions[0] is (batch, seq_len_att)
                att_batch_layer = outputs.attentions[0].cpu().numpy() if outputs.attentions and outputs.attentions[0] is not None else None
                if att_batch_layer is None: # Fallback
                    att_batch_layer = np.ones((batch_on_device['input_ids'].shape[0], batch_on_device['input_ids'].shape[1])) / batch_on_device['input_ids'].shape[1]

                # Custom: hidden_states[0] is (batch, hidden_dim*2)
                features_batch = outputs.hidden_states[0].cpu().numpy() if outputs.hidden_states else \
                                 np.zeros((batch_on_device['input_ids'].shape[0], self.config.NLP_HIDDEN_DIM *2))


            features_list.extend(list(features_batch))
            input_ids_batch_cpu = batch_on_device['input_ids'].cpu().numpy()

            for i in range(len(input_ids_batch_cpu)):
                sample_input_ids = input_ids_batch_cpu[i]
                if self.is_hf_model:
                    # Use CLS token's attention to other tokens from the specified layer
                    cls_token_att_to_seq = np.mean(att_batch_layer[i, :, 0, :], axis=0)
                    metrics = self._compute_attention_metrics_hf(cls_token_att_to_seq, sample_input_ids)
                else: # Custom model
                    # att_batch_layer is already (batch, seq_len_att)
                    sample_att_scores = att_batch_layer[i]
                    metrics = self._compute_attention_metrics_custom(sample_att_scores, sample_input_ids)
                attention_metrics_list.append(metrics)
        
        return pd.DataFrame(attention_metrics_list), features_list

    def _compute_attention_metrics_hf(self, cls_attentions_np, input_ids_sample): # For HF model attentions
        valid_indices = np.where(
            (input_ids_sample != self.tokenizer.pad_token_id) &
            (input_ids_sample != self.tokenizer.cls_token_id) &
            (input_ids_sample != self.tokenizer.sep_token_id)
        )[0]
        
        if len(valid_indices) == 0: return {'max_attention': 0, 'top_k_avg_attention': 0, 'attention_entropy': 0}
        
        # cls_attentions_np are scores for each token FROM the CLS token.
        # We only care about these scores for non-special, non-pad tokens.
        token_attentions = cls_attentions_np[valid_indices]
        
        max_att = np.max(token_attentions) if len(token_attentions) > 0 else 0
        k = min(self.config.TOP_K_ATTENTION, len(token_attentions))
        top_k_avg_att = np.mean(np.sort(token_attentions)[-k:]) if k > 0 else 0
        
        # Softmax to get probabilities for entropy calculation
        probs = F.softmax(torch.tensor(token_attentions), dim=0).numpy()
        att_entropy = entropy(probs) if len(probs) > 1 else 0.0
        return {'max_attention': max_att, 'top_k_avg_attention': top_k_avg_att, 'attention_entropy': att_entropy}

    def _compute_attention_metrics_custom(self, attention_scores_sample, input_ids_sample): # For custom model attentions
        # attention_scores_sample are from the model's internal attention layer (e.g., on RNN outputs)
        # input_ids_sample are the token IDs.
        # We need to consider scores for tokens that are not PAD, UNK, CLS, SEP
        valid_indices = np.where(
            (input_ids_sample != self.tokenizer.pad_token_id) &
            (input_ids_sample != self.tokenizer.unk_token_id) &
            (input_ids_sample != self.tokenizer.cls_token_id) & # Exclude CLS/SEP from metric calculation itself
            (input_ids_sample != self.tokenizer.sep_token_id)
        )[0]

        if len(valid_indices) == 0: return {'max_attention': 0, 'top_k_avg_attention': 0, 'attention_entropy': 0}

        # attention_scores_sample has one score per input token position
        token_attentions = attention_scores_sample[valid_indices]

        max_att = np.max(token_attentions) if len(token_attentions) > 0 else 0
        k = min(self.config.TOP_K_ATTENTION, len(token_attentions))
        top_k_avg_att = np.mean(np.sort(token_attentions)[-k:]) if k > 0 else 0
        
        # These attention_scores are already softmaxed in the custom model
        # If they are not, they should be softmaxed first. Assuming they are probabilities.
        # The `att_weights` from `CustomNLPClassifier` are already softmaxed.
        probs = token_attentions / (np.sum(token_attentions) + 1e-9) # Re-normalize just in case
        att_entropy = entropy(probs) if len(probs) > 1 and np.sum(probs) > 1e-6 else 0.0
        return {'max_attention': max_att, 'top_k_avg_attention': top_k_avg_att, 'attention_entropy': att_entropy}


    def compute_removed_word_attention(self, df: pd.DataFrame, attention_analyzer: EnhancedAttentionAnalyzer) -> pd.DataFrame:
        print("Computing removed word attention scores...")
        text_col_name = Config.NLP_ID_DATASETS[self.config.CURRENT_NLP_ID_DATASET]['text_column']
        if 'top_attention_words' not in df.columns or text_col_name not in df.columns:
            df['removed_avg_attention'] = 0.0
            return df
        
        # We need original texts to get original word attentions
        texts_for_original_attention = df[text_col_name].tolist()
        original_word_attentions_list = attention_analyzer.get_word_attention_scores(texts_for_original_attention)
            
        removed_attentions = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing removed_avg_attention", leave=False):
            top_words = safe_literal_eval(row['top_attention_words'])
            
            if top_words and idx < len(original_word_attentions_list):
                word_scores_dict_original = original_word_attentions_list[idx]
                # Get the attention scores of the words that were removed (top_words) from the *original* text's attention map
                removed_scores = [word_scores_dict_original.get(word.lower(), # Compare lowercased
                                  word_scores_dict_original.get(word, 0)) # Try original case if lower not found
                                  for word in top_words]
                removed_attentions.append(np.mean([s for s in removed_scores if s is not None]) if any(s is not None for s in removed_scores) else 0.0)
            else:
                removed_attentions.append(0.0)
        
        df['removed_avg_attention'] = removed_attentions
        return df

    def extract_oe_datasets(self, df_with_metrics: pd.DataFrame) -> None: # Extracts OE from ID data
        print("Extracting OE datasets from ID data using calculated metrics...")
        # Always use all available data in df_with_metrics for OE extraction
        # (as it's derived from the ID set)
        
        for metric, settings in self.config.METRIC_SETTINGS.items():
            if metric not in df_with_metrics.columns:
                print(f"Skipping OE for {metric} - column not found.")
                continue
            self._extract_single_metric_oe(df_with_metrics, metric, settings)
        
        self._extract_sequential_filtering_oe(df_with_metrics)

    def _extract_single_metric_oe(self, df: pd.DataFrame, metric: str, settings: dict):
        scores = np.nan_to_num(df[metric].values, nan=0.0) # Handle potential NaNs
        if settings['mode'] == 'higher': # Higher score is more "outlier-like" for this metric
            threshold = np.percentile(scores, 100 - settings['percentile'])
            selected_indices = np.where(scores >= threshold)[0]
        else: # Lower score is more "outlier-like"
            threshold = np.percentile(scores, settings['percentile'])
            selected_indices = np.where(scores <= threshold)[0]
        
        if len(selected_indices) > 0:
            # Save only the masked text for OSR training
            oe_df_simple = df.iloc[selected_indices][[self.config.TEXT_COLUMN_IN_OE_FILES]].copy()
            # Save extended info for analysis
            extended_cols = [self.config.TEXT_COLUMN_IN_OE_FILES, 
                             Config.NLP_ID_DATASETS[self.config.CURRENT_NLP_ID_DATASET]['text_column'], 
                             'top_attention_words', metric]
            extended_cols = [col for col in extended_cols if col in df.columns] # Ensure cols exist
            oe_df_extended = df.iloc[selected_indices][extended_cols].copy()

            mode_desc = f"{settings['mode']}{settings['percentile']}pct"
            base_filename = f"derived_oe_{self.config.CURRENT_NLP_ID_DATASET}_{metric}_{mode_desc}"
            
            oe_df_simple.to_csv(os.path.join(self.config.OE_DATA_DIR, f"{base_filename}.csv"), index=False)
            oe_df_extended.to_csv(os.path.join(self.config.OE_DATA_DIR, f"{base_filename}_extended.csv"), index=False)
            print(f"Saved derived OE dataset ({len(oe_df_simple)} samples) for {metric} {mode_desc}")

    def _extract_sequential_filtering_oe(self, df: pd.DataFrame):
        print("Applying sequential filtering for derived OE extraction...")
        current_df = df.copy()
        
        filter_desc_parts = []
        for step, (metric, settings) in enumerate(self.config.FILTERING_SEQUENCE):
            if metric not in current_df.columns:
                print(f"Seq. Filter Step {step+1}: {metric} not found. Skipping.")
                continue
            if current_df.empty:
                print("No samples left for sequential filtering.")
                break

            scores = np.nan_to_num(current_df[metric].values, nan=0.0)
            if settings['mode'] == 'higher':
                threshold = np.percentile(scores, 100 - settings['percentile'])
                current_df = current_df[scores >= threshold]
            else:
                threshold = np.percentile(scores, settings['percentile'])
                current_df = current_df[scores <= threshold]
            
            filter_desc_parts.append(f"{metric}_{settings['mode']}{settings['percentile']}")
            print(f"Sequential Filter {step+1} ({metric} {settings['mode']} {settings['percentile']}%): {len(current_df)} samples remaining")

        if not current_df.empty:
            oe_df_simple = current_df[[self.config.TEXT_COLUMN_IN_OE_FILES]].copy()
            
            extended_cols = [self.config.TEXT_COLUMN_IN_OE_FILES, 
                             Config.NLP_ID_DATASETS[self.config.CURRENT_NLP_ID_DATASET]['text_column'], 
                             'top_attention_words']
            extended_cols.extend([m for m, _ in self.config.FILTERING_SEQUENCE if m in df.columns])
            oe_df_extended = current_df[extended_cols].copy()

            filter_desc = "_then_".join(filter_desc_parts)
            base_filename = f"derived_oe_{self.config.CURRENT_NLP_ID_DATASET}_sequential_{filter_desc}"
            
            oe_df_simple.to_csv(os.path.join(self.config.OE_DATA_DIR, f"{base_filename}.csv"), index=False)
            oe_df_extended.to_csv(os.path.join(self.config.OE_DATA_DIR, f"{base_filename}_extended.csv"), index=False)
            print(f"Saved sequential derived OE dataset ({len(oe_df_simple)} samples)")
        else:
            print("No samples selected by sequential filtering for derived OE.")


class EnhancedVisualizer: # Simplified
    def __init__(self, config: Config):
        self.config = config
    
    def plot_metric_distribution(self, scores: np.ndarray, metric_name: str, title: str, save_path: str):
        if len(scores) == 0: return
        plt.figure(figsize=(10, 6))
        if SNS_AVAILABLE: sns.histplot(scores, bins=50, kde=True, stat='density')
        else: plt.hist(scores, bins=50, density=True, alpha=0.7)
        plt.title(title); plt.xlabel(metric_name); plt.ylabel('Density'); plt.grid(alpha=0.3)
        mean_val = np.mean(scores); plt.axvline(mean_val, color='r', ls='--', label=f'Mean: {mean_val:.4f}')
        plt.legend(); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
        # print(f"Dist plot saved: {save_path}")

    def plot_tsne(self, features: np.ndarray, labels: np.ndarray, title: str, save_path: str,
                highlight_indices: Optional[np.ndarray] = None, highlight_label: str = 'OE Candidate',
                class_names: Optional[Dict] = None):
        """
        특징 공간의 t-SNE 시각화를 생성합니다.
        
        Args:
            features: 특징 행렬 (n_samples, n_features)
            labels: 클래스 레이블 (n_samples,)
            title: 플롯 제목
            save_path: 그림을 저장할 경로
            highlight_indices: 강조할 포인트 인덱스
            highlight_label: 강조된 포인트의 레이블
            class_names: 숫자 레이블을 클래스 이름에 매핑하는 딕셔너리
        """
        if len(features) == 0 or features.shape[0] <= 1: 
            print(f"Skipping t-SNE: Not enough samples ({len(features)}) for visualization.")
            return  # TSNE에는 1개 이상의 샘플이 필요
        
        print(f"Running t-SNE for {features.shape[0]} samples...")
        perplexity_val = min(30.0, float(features.shape[0] - 1))
        if perplexity_val <= 1:  # perplexity는 1보다 커야 함
            print(f"Skipping t-SNE: Not enough samples ({features.shape[0]}) for perplexity {perplexity_val}.")
            return
        
        try:
            tsne = TSNE(
                n_components=2, 
                random_state=self.config.RANDOM_STATE, 
                perplexity=perplexity_val,
                max_iter=1000, 
                init='pca', 
                learning_rate='auto'
            )
            tsne_results = tsne.fit_transform(features)
        except Exception as e:
            print(f"Error in t-SNE: {e}. Skipping plot.")
            return
        
        df_tsne = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
        df_tsne['label_val'] = labels  # 레이블 값에 일반 이름 사용
        df_tsne['is_highlighted'] = False
        
        if highlight_indices is not None and len(highlight_indices) > 0:  # highlight_indices가 비어 있지 않은지 확인
            valid_indices = highlight_indices[highlight_indices < len(df_tsne)]
            if len(valid_indices) > 0:
                df_tsne.loc[valid_indices, 'is_highlighted'] = True  # 인덱스가 유효한지 확인
        
        plt.figure(figsize=(12, 8))
        
        # 고유 레이블 가져오기 및 컬러맵 준비
        unique_label_vals = sorted(df_tsne['label_val'].unique())
        
        # 다중 클래스에 더 구분하기 쉬운 컬러맵 생성
        if len(unique_label_vals) > 10:
            # 많은 클래스의 경우 연속적인 컬러맵 사용
            cmap = plt.cm.get_cmap('tab20' if len(unique_label_vals) <= 20 else 'viridis', len(unique_label_vals))
            colors = [cmap(i) for i in range(len(unique_label_vals))]
        else:
            # 더 적은 클래스에는 구분되는 색상 사용
            distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            colors = [distinct_colors[i % len(distinct_colors)] for i in range(len(unique_label_vals))]
        
        # 각 클래스 플롯
        for i, label_val_item in enumerate(unique_label_vals):
            subset = df_tsne[(df_tsne['label_val'] == label_val_item) & (~df_tsne['is_highlighted'])]
            if len(subset) > 0:
                c_name = class_names.get(label_val_item, f'Label {label_val_item}') if class_names else f'Label {label_val_item}'
                plt.scatter(
                    subset['tsne1'], 
                    subset['tsne2'], 
                    color=colors[i],
                    label=c_name, 
                    alpha=0.7, 
                    s=30,
                    edgecolors='none'
                )
        
        # 강조된 포인트가 있으면 플롯
        if highlight_indices is not None and len(df_tsne[df_tsne['is_highlighted']]) > 0:
            highlight_subset = df_tsne[df_tsne['is_highlighted']]
            plt.scatter(
                highlight_subset['tsne1'], 
                highlight_subset['tsne2'],
                color='red', 
                marker='x', 
                s=80, 
                label=highlight_label, 
                alpha=0.9
            )
        
        plt.title(title, fontsize=14)
        plt.xlabel("t-SNE Dimension 1", fontsize=12)
        plt.ylabel("t-SNE Dimension 2", fontsize=12)
        plt.grid(alpha=0.2)
        
        # 범례 처리: 많은 클래스는 외부에, 적은 클래스는 내부에
        if len(unique_label_vals) > 5:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
            plt.tight_layout(rect=[0, 0, 0.85, 1])  # 범례 공간 확보
        else:
            plt.legend(loc='best', fontsize=10)
            plt.tight_layout()
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"t-SNE plot saved: {save_path}")

    def visualize_all_metrics(self, df_with_metrics: pd.DataFrame):
        metric_cols = ['max_attention', 'top_k_avg_attention', 'attention_entropy', 'removed_avg_attention']
        for metric in metric_cols:
            if metric in df_with_metrics.columns and not df_with_metrics[metric].isnull().all():
                self.plot_metric_distribution(
                    df_with_metrics[metric].dropna().values, metric, f'Dist. of {metric} (from masked texts)',
                    os.path.join(self.config.VIS_DIR, f'{metric}_distribution.png')
                )

    def visualize_oe_candidates(self, df_with_metrics: pd.DataFrame, features: List[np.ndarray],
                                id_label2id: dict, id_id2label: dict): # From ID dataset
        if not features or len(features) != len(df_with_metrics): return

        # Labels for t-SNE are the original ID labels
        tsne_labels_numeric = df_with_metrics['label_id'].values # Assuming 'label_id' column from DataModule
        
        # Class names for legend
        class_names_viz = {id_val: str(name) for id_val, name in id_id2label.items()}
        class_names_viz[-1] = "OOD/Unknown" # Generic, though not expected in ID data metrics

        # Features should be a single numpy array
        features_np = np.array(features)
        if features_np.ndim == 1: # If features_list was not a list of arrays but list of lists/numbers
            try:
                features_np = np.vstack(features)
            except:
                print("Could not stack features for t-SNE visualization.")
                return


        # Plot for each OE extraction method
        for metric, settings in self.config.METRIC_SETTINGS.items():
            if metric not in df_with_metrics.columns: continue
            
            scores = np.nan_to_num(df_with_metrics[metric].values, nan=0.0)
            if settings['mode'] == 'higher':
                threshold = np.percentile(scores, 100 - settings['percentile'])
                oe_indices = np.where(scores >= threshold)[0]
            else:
                threshold = np.percentile(scores, settings['percentile'])
                oe_indices = np.where(scores <= threshold)[0]
            
            mode_desc = f"{settings['mode']}{settings['percentile']}%"
            title = f't-SNE of ID data, highlighting derived OE candidates by {metric} ({mode_desc})'
            save_path = os.path.join(self.config.VIS_DIR, f'tsne_derived_oe_cand_{metric}_{mode_desc}.png')
            self.plot_tsne(features_np, tsne_labels_numeric, title, save_path,
                             highlight_indices=oe_indices, highlight_label=f'Derived OE ({metric})',
                             class_names=class_names_viz)
        
        # Plot for sequential filtering (if used)
        # (Reconstruct the sequential filtering logic to get final_indices_seq)
        # ... This part can be complex to reimplement here just for viz, consider if essential.

# OSR functions (evaluate_nlp_osr, plot_osr_...) would largely remain the same,
# but ensure they are called with correct model, tokenizer, and data.
# The main pipeline needs the biggest overhaul for the new experiment structure.

# --- Main Pipeline Class (Heavily Modified for NLP Focus) ---
# (The rest of the script including EnhancedOEPipeline, main() will be provided in the next part
# due to length constraints. It will incorporate these changes and focus on the NLP experiment flow.)
# ```

# This is the first part of the refactoring, covering the configuration, data loaders, tokenizers, and the base NLP models, along with initial versions of the analyzer, extractor, and visualizer. The core logic for the pipeline execution, especially the OSR experiments, will follow.

# Let me know if this initial direction aligns with your expectations before I generate the rest of the `EnhancedOEPipeline` and `main` function. The key changes are:
# *   Syslog removal.
# *   RoBERTa-base as a model option throughout.
# *   Specific data loaders for ID, standard OE (WikiText2), and OOD Test (WMT16 etc.).
# *   OE data derived from ID data using attention metrics.
# *   Restructured OSR experiments to compare these different OE sources.

# Continuing with the rest ofr the `EnhancedOEPipeline` and `main` function:

# ```python
# === OSR 평가 함수 (NLP) ===
# (evaluate_nlp_osr and plotting functions like plot_confidence_histograms_osr etc.
# are assumed to be mostly correct from the original script, but ensure they take the
# OSR model and appropriate tokenizer if it differs from base classifier)

# Placeholder for OSR evaluation and plotting functions (can be copied from original oe2.py or your current script)
def evaluate_nlp_osr(model: nn.Module, id_loader: Optional[DataLoader], ood_loader: Optional[DataLoader], 
                     device: torch.device, temperature: float = 1.0, threshold_percentile: float = 5.0, 
                     return_data: bool = False, is_hf_osr_model: bool = False, 
                     osr_tokenizer_for_custom: Optional[NLPTokenizer] = None) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, np.ndarray]]]:
    model.eval()
    id_logits_all, id_scores_all, id_labels_true, id_labels_pred, id_features_all = [], [], [], [], []
    ood_logits_all, ood_scores_all, ood_features_all = [], [], []

    # ID 데이터가 있는 경우 처리
    if id_loader is not None:
        with torch.no_grad():
            for batch in tqdm(id_loader, desc="Evaluating ID for OSR", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                # --- 레이블 접근 방식 ---
                if 'labels' in batch:
                    labels_tensor = batch['labels'] # CPU에 유지
                elif 'label' in batch:
                    labels_tensor = batch['label']
                else:
                    # ID 데이터에는 레이블이 항상 있어야 함
                    raise KeyError("ID batch for OSR evaluation does not contain 'label' or 'labels' key.")

                if is_hf_osr_model or not hasattr(model, 'forward_features'):
                     logits, features = model(input_ids, attention_mask, output_features=True)
                else:
                     logits, features = model.forward_features(input_ids, attention_mask)

                softmax_probs = F.softmax(logits / temperature, dim=1)
                max_probs, preds = softmax_probs.max(dim=1)

                id_logits_all.append(logits.cpu())
                id_scores_all.append(max_probs.cpu())
                id_labels_true.extend(labels_tensor.numpy()) # labels_tensor 사용
                id_labels_pred.extend(preds.cpu().numpy())
                if features is not None: id_features_all.append(features.cpu())

    # OOD 데이터가 있는 경우 처리
    if ood_loader:
        with torch.no_grad():
            for batch in tqdm(ood_loader, desc="Evaluating OOD for OSR", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                if is_hf_osr_model or not hasattr(model, 'forward_features'):
                     logits, features = model(input_ids, attention_mask, output_features=True)
                else:
                     logits, features = model.forward_features(input_ids, attention_mask)

                softmax_probs = F.softmax(logits / temperature, dim=1)
                max_probs, _ = softmax_probs.max(dim=1)
                ood_logits_all.append(logits.cpu())
                ood_scores_all.append(max_probs.cpu())
                if features is not None: ood_features_all.append(features.cpu())

    # 결과 조합 및 처리
    id_scores = torch.cat(id_scores_all).numpy() if id_scores_all else np.array([])
    id_features_list = [f for f in id_features_all if f is not None] # None 필터링
    id_features = torch.cat(id_features_list).numpy() if id_features_list and len(id_features_list) > 0 else np.array([])
    
    id_labels_true_np = np.array(id_labels_true) if id_labels_true else np.array([])
    id_labels_pred_np = np.array(id_labels_pred) if id_labels_pred else np.array([])

    ood_scores = torch.cat(ood_scores_all).numpy() if ood_scores_all else np.array([])
    ood_features_list = [f for f in ood_features_all if f is not None] # None 필터링
    ood_features = torch.cat(ood_features_list).numpy() if ood_features_list and len(ood_features_list) > 0 else np.array([])
    
    results = {"Closed_Set_Accuracy": 0.0, "F1_Macro": 0.0, "AUROC": 0.0, "FPR@TPR90": 1.0, 
               "AUPR_In": 0.0, "AUPR_Out": 0.0, "DetectionAccuracy": 0.0, "OSCR": 0.0, "Threshold_Used": 0.0}
    all_data_dict = {"id_scores": id_scores, "ood_scores": ood_scores, "id_labels_true": id_labels_true_np, 
                     "id_labels_pred": id_labels_pred_np, "id_features": id_features, "ood_features": ood_features}

    if len(id_labels_true_np) == 0: return (results, all_data_dict) if return_data else results
    results["Closed_Set_Accuracy"] = accuracy_score(id_labels_true_np, id_labels_pred_np)
    results["F1_Macro"] = f1_score(id_labels_true_np, id_labels_pred_np, average='macro', zero_division=0)

    if len(ood_scores) == 0: 
        print("Warning: No OOD scores for OSR AUROC calculation. Skipping OSR metrics.")
        return (results, all_data_dict) if return_data else results # OSR 관련 메트릭 없이 반환
    
    y_true_osr = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_scores_osr = np.concatenate([id_scores, ood_scores])
    valid_indices = ~np.isnan(y_scores_osr)
    y_true_osr, y_scores_osr = y_true_osr[valid_indices], y_scores_osr[valid_indices]

    if len(np.unique(y_true_osr)) < 2: print("Warning: Only one class type for OSR AUROC.")
    else:
        results["AUROC"] = roc_auc_score(y_true_osr, y_scores_osr)
        fpr, tpr, thresholds_roc = roc_curve(y_true_osr, y_scores_osr)
        idx_tpr90 = np.where(tpr >= 0.90)[0]
        results["FPR@TPR90"] = fpr[idx_tpr90[0]] if len(idx_tpr90) > 0 else 1.0
        precision_in, recall_in, _ = precision_recall_curve(y_true_osr, y_scores_osr, pos_label=1)
        results["AUPR_In"] = auc(recall_in, precision_in)
        precision_out, recall_out, _ = precision_recall_curve(1 - y_true_osr, 1 - y_scores_osr, pos_label=1) # OOD를 양성으로
        results["AUPR_Out"] = auc(recall_out, precision_out)

    chosen_threshold = np.percentile(id_scores, threshold_percentile) if len(id_scores) > 0 else 0.5
    results["Threshold_Used"] = chosen_threshold
    id_preds_binary = (id_scores >= chosen_threshold).astype(int) # 올바르게 ID로 분류
    ood_preds_binary = (ood_scores < chosen_threshold).astype(int) # 올바르게 OOD로 분류
    if (len(id_scores) + len(ood_scores)) > 0:
        results["DetectionAccuracy"] = (np.sum(id_preds_binary) + np.sum(ood_preds_binary)) / (len(id_scores) + len(ood_scores))
    
    known_mask = (id_scores >= chosen_threshold)
    ccr = accuracy_score(id_labels_true_np[known_mask], id_labels_pred_np[known_mask]) if np.sum(known_mask) > 0 else 0.0
    oer = np.sum(ood_scores >= chosen_threshold) / len(ood_scores) if len(ood_scores) > 0 else 0.0 # OOD를 ID로 잘못 분류
    results["OSCR"] = ccr * (1.0 - oer)

    return (results, all_data_dict) if return_data else results

def plot_confidence_histograms_osr(id_scores, ood_scores, title, save_path):
    plt.figure(figsize=(8, 5))
    sns.histplot(id_scores, color="blue", label='ID Scores', kde=True, stat="density", common_norm=False)
    if ood_scores is not None and len(ood_scores) > 0:
        sns.histplot(ood_scores, color="red", label='OOD Scores', kde=True, stat="density", common_norm=False)
    plt.title(title); plt.xlabel('Confidence Score'); plt.ylabel('Density')
    plt.legend(); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_roc_curve_osr(id_scores, ood_scores, title, save_path):
    if ood_scores is None or len(ood_scores) == 0: return
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title(title)
    plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_tsne_osr(id_features, ood_features, title, save_path, seed=42):
    if (id_features is None or len(id_features) == 0) and \
       (ood_features is None or len(ood_features) == 0):
        return
    
    features_list = []
    labels_list = []
    if id_features is not None and len(id_features) > 0:
        features_list.append(id_features)
        labels_list.extend([1] * len(id_features)) # 1 for ID
    if ood_features is not None and len(ood_features) > 0:
        features_list.append(ood_features)
        labels_list.extend([0] * len(ood_features)) # 0 for OOD

    if not features_list: return
    
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.array(labels_list)

    if all_features.shape[0] <=1: return # TSNE needs more than 1 sample
    perplexity_val = min(30.0, float(all_features.shape[0] - 1))
    if perplexity_val <= 1: return

    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity_val, max_iter=1000, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(all_features)
    
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=all_labels, cmap='coolwarm', alpha=0.7, s=10)
    plt.title(title); plt.xlabel("t-SNE Dim 1"); plt.ylabel("t-SNE Dim 2")
    handles, _ = scatter.legend_elements(prop="colors", alpha=0.6)
    if len(handles) == 2 : # If both ID and OOD are present
        plt.legend(handles, ['OOD', 'ID'], title="Classes")
    elif len(handles) == 1 and all_labels[0] == 1:
         plt.legend(handles, ['ID'], title="Classes")
    elif len(handles) == 1 and all_labels[0] == 0:
         plt.legend(handles, ['OOD'], title="Classes")

    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


# === Main Pipeline Class ===
class EnhancedOEPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.data_module: Optional[EnhancedDataModule] = None
        self.model: Optional[EnhancedModel] = None # Base classifier
        self.attention_analyzer: Optional[EnhancedAttentionAnalyzer] = None
        self.oe_extractor: Optional[OEExtractorEnhanced] = None
        self.visualizer = EnhancedVisualizer(config)
        
        config.create_directories() # This also calls update_derived_paths
        config.save_config()
        set_seed(config.RANDOM_STATE)
    
    def run_stage1_model_training(self):
            if not self.config.STAGE_MODEL_TRAINING:
                print("Skipping Stage 1: Base Model Training")
                # ... (rest of skip logic) ...
                return

            print(f"\n{'='*50}\nSTAGE 1: BASE MODEL TRAINING (ID: {self.config.CURRENT_NLP_ID_DATASET}, Model: {self.config.NLP_MODEL_TYPE})\n{'='*50}")
            self.data_module = EnhancedDataModule(self.config)
            self.data_module.prepare_data() 
            self.data_module.setup()
            
            self.model = EnhancedModel(
                config=self.config,
                num_labels=self.data_module.num_labels,
                label2id=self.data_module.label2id,
                id2label=self.data_module.id2label,
                class_weights=self.data_module.class_weights,
                tokenizer_for_custom_model=self.data_module.tokenizer if not self.data_module.is_hf_tokenizer else None
            )
            
            monitor_metric = 'val_f1_macro' # This is a string, e.g., "val_f1_macro"
            
            # CORRECTED filename construction for ModelCheckpoint
            filename_prefix_for_ckpt = f"{self.config.CURRENT_NLP_ID_DATASET}-{self.config.NLP_MODEL_TYPE}"
            # PyTorch Lightning will replace {epoch} and the metric name (e.g., {val_f1_macro:.4f})
            filename_template_for_pl = f"{filename_prefix_for_ckpt}-{{epoch}}-{{{monitor_metric}:.4f}}"
            
            checkpoint_cb = ModelCheckpoint(
                dirpath=self.config.MODEL_SAVE_DIR, 
                save_top_k=1, 
                monitor=monitor_metric, 
                mode='max',
                filename=filename_template_for_pl # Use the constructed template
            )
            early_stop_cb = EarlyStopping(monitor=monitor_metric, patience=3, mode='max', verbose=True)
            csv_logger = CSVLogger(save_dir=self.config.LOG_DIR, name=f"{self.config.CURRENT_NLP_ID_DATASET}_{self.config.NLP_MODEL_TYPE}_training")
            
            trainer = pl.Trainer(
                max_epochs=self.config.NLP_NUM_EPOCHS, accelerator=self.config.ACCELERATOR, devices=self.config.DEVICES,
                precision=self.config.PRECISION, logger=csv_logger, callbacks=[checkpoint_cb, early_stop_cb],
                log_every_n_steps=self.config.LOG_EVERY_N_STEPS, gradient_clip_val=self.config.GRADIENT_CLIP_VAL,
                deterministic=False 
            )
            trainer.fit(self.model, datamodule=self.data_module)
            self._load_best_model(checkpoint_cb)
            
    def run_stage2_attention_extraction(self) -> Optional[pd.DataFrame]:
        if not self.config.STAGE_ATTENTION_EXTRACTION:
            print("Skipping Stage 2: Attention Extraction for ID data")
            return self._load_attention_results() if (self.config.STAGE_OE_EXTRACTION or self.config.STAGE_VISUALIZATION) else None

        print(f"\n{'='*50}\nSTAGE 2: ATTENTION EXTRACTION (on ID: {self.config.CURRENT_NLP_ID_DATASET})\n{'='*50}")
        if self.model is None: self._load_existing_model()
        if self.data_module is None: # Should have been set up in stage 1 or _load_existing_model
            self.data_module = EnhancedDataModule(self.config); self.data_module.setup()

        self.attention_analyzer = EnhancedAttentionAnalyzer(self.config, self.model, self.model.device)
        
        full_id_df = self.data_module.get_full_dataframe() # This is the full ID dataset (train+test)
        # We want to process samples that were part of the training of the base classifier, or all ID data
        # For deriving OE, typically use the training portion of ID data.
        # Let's use the combined train_df_final and val_df_for_train from datamodule.
        df_to_analyze = pd.concat([self.data_module.train_df_final, self.data_module.val_df_for_train], ignore_index=True)

        processed_df = self.attention_analyzer.process_full_dataset(df_to_analyze)
        
        output_path = os.path.join(self.config.ATTENTION_DATA_DIR, f"df_id_{self.config.CURRENT_NLP_ID_DATASET}_with_attention.csv")
        processed_df.to_csv(output_path, index=False)
        print(f"ID Attention analysis results saved: {output_path}")
        return processed_df

    def run_stage3_oe_extraction(self, df_with_attention: Optional[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], Optional[List[np.ndarray]]]:
        if not self.config.STAGE_OE_EXTRACTION:
            print("Skipping Stage 3: OE Data Extraction from ID Attention")
            return self._load_final_metrics_and_features() if (self.config.STAGE_VISUALIZATION or self.config.STAGE_OSR_EXPERIMENTS) else (None,None)

        print(f"\n{'='*50}\nSTAGE 3: OE DATA EXTRACTION (from ID: {self.config.CURRENT_NLP_ID_DATASET} attention)\n{'='*50}")
        if df_with_attention is None: df_with_attention = self._load_attention_results()
        if df_with_attention is None: print("Error: DataFrame with attention not available."); return None, None
        
        if self.model is None: self._load_existing_model()
        self.oe_extractor = OEExtractorEnhanced(self.config, self.model, self.model.device)

        masked_texts_col = self.config.TEXT_COLUMN_IN_OE_FILES
        if masked_texts_col not in df_with_attention.columns:
             print(f"Error: Column '{masked_texts_col}' for masked texts not found."); return df_with_attention, None

        all_texts_for_metrics = df_with_attention[masked_texts_col].tolist()
        dataset_metrics = MaskedTextDatasetForMetrics(
            all_texts_for_metrics, self.model.tokenizer, self.config.NLP_MAX_LENGTH, self.model.is_hf_model
        )
        dataloader_metrics = DataLoader(
            dataset_metrics, 
            batch_size=self.config.NLP_BATCH_SIZE, 
            num_workers=self.config.NUM_WORKERS, 
            shuffle=False,
            multiprocessing_context='spawn' if self.config.NUM_WORKERS > 0 else None,  # 추가된 부분
            collate_fn=DataCollatorWithPadding(self.model.tokenizer) if self.model.is_hf_model else None,
            pin_memory=torch.cuda.is_available()  # 추가: 메모리 전송 최적화
        )    
        
        attention_metrics_df, features = self.oe_extractor.extract_attention_metrics(dataloader_metrics, df_with_attention)
        
        df_with_metrics = pd.concat([df_with_attention.reset_index(drop=True), attention_metrics_df.reset_index(drop=True)], axis=1)
        if self.attention_analyzer: # Needs to be initialized
             df_with_metrics = self.oe_extractor.compute_removed_word_attention(df_with_metrics, self.attention_analyzer)
        
        self.oe_extractor.extract_oe_datasets(df_with_metrics) # Saves derived OE files
        
        metrics_output_path = os.path.join(self.config.ATTENTION_DATA_DIR, f"df_id_{self.config.CURRENT_NLP_ID_DATASET}_all_metrics.csv")
        df_with_metrics.to_csv(metrics_output_path, index=False)
        features_path = os.path.join(self.config.ATTENTION_DATA_DIR, f"features_id_{self.config.CURRENT_NLP_ID_DATASET}.npy")
        if features: np.save(features_path, np.array(features))
        print(f"Derived OE metrics and features saved.")
        return df_with_metrics, features


    def run_stage4_visualization(self, df_with_metrics: Optional[pd.DataFrame], features: Optional[List[np.ndarray]]):
        if not self.config.STAGE_VISUALIZATION: 
            print("Skipping Stage 4: Visualization")
            return
        
        print(f"\n{'='*50}\nSTAGE 4: VISUALIZATION\n{'='*50}")
        if df_with_metrics is None or features is None:
            df_with_metrics, features = self._load_final_metrics_and_features()
        if df_with_metrics is None: 
            print("Error: Metrics DF not available for viz.")
            return
        
        if self.data_module is None: 
            self.data_module = EnhancedDataModule(self.config)
            self.data_module.prepare_data()
            self.data_module.setup()

        # 메트릭 분포 시각화
        self.visualizer.visualize_all_metrics(df_with_metrics)
        
        # OE 후보를 사용하여 ID 특징 시각화
        if features:
            self.visualizer.visualize_oe_candidates(df_with_metrics, features, 
                                                self.data_module.label2id, 
                                                self.data_module.id2label)
        
        # ID vs 외부 OE 데이터셋(WikiText-2 등) 시각화
        self._visualize_id_vs_external_oe(features)
        
        print("Visualization complete!")

    def _visualize_id_vs_external_oe(self, id_features: Optional[List[np.ndarray]]):
        """ID 특징과 WikiText-2와 같은 외부 OE 데이터셋을 시각화합니다."""
        if id_features is None or len(id_features) == 0:
            print("ID features not available for external OE visualization")
            return
        
        # 필요한 경우 특징 리스트를 numpy로 변환
        id_features_np = np.array(id_features)
        if id_features_np.ndim == 1:
            try:
                id_features_np = np.vstack(id_features)
            except Exception as e:
                print(f"Error converting ID features for visualization: {e}")
                return
        
        # OE 특징 디렉토리 순회
        vis_base_dir = os.path.join(self.config.VIS_DIR, self.config.CURRENT_NLP_ID_DATASET)
        os.makedirs(vis_base_dir, exist_ok=True)
        
        external_oe_features = {}
        
        # WikiText-2 또는 표준 OE 소스를 먼저 확인
        wikitext_path = os.path.join(vis_base_dir, self.config.DEFAULT_OE_DATASET, 
                                f"oe_features_{self.config.DEFAULT_OE_DATASET}.npy")
        if os.path.exists(wikitext_path):
            try:
                wikitext_features = np.load(wikitext_path)
                external_oe_features[self.config.DEFAULT_OE_DATASET] = wikitext_features
                print(f"Loaded external OE features: {self.config.DEFAULT_OE_DATASET} ({len(wikitext_features)} samples)")
            except Exception as e:
                print(f"Error loading {self.config.DEFAULT_OE_DATASET} features: {e}")
        
        # 다른 OE 하위 디렉토리 찾기
        for oe_source_dir in os.listdir(self.config.VIS_DIR):
            oe_dir_path = os.path.join(self.config.VIS_DIR, oe_source_dir)
            if not os.path.isdir(oe_dir_path) or oe_source_dir == self.config.CURRENT_NLP_ID_DATASET:
                continue
                
            # 이 디렉토리에 OE 특징이 있는지 확인
            for filename in os.listdir(oe_dir_path):
                if filename.startswith('oe_features_') and filename.endswith('.npy'):
                    oe_name = os.path.splitext(filename)[0].replace('oe_features_', '')
                    
                    # 이미 로드한 OE 소스는 건너뜀
                    if oe_name in external_oe_features:
                        continue
                        
                    try:
                        oe_features = np.load(os.path.join(oe_dir_path, filename))
                        external_oe_features[oe_name] = oe_features
                        print(f"Loaded external OE features: {oe_name} ({len(oe_features)} samples)")
                    except Exception as e:
                        print(f"Error loading OE features from {filename}: {e}")
        
        # 각 OE 소스에 대해 ID와 비교하는 t-SNE 시각화 생성
        id_labels = np.full(len(id_features_np), 1)  # 1 = ID
        
        for oe_name, oe_features in external_oe_features.items():
            # 결합된 데이터셋 준비
            all_features = np.vstack([id_features_np, oe_features])
            all_labels = np.concatenate([id_labels, np.zeros(len(oe_features))])  # 0 = OOD
            
            # 범례를 위한 ID 레이블 매핑
            class_names = {0: f"OE ({oe_name})", 1: f"ID ({self.config.CURRENT_NLP_ID_DATASET})"}
            
            # 시각화 생성
            save_path = os.path.join(vis_base_dir, f"tsne_id_vs_{oe_name}.png")
            title = f't-SNE: {self.config.CURRENT_NLP_ID_DATASET} (ID) vs {oe_name} (OE)'
            
            # visualizer의 plot_tsne 메서드 사용
            self.visualizer.plot_tsne(all_features, all_labels, title, save_path, 
                                    class_names=class_names)
            print(f"Generated t-SNE visualization: ID vs {oe_name}")

        
    def run_stage5_osr_experiments(self):
        if not self.config.STAGE_OSR_EXPERIMENTS: print("Skipping Stage 5: OSR Experiments"); return
        print(f"\n{'='*50}\nSTAGE 5: OSR EXPERIMENTS (ID: {self.config.CURRENT_NLP_ID_DATASET}, OSR Model: {self.config.OSR_NLP_MODEL_TYPE})\n{'='*50}")

        if self.data_module is None: # Ensure datamodule is loaded for ID data paths
            self.data_module = EnhancedDataModule(self.config)
            self.data_module.prepare_data()
            self.data_module.setup()

        self._run_nlp_osr_experiments()

    def _run_nlp_osr_experiments(self):
        # --- OSR Tokenizer Setup ---
        is_hf_osr_model = self.config.OSR_NLP_MODEL_TYPE == "roberta-base"
        osr_tokenizer_vocab_size = None
        if is_hf_osr_model:
            osr_tokenizer = AutoTokenizer.from_pretrained(self.config.OSR_NLP_MODEL_TYPE, cache_dir=self.config.CACHE_DIR_HF)
        else: # GRU/LSTM OSR model
            # Use the tokenizer from the base model training if compatible, or build a new one
            # For simplicity, assume base model's custom tokenizer is used for custom OSR model
            if self.data_module.is_hf_tokenizer and not is_hf_osr_model : # Base was HF, OSR is custom
                 print("Warning: Mismatch in tokenizer types between base and OSR. Re-initializing custom tokenizer for OSR.")
                 osr_tokenizer = NLPTokenizer(vocab_size=self.config.NLP_VOCAB_SIZE)
                 # This tokenizer would need to be built on ID train data.
                 # This scenario is complex. Assuming NLP_MODEL_TYPE and OSR_NLP_MODEL_TYPE are consistent for custom tokenizers.
                 # For now, let's assume if OSR is custom, base was custom.
                 if self.data_module.tokenizer is None or not isinstance(self.data_module.tokenizer, NLPTokenizer):
                     raise ValueError("Custom OSR model requires a custom NLPTokenizer from DataModule.")
                 osr_tokenizer = self.data_module.tokenizer
            else: # Both custom or base is custom
                osr_tokenizer = self.data_module.tokenizer # This is NLPTokenizer instance
            osr_tokenizer_vocab_size = len(osr_tokenizer.vocab)


        # --- ID Data for OSR ---
        # Use train_df_final for OSR training, val_df_final (original test split) for OSR ID testing
        id_train_texts = self.data_module.train_df_final['text'].tolist()
        id_train_labels = self.data_module.train_df_final['label_id'].tolist()
        id_test_texts = self.data_module.val_df_final['text'].tolist() # val_df_final is the original test split
        id_test_labels = self.data_module.val_df_final['label_id'].tolist()

        osr_id_train_dataset = OSRNNLPTorchDataset(id_train_texts, id_train_labels, osr_tokenizer, self.config.OSR_NLP_MAX_LENGTH, is_hf_osr_model)
        osr_id_test_dataset = OSRNNLPTorchDataset(id_test_texts, id_test_labels, osr_tokenizer, self.config.OSR_NLP_MAX_LENGTH, is_hf_osr_model)
        
        osr_id_train_loader = DataLoader(
            osr_id_train_dataset, 
            batch_size=self.config.OSR_NLP_BATCH_SIZE, 
            shuffle=True, 
            num_workers=self.config.OSR_NUM_DATALOADER_WORKERS,
            multiprocessing_context='spawn' if self.config.OSR_NUM_DATALOADER_WORKERS > 0 else None,  # 추가된 부분
            collate_fn=DataCollatorWithPadding(osr_tokenizer) if is_hf_osr_model else None, 
            persistent_workers=self.config.OSR_NUM_DATALOADER_WORKERS > 0,
            pin_memory=torch.cuda.is_available()  # 추가: 메모리 전송 최적화
        )
        
        osr_id_test_loader = DataLoader(
            osr_id_test_dataset, 
            batch_size=self.config.OSR_NLP_BATCH_SIZE, 
            num_workers=self.config.OSR_NUM_DATALOADER_WORKERS,
            multiprocessing_context='spawn' if self.config.OSR_NUM_DATALOADER_WORKERS > 0 else None,  # 추가된 부분
            collate_fn=DataCollatorWithPadding(osr_tokenizer) if is_hf_osr_model else None, 
            persistent_workers=self.config.OSR_NUM_DATALOADER_WORKERS > 0,
            pin_memory=torch.cuda.is_available()  # 추가: 메모리 전송 최적화
        )
        num_osr_classes = self.data_module.num_labels
        osr_class_names = [str(self.data_module.id2label.get(i,f"C{i}")) for i in range(num_osr_classes)]

        # --- OOD Test Data ---
        ood_test_data_name = self.config.DEFAULT_OOD_TEST_DATASET
        ood_test_raw = NLPDatasetLoader.load_ood_test_dataset(ood_test_data_name)
        ood_test_loader = None
        if ood_test_raw and ood_test_raw['text']:
            ood_test_dataset = OSRNNLPTorchDataset(ood_test_raw['text'], None, osr_tokenizer, self.config.OSR_NLP_MAX_LENGTH, is_hf_osr_model)
            ood_test_loader = DataLoader(
                ood_test_dataset, 
                batch_size=self.config.OSR_NLP_BATCH_SIZE, 
                num_workers=self.config.OSR_NUM_DATALOADER_WORKERS,
                multiprocessing_context='spawn' if self.config.OSR_NUM_DATALOADER_WORKERS > 0 else None,  # 추가된 부분
                collate_fn=DataCollatorWithPadding(osr_tokenizer) if is_hf_osr_model else None, 
                persistent_workers=self.config.OSR_NUM_DATALOADER_WORKERS > 0,
                pin_memory=torch.cuda.is_available()  # 추가: 메모리 전송 최적화
            )        
        else:
            print(f"Warning: Could not load OOD test dataset {ood_test_data_name}. OSR evaluation will be limited.")

        all_osr_results = {}

        # Exp 1: Standard OSR (No OE)
        if not self.config.OSR_SKIP_STANDARD_MODEL:
            print(f"\n--- OSR: Standard Model (ID: {self.config.CURRENT_NLP_ID_DATASET}) ---")
            results, _ = self._run_single_nlp_osr_experiment(
                num_osr_classes, osr_class_names, osr_id_train_loader, osr_id_test_loader, ood_test_loader, ood_test_data_name,
                is_hf_osr_model, osr_tokenizer, osr_tokenizer_vocab_size,
                oe_source_name="Standard_NoOE", oe_dataset_for_training=None
            )
            all_osr_results.update(results)



        # Exp 2: OE with WikiText2
        print(f"\n--- OSR: OE with {self.config.DEFAULT_OE_DATASET} (ID: {self.config.CURRENT_NLP_ID_DATASET}) ---")
        wikitext_oe_raw = NLPDatasetLoader.load_oe_dataset(self.config.DEFAULT_OE_DATASET)
        if wikitext_oe_raw and wikitext_oe_raw['text']:
            # Sample only 40% of the WikiText data to reduce processing time
            original_size = len(wikitext_oe_raw['text'])
            sampled_size = int(original_size * 0.4)  # Use 40% of the data
            
            # Use random sampling to get a representative subset
            import random
            random.seed(self.config.RANDOM_STATE)  # For reproducibility
            sampled_indices = random.sample(range(original_size), sampled_size)
            sampled_texts = [wikitext_oe_raw['text'][i] for i in sampled_indices]
            
            print(f"Sampled WikiText-2 data: {sampled_size} examples (40% of {original_size})")
            
            # Use the sampled data instead of the full dataset
            wikitext_oe_dataset = OSRNNLPTorchDataset(sampled_texts, None, osr_tokenizer, self.config.OSR_NLP_MAX_LENGTH, is_hf_osr_model)
            results, _ = self._run_single_nlp_osr_experiment(
                num_osr_classes, osr_class_names, osr_id_train_loader, osr_id_test_loader, ood_test_loader, ood_test_data_name,
                is_hf_osr_model, osr_tokenizer, osr_tokenizer_vocab_size,
                oe_source_name=self.config.DEFAULT_OE_DATASET, oe_dataset_for_training=wikitext_oe_dataset
            )
            all_osr_results.update(results)
        else:
            print(f"Warning: Could not load OE dataset {self.config.DEFAULT_OE_DATASET}. Skipping this OE experiment.")

        # Exp 3+: OE with Attention-Derived Data from ID
        print(f"\n--- OSR: OE with Attention-Derived data from {self.config.CURRENT_NLP_ID_DATASET} ---")
        derived_oe_files = [f for f in os.listdir(self.config.OE_DATA_DIR) if f.startswith(f"derived_oe_{self.config.CURRENT_NLP_ID_DATASET}") and f.endswith('.csv') and 'extended' not in f]
        
        for oe_filename in derived_oe_files:
            oe_data_path = os.path.join(self.config.OE_DATA_DIR, oe_filename)
            oe_source_name = os.path.splitext(oe_filename)[0] # More descriptive name
            print(f"\n--- OSR using Derived OE: {oe_source_name} ---")
            try:
                df_oe = pd.read_csv(oe_data_path)
                if self.config.TEXT_COLUMN_IN_OE_FILES not in df_oe.columns:
                    print(f"Warning: OE file {oe_filename} missing column '{self.config.TEXT_COLUMN_IN_OE_FILES}'. Skipping."); continue
                
                oe_texts = df_oe[self.config.TEXT_COLUMN_IN_OE_FILES].dropna().astype(str).tolist()
                if not oe_texts: print(f"Warning: No texts in OE file {oe_filename}. Skipping."); continue

                derived_oe_dataset = OSRNNLPTorchDataset(oe_texts, None, osr_tokenizer, self.config.OSR_NLP_MAX_LENGTH, is_hf_osr_model)
                results, _ = self._run_single_nlp_osr_experiment(
                    num_osr_classes, osr_class_names, osr_id_train_loader, osr_id_test_loader, ood_test_loader, ood_test_data_name,
                    is_hf_osr_model, osr_tokenizer, osr_tokenizer_vocab_size,
                    oe_source_name=oe_source_name, oe_dataset_for_training=derived_oe_dataset
                )
                all_osr_results.update(results)
            except Exception as e:
                print(f"Error processing derived OE file {oe_filename}: {e}. Skipping.")
        
        self._save_osr_results(all_osr_results, f"NLP_{self.config.CURRENT_NLP_ID_DATASET}_VS_{ood_test_data_name}")


    def _run_single_nlp_osr_experiment(self, num_classes: int, class_names: List[str],
                                    id_train_loader: DataLoader, id_test_loader: DataLoader, 
                                    ood_eval_loader: Optional[DataLoader], ood_eval_name: str,
                                    is_hf_osr_model: bool, osr_tokenizer: Union[NLPTokenizer, AutoTokenizer], 
                                    osr_tokenizer_vocab_size: Optional[int], # 커스텀 모델용
                                    oe_source_name: str, 
                                    oe_dataset_for_training: Optional[OSRNNLPTorchDataset]) -> Tuple[Dict, Dict]:
        
        experiment_tag = f"ID_{self.config.CURRENT_NLP_ID_DATASET}_OSRModel_{self.config.OSR_NLP_MODEL_TYPE}_OE_{oe_source_name}"
        print(f"\n===== NLP OSR Experiment: {experiment_tag} =====")
        
        # 디렉토리 경로 생성
        current_result_dir = os.path.join(self.config.OSR_RESULT_DIR, self.config.CURRENT_NLP_ID_DATASET, oe_source_name)
        current_model_dir = os.path.join(self.config.OSR_MODEL_DIR, self.config.CURRENT_NLP_ID_DATASET, oe_source_name)
        current_feature_dir = os.path.join(self.config.VIS_DIR, self.config.CURRENT_NLP_ID_DATASET, oe_source_name)
        
        # 디렉토리가 없으면 생성 - 모든 케이스에서 디렉토리를 확실히 생성
        os.makedirs(current_result_dir, exist_ok=True)
        os.makedirs(current_model_dir, exist_ok=True)  # OSR_SAVE_MODEL_PER_EXPERIMENT 조건 제거
        os.makedirs(current_feature_dir, exist_ok=True)

        model_osr = NLPModelOOD(self.config, num_classes, tokenizer_vocab_size=osr_tokenizer_vocab_size).to(DEVICE_OSR)
        model_filename = f"osr_model_{experiment_tag}.pt"
        model_save_path = os.path.join(current_model_dir, model_filename)

        epoch_losses = []
        if self.config.OSR_EVAL_ONLY and os.path.exists(model_save_path):
            print(f"Loading pre-trained OSR model: {model_save_path}")
            model_osr.load_state_dict(torch.load(model_save_path, map_location=DEVICE_OSR))
        else:
            if self.config.OSR_EVAL_ONLY: print(f"Eval only but model not found: {model_save_path}. Training new model.")
            
            optimizer = AdamW(model_osr.parameters(), lr=self.config.OSR_NLP_LEARNING_RATE) if is_hf_osr_model \
                        else optim.Adam(model_osr.parameters(), lr=self.config.OSR_NLP_LEARNING_RATE)

            model_osr.train()
            for epoch in range(self.config.OSR_NLP_NUM_EPOCHS):
                total_loss_epoch, id_loss_epoch, oe_loss_epoch = 0, 0, 0
                
                id_iter = iter(id_train_loader)
                
                num_batches = len(id_train_loader)
                oe_loader_for_training = None
                # OE 데이터셋으로 DataLoader 생성 부분 수정
                if oe_dataset_for_training:
                    # 'spawn' 방식 추가, 워커 수 최적화
                    oe_loader_for_training = DataLoader(
                        oe_dataset_for_training, 
                        batch_size=self.config.OSR_NLP_BATCH_SIZE, 
                        shuffle=True, 
                        num_workers=self.config.OSR_NUM_DATALOADER_WORKERS,
                        multiprocessing_context='spawn' if self.config.OSR_NUM_DATALOADER_WORKERS > 0 else None,  # 추가된 부분
                        collate_fn=DataCollatorWithPadding(osr_tokenizer) if is_hf_osr_model else None,
                        persistent_workers=self.config.OSR_NUM_DATALOADER_WORKERS > 0,
                        pin_memory=torch.cuda.is_available()  # 추가: 메모리 전송 최적화
                    )
                    num_batches = max(num_batches, len(oe_loader_for_training))
                    oe_iter = iter(oe_loader_for_training)

                progress_bar = tqdm(range(num_batches), desc=f"OSR Epoch {epoch+1}/{self.config.OSR_NLP_NUM_EPOCHS}", leave=False)
                
                for batch_idx in progress_bar:
                    optimizer.zero_grad()
                    try: 
                        id_batch = next(id_iter)
                    except StopIteration: 
                        id_iter = iter(id_train_loader)
                        id_batch = next(id_iter)

                    id_input_ids = id_batch['input_ids'].to(DEVICE_OSR)
                    id_attention_mask = id_batch['attention_mask'].to(DEVICE_OSR)
                    
                    # 레이블 접근
                    if 'labels' in id_batch:
                        id_labels = id_batch['labels'].to(DEVICE_OSR)
                    elif 'label' in id_batch:
                        id_labels = id_batch['label'].to(DEVICE_OSR)
                    else:
                        raise KeyError("ID batch for OSR training does not contain 'label' or 'labels' key.")
                    
                    id_logits = model_osr(id_input_ids, id_attention_mask) 
                    loss_id = F.cross_entropy(id_logits, id_labels)
                    
                    loss_oe = torch.tensor(0.0).to(DEVICE_OSR)
                    if oe_loader_for_training:
                        try: 
                            oe_batch = next(oe_iter)
                        except StopIteration: 
                            oe_iter = iter(oe_loader_for_training)
                            oe_batch = next(oe_iter)

                        oe_input_ids = oe_batch['input_ids'].to(DEVICE_OSR)
                        oe_attention_mask = oe_batch['attention_mask'].to(DEVICE_OSR)
                        
                        oe_logits = model_osr(oe_input_ids, oe_attention_mask)
                        log_softmax_oe = F.log_softmax(oe_logits, dim=1)
                        uniform_target = torch.full_like(oe_logits, 1.0 / num_classes)
                        loss_oe = F.kl_div(log_softmax_oe, uniform_target, reduction='batchmean', log_target=False)

                    total_batch_loss = loss_id + self.config.OSR_OE_LAMBDA * loss_oe
                    total_batch_loss.backward()
                    optimizer.step()

                    total_loss_epoch += total_batch_loss.item()
                    id_loss_epoch += loss_id.item()
                    oe_loss_epoch += loss_oe.item()
                    progress_bar.set_postfix({
                        'L_total': f"{total_batch_loss.item():.3f}", 
                        'L_id': f"{loss_id.item():.3f}", 
                        'L_oe': f"{loss_oe.item():.3f}"
                    })
                
                avg_loss = total_loss_epoch / num_batches
                epoch_losses.append(avg_loss)
                print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f} (ID: {id_loss_epoch/num_batches:.4f}, OE: {oe_loss_epoch/num_batches:.4f})")

            # 항상 모델 저장 (조건 제거 - 저장 여부는 config에서만 제어)
            if self.config.OSR_SAVE_MODEL_PER_EXPERIMENT: 
                torch.save(model_osr.state_dict(), model_save_path)
                print(f"OSR Model saved: {model_save_path}")
            
            # 항상 학습 곡선 그리기 (조건 완화)
            if epoch_losses and not self.config.OSR_NO_PLOT_PER_EXPERIMENT: 
                self._plot_training_curve(epoch_losses, experiment_tag, current_result_dir)
                
        # WikiText-2나 다른 외부 OE 데이터셋의 특징 추출 (시각화용)
            if oe_dataset_for_training and self.config.STAGE_VISUALIZATION:
                print(f"Extracting features from OE dataset: {oe_source_name} for visualization")
                oe_features_path = os.path.join(current_feature_dir, f"oe_features_{oe_source_name}.npy")
                
                # 이미 추출된 특징이 있는지 확인
                if os.path.exists(oe_features_path):
                    print(f"OE features already exist at {oe_features_path}")
                else:
                    # 여기도 'spawn' 방식 추가
                    oe_loader_for_features = DataLoader(
                        oe_dataset_for_training, 
                        batch_size=self.config.OSR_NLP_BATCH_SIZE, 
                        shuffle=False,
                        num_workers=self.config.OSR_NUM_DATALOADER_WORKERS,
                        multiprocessing_context='spawn' if self.config.OSR_NUM_DATALOADER_WORKERS > 0 else None,  # 추가된 부분
                        collate_fn=DataCollatorWithPadding(osr_tokenizer) if is_hf_osr_model else None,
                        persistent_workers=self.config.OSR_NUM_DATALOADER_WORKERS > 0,
                        pin_memory=torch.cuda.is_available()  # 추가: 메모리 전송 최적화
                    )
                
                # 특징만 추출하기 위해 평가 함수 사용 (id_loader=None)
                model_osr.eval()
                _, oe_data_dict = evaluate_nlp_osr(
                    model_osr, id_loader=None, ood_loader=oe_loader_for_features, device=DEVICE_OSR,
                    temperature=self.config.OSR_TEMPERATURE, threshold_percentile=self.config.OSR_THRESHOLD_PERCENTILE,
                    return_data=True, is_hf_osr_model=is_hf_osr_model,
                    osr_tokenizer_for_custom=osr_tokenizer if not is_hf_osr_model else None
                )
                
                # OE 특징 저장
                if oe_data_dict.get("ood_features") is not None and len(oe_data_dict["ood_features"]) > 0:
                    np.save(oe_features_path, oe_data_dict["ood_features"])
                    print(f"Saved OE features for {oe_source_name} ({len(oe_data_dict['ood_features'])} samples)")
                else:
                    print(f"Warning: Could not extract features from {oe_source_name} dataset")
        
        # ID 및 OOD 테스트 데이터로 평가
        results_this_exp, data_for_plots_this_exp = evaluate_nlp_osr(
            model_osr, id_test_loader, ood_eval_loader, DEVICE_OSR,
            self.config.OSR_TEMPERATURE, self.config.OSR_THRESHOLD_PERCENTILE, return_data=True,
            is_hf_osr_model=is_hf_osr_model, osr_tokenizer_for_custom=osr_tokenizer if not is_hf_osr_model else None
        )
        print(f"  Results ({experiment_tag} vs {ood_eval_name}): {results_this_exp}")
        
        # 고유한 키로 결과 저장
        metric_key = f"{experiment_tag}_VS_{ood_eval_name}"
        final_results = {metric_key: results_this_exp}
        final_data_for_plots = {metric_key: data_for_plots_this_exp}

        # ID 특징 저장 (시각화용)
        if self.config.STAGE_VISUALIZATION and data_for_plots_this_exp.get("id_features") is not None:
            id_features_path = os.path.join(current_feature_dir, f"id_features_{self.config.CURRENT_NLP_ID_DATASET}.npy")
            np.save(id_features_path, data_for_plots_this_exp["id_features"])
            print(f"Saved ID features for {self.config.CURRENT_NLP_ID_DATASET} ({len(data_for_plots_this_exp['id_features'])} samples)")

        # 평가 결과에 대한 플롯 생성 - 항상 실행 (조건 완화)
        if not self.config.OSR_NO_PLOT_PER_EXPERIMENT:
            plot_filename_prefix = re.sub(r'[^\w\-]+', '_', metric_key) # 파일 이름 정리
            
            # 스코어 플롯 생성 - id_scores와 ood_scores가 있는지 확인
            if (data_for_plots_this_exp.get('id_scores') is not None and 
                data_for_plots_this_exp.get('ood_scores') is not None and
                len(data_for_plots_this_exp['id_scores']) > 0 and 
                len(data_for_plots_this_exp['ood_scores']) > 0):
                
                plot_confidence_histograms_osr(
                    data_for_plots_this_exp['id_scores'], 
                    data_for_plots_this_exp['ood_scores'],
                    f'Conf - {metric_key}', 
                    os.path.join(current_result_dir, f'{plot_filename_prefix}_hist.png')
                )
                plot_roc_curve_osr(
                    data_for_plots_this_exp['id_scores'], 
                    data_for_plots_this_exp['ood_scores'],
                    f'ROC - {metric_key}', 
                    os.path.join(current_result_dir, f'{plot_filename_prefix}_roc.png')
                )
                print(f"Generated histogram and ROC curve plots for {oe_source_name}")
            else:
                print(f"Warning: Could not generate score plots for {oe_source_name} (missing score data)")
            
            # 특징 t-SNE 플롯 생성 - id_features와 ood_features가 있는지 확인
            if (data_for_plots_this_exp.get('id_features') is not None and 
                data_for_plots_this_exp.get('ood_features') is not None and
                len(data_for_plots_this_exp['id_features']) > 0 and 
                len(data_for_plots_this_exp['ood_features']) > 0):
                
                plot_tsne_osr(
                    data_for_plots_this_exp['id_features'], 
                    data_for_plots_this_exp['ood_features'],
                    f't-SNE - {metric_key}', 
                    os.path.join(current_result_dir, f'{plot_filename_prefix}_tsne.png'),
                    seed=self.config.RANDOM_STATE
                )
                print(f"Generated t-SNE plot for {oe_source_name}")
            else:
                print(f"Warning: Could not generate t-SNE plot for {oe_source_name} (missing feature data)")
        
        del model_osr; gc.collect(); torch.cuda.empty_cache()
        return final_results, final_data_for_plots
    def _save_osr_results(self, results: Dict, summary_name_suffix: str):
        print(f"\n===== OSR Experiments Overall Summary ({summary_name_suffix}) =====")
        if not results: print("No OSR results to save."); return

        results_df = pd.DataFrame.from_dict(results, orient='index').sort_index()
        print(results_df)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_fname = f"osr_summary_{summary_name_suffix}_{ts}"
        results_df.to_csv(os.path.join(self.config.OSR_RESULT_DIR, f"{base_fname}.csv"))
        
        config_snapshot = {k: str(v) for k, v in self.config.__class__.__dict__.items() if not k.startswith('_') and not callable(v)}
        summary_content = f"--- Config ---\n{json.dumps(config_snapshot, indent=2)}\n\n--- Results ---\n{results_df.to_string()}"
        with open(os.path.join(self.config.OSR_RESULT_DIR, f"{base_fname}.txt"), 'w') as f: f.write(summary_content)
        print(f"Overall OSR results saved in {self.config.OSR_RESULT_DIR}")

    def _plot_training_curve(self, losses: List[float], experiment_name: str, save_dir: str):
        plt.figure(figsize=(8,5)); plt.plot(losses, label='Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Training Loss: {experiment_name}')
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{experiment_name}_train_loss.png"), dpi=150); plt.close()

    def run_full_pipeline(self):
        print(f"Starting NLP OE Pipeline (ID: {self.config.CURRENT_NLP_ID_DATASET}, Model: {self.config.NLP_MODEL_TYPE})...")
        df_with_attention, df_with_metrics, features = None, None, None
        
        self.run_stage1_model_training()
        df_with_attention = self.run_stage2_attention_extraction()
        df_with_metrics, features = self.run_stage3_oe_extraction(df_with_attention)
        self.run_stage4_visualization(df_with_metrics, features)
        self.run_stage5_osr_experiments()
        
        self._print_final_summary()
        print(f"\nNLP OE Pipeline Complete!")

    def _load_existing_model(self): # For base classifier
        if self.data_module is None:
            self.data_module = EnhancedDataModule(self.config)
            self.data_module.prepare_data(); self.data_module.setup()

        ckpt_files = [os.path.join(self.config.MODEL_SAVE_DIR, f) for f in os.listdir(self.config.MODEL_SAVE_DIR) if f.endswith('.ckpt')]
        if not ckpt_files: raise FileNotFoundError(f"No .ckpt found in {self.config.MODEL_SAVE_DIR}")
        
        latest_ckpt = max(ckpt_files, key=os.path.getctime)
        print(f"Loading existing base model from: {latest_ckpt}")
        self.model = EnhancedModel.load_from_checkpoint(
            latest_ckpt, config=self.config, num_labels=self.data_module.num_labels,
            label2id=self.data_module.label2id, id2label=self.data_module.id2label,
            class_weights=self.data_module.class_weights,
            tokenizer_for_custom_model=self.data_module.tokenizer if not self.data_module.is_hf_tokenizer else None
        )
        print("Base model loaded.")

    def _load_best_model(self, checkpoint_callback: ModelCheckpoint):
        if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
            print(f"Loading best base model: {checkpoint_callback.best_model_path}")
            # Ensure datamodule is set up if called standalone
            if self.data_module is None or self.data_module.num_labels is None: 
                 self.data_module = EnhancedDataModule(self.config); self.data_module.prepare_data(); self.data_module.setup()

            self.model = EnhancedModel.load_from_checkpoint(
                checkpoint_callback.best_model_path, config=self.config, num_labels=self.data_module.num_labels,
                label2id=self.data_module.label2id, id2label=self.data_module.id2label,
                class_weights=self.data_module.class_weights,
                tokenizer_for_custom_model=self.data_module.tokenizer if not self.data_module.is_hf_tokenizer else None
            )
            print("Best base model loaded.")
        else:
            print("Warning: Best model path not found or invalid. Using current model state.")

    def _load_attention_results(self) -> Optional[pd.DataFrame]:
        path = os.path.join(self.config.ATTENTION_DATA_DIR, f"df_id_{self.config.CURRENT_NLP_ID_DATASET}_with_attention.csv")
        if os.path.exists(path):
            print(f"Loading attention results from: {path}")
            df = pd.read_csv(path)
            if 'top_attention_words' in df.columns: df['top_attention_words'] = df['top_attention_words'].apply(safe_literal_eval)
            return df
        print(f"Attention results file not found: {path}"); return None

    def _load_final_metrics_and_features(self) -> Tuple[Optional[pd.DataFrame], Optional[List[np.ndarray]]]:
        metrics_path = os.path.join(self.config.ATTENTION_DATA_DIR, f"df_id_{self.config.CURRENT_NLP_ID_DATASET}_all_metrics.csv")
        features_path = os.path.join(self.config.ATTENTION_DATA_DIR, f"features_id_{self.config.CURRENT_NLP_ID_DATASET}.npy")
        df, feats = None, None
        if os.path.exists(metrics_path):
            print(f"Loading metrics DF from: {metrics_path}")
            df = pd.read_csv(metrics_path)
            if 'top_attention_words' in df.columns: df['top_attention_words'] = df['top_attention_words'].apply(safe_literal_eval)
        if os.path.exists(features_path):
            print(f"Loading features from: {features_path}")
            feats = np.load(features_path, allow_pickle=True).tolist() # Assuming list of arrays
        return df, feats
    
    def _print_final_summary(self):
        print(f"\n{'='*50}\nPIPELINE SUMMARY (ID: {self.config.CURRENT_NLP_ID_DATASET})\n{'='*50}")
        print(f"Output directory: {self.config.OUTPUT_DIR}")
        print("\nGenerated Files (Examples):")
        max_files_to_list = 5
        for sub_dir in [self.config.MODEL_SAVE_DIR, self.config.OE_DATA_DIR, self.config.OSR_RESULT_DIR, self.config.VIS_DIR]:
            if os.path.exists(sub_dir):
                print(f"  In {os.path.basename(sub_dir)}:")
                files_in_subdir = [f for f in os.listdir(sub_dir) if os.path.isfile(os.path.join(sub_dir, f))]
                for f_idx, f_name in enumerate(files_in_subdir[:max_files_to_list]):
                    print(f"    - {f_name}")
                if len(files_in_subdir) > max_files_to_list: print("    ...")


# === Main Function ===
def main():
    parser = argparse.ArgumentParser(description="NLP Outlier Exposure Pipeline")
    parser.add_argument('--id_dataset', type=str, default=Config.CURRENT_NLP_ID_DATASET, choices=list(Config.NLP_ID_DATASETS.keys()))
    parser.add_argument('--ood_test_dataset', type=str, default=Config.DEFAULT_OOD_TEST_DATASET, choices=list(Config.NLP_OOD_TEST_DATASETS.keys()))
    parser.add_argument('--nlp_model_type', type=str, default=Config.NLP_MODEL_TYPE, choices=['gru', 'lstm', 'roberta-base'])
    parser.add_argument('--osr_nlp_model_type', type=str, default=Config.OSR_NLP_MODEL_TYPE, choices=['gru', 'lstm', 'roberta-base'])
    
    parser.add_argument('--nlp_epochs', type=int, default=Config.NLP_NUM_EPOCHS)
    parser.add_argument('--osr_epochs', type=int, default=Config.OSR_NLP_NUM_EPOCHS)
    parser.add_argument('--attention_percent', type=float, default=Config.ATTENTION_TOP_PERCENT)
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR)
    
    parser.add_argument('--skip_base_training', action='store_true')
    parser.add_argument('--skip_attention_extraction', action='store_true')
    parser.add_argument('--skip_oe_extraction', action='store_true')
    parser.add_argument('--skip_visualization', action='store_true')
    parser.add_argument('--skip_osr_experiments', action='store_true')
    parser.add_argument('--osr_eval_only', action='store_true')
    args = parser.parse_args()

    # Update Config class with parsed arguments
    Config.CURRENT_NLP_ID_DATASET = args.id_dataset
    Config.DEFAULT_OOD_TEST_DATASET = args.ood_test_dataset
    Config.NLP_MODEL_TYPE = args.nlp_model_type
    Config.OSR_NLP_MODEL_TYPE = args.osr_nlp_model_type # Make OSR model same as base or allow different
    
    Config.NLP_NUM_EPOCHS = args.nlp_epochs
    Config.OSR_NLP_NUM_EPOCHS = args.osr_epochs
    Config.ATTENTION_TOP_PERCENT = args.attention_percent
    Config.OUTPUT_DIR = args.output_dir # Crucial: set this first

    # Update learning rates based on model type AFTER model type is set
    Config.NLP_LEARNING_RATE = 2e-5 if Config.NLP_MODEL_TYPE == "roberta-base" else 1e-3
    Config.OSR_NLP_LEARNING_RATE = 2e-5 if Config.OSR_NLP_MODEL_TYPE == "roberta-base" else 1e-3

    Config.STAGE_MODEL_TRAINING = not args.skip_base_training
    Config.STAGE_ATTENTION_EXTRACTION = not args.skip_attention_extraction
    Config.STAGE_OE_EXTRACTION = not args.skip_oe_extraction
    Config.STAGE_VISUALIZATION = not args.skip_visualization
    Config.STAGE_OSR_EXPERIMENTS = not args.skip_osr_experiments
    Config.OSR_EVAL_ONLY = args.osr_eval_only
    
    print(f"--- NLP Outlier Exposure Pipeline ---")
    print(f"ID Dataset: {Config.CURRENT_NLP_ID_DATASET}, Base Model: {Config.NLP_MODEL_TYPE}")
    print(f"OOD Test Dataset: {Config.DEFAULT_OOD_TEST_DATASET}, OSR Model: {Config.OSR_NLP_MODEL_TYPE}")
    print(f"Output Dir: {Config.OUTPUT_DIR}")

    pipeline = EnhancedOEPipeline(Config)
    pipeline.run_full_pipeline()

if __name__ == '__main__':
    main()
# ```

# **Key changes in this final part:**

# 1.  **`EnhancedOEPipeline`:**
#     *   `__init__`: Simplified, calls `config.create_directories()` which now also calls `update_derived_paths`.
#     *   `run_stage1_model_training`: Initializes `EnhancedDataModule` and `EnhancedModel` (base classifier). Passes the custom tokenizer to `EnhancedModel` if a GRU/LSTM is used.
#     *   `run_stage2_attention_extraction`: Gets the ID dataset from `data_module` (specifically the parts used for training the base classifier) to perform attention analysis on.
#     *   `run_stage3_oe_extraction`: Uses masked texts from `df_with_attention` to compute OE metrics and extract features. Saves derived OE datasets to `Config.OE_DATA_DIR`.
#     *   `run_stage4_visualization`: Visualizes metrics from derived OE data and highlights candidates on t-SNE plots of ID data features.
#     *   `_run_nlp_osr_experiments`:
#         *   Sets up the OSR tokenizer (HuggingFace or custom) based on `Config.OSR_NLP_MODEL_TYPE`.
#         *   Prepares ID train/test data for OSR using `OSRNNLPTorchDataset`.
#         *   Loads the OOD Test dataset (e.g., WMT16).
#         *   Iterates through OE sources:
#             1.  Standard (no OE).
#             2.  WikiText-2 (loaded via `NLPDatasetLoader`).
#             3.  Attention-derived OE files from `Config.OE_DATA_DIR`.
#         *   Calls `_run_single_nlp_osr_experiment` for each.
#     *   `_run_single_nlp_osr_experiment`:
#         *   Initializes `NLPModelOOD` (the OSR model).
#         *   Handles training (standard or with OE loss term) or loading of the OSR model.
#         *   Performs evaluation using `evaluate_nlp_osr`.
#         *   Saves results and plots.
#     *   Helper methods (`_load_existing_model`, `_load_best_model`, etc.) are updated for new paths and NLP focus.

# 2.  **OSR Evaluation and Plotting:**
#     *   The `evaluate_nlp_osr` function and plotting helpers (`plot_confidence_histograms_osr`, etc.) are assumed to be largely functional from your existing script or `oe2.py`. I've included simplified versions with adaptations for NLPModelOOD's potential outputs. You might need to fine-tune these based on the exact feature/logit structure from `NLPModelOOD`.

# 3.  **`main()` function:**
#     *   Parses NLP-specific arguments.
#     *   Updates the `Config` object. Importantly, `Config.OUTPUT_DIR` is set first, then `Config.update_derived_paths()` is called (implicitly by `create_directories` in `EnhancedOEPipeline.__init__`). Learning rates are also updated based on chosen model types.
#     *   Instantiates and runs the pipeline.

# **To make this fully runnable, you would:**

# 1.  **Review `NLPModelOOD`'s `forward` method:** Ensure it consistently returns `(logits, features)` when `output_features=True`, regardless of whether it's an HF model or a custom one. The `evaluate_nlp_osr` function relies on this.
# 2.  **Test Tokenizer Logic:** Especially the interaction between `NLPTokenizer` (custom) and `AutoTokenizer` (HF) if you mix model types for base vs. OSR. The current code tries to handle this by re-initializing tokenizers where needed, but consistency is simpler.
# 3.  **Attention Extraction for RoBERTa:** The `_process_attention_batch_hf` and `_extract_word_scores_from_cls_attention_hf` methods are crucial for RoBERTa. They aim to get word scores from CLS token attention. This is a common approach but might need tuning.
# 4.  **Attention Extraction for Custom Models:** `_process_attention_batch_custom_nlp` assumes the custom model's attention output can be mapped back to words. This depends on the specifics of your `CustomNLPClassifier`'s attention.
# 5.  **Memory Management:** Deep learning with NLP can be memory-intensive. Ensure `gc.collect()` and `torch.cuda.empty_cache()` are used appropriately, especially in loops running many experiments.

# This refactoring provides a strong foundation for your specific NLP Outlier Exposure research. Remember to test each stage incrementally.