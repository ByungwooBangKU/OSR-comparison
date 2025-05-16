# -*- coding: utf-8 -*-
"""
Unified OE (Out-of-Distribution) Extractor and OSR Experimentation
(Extended with external OE dataset comparison)
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
from datasets import Dataset as HFDataset, DatasetDict, load_dataset # Hugging Face datasets for oe.py part and external OE

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    RobertaConfig,
    AdamW
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

import nltk
from nltk.tokenize import word_tokenize
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

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Configuration Class ---
class Config:
    ORIGINAL_DATA_PATH = 'data_syslog/log_all_critical.csv'
    TEXT_COLUMN = 'text'
    CLASS_COLUMN = 'class'
    EXCLUDE_CLASS_FOR_TRAINING = "unknown"
    
    OUTPUT_DIR = 'unified_oe_osr_results_extOE'
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "base_classifier_model")
    LOG_DIR = os.path.join(OUTPUT_DIR, "lightning_logs")
    CONFUSION_MATRIX_DIR = os.path.join(LOG_DIR, "confusion_matrices")
    VIS_DIR = os.path.join(OUTPUT_DIR, "oe_extraction_visualizations")
    OE_DATA_DIR = os.path.join(OUTPUT_DIR, "extracted_oe_datasets")
    ATTENTION_DATA_DIR = os.path.join(OUTPUT_DIR, "attention_analysis")
    
    MODEL_NAME = "roberta-base"
    MAX_LENGTH = 256
    BATCH_SIZE = 64
    NUM_TRAIN_EPOCHS = 20 # Base classifier epochs
    LEARNING_RATE = 2e-5
    MIN_SAMPLES_PER_CLASS_FOR_TRAIN_VAL = 2
    
    ACCELERATOR = "auto"
    DEVICES = "auto"
    PRECISION = "16-mixed" if torch.cuda.is_available() else "32-true"
    NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1
    
    LOG_EVERY_N_STEPS = 70
    GRADIENT_CLIP_VAL = 1.0
    USE_WEIGHTED_LOSS = True
    USE_LR_SCHEDULER = True
    RANDOM_STATE = 42
    
    ATTENTION_TOP_PERCENT = 0.20
    MIN_TOP_WORDS = 1
    TOP_K_ATTENTION = 3
    ATTENTION_LAYER = -1
    
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

    OSR_EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, "osr_experiments")
    OSR_MODEL_DIR = os.path.join(OSR_EXPERIMENT_DIR, "models")
    OSR_RESULT_DIR = os.path.join(OSR_EXPERIMENT_DIR, "results")
    
    OOD_SYSLOG_UNKNOWN_PATH_OSR = 'data_syslog/log_unknown.csv'
    OOD_TARGET_CLASS_OSR = "unknown"

    OSR_MODEL_TYPE = 'roberta-base'
    OSR_MAX_LENGTH = 128
    OSR_BATCH_SIZE = 64
    OSR_NUM_EPOCHS = 30 # OSR experiment epochs
    OSR_LEARNING_RATE = 2e-5
    OSR_OE_LAMBDA = 1.0
    OSR_TEMPERATURE = 1.0
    OSR_THRESHOLD_PERCENTILE = 5.0
    OSR_NUM_DATALOADER_WORKERS = NUM_WORKERS
    
    OSR_EARLY_STOPPING_PATIENCE = 5
    OSR_EARLY_STOPPING_MIN_DELTA = 0.001
    OSR_WARMUP_RATIO = 0.1

    # NEW: External OE sources for OSR experiments
    # Example: ['wikitext', 'snli', 'imdb']
    # wikitext config: 'wikitext-103-raw-v1' or 'wikitext-2-raw-v1'
    OSR_OE_SOURCES_EXTERNAL = ['wikitext:wikitext-2-raw-v1'] # Specify dataset name and optionally config_name
    
    DATA_DIR_EXTERNAL_HF = os.path.join(OUTPUT_DIR, 'data_external_hf') 
    CACHE_DIR_HF = os.path.join(DATA_DIR_EXTERNAL_HF, "hf_cache")

    OSR_SAVE_MODEL_PER_EXPERIMENT = True
    OSR_EVAL_ONLY = False
    OSR_NO_PLOT_PER_EXPERIMENT = False
    OSR_SKIP_STANDARD_MODEL = False
    OSR_SKIP_ATTENTION_OE_MODELS = False # New flag to skip locally generated OE models
    OSR_SKIP_EXTERNAL_OE_MODELS = False # New flag to skip external OE models

    STAGE_MODEL_TRAINING = True
    STAGE_ATTENTION_EXTRACTION = True
    STAGE_OE_EXTRACTION = True
    STAGE_VISUALIZATION = True
    STAGE_OSR_EXPERIMENTS = True
        
    @classmethod
    def create_directories(cls):
        dirs = [
            cls.OUTPUT_DIR, cls.MODEL_SAVE_DIR, cls.LOG_DIR,
            cls.CONFUSION_MATRIX_DIR, cls.VIS_DIR, cls.OE_DATA_DIR,
            cls.ATTENTION_DATA_DIR,
            cls.OSR_EXPERIMENT_DIR, cls.OSR_MODEL_DIR, cls.OSR_RESULT_DIR,
            cls.DATA_DIR_EXTERNAL_HF, cls.CACHE_DIR_HF
        ]
        for dir_path in dirs: os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def save_config(cls, filepath=None):
        if filepath is None: filepath = os.path.join(cls.OUTPUT_DIR, 'config_unified.json')
        config_dict = {attr: getattr(cls, attr) for attr in dir(cls)
                       if not attr.startswith('_') and not callable(getattr(cls, attr)) and
                       isinstance(getattr(cls, attr), (str, int, float, bool, list, dict, type(None)))}
        with open(filepath, 'w') as f: json.dump(config_dict, f, indent=2, default=str)
        print(f"Configuration saved to {filepath}")

# === Global Helpers & OSR Components (mostly same as oe2.py, with minor additions/checks) ===
DEVICE_OSR = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    print(f"Seed set to {seed}")

def preprocess_text_for_roberta(text):
    if not isinstance(text, str): return ""
    return re.sub(r'\s+', ' ', text).strip()

def tokenize_nltk(text):
    if not text: return []
    try: return word_tokenize(text)
    except Exception: return text.split()

def create_masked_sentence(original_text, important_words):
    if not isinstance(original_text, str): return ""
    if not important_words: return original_text
    processed_text = preprocess_text_for_roberta(original_text)
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
    except (ValueError, SyntaxError): return []

class OSRTextDataset(TorchDataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.labels = labels
        valid_texts = [str(t) if pd.notna(t) else "" for t in texts]
        self.encodings = tokenizer(valid_texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    def __len__(self): return len(self.labels)
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
            features = outputs.hidden_states[-1][:, 0, :]
            return logits, features
        return logits

def prepare_id_data_for_osr(datamodule: 'UnifiedDataModule', tokenizer, max_length: int) -> Tuple[Optional[OSRTextDataset], Optional[OSRTextDataset], int, Optional[LabelEncoder], Dict, Dict]:
    print(f"\n--- Preparing ID data for OSR from UnifiedDataModule ---")
    if datamodule.train_df_final is None or datamodule.val_df_final is None:
        print("Error: DataModule not set up or train/val split not performed.")
        return None, None, 0, None, {}, {}
    train_df = datamodule.train_df_final
    test_df = datamodule.val_df_final
    num_classes = datamodule.num_labels
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array([datamodule.id2label[i] for i in range(num_classes)])
    id_label2id = datamodule.label2id
    id_id2label = datamodule.id2label
    print(f"  - Using {num_classes} known classes. Label mapping: {id_label2id}")
    train_df['label'] = train_df['label'].astype(int)
    test_df['label'] = test_df['label'].astype(int)
    train_dataset = OSRTextDataset(train_df[Config.TEXT_COLUMN].tolist(), train_df['label'].tolist(), tokenizer, max_length)
    id_test_dataset = OSRTextDataset(test_df[Config.TEXT_COLUMN].tolist(), test_df['label'].tolist(), tokenizer, max_length)
    print(f"  - OSR Train: {len(train_dataset)}, OSR ID Test: {len(id_test_dataset)}")
    return train_dataset, id_test_dataset, num_classes, label_encoder, id_label2id, id_id2label

def prepare_syslog_ood_data_for_osr(tokenizer, max_length: int, ood_data_path: str, text_col: str, class_col: str, ood_target_class: str) -> Optional[OSRTextDataset]:
    print(f"\n--- Preparing Syslog OOD data (class: '{ood_target_class}') from: {ood_data_path} for OSR ---")
    if not os.path.exists(ood_data_path):
        print(f"Error: OOD data path not found: {ood_data_path}"); return None
    try:
        df = pd.read_csv(ood_data_path)
        if not all(c in df.columns for c in [text_col, class_col]):
            raise ValueError(f"OOD CSV must contain '{text_col}' and '{class_col}'.")
        df = df.dropna(subset=[text_col, class_col])
        df[class_col] = df[class_col].astype(str).str.lower()
        df_ood = df[df[class_col] == ood_target_class.lower()].copy()
        if df_ood.empty: print(f"Warning: No data for OOD class '{ood_target_class}'."); return None
        texts = df_ood[text_col].tolist()
        ood_labels = np.full(len(texts), -1, dtype=int).tolist()
        ood_dataset = OSRTextDataset(texts, ood_labels, tokenizer, max_length)
        print(f"  - Loaded {len(ood_dataset)} OOD samples (class: '{ood_target_class}').")
        return ood_dataset
    except Exception as e: print(f"Error preparing Syslog OOD: {e}"); return None

def prepare_generated_oe_data_for_osr(tokenizer, max_length: int, oe_data_path: str, oe_text_col: str) -> Optional[OSRTextDataset]:
    print(f"\n--- Preparing Generated OE data from: {oe_data_path} for OSR ---")
    if not os.path.exists(oe_data_path):
        print(f"Error: OE data path not found: {oe_data_path}"); return None
    try:
        df = pd.read_csv(oe_data_path)
        oe_text_col_actual = oe_text_col
        if oe_text_col not in df.columns:
            fallback_cols = ['masked_text_attention', 'text', Config.TEXT_COLUMN]
            found_col = False
            for col_attempt in fallback_cols:
                if col_attempt in df.columns:
                    oe_text_col_actual = col_attempt
                    print(f"  Warning: Specified OE text col '{oe_text_col}' not found. Using fallback '{oe_text_col_actual}'.")
                    found_col = True; break
            if not found_col: raise ValueError(f"OE CSV must contain a text column (tried '{oe_text_col}' and fallbacks).")
        
        df = df.dropna(subset=[oe_text_col_actual])
        texts = df[oe_text_col_actual].astype(str).tolist()
        if not texts: print(f"Warning: No valid OE texts in '{oe_text_col_actual}'."); return None
        oe_labels = np.full(len(texts), -1, dtype=int).tolist()
        oe_dataset = OSRTextDataset(texts, oe_labels, tokenizer, max_length)
        print(f"  - Loaded {len(oe_dataset)} OE samples from {oe_data_path} (col '{oe_text_col_actual}').")
        return oe_dataset
    except Exception as e: print(f"Error preparing Generated OE: {e}"); return None

# NEW: Function to prepare external OE data for OSR
def prepare_external_oe_data_for_osr(tokenizer, max_length: int, oe_source_name_full: str,
                                      data_dir_hf: str = 'data_external_hf', 
                                      cache_dir_hf: Optional[str] = None) -> Optional[OSRTextDataset]:
    # oe_source_name_full can be "dataset_name" or "dataset_name:config_name"
    parts = oe_source_name_full.split(':')
    dataset_name = parts[0]
    config_name = parts[1] if len(parts) > 1 else None

    print(f"\n--- Preparing External OE data source: {dataset_name} (Config: {config_name}) for OSR ---")
    if cache_dir_hf is None: cache_dir_hf = os.path.join(data_dir_hf, "hf_cache")
    os.makedirs(cache_dir_hf, exist_ok=True)

    dataset_config_map = {
        "snli": {"split": "train", "text_col": "hypothesis"},
        "imdb": {"split": "train", "text_col": "text"},
        "wikitext": {"split": "train", "text_col": "text"}, # Default to wikitext-103-raw-v1 if no config_name
        # Add more dataset specifics here if needed
    }
    
    if dataset_name not in dataset_config_map:
        print(f"Error: Unknown external OE source name '{dataset_name}'. Add its config to dataset_config_map.")
        return None
    
    current_ds_config = dataset_config_map[dataset_name]
    # Use provided config_name if available, otherwise stick to default for wikitext or None for others
    hf_config_name_to_load = config_name if config_name else (current_ds_config.get('default_config_name', None))
    if dataset_name == "wikitext" and not hf_config_name_to_load: # Default wikitext if not specified
        hf_config_name_to_load = "wikitext-103-raw-v1" # Common default

    try:
        print(f"  Loading {dataset_name} (HF Config: {hf_config_name_to_load}, Split: {current_ds_config['split']})...")
        ds = load_dataset(dataset_name, name=hf_config_name_to_load, split=current_ds_config['split'], cache_dir=cache_dir_hf)
        
        if isinstance(ds, DatasetDict):
            if current_ds_config['split'] in ds: ds = ds[current_ds_config['split']]
            else: raise ValueError(f"Split '{current_ds_config['split']}' not found for {dataset_name}")

        texts = [item for item in ds[current_ds_config['text_col']] if isinstance(item, str) and item.strip()]
        if dataset_name == "wikitext": # Specific cleaning for wikitext
            texts = [text for text in texts if not text.strip().startswith("=") and len(text.strip().split()) > 3]
        
        if not texts: print(f"Warning: No valid texts for OE source {dataset_name}."); return None
        
        # Limit sample size for very large datasets to speed up OE training if needed
        # max_oe_samples = 50000 # Example limit
        # if len(texts) > max_oe_samples:
        #     print(f"  Sampling {max_oe_samples} from {len(texts)} for {dataset_name} OE.")
        #     random.shuffle(texts)
        #     texts = texts[:max_oe_samples]

        oe_labels = np.full(len(texts), -1, dtype=int).tolist()
        oe_dataset = OSRTextDataset(texts, oe_labels, tokenizer, max_length)
        print(f"  - Loaded {len(oe_dataset)} samples for OE training from {dataset_name}.")
        return oe_dataset
    except Exception as e: print(f"Error loading external OE dataset {dataset_name}: {e}"); return None


# OSR Training and Evaluation functions (evaluate_osr, plot_confidence_histograms_osr, etc.)
# remain the same as in oe2.py.
# ... (Rest of the OSR helper functions, plotting functions, and PyTorch Lightning modules from oe2.py) ...
# ... (UnifiedDataModule, UnifiedModel, AttentionAnalyzer, OEExtractor, Visualizer) ...
# These are assumed to be identical to your `oe2.py` and are omitted for brevity here,
# but they should be included in the final complete script.

# === PyTorch Lightning 컴포넌트 (OE Extraction Part) - Assuming these are as in oe2.py ===
class UnifiedDataModule(pl.LightningDataModule):
    # ... (Implementation from oe2.py) ...
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['config']) 
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.df_full = None; self.df_known_for_train_val = None; self.train_df_final = None
        self.val_df_final = None; self.label2id = None; self.id2label = None
        self.num_labels = None; self.tokenized_train_val_datasets = None; self.class_weights = None
    def prepare_data(self): pass
    def setup(self, stage=None):
        if self.df_full is not None: return
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
        known_classes_str = sorted(df_known[self.config.CLASS_COLUMN].unique())
        self.label2id = {label: i for i, label in enumerate(known_classes_str)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.num_labels = len(known_classes_str)
        if self.num_labels == 0: raise ValueError("No known classes found after excluding.")
        df_known['label'] = df_known[self.config.CLASS_COLUMN].map(self.label2id)
        df_known = df_known.dropna(subset=['label']); df_known['label'] = df_known['label'].astype(int)
        label_counts = df_known['label'].value_counts()
        valid_labels = label_counts[label_counts >= self.config.MIN_SAMPLES_PER_CLASS_FOR_TRAIN_VAL].index
        self.df_known_for_train_val = df_known[df_known['label'].isin(valid_labels)].copy()
        if len(valid_labels) < self.num_labels:
            final_classes_str = sorted(self.df_known_for_train_val[self.config.CLASS_COLUMN].unique())
            self.label2id = {label: i for i, label in enumerate(final_classes_str)}
            self.id2label = {i: label for label, i in self.label2id.items()}
            self.num_labels = len(final_classes_str)
            self.df_known_for_train_val['label'] = self.df_known_for_train_val[self.config.CLASS_COLUMN].map(self.label2id)
            self.df_known_for_train_val['label'] = self.df_known_for_train_val['label'].astype(int)
        if len(self.df_known_for_train_val) == 0: raise ValueError("No data available after filtering.")
        if self.config.USE_WEIGHTED_LOSS: self._compute_class_weights()
        self._split_train_val(); self._tokenize_datasets()
    def _compute_class_weights(self):
        labels = self.df_known_for_train_val['label'].values; unique_labels = np.unique(labels)
        try:
            weights_arr = compute_class_weight('balanced', classes=unique_labels, y=labels)
            self.class_weights = torch.ones(self.num_labels)
            for i, lbl_idx in enumerate(unique_labels):
                if lbl_idx < self.num_labels: self.class_weights[lbl_idx] = weights_arr[i]
        except ValueError: self.config.USE_WEIGHTED_LOSS = False; self.class_weights = None
    def _split_train_val(self):
        min_c = self.df_known_for_train_val['label'].value_counts().min()
        stratify = self.df_known_for_train_val['label'] if min_c > 1 else None
        try:
            self.train_df_final, self.val_df_final = train_test_split(
                self.df_known_for_train_val, test_size=0.2, random_state=self.config.RANDOM_STATE, stratify=stratify)
        except ValueError:
            self.train_df_final, self.val_df_final = train_test_split(
                self.df_known_for_train_val, test_size=0.2, random_state=self.config.RANDOM_STATE)
    def _tokenize_datasets(self):
        raw_ds = DatasetDict({'train': HFDataset.from_pandas(self.train_df_final), 'validation': HFDataset.from_pandas(self.val_df_final)})
        def tok_fn(ex): return self.tokenizer([preprocess_text_for_roberta(t) for t in ex[self.config.TEXT_COLUMN]], truncation=True, padding=False, max_length=self.config.MAX_LENGTH)
        self.tokenized_train_val_datasets = raw_ds.map(tok_fn, batched=True, num_proc=max(1,self.config.NUM_WORKERS//2), remove_columns=[c for c in raw_ds['train'].column_names if c not in ['label','input_ids','attention_mask']])
        self.tokenized_train_val_datasets.set_format(type='torch', columns=['input_ids','attention_mask','label'])
    def train_dataloader(self): return DataLoader(self.tokenized_train_val_datasets['train'], batch_size=self.config.BATCH_SIZE, collate_fn=self.data_collator, num_workers=self.config.NUM_WORKERS, shuffle=True, pin_memory=True, persistent_workers=self.config.NUM_WORKERS > 0)
    def val_dataloader(self): return DataLoader(self.tokenized_train_val_datasets['validation'], batch_size=self.config.BATCH_SIZE, collate_fn=self.data_collator, num_workers=self.config.NUM_WORKERS, pin_memory=True, persistent_workers=self.config.NUM_WORKERS > 0)
    def get_full_dataframe(self):
        if self.df_full is None: self.setup()
        return self.df_full

class UnifiedModel(pl.LightningModule):
    # ... (Implementation from oe2.py) ...
    def __init__(self, config: Config, num_labels: int, label2id: dict, id2label: dict, class_weights=None):
        super().__init__(); self.config_params = config; self.save_hyperparameters(ignore=['config_params', 'class_weights'])
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config_params.MODEL_NAME, num_labels=num_labels, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True, output_attentions=True, output_hidden_states=True)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights) if self.config_params.USE_WEIGHTED_LOSS and class_weights is not None else nn.CrossEntropyLoss()
        metrics = torchmetrics.MetricCollection({'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=num_labels), 'f1_weighted': torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='weighted'), 'f1_macro': torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='macro')})
        self.train_metrics = metrics.clone(prefix='train_'); self.val_metrics = metrics.clone(prefix='val_'); self.val_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_labels)
    def setup(self, stage=None):
        if self.config_params.USE_WEIGHTED_LOSS and hasattr(self.loss_fn, 'weight') and self.loss_fn.weight is not None: self.loss_fn.weight = self.loss_fn.weight.to(self.device)
    def forward(self, batch, output_features=False, output_attentions=False):
        input_ids=batch.get('input_ids'); attention_mask=batch.get('attention_mask')
        if input_ids is None or attention_mask is None: raise ValueError("Batch missing input_ids or attention_mask")
        return self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=output_features, output_attentions=output_attentions)
    def _common_step(self, batch, batch_idx):
        if 'label' in batch: batch['labels'] = batch.pop('label')
        outputs = self.model(**batch); loss = outputs.loss; preds = torch.argmax(outputs.logits, dim=1)
        return loss, preds, batch['labels']
    def training_step(self, batch, batch_idx): loss, preds, labels = self._common_step(batch, batch_idx); self.log_dict(self.train_metrics(preds, labels), on_step=False, on_epoch=True, prog_bar=True); self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True); return loss
    def validation_step(self, batch, batch_idx): loss, preds, labels = self._common_step(batch, batch_idx); self.val_metrics.update(preds, labels); self.val_cm.update(preds, labels); self.log_dict(self.val_metrics, on_epoch=True, prog_bar=True); self.log('val_loss', loss, on_epoch=True, prog_bar=True); return loss
    def on_validation_epoch_end(self):
        try:
            cm_comp = self.val_cm.compute(); class_names = list(self.hparams.id2label.values())
            cm_df = pd.DataFrame(cm_comp.cpu().numpy(), index=class_names, columns=class_names)
            cm_filename = os.path.join(self.config_params.CONFUSION_MATRIX_DIR, f"base_clf_val_cm_epoch_{self.current_epoch}.csv"); cm_df.to_csv(cm_filename)
        except Exception as e: print(f"Error in base CM: {e}")
        finally: self.val_cm.reset()
    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.config_params.LEARNING_RATE)
        if self.config_params.USE_LR_SCHEDULER and self.trainer and hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches > 0:
            steps = self.trainer.estimated_stepping_batches; sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=steps)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}
        return opt

class AttentionAnalyzer:
    # ... (Implementation from oe2.py, ensure process_full_dataset handles exclude_class correctly for analysis scope) ...
    def __init__(self, config: Config, model_pl: UnifiedModel, tokenizer, device):
        self.config = config; self.model_pl = model_pl.to(device); self.model_pl.eval(); self.model_pl.freeze()
        self.tokenizer = tokenizer; self.device = device
    @torch.no_grad()
    def get_word_attention_scores(self, texts: List[str], layer_idx: int = -1) -> List[Dict[str, float]]:
        all_scores = []
        for i in tqdm(range(0, len(texts), self.config.BATCH_SIZE), desc="Word Attention", leave=False):
            all_scores.extend(self._process_attention_batch(texts[i:i+self.config.BATCH_SIZE], layer_idx))
        return all_scores
    def _process_attention_batch(self, batch_texts: List[str], layer_idx: int) -> List[Dict[str, float]]:
        if not batch_texts: return []
        proc_texts = [preprocess_text_for_roberta(t) for t in batch_texts]
        inputs = self.tokenizer(proc_texts, return_tensors='pt', truncation=True, max_length=self.config.MAX_LENGTH, padding=True, return_offsets_mapping=True)
        offsets = inputs.pop('offset_mapping').cpu().numpy(); ids_batch = inputs['input_ids'].cpu().numpy()
        inputs_dev = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad(): outputs = self.model_pl.model(**inputs_dev, output_attentions=True); attentions = outputs.attentions[layer_idx].cpu().numpy()
        batch_word_scores = [self._extract_word_scores_from_attention(attentions[i], ids_batch[i], offsets[i], proc_texts[i]) for i in range(len(batch_texts))]
        del inputs, inputs_dev, outputs, attentions; gc.collect(); return batch_word_scores
    def _extract_word_scores_from_attention(self, att_sample, input_ids, offset_map, orig_text):
        mean_att = np.mean(att_sample, axis=0); cls_att = mean_att[0, :]
        word_sc = defaultdict(list); curr_indices = []; last_end_off = 0
        for j, (tok_id, off) in enumerate(zip(input_ids, offset_map)):
            if off[0] == off[1] or tok_id in self.tokenizer.all_special_ids: continue
            is_cont = (j > 0 and off[0] == last_end_off and tok_id != self.tokenizer.unk_token_id)
            if not is_cont and curr_indices:
                s, e = offset_map[curr_indices[0]][0], offset_map[curr_indices[-1]][1]
                word = orig_text[s:e]; avg_s = np.mean(cls_att[curr_indices])
                if word.strip(): word_sc[word.strip()].append(avg_s)
                curr_indices = []
            curr_indices.append(j); last_end_off = off[1]
        if curr_indices:
            s, e = offset_map[curr_indices[0]][0], offset_map[curr_indices[-1]][1]
            word = orig_text[s:e]; avg_s = np.mean(cls_att[curr_indices])
            if word.strip(): word_sc[word.strip()].append(avg_s)
        return {w: np.mean(s) for w, s in word_sc.items()}
    def extract_top_attention_words(self, word_scores_dict: Dict[str, float]) -> List[str]:
        if not word_scores_dict: return []
        sorted_w = sorted(word_scores_dict.items(), key=lambda x: x[1], reverse=True)
        n_top = max(self.config.MIN_TOP_WORDS, math.ceil(len(sorted_w) * self.config.ATTENTION_TOP_PERCENT))
        stopwords = {'__arg__', '__num__', 'a','an','the','is','was','to','of','for','on','in','at'}
        top_filt = [w for w,s in sorted_w[:n_top] if w.lower() not in stopwords and len(w)>1]
        return top_filt if top_filt else [w for w,s in sorted_w[:n_top]]
    def process_full_dataset(self, df: pd.DataFrame, exclude_class: str = None) -> pd.DataFrame:
        # ... (Implementation from oe2.py, ensuring exclude_class filters df_for_analysis correctly) ...
        print("Processing dataset for attention analysis (base classifier)...")
        df_for_analysis = df[df[self.config.CLASS_COLUMN].str.lower() != exclude_class.lower()].copy() if exclude_class else df.copy()
        if df_for_analysis.empty:
            result_df = df.copy(); result_df['top_attention_words'] = pd.Series([[]]*len(df),index=df.index,dtype=object)
            result_df[self.config.TEXT_COLUMN_IN_OE_FILES] = df[self.config.TEXT_COLUMN]; return result_df
        texts = df_for_analysis[self.config.TEXT_COLUMN].tolist()
        all_word_scores = self.get_word_attention_scores(texts, self.config.ATTENTION_LAYER)
        all_top_words, masked_texts_list = [], []
        for i, (text, word_scores) in enumerate(zip(texts, all_word_scores)):
            top_words = self.extract_top_attention_words(word_scores)
            all_top_words.append(top_words); masked_texts_list.append(create_masked_sentence(text, top_words))
        result_df = df.copy()
        result_df['top_attention_words'] = pd.Series([[]]*len(df),index=df.index,dtype=object)
        result_df[self.config.TEXT_COLUMN_IN_OE_FILES] = df[self.config.TEXT_COLUMN]
        
        analysis_indices = df_for_analysis.index # Get original indices of the analyzed part
        for i, original_idx in enumerate(analysis_indices):
            result_df.loc[original_idx, 'top_attention_words'] = all_top_words[i]
            result_df.loc[original_idx, self.config.TEXT_COLUMN_IN_OE_FILES] = masked_texts_list[i]
        return result_df


class MaskedTextDatasetForMetrics(TorchDataset):
    # ... (Implementation from oe2.py) ...
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.texts=texts; valid_texts=[str(t) if pd.notna(t) else "" for t in texts]
        self.encodings=tokenizer(valid_texts,max_length=max_length,padding='max_length',truncation=True,return_tensors='pt')
    def __len__(self): return len(self.texts)
    def __getitem__(self,idx): return {k: v[idx].clone().detach() for k,v in self.encodings.items()}

class OEExtractor:
    # ... (Implementation from oe2.py, ensure extract_attention_metrics and compute_removed_word_attention
    #      correctly handle the full dataframe for feature extraction vs. filtered for metric calculation/OE selection) ...
    def __init__(self, config: Config, model_pl: UnifiedModel, tokenizer, device):
        self.config=config; self.model_pl=model_pl.to(device); self.model_pl.eval(); self.model_pl.freeze()
        self.tokenizer=tokenizer; self.device=device
    @torch.no_grad()
    def extract_attention_metrics(self, dataloader: DataLoader, original_df: pd.DataFrame = None, exclude_class: str = None) -> Tuple[pd.DataFrame, List[np.ndarray]]:
        att_metrics_list, features_list = [], []
        for batch_enc in tqdm(dataloader, desc="Extracting Metrics/Features", leave=False):
            batch_dev = {k: v.to(self.device) for k,v in batch_enc.items()}
            outputs = self.model_pl.forward(batch_dev, output_features=True, output_attentions=True)
            att_batch = outputs.attentions[-1].cpu().numpy(); feat_batch = outputs.hidden_states[-1][:,0,:].cpu().numpy()
            features_list.extend(list(feat_batch)); ids_batch = batch_enc['input_ids'].cpu().numpy()
            for i in range(len(ids_batch)): att_metrics_list.append(self._compute_attention_metrics(att_batch[i],ids_batch[i]))
        del outputs, att_batch, feat_batch; gc.collect()
        
        # This logic ensures features_list corresponds to original_df, and metrics are correctly nulled for excluded class
        if original_df is not None and exclude_class and len(original_df) == len(att_metrics_list): # Assuming dataloader processed all original_df rows
            excl_mask = original_df[self.config.CLASS_COLUMN].str.lower() == exclude_class.lower()
            default_metrics = {'max_attention':0,'top_k_avg_attention':0,'attention_entropy':0}
            for idx, is_excl in enumerate(excl_mask):
                if is_excl: att_metrics_list[idx] = default_metrics.copy()
        return pd.DataFrame(att_metrics_list), features_list
        
    def _compute_attention_metrics(self, att_sample, input_ids):
        valid_idx = np.where((input_ids!=self.tokenizer.pad_token_id)&(input_ids!=self.tokenizer.cls_token_id)&(input_ids!=self.tokenizer.sep_token_id))[0]
        if len(valid_idx)==0: return {'max_attention':0,'top_k_avg_attention':0,'attention_entropy':0}
        cls_att = np.mean(att_sample[:,0,:],axis=0)[valid_idx]
        max_a = np.max(cls_att) if len(cls_att)>0 else 0; k=min(self.config.TOP_K_ATTENTION,len(cls_att))
        top_k_avg = np.mean(np.sort(cls_att)[-k:]) if k>0 else 0
        att_p = F.softmax(torch.tensor(cls_att),dim=0).numpy(); att_e = entropy(att_p) if len(att_p)>1 else 0
        return {'max_attention':max_a,'top_k_avg_attention':top_k_avg,'attention_entropy':att_e}
    def compute_removed_word_attention(self, df: pd.DataFrame, attention_analyzer: AttentionAnalyzer, exclude_class: str = None) -> pd.DataFrame:
        # ... (Implementation from oe2.py, ensure exclude_class filters df_for_processing correctly) ...
        if 'top_attention_words' not in df.columns: df['removed_avg_attention']=0.0; return df
        df_proc = df[df[self.config.CLASS_COLUMN].str.lower() != exclude_class.lower()].copy() if exclude_class else df.copy()
        if df_proc.empty: df['removed_avg_attention']=0.0; return df
        texts = df_proc[self.config.TEXT_COLUMN].tolist()
        word_att_list = attention_analyzer.get_word_attention_scores(texts)
        
        removed_att_values = np.zeros(len(df)) # Initialize for all rows
        proc_indices = df_proc.index # Original indices of rows being processed
        
        for i, original_idx in enumerate(proc_indices):
            row = df_proc.loc[original_idx] # Use df_proc to get 'top_attention_words'
            top_words = safe_literal_eval(row['top_attention_words'])
            if top_words and i < len(word_att_list):
                word_scores = word_att_list[i]
                scores = [word_scores.get(w,0) for w in top_words]
                removed_att_values[df.index.get_loc(original_idx)] = np.mean(scores) if scores else 0 # Assign to correct pos in original df
        df['removed_avg_attention'] = removed_att_values
        return df

    def extract_oe_datasets(self, df: pd.DataFrame, exclude_class: str = None) -> None:
        # ... (Implementation from oe2.py, ensure exclude_class filters df_for_oe correctly) ...
        df_oe = df[df[self.config.CLASS_COLUMN].str.lower() != exclude_class.lower()].copy() if exclude_class else df.copy()
        if df_oe.empty: return
        for metric, settings in self.config.METRIC_SETTINGS.items():
            if metric not in df_oe.columns: continue
            self._extract_single_metric_oe(df_oe, metric, settings)
        self._extract_sequential_filtering_oe(df_oe)
    def _extract_single_metric_oe(self, df: pd.DataFrame, metric: str, settings: dict):
        # ... (Implementation from oe2.py) ...
        scores = np.nan_to_num(df[metric].values, nan=0.0)
        thresh = np.percentile(scores, 100-settings['percentile'] if settings['mode']=='higher' else settings['percentile'])
        sel_idx = np.where(scores >= thresh if settings['mode']=='higher' else scores <= thresh)[0]
        if len(sel_idx)>0:
            oe_simple = df.iloc[sel_idx][[self.config.TEXT_COLUMN_IN_OE_FILES]].copy()
            mode_desc = f"{settings['mode']}{settings['percentile']}pct"
            fname = os.path.join(self.config.OE_DATA_DIR, f"oe_data_{metric}_{mode_desc}.csv")
            oe_simple.to_csv(fname, index=False)
            print(f"Saved OE: {fname} ({len(oe_simple)} samples)")
    def _extract_sequential_filtering_oe(self, df: pd.DataFrame):
        # ... (Implementation from oe2.py) ...
        sel_mask = np.ones(len(df), dtype=bool)
        for step, (metric, settings) in enumerate(self.config.FILTERING_SEQUENCE):
            if metric not in df.columns: continue
            curr_sel_df = df[sel_mask];
            if curr_sel_df.empty: sel_mask[:]=False; break
            scores = np.nan_to_num(curr_sel_df[metric].values, nan=0.0)
            thresh = np.percentile(scores, 100-settings['percentile'] if settings['mode']=='higher' else settings['percentile'])
            step_m = scores >= thresh if settings['mode']=='higher' else scores <= thresh
            curr_idx = np.where(sel_mask)[0]; idx_keep = curr_idx[step_m]
            sel_mask = np.zeros_like(sel_mask)
            if len(idx_keep)>0: sel_mask[idx_keep]=True
        final_idx = np.where(sel_mask)[0]
        if len(final_idx)>0:
            oe_simple = df.iloc[final_idx][[self.config.TEXT_COLUMN_IN_OE_FILES]].copy()
            f_desc = "_".join([f"{m}_{s['mode']}{s['percentile']}" for m,s in self.config.FILTERING_SEQUENCE])
            fname = os.path.join(self.config.OE_DATA_DIR, f"oe_data_sequential_{f_desc}.csv")
            oe_simple.to_csv(fname, index=False)
            print(f"Saved Seq OE: {fname} ({len(oe_simple)} samples)")

class Visualizer:
    # ... (Implementation from oe2.py) ...
    def __init__(self, config: Config): self.config = config
    def plot_metric_distribution(self, scores: np.ndarray, metric_name: str, title: str, save_path: str):
        if len(scores)==0: return
        plt.figure(figsize=(10,6));
        if SNS_AVAILABLE: sns.histplot(scores,bins=50,kde=True,stat='density')
        else: plt.hist(scores,bins=50,density=True,alpha=0.7)
        plt.title(title); plt.xlabel(metric_name); plt.ylabel('Density'); plt.grid(alpha=0.3)
        mean_v=np.mean(scores); plt.axvline(mean_v,color='r',ls='--',label=f'Mean: {mean_v:.4f}')
        plt.legend(); plt.tight_layout(); plt.savefig(save_path,dpi=300,bbox_inches='tight'); plt.close()
    def plot_tsne(self, features: np.ndarray, labels: np.ndarray, title: str, save_path: str,
                  highlight_indices: Optional[np.ndarray]=None, highlight_label:str='OE Candidate',
                  class_names:Optional[Dict]=None,seed:int=42):
        if len(features)==0:return
        try:
            perp = min(30, features.shape[0]-1)
            if perp<=1: print(f"tSNE perplexity too low ({perp}). Skipping."); return
            tsne_res = TSNE(n_components=2,random_state=seed,perplexity=perp,max_iter=1000,init='pca',learning_rate='auto').fit_transform(features)
        except Exception as e: print(f"Error tSNE: {e}"); return
        df_tsne = pd.DataFrame(tsne_res, columns=['tsne1','tsne2']); df_tsne['label']=labels
        df_tsne['is_highlighted']=False
        if highlight_indices is not None: df_tsne.loc[highlight_indices,'is_highlighted']=True
        plt.figure(figsize=(14,10)); unique_labels=sorted(df_tsne['label'].unique())
        colors = plt.cm.tab20(np.linspace(0,1,len(unique_labels)))
        for i,lbl_val in enumerate(unique_labels):
            subset = df_tsne[(df_tsne['label']==lbl_val)&(~df_tsne['is_highlighted'])]
            if len(subset)>0:
                cname=class_names.get(lbl_val,f'Class {lbl_val}') if class_names else f'Class {lbl_val}'
                plt.scatter(subset['tsne1'],subset['tsne2'],color=colors[i],label=cname,alpha=0.7,s=30)
        if highlight_indices is not None and len(df_tsne[df_tsne['is_highlighted']])>0:
            plt.scatter(df_tsne[df_tsne['is_highlighted']]['tsne1'],df_tsne[df_tsne['is_highlighted']]['tsne2'],color='r',marker='x',s=100,label=highlight_label,alpha=0.9)
        plt.title(title,fontsize=16,pad=20); plt.xlabel("t-SNE Dim 1"); plt.ylabel("t-SNE Dim 2")
        plt.grid(alpha=0.3,ls='--'); plt.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=10)
        plt.tight_layout(); plt.subplots_adjust(right=0.75); plt.savefig(save_path,dpi=300,bbox_inches='tight'); plt.close()
    def visualize_all_metrics(self, df: pd.DataFrame):
        metrics=['max_attention','top_k_avg_attention','attention_entropy','removed_avg_attention']
        for m in metrics:
            if m in df.columns and not df[m].isnull().all():
                self.plot_metric_distribution(df[m].dropna().values,m,f'Distribution of {m}',os.path.join(self.config.VIS_DIR,f'{m}_distribution.png'))
    def visualize_oe_candidates(self, df:pd.DataFrame, features:List[np.ndarray], label2id:dict, id2label:dict):
        if not features or len(features)!=len(df): print(f"Feature/DF mismatch for OE tSNE. Feat:{len(features)}, DF:{len(df)}"); return
        tsne_labels=[]; unk_lower=self.config.EXCLUDE_CLASS_FOR_TRAINING.lower()
        for cls_val in df[self.config.CLASS_COLUMN]:
            cls_str=str(cls_val).lower()
            tsne_labels.append(-1 if cls_str==unk_lower else label2id.get(cls_str,-2))
        tsne_labels_np=np.array(tsne_labels)
        class_names_viz={**{k:str(v) for k,v in id2label.items()},-1:'Unknown (Excluded)',-2:'Other/Filtered'}
        for metric,settings in self.config.METRIC_SETTINGS.items():
            if metric not in df.columns:continue
            scores=np.nan_to_num(df[metric].values,nan=0.0)
            thresh=np.percentile(scores,100-settings['percentile'] if settings['mode']=='higher' else settings['percentile'])
            oe_idx=np.where(scores>=thresh if settings['mode']=='higher' else scores<=thresh)[0]
            mode_desc=f"{settings['mode']}{settings['percentile']}%"
            self.plot_tsne(np.array(features),tsne_labels_np,f't-SNE (OE Viz): Candidates by {metric} ({mode_desc})',os.path.join(self.config.VIS_DIR,f'tsne_oe_cand_{metric}_{mode_desc}.png'),highlight_indices=oe_idx,highlight_label=f'OE Candidate ({metric} {mode_desc})',class_names=class_names_viz,seed=self.config.RANDOM_STATE)
        if hasattr(self.config,'FILTERING_SEQUENCE') and self.config.FILTERING_SEQUENCE:
            sel_mask=np.ones(len(df),dtype=bool); filt_steps_desc=[]
            for step,(metric,settings) in enumerate(self.config.FILTERING_SEQUENCE):
                if metric not in df.columns:continue
                curr_sel_df=df[sel_mask];
                if curr_sel_df.empty:break
                scores=np.nan_to_num(curr_sel_df[metric].values,nan=0.0)
                thresh=np.percentile(scores,100-settings['percentile'] if settings['mode']=='higher' else settings['percentile'])
                step_m_sub=scores>=thresh if settings['mode']=='higher' else scores<=thresh
                curr_idx=np.where(sel_mask)[0]; idx_keep=curr_idx[step_m_sub]; sel_mask=np.zeros_like(sel_mask)
                if len(idx_keep)>0:sel_mask[idx_keep]=True
                filt_steps_desc.append(f"{metric}({settings['mode']}{settings['percentile']}%)")
            final_idx_seq=np.where(sel_mask)[0]
            if len(final_idx_seq)>0:
                seq_desc=" -> ".join(filt_steps_desc)
                self.plot_tsne(np.array(features),tsne_labels_np,f't-SNE (OE Viz): Sequential Filter Candidates\n{seq_desc} -> {len(final_idx_seq)} samples',os.path.join(self.config.VIS_DIR,f'tsne_oe_cand_sequential_{"_".join(filt_steps_desc)}.png'),highlight_indices=final_idx_seq,highlight_label=f'Sequential OE Candidate ({len(final_idx_seq)} samples)',class_names=class_names_viz,seed=self.config.RANDOM_STATE)

# === 메인 파이프라인 클래스 ===
class UnifiedOEExtractor:
    # ... (Implementation from oe2.py, with Stage 5 modifications) ...
    def __init__(self, config: Config):
        self.config=config; self.data_module:Optional[UnifiedDataModule]=None; self.model:Optional[UnifiedModel]=None
        self.attention_analyzer:Optional[AttentionAnalyzer]=None; self.oe_extractor:Optional[OEExtractor]=None
        self.visualizer=Visualizer(config); config.create_directories(); config.save_config(); set_seed(config.RANDOM_STATE)
    def run_stage1_model_training(self):
        if not self.config.STAGE_MODEL_TRAINING:
            if self._check_existing_model(): self._load_existing_model()
            else: print("Error: Stage 1 skipped, no existing model."); sys.exit(1)
            return
        print("\n" + "="*50 + "\nSTAGE 1: BASE MODEL TRAINING\n" + "="*50)
        self.data_module=UnifiedDataModule(self.config); self.data_module.setup()
        self.model=UnifiedModel(config=self.config,num_labels=self.data_module.num_labels,label2id=self.data_module.label2id,id2label=self.data_module.id2label,class_weights=self.data_module.class_weights)
        monitor='val_f1_macro'; ckpt_cb=ModelCheckpoint(dirpath=self.config.MODEL_SAVE_DIR,filename=f'base-clf-{{epoch:02d}}-{{{monitor}:.4f}}',save_top_k=1,monitor=monitor,mode='max')
        es_cb=EarlyStopping(monitor=monitor,patience=3,mode='max',verbose=True); csv_log=CSVLogger(save_dir=self.config.LOG_DIR,name="base_model_training")
        if self._check_existing_model() and self.config.OSR_EVAL_ONLY: self._load_existing_model(ckpt_cb)
        else:
            trainer=pl.Trainer(max_epochs=self.config.NUM_TRAIN_EPOCHS,accelerator=self.config.ACCELERATOR,devices=self.config.DEVICES,precision=self.config.PRECISION,logger=csv_log,callbacks=[ckpt_cb,es_cb],log_every_n_steps=self.config.LOG_EVERY_N_STEPS,gradient_clip_val=self.config.GRADIENT_CLIP_VAL)
            trainer.fit(self.model,datamodule=self.data_module); self._load_best_model(ckpt_cb)
    def run_stage2_attention_extraction(self) -> Optional[pd.DataFrame]:
        if not self.config.STAGE_ATTENTION_EXTRACTION:
            if self.config.STAGE_OE_EXTRACTION or self.config.STAGE_VISUALIZATION:
                try: return self._load_attention_results()
                except FileNotFoundError: return None
            return None
        print("\n" + "="*50 + "\nSTAGE 2: ATTENTION EXTRACTION\n" + "="*50)
        if self.model is None: self._load_existing_model()
        if self.data_module is None: self.data_module=UnifiedDataModule(self.config); self.data_module.setup()
        curr_dev = self.model.device if hasattr(self.model,'device') else DEVICE_OSR
        self.attention_analyzer = AttentionAnalyzer(config=self.config,model_pl=self.model,tokenizer=self.data_module.tokenizer,device=curr_dev)
        full_df = self.data_module.get_full_dataframe()
        proc_df = self.attention_analyzer.process_full_dataset(full_df,exclude_class=self.config.EXCLUDE_CLASS_FOR_TRAINING)
        out_path = os.path.join(self.config.ATTENTION_DATA_DIR, "df_with_attention.csv"); proc_df.to_csv(out_path,index=False)
        self._print_attention_samples(proc_df); return proc_df
    def run_stage3_oe_extraction(self, df_with_attention: Optional[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], Optional[List[np.ndarray]]]:
        if not self.config.STAGE_OE_EXTRACTION:
            if self.config.STAGE_VISUALIZATION or self.config.STAGE_OSR_EXPERIMENTS:
                try: return self._load_final_metrics_and_features()
                except FileNotFoundError: return None,None
            return None,None
        print("\n" + "="*50 + "\nSTAGE 3: OE EXTRACTION (FIXED)\n" + "="*50) # Updated print
        if df_with_attention is None: df_with_attention = self._load_attention_results()
        if df_with_attention is None: return None,None
        if self.model is None: self._load_existing_model()
        if self.data_module is None: self.data_module=UnifiedDataModule(self.config); self.data_module.setup()
        curr_dev = self.model.device if hasattr(self.model,'device') else DEVICE_OSR
        self.oe_extractor = OEExtractor(config=self.config,model_pl=self.model,tokenizer=self.data_module.tokenizer,device=curr_dev)
        
        all_texts = [] # For feature extraction, use appropriate text (masked or original)
        masked_col = self.config.TEXT_COLUMN_IN_OE_FILES
        unk_lower = self.config.EXCLUDE_CLASS_FOR_TRAINING.lower()
        for _, row in df_with_attention.iterrows():
            all_texts.append(row[self.config.TEXT_COLUMN] if str(row[self.config.CLASS_COLUMN]).lower()==unk_lower else row[masked_col])
            
        dataset = MaskedTextDatasetForMetrics(all_texts, self.data_module.tokenizer, self.config.MAX_LENGTH)
        dataloader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE, num_workers=self.config.NUM_WORKERS, shuffle=False)
        
        att_metrics_df, features = self.oe_extractor.extract_attention_metrics(dataloader, original_df=df_with_attention, exclude_class=self.config.EXCLUDE_CLASS_FOR_TRAINING)
        assert len(df_with_attention)==len(att_metrics_df)==len(features), "Length mismatch post metrics/feature extraction"
        
        df_w_metrics = pd.concat([df_with_attention.reset_index(drop=True), att_metrics_df.reset_index(drop=True)], axis=1)
        if self.attention_analyzer: df_w_metrics = self.oe_extractor.compute_removed_word_attention(df_w_metrics, self.attention_analyzer, exclude_class=self.config.EXCLUDE_CLASS_FOR_TRAINING)
        self.oe_extractor.extract_oe_datasets(df_w_metrics, exclude_class=self.config.EXCLUDE_CLASS_FOR_TRAINING)
        
        metrics_out_path = os.path.join(self.config.ATTENTION_DATA_DIR, "df_with_all_metrics.csv"); df_w_metrics.to_csv(metrics_out_path,index=False)
        if features: np.save(os.path.join(self.config.ATTENTION_DATA_DIR,"extracted_features.npy"),np.array(features))
        return df_w_metrics, features
    def _plot_training_curve(self, losses:List[float], exp_name:str, save_dir:str):
        plt.figure(figsize=(10,6)); plt.plot(range(1,len(losses)+1),losses,'b-',label='Training Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.title(f'Training Loss Curve - {exp_name}'); plt.grid(True,alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir,f'{exp_name}_training_curve.png'),dpi=300,bbox_inches='tight'); plt.close()
    def run_stage4_visualization(self, df_with_metrics:Optional[pd.DataFrame], features:Optional[List[np.ndarray]]):
        if not self.config.STAGE_VISUALIZATION: return
        print("\n" + "="*50 + "\nSTAGE 4: OE EXTRACTION VISUALIZATION\n" + "="*50)
        if df_with_metrics is None or features is None: df_with_metrics, features = self._load_final_metrics_and_features()
        if df_with_metrics is None: return
        self.visualizer.visualize_all_metrics(df_with_metrics)
        if features and self.data_module: self.visualizer.visualize_oe_candidates(df_with_metrics,features,self.data_module.label2id,self.data_module.id2label)
    def _run_single_osr_experiment(self, osr_tokenizer, num_osr_classes:int, osr_id_label2id:Dict, osr_id_id2label:Dict, osr_known_class_names:List[str], id_train_loader_osr:DataLoader, id_test_loader_osr:DataLoader, ood_eval_loader_osr:Optional[DataLoader], current_oe_source_name:Optional[str], current_oe_data_path:Optional[str], ood_dataset_eval_name_tag:str, is_external_oe:bool=False) -> Tuple[Dict,Dict]:
        exp_tag_base = "ExternalOE" if is_external_oe else "AttentionOE"
        exp_tag = f"SyslogOSR_{exp_tag_base}_{current_oe_source_name}" if current_oe_source_name else f"SyslogOSR_Standard"
        
        print(f"\n\n===== Starting Single OSR Experiment: {exp_tag} =====")
        sanitized_oe_name = re.sub(r'[^\w\-.]+', '_', current_oe_source_name) if current_oe_source_name else "Standard"
        exp_res_subdir = os.path.join("SyslogOSR", f"{exp_tag_base}_{sanitized_oe_name}" if current_oe_source_name else "Standard")
        
        curr_res_dir = os.path.join(self.config.OSR_RESULT_DIR, exp_res_subdir); os.makedirs(curr_res_dir, exist_ok=True)
        curr_model_dir = os.path.join(self.config.OSR_MODEL_DIR, exp_res_subdir)
        if self.config.OSR_SAVE_MODEL_PER_EXPERIMENT: os.makedirs(curr_model_dir, exist_ok=True)

        model_osr = RoBERTaOOD(num_osr_classes, self.config.OSR_MODEL_TYPE).to(DEVICE_OSR)
        if osr_id_label2id and osr_id_id2label: model_osr.roberta.config.label2id=osr_id_label2id; model_osr.roberta.config.id2label=osr_id_id2label
        
        model_fname = f"roberta_osr_{sanitized_oe_name}_{num_osr_classes}cls_seed{self.config.RANDOM_STATE}.pt"
        model_save_p = os.path.join(curr_model_dir, model_fname)
        exp_results, exp_data_plots = {}, {}
        epoch_losses = []

        if self.config.OSR_EVAL_ONLY:
            if os.path.exists(model_save_p): model_osr.load_state_dict(torch.load(model_save_p, map_location=DEVICE_OSR))
            else: return {}, {}
        else:
            opt = AdamW(model_osr.parameters(), lr=self.config.OSR_LEARNING_RATE)
            total_s = len(id_train_loader_osr) * self.config.OSR_NUM_EPOCHS
            sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(self.config.OSR_WARMUP_RATIO*total_s), num_training_steps=total_s)
            pat, min_d = self.config.OSR_EARLY_STOPPING_PATIENCE, self.config.OSR_EARLY_STOPPING_MIN_DELTA
            
            oe_train_dataset_osr = None
            if current_oe_data_path: # Path to a generated OE CSV
                oe_train_dataset_osr = prepare_generated_oe_data_for_osr(osr_tokenizer, self.config.OSR_MAX_LENGTH, current_oe_data_path, self.config.TEXT_COLUMN_IN_OE_FILES)
            elif is_external_oe and current_oe_source_name: # Name of an external dataset
                 oe_train_dataset_osr = prepare_external_oe_data_for_osr(osr_tokenizer, self.config.OSR_MAX_LENGTH, current_oe_source_name, self.config.DATA_DIR_EXTERNAL_HF, self.config.CACHE_DIR_HF)

            if oe_train_dataset_osr:
                oe_loader = DataLoader(oe_train_dataset_osr, batch_size=self.config.OSR_BATCH_SIZE, shuffle=True, num_workers=self.config.OSR_NUM_DATALOADER_WORKERS, pin_memory=True, persistent_workers=self.config.OSR_NUM_DATALOADER_WORKERS > 0)
                epoch_losses = self._train_with_oe_uniform_loss_osr_with_early_stopping(model_osr, id_train_loader_osr, oe_loader, opt, sched, DEVICE_OSR, self.config.OSR_NUM_EPOCHS, self.config.OSR_OE_LAMBDA, exp_tag, pat, min_d)
                del oe_train_dataset_osr, oe_loader; gc.collect()
            else: # Standard training or OE data prep failed
                epoch_losses = self._train_standard_osr_with_early_stopping(model_osr, id_train_loader_osr, opt, sched, DEVICE_OSR, self.config.OSR_NUM_EPOCHS, exp_tag, pat, min_d)
            
            if self.config.OSR_SAVE_MODEL_PER_EXPERIMENT: torch.save(model_osr.state_dict(), model_save_p)
            if not self.config.OSR_NO_PLOT_PER_EXPERIMENT and epoch_losses: self._plot_training_curve(epoch_losses, exp_tag, curr_res_dir)

        res_osr, data_osr = evaluate_osr(model_osr,id_test_loader_osr,ood_eval_loader_osr,DEVICE_OSR,self.config.OSR_TEMPERATURE,self.config.OSR_THRESHOLD_PERCENTILE,return_data=True)
        metric_key = f"{exp_tag}+{ood_dataset_eval_name_tag}"
        exp_results[metric_key] = res_osr; exp_data_plots[metric_key] = data_osr
        
        if not self.config.OSR_NO_PLOT_PER_EXPERIMENT:
            plot_prefix = re.sub(r'[^\w\-]+','_',metric_key)
            if data_osr['id_scores'] is not None and data_osr['ood_scores'] is not None and len(data_osr['ood_scores'])>0:
                plot_confidence_histograms_osr(data_osr['id_scores'],data_osr['ood_scores'],f'Conf - {exp_tag} vs {ood_dataset_eval_name_tag}',os.path.join(curr_res_dir,f'{plot_prefix}_hist.png'))
                plot_roc_curve_osr(data_osr['id_scores'],data_osr['ood_scores'],f'ROC - {exp_tag} vs {ood_dataset_eval_name_tag}',os.path.join(curr_res_dir,f'{plot_prefix}_roc.png'))
                plot_tsne_osr(data_osr['id_features'],data_osr['ood_features'],f't-SNE - {exp_tag} (ID vs OOD: {ood_dataset_eval_name_tag})',os.path.join(curr_res_dir,f'{plot_prefix}_tsne.png'),seed=self.config.RANDOM_STATE)
            if data_osr['id_labels_true'] is not None and len(data_osr['id_labels_true'])>0 and num_osr_classes>0:
                cm = confusion_matrix(data_osr['id_labels_true'],data_osr['id_labels_pred'],labels=np.arange(num_osr_classes))
                plot_confusion_matrix_osr(cm,osr_known_class_names,f'CM - {exp_tag} (ID Test)',os.path.join(curr_res_dir,f'{plot_prefix}_cm.png'))
        del model_osr; gc.collect(); torch.cuda.empty_cache(); return exp_results, exp_data_plots
    def run_stage5_osr_experiments(self):
        if not self.config.STAGE_OSR_EXPERIMENTS: print("Skipping Stage 5: OSR Experiments"); return
        print("\n" + "="*50 + "\nSTAGE 5: OSR EXPERIMENTS\n" + "="*50)
        if self.data_module is None or self.data_module.num_labels is None:
            if not self.config.STAGE_MODEL_TRAINING: self.data_module=UnifiedDataModule(self.config); self.data_module.setup()
            if self.data_module.num_labels is None: print("Critical Error: DataModule setup failed."); return
        
        osr_tok = RobertaTokenizer.from_pretrained(self.config.OSR_MODEL_TYPE)
        id_train_ds,id_test_ds,num_cls,_,id_l2i,id_i2l = prepare_id_data_for_osr(self.data_module,osr_tok,self.config.OSR_MAX_LENGTH)
        if id_train_ds is None or num_cls==0: print("Error: Failed to prep ID data for OSR."); return
        
        known_cls_names = list(id_i2l.values()) if id_i2l else [f"C_{i}" for i in range(num_cls)]
        id_train_load = DataLoader(id_train_ds,batch_size=self.config.OSR_BATCH_SIZE,shuffle=True,num_workers=self.config.OSR_NUM_DATALOADER_WORKERS,pin_memory=True,persistent_workers=self.config.OSR_NUM_DATALOADER_WORKERS > 0)
        id_test_load = DataLoader(id_test_ds,batch_size=self.config.OSR_BATCH_SIZE,num_workers=self.config.OSR_NUM_DATALOADER_WORKERS,pin_memory=True)
        ood_eval_ds = prepare_syslog_ood_data_for_osr(osr_tok,self.config.OSR_MAX_LENGTH,self.config.OOD_SYSLOG_UNKNOWN_PATH_OSR,self.config.TEXT_COLUMN,self.config.CLASS_COLUMN,self.config.OOD_TARGET_CLASS_OSR)
        ood_eval_load = DataLoader(ood_eval_ds,batch_size=self.config.OSR_BATCH_SIZE,num_workers=self.config.OSR_NUM_DATALOADER_WORKERS,pin_memory=True) if ood_eval_ds else None
        ood_tag = self.config.OOD_TARGET_CLASS_OSR
        all_osr_res = {}

        if not self.config.OSR_SKIP_STANDARD_MODEL:
            std_res,_ = self._run_single_osr_experiment(osr_tok,num_cls,id_l2i,id_i2l,known_cls_names,id_train_load,id_test_load,ood_eval_load,None,None,ood_tag)
            all_osr_res.update(std_res)
        
        if not self.config.OSR_SKIP_ATTENTION_OE_MODELS:
            print(f"\n--- OSR with Attention-based OE from: {self.config.OE_DATA_DIR} ---")
            oe_files = [f for f in os.listdir(self.config.OE_DATA_DIR) if f.endswith('.csv') and 'extended' not in f]
            for oe_f in oe_files:
                oe_path = os.path.join(self.config.OE_DATA_DIR,oe_f); oe_name=os.path.splitext(oe_f)[0]
                att_oe_res,_ = self._run_single_osr_experiment(osr_tok,num_cls,id_l2i,id_i2l,known_cls_names,id_train_load,id_test_load,ood_eval_load,oe_name,oe_path,ood_tag,is_external_oe=False)
                all_osr_res.update(att_oe_res)
        
        if not self.config.OSR_SKIP_EXTERNAL_OE_MODELS and self.config.OSR_OE_SOURCES_EXTERNAL:
            print(f"\n--- OSR with External OE sources: {self.config.OSR_OE_SOURCES_EXTERNAL} ---")
            for ext_oe_src_full in self.config.OSR_OE_SOURCES_EXTERNAL:
                # Use full name (e.g., wikitext:wikitext-2-raw-v1) as oe_name for uniqueness
                ext_oe_res,_ = self._run_single_osr_experiment(osr_tok,num_cls,id_l2i,id_i2l,known_cls_names,id_train_load,id_test_load,ood_eval_load,ext_oe_src_full,None,ood_tag,is_external_oe=True)
                all_osr_res.update(ext_oe_res)

        if all_osr_res:
            final_df = pd.DataFrame.from_dict(all_osr_res,orient='index').sort_index()
            print("\nOverall OSR Performance Metrics:\n", final_df)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S"); base_name = f"osr_overall_summary_SyslogOSR_{ts}"
            final_df.to_csv(os.path.join(self.config.OSR_RESULT_DIR,f'{base_name}.csv'))
            osr_args = {k:v for k,v in self.config.__class__.__dict__.items() if k.startswith('OSR_') or k in ['RANDOM_STATE','TEXT_COLUMN','CLASS_COLUMN','ORIGINAL_DATA_PATH','OE_DATA_DIR_USED']}
            with open(os.path.join(self.config.OSR_RESULT_DIR,f'{base_name}.txt'),'w') as f:
                f.write("--- OSR Args ---\n" + json.dumps(osr_args,indent=2,default=str) + "\n\n--- OSR Metrics ---\n" + final_df.to_string())
            with open(os.path.join(self.config.OSR_RESULT_DIR,f'{base_name}.json'),'w') as f:
                json.dump({'arguments_osr':osr_args,'timestamp':ts,'results_osr':all_osr_res},f,indent=2,default=str)
        print("\nOSR Experiments Finished.")
    def run_full_pipeline(self):
        print("Starting Unified OE Extraction & OSR Pipeline...")
        df_att, df_metrics, feats = None,None,None
        self.run_stage1_model_training()
        df_att = self.run_stage2_attention_extraction()
        df_metrics, feats = self.run_stage3_oe_extraction(df_att)
        self.run_stage4_visualization(df_metrics, feats)
        self.run_stage5_osr_experiments()
        self._print_final_summary()
        print("\nUnified OE Extraction & OSR Pipeline Complete!")
    def _train_standard_osr_with_early_stopping(self, model:nn.Module,train_loader:DataLoader,optimizer:optim.Optimizer,scheduler,device:torch.device,num_epochs:int,exp_name:str,patience:int=5,min_delta:float=0.001):
        model.train(); use_amp=(device.type=='cuda'); scaler=torch.cuda.amp.GradScaler(enabled=use_amp)
        best_loss=float('inf'); pat_count=0; epoch_losses=[]
        for ep in range(num_epochs):
            tot_loss=0; prog_bar=tqdm(train_loader,desc=f"Std OSR Ep {ep+1}/{num_epochs} ({exp_name})",leave=False)
            for batch in prog_bar:
                ids,mask,lbls = batch['input_ids'].to(device),batch['attention_mask'].to(device),batch['label'].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=str(device).split(':')[0],enabled=use_amp):
                    logits=model(ids,mask); loss=F.cross_entropy(logits,lbls)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); tot_loss+=loss.item()
                prog_bar.set_postfix({'loss':f"{loss.item():.3f}"})
            if scheduler:scheduler.step()
            avg_loss=tot_loss/len(train_loader); epoch_losses.append(avg_loss)
            if avg_loss<best_loss-min_delta: best_loss=avg_loss; pat_count=0
            else: pat_count+=1
            if pat_count>=patience: print(f"Early stopping at epoch {ep+1}"); break
        return epoch_losses
    def _train_with_oe_uniform_loss_osr_with_early_stopping(self, model:nn.Module,train_loader:DataLoader,oe_loader:DataLoader,optimizer:optim.Optimizer,scheduler,device:torch.device,num_epochs:int,oe_lambda:float,exp_name:str,patience:int=5,min_delta:float=0.001):
        model.train(); use_amp=(device.type=='cuda'); scaler=torch.cuda.amp.GradScaler(enabled=use_amp)
        best_loss=float('inf'); pat_count=0; epoch_losses=[]
        for ep in range(num_epochs):
            oe_iter=iter(oe_loader); tot_loss,tot_id_loss,tot_oe_loss=0,0,0
            prog_bar=tqdm(train_loader,desc=f"OE OSR Ep {ep+1}/{num_epochs} ({exp_name})",leave=False)
            for batch in prog_bar:
                ids,mask,lbls=batch['input_ids'].to(device),batch['attention_mask'].to(device),batch['label'].to(device)
                try: oe_batch=next(oe_iter)
                except StopIteration: oe_iter=iter(oe_loader); oe_batch=next(oe_iter)
                oe_ids,oe_mask=oe_batch['input_ids'].to(device),oe_batch['attention_mask'].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=str(device).split(':')[0],enabled=use_amp):
                    id_logits=model(ids,mask); id_loss=F.cross_entropy(id_logits,lbls)
                    oe_logits=model(oe_ids,oe_mask); n_cls=oe_logits.size(1)
                    log_sm_oe=F.log_softmax(oe_logits,dim=1); uni_targ=torch.full_like(oe_logits,1.0/n_cls)
                    oe_loss=F.kl_div(log_sm_oe,uni_targ,reduction='batchmean',log_target=False)
                    batch_loss=id_loss+oe_lambda*oe_loss
                scaler.scale(batch_loss).backward(); scaler.step(optimizer); scaler.update()
                tot_loss+=batch_loss.item(); tot_id_loss+=id_loss.item(); tot_oe_loss+=oe_loss.item()
                prog_bar.set_postfix({'Total':f"{batch_loss.item():.3f}",'ID':f"{id_loss.item():.3f}",'OE':f"{oe_loss.item():.3f}"})
            if scheduler:scheduler.step()
            avg_loss=tot_loss/len(train_loader); epoch_losses.append(avg_loss)
            if avg_loss<best_loss-min_delta: best_loss=avg_loss; pat_count=0
            else: pat_count+=1
            if pat_count>=patience: print(f"Early stopping at epoch {ep+1}"); break
        return epoch_losses
    def _check_existing_model(self) -> bool: return os.path.exists(self.config.MODEL_SAVE_DIR) and any(f.endswith('.ckpt') for f in os.listdir(self.config.MODEL_SAVE_DIR))
    def _load_existing_model(self, ckpt_cb=None):
        if self.data_module is None: self.data_module=UnifiedDataModule(self.config); self.data_module.setup()
        model_p = None
        if ckpt_cb and hasattr(ckpt_cb,'best_model_path') and ckpt_cb.best_model_path: model_p=ckpt_cb.best_model_path
        else:
            files=[f for f in os.listdir(self.config.MODEL_SAVE_DIR) if f.endswith('.ckpt')]
            if files: model_p=os.path.join(self.config.MODEL_SAVE_DIR,sorted(files)[-1])
        if model_p and os.path.exists(model_p):
            self.model=UnifiedModel.load_from_checkpoint(model_p,config=self.config,num_labels=self.data_module.num_labels,label2id=self.data_module.label2id,id2label=self.data_module.id2label,class_weights=self.data_module.class_weights)
        else: raise FileNotFoundError("Base model checkpoint not found.")
    def _load_best_model(self, ckpt_cb):
        if ckpt_cb.best_model_path and os.path.exists(ckpt_cb.best_model_path):
            if self.data_module is None: self.data_module=UnifiedDataModule(self.config); self.data_module.setup()
            self.model=UnifiedModel.load_from_checkpoint(ckpt_cb.best_model_path,config=self.config,num_labels=self.data_module.num_labels,label2id=self.data_module.label2id,id2label=self.data_module.id2label,class_weights=self.data_module.class_weights)
    def _load_attention_results(self) -> Optional[pd.DataFrame]:
        att_file=os.path.join(self.config.ATTENTION_DATA_DIR,"df_with_attention.csv")
        if os.path.exists(att_file):
            df=pd.read_csv(att_file)
            if 'top_attention_words' in df.columns: df['top_attention_words']=df['top_attention_words'].apply(safe_literal_eval)
            return df
        return None
    def _load_final_metrics_and_features(self) -> Tuple[Optional[pd.DataFrame],Optional[List[np.ndarray]]]:
        met_file=os.path.join(self.config.ATTENTION_DATA_DIR,"df_with_all_metrics.csv")
        feat_file=os.path.join(self.config.ATTENTION_DATA_DIR,"extracted_features.npy")
        df_met,feat_arr = None,None
        if os.path.exists(met_file):
            df_met=pd.read_csv(met_file)
            if 'top_attention_words' in df_met.columns: df_met['top_attention_words']=df_met['top_attention_words'].apply(safe_literal_eval)
        if os.path.exists(feat_file): feat_arr=np.load(feat_file,allow_pickle=True).tolist()
        return df_met,feat_arr
    def _print_attention_samples(self, df:pd.DataFrame, num_samples:int=3):
        if df is None or df.empty: return
        sample_df=df.sample(min(num_samples,len(df)))
        for _,row in sample_df.iterrows():
            print(f"Original: {str(row[self.config.TEXT_COLUMN])[:100]}...")
            print(f"Top Words: {row['top_attention_words']}")
            print(f"Masked: {str(row[self.config.TEXT_COLUMN_IN_OE_FILES])[:100]}...")
    def _print_final_summary(self):
        print("\n" + "="*50 + "\nPIPELINE SUMMARY\n" + "="*50)
        print(f"Output directory: {self.config.OUTPUT_DIR}")
        count=0
        for root,_,files in os.walk(self.config.OUTPUT_DIR):
            if count>=15:break
            for file in files:
                if count>=15:break
                if file.endswith(('.csv','.png','.json','.txt','.pt','.ckpt')): print(f"  - {os.path.join(root,file)}"); count+=1
            if count>=15 and root==self.config.OUTPUT_DIR: print("  ... (many more files generated)")

# === 메인 함수 ===
def main():
    parser = argparse.ArgumentParser(description="Unified OE Extraction and OSR Experimentation Pipeline")
    parser.add_argument('--attention_percent', type=float, default=Config.ATTENTION_TOP_PERCENT)
    parser.add_argument('--top_words', type=int, default=Config.MIN_TOP_WORDS)
    parser.add_argument('--data_path', type=str, default=Config.ORIGINAL_DATA_PATH)
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR)
    parser.add_argument('--oe_model_name', type=str, default=Config.MODEL_NAME)
    parser.add_argument('--oe_epochs', type=int, default=Config.NUM_TRAIN_EPOCHS)
    parser.add_argument('--osr_model_type', type=str, default=Config.OSR_MODEL_TYPE)
    parser.add_argument('--osr_epochs', type=int, default=Config.OSR_NUM_EPOCHS)
    parser.add_argument('--ood_data_path_osr', type=str, default=Config.OOD_SYSLOG_UNKNOWN_PATH_OSR)
    parser.add_argument('--osr_oe_sources_external', nargs='*', default=Config.OSR_OE_SOURCES_EXTERNAL, help="List of external OE sources, e.g., wikitext:wikitext-2-raw-v1 snli")
    
    parser.add_argument('--skip_base_training', action='store_true')
    parser.add_argument('--skip_attention_extraction', action='store_true')
    parser.add_argument('--skip_oe_extraction', action='store_true')
    parser.add_argument('--skip_oe_visualization', action='store_true')
    parser.add_argument('--skip_osr_experiments', action='store_true')
    parser.add_argument('--osr_eval_only', action='store_true')
    parser.add_argument('--osr_skip_standard_model', action='store_true', default=Config.OSR_SKIP_STANDARD_MODEL)
    parser.add_argument('--osr_skip_attention_oe_models', action='store_true', default=Config.OSR_SKIP_ATTENTION_OE_MODELS)
    parser.add_argument('--osr_skip_external_oe_models', action='store_true', default=Config.OSR_SKIP_EXTERNAL_OE_MODELS)


    args = parser.parse_args()
    
    Config.ATTENTION_TOP_PERCENT = args.attention_percent
    Config.MIN_TOP_WORDS = args.top_words
    Config.ORIGINAL_DATA_PATH = args.data_path
    Config.OUTPUT_DIR = args.output_dir
    Config.MODEL_NAME = args.oe_model_name
    Config.NUM_TRAIN_EPOCHS = args.oe_epochs
    Config.OSR_MODEL_TYPE = args.osr_model_type
    Config.OSR_NUM_EPOCHS = args.osr_epochs
    Config.OOD_SYSLOG_UNKNOWN_PATH_OSR = args.ood_data_path_osr
    Config.OSR_OE_SOURCES_EXTERNAL = args.osr_oe_sources_external
    Config.OSR_EVAL_ONLY = args.osr_eval_only
    Config.OSR_SKIP_STANDARD_MODEL = args.osr_skip_standard_model
    Config.OSR_SKIP_ATTENTION_OE_MODELS = args.osr_skip_attention_oe_models
    Config.OSR_SKIP_EXTERNAL_OE_MODELS = args.osr_skip_external_oe_models
    
    Config.STAGE_MODEL_TRAINING = not args.skip_base_training
    Config.STAGE_ATTENTION_EXTRACTION = not args.skip_attention_extraction
    Config.STAGE_OE_EXTRACTION = not args.skip_oe_extraction
    Config.STAGE_VISUALIZATION = not args.skip_oe_visualization
    Config.STAGE_OSR_EXPERIMENTS = not args.skip_osr_experiments
    
    # Update derived paths
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

    print(f"--- Unified OE/OSR Pipeline --- Output Dir: {Config.OUTPUT_DIR}")
    pipeline = UnifiedOEExtractor(Config); pipeline.run_full_pipeline()
    
if __name__ == '__main__':
    main()
