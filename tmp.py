# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import math
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from scipy.stats import weibull_min, norm # Added norm for DOC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # 한글 글꼴 설정용 임포트
import seaborn as sns
import shutil # Added for cleaning up trial checkpoints
import copy # --- 수정: args 복사를 위해 추가 ---

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
# --- 수정: Callback Base 임포트 ---
from pytorch_lightning.callbacks import Callback
# --- 수정: sklearn, pathlib 임포트 ---
from sklearn.metrics import roc_curve, auc # roc_curve 임포트 확인
from sklearn.manifold import TSNE # t-SNE 임포트
import pathlib
# ---

from transformers import (
    RobertaModel, RobertaTokenizer, RobertaConfig,
    get_linear_schedule_with_warmup,
    logging as hf_logging
)

# Custom Modules
from hyperparameter_tuning import (
    OptunaHyperparameterTuner, load_best_params, get_default_best_params
)
# Import the specific prepare functions needed
from dataset_utils import (
    prepare_newsgroup20_dataset, prepare_bbc_news_dataset, prepare_trec_dataset,
    prepare_reuters8_dataset, prepare_acm_dataset, prepare_chemprot_dataset,
    prepare_banking77_dataset, prepare_oos_dataset, prepare_stackoverflow_dataset,
    prepare_atis_dataset, prepare_snips_dataset, prepare_financial_phrasebank_dataset,
    prepare_arxiv10_dataset, prepare_custom_syslog_dataset # Added custom_syslog
)

# --- Basic Setup ---
hf_logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*DataLoader running processes.*")
warnings.filterwarnings("ignore", ".*Checkpoint directory.*exists but is not empty.*")

# --- Matplotlib Korean Font Setup ---
def setup_korean_font():
    font_name = None
    try:
        possible_fonts = ['Malgun Gothic', 'NanumGothic', 'Apple SD Gothic Neo', 'Noto Sans KR']
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        for font in possible_fonts:
            if font in available_fonts:
                font_name = font
                break
        if font_name:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
        else:
            print("Warning: No common Korean font found. Install 'Malgun Gothic' or 'NanumGothic'. Plots might not display Korean characters correctly.")
            plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"Error setting up Korean font: {e}")
        plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()

# =============================================================================
# Data Processing (TextDataset, DataModule)
# =============================================================================
class TextDataset(Dataset):
    """PyTorch Dataset for text classification."""
    def __init__(self, texts, labels, tokenizer, max_length=384):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]) if self.texts[idx] is not None else ""
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_length,
            padding="max_length", truncation=True, return_attention_mask=True,
            return_token_type_ids=False, return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
        return item

class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for handling datasets."""
    def __init__(
        self, dataset_name, tokenizer, batch_size=64, seen_class_ratio=0.5,
        random_seed=42, max_length=384, train_ratio=0.7, val_ratio=0.15,
        test_ratio=0.15, data_dir="data"
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seen_class_ratio = seen_class_ratio
        self.random_seed = random_seed
        self.max_length = max_length
        self.data_dir = data_dir
        self.num_classes = None
        self.num_seen_classes = None
        self.seen_classes = None
        self.unseen_classes = None
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            print(f"Warning: Data split ratios sum to {total}. Normalizing...")
            self.train_ratio /= total; self.val_ratio /= total; self.test_ratio /= total
            print(f"Normalized ratios: train={self.train_ratio:.3f}, val={self.val_ratio:.3f}, test={self.test_ratio:.3f}")

        self.train_dataset = None; self.val_dataset = None; self.test_dataset = None
        self.class_names = None
        self.original_seen_indices = None; self.original_unseen_indices = None
        self.seen_class_mapping = None

        self.prepare_func_map = {
            "newsgroup20": prepare_newsgroup20_dataset, "bbc_news": prepare_bbc_news_dataset,
            "trec": prepare_trec_dataset, "reuters8": prepare_reuters8_dataset,
            "acm": prepare_acm_dataset, "chemprot": prepare_chemprot_dataset,
            "banking77": prepare_banking77_dataset, "oos": prepare_oos_dataset,
            "stackoverflow": prepare_stackoverflow_dataset, "atis": prepare_atis_dataset,
            "snips": prepare_snips_dataset, "financial_phrasebank": prepare_financial_phrasebank_dataset,
            "arxiv10": prepare_arxiv10_dataset, "custom_syslog": prepare_custom_syslog_dataset,
        }

    def prepare_data(self):
        print(f"Preparing data for dataset: {self.dataset_name}...")
        if self.dataset_name in self.prepare_func_map:
            try:
                _ = self.prepare_func_map[self.dataset_name](data_dir=self.data_dir)
                print(f"{self.dataset_name} data preparation check complete.")
            except FileNotFoundError as e:
                 if self.dataset_name == 'custom_syslog':
                      print(f"\n{'='*20} ACTION REQUIRED {'='*20}\n{e}\n{'='*58}\n")
                 else: print(f"Error during prepare_data for {self.dataset_name}: {e}")
                 sys.exit(1)
            except Exception as e:
                print(f"Error during prepare_data for {self.dataset_name}: {e}"); import traceback; traceback.print_exc(); raise
        else: print(f"Warning: No specific prepare_data action defined for dataset '{self.dataset_name}'.")

    def setup(self, stage=None):
        if self.train_dataset is not None and stage == 'fit': return
        if self.test_dataset is not None and stage == 'test': return

        pl.seed_everything(self.random_seed) # Use PL seed_everything

        print(f"\n--- Setting up DataModule for dataset: {self.dataset_name} (Seen Ratio: {self.seen_class_ratio}) ---")

        if self.dataset_name in self.prepare_func_map:
            print(f"Loading data using prepare_{self.dataset_name}_dataset...")
            try: texts, labels, self.class_names = self.prepare_func_map[self.dataset_name](data_dir=self.data_dir)
            except Exception as e: raise ValueError(f"Data loading failed for {self.dataset_name}") from e
        else: raise ValueError(f"Unknown or unprepared dataset: {self.dataset_name}")

        if not texts: raise ValueError(f"Failed to load any text data for dataset '{self.dataset_name}'.")

        labels = np.array(labels, dtype=int)
        if labels.ndim == 0 or len(labels) == 0: raise ValueError(f"Loaded labels for {self.dataset_name} are invalid or empty.")
        if len(texts) != len(labels): raise ValueError(f"Mismatch between texts ({len(texts)}) and labels ({len(labels)}) for {self.dataset_name}.")

        print("Splitting data into train/validation/test sets...")
        min_samples_per_class = 2
        unique_labels_split, counts = np.unique(labels, return_counts=True)
        stratify_param = labels if not np.any(counts < min_samples_per_class) else None
        if stratify_param is None: print(f"Warning: Stratification disabled due to classes with < {min_samples_per_class} samples.")

        try:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                texts, labels, test_size=self.test_ratio, random_state=self.random_seed, stratify=stratify_param
            )
            if self.train_ratio + self.val_ratio <= 1e-6: val_size_relative = 0; X_train, X_val, y_train, y_val = X_train_val, [], y_train_val, []
            else:
                 val_size_relative = self.val_ratio / (self.train_ratio + self.val_ratio)
                 unique_tv, counts_tv = np.unique(y_train_val, return_counts=True)
                 stratify_tv = y_train_val if not np.any(counts_tv < min_samples_per_class) else None
                 if stratify_tv is None: print("Warning: Stratification for train/val split disabled.")
                 X_train, X_val, y_train, y_val = train_test_split(
                     X_train_val, y_train_val, test_size=val_size_relative, random_state=self.random_seed, stratify=stratify_tv
                 )
        except ValueError as e:
             print(f"Error during stratified split: {e}. Retrying without stratification...")
             X_train_val, X_test, y_train_val, y_test = train_test_split(texts, labels, test_size=self.test_ratio, random_state=self.random_seed)
             if self.train_ratio + self.val_ratio <= 1e-6: val_size_relative = 0
             else: val_size_relative = self.val_ratio / (self.train_ratio + self.val_ratio)
             X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_relative, random_state=self.random_seed)

        all_original_indices = np.unique(labels)
        self.num_classes = len(all_original_indices)
        print(f"Total original classes found: {self.num_classes} -> {all_original_indices}")
        if self.class_names is None: self.class_names = [str(i) for i in all_original_indices]; print(f"Warning: class_names not set. Using: {self.class_names}")
        elif len(self.class_names) != self.num_classes:
             print(f"Warning: Mismatch class names ({len(self.class_names)}) vs unique labels ({self.num_classes}). Adjusting."); self.class_names = [self.class_names[i] if i < len(self.class_names) else f"Class_{i}" for i in all_original_indices]

        if self.seen_class_ratio < 1.0:
            print(f"Splitting classes: {self.seen_class_ratio*100:.1f}% Seen / {(1-self.seen_class_ratio)*100:.1f}% Unseen")
            num_seen = max(1, int(np.round(self.num_classes * self.seen_class_ratio)))
            if num_seen >= self.num_classes: print("Warning: num_seen >= total classes. Setting ratio to 1.0."); self.seen_class_ratio = 1.0

        if self.seen_class_ratio < 1.0:
            np.random.seed(self.random_seed); all_classes_shuffled = np.random.permutation(all_original_indices)
            self.original_seen_indices = np.sort(all_classes_shuffled[:num_seen])
            self.original_unseen_indices = np.sort(all_classes_shuffled[num_seen:])
            print(f"  Original Seen Indices: {self.original_seen_indices}"); print(f"  Original Unseen Indices: {self.original_unseen_indices}")
            self.seen_class_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(self.original_seen_indices)}
            self.num_seen_classes = len(self.original_seen_indices); self.seen_classes = np.arange(self.num_seen_classes)

            train_seen_mask = np.isin(y_train, self.original_seen_indices); X_train = [X_train[i] for i, keep in enumerate(train_seen_mask) if keep]; y_train_original_kept = y_train[train_seen_mask]; y_train_mapped = np.array([self.seen_class_mapping[lbl] for lbl in y_train_original_kept])
            val_seen_mask = np.isin(y_val, self.original_seen_indices); X_val = [X_val[i] for i, keep in enumerate(val_seen_mask) if keep]; y_val_original_kept = y_val[val_seen_mask]; y_val_mapped = np.array([self.seen_class_mapping[lbl] for lbl in y_val_original_kept])
            y_test_final = y_test.copy().astype(int); unseen_test_mask = np.isin(y_test, self.original_unseen_indices); y_test_final[unseen_test_mask] = -1
            y_train_final = y_train_mapped; y_val_final = y_val_mapped
        else:
            print("All classes are Known (seen_class_ratio = 1.0)")
            self.original_seen_indices = all_original_indices.copy(); self.original_unseen_indices = np.array([])
            self.num_seen_classes = self.num_classes; self.seen_classes = all_original_indices.copy()
            self.seen_class_mapping = {orig_idx: orig_idx for orig_idx in all_original_indices}
            y_train_final = y_train; y_val_final = y_val; y_test_final = y_test

        self.train_dataset = TextDataset(list(X_train), y_train_final, self.tokenizer, self.max_length)
        self.val_dataset = TextDataset(list(X_val), y_val_final, self.tokenizer, self.max_length)
        self.test_dataset = TextDataset(list(X_test), y_test_final, self.tokenizer, self.max_length)

        print(f"\nDataset sizes: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")
        print(f"Known classes for training: {self.num_seen_classes}")
        if self.seen_class_ratio < 1.0: print(f"Unknown samples in test set: {np.sum(y_test_final == -1)}")
        if self.class_names and self.original_seen_indices is not None:
             try: seen_names_list = [self.class_names[i] for i in self.original_seen_indices]; print(f"Known class names: {seen_names_list}")
             except IndexError: print(f"Warning: Could not map seen_indices to class_names.")
        print("--- Finished DataModule setup ---")

    def train_dataloader(self):
        if self.train_dataset is None: raise ValueError("Train dataset not initialized.")
        num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 0)
        persistent = num_workers > 0
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, persistent_workers=persistent, pin_memory=True)

    def val_dataloader(self):
        if self.val_dataset is None: raise ValueError("Validation dataset not initialized.")
        num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 0)
        persistent = num_workers > 0
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent, pin_memory=True)

    def test_dataloader(self):
        if self.test_dataset is None: raise ValueError("Test dataset not initialized.")
        num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 0)
        persistent = num_workers > 0
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent, pin_memory=True)

    def _determine_unknown_labels(self, labels_np):
        return (np.array(labels_np) == -1)

# =============================================================================
# Model Definitions
# =============================================================================
# ... (RobertaClassifier, RobertaAutoencoder, DOCRobertaClassifier, RobertaADB 클래스 기존 코드 유지) ...
class RobertaClassifier(pl.LightningModule):
    """Standard RoBERTa model with a classification head."""
    def __init__(
        self,
        model_name="roberta-base",
        num_classes=20, # Number of *known* classes for the classifier
        learning_rate=2e-5, # Renamed from lr for consistency
        weight_decay=0.01,
        warmup_steps=0,
        total_steps=0
    ):
        super().__init__()
        # Use save_hyperparameters() correctly
        self.save_hyperparameters(ignore=['model_name']) # Saves num_classes, learning_rate, etc.
        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        self.classifier = nn.Linear(self.config.hidden_size, self.hparams.num_classes) # Use hparams
        # Store scheduler params if needed later, though configure_optimizers uses hparams
        # self.learning_rate = learning_rate # Already saved in hparams
        # self.weight_decay = weight_decay # Already saved in hparams
        self.warmup_steps = warmup_steps # Store for configure_optimizers
        self.total_steps = total_steps   # Store for configure_optimizers
        self.num_classes = self.hparams.num_classes # Explicitly set for OSRAlgorithm use

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Forward pass returning logits and CLS token embedding."""
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids # Will be None if tokenizer doesn't provide it
        )
        cls_output = outputs.last_hidden_state[:, 0, :] # [CLS] embedding
        logits = self.classifier(cls_output)
        return logits, cls_output

    def training_step(self, batch, batch_idx):
        """Single training step."""
        # Assumes labels are 0..N-1 mapped known classes
        logits, _ = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        # Ensure labels are not -1 during training
        valid_labels = batch['label'][batch['label'] >= 0]
        valid_logits = logits[batch['label'] >= 0]
        if valid_labels.numel() == 0: return None # Skip batch if no valid labels

        loss = F.cross_entropy(valid_logits, valid_labels)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """Single validation step."""
        logits, _ = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        # Validation set should also only contain known classes (0..N-1)
        valid_labels = batch['label'][batch['label'] >= 0]
        valid_logits = logits[batch['label'] >= 0]
        if valid_labels.numel() == 0: return {'val_loss': torch.tensor(0.0), 'val_acc': torch.tensor(0.0)}

        loss = F.cross_entropy(valid_logits, valid_labels)
        preds = torch.argmax(valid_logits, dim=1)
        acc = accuracy_score(valid_labels.cpu(), preds.cpu())
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        """Single test step."""
        logits, embeddings = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        labels_original = batch['label'] # Original labels (can include -1 and original seen indices 0..Total-1)
        preds_mapped = torch.argmax(logits, dim=1) # Predictions indices (0..num_seen-1)

        # Calculate loss & acc only on known samples (where label >= 0)
        known_mask = labels_original >= 0
        loss = torch.tensor(0.0, device=self.device)
        acc = torch.tensor(0.0, device=self.device)
        if known_mask.any():
             # Map true known labels (original indices 0..Total-1) to the model's output space (0..num_seen-1) for loss/acc
             original_indices_known = labels_original[known_mask].cpu().numpy()
             try:
                 # Use the mapping stored in the datamodule
                 mapped_true_labels_known = torch.tensor([self.trainer.datamodule.seen_class_mapping[idx] for idx in original_indices_known], device=self.device)
                 loss = F.cross_entropy(logits[known_mask], mapped_true_labels_known)
                 acc = accuracy_score(mapped_true_labels_known.cpu(), preds_mapped[known_mask].cpu())
                 self.log('test_loss', loss, prog_bar=False, logger=True, batch_size=self.trainer.datamodule.batch_size)
                 self.log('test_acc_known', acc, prog_bar=False, logger=True, batch_size=self.trainer.datamodule.batch_size) # Log accuracy on knowns
             except KeyError as e:
                 print(f"Error: Test label {e} not found in seen_class_mapping during test_step.")
                 # Handle error, maybe log or skip accuracy calculation for this batch


        return {
            'preds_mapped': preds_mapped, # Predictions indices (0..num_seen-1) for all samples
            'labels_original': labels_original,    # Original labels (can include -1 and original seen indices 0..Total-1)
            'logits': logits,             # Raw logits (size num_seen_classes)
            'embeddings': embeddings      # CLS embeddings
        }

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        # Ensure total_steps is calculated correctly before scheduler creation
        if not hasattr(self, 'total_steps') or self.total_steps <= 0:
             print("Warning: total_steps not set for scheduler. Estimating from trainer.")
             try:
                 self.total_steps = self.trainer.estimated_stepping_batches
                 if self.total_steps is None or self.total_steps <= 0: # Check if estimation failed
                      print("Warning: Failed to estimate total_steps from trainer. Using fallback.")
                      self.total_steps = 10000 # Fallback
             except Exception:
                 print("Warning: Exception during total_steps estimation. Using fallback.")
                 self.total_steps = 10000 # Fallback if trainer not available yet
             print(f"Using estimated/fallback total_steps: {self.total_steps}")

        if not hasattr(self, 'warmup_steps'):
             print("Warning: warmup_steps not set explicitly. Using default.")
             self.warmup_steps = 0

        # Ensure warmup_steps doesn't exceed total_steps
        actual_warmup_steps = min(self.warmup_steps, self.total_steps) if self.total_steps > 0 else 0
        if actual_warmup_steps != self.warmup_steps:
             print(f"Warning: Adjusted warmup_steps from {self.warmup_steps} to {actual_warmup_steps}")

        if self.total_steps > 0:
             scheduler = get_linear_schedule_with_warmup(
                 optimizer, num_warmup_steps=actual_warmup_steps, num_training_steps=self.total_steps
             )
             return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        else:
             print("Warning: total_steps is zero or negative. No LR scheduler will be used.")
             return optimizer

class RobertaAutoencoder(pl.LightningModule):
    """RoBERTa-based Autoencoder for CROSR."""
    def __init__( self, model_name="roberta-base", num_classes=20, learning_rate=2e-5, weight_decay=0.01,
                  warmup_steps=0, total_steps=0, latent_dim=256, reconstruction_weight=0.5 ):
        super().__init__()
        self.save_hyperparameters(ignore=['model_name'])
        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        self.classifier = nn.Linear(self.config.hidden_size, self.hparams.num_classes)
        # Encoder/Decoder Layers
        self.encoder = nn.Sequential( nn.Linear(self.config.hidden_size, self.hparams.latent_dim), nn.ReLU() )
        self.decoder = nn.Sequential( nn.Linear(self.hparams.latent_dim, self.config.hidden_size) )
        # Store scheduler params
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.num_classes = self.hparams.num_classes # For OSRAlgorithm use

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        encoded = self.encoder(cls_output)
        reconstructed = self.decoder(encoded)
        return logits, cls_output, encoded, reconstructed

    def training_step(self, batch, batch_idx):
        logits, cls_output, _, reconstructed = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        # Ensure labels are valid (0..N-1)
        valid_labels = batch['label'][batch['label'] >= 0]
        valid_logits = logits[batch['label'] >= 0]
        valid_cls_output = cls_output[batch['label'] >= 0]
        valid_reconstructed = reconstructed[batch['label'] >= 0]
        if valid_labels.numel() == 0: return None

        class_loss = F.cross_entropy(valid_logits, valid_labels)
        recon_loss = F.mse_loss(valid_reconstructed, valid_cls_output)
        loss = class_loss + self.hparams.reconstruction_weight * recon_loss
        self.log_dict({'train_loss': loss, 'train_class_loss': class_loss, 'train_recon_loss': recon_loss},
                      prog_bar=False, on_step=True, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, cls_output, _, reconstructed = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        # Ensure labels are valid (0..N-1)
        valid_labels = batch['label'][batch['label'] >= 0]
        valid_logits = logits[batch['label'] >= 0]
        valid_cls_output = cls_output[batch['label'] >= 0]
        valid_reconstructed = reconstructed[batch['label'] >= 0]
        if valid_labels.numel() == 0: return {'val_loss': torch.tensor(0.0), 'val_acc': torch.tensor(0.0)}

        class_loss = F.cross_entropy(valid_logits, valid_labels)
        recon_loss = F.mse_loss(valid_reconstructed, valid_cls_output)
        loss = class_loss + self.hparams.reconstruction_weight * recon_loss
        preds = torch.argmax(valid_logits, dim=1)
        acc = accuracy_score(valid_labels.cpu(), preds.cpu())
        self.log_dict({'val_loss': loss, 'val_class_loss': class_loss, 'val_recon_loss': recon_loss, 'val_acc': acc},
                       prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        logits, cls_output, encoded, reconstructed = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        labels_original = batch['label'] # Original labels (-1 possible, original seen indices 0..Total-1)
        preds_mapped = torch.argmax(logits, dim=1) # Predictions (0..num_seen-1)
        recon_errors = torch.norm(reconstructed - cls_output, p=2, dim=1)

        # Calculate loss/acc on knowns only
        known_mask = labels_original >= 0
        loss = torch.tensor(0.0, device=self.device)
        acc = torch.tensor(0.0, device=self.device)
        if known_mask.any():
             original_indices_known = labels_original[known_mask].cpu().numpy()
             try:
                 mapped_true_labels_known = torch.tensor([self.trainer.datamodule.seen_class_mapping[idx] for idx in original_indices_known], device=self.device)
                 class_loss = F.cross_entropy(logits[known_mask], mapped_true_labels_known)
                 recon_loss = F.mse_loss(reconstructed[known_mask], cls_output[known_mask])
                 loss = class_loss + self.hparams.reconstruction_weight * recon_loss
                 acc = accuracy_score(mapped_true_labels_known.cpu(), preds_mapped[known_mask].cpu())
                 self.log('test_loss', loss, prog_bar=False, logger=True, batch_size=self.trainer.datamodule.batch_size)
                 self.log('test_acc_known', acc, prog_bar=False, logger=True, batch_size=self.trainer.datamodule.batch_size)
             except KeyError as e:
                 print(f"Error: Test label {e} not found in seen_class_mapping during test_step.")

        return {
            'preds_mapped': preds_mapped, 'labels_original': labels_original, 'logits': logits,
            'embeddings': cls_output, 'encoded': encoded, 'reconstructed': reconstructed,
            'recon_errors': recon_errors # Crucial for CROSR evaluation
        }

    def configure_optimizers(self):
        # --- 수정: learning_rate 사용 ---
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        if not hasattr(self, 'total_steps') or self.total_steps <= 0:
             print("Warning: total_steps not set for scheduler. Estimating from trainer.")
             try:
                 self.total_steps = self.trainer.estimated_stepping_batches or 10000
             except Exception: self.total_steps = 10000
             print(f"Using estimated/fallback total_steps: {self.total_steps}")
        if not hasattr(self, 'warmup_steps'): self.warmup_steps = 0

        actual_warmup_steps = min(self.warmup_steps, self.total_steps) if self.total_steps > 0 else 0

        if self.total_steps > 0:
             scheduler = get_linear_schedule_with_warmup(
                 optimizer, num_warmup_steps=actual_warmup_steps, num_training_steps=self.total_steps
             )
             return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        else:
             return optimizer


class DOCRobertaClassifier(pl.LightningModule):
    """RoBERTa model adapted for DOC (one-vs-rest binary classifiers)."""
    def __init__( self, model_name="roberta-base", num_classes=20, learning_rate=2e-5, weight_decay=0.01,
                  warmup_steps=0, total_steps=0 ):
        super().__init__()
        self.save_hyperparameters(ignore=['model_name'])
        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        # One binary classifier head per known class
        self.classifiers = nn.ModuleList([nn.Linear(self.config.hidden_size, 1) for _ in range(self.hparams.num_classes)])
        # Store scheduler params
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.num_classes = self.hparams.num_classes # For OSRAlgorithm use

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Apply each binary classifier and concatenate logits
        logits = torch.cat([clf(cls_output) for clf in self.classifiers], dim=1) # Shape: (batch_size, num_classes)
        return logits, cls_output

    def training_step(self, batch, batch_idx):
        logits, _ = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        # Ensure labels are valid (0..N-1)
        valid_labels = batch['label'][batch['label'] >= 0]
        valid_logits = logits[batch['label'] >= 0]
        if valid_labels.numel() == 0: return None

        # Create multi-label binary targets (one-hot)
        one_hot_labels = F.one_hot(valid_labels, num_classes=self.hparams.num_classes).float()
        loss = F.binary_cross_entropy_with_logits(valid_logits, one_hot_labels)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, _ = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        # Ensure labels are valid (0..N-1)
        valid_labels = batch['label'][batch['label'] >= 0]
        valid_logits = logits[batch['label'] >= 0]
        if valid_labels.numel() == 0: return {'val_loss': torch.tensor(0.0), 'val_acc': torch.tensor(0.0)}

        one_hot_labels = F.one_hot(valid_labels, num_classes=self.hparams.num_classes).float()
        loss = F.binary_cross_entropy_with_logits(valid_logits, one_hot_labels)
        # Predict based on highest sigmoid score
        sigmoid_scores = torch.sigmoid(valid_logits)
        preds = torch.argmax(sigmoid_scores, dim=1)
        acc = accuracy_score(valid_labels.cpu(), preds.cpu())
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        logits, embeddings = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        labels_original = batch['label'] # Original labels (-1 possible, original seen indices 0..Total-1)
        sigmoid_scores = torch.sigmoid(logits)
        preds_mapped = torch.argmax(sigmoid_scores, dim=1) # Predicted class index (0..num_seen-1)

        # Calculate loss/acc on knowns only
        known_mask = labels_original >= 0
        loss = torch.tensor(0.0, device=self.device)
        acc = torch.tensor(0.0, device=self.device)
        if known_mask.any():
             original_indices_known = labels_original[known_mask].cpu().numpy()
             try:
                 mapped_true_labels_known = torch.tensor([self.trainer.datamodule.seen_class_mapping[idx] for idx in original_indices_known], device=self.device)
                 one_hot_labels = F.one_hot(mapped_true_labels_known, num_classes=self.hparams.num_classes).float()
                 loss = F.binary_cross_entropy_with_logits(logits[known_mask], one_hot_labels)
                 acc = accuracy_score(mapped_true_labels_known.cpu(), preds_mapped[known_mask].cpu())
                 self.log('test_loss', loss, prog_bar=False, logger=True, batch_size=self.trainer.datamodule.batch_size)
                 self.log('test_acc_known', acc, prog_bar=False, logger=True, batch_size=self.trainer.datamodule.batch_size)
             except KeyError as e:
                 print(f"Error: Test label {e} not found in seen_class_mapping during test_step.")

        return {
            'preds_mapped': preds_mapped,           # Predicted class index (0..num_seen-1)
            'labels_original': labels_original,     # Original labels
            'logits': logits,                       # Raw one-vs-rest logits
            'sigmoid_scores': sigmoid_scores,       # Sigmoid scores (crucial for DOC eval)
            'embeddings': embeddings                # CLS embeddings
        }

    def configure_optimizers(self):
        # --- 수정: learning_rate 사용 ---
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        if not hasattr(self, 'total_steps') or self.total_steps <= 0:
             print("Warning: total_steps not set for scheduler. Estimating from trainer.")
             try:
                 self.total_steps = self.trainer.estimated_stepping_batches or 10000
             except Exception: self.total_steps = 10000
             print(f"Using estimated/fallback total_steps: {self.total_steps}")
        if not hasattr(self, 'warmup_steps'): self.warmup_steps = 0

        actual_warmup_steps = min(self.warmup_steps, self.total_steps) if self.total_steps > 0 else 0

        if self.total_steps > 0:
             scheduler = get_linear_schedule_with_warmup(
                 optimizer, num_warmup_steps=actual_warmup_steps, num_training_steps=self.total_steps
             )
             return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        else:
             return optimizer


class RobertaADB(pl.LightningModule):
    """RoBERTa model with Adaptive Decision Boundary (ADB) components."""
    def __init__( self,
                  model_name: str = "roberta-base",
                  num_classes: int = 20,
                  lr: float = 2e-5, # Backbone fine-tuning LR (from args.lr)
                  lr_adb: float = 5e-4, # ADB parameter LR (from args.lr_adb)
                  weight_decay: float = 0.0,
                  warmup_steps: int = 0, # For backbone scheduler
                  total_steps: int = 0,  # For backbone scheduler
                  param_adb_delta: float = 0.1, # Renamed to match arg name
                  param_adb_alpha: float = 0.5, # Renamed to match arg name
                  adb_freeze_backbone: bool = True ): # Renamed to match arg name
        super().__init__()
        # Save hyperparameters passed during initialization
        self.save_hyperparameters(ignore=['model_name']) # Saves num_classes, lr, lr_adb, etc.
        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        self.feat_dim = self.config.hidden_size

        # Learnable centers (initialized using hparams)
        self.centers = nn.Parameter(torch.empty(self.hparams.num_classes, self.feat_dim), requires_grad=True)
        nn.init.normal_(self.centers, std=0.05)

        # Learnable Logits for Radii (Delta' in paper)
        initial_delta_prime = -1.5 # Initial value aiming for radius ~0.2
        self.delta_prime = nn.Parameter(torch.full((self.hparams.num_classes,), initial_delta_prime), requires_grad=True)

        # Store scheduler params (needed by configure_optimizers)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.num_classes = self.hparams.num_classes # For OSRAlgorithm use

        # Freeze backbone based on hparams
        if self.hparams.adb_freeze_backbone:
            print("[RobertaADB Init] Freezing RoBERTa backbone parameters.")
            for param in self.roberta.parameters(): param.requires_grad = False
        else: print("[RobertaADB Init] RoBERTa backbone parameters will be fine-tuned.")

    def get_radii(self):
        """Calculate actual radii using Softplus."""
        return F.softplus(self.delta_prime)

    @staticmethod
    def _cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates pairwise cosine distance (1 - similarity). Input x assumed normalized."""
        # Ensure y is normalized for cosine distance calculation
        y_norm = F.normalize(y, p=2, dim=-1)
        similarity = torch.matmul(x, y_norm.t())
        similarity = torch.clamp(similarity, -1.0 + 1e-7, 1.0 - 1e-7)
        distance = 1.0 - similarity
        return distance

    def adb_margin_loss(self, feat_norm: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculates ADB margin loss: max(0, d(feat, c_y) - r_y + delta)."""
        distances = self._cosine_distance(feat_norm, self.centers) # feat_norm is normalized
        # Get distances corresponding to the true labels
        d_y = distances.gather(1, labels.unsqueeze(1)).squeeze(1) # More robust way

        radii = self.get_radii() # Get positive radii via Softplus
        r_y = radii[labels] # Get radii for the true labels

        # Loss = max(0, distance_to_correct_center - radius_of_correct_center + margin)
        # Use hparams for delta
        loss_per_sample = torch.relu(d_y - r_y + self.hparams.param_adb_delta)
        loss = loss_per_sample.mean()
        return loss

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Forward pass: get CLS embedding, normalize it, calculate similarity logits."""
        # Control gradient flow based on freeze_backbone hparam
        with torch.set_grad_enabled(not self.hparams.adb_freeze_backbone):
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        feat = outputs.last_hidden_state[:, 0, :]
        feat_norm = F.normalize(feat, p=2, dim=-1)
        # Logits are similarity scores (1 - cosine distance)
        logits = 1.0 - self._cosine_distance(feat_norm, self.centers)
        return logits, feat_norm

    def training_step(self, batch, batch_idx):
        # Ensure labels are valid (0..N-1)
        valid_labels = batch['label'][batch['label'] >= 0]
        valid_input_ids = batch["input_ids"][batch['label'] >= 0]
        valid_attention_mask = batch["attention_mask"][batch['label'] >= 0]
        valid_token_type_ids = batch.get("token_type_ids")
        if valid_token_type_ids is not None:
            valid_token_type_ids = valid_token_type_ids[batch['label'] >= 0]
        if valid_labels.numel() == 0: return None

        logits, feat_norm = self(valid_input_ids, valid_attention_mask, valid_token_type_ids)
        ce_loss = F.cross_entropy(logits, valid_labels)
        adb_loss = self.adb_margin_loss(feat_norm, valid_labels)
        # Use hparams for alpha
        loss = ce_loss + self.hparams.param_adb_alpha * adb_loss
        self.log_dict({'train_loss': loss, 'train_ce_loss': ce_loss, 'train_adb_loss': adb_loss},
                      prog_bar=True, on_step=True, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        # Log actual average radius
        avg_radius = self.get_radii().mean().item()
        self.log("avg_radius", avg_radius, on_step=False, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        # Ensure labels are valid (0..N-1)
        valid_labels = batch['label'][batch['label'] >= 0]
        valid_input_ids = batch["input_ids"][batch['label'] >= 0]
        valid_attention_mask = batch["attention_mask"][batch['label'] >= 0]
        valid_token_type_ids = batch.get("token_type_ids")
        if valid_token_type_ids is not None:
            valid_token_type_ids = valid_token_type_ids[batch['label'] >= 0]
        if valid_labels.numel() == 0: return {"val_loss": torch.tensor(0.0), "val_acc": torch.tensor(0.0)}

        logits, feat_norm = self(valid_input_ids, valid_attention_mask, valid_token_type_ids)
        ce_loss = F.cross_entropy(logits, valid_labels)
        adb_loss = self.adb_margin_loss(feat_norm, valid_labels)
        loss = ce_loss + self.hparams.param_adb_alpha * adb_loss
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(valid_labels.cpu(), preds.cpu())
        self.log_dict({'val_loss': loss, 'val_ce_loss': ce_loss, 'val_adb_loss': adb_loss, 'val_acc': acc},
                      prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        labels_original = batch["label"] # Original labels (-1 possible, original seen indices 0..Total-1)
        logits, feat_norm = self(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"))
        preds_mapped = torch.argmax(logits, dim=1) # Predictions based on highest similarity (0..num_seen-1)

        # Calculate loss/acc on knowns only
        known_mask = labels_original >= 0
        loss = torch.tensor(0.0, device=self.device)
        acc = torch.tensor(0.0, device=self.device)
        if known_mask.any():
             original_indices_known = labels_original[known_mask].cpu().numpy()
             try:
                 mapped_true_labels_known = torch.tensor([self.trainer.datamodule.seen_class_mapping[idx] for idx in original_indices_known], device=self.device)
                 ce_loss = F.cross_entropy(logits[known_mask], mapped_true_labels_known)
                 adb_loss = self.adb_margin_loss(feat_norm[known_mask], mapped_true_labels_known)
                 loss = ce_loss + self.hparams.param_adb_alpha * adb_loss
                 acc = accuracy_score(mapped_true_labels_known.cpu(), preds_mapped[known_mask].cpu())
                 self.log('test_loss', loss, prog_bar=False, logger=True, batch_size=self.trainer.datamodule.batch_size)
                 self.log('test_acc_known', acc, prog_bar=False, logger=True, batch_size=self.trainer.datamodule.batch_size)
             except KeyError as e:
                 print(f"Error: Test label {e} not found in seen_class_mapping during test_step.")

        return {
            'preds_mapped': preds_mapped,        # Predicted class index (0..num_seen-1) based on similarity
            'labels_original': labels_original,  # Original labels
            'logits': logits,                    # Similarity-based logits
            'features': feat_norm                # Normalized features (crucial for ADB eval)
        }

    def configure_optimizers(self):
        """Configure optimizer(s) and scheduler(s)."""
        params_to_optimize = []
        if not self.hparams.adb_freeze_backbone:
             # Use the general 'lr' hyperparameter for the backbone
             backbone_lr = self.hparams.lr
             params_to_optimize.append({'params': self.roberta.parameters(), 'lr': backbone_lr})
             print(f"[ADB Optim] Fine-tuning RoBERTa with LR: {backbone_lr}")

        # Optimize centers and delta_prime with the specific ADB LR ('lr_adb' hyperparameter)
        adb_param_lr = self.hparams.lr_adb
        params_to_optimize.append({'params': self.centers, 'lr': adb_param_lr})
        params_to_optimize.append({'params': self.delta_prime, 'lr': adb_param_lr})
        print(f"[ADB Optim] Optimizing centers/delta_prime with LR: {adb_param_lr}")

        optimizer = AdamW(params_to_optimize, weight_decay=self.hparams.weight_decay)

        # Add scheduler only if backbone is being trained and total_steps is valid
        if not self.hparams.adb_freeze_backbone:
             if not hasattr(self, 'total_steps') or self.total_steps <= 0:
                 print("Warning: total_steps not set for scheduler. Estimating from trainer.")
                 try:
                     self.total_steps = self.trainer.estimated_stepping_batches or 10000
                 except Exception: self.total_steps = 10000
                 print(f"Using estimated/fallback total_steps: {self.total_steps}")
             if not hasattr(self, 'warmup_steps'): self.warmup_steps = 0

             actual_warmup_steps = min(self.warmup_steps, self.total_steps) if self.total_steps > 0 else 0

             if self.total_steps > 0:
                 scheduler = get_linear_schedule_with_warmup(
                     optimizer, num_warmup_steps=actual_warmup_steps, num_training_steps=self.total_steps
                 )
                 print("[ADB Optim] Using learning rate scheduler for backbone.")
                 return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
             else:
                 print("[ADB Optim] No learning rate scheduler used (total_steps=0).")
                 return optimizer
        else:
             print("[ADB Optim] No learning rate scheduler used (backbone frozen).")
             return optimizer

# =============================================================================
# OSR Algorithms
# =============================================================================
class OSRAlgorithm:
    """Base class for Open Set Recognition algorithms."""
    def __init__(self, model, datamodule, args):
        self.model = model
        self.datamodule = datamodule
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu')
        self.num_known_classes = model.num_classes if hasattr(model, 'num_classes') else datamodule.num_seen_classes
        if self.num_known_classes is None: raise ValueError("Could not determine number of known classes.")
        print(f"[{self.__class__.__name__}] Initialized for {self.num_known_classes} known classes.")

    def predict(self, dataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts labels, including potential 'unknown' (-1).
        Must be implemented by subclass.

        Returns:
            tuple: (all_scores, all_preds_final, all_labels_original, scores_for_ranking, all_embeddings)
                - all_scores: Method-specific scores (e.g., probs, recon errors). Shape: (N, num_classes or 1)
                - all_preds_final: Final predicted labels (-1 for unknown, original index for known). Shape: (N,)
                - all_labels_original: Ground truth labels (-1 or original index). Shape: (N,)
                - scores_for_ranking: Scores where higher means more likely unknown (for AUROC/FPR). Shape: (N,)
                - all_embeddings: Feature embeddings for t-SNE. Shape: (N, embedding_dim)
        """
        raise NotImplementedError("Predict method must be implemented by subclass.")

    def evaluate(self, dataloader):
        """Evaluates OSR performance on the dataloader."""
        all_scores, all_preds, all_labels, scores_for_ranking, all_embeddings = self.predict(dataloader)

        if len(all_labels) == 0:
            print(f"Warning: No data to evaluate for {self.__class__.__name__}.")
            return { 'accuracy': 0, 'auroc': float('nan'), 'f1_score': 0, 'unknown_detection_rate': 0,
                     'fpr_at_tpr90': float('nan'), 'confusion_matrix': None, 'confusion_matrix_labels': [],
                     'confusion_matrix_names': [], 'predictions': [], 'labels': [],
                     'scores_for_ranking': [], 'embeddings': [] }

        unknown_labels_mask = self.datamodule._determine_unknown_labels(all_labels)
        unknown_preds_mask = (all_preds == -1)
        known_mask = ~unknown_labels_mask

        accuracy = accuracy_score(all_labels[known_mask], all_preds[known_mask]) if known_mask.any() else 0.0
        unknown_correct = np.sum(unknown_preds_mask & unknown_labels_mask)
        unknown_total = np.sum(unknown_labels_mask)
        unknown_detection_rate = unknown_correct / unknown_total if unknown_total > 0 else 0.0

        auroc = float('nan'); fpr_at_tpr90 = float('nan')
        if len(np.unique(unknown_labels_mask)) > 1 and len(scores_for_ranking) == len(unknown_labels_mask):
            try: auroc = roc_auc_score(unknown_labels_mask, scores_for_ranking)
            except ValueError as e: print(f"AUROC calculation failed: {e}") # Handle cases with only one class in scores

            try:
                fpr, tpr, thresholds = roc_curve(unknown_labels_mask, scores_for_ranking)
                if np.any(tpr >= 0.90):
                    idx = np.where(tpr >= 0.90)[0][0]
                    fpr_at_tpr90 = fpr[idx]
                else:
                    print("Warning: TPR did not reach 0.90. Setting FPR@TPR90 to 1.0."); fpr_at_tpr90 = 1.0
            except Exception as e: print(f"Error calculating FPR@TPR90: {e}")
        else: print("Skipping AUROC and FPR@TPR90 calculation (only one class or data mismatch).")

        labels_mapped_for_cm = all_labels.copy()
        cm_axis_labels_int, cm_axis_labels_names = self._get_cm_labels()
        valid_cm_labels = set(cm_axis_labels_int)
        filtered_labels_true = [l if l in valid_cm_labels else -1 for l in labels_mapped_for_cm]
        filtered_labels_pred = [p if p in valid_cm_labels else -1 for p in all_preds]

        conf_matrix = confusion_matrix(filtered_labels_true, filtered_labels_pred, labels=cm_axis_labels_int)
        precision, recall, f1_by_class, _ = precision_recall_fscore_support(
            filtered_labels_true, filtered_labels_pred, labels=cm_axis_labels_int, average=None, zero_division=0
        )
        f1_known_classes = [f1 for i, f1 in enumerate(f1_by_class) if cm_axis_labels_int[i] != -1]
        macro_f1 = np.mean(f1_known_classes) if len(f1_known_classes) > 0 else 0.0

        print(f"\n{self.__class__.__name__} Evaluation Summary:")
        print(f"  Accuracy (Known): {accuracy:.4f}")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  FPR@TPR90: {fpr_at_tpr90:.4f}")
        print(f"  Unknown Detection Rate: {unknown_detection_rate:.4f}")
        print(f"  Macro F1 Score (Known Only): {macro_f1:.4f}")

        results = {
            'accuracy': accuracy, 'auroc': auroc, 'f1_score': macro_f1,
            'unknown_detection_rate': unknown_detection_rate, 'fpr_at_tpr90': fpr_at_tpr90,
            'confusion_matrix': conf_matrix, 'confusion_matrix_labels': cm_axis_labels_int,
            'confusion_matrix_names': cm_axis_labels_names, 'predictions': all_preds, 'labels': all_labels,
            'scores_for_ranking': scores_for_ranking, 'embeddings': all_embeddings
        }
        return results

    def _visualize_tsne(self, results, base_filename):
        if 'embeddings' not in results or results['embeddings'] is None or len(results['embeddings']) < 100:
            print("  Skipping t-SNE plot (embeddings not available or insufficient samples)."); return
        if results['embeddings'].shape[1] <= 2:
             print("  Skipping t-SNE plot (embedding dimension <= 2)."); return

        print("  Generating t-SNE plot (this may take a while)...")
        features = results['embeddings']; labels_np = results['labels']
        unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)
        n_samples = features.shape[0]; max_tsne = 5000
        indices = np.random.choice(n_samples, min(n_samples, max_tsne), replace=False)
        features_sub = features[indices]; unknown_sub = unknown_labels_mask[indices]

        try:
            sample_norm = np.linalg.norm(features_sub[0])
            tsne_metric = 'cosine' if np.isclose(sample_norm, 1.0) else 'euclidean'
            print(f"    Using t-SNE metric: {tsne_metric} (based on sample norm: {sample_norm:.2f})")

            tsne = TSNE(n_components=2, random_state=self.args.random_seed, perplexity=min(30, features_sub.shape[0]-1),
                        n_iter=300, init='pca', learning_rate='auto', metric=tsne_metric)
            reduced_feats = tsne.fit_transform(features_sub)

            plt.figure(figsize=(10, 8))
            known_label_display = 'Known' if np.any(~unknown_sub) else None
            unknown_label_display = 'Unknown' if np.any(unknown_sub) else None
            center_label_display = None # Default

            if known_label_display: plt.scatter(reduced_feats[~unknown_sub, 0], reduced_feats[~unknown_sub, 1], c='blue', alpha=0.4, s=8, label=known_label_display)
            if unknown_label_display: plt.scatter(reduced_feats[unknown_sub, 0], reduced_feats[unknown_sub, 1], c='red', alpha=0.4, s=8, label=unknown_label_display)

            # Add centers for ADB
            if isinstance(self, ADBOSR) and hasattr(self.model, 'centers'):
                 try:
                      print("    Projecting centers with t-SNE...")
                      centers = self.model.centers.detach().cpu().numpy()
                      centers_norm = F.normalize(torch.from_numpy(centers), p=2, dim=-1).numpy()
                      combined_for_tsne = np.vstack([features_sub, centers_norm])
                      tsne_combined = TSNE(n_components=2, random_state=self.args.random_seed, perplexity=min(30, combined_for_tsne.shape[0]-1),
                                           n_iter=300, init='pca', learning_rate='auto', metric=tsne_metric)
                      reduced_combined = tsne_combined.fit_transform(combined_for_tsne)
                      reduced_centers = reduced_combined[len(features_sub):]
                      plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], c='black', marker='X', s=100, edgecolors='w', linewidth=1, label='Centers')
                      center_label_display = 'Centers'
                 except Exception as tsne_center_e: print(f"    Could not project centers with t-SNE: {tsne_center_e}")

            plt.title(f't-SNE Visualization ({self.__class__.__name__} Embeddings)')
            plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
            if known_label_display or unknown_label_display or center_label_display: plt.legend(markerscale=1.5)
            plt.grid(alpha=0.4); plt.tight_layout(); plt.savefig(f"{base_filename}_tsne.png"); plt.close()
            print("  t-SNE plot saved.")
        except ImportError: print("  Skipping t-SNE plot: scikit-learn (sklearn) is not installed.")
        except Exception as e: print(f"  Error during t-SNE visualization: {e}"); import traceback; traceback.print_exc()

    def visualize(self, results):
        print(f"[{self.__class__.__name__} Visualize] Generating plots...")
        os.makedirs("results", exist_ok=True)
        # --- 수정: base_filename 생성 시 self.args.osr_method 사용 ---
        method_name_for_file = self.args.osr_method if self.args.osr_method != 'all' else self.__class__.__name__.lower().replace("osr","")
        base_filename = f"results/{method_name_for_file}_osr_{self.args.dataset}_{self.args.seen_class_ratio}"
        # ---

        if 'labels' not in results or len(results['labels']) == 0: print("No data to visualize."); return

        self._visualize_confusion_matrix(results, base_filename)
        self._visualize_roc_curve(results, base_filename)
        self._visualize_tsne(results, base_filename) # 공통 t-SNE 호출

    def _visualize_confusion_matrix(self, results, base_filename):
        if 'confusion_matrix' in results and results['confusion_matrix'] is not None:
            f1_score_val = results.get('f1_score', float('nan')); fpr_tpr_val = results.get('fpr_at_tpr90', float('nan'))
            f1_str = f"Macro F1 (Known): {f1_score_val:.4f}" if pd.notna(f1_score_val) else "N/A"
            fpr_str = f"FPR@TPR90: {fpr_tpr_val:.4f}" if pd.notna(fpr_tpr_val) else "N/A"
            method_display_name = self.__class__.__name__.replace("OSR", "")

            plt.figure(figsize=(max(6, len(results['confusion_matrix_labels'])*0.6), max(5, len(results['confusion_matrix_labels'])*0.5)))
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                        xticklabels=results['confusion_matrix_names'], yticklabels=results['confusion_matrix_names'], annot_kws={"size": 8})
            plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Confusion Matrix ({method_display_name})\n({f1_str}, {fpr_str})')
            plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(rotation=0, fontsize=9)
            plt.tight_layout(); plt.savefig(f"{base_filename}_confusion.png"); plt.close()
            print(f"  Confusion matrix saved.")
        else: print("  Skipping confusion matrix (not found in results).")

    def _visualize_roc_curve(self, results, base_filename):
         if 'labels' not in results or 'scores_for_ranking' not in results: print("  Skipping ROC curve (missing labels or scores)."); return
         labels_np = results['labels']; scores = results['scores_for_ranking']
         unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)

         if len(np.unique(unknown_labels_mask)) > 1 and len(scores) == len(unknown_labels_mask):
             fpr, tpr, _ = roc_curve(unknown_labels_mask, scores)
             roc_auc_val = auc(fpr, tpr)
             plt.figure(figsize=(7, 6)); plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc_val:.3f}'); plt.plot([0, 1], [0, 1], 'k--')
             fpr90 = results.get('fpr_at_tpr90', float('nan'))
             if pd.notna(fpr90): plt.plot(fpr90, 0.9, 'ro', markersize=8, label=f'FPR@TPR90 ({fpr90:.3f})')
             plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC Curve ({self.__class__.__name__.replace("OSR", "")})')
             plt.legend(); plt.grid(); plt.tight_layout(); plt.savefig(f"{base_filename}_roc.png"); plt.close()
             print("  ROC curve saved.")
         else: print("  Skipping ROC curve (only one class or data mismatch).")

    def _get_seen_class_names(self):
         if self.datamodule.class_names is None or self.datamodule.original_seen_indices is None:
              print("Warning: Class names or seen indices not available. Using generic names.")
              return {i: f"Known_{i}" for i in range(self.num_known_classes)}
         seen_names = {}
         original_seen_indices = sorted(list(self.datamodule.original_seen_indices))
         for original_idx in original_seen_indices:
              if 0 <= original_idx < len(self.datamodule.class_names): seen_names[original_idx] = self.datamodule.class_names[original_idx]
              else: print(f"Warning: Index {original_idx} out of bounds for class names (len={len(self.datamodule.class_names)})."); seen_names[original_idx] = f"Class_{original_idx}"
         return seen_names

    def _get_cm_labels(self):
         seen_class_names_map = self._get_seen_class_names()
         cm_axis_labels_int = [-1] + sorted(list(self.datamodule.original_seen_indices))
         cm_axis_labels_names = ["Unknown"] + [seen_class_names_map.get(lbl, str(lbl)) for lbl in cm_axis_labels_int if lbl != -1]
         return cm_axis_labels_int, cm_axis_labels_names

    def _map_preds_to_original(self, preds_mapped_batch):
        original_seen_indices = self.datamodule.original_seen_indices
        if original_seen_indices is None: print("Warning: original_seen_indices not found. Returning mapped indices."); return preds_mapped_batch
        if not isinstance(original_seen_indices, np.ndarray): original_seen_indices = np.array(original_seen_indices)
        if isinstance(preds_mapped_batch, torch.Tensor): preds_mapped_batch = preds_mapped_batch.cpu().numpy()
        elif not isinstance(preds_mapped_batch, np.ndarray): preds_mapped_batch = np.array(preds_mapped_batch)

        preds_original_batch = np.full_like(preds_mapped_batch, -1, dtype=int) # Default to -1
        valid_mask = (preds_mapped_batch >= 0) & (preds_mapped_batch < len(original_seen_indices))
        preds_original_batch[valid_mask] = original_seen_indices[preds_mapped_batch[valid_mask]]
        if not np.all(valid_mask): print(f"Warning: Found {np.sum(~valid_mask)} invalid mapped indices during prediction mapping.")
        return preds_original_batch


class ThresholdOSR(OSRAlgorithm):
    """OSR using a simple threshold on the maximum softmax probability."""
    def __init__(self, model, datamodule, args):
        super().__init__(model, datamodule, args)
        self.threshold = getattr(args, 'param_threshold', 0.5)
        print(f"[ThresholdOSR Init] Using softmax threshold: {self.threshold:.4f}")

    def predict(self, dataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval().to(self.device)
        all_probs, all_preds_final, all_labels_original, all_max_probs, all_embeddings = [], [], [], [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (Threshold OSR)"):
                input_ids = batch['input_ids'].to(self.device); attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch.get('token_type_ids'); token_type_ids = token_type_ids.to(self.device) if token_type_ids is not None else None
                labels_orig = batch['label'].cpu().numpy()

                logits, embeddings = self.model(input_ids, attention_mask, token_type_ids)
                preds_mapped = torch.argmax(logits, dim=1)
                probs = F.softmax(logits, dim=1); max_probs, _ = torch.max(probs, dim=1)
                preds_mapped_cpu = preds_mapped.cpu().numpy(); max_probs_cpu = max_probs.cpu().numpy()
                final_batch_preds = np.full_like(preds_mapped_cpu, -1, dtype=int)
                accept_mask = max_probs_cpu >= self.threshold
                if np.any(accept_mask): final_batch_preds[accept_mask] = self._map_preds_to_original(preds_mapped_cpu[accept_mask])

                all_probs.append(probs.cpu().numpy()); all_preds_final.extend(final_batch_preds)
                all_labels_original.extend(labels_orig); all_max_probs.append(max_probs_cpu)
                all_embeddings.append(embeddings.cpu().numpy())

        all_probs = np.concatenate(all_probs) if all_probs else np.empty((0, self.num_known_classes))
        all_preds_final = np.array(all_preds_final); all_labels_original = np.array(all_labels_original)
        all_max_probs = np.concatenate(all_max_probs) if all_max_probs else np.array([])
        all_embeddings = np.concatenate(all_embeddings) if all_embeddings else np.empty((0, self.model.config.hidden_size))
        scores_for_ranking = -all_max_probs # Higher score = more unknown
        return all_probs, all_preds_final, all_labels_original, scores_for_ranking, all_embeddings

    def visualize(self, results):
        super().visualize(results) # Call common visualizations
        print(f"[{self.__class__.__name__} Visualize] Generating specific plots...")
        method_name_for_file = self.args.osr_method if self.args.osr_method != 'all' else self.__class__.__name__.lower().replace("osr","")
        base_filename = f"results/{method_name_for_file}_osr_{self.args.dataset}_{self.args.seen_class_ratio}"

        if 'max_probs' in results and len(results['max_probs']) > 0:
             labels_np = results['labels']; unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)
             max_probs = results['max_probs'] # Already collected in predict
             plt.figure(figsize=(7, 5)); sns.histplot(data=pd.DataFrame({'score': max_probs, 'Known': ~unknown_labels_mask}), x='score', hue='Known', kde=True, stat='density', common_norm=False, bins=50)
             plt.axvline(self.threshold, color='r', linestyle='--', label=f'Threshold={self.threshold:.2f}'); plt.title('Confidence Distribution (Threshold)'); plt.xlabel('Max Softmax Probability')
             plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_confidence.png"); plt.close()
             print("  Confidence distribution saved.")
        else: print("  Skipping confidence distribution (no scores).")
        print(f"[{self.__class__.__name__} Visualize] Finished specific plots.")


class OpenMaxOSR(OSRAlgorithm):
    """OpenMax OSR algorithm implementation."""
    def __init__(self, model, datamodule, args):
        super().__init__(model, datamodule, args)
        self.tail_size = getattr(args, 'param_openmax_tailsize', 50)
        self.alpha = getattr(args, 'param_openmax_alpha', 10)
        print(f"[OpenMaxOSR Init] Tail size: {self.tail_size}, Alpha: {self.alpha}, Known Classes: {self.num_known_classes}")
        self.mav: dict[int, np.ndarray] = {}; self.weibull_models: dict[int, tuple[float, float, float]] = {}
        self.feat_dim = model.config.hidden_size if hasattr(model, 'config') else 768

    def fit_weibull(self, dataloader):
        print("[OpenMaxOSR Fit] Fitting Weibull models...")
        self.model.eval().to(self.device); av_per_class = {c: [] for c in range(self.num_known_classes)}
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="OpenMax Fit: Collecting embeddings"):
                ids, attn = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                labels_mapped = batch["label"].to(self.device)
                logits, embeddings = self.model(ids, attn, tok); preds_mapped = torch.argmax(logits, dim=1)
                for i, true_mapped_idx in enumerate(labels_mapped):
                    true_idx = true_mapped_idx.item(); pred_idx = preds_mapped[i].item()
                    if true_idx >= 0 and pred_idx == true_idx and 0 <= true_idx < self.num_known_classes:
                        av_per_class[true_idx].append(embeddings[i].cpu().numpy())
        self.mav.clear(); self.weibull_models.clear()
        print("[OpenMaxOSR Fit] Calculating MAVs and fitting Weibull models...")
        for c_idx, av_list in tqdm(av_per_class.items(), desc="OpenMax Fit: Weibull Fitting"):
            if not av_list: print(f"  Warning: No samples for class index {c_idx}. Skipping."); continue
            avs = np.stack(av_list); self.mav[c_idx] = np.mean(avs, axis=0)
            distances = np.linalg.norm(avs - self.mav[c_idx], axis=1); distances_sorted = np.sort(distances)
            current_tail_size = min(self.tail_size, len(distances_sorted))
            if current_tail_size < 2:
                print(f"  Warning: Insufficient tail ({current_tail_size}) for class {c_idx}. Using default."); mean_dist = np.mean(distances_sorted) if len(distances_sorted) > 0 else 1.0; shape, loc, scale = 1.0, 0.0, mean_dist
            else:
                tail_distances = distances_sorted[-current_tail_size:]
                try:
                    shape, loc, scale = weibull_min.fit(tail_distances, floc=0)
                    if not np.isfinite([shape, scale]).all() or scale <= 1e-9 or shape <= 1e-9: print(f"  Warning: Weibull fit failed for class {c_idx}. Using default."); shape, loc, scale = 1.0, 0.0, np.mean(tail_distances) if len(tail_distances) > 0 else 1.0
                except Exception as e: print(f"  Warning: Weibull fit exception for class {c_idx}: {e}. Using default."); shape, loc, scale = 1.0, 0.0, np.mean(tail_distances) if len(tail_distances) > 0 else 1.0
            self.weibull_models[c_idx] = (shape, loc, scale)
        print("[OpenMaxOSR Fit] Weibull fitting complete.")

    def openmax_probability(self, embedding_av: np.ndarray, logits: np.ndarray) -> np.ndarray:
        if not self.mav or not self.weibull_models:
            print("Error: MAV/Weibull not calculated. Returning uncalibrated softmax."); exp_logits = np.exp(logits - np.max(logits)); softmax_probs = exp_logits / np.sum(exp_logits); return np.append(softmax_probs, 0.0)
        num_known = len(logits); current_alpha = self.alpha
        if num_known != self.num_known_classes: print(f"Warning: Logits dim ({num_known}) != expected ({self.num_known_classes}). Adjusting alpha."); current_alpha = min(self.alpha, num_known)

        distances = np.full(num_known, np.inf); cdf_scores = np.ones(num_known)
        for c_idx in range(num_known):
             if c_idx in self.mav: distances[c_idx] = np.linalg.norm(embedding_av - self.mav[c_idx])
             if c_idx in self.weibull_models and np.isfinite(distances[c_idx]):
                 shape, loc, scale = self.weibull_models[c_idx]; cdf_scores[c_idx] = weibull_min.cdf(distances[c_idx], shape, loc=loc, scale=scale)

        revised_logits = logits.copy(); sorted_indices = np.argsort(logits)[::-1]
        for rank, c_idx in enumerate(sorted_indices):
            if rank < current_alpha: revised_logits[c_idx] *= (1.0 - cdf_scores[c_idx])
        unknown_logit_score = np.sum(logits[sorted_indices[:current_alpha]] * cdf_scores[sorted_indices[:current_alpha]])
        final_logits = np.append(revised_logits, unknown_logit_score); exp_logits = np.exp(final_logits - np.max(final_logits))
        return exp_logits / np.sum(exp_logits)

    def predict(self, dataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not hasattr(self, 'mav') or not self.mav or not self.weibull_models:
            print("[OpenMaxOSR Predict] Fitting Weibull models..."); train_loader = self.datamodule.train_dataloader(); self.fit_weibull(train_loader)
            if not self.mav or not self.weibull_models: print("Error: Weibull fitting failed. Predictions might be unreliable.")

        self.model.eval().to(self.device)
        openmax_probs_list, preds_final_list, labels_original_list, all_embeddings = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (OpenMax OSR)"):
                ids, attn = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                labels_orig = batch["label"].cpu().numpy()
                logits_batch_gpu, embeddings_batch_gpu = self.model(ids, attn, tok)
                logits_batch = logits_batch_gpu.cpu().numpy(); embeddings_batch = embeddings_batch_gpu.cpu().numpy()
                all_embeddings.append(embeddings_batch) # Store embeddings

                for i in range(len(labels_orig)):
                    om_probs = self.openmax_probability(embeddings_batch[i], logits_batch[i])
                    openmax_probs_list.append(om_probs); pred_idx_with_unknown = np.argmax(om_probs)
                    if pred_idx_with_unknown == self.num_known_classes: pred_final = -1
                    else:
                        if 0 <= pred_idx_with_unknown < self.num_known_classes: pred_final = self._map_preds_to_original([pred_idx_with_unknown])[0]
                        else: print(f"Warning: Invalid OpenMax index {pred_idx_with_unknown}. Assigning -1."); pred_final = -1
                    preds_final_list.append(pred_final); labels_original_list.append(labels_orig[i])

        all_openmax_probs = np.vstack(openmax_probs_list) if openmax_probs_list else np.empty((0, self.num_known_classes + 1))
        all_preds_final = np.array(preds_final_list); all_labels_original = np.array(labels_original_list)
        all_embeddings = np.concatenate(all_embeddings) if all_embeddings else np.empty((0, self.feat_dim))
        scores_for_ranking = all_openmax_probs[:, -1] if all_openmax_probs.shape[1] > self.num_known_classes else np.zeros(len(all_labels_original)) # Higher unknown prob = more unknown
        return all_openmax_probs, all_preds_final, all_labels_original, scores_for_ranking, all_embeddings

    def visualize(self, results):
        super().visualize(results) # Call common visualizations
        print(f"[{self.__class__.__name__} Visualize] Generating specific plots...")
        method_name_for_file = self.args.osr_method if self.args.osr_method != 'all' else self.__class__.__name__.lower().replace("osr","")
        base_filename = f"results/{method_name_for_file}_osr_{self.args.dataset}_{self.args.seen_class_ratio}"

        unknown_probs = results.get("scores_for_ranking") # Use the ranking score which is unknown prob
        if unknown_probs is not None and len(unknown_probs) > 0:
            labels_np = results['labels']; unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)
            plt.figure(figsize=(7, 5)); sns.histplot(data=pd.DataFrame({'score': unknown_probs, 'Known': ~unknown_labels_mask}), x='score', hue='Known', kde=True, stat='density', common_norm=False, bins=50)
            plt.title('Unknown Probability Distribution (OpenMax)'); plt.xlabel('OpenMax Unknown Probability'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_confidence.png"); plt.close()
            print("  Unknown probability distribution saved.")
        else: print("  Skipping unknown probability distribution (no scores).")
        print(f"[{self.__class__.__name__} Visualize] Finished specific plots.")


class CROSROSR(OSRAlgorithm):
    """CROSR algorithm using reconstruction error."""
    def __init__(self, model, datamodule, args):
        if not isinstance(model, RobertaAutoencoder): raise TypeError("CROSR needs RobertaAutoencoder.")
        super().__init__(model, datamodule, args)
        self.threshold = getattr(args, 'param_crosr_reconstruction_threshold', 0.9)
        self.tail_size = getattr(args, 'param_crosr_tailsize', 100)
        print(f"[CROSROSR Init] Threshold (CDF): {self.threshold:.4f}, Tail Size: {self.tail_size}")
        self.weibull_model = None

    def fit_evt_model(self, dataloader):
        print("[CROSROSR Fit] Fitting EVT model on reconstruction errors...")
        self.model.eval().to(self.device); errors = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="CROSR Fit: Collecting errors"):
                ids, attn = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                valid_mask = batch['label'] >= 0
                if not valid_mask.any(): continue
                ids, attn = ids[valid_mask], attn[valid_mask]; tok = tok[valid_mask] if tok is not None else None
                _, cls_output, _, reconstructed = self.model(ids, attn, tok)
                errors.extend(torch.norm(reconstructed - cls_output, p=2, dim=1).cpu().numpy())
        if not errors: print("Warning: No errors collected for EVT fit."); self.weibull_model = (1.0, 0.0, 1.0); return
        errors_np = np.sort(np.array(errors)); tail = errors_np[-min(self.tail_size, len(errors_np)):]
        if len(tail) < 2: print(f"Warning: Insufficient tail ({len(tail)}). Using default."); self.weibull_model = (1.0, 0.0, np.mean(errors_np) if errors_np.size > 0 else 1.0); return
        try:
            shape, loc, scale = weibull_min.fit(tail, floc=0)
            if not np.isfinite([shape, scale]).all() or scale <= 1e-9 or shape <= 1e-9: raise ValueError("Invalid fit")
            self.weibull_model = (shape, loc, scale); print(f"  CROSR Fitted Weibull: shape={shape:.4f}, scale={scale:.4f}")
        except Exception as e: print(f"Warning: CROSR Weibull fit exception: {e}. Using default."); self.weibull_model = (1.0, 0.0, np.mean(tail) if len(tail) > 0 else 1.0)
        print("[CROSROSR Fit] Complete.")

    def predict(self, dataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.weibull_model is None:
            print("[CROSROSR Predict] Fitting EVT model..."); train_loader = self.datamodule.train_dataloader(); self.fit_evt_model(train_loader)
            if self.weibull_model is None: print("Error: EVT fitting failed."); self.weibull_model = (1.0, 0.0, 1.0)

        self.model.eval().to(self.device)
        all_recon_errors, all_unknown_probs, all_preds_final, all_labels_original, all_embeddings = [], [], [], [], []
        shape, loc, scale = self.weibull_model
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (CROSR)"):
                ids, attn = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                labels_orig = batch['label'].cpu().numpy()
                logits, cls_output, _, reconstructed = self.model(ids, attn, tok)
                preds_mapped = torch.argmax(logits, dim=1); recon_errors_batch = torch.norm(reconstructed - cls_output, p=2, dim=1).cpu().numpy()
                unknown_probs_batch = weibull_min.cdf(recon_errors_batch, shape, loc=loc, scale=scale)
                preds_mapped_cpu = preds_mapped.cpu().numpy(); batch_preds_final = np.full_like(preds_mapped_cpu, -1, dtype=int)
                accept_mask = unknown_probs_batch <= self.threshold
                if np.any(accept_mask): batch_preds_final[accept_mask] = self._map_preds_to_original(preds_mapped_cpu[accept_mask])

                all_recon_errors.extend(recon_errors_batch); all_unknown_probs.extend(unknown_probs_batch)
                all_preds_final.extend(batch_preds_final); all_labels_original.extend(labels_orig)
                all_embeddings.append(cls_output.cpu().numpy()) # Store embeddings

        all_recon_errors = np.array(all_recon_errors); all_unknown_probs = np.array(all_unknown_probs)
        all_preds_final = np.array(all_preds_final); all_labels_original = np.array(all_labels_original)
        all_embeddings = np.concatenate(all_embeddings) if all_embeddings else np.empty((0, self.model.config.hidden_size))
        scores_for_ranking = all_unknown_probs # Higher CDF prob = more unknown
        return all_recon_errors, all_preds_final, all_labels_original, scores_for_ranking, all_embeddings

    def visualize(self, results):
        super().visualize(results) # Call common visualizations
        print(f"[{self.__class__.__name__} Visualize] Generating specific plots...")
        method_name_for_file = self.args.osr_method if self.args.osr_method != 'all' else self.__class__.__name__.lower().replace("osr","")
        base_filename = f"results/{method_name_for_file}_osr_{self.args.dataset}_{self.args.seen_class_ratio}"

        recon_errors = results.get('scores') # Predict returns errors as first element
        unknown_probs = results.get('scores_for_ranking') # Predict returns unknown_probs as ranking score

        if recon_errors is not None and len(recon_errors) > 0:
             labels_np = results['labels']; unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)
             plt.figure(figsize=(7, 5)); sns.histplot(data=pd.DataFrame({'error': recon_errors, 'Known': ~unknown_labels_mask}), x='error', hue='Known', kde=True, stat='density', common_norm=False, bins=50)
             plt.title('Reconstruction Error Distribution (CROSR)'); plt.xlabel('L2 Reconstruction Error'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_error.png"); plt.close(); print("  Error dist saved.")
        else: print("  Skipping error distribution.")

        if unknown_probs is not None and len(unknown_probs) > 0:
             labels_np = results['labels']; unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)
             plt.figure(figsize=(7, 5)); sns.histplot(data=pd.DataFrame({'score': unknown_probs, 'Known': ~unknown_labels_mask}), x='score', hue='Known', kde=True, stat='density', common_norm=False, bins=50)
             plt.axvline(self.threshold, color='r', linestyle='--', label=f'Threshold={self.threshold:.2f}'); plt.title('Unknown Probability Distribution (CROSR)'); plt.xlabel('Weibull CDF of Recon Error'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_prob.png"); plt.close(); print("  Prob dist saved.")
        else: print("  Skipping probability distribution.")
        print(f"[{self.__class__.__name__} Visualize] Finished specific plots.")


class DOCOSR(OSRAlgorithm):
    """DOC algorithm using class-specific thresholds on sigmoid scores."""
    def __init__(self, model, datamodule, args):
        if not isinstance(model, DOCRobertaClassifier): raise TypeError("DOC needs DOCRobertaClassifier.")
        super().__init__(model, datamodule, args)
        self.k_sigma = getattr(args, 'param_doc_k', 3.0)
        print(f"[DOCOSR Init] k-sigma: {self.k_sigma}")
        self.gaussian_params: dict[int, tuple[float, float]] = {}; self.class_thresholds: dict[int, float] = {}

    def fit_gaussian(self, dataloader):
        print("[DOCOSR Fit] Fitting Gaussian models...")
        self.model.eval().to(self.device); scores_per_class = {c: [] for c in range(self.num_known_classes)}
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="DOC Fit: Collecting scores"):
                ids, attn = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                labels_mapped = batch['label'].to(self.device)
                valid_mask = labels_mapped >= 0
                if not valid_mask.any(): continue
                ids, attn = ids[valid_mask], attn[valid_mask]; tok = tok[valid_mask] if tok is not None else None; labels_mapped = labels_mapped[valid_mask]
                logits, _ = self.model(ids, attn, tok); sigmoid_scores = torch.sigmoid(logits)
                for i, true_mapped_idx in enumerate(labels_mapped):
                     idx = true_mapped_idx.item()
                     if 0 <= idx < self.num_known_classes: scores_per_class[idx].append(sigmoid_scores[i, idx].item())
        self.gaussian_params.clear(); self.class_thresholds.clear()
        print("[DOCOSR Fit] Calculating parameters and thresholds...")
        for c_idx, scores in tqdm(scores_per_class.items(), desc="DOC Fit: Fitting"):
            if len(scores) >= 2:
                scores_np = np.array([s for s in scores if s > 0.1])
                if len(scores_np) < 2: print(f"Warning: Insufficient scores ({len(scores_np)}) for class {c_idx}. Using default."); self.gaussian_params[c_idx] = (0.5, 0.5); self.class_thresholds[c_idx] = 0.5; continue
                mirrored_scores = 1.0 + (1.0 - scores_np); combined_scores = np.concatenate([scores_np, mirrored_scores])
                _, std_combined = norm.fit(combined_scores); std_combined = max(std_combined, 1e-6)
                threshold = max(0.5, 1.0 - self.k_sigma * std_combined)
                self.gaussian_params[c_idx] = (np.mean(scores_np), std_combined); self.class_thresholds[c_idx] = threshold
            else: print(f"Warning: Insufficient samples ({len(scores)}) for class {c_idx}. Using default."); self.gaussian_params[c_idx] = (0.5, 0.5); self.class_thresholds[c_idx] = 0.5
        print("[DOCOSR Fit] Complete.")

    def predict(self, dataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.class_thresholds:
            print("[DOCOSR Predict] Fitting Gaussian models..."); train_loader = self.datamodule.train_dataloader(); self.fit_gaussian(train_loader)
            if not self.class_thresholds: print("Error: Gaussian fitting failed."); self.class_thresholds = {i: 0.5 for i in range(self.num_known_classes)}; self.gaussian_params = {i: (0.5, 0.5) for i in range(self.num_known_classes)}

        self.model.eval().to(self.device)
        all_sigmoid_scores, all_max_scores, all_preds_final, all_labels_original, all_embeddings = [], [], [], [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (DOC OSR)"):
                ids, attn = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                labels_orig = batch['label'].cpu().numpy()
                logits, embeddings = self.model(ids, attn, tok) # Get embeddings
                sigmoid_scores_batch = torch.sigmoid(logits); preds_mapped = torch.argmax(sigmoid_scores_batch, dim=1)
                max_scores_batch, _ = torch.max(sigmoid_scores_batch, dim=1)
                pred_indices_np = preds_mapped.cpu().numpy(); max_scores_np = max_scores_batch.cpu().numpy()
                batch_preds_final = np.full_like(pred_indices_np, -1, dtype=int); accept_mask = np.zeros_like(pred_indices_np, dtype=bool)
                for i in range(len(labels_orig)):
                     pred_mapped_idx = pred_indices_np[i]; threshold = self.class_thresholds.get(pred_mapped_idx, 0.5)
                     if max_scores_np[i] >= threshold: accept_mask[i] = True
                if np.any(accept_mask): batch_preds_final[accept_mask] = self._map_preds_to_original(pred_indices_np[accept_mask])

                all_sigmoid_scores.append(sigmoid_scores_batch.cpu().numpy()); all_max_scores.extend(max_scores_np)
                all_preds_final.extend(batch_preds_final); all_labels_original.extend(labels_orig)
                all_embeddings.append(embeddings.cpu().numpy()) # Store embeddings

        all_sigmoid_scores = np.vstack(all_sigmoid_scores) if all_sigmoid_scores else np.empty((0, self.num_known_classes))
        all_max_scores = np.array(all_max_scores); all_preds_final = np.array(all_preds_final); all_labels_original = np.array(all_labels_original)
        all_embeddings = np.concatenate(all_embeddings) if all_embeddings else np.empty((0, self.model.config.hidden_size))
        scores_for_ranking = -all_max_scores # Higher score = more unknown
        return all_sigmoid_scores, all_preds_final, all_labels_original, scores_for_ranking, all_embeddings

    def visualize(self, results):
        super().visualize(results) # Call common visualizations
        print(f"[{self.__class__.__name__} Visualize] Generating specific plots...")
        method_name_for_file = self.args.osr_method if self.args.osr_method != 'all' else self.__class__.__name__.lower().replace("osr","")
        base_filename = f"results/{method_name_for_file}_osr_{self.args.dataset}_{self.args.seen_class_ratio}"

        max_scores = -results.get('scores_for_ranking', np.array([])) # Convert back from ranking score

        if len(max_scores) > 0:
            labels_np = results['labels']; unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)
            plt.figure(figsize=(7, 5)); sns.histplot(data=pd.DataFrame({'score': max_scores, 'Known': ~unknown_labels_mask}), x='score', hue='Known', kde=True, stat='density', common_norm=False, bins=50)
            avg_threshold = np.mean(list(self.class_thresholds.values())) if self.class_thresholds else 0.5
            plt.axvline(avg_threshold, color='g', linestyle=':', label=f'Avg Thresh~{avg_threshold:.2f}')
            plt.title('Max Sigmoid Score Distribution (DOC)'); plt.xlabel('Max Sigmoid Score'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_score.png"); plt.close(); print("  Score dist saved.")
        else: print("  Skipping score distribution.")

        # Z-Score plot can be added here if needed, using gaussian_params and max_scores
        print(f"[{self.__class__.__name__} Visualize] Finished specific plots.")


class ADBOSR(OSRAlgorithm):
    """OSR algorithm using Adaptive Decision Boundaries (ADB)."""
    def __init__(self, model, datamodule, args):
        if not isinstance(model, RobertaADB): raise TypeError("ADBOSR needs RobertaADB.")
        super().__init__(model, datamodule, args)
        self.distance_metric = getattr(args, 'param_adb_distance', 'cosine')
        print(f"[ADBOSR Init] Distance metric: {self.distance_metric}")

    def compute_distances(self, features_norm: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        if self.distance_metric == 'cosine':
            centers_norm = F.normalize(centers, p=2, dim=-1); similarity = torch.matmul(features_norm, centers_norm.t())
            similarity = torch.clamp(similarity, -1.0 + 1e-7, 1.0 - 1e-7); return 1.0 - similarity
        elif self.distance_metric == 'euclidean':
             centers_norm = F.normalize(centers, p=2, dim=-1); return torch.cdist(features_norm, centers_norm, p=2)
        else: raise ValueError(f"Unknown distance: {self.distance_metric}")

    def predict(self, dataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval().to(self.device)
        all_features, all_distances, all_preds_final, all_labels_original, all_min_distances = [], [], [], [], []
        centers = self.model.centers.detach(); radii = self.model.get_radii().detach()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (ADB OSR)"):
                input_ids = batch['input_ids'].to(self.device); attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch.get('token_type_ids'); token_type_ids = token_type_ids.to(self.device) if token_type_ids is not None else None
                labels_orig = batch['label'].cpu().numpy()
                logits, features_norm = self.model(input_ids, attention_mask, token_type_ids); preds_mapped = torch.argmax(logits, dim=1)
                distances_batch = self.compute_distances(features_norm, centers); min_distances_batch, _ = torch.min(distances_batch, dim=1)
                closest_radii_batch = radii[preds_mapped]
                pred_indices_np = preds_mapped.cpu().numpy(); min_distances_np = min_distances_batch.cpu().numpy(); closest_radii_np = closest_radii_batch.cpu().numpy()
                batch_preds_final = np.full_like(pred_indices_np, -1, dtype=int); accept_mask = min_distances_np <= closest_radii_np
                if np.any(accept_mask): batch_preds_final[accept_mask] = self._map_preds_to_original(pred_indices_np[accept_mask])

                all_features.append(features_norm.cpu().numpy()); all_distances.append(distances_batch.cpu().numpy())
                all_preds_final.extend(batch_preds_final); all_labels_original.extend(labels_orig); all_min_distances.extend(min_distances_np)

        all_features = np.concatenate(all_features) if all_features else np.empty((0, self.model.feat_dim))
        all_distances = np.concatenate(all_distances) if all_distances else np.empty((0, self.num_known_classes))
        all_preds_final = np.array(all_preds_final); all_labels_original = np.array(all_labels_original); all_min_distances = np.array(all_min_distances)
        scores_for_ranking = all_min_distances # Higher distance = more unknown
        return all_distances, all_preds_final, all_labels_original, scores_for_ranking, all_features # Return features as embeddings

    def visualize(self, results: dict):
        super().visualize(results) # Call common visualizations
        print(f"[{self.__class__.__name__} Visualize] Generating specific plots...")
        method_name_for_file = self.args.osr_method if self.args.osr_method != 'all' else self.__class__.__name__.lower().replace("osr","")
        base_filename = f"results/{method_name_for_file}_osr_{self.args.dataset}_{self.args.seen_class_ratio}"

        min_distances = results.get('scores_for_ranking') # Use ranking score which is min distance
        if min_distances is not None and len(min_distances) > 0:
            labels_np = results['labels']; unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)
            plt.figure(figsize=(7, 5)); sns.histplot(data=pd.DataFrame({'dist': min_distances, 'Known': ~unknown_labels_mask}), x='dist', hue='Known', kde=True, stat='density', common_norm=False, bins=50)
            mean_radius = self.model.get_radii().detach().mean().item()
            plt.axvline(mean_radius, color='g', linestyle=':', label=f'Avg Radius~{mean_radius:.3f}')
            plt.title(f'Min Distance Distribution (ADB - {self.distance_metric})'); plt.xlabel(f'Min {self.distance_metric.capitalize()} Distance'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_distance.png"); plt.close(); print("  Distance dist saved.")
        else: print("  Skipping distance distribution.")
        print(f"[{self.__class__.__name__} Visualize] Finished specific plots.")


# =============================================================================
# Training and Evaluation Functions
# =============================================================================
class FinalizeLoggerCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        if hasattr(trainer.logger, 'finalize'): trainer.logger.finalize("finished"); print("Logger finalized.")
    def on_exception(self, trainer, pl_module, exception):
         if hasattr(trainer.logger, 'finalize'): trainer.logger.finalize("interrupted"); print("Logger finalized after exception.")

def train_model(model, datamodule, args):
    """Trains a PyTorch Lightning model."""
    print(f"\n--- Training Model: {model.__class__.__name__} ---")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name_prefix = args.osr_method; run_name = f"{args.dataset}_{run_name_prefix}_{args.seen_class_ratio}_{timestamp}"
    output_dir = os.path.join("checkpoints", run_name); log_dir = os.path.join("logs", run_name)
    os.makedirs(output_dir, exist_ok=True); os.makedirs(log_dir, exist_ok=True)

    total_steps, train_batches = 0, 0
    try:
        train_loader = datamodule.train_dataloader(); train_batches = len(train_loader); total_steps = train_batches * args.epochs
        warmup_steps = min(int(args.warmup_ratio * total_steps), args.max_warmup_steps)
        if total_steps <= 0: raise ValueError("Total steps cannot be zero.")
    except Exception as e: print(f"Warning: Could not determine steps: {e}. Using defaults."); total_steps = 10000; warmup_steps = 500; train_batches = total_steps // args.epochs

    if hasattr(model, 'total_steps'): model.total_steps = total_steps
    if hasattr(model, 'warmup_steps'): model.warmup_steps = warmup_steps
    print(f"Scheduler: Total steps={total_steps}, Warmup steps={warmup_steps}")

    monitor_metric = "val_loss"; monitor_mode = "min"
    checkpoint_callback = ModelCheckpoint(dirpath=output_dir, filename=f"{model.__class__.__name__}-{{epoch:02d}}-{{{monitor_metric}:.4f}}", save_top_k=1, verbose=False, monitor=monitor_metric, mode=monitor_mode)
    early_stopping_callback = EarlyStopping(monitor=monitor_metric, patience=args.early_stopping_patience, min_delta=args.early_stopping_delta, verbose=True, mode=monitor_mode)
    finalize_logger_callback = FinalizeLoggerCallback()

    try: logger = TensorBoardLogger(save_dir=os.path.dirname(log_dir), name=os.path.basename(log_dir), version=""); print(f"Using TensorBoardLogger in: {logger.log_dir}")
    except ImportError: print("TensorBoard not available. Using CSVLogger."); logger = CSVLogger(save_dir=os.path.dirname(log_dir), name=os.path.basename(log_dir), version=""); print(f"Using CSVLogger in: {logger.log_dir}")

    use_gpu = args.force_gpu or torch.cuda.is_available()
    trainer_kwargs = { "max_epochs": args.epochs, "callbacks": [checkpoint_callback, early_stopping_callback, finalize_logger_callback], "logger": logger, "log_every_n_steps": max(1, train_batches // 10) if train_batches > 0 else 50, "precision": "16-mixed" if use_gpu else 32, "gradient_clip_val": args.gradient_clip_val, "deterministic": "warn", "benchmark": False if args.random_seed else True, "enable_progress_bar": False }
    if use_gpu: trainer_kwargs["accelerator"] = "gpu"; trainer_kwargs["devices"] = [args.gpu_id]
    else: print("Using CPU for training."); trainer_kwargs["accelerator"] = "cpu"

    trainer = pl.Trainer(**trainer_kwargs); print(f"Starting training for {args.epochs} epochs (run: {run_name})...")
    best_checkpoint_path = None
    try: trainer.fit(model, datamodule=datamodule); best_checkpoint_path = checkpoint_callback.best_model_path; print(f"Training finished. Best model saved at: {best_checkpoint_path}")
    except Exception as e: print(f"\nError during training: {e}"); import traceback; traceback.print_exc(); print("Attempting to use last saved checkpoint..."); best_checkpoint_path = checkpoint_callback.best_model_path

    if best_checkpoint_path and os.path.exists(best_checkpoint_path): return best_checkpoint_path
    else:
         print(f"Warning: No valid checkpoint found in {output_dir}."); existing_ckpts = [f for f in os.listdir(output_dir) if f.endswith('.ckpt')]
         if existing_ckpts:
             best_guess = next((os.path.join(output_dir, ckpt) for ckpt in existing_ckpts if 'epoch' in ckpt and 'val_loss' in ckpt), None)
             if best_guess: print(f"  Returning best guess: {best_guess}"); return best_guess
             else: print(f"  Found other checkpoints: {existing_ckpts}. Returning first one."); return os.path.join(output_dir, existing_ckpts[0])
         return None

# --- OSR Evaluation Wrappers ---
MODEL_CLASS_MAP = { 'standard': RobertaClassifier, 'crosr': RobertaAutoencoder, 'doc': DOCRobertaClassifier, 'adb': RobertaADB }
METHODS_NEEDING_SPECIAL_MODEL = ['crosr', 'doc', 'adb']
METHODS_NEEDING_RETRAINING_PER_TRIAL = ['crosr', 'doc', 'adb']
METHODS_USING_STANDARD_MODEL = ['threshold', 'openmax']

def _prepare_evaluation(method_name, base_model, datamodule, args, osr_algorithm_class):
    """Handles model checking, parameter setup, and potential retraining."""
    print(f"\n--- Preparing for {method_name.upper()} OSR Evaluation ---")
    target_model_class_name = method_name if method_name in METHODS_NEEDING_SPECIAL_MODEL else 'standard'
    target_model_class = MODEL_CLASS_MAP[target_model_class_name]
    model_for_final_eval = None; needs_final_training = False; best_params_from_tuning = {}

    if args.parameter_search and (args.osr_method == 'all' or method_name == args.osr_method):
        print(f"Starting Optuna search for {method_name.upper()}..."); tuner = OptunaHyperparameterTuner(method_name, datamodule, args)
        if method_name in METHODS_NEEDING_RETRAINING_PER_TRIAL:
            print("  Tuning Mode: Retraining model per trial.")
            def train_and_evaluate_trial(trial_args):
                print(f"\n  Starting Training & Eval for Trial..."); trial_start_time = time.time()
                num_classes = datamodule.num_seen_classes
                # 모델 초기화 인자 설정
                init_kwargs = { 'model_name': trial_args.model, 'num_classes': num_classes, 'weight_decay': trial_args.weight_decay, 'warmup_steps': trial_args.max_warmup_steps, 'total_steps': 0 }
                if target_model_class == RobertaAutoencoder: init_kwargs['learning_rate'] = trial_args.lr; init_kwargs['reconstruction_weight'] = trial_args.param_crosr_recon_weight
                elif target_model_class == DOCRobertaClassifier: init_kwargs['learning_rate'] = trial_args.lr
                elif target_model_class == RobertaADB: init_kwargs['lr'] = trial_args.lr; init_kwargs['lr_adb'] = trial_args.lr_adb; init_kwargs['param_adb_delta'] = trial_args.param_adb_delta; init_kwargs['param_adb_alpha'] = trial_args.param_adb_alpha; init_kwargs['adb_freeze_backbone'] = trial_args.adb_freeze_backbone

                # 모델 초기화 시도
                print(f"    Initializing {target_model_class.__name__}...")
                try:
                    trial_model = target_model_class(**init_kwargs)
                except Exception as e:
                    import traceback
                    print(f"    Error initializing model: {e}")
                    traceback.print_exc() # 초기화 에러 시 상세 정보 출력
                    return {}, -1e9 # Optuna에 실패 알림

                # Trial용 학습 설정
                tuning_epochs = max(3, args.epochs // 2); print(f"    Training trial model for {tuning_epochs} epochs...")
                trial_args_copy = copy.deepcopy(trial_args); trial_args_copy.epochs = tuning_epochs
                trial_run_id = f"trial_{method_name}_{trial_args.dataset}_{datetime.now().strftime('%H%M%S%f')}"; trial_args_copy.osr_method = trial_run_id

                # Trial 모델 학습
                checkpoint_path = train_model(trial_model, datamodule, trial_args_copy); trial_ckpt_dir = os.path.dirname(checkpoint_path) if checkpoint_path else None
                if checkpoint_path is None or not os.path.exists(checkpoint_path):
                    print("    Trial Training Failed.")
                    if trial_ckpt_dir and os.path.exists(trial_ckpt_dir): shutil.rmtree(trial_ckpt_dir, ignore_errors=True) # 실패 시 체크포인트 정리
                    return {}, -1e9 # Optuna에 실패 알림

                # 학습된 Trial 모델 로드 시도
                print(f"    Loading best model from trial checkpoint: {checkpoint_path}")
                try:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    trained_trial_model = target_model_class.load_from_checkpoint(checkpoint_path, map_location=device)
                except Exception as load_e:
                    import traceback
                    print(f"    Failed to load trial checkpoint '{checkpoint_path}': {load_e}")
                    traceback.print_exc() # 로딩 에러 시 상세 정보 출력
                    if trial_ckpt_dir and os.path.exists(trial_ckpt_dir): shutil.rmtree(trial_ckpt_dir, ignore_errors=True) # 실패 시 체크포인트 정리
                    return {}, -1e9 # Optuna에 실패 알림

                # Trial 모델 평가 시도
                print("    Evaluating trial model..."); evaluator = osr_algorithm_class(trained_trial_model, datamodule, trial_args_copy)
                results = {} # 결과 초기화
                try:
                    results = evaluator.evaluate(datamodule.test_dataloader())
                except Exception as eval_e:
                    import traceback
                    print(f"    Error during trial evaluation: {eval_e}")
                    traceback.print_exc() # 평가 에러 시 상세 정보 출력
                    # 결과는 비어있는 상태로 유지
                finally:
                    # 평가 성공/실패 여부와 관계없이 생성된 파일 정리
                    if trial_ckpt_dir and os.path.exists(trial_ckpt_dir):
                        shutil.rmtree(trial_ckpt_dir, ignore_errors=True)
                        print(f"    Cleaned up trial checkpoint directory: {trial_ckpt_dir}")
                    log_dir_path = os.path.join("logs", f"{args.dataset}_{trial_run_id}_{args.seen_class_ratio}") # 로그 디렉토리 경로 수정 확인 필요
                    if os.path.exists(log_dir_path):
                        shutil.rmtree(log_dir_path, ignore_errors=True)
                        print(f"    Cleaned up trial log directory: {log_dir_path}")

                # 점수 계산 및 반환
                score = results.get(args.tuning_metric)
                score = score if score is not None and np.isfinite(score) else -1e9 # 유효하지 않으면 실패 점수
                print(f"  Trial completed in {time.time() - trial_start_time:.2f}s. Score ({args.tuning_metric}): {score:.4f if score > -1e8 else 'Fail'})")
                return results, float(score) # Optuna에 결과와 점수 반환
            best_params_from_tuning, _ = tuner.tune(tuner._objective_with_retraining, train_and_evaluate_trial); needs_final_training = True
        else:
            print("  Tuning Mode: Evaluating pre-trained model.");
            def evaluate_trial_no_retraining(trial_args):
                trial_start_time = time.time(); print("    Evaluating trial..."); evaluator = osr_algorithm_class(base_model, datamodule, trial_args)
                try: results = evaluator.evaluate(datamodule.test_dataloader())
                except Exception as e: print(f"    Error during trial evaluation: {e}"); results = {}
                score = results.get(args.tuning_metric); score = score if score is not None and np.isfinite(score) else -1e9; print(f"  Trial completed in {time.time() - trial_start_time:.2f}s. Score ({args.tuning_metric}): {score:.4f if score > -1e8 else 'Fail'})"); return results, float(score)
            best_params_from_tuning, _ = tuner.tune(tuner._objective_evaluate_only, evaluate_trial_no_retraining); needs_final_training = False
        print(f"\nApplying best tuned parameters for final {method_name.upper()} run:"); [setattr(args, name, value) or print(f"  {name}: {value}") for name, value in best_params_from_tuning.items()]
    else:
        needs_final_training = False; loaded_params = load_best_params(method_name, args.dataset, args.seen_class_ratio)
        param_source = "loaded" if loaded_params else "defaults"; loaded_params = loaded_params or get_default_best_params(method_name)
        print(f"Applying parameters ({param_source}) for final {method_name.upper()} evaluation:"); [setattr(args, name, value) or print(f"  {name}: {value}") if hasattr(args, name) else print(f"  (Skipping '{name}')") for name, value in loaded_params.items()]
        if method_name in METHODS_NEEDING_SPECIAL_MODEL and not isinstance(base_model, target_model_class): print(f"Warning: Model type mismatch. Retraining required."); needs_final_training = True

    if needs_final_training:
        print(f"\nTraining final {target_model_class.__name__} model..."); num_classes = datamodule.num_seen_classes
        init_kwargs = { 'model_name': args.model, 'num_classes': num_classes, 'weight_decay': args.weight_decay, 'warmup_steps': args.max_warmup_steps, 'total_steps': 0 }
        if target_model_class == RobertaAutoencoder: init_kwargs['learning_rate'] = args.lr; init_kwargs['reconstruction_weight'] = args.param_crosr_recon_weight
        elif target_model_class == DOCRobertaClassifier: init_kwargs['learning_rate'] = args.lr
        elif target_model_class == RobertaADB: init_kwargs['lr'] = args.lr; init_kwargs['lr_adb'] = args.lr_adb; init_kwargs['param_adb_delta'] = args.param_adb_delta; init_kwargs['param_adb_alpha'] = args.param_adb_alpha; init_kwargs['adb_freeze_backbone'] = args.adb_freeze_backbone
        final_model_instance = target_model_class(**init_kwargs); final_args = copy.deepcopy(args); final_args.epochs = args.epochs; final_args.osr_method = f"final_{method_name}"
        final_checkpoint_path = train_model(final_model_instance, datamodule, final_args)
        if final_checkpoint_path is None or not os.path.exists(final_checkpoint_path): raise RuntimeError(f"Failed to train final model for {method_name.upper()}.")
        print(f"Loading final trained model from: {final_checkpoint_path}")
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_for_final_eval = target_model_class.load_from_checkpoint(final_checkpoint_path, map_location=device)
        except Exception as load_e: raise RuntimeError(f"Failed to load final checkpoint '{final_checkpoint_path}': {load_e}") # Include path in error
    else:
         print(f"Using the initially provided/loaded model ({type(base_model).__name__}) for final evaluation.")
         if method_name in METHODS_NEEDING_SPECIAL_MODEL and not isinstance(base_model, target_model_class): raise TypeError(f"Method {method_name} requires {target_model_class.__name__}, but received {type(base_model).__name__}.")
         model_for_final_eval = base_model
    return model_for_final_eval

def evaluate_threshold_osr(base_model, datamodule, args, all_results):
    method_name = 'threshold'; model_for_eval = _prepare_evaluation(method_name, base_model, datamodule, args, ThresholdOSR)
    print(f"\n--- Running Final {method_name.upper()} Evaluation ---"); evaluator = ThresholdOSR(model_for_eval, datamodule, args)
    results = evaluator.evaluate(datamodule.test_dataloader()); evaluator.visualize(results); all_results[method_name] = results; return results

def evaluate_openmax_osr(base_model, datamodule, args, all_results):
    method_name = 'openmax'; model_for_eval = _prepare_evaluation(method_name, base_model, datamodule, args, OpenMaxOSR)
    print(f"\n--- Running Final {method_name.upper()} Evaluation ---"); evaluator = OpenMaxOSR(model_for_eval, datamodule, args)
    results = evaluator.evaluate(datamodule.test_dataloader()); evaluator.visualize(results); all_results[method_name] = results; return results

def evaluate_crosr_osr(base_model, datamodule, args, all_results):
    method_name = 'crosr'; model_for_eval = _prepare_evaluation(method_name, base_model, datamodule, args, CROSROSR)
    print(f"\n--- Running Final {method_name.upper()} Evaluation ---"); evaluator = CROSROSR(model_for_eval, datamodule, args)
    results = evaluator.evaluate(datamodule.test_dataloader()); evaluator.visualize(results); all_results[method_name] = results; return results

def evaluate_doc_osr(base_model, datamodule, args, all_results):
    method_name = 'doc'; model_for_eval = _prepare_evaluation(method_name, base_model, datamodule, args, DOCOSR)
    print(f"\n--- Running Final {method_name.upper()} Evaluation ---"); evaluator = DOCOSR(model_for_eval, datamodule, args)
    results = evaluator.evaluate(datamodule.test_dataloader()); evaluator.visualize(results); all_results[method_name] = results; return results

def evaluate_adb_osr(base_model, datamodule, args, all_results):
    method_name = 'adb'; model_for_eval = _prepare_evaluation(method_name, base_model, datamodule, args, ADBOSR)
    print(f"\n--- Running Final {method_name.upper()} Evaluation ---"); evaluator = ADBOSR(model_for_eval, datamodule, args)
    results = evaluator.evaluate(datamodule.test_dataloader()); evaluator.visualize(results); all_results[method_name] = results; return results

# --- OSCR Curve Calculation and Visualization ---
def calculate_oscr_curve(results, datamodule):
    """Calculates CCR vs FPR for the OSCR curve."""
    if 'predictions' not in results or 'labels' not in results or 'scores_for_ranking' not in results: return np.array([0,1]), np.array([0,0])
    preds = np.array(results['predictions']); labels = np.array(results['labels']); scores_for_ranking = np.array(results['scores_for_ranking'])
    if len(preds) != len(labels) or len(preds) != len(scores_for_ranking): print("Warning: Length mismatch in OSCR data."); return np.array([0,1]), np.array([0,0])

    unknown_labels_mask = datamodule._determine_unknown_labels(labels); known_mask = ~unknown_labels_mask
    is_correct_known = (preds == labels) & known_mask; is_false_positive = (preds != -1) & unknown_labels_mask
    valid_score_mask = ~np.isnan(scores_for_ranking)
    if not np.all(valid_score_mask): print(f"Warning: Filtering {np.sum(~valid_score_mask)} NaNs in OSCR scores."); scores_for_ranking = scores_for_ranking[valid_score_mask]; is_correct_known = is_correct_known[valid_score_mask]; is_false_positive = is_false_positive[valid_score_mask]

    n_known = np.sum(known_mask); n_unknown = np.sum(unknown_labels_mask)
    if n_known == 0 or n_unknown == 0: print("Warning: No known or unknown samples for OSCR."); return np.array([0,1]), np.array([0,0])

    sorted_indices = np.argsort(scores_for_ranking); sorted_correct_known = is_correct_known[sorted_indices]; sorted_false_positive = is_false_positive[sorted_indices]
    ccr = np.cumsum(sorted_correct_known) / n_known; fpr = np.cumsum(sorted_false_positive) / n_unknown
    fpr = np.insert(fpr, 0, 0.0); ccr = np.insert(ccr, 0, 0.0)
    return fpr, ccr

def visualize_oscr_curves(all_results, datamodule, args):
    """Plots OSCR curves for comparing multiple OSR methods."""
    print("\nGenerating OSCR Comparison Curve..."); plt.figure(figsize=(8, 7)); method_found = False; plotted_methods = []
    sorted_methods = sorted(all_results.keys())
    for method in sorted_methods:
        results = all_results.get(method)
        if results and isinstance(results, dict) and 'error' not in results:
            try:
                fpr, ccr = calculate_oscr_curve(results, datamodule)
                if len(fpr) > 1 and len(ccr) > 1: oscr_auc = np.trapz(ccr, fpr); plt.plot(fpr, ccr, lw=2.5, label=f'{method.upper()} (AUC = {oscr_auc:.3f})', alpha=0.8); method_found = True; plotted_methods.append(method)
                else: print(f"  Skipping OSCR plot for {method}: Insufficient data points.")
            except Exception as e: print(f"  Error calculating OSCR for {method}: {e}")
    if not method_found: print("No valid results found to plot OSCR."); plt.close(); return
    plt.plot([0, 1], [1, 0], color='grey', lw=1.5, linestyle='--', label='Ideal Closed-Set'); plt.xlim([-0.02, 1.02]); plt.ylim([-0.02, 1.05])
    plt.xlabel('FPR', fontsize=12); plt.ylabel('CCR', fontsize=12); plt.title(f'OSCR Curves ({args.dataset}, Seen: {args.seen_class_ratio*100:.0f}%)', fontsize=14)
    plt.legend(loc="lower left", fontsize=10); plt.grid(True, linestyle=':', alpha=0.6); save_path = f"results/oscr_comparison_{args.dataset}_{args.seen_class_ratio}.png"
    plt.tight_layout(); plt.savefig(save_path); plt.close(); print(f"OSCR comparison curve saved to: {save_path}")

# --- Main Evaluation Orchestrator ---
def evaluate_osr_main(initial_trained_model, datamodule, args):
    """Runs evaluation for the selected OSR method(s)."""
    all_results = {}; os.makedirs("results", exist_ok=True)
    if args.parameter_search: print("\n" + "="*70 + f"\n{' ' * 15}Hyperparameter Tuning Mode (Optuna)\n" + "="*70 + f"\nTuning Metric: {args.tuning_metric}, Trials: {args.n_trials}, Methods: {args.osr_method}\n" + "="*70 + "\n")

    method_map = { "threshold": evaluate_threshold_osr, "openmax": evaluate_openmax_osr, "crosr": evaluate_crosr_osr, "doc": evaluate_doc_osr, "adb": evaluate_adb_osr }
    methods_to_run = list(method_map.keys()) if args.osr_method == "all" else [args.osr_method]

    for method in methods_to_run:
        if method in method_map:
            try: print(f"\n>>> Starting evaluation for: {method.upper()} <<<"); method_map[method](initial_trained_model, datamodule, args, all_results)
            except Exception as e: print(f"\n!!!!! Error evaluating method {method.upper()}: {e} !!!!!"); import traceback; traceback.print_exc(); all_results[method] = {"error": str(e)}
        else: print(f"Warning: Unknown OSR method '{method}' skipped.")

    # --- Save Consolidated Results ---
    results_suffix = "_tuned" if args.parameter_search else ""
    results_filename = f"results/final_{args.model.replace('/','_')}_{args.dataset}_{args.osr_method}_{args.seen_class_ratio}{results_suffix}.json"
    def json_converter(obj):
        if isinstance(obj, (np.integer, np.floating)): return obj.item() # Convert numpy scalars
        elif isinstance(obj, (np.ndarray,)): return obj.tolist()
        elif isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
        elif isinstance(obj, (torch.Tensor)): return obj.cpu().numpy().tolist()
        elif isinstance(obj, set): return list(obj)
        elif isinstance(obj, pathlib.Path): return str(obj)
        elif pd.isna(obj): return None # Handle pandas NaT etc.
        try: return obj.__dict__
        except AttributeError: raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    try:
        summary_results = {}
        keys_to_exclude = [ 'predictions', 'labels', 'probs', 'scores', 'features', 'distances', 'max_probs', 'unknown_probs', 'max_scores', 'min_distances', 'reconstruction_errors', 'z_scores', 'embeddings', 'logits', 'sigmoid_scores', 'encoded', 'reconstructed', 'preds_mapped', 'labels_original', 'confusion_matrix', 'scores_for_ranking' ]
        for method, res in all_results.items():
             if isinstance(res, dict): summary_results[method] = {k: v for k, v in res.items() if k not in keys_to_exclude}; summary_results[method]['confusion_matrix_labels'] = res.get('confusion_matrix_labels'); summary_results[method]['confusion_matrix_names'] = res.get('confusion_matrix_names')
             else: summary_results[method] = res
        with open(results_filename, 'w', encoding='utf-8') as f: json.dump(summary_results, f, indent=2, ensure_ascii=False, default=json_converter)
        print(f"\nConsolidated results summary saved to: {results_filename}")
    except Exception as e:
        print(f"\nError saving summary results to JSON: {e}"); pickle_filename = results_filename.replace(".json", "_full.pkl")
        try:
            import pickle
            with open(pickle_filename, 'wb') as pf: pickle.dump(all_results, pf)
            print(f"Warning: JSON saving failed due to '{e}'. Saved full results as pickle: {pickle_filename}") # Include original JSON error
        except Exception as pe:
            import traceback
            print(f"Error saving full results as pickle ({pickle_filename}): {pe}")
            print("Pickling error traceback:")
            traceback.print_exc() # Print detailed traceback for the pickle error

    # --- Print Summary Table ---
    metrics_to_display = ["accuracy", "auroc", "fpr_at_tpr90", "unknown_detection_rate", "f1_score"]
    metric_names_display = ["Acc(Known)", "AUROC", "FPR@TPR90", "UnkDetect", "F1(Known)"]
    methods_evaluated = [m for m in all_results if isinstance(all_results.get(m), dict) and 'error' not in all_results[m]]
    if not methods_evaluated: print("\nNo successful evaluation results to display."); return all_results

    print("\n" + "="*110); print(f"{' ' * 43}Experiment Results Summary"); print("="*110) # Adjusted width
    header = "{:<20}".format("Metric")
    for method in methods_evaluated: header += "{:<18}".format(method.upper())
    print(header); print("-"*len(header))
    for i, metric_key in enumerate(metrics_to_display):
        row = "{:<20}".format(metric_names_display[i])
        if args.parameter_search and metric_key == args.tuning_metric: row = "* " + row.strip(); row = "{:<20}".format(row)
        for method in methods_evaluated:
            val = all_results[method].get(metric_key, "N/A")
            try: formatted_val = "{:<18.4f}".format(float(val)) if pd.notna(val) else "{:<18}".format("NaN")
            except (TypeError, ValueError): formatted_val = "{:<18}".format(str(val))
            row += formatted_val
        print(row)
    if args.parameter_search: print("\n* Metric used for hyperparameter tuning.")
    print("="*len(header))

    if len(methods_evaluated) > 1: visualize_oscr_curves(all_results, datamodule, args)
    print("\nEvaluation finished!"); return all_results

# =============================================================================
# Argument Parser and Main Execution Block
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Open-Set Recognition Experiments with RoBERTa')
    parser.add_argument('-dataset', type=str, default='acm', choices=['newsgroup20', 'bbc_news', 'trec', 'reuters8', 'acm', 'chemprot', 'banking77', 'oos', 'stackoverflow', 'atis', 'snips', 'financial_phrasebank', 'arxiv10', 'custom_syslog'], help='Dataset to use.')
    parser.add_argument('-model', type=str, default='roberta-base', help='Pre-trained RoBERTa model name.')
    parser.add_argument('-osr_method', type=str, default='all', choices=['threshold', 'openmax', 'crosr', 'doc', 'adb', 'all'], help='OSR method(s) to evaluate.')
    parser.add_argument('-seen_class_ratio', type=float, default=0.5, help='Ratio of classes used as known/seen (0.0 to 1.0).')
    parser.add_argument('-random_seed', type=int, default=42, help='Random seed.')
    parser.add_argument('-epochs', type=int, default=10, help='Training epochs.')
    parser.add_argument('-batch_size', type=int, default=32, help='Batch size.') # Default changed
    parser.add_argument('-lr', type=float, default=2e-5, help='Backbone/standard LR.')
    parser.add_argument('-lr_adb', type=float, default=5e-4, help='ADB centers/radii LR.')
    parser.add_argument('-weight_decay', type=float, default=0.01, help='Weight decay.')
    parser.add_argument('-warmup_ratio', type=float, default=0.1, help='LR warmup ratio.')
    parser.add_argument('-max_warmup_steps', type=int, default=500, help='Max warmup steps.')
    parser.add_argument('-gradient_clip_val', type=float, default=1.0, help='Gradient clipping.')
    parser.add_argument('-early_stopping_patience', type=int, default=3, help='Early stopping patience.')
    parser.add_argument('-early_stopping_delta', type=float, default=0.001, help='Early stopping delta.')
    parser.add_argument('-train_ratio', type=float, default=0.7, help='Train split.')
    parser.add_argument('-val_ratio', type=float, default=0.15, help='Validation split.')
    parser.add_argument('-test_ratio', type=float, default=0.15, help='Test split.')
    parser.add_argument('-force_gpu', action='store_true', help='Force GPU usage.')
    parser.add_argument('-gpu_id', type=int, default=0, help='GPU ID.')
    parser.add_argument('-param_threshold', type=float, default=None, help='ThresholdOSR threshold.')
    parser.add_argument('-param_openmax_tailsize', type=int, default=None, help='OpenMax tail size.')
    parser.add_argument('-param_openmax_alpha', type=int, default=None, help='OpenMax alpha.')
    parser.add_argument('-param_crosr_reconstruction_threshold', type=float, default=None, help='CROSR CDF threshold.')
    parser.add_argument('-param_crosr_tailsize', type=int, default=None, help='CROSR EVT tail size.')
    parser.add_argument('-param_crosr_recon_weight', type=float, default=0.5, help='CROSR recon loss weight.')
    parser.add_argument('-param_doc_k', type=float, default=None, help='DOC k-sigma.')
    parser.add_argument('-param_adb_distance', type=str, default='cosine', choices=['cosine', 'euclidean'], help='ADB distance metric.')
    parser.add_argument('-param_adb_delta', type=float, default=0.1, help='ADB margin delta.')
    parser.add_argument('-param_adb_alpha', type=float, default=0.5, help='ADB loss alpha.')
    parser.add_argument('--adb_freeze_backbone', action=argparse.BooleanOptionalAction, default=True, help='Freeze backbone for ADB.')
    parser.add_argument('-parameter_search', action='store_true', help='Enable Optuna tuning.')
    parser.add_argument('-tuning_metric', type=str, default='f1_score', choices=['accuracy', 'auroc', 'f1_score', 'unknown_detection_rate', 'fpr_at_tpr90'], help='Metric to optimize.') # Added fpr_at_tpr90
    parser.add_argument('-n_trials', type=int, default=20, help='Number of Optuna trials.')
    return parser.parse_args()

def check_gpu():
    print("\n----- GPU Diagnostics -----")
    if torch.cuda.is_available(): print(f"CUDA Available: Yes, Devices: {torch.cuda.device_count()}"); [print(f"  GPU {i}: {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]; print(f"Current Device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
    else: print("CUDA Available: No")
    print("-------------------------\n")

def main():
    args = parse_args(); print("\n----- Args -----"); print(json.dumps(vars(args), indent=2)); print("----------------\n")
    check_gpu(); print(f"Setting random seed: {args.random_seed}"); pl.seed_everything(args.random_seed, workers=True); tokenizer = None # Initialize tokenizer
    print(f"Loading tokenizer: {args.model}...")
    try:
        tokenizer = RobertaTokenizer.from_pretrained(args.model)
    except OSError as e:
        print(f"Error loading tokenizer '{args.model}': Model not found or network issue. {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error loading tokenizer '{args.model}': Invalid configuration or model name format. {e}")
        sys.exit(1)
    except Exception as e: print(f"Error loading tokenizer '{args.model}': {e}"); sys.exit(1)

    print(f"Preparing DataModule: {args.dataset}..."); datamodule = DataModule( dataset_name=args.dataset, tokenizer=tokenizer, batch_size=args.batch_size, seen_class_ratio=args.seen_class_ratio, random_seed=args.random_seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio, max_length=384, data_dir="data" )
    datamodule.prepare_data(); datamodule.setup(stage=None); num_model_classes = datamodule.num_seen_classes
    if num_model_classes is None or num_model_classes <= 0: raise ValueError(f"Invalid num_seen_classes ({num_model_classes}).")
    print(f"Model training targets {num_model_classes} known classes.")

    print("\nStep 1: Training Standard Base Model (RobertaClassifier)...")
    initial_model_class = RobertaClassifier
    init_kwargs = { 'model_name': args.model, 'num_classes': num_model_classes, 'learning_rate': args.lr, 'weight_decay': args.weight_decay, 'warmup_steps': args.max_warmup_steps, 'total_steps': 0 }
    initial_model = initial_model_class(**init_kwargs)
    train_args = copy.deepcopy(args); train_args.osr_method = "initial_standard"
    initial_checkpoint_path = train_model(initial_model, datamodule, train_args)
    if not initial_checkpoint_path or not os.path.exists(initial_checkpoint_path): print("Error: Initial training failed."); sys.exit(1)

    print(f"\nStep 2: Loading initially trained model: {initial_checkpoint_path}")
    try: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); loaded_standard_model = initial_model_class.load_from_checkpoint(initial_checkpoint_path, map_location=device); print("Model loaded.")
    except Exception as e: print(f"Error loading model: {e}"); sys.exit(1)

    print("\nStep 3: Evaluating OSR algorithm(s)...")
    evaluate_osr_main(loaded_standard_model, datamodule, args)
    print("\nExperiment finished.")

if __name__ == "__main__":
    main()
