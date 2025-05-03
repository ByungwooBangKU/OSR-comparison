
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
import copy # For deep copying args
import pickle # For fallback result saving
import pathlib # For path handling in JSON saving

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from sklearn.manifold import TSNE # t-SNE import

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
warnings.filterwarnings("ignore", ".*The dataloader.*does not have many workers.*") # Added more specific ignore

# --- Constants ---
RESULTS_DIR = "results"
LOGS_DIR = "logs"
CHECKPOINTS_DIR = "checkpoints"
DATA_DIR = "data"
DEFAULT_FALLBACK_TOTAL_STEPS = 10000
DEFAULT_MAX_TSNE_SAMPLES = 5000
INITIAL_ADB_DELTA_PRIME = -1.5 # Initial value aiming for radius ~0.2
MIN_SAMPLES_FOR_STRATIFY = 2
NUM_DATALOADER_WORKERS = min(4, os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 0)

# --- Matplotlib Korean Font Setup ---
def setup_korean_font():
    """Attempts to set up a Korean font for Matplotlib."""
    font_name = None
    try:
        possible_fonts = ['Malgun Gothic', 'NanumGothic', 'Apple SD Gothic Neo', 'Noto Sans KR']
        # Use findSystemFonts to be more robust
        available_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        available_font_names = [fm.FontProperties(fname=font_path).get_name() for font_path in available_fonts]

        for font in possible_fonts:
            if font in available_font_names:
                font_name = font
                break
        if font_name:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
            print(f"Korean font '{font_name}' set for Matplotlib.")
        else:
            print("Warning: No common Korean font found (Malgun Gothic, NanumGothic, etc.). Install one for proper Korean text in plots.")
            plt.rcParams['axes.unicode_minus'] = False # Still disable minus sign issue
    except Exception as e:
        print(f"Error setting up Korean font: {e}")
        plt.rcParams['axes.unicode_minus'] = False # Fallback

setup_korean_font()

# =============================================================================
# Data Processing (TextDataset, DataModule)
# =============================================================================
class TextDataset(Dataset):
    """PyTorch Dataset for text classification."""
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int = 384):
        self.texts = texts
        self.labels = labels # Should be numpy array for efficient indexing
        self.tokenizer = tokenizer
        self.max_length = max_length
        if len(texts) != len(labels):
            raise ValueError(f"Texts and labels length mismatch: {len(texts)} != {len(labels)}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Ensure text is string, handle potential None values gracefully
        text = str(self.texts[idx]) if self.texts[idx] is not None else ""
        label = self.labels[idx] # Get label directly from numpy array

        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False, # RoBERTa doesn't use token_type_ids
            return_tensors='pt'
        )

        # Return dictionary matching model input names
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long) # Ensure label is torch tensor
        }
        return item

class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for handling datasets and OSR splits."""
    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        batch_size: int = 64,
        seen_class_ratio: float = 0.5,
        random_seed: int = 42,
        max_length: int = 384,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        data_dir: str = DATA_DIR,
        num_workers: int = NUM_DATALOADER_WORKERS
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seen_class_ratio = seen_class_ratio
        self.random_seed = random_seed
        self.max_length = max_length
        self.data_dir = data_dir
        self.num_workers = num_workers

        # Placeholders, will be populated in setup()
        self.num_classes = None
        self.num_seen_classes = None
        self.seen_classes = None # Indices in the mapped space (0 to num_seen_classes-1)
        self.unseen_classes = None # Indices in the original space
        self.original_seen_indices = None # Indices in the original space
        self.original_unseen_indices = None # Indices in the original space
        self.seen_class_mapping = None # Maps original seen index -> new mapped index (0..N-1)
        self.class_names = None # List of original class names

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Normalize ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            print(f"Warning: Data split ratios sum to {total:.3f}. Normalizing...")
            self.train_ratio = train_ratio / total
            self.val_ratio = val_ratio / total
            self.test_ratio = test_ratio / total
            print(f"Normalized ratios: train={self.train_ratio:.3f}, val={self.val_ratio:.3f}, test={self.test_ratio:.3f}")
        else:
            self.train_ratio = train_ratio
            self.val_ratio = val_ratio
            self.test_ratio = test_ratio

        # Map dataset names to their preparation functions
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
        """Downloads or prepares data if necessary. Runs only once."""
        print(f"Preparing data for dataset: {self.dataset_name}...")
        prepare_func = self.prepare_func_map.get(self.dataset_name)
        if prepare_func:
            try:
                # Call the prepare function - it should handle download/extraction
                _ = prepare_func(data_dir=self.data_dir)
                print(f"{self.dataset_name} data preparation check complete.")
            except FileNotFoundError as e:
                 # Specific user instruction for custom dataset
                 if self.dataset_name == 'custom_syslog':
                      print(f"\n{'='*20} ACTION REQUIRED {'='*20}\n{e}\nPlease ensure your custom syslog data is placed correctly according to the dataset_utils documentation.\n{'='*78}\n")
                 else:
                     print(f"Error during prepare_data for {self.dataset_name}: {e}")
                 sys.exit(1) # Exit if data is missing (except maybe custom)
            except Exception as e:
                print(f"Unexpected error during prepare_data for {self.dataset_name}: {e}")
                import traceback; traceback.print_exc()
                raise # Re-raise the exception
        else:
            print(f"Warning: No specific prepare_data action defined for dataset '{self.dataset_name}'. Assuming data exists.")

    def setup(self, stage: str | None = None):
        """Loads data, performs splits, and creates datasets. Runs on each process in DDP."""
        # Prevent redundant setup
        if stage == 'fit' and self.train_dataset is not None: return
        if stage == 'test' and self.test_dataset is not None: return
        if stage == 'predict' and self.test_dataset is not None: return # Handle predict stage too

        pl.seed_everything(self.random_seed) # Ensure reproducibility across processes

        print(f"\n--- Setting up DataModule for dataset: {self.dataset_name} (Seen Ratio: {self.seen_class_ratio}) ---")

        # Load data using the appropriate function
        load_func = self.prepare_func_map.get(self.dataset_name)
        if not load_func:
            raise ValueError(f"Unknown or unprepared dataset: {self.dataset_name}")

        print(f"Loading data using {load_func.__name__}...")
        try:
            texts, labels, self.class_names = load_func(data_dir=self.data_dir)
        except Exception as e:
            raise ValueError(f"Data loading failed for {self.dataset_name}") from e

        if not texts: raise ValueError(f"Failed to load any text data for dataset '{self.dataset_name}'. Check data source.")
        if not isinstance(labels, (list, np.ndarray)) or len(labels) == 0: raise ValueError(f"Loaded labels for {self.dataset_name} are invalid or empty.")
        if len(texts) != len(labels): raise ValueError(f"Mismatch between texts ({len(texts)}) and labels ({len(labels)}) for {self.dataset_name}.")

        # Convert to numpy arrays for easier processing
        texts_np = np.array(texts, dtype=object) # Use object dtype for strings
        labels_np = np.array(labels, dtype=int)

        # --- Data Splitting (Train/Val/Test) ---
        print("Splitting data into train/validation/test sets...")
        X_train_val, X_test, y_train_val, y_test = self._split_data(texts_np, labels_np, test_size=self.test_ratio)

        # Calculate relative validation size for the second split
        if abs(self.train_ratio + self.val_ratio) < 1e-6: # Handle edge case: no training/validation data needed
            val_size_relative = 0.0
            X_train, X_val, y_train, y_val = X_train_val, np.array([], dtype=object), y_train_val, np.array([], dtype=int)
        else:
            val_size_relative = self.val_ratio / (self.train_ratio + self.val_ratio)
            X_train, X_val, y_train, y_val = self._split_data(X_train_val, y_train_val, test_size=val_size_relative)

        # --- Class Splitting (Seen/Unseen) ---
        all_original_indices = np.unique(labels_np)
        self.num_classes = len(all_original_indices)
        print(f"Total original classes found: {self.num_classes} -> {all_original_indices.tolist()}")

        # Validate or generate class names
        if self.class_names is None:
            print(f"Warning: class_names not provided by loader. Generating generic names.")
            self.class_names = [f"Class_{i}" for i in all_original_indices]
        elif len(self.class_names) != self.num_classes:
             print(f"Warning: Mismatch between provided class names ({len(self.class_names)}) and unique labels ({self.num_classes}). Adjusting names.")
             # Ensure class_names list matches the indices found
             self.class_names = [self.class_names[i] if i < len(self.class_names) else f"Class_{i}" for i in all_original_indices]

        # Determine seen/unseen classes based on ratio
        if self.seen_class_ratio < 1.0:
            print(f"Splitting classes: {self.seen_class_ratio*100:.1f}% Seen / {(1-self.seen_class_ratio)*100:.1f}% Unseen")
            num_seen = max(1, int(np.round(self.num_classes * self.seen_class_ratio)))
            if num_seen >= self.num_classes:
                print("Warning: num_seen >= total classes. Adjusting seen_class_ratio to 1.0.")
                self.seen_class_ratio = 1.0 # Treat as all known

        # Perform the class split if ratio < 1.0
        if self.seen_class_ratio < 1.0:
            np.random.seed(self.random_seed) # Ensure consistent class split
            all_classes_shuffled = np.random.permutation(all_original_indices)
            self.original_seen_indices = np.sort(all_classes_shuffled[:num_seen])
            self.original_unseen_indices = np.sort(all_classes_shuffled[num_seen:])
            print(f"  Original Seen Indices: {self.original_seen_indices.tolist()}")
            print(f"  Original Unseen Indices: {self.original_unseen_indices.tolist()}")

            # Create mapping from original seen index to new 0..N-1 index
            self.seen_class_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(self.original_seen_indices)}
            self.num_seen_classes = len(self.original_seen_indices)
            self.seen_classes = np.arange(self.num_seen_classes) # Mapped indices 0..N-1

            # Filter train/val sets to contain only seen classes and map labels
            X_train, y_train_final = self._filter_and_map_labels(X_train, y_train, self.original_seen_indices, self.seen_class_mapping)
            X_val, y_val_final = self._filter_and_map_labels(X_val, y_val, self.original_seen_indices, self.seen_class_mapping)

            # Mark unseen classes in the test set with -1
            y_test_final = y_test.copy()
            unseen_test_mask = np.isin(y_test_final, self.original_unseen_indices)
            y_test_final[unseen_test_mask] = -1 # Mark unknowns
            # Keep original labels for seen classes in test set (will be mapped during evaluation if needed)

        else: # All classes are seen
            print("All classes are Known (seen_class_ratio = 1.0)")
            self.original_seen_indices = all_original_indices.copy()
            self.original_unseen_indices = np.array([], dtype=int)
            self.num_seen_classes = self.num_classes
            self.seen_classes = all_original_indices.copy() # Original indices are used directly
            # Mapping is identity (original index -> same original index)
            self.seen_class_mapping = {orig_idx: orig_idx for orig_idx in all_original_indices}
            # Labels remain as original indices
            y_train_final = y_train
            y_val_final = y_val
            y_test_final = y_test

        # --- Create PyTorch Datasets ---
        self.train_dataset = TextDataset(list(X_train), y_train_final, self.tokenizer, self.max_length)
        self.val_dataset = TextDataset(list(X_val), y_val_final, self.tokenizer, self.max_length)
        self.test_dataset = TextDataset(list(X_test), y_test_final, self.tokenizer, self.max_length)

        print(f"\nDataset sizes: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")
        print(f"Number of known classes for model training: {self.num_seen_classes}")
        if self.seen_class_ratio < 1.0:
            unknown_in_test = np.sum(y_test_final == -1)
            print(f"Unknown samples (-1) in test set: {unknown_in_test}")
        if self.class_names and self.original_seen_indices is not None:
             try:
                 # Get names corresponding to the *original* seen indices
                 seen_names_list = [self.class_names[i] for i in self.original_seen_indices]
                 print(f"Known class names (original): {seen_names_list}")
             except IndexError:
                 print(f"Warning: Could not map all original_seen_indices to class_names (check consistency).")
        print("--- Finished DataModule setup ---")

    def _split_data(self, X, y, test_size):
        """Helper function to split data with stratification handling."""
        if test_size <= 0.0 or test_size >= 1.0: # Handle edge cases
             if test_size <= 0.0: return X, np.array([], dtype=X.dtype), y, np.array([], dtype=y.dtype)
             else: return np.array([], dtype=X.dtype), X, np.array([], dtype=y.dtype), y

        unique_labels, counts = np.unique(y, return_counts=True)
        # Stratify only if all classes have at least MIN_SAMPLES_FOR_STRATIFY samples
        can_stratify = not np.any(counts < MIN_SAMPLES_FOR_STRATIFY)
        stratify_param = y if can_stratify else None

        if not can_stratify and len(unique_labels) > 1:
            print(f"Warning: Stratification disabled for split due to classes with < {MIN_SAMPLES_FOR_STRATIFY} samples.")

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_seed, stratify=stratify_param
            )
        except ValueError as e:
             # This might happen if stratification fails even with the check (e.g., very small dataset)
             print(f"Warning: Stratified split failed ('{e}'). Retrying without stratification.")
             X_train, X_test, y_train, y_test = train_test_split(
                 X, y, test_size=test_size, random_state=self.random_seed, stratify=None
             )
        return X_train, X_test, y_train, y_test

    def _filter_and_map_labels(self, X, y, original_seen_indices, seen_class_mapping):
        """Filters data to keep only seen classes and maps their labels."""
        if len(y) == 0: # Handle empty input
            return X, y
        seen_mask = np.isin(y, original_seen_indices)
        X_filtered = X[seen_mask]
        y_original_kept = y[seen_mask]
        # Map the original labels (which are in original_seen_indices) to the new 0..N-1 space
        y_mapped = np.array([seen_class_mapping[lbl] for lbl in y_original_kept], dtype=int)
        return X_filtered, y_mapped

    def _create_dataloader(self, dataset: TextDataset | None, shuffle: bool) -> DataLoader | None:
        """Helper to create a DataLoader."""
        if dataset is None or len(dataset) == 0:
            # Return None or an empty DataLoader if needed, but None is simpler
            return None
        # Determine if persistent workers can be used
        persistent = self.num_workers > 0
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=persistent if persistent else False, # Only set True if num_workers > 0
            pin_memory=True # Generally good practice with GPUs
        )

    def train_dataloader(self) -> DataLoader | None:
        if self.train_dataset is None: raise ValueError("Train dataset not initialized. Call setup() first.")
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader | None:
        if self.val_dataset is None: raise ValueError("Validation dataset not initialized. Call setup() first.")
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader | None:
        if self.test_dataset is None: raise ValueError("Test dataset not initialized. Call setup() first.")
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader | None:
        """Provide a dataloader for prediction, typically the test set."""
        if self.test_dataset is None: raise ValueError("Test dataset not initialized. Call setup() first.")
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def _determine_unknown_labels(self, labels_np: np.ndarray) -> np.ndarray:
        """Consistently identifies unknown labels marked as -1."""
        return (np.array(labels_np) == -1)

# =============================================================================
# Base Lightning Module (for common logic)
# =============================================================================
class BaseRobertaModule(pl.LightningModule):
    """Base class for RoBERTa-based Lightning Modules to share optimizer config."""
    def __init__(self, model_name: str, num_classes: int, learning_rate: float,
                 weight_decay: float, warmup_steps: int, total_steps: int):
        super().__init__()
        # Use save_hyperparameters() to automatically save arguments to self.hparams
        # Ignore model_name as it's not a numerical hyperparameter for logging/checkpointing
        self.save_hyperparameters(ignore=['model_name'])

        self.model_name = model_name # Store separately if needed
        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)

        # Store scheduler parameters explicitly if needed by configure_optimizers logic
        # Note: They are also available via self.hparams.warmup_steps etc.
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        # Ensure num_classes is accessible for OSRAlgorithm
        self.num_classes = self.hparams.num_classes

    def _calculate_total_steps(self) -> int:
        """Estimates total training steps if not provided."""
        if hasattr(self, 'total_steps') and self.total_steps > 0:
            return self.total_steps

        print("Warning: total_steps not explicitly set. Estimating from trainer...")
        try:
            if self.trainer and hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches > 0:
                 estimated_steps = self.trainer.estimated_stepping_batches
                 print(f"Using estimated total_steps from trainer: {estimated_steps}")
                 # Store it back for potential reuse if needed, though hparams should be primary
                 self.total_steps = estimated_steps
                 return estimated_steps
            else:
                 print(f"Warning: Could not estimate total_steps from trainer. Using fallback: {DEFAULT_FALLBACK_TOTAL_STEPS}")
                 self.total_steps = DEFAULT_FALLBACK_TOTAL_STEPS
                 return DEFAULT_FALLBACK_TOTAL_STEPS
        except Exception as e:
            print(f"Warning: Exception during total_steps estimation ({e}). Using fallback: {DEFAULT_FALLBACK_TOTAL_STEPS}")
            self.total_steps = DEFAULT_FALLBACK_TOTAL_STEPS
            return DEFAULT_FALLBACK_TOTAL_STEPS

    def _get_scheduler(self, optimizer):
        """Creates the learning rate scheduler."""
        effective_total_steps = self._calculate_total_steps()

        # Ensure warmup_steps is valid
        if not hasattr(self, 'warmup_steps') or self.warmup_steps < 0:
             print("Warning: warmup_steps not set or invalid. Using 0.")
             self.warmup_steps = 0

        # Clamp warmup steps to not exceed total steps
        actual_warmup_steps = min(self.warmup_steps, effective_total_steps) if effective_total_steps > 0 else 0
        if actual_warmup_steps != self.warmup_steps:
            print(f"Warning: Adjusted warmup_steps from {self.warmup_steps} to {actual_warmup_steps} (total steps: {effective_total_steps})")

        if effective_total_steps > 0:
            print(f"Scheduler: Using linear warmup for {actual_warmup_steps} steps and decay over {effective_total_steps} steps.")
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=actual_warmup_steps,
                num_training_steps=effective_total_steps
            )
            return {'scheduler': scheduler, 'interval': 'step'}
        else:
            print("Warning: total_steps is zero or negative. No LR scheduler will be used.")
            return None

    def _get_mapped_labels_for_loss(self, original_labels: torch.Tensor) -> torch.Tensor | None:
        """Maps original known labels (0..Total-1) to model output space (0..num_seen-1)."""
        known_mask = original_labels >= 0
        if not known_mask.any():
            return None # No known labels in this batch

        original_indices_known_cpu = original_labels[known_mask].cpu().numpy()
        try:
            # Use the mapping stored in the datamodule
            datamodule = self.trainer.datamodule
            if not hasattr(datamodule, 'seen_class_mapping') or datamodule.seen_class_mapping is None:
                print("Error: seen_class_mapping not found in datamodule during label mapping.")
                return None

            mapped_labels = [datamodule.seen_class_mapping[idx] for idx in original_indices_known_cpu]
            return torch.tensor(mapped_labels, device=self.device, dtype=torch.long)

        except KeyError as e:
            print(f"Error: Test label {e} not found in seen_class_mapping during label mapping.")
            # Decide how to handle: return None, raise error, etc.
            return None
        except Exception as e:
            print(f"Unexpected error during label mapping: {e}")
            return None

    # Default configure_optimizers, subclasses can override if needed (like ADB)
    def configure_optimizers(self):
        """Configures the optimizer and scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler_config = self._get_scheduler(optimizer)

        if scheduler_config:
            return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
        else:
            return optimizer

# =============================================================================
# Model Definitions
# =============================================================================

class RobertaClassifier(BaseRobertaModule):
    """Standard RoBERTa model with a classification head."""
    def __init__(
        self,
        model_name: str = "roberta-base",
        num_classes: int = 20, # Number of *known* classes for the classifier
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        total_steps: int = 0
    ):
        # Call BaseRobertaModule's init
        super().__init__(model_name, num_classes, learning_rate, weight_decay, warmup_steps, total_steps)
        # Define the classification head specific to this model
        self.classifier = nn.Linear(self.config.hidden_size, self.hparams.num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Forward pass returning logits and CLS token embedding."""
        # RoBERTa forward pass (token_type_ids is usually None for RoBERTa)
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids # Pass if provided, otherwise None
        )
        # Extract the [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Pass CLS embedding through the classifier
        logits = self.classifier(cls_output)
        return logits, cls_output

    def training_step(self, batch, batch_idx):
        """Single training step."""
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        labels = batch['label'] # Labels are already mapped to 0..N-1 for known classes

        # Forward pass
        logits, _ = self(input_ids, attention_mask)

        # Filter out any potential invalid labels (shouldn't happen in train/val)
        valid_mask = labels >= 0
        if not valid_mask.any(): return None # Skip batch if no valid labels

        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]

        # Calculate cross-entropy loss
        loss = F.cross_entropy(valid_logits, valid_labels)

        # Log training loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, batch_size=batch['input_ids'].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        """Single validation step."""
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        labels = batch['label'] # Labels are already mapped to 0..N-1

        # Forward pass
        logits, _ = self(input_ids, attention_mask)

        # Filter out potential invalid labels
        valid_mask = labels >= 0
        if not valid_mask.any():
            # Log zero loss/acc if no valid samples, or return None
            self.log_dict({'val_loss': 0.0, 'val_acc': 0.0}, prog_bar=True, on_epoch=True, logger=True, batch_size=batch['input_ids'].size(0))
            return None

        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]

        # Calculate loss and accuracy
        loss = F.cross_entropy(valid_logits, valid_labels)
        preds = torch.argmax(valid_logits, dim=1)
        acc = accuracy_score(valid_labels.cpu().numpy(), preds.cpu().numpy())

        # Log validation metrics
        self.log_dict({'val_loss': loss, 'val_acc': acc}, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=batch['input_ids'].size(0))
        return {'val_loss': loss, 'val_acc': acc} # Return dict for potential callbacks

    def test_step(self, batch, batch_idx):
        """Single test step. Handles original labels including -1 for unknowns."""
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        labels_original = batch['label'] # Original labels (-1 for unknown, original index for known)

        # Forward pass
        logits, embeddings = self(input_ids, attention_mask)
        preds_mapped = torch.argmax(logits, dim=1) # Predictions indices (0..num_seen-1)

        # Calculate loss & acc only on known samples (where label >= 0)
        known_mask = labels_original >= 0
        test_loss = torch.tensor(0.0, device=self.device)
        test_acc_known = torch.tensor(0.0, device=self.device)

        if known_mask.any():
             # Map the *true* original known labels to the model's output space (0..num_seen-1)
             mapped_true_labels_known = self._get_mapped_labels_for_loss(labels_original)

             if mapped_true_labels_known is not None:
                 # Calculate loss using mapped true labels and logits for known samples
                 test_loss = F.cross_entropy(logits[known_mask], mapped_true_labels_known)
                 # Calculate accuracy comparing mapped predictions and mapped true labels for known samples
                 test_acc_known = accuracy_score(
                     mapped_true_labels_known.cpu().numpy(),
                     preds_mapped[known_mask].cpu().numpy()
                 )
                 self.log('test_loss', test_loss, prog_bar=False, logger=True, batch_size=batch['input_ids'].size(0))
                 self.log('test_acc_known', test_acc_known, prog_bar=False, logger=True, batch_size=batch['input_ids'].size(0))

        # Return dictionary containing predictions, labels, logits, and embeddings for OSR evaluation
        return {
            'preds_mapped': preds_mapped,           # Predictions indices (0..num_seen-1) for all samples
            'labels_original': labels_original,     # Original labels (-1 or original index 0..Total-1)
            'logits': logits,                       # Raw logits (size num_seen_classes)
            'embeddings': embeddings                # CLS embeddings
        }

    # configure_optimizers is inherited from BaseRobertaModule


class RobertaAutoencoder(BaseRobertaModule):
    """RoBERTa-based Autoencoder for CROSR."""
    def __init__(
        self,
        model_name: str = "roberta-base",
        num_classes: int = 20,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        total_steps: int = 0,
        latent_dim: int = 256,
        reconstruction_weight: float = 0.5 # Added hparam saving
    ):
        super().__init__(model_name, num_classes, learning_rate, weight_decay, warmup_steps, total_steps)
        # Save AE specific hyperparameters
        self.save_hyperparameters('latent_dim', 'reconstruction_weight')

        # Classification head
        self.classifier = nn.Linear(self.config.hidden_size, self.hparams.num_classes)
        # Encoder/Decoder Layers
        self.encoder = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.hparams.latent_dim),
            nn.ReLU() # Or another activation like Tanh
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.latent_dim, self.config.hidden_size)
            # Potentially add activation if needed, depends on reconstruction target
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Forward pass for classification and reconstruction."""
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :] # [CLS] embedding
        logits = self.classifier(cls_output) # Classification logits
        encoded = self.encoder(cls_output) # Encode CLS embedding
        reconstructed = self.decoder(encoded) # Decode back to original embedding dim
        return logits, cls_output, encoded, reconstructed

    def training_step(self, batch, batch_idx):
        """Single training step with combined loss."""
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        labels = batch['label']

        # Forward pass
        logits, cls_output, _, reconstructed = self(input_ids, attention_mask)

        # Filter valid samples
        valid_mask = labels >= 0
        if not valid_mask.any(): return None

        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]
        valid_cls_output = cls_output[valid_mask]
        valid_reconstructed = reconstructed[valid_mask]

        # Calculate losses
        class_loss = F.cross_entropy(valid_logits, valid_labels)
        recon_loss = F.mse_loss(valid_reconstructed, valid_cls_output) # MSE between original and reconstructed CLS

        # Combine losses using the weight hyperparameter
        loss = class_loss + self.hparams.reconstruction_weight * recon_loss

        # Log metrics
        self.log_dict(
            {'train_loss': loss, 'train_class_loss': class_loss, 'train_recon_loss': recon_loss},
            prog_bar=False, on_step=True, on_epoch=True, logger=True, batch_size=batch['input_ids'].size(0)
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Single validation step."""
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        labels = batch['label']

        # Forward pass
        logits, cls_output, _, reconstructed = self(input_ids, attention_mask)

        # Filter valid samples
        valid_mask = labels >= 0
        if not valid_mask.any():
            self.log_dict({'val_loss': 0.0, 'val_acc': 0.0, 'val_class_loss': 0.0, 'val_recon_loss': 0.0},
                          prog_bar=True, on_epoch=True, logger=True, batch_size=batch['input_ids'].size(0))
            return None

        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]
        valid_cls_output = cls_output[valid_mask]
        valid_reconstructed = reconstructed[valid_mask]

        # Calculate losses and accuracy
        class_loss = F.cross_entropy(valid_logits, valid_labels)
        recon_loss = F.mse_loss(valid_reconstructed, valid_cls_output)
        loss = class_loss + self.hparams.reconstruction_weight * recon_loss
        preds = torch.argmax(valid_logits, dim=1)
        acc = accuracy_score(valid_labels.cpu().numpy(), preds.cpu().numpy())

        # Log metrics
        self.log_dict(
            {'val_loss': loss, 'val_class_loss': class_loss, 'val_recon_loss': recon_loss, 'val_acc': acc},
            prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=batch['input_ids'].size(0)
        )
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        """Single test step, returning reconstruction errors."""
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        labels_original = batch['label']

        # Forward pass
        logits, cls_output, encoded, reconstructed = self(input_ids, attention_mask)
        preds_mapped = torch.argmax(logits, dim=1) # Predictions (0..num_seen-1)

        # Calculate reconstruction error (L2 norm) for all samples
        recon_errors = torch.norm(reconstructed - cls_output, p=2, dim=1)

        # Calculate loss/acc on knowns only
        known_mask = labels_original >= 0
        test_loss = torch.tensor(0.0, device=self.device)
        test_acc_known = torch.tensor(0.0, device=self.device)

        if known_mask.any():
            mapped_true_labels_known = self._get_mapped_labels_for_loss(labels_original)
            if mapped_true_labels_known is not None:
                valid_logits = logits[known_mask]
                valid_cls_output = cls_output[known_mask]
                valid_reconstructed = reconstructed[known_mask]

                class_loss = F.cross_entropy(valid_logits, mapped_true_labels_known)
                recon_loss = F.mse_loss(valid_reconstructed, valid_cls_output)
                test_loss = class_loss + self.hparams.reconstruction_weight * recon_loss
                test_acc_known = accuracy_score(
                    mapped_true_labels_known.cpu().numpy(),
                    preds_mapped[known_mask].cpu().numpy()
                )
                self.log('test_loss', test_loss, prog_bar=False, logger=True, batch_size=batch['input_ids'].size(0))
                self.log('test_acc_known', test_acc_known, prog_bar=False, logger=True, batch_size=batch['input_ids'].size(0))

        # Return values needed for CROSR evaluation
        return {
            'preds_mapped': preds_mapped,
            'labels_original': labels_original,
            'logits': logits,
            'embeddings': cls_output, # Original CLS embeddings
            'encoded': encoded, # Latent space representation
            'reconstructed': reconstructed, # Reconstructed embeddings
            'recon_errors': recon_errors # Crucial for CROSR OSR scoring
        }

    # configure_optimizers is inherited from BaseRobertaModule


class DOCRobertaClassifier(BaseRobertaModule):
    """RoBERTa model adapted for DOC (one-vs-rest binary classifiers)."""
    def __init__(
        self,
        model_name: str = "roberta-base",
        num_classes: int = 20,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        total_steps: int = 0
    ):
        super().__init__(model_name, num_classes, learning_rate, weight_decay, warmup_steps, total_steps)
        # Create a separate binary classifier head for each known class
        self.classifiers = nn.ModuleList([
            nn.Linear(self.config.hidden_size, 1) for _ in range(self.hparams.num_classes)
        ])

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Forward pass applying all binary classifiers."""
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Apply each binary classifier to the CLS embedding
        # Concatenate the single logit output from each classifier
        logits = torch.cat([clf(cls_output) for clf in self.classifiers], dim=1) # Shape: (batch_size, num_classes)
        return logits, cls_output

    def training_step(self, batch, batch_idx):
        """Single training step using binary cross-entropy."""
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        labels = batch['label']

        # Forward pass
        logits, _ = self(input_ids, attention_mask)

        # Filter valid samples
        valid_mask = labels >= 0
        if not valid_mask.any(): return None

        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]

        # Create multi-label binary targets (one-hot encoding) for BCEWithLogitsLoss
        one_hot_labels = F.one_hot(valid_labels, num_classes=self.hparams.num_classes).float()
        loss = F.binary_cross_entropy_with_logits(valid_logits, one_hot_labels) # Use BCEWithLogitsLoss for numerical stability

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, batch_size=batch['input_ids'].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        """Single validation step."""
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        labels = batch['label']

        # Forward pass
        logits, _ = self(input_ids, attention_mask)

        # Filter valid samples
        valid_mask = labels >= 0
        if not valid_mask.any():
            self.log_dict({'val_loss': 0.0, 'val_acc': 0.0}, prog_bar=True, on_epoch=True, logger=True, batch_size=batch['input_ids'].size(0))
            return None

        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]

        # Calculate loss
        one_hot_labels = F.one_hot(valid_labels, num_classes=self.hparams.num_classes).float()
        loss = F.binary_cross_entropy_with_logits(valid_logits, one_hot_labels)

        # Calculate accuracy: predict class with the highest sigmoid score
        sigmoid_scores = torch.sigmoid(valid_logits)
        preds = torch.argmax(sigmoid_scores, dim=1)
        acc = accuracy_score(valid_labels.cpu().numpy(), preds.cpu().numpy())

        self.log_dict({'val_loss': loss, 'val_acc': acc}, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=batch['input_ids'].size(0))
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        """Single test step, returning sigmoid scores."""
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        labels_original = batch['label']

        # Forward pass
        logits, embeddings = self(input_ids, attention_mask)
        sigmoid_scores = torch.sigmoid(logits) # Sigmoid scores for each class (crucial for DOC OSR)
        preds_mapped = torch.argmax(sigmoid_scores, dim=1) # Predicted class index (0..num_seen-1)

        # Calculate loss/acc on knowns only
        known_mask = labels_original >= 0
        test_loss = torch.tensor(0.0, device=self.device)
        test_acc_known = torch.tensor(0.0, device=self.device)

        if known_mask.any():
            mapped_true_labels_known = self._get_mapped_labels_for_loss(labels_original)
            if mapped_true_labels_known is not None:
                valid_logits = logits[known_mask]
                # Create one-hot labels for BCE loss calculation
                one_hot_labels = F.one_hot(mapped_true_labels_known, num_classes=self.hparams.num_classes).float()
                test_loss = F.binary_cross_entropy_with_logits(valid_logits, one_hot_labels)
                test_acc_known = accuracy_score(
                    mapped_true_labels_known.cpu().numpy(),
                    preds_mapped[known_mask].cpu().numpy()
                )
                self.log('test_loss', test_loss, prog_bar=False, logger=True, batch_size=batch['input_ids'].size(0))
                self.log('test_acc_known', test_acc_known, prog_bar=False, logger=True, batch_size=batch['input_ids'].size(0))

        # Return values needed for DOC OSR evaluation
        return {
            'preds_mapped': preds_mapped,           # Predicted class index (0..num_seen-1)
            'labels_original': labels_original,     # Original labels
            'logits': logits,                       # Raw one-vs-rest logits
            'sigmoid_scores': sigmoid_scores,       # Sigmoid scores (crucial for DOC eval)
            'embeddings': embeddings                # CLS embeddings
        }

    # configure_optimizers is inherited from BaseRobertaModule


class RobertaADB(BaseRobertaModule):
    """RoBERTa model with Adaptive Decision Boundary (ADB) components."""
    def __init__(
        self,
        model_name: str = "roberta-base",
        num_classes: int = 20,
        # Learning rates now use hparams naming convention
        learning_rate: float = 2e-5,      # Backbone fine-tuning LR (maps to args.lr)
        lr_adb: float = 5e-4,             # ADB parameter LR (maps to args.lr_adb)
        weight_decay: float = 0.0,        # Note: Original paper might use 0 for ADB params
        warmup_steps: int = 0,            # For backbone scheduler
        total_steps: int = 0,             # For backbone scheduler
        # ADB specific parameters now use hparams naming
        param_adb_delta: float = 0.1,     # Margin delta
        param_adb_alpha: float = 0.5,     # Loss weighting alpha
        adb_freeze_backbone: bool = True  # Whether to freeze RoBERTa
    ):
        # Initialize BaseRobertaModule - note the LR mapping
        super().__init__(model_name, num_classes, learning_rate, weight_decay, warmup_steps, total_steps)
        # Save ADB-specific hyperparameters
        self.save_hyperparameters('lr_adb', 'param_adb_delta', 'param_adb_alpha', 'adb_freeze_backbone')

        self.feat_dim = self.config.hidden_size

        # Learnable class centers (initialized normally)
        self.centers = nn.Parameter(torch.empty(self.hparams.num_classes, self.feat_dim), requires_grad=True)
        nn.init.normal_(self.centers, std=0.05)

        # Learnable Logits for Radii (Delta' in paper, ensures radii are positive via softplus)
        # Initialize delta_prime such that initial radius is small but non-zero
        self.delta_prime = nn.Parameter(torch.full((self.hparams.num_classes,), INITIAL_ADB_DELTA_PRIME), requires_grad=True)

        # Freeze backbone based on hyperparameter AFTER initialization
        if self.hparams.adb_freeze_backbone:
            print("[RobertaADB Init] Freezing RoBERTa backbone parameters.")
            for param in self.roberta.parameters():
                param.requires_grad = False
        else:
            print("[RobertaADB Init] RoBERTa backbone parameters will be fine-tuned.")

    def get_radii(self) -> torch.Tensor:
        """Calculate actual positive radii using Softplus on the learnable delta_prime."""
        return F.softplus(self.delta_prime)

    @staticmethod
    def _cosine_distance(x_norm: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates pairwise cosine distance (1 - similarity).
        Assumes input x_norm is already L2 normalized.
        Normalizes y internally.
        """
        # Ensure centers (y) are normalized for cosine distance calculation
        y_norm = F.normalize(y, p=2, dim=-1)
        # Calculate cosine similarity
        similarity = torch.matmul(x_norm, y_norm.t())
        # Clamp similarity to avoid numerical issues with acos or sqrt
        similarity = torch.clamp(similarity, -1.0 + 1e-7, 1.0 - 1e-7)
        # Distance = 1 - Similarity
        distance = 1.0 - similarity
        return distance

    def adb_margin_loss(self, feat_norm: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculates ADB margin loss: max(0, d(feat, c_y) - r_y + delta).
        feat_norm: L2 normalized features.
        labels: Target class indices (mapped 0..N-1).
        """
        # Calculate distances from features to all centers
        # Assumes self.centers might not be normalized yet, _cosine_distance handles it
        distances = self._cosine_distance(feat_norm, self.centers) # Shape: (batch_size, num_classes)

        # Get distances corresponding to the true labels for each sample
        # gather(dim, index) -> selects values along dim using index
        d_y = distances.gather(1, labels.unsqueeze(1)).squeeze(1) # Shape: (batch_size,)

        # Get positive radii using the helper function
        radii = self.get_radii() # Shape: (num_classes,)
        # Get radii corresponding to the true labels
        r_y = radii[labels] # Shape: (batch_size,)

        # Calculate margin loss per sample: max(0, distance_to_correct_center - radius_of_correct_center + margin_delta)
        # Use delta from hyperparameters
        loss_per_sample = torch.relu(d_y - r_y + self.hparams.param_adb_delta)

        # Average loss over the batch
        loss = loss_per_sample.mean()
        return loss

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Forward pass: get CLS embedding, normalize it, calculate similarity 'logits'."""
        # Control gradient flow for the backbone based on the freeze hyperparameter
        with torch.set_grad_enabled(not self.hparams.adb_freeze_backbone):
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        feat = outputs.last_hidden_state[:, 0, :] # CLS embedding
        # L2 normalize the features - crucial for cosine distance/similarity
        feat_norm = F.normalize(feat, p=2, dim=-1)

        # Calculate 'logits' based on cosine similarity (which is 1 - cosine_distance)
        # Higher similarity = higher logit = more likely to be that class
        logits = 1.0 - self._cosine_distance(feat_norm, self.centers)
        # Note: These logits represent similarity scores, not standard softmax logits.

        return logits, feat_norm # Return similarity logits and normalized features

    def training_step(self, batch, batch_idx):
        """Single training step combining CE loss and ADB margin loss."""
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        labels = batch['label']

        # Filter valid samples
        valid_mask = labels >= 0
        if not valid_mask.any(): return None

        # Use only valid samples for forward pass and loss calculation
        valid_input_ids = input_ids[valid_mask]
        valid_attention_mask = attention_mask[valid_mask]
        valid_labels = labels[valid_mask]
        # Handle token_type_ids if present
        valid_token_type_ids = batch.get("token_type_ids")
        if valid_token_type_ids is not None:
            valid_token_type_ids = valid_token_type_ids[valid_mask]

        # Forward pass
        logits, feat_norm = self(valid_input_ids, valid_attention_mask, valid_token_type_ids)

        # Calculate Cross-Entropy loss based on similarity logits
        ce_loss = F.cross_entropy(logits, valid_labels)

        # Calculate ADB margin loss using normalized features
        adb_loss = self.adb_margin_loss(feat_norm, valid_labels)

        # Combine losses using alpha from hyperparameters
        loss = ce_loss + self.hparams.param_adb_alpha * adb_loss

        # Log metrics
        self.log_dict(
            {'train_loss': loss, 'train_ce_loss': ce_loss, 'train_adb_loss': adb_loss},
            prog_bar=True, on_step=True, on_epoch=True, logger=True, batch_size=valid_labels.size(0)
        )
        # Log average radius for monitoring
        avg_radius = self.get_radii().mean().item()
        self.log("avg_radius", avg_radius, on_step=False, on_epoch=True, logger=True, batch_size=valid_labels.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        """Single validation step."""
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        labels = batch['label']

        # Filter valid samples
        valid_mask = labels >= 0
        if not valid_mask.any():
             self.log_dict({'val_loss': 0.0, 'val_acc': 0.0, 'val_ce_loss': 0.0, 'val_adb_loss': 0.0},
                           prog_bar=True, on_epoch=True, logger=True, batch_size=batch['input_ids'].size(0))
             return None

        valid_input_ids = input_ids[valid_mask]
        valid_attention_mask = attention_mask[valid_mask]
        valid_labels = labels[valid_mask]
        valid_token_type_ids = batch.get("token_type_ids")
        if valid_token_type_ids is not None:
            valid_token_type_ids = valid_token_type_ids[valid_mask]

        # Forward pass
        logits, feat_norm = self(valid_input_ids, valid_attention_mask, valid_token_type_ids)

        # Calculate losses
        ce_loss = F.cross_entropy(logits, valid_labels)
        adb_loss = self.adb_margin_loss(feat_norm, valid_labels)
        loss = ce_loss + self.hparams.param_adb_alpha * adb_loss

        # Calculate accuracy based on highest similarity logit
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(valid_labels.cpu().numpy(), preds.cpu().numpy())

        # Log metrics
        self.log_dict(
            {'val_loss': loss, 'val_ce_loss': ce_loss, 'val_adb_loss': adb_loss, 'val_acc': acc},
            prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=valid_labels.size(0)
        )
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        """Single test step, returning normalized features."""
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        labels_original = batch["label"] # Original labels (-1 possible)
        token_type_ids = batch.get("token_type_ids")

        # Forward pass for all samples
        logits, feat_norm = self(input_ids, attention_mask, token_type_ids)
        # Predictions based on highest similarity (mapped 0..num_seen-1)
        preds_mapped = torch.argmax(logits, dim=1)

        # Calculate loss/acc on knowns only
        known_mask = labels_original >= 0
        test_loss = torch.tensor(0.0, device=self.device)
        test_acc_known = torch.tensor(0.0, device=self.device)

        if known_mask.any():
            mapped_true_labels_known = self._get_mapped_labels_for_loss(labels_original)
            if mapped_true_labels_known is not None:
                valid_logits = logits[known_mask]
                valid_feat_norm = feat_norm[known_mask]

                # Calculate losses for known samples
                ce_loss = F.cross_entropy(valid_logits, mapped_true_labels_known)
                adb_loss = self.adb_margin_loss(valid_feat_norm, mapped_true_labels_known)
                test_loss = ce_loss + self.hparams.param_adb_alpha * adb_loss

                # Calculate accuracy for known samples
                test_acc_known = accuracy_score(
                    mapped_true_labels_known.cpu().numpy(),
                    preds_mapped[known_mask].cpu().numpy()
                )
                self.log('test_loss', test_loss, prog_bar=False, logger=True, batch_size=batch['input_ids'].size(0))
                self.log('test_acc_known', test_acc_known, prog_bar=False, logger=True, batch_size=batch['input_ids'].size(0))

        # Return values needed for ADB OSR evaluation
        return {
            'preds_mapped': preds_mapped,        # Predicted class index (0..num_seen-1) based on similarity
            'labels_original': labels_original,  # Original labels
            'logits': logits,                    # Similarity-based logits
            'features': feat_norm                # Normalized features (crucial for ADB eval)
        }

    def configure_optimizers(self):
        """Configure optimizer(s) and scheduler(s) for ADB.
           Allows different LRs for backbone and ADB parameters.
        """
        params_to_optimize = []

        # Add backbone parameters if not frozen
        if not self.hparams.adb_freeze_backbone:
            # Use the general 'learning_rate' hyperparameter for the backbone
            backbone_lr = self.hparams.learning_rate
            params_to_optimize.append({'params': self.roberta.parameters(), 'lr': backbone_lr})
            print(f"[ADB Optim] Configuring RoBERTa fine-tuning with LR: {backbone_lr}")
        else:
            print("[ADB Optim] RoBERTa backbone is frozen.")

        # Add ADB-specific parameters (centers, delta_prime) with their own LR
        adb_param_lr = self.hparams.lr_adb
        params_to_optimize.append({'params': self.centers, 'lr': adb_param_lr})
        params_to_optimize.append({'params': self.delta_prime, 'lr': adb_param_lr})
        print(f"[ADB Optim] Configuring centers/delta_prime optimization with LR: {adb_param_lr}")

        # Create optimizer with potentially different parameter groups
        optimizer = AdamW(params_to_optimize, weight_decay=self.hparams.weight_decay)

        # Add scheduler only if the backbone is being trained
        if not self.hparams.adb_freeze_backbone:
            scheduler_config = self._get_scheduler(optimizer)
            if scheduler_config:
                print("[ADB Optim] Using learning rate scheduler for backbone.")
                return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
            else:
                print("[ADB Optim] No learning rate scheduler used (total_steps=0 or estimation failed).")
                return optimizer
        else:
            # No scheduler needed if only ADB params are trained (or if their LR is constant)
            print("[ADB Optim] No learning rate scheduler used (backbone frozen).")
            return optimizer


# =============================================================================
# OSR Algorithms
# =============================================================================
class OSRAlgorithm:
    """Base class for Open Set Recognition algorithms."""
    def __init__(self, model: pl.LightningModule, datamodule: DataModule, args: argparse.Namespace):
        self.model = model
        self.datamodule = datamodule
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu')
        self.num_known_classes = self._get_num_known_classes()
        print(f"[{self.__class__.__name__}] Initialized for {self.num_known_classes} known classes on device {self.device}.")

    def _get_num_known_classes(self) -> int:
        """Determine the number of known classes the model was trained on."""
        if hasattr(self.model, 'num_classes') and self.model.num_classes is not None:
            return self.model.num_classes
        elif hasattr(self.model, 'hparams') and 'num_classes' in self.model.hparams:
             return self.model.hparams.num_classes
        elif self.datamodule.num_seen_classes is not None:
            print("Warning: Model doesn't explicitly state num_classes. Using datamodule.num_seen_classes.")
            return self.datamodule.num_seen_classes
        else:
            raise ValueError("Could not determine number of known classes from model or datamodule.")

    def predict(self, dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts labels, including potential 'unknown' (-1). Must be implemented by subclass.

        Args:
            dataloader: DataLoader providing test/prediction data.

        Returns:
            tuple:
                - all_scores: Method-specific scores (e.g., probs, recon errors). Shape: (N, num_classes or 1)
                - all_preds_final: Final predicted labels (-1 for unknown, original index for known). Shape: (N,)
                - all_labels_original: Ground truth labels (-1 or original index). Shape: (N,)
                - scores_for_ranking: Scores where higher means more likely unknown (for AUROC/FPR). Shape: (N,)
                - all_embeddings: Feature embeddings for t-SNE. Shape: (N, embedding_dim)
        """
        raise NotImplementedError("Predict method must be implemented by subclass.")

    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluates OSR performance on the dataloader using the predict method."""
        print(f"[{self.__class__.__name__} Evaluate] Starting evaluation...")
        eval_start_time = time.time()
        self.model.eval().to(self.device) # Ensure model is in eval mode and on correct device

        # Get predictions and ground truth from the specific algorithm's predict method
        try:
            all_scores, all_preds_final, all_labels_original, scores_for_ranking, all_embeddings = self.predict(dataloader)
        except Exception as e:
            print(f"Error during {self.__class__.__name__}.predict(): {e}")
            import traceback; traceback.print_exc()
            return self._get_empty_results("Prediction failed")

        if len(all_labels_original) == 0:
            print(f"Warning: No data returned from predict method for {self.__class__.__name__}.")
            return self._get_empty_results("No data predicted")

        print(f"[{self.__class__.__name__} Evaluate] Calculating metrics...")
        # --- Metric Calculation ---
        unknown_labels_mask = self.datamodule._determine_unknown_labels(all_labels_original)
        known_mask = ~unknown_labels_mask
        unknown_preds_mask = (all_preds_final == -1) # Predicted as unknown

        # Accuracy on Known samples only
        accuracy = 0.0
        if known_mask.any():
            # Compare final predictions (original indices or -1) with original labels for known samples
            accuracy = accuracy_score(all_labels_original[known_mask], all_preds_final[known_mask])

        # Unknown Detection Rate (Recall for the 'unknown' class)
        unknown_correct = np.sum(unknown_preds_mask & unknown_labels_mask)
        unknown_total = np.sum(unknown_labels_mask)
        unknown_detection_rate = unknown_correct / unknown_total if unknown_total > 0 else 0.0

        # AUROC and FPR@TPR90 (using scores where higher means more unknown)
        auroc = float('nan')
        fpr_at_tpr90 = float('nan')
        if len(np.unique(unknown_labels_mask)) > 1 and len(scores_for_ranking) == len(unknown_labels_mask):
            try:
                # Ensure scores are finite
                finite_scores_mask = np.isfinite(scores_for_ranking)
                if not finite_scores_mask.all():
                    print(f"Warning: Found non-finite values in scores_for_ranking. Filtering {np.sum(~finite_scores_mask)} samples.")
                valid_labels = unknown_labels_mask[finite_scores_mask]
                valid_scores = scores_for_ranking[finite_scores_mask]

                if len(np.unique(valid_labels)) > 1: # Check again after filtering
                    auroc = roc_auc_score(valid_labels, valid_scores)
                    fpr, tpr, thresholds = roc_curve(valid_labels, valid_scores)
                    # Find the FPR at the first point where TPR >= 0.90
                    tpr90_indices = np.where(tpr >= 0.90)[0]
                    if len(tpr90_indices) > 0:
                        fpr_at_tpr90 = fpr[tpr90_indices[0]]
                    else:
                        # If TPR never reaches 0.90, FPR@TPR90 is arguably 1.0 (worst case)
                        print("Warning: TPR did not reach 0.90. Setting FPR@TPR90 to 1.0.")
                        fpr_at_tpr90 = 1.0
                else:
                     print("Skipping AUROC/FPR@TPR90 calculation (only one class present after filtering scores).")

            except ValueError as e:
                print(f"AUROC/FPR calculation failed: {e}") # Handle cases like only one class in scores
            except Exception as e:
                print(f"Unexpected error calculating AUROC/FPR@TPR90: {e}")
        else:
            print("Skipping AUROC/FPR@TPR90 calculation (only one class in labels or score/label length mismatch).")

        # Confusion Matrix and F1 Score (Macro F1 on Known Classes)
        cm_axis_labels_int, cm_axis_labels_names = self._get_cm_labels()
        valid_cm_labels_set = set(cm_axis_labels_int)

        # Filter true and predicted labels to only include those in the CM axes
        # This ensures consistency if some classes were entirely absent in the test set
        filtered_labels_true = [l if l in valid_cm_labels_set else -1 for l in all_labels_original]
        filtered_labels_pred = [p if p in valid_cm_labels_set else -1 for p in all_preds_final]

        conf_matrix = confusion_matrix(filtered_labels_true, filtered_labels_pred, labels=cm_axis_labels_int)

        # Calculate precision, recall, F1 per class (including 'Unknown' if present)
        precision, recall, f1_by_class, support = precision_recall_fscore_support(
            filtered_labels_true, filtered_labels_pred, labels=cm_axis_labels_int,
            average=None, zero_division=0
        )

        # Calculate Macro F1 score considering only the *known* classes present in the CM labels
        known_class_indices_in_cm = [i for i, lbl in enumerate(cm_axis_labels_int) if lbl != -1]
        if known_class_indices_in_cm:
            f1_known_classes = f1_by_class[known_class_indices_in_cm]
            macro_f1 = np.mean(f1_known_classes)
        else:
            macro_f1 = 0.0 # No known classes evaluated

        eval_duration = time.time() - eval_start_time
        print(f"[{self.__class__.__name__} Evaluate] Metrics calculation finished in {eval_duration:.2f}s.")

        # --- Print Summary ---
        print(f"\n--- {self.__class__.__name__} Evaluation Summary ---")
        print(f"  Accuracy (Known Samples Only): {accuracy:.4f}")
        print(f"  AUROC (Unknown vs Known):      {auroc:.4f}")
        print(f"  FPR@TPR90:                     {fpr_at_tpr90:.4f}")
        print(f"  Unknown Detection Rate:        {unknown_detection_rate:.4f}")
        print(f"  Macro F1 Score (Known Only):   {macro_f1:.4f}")
        print("--------------------------------------------------")

        # --- Store Results ---
        results = {
            'accuracy': accuracy,
            'auroc': auroc,
            'f1_score': macro_f1, # Macro F1 on known classes
            'unknown_detection_rate': unknown_detection_rate,
            'fpr_at_tpr90': fpr_at_tpr90,
            'confusion_matrix': conf_matrix,
            'confusion_matrix_labels': cm_axis_labels_int, # Integer labels for CM axes
            'confusion_matrix_names': cm_axis_labels_names, # String names for CM axes
            'predictions': all_preds_final, # Final predictions (-1 or original index)
            'labels': all_labels_original, # Ground truth labels (-1 or original index)
            'scores_for_ranking': scores_for_ranking, # Scores for AUROC (higher=unknown)
            'embeddings': all_embeddings, # Embeddings for t-SNE
            'raw_scores': all_scores, # Raw scores from the method (optional, for debugging/details)
            'eval_duration_sec': eval_duration,
        }
        return results

    def _get_empty_results(self, reason: str = "Evaluation failed") -> dict:
        """Returns a dictionary with default values for failed evaluation."""
        print(f"Warning: Returning empty results because: {reason}")
        return {
            'accuracy': 0.0, 'auroc': float('nan'), 'f1_score': 0.0,
            'unknown_detection_rate': 0.0, 'fpr_at_tpr90': float('nan'),
            'confusion_matrix': None, 'confusion_matrix_labels': [],
            'confusion_matrix_names': [], 'predictions': np.array([]), 'labels': np.array([]),
            'scores_for_ranking': np.array([]), 'embeddings': None, 'raw_scores': None,
            'eval_duration_sec': 0.0, 'error': reason
        }

    def _get_seen_class_names_map(self) -> dict[int, str]:
         """Gets a map from original seen class indices to their names."""
         if self.datamodule.class_names is None or self.datamodule.original_seen_indices is None:
              print("Warning: Class names or original_seen_indices not available in datamodule. Using generic names.")
              # Fallback: Generate generic names based on the number of known classes
              return {i: f"Known_{i}" for i in range(self.num_known_classes)}

         seen_names_map = {}
         original_seen_indices = sorted(list(self.datamodule.original_seen_indices))
         num_total_classes = len(self.datamodule.class_names)

         for original_idx in original_seen_indices:
              if 0 <= original_idx < num_total_classes:
                  seen_names_map[original_idx] = self.datamodule.class_names[original_idx]
              else:
                  # This case should ideally not happen if setup is correct
                  print(f"Warning: Original seen index {original_idx} is out of bounds for class names list (len={num_total_classes}). Using generic name.")
                  seen_names_map[original_idx] = f"Class_{original_idx}"
         return seen_names_map

    def _get_cm_labels(self) -> tuple[list[int], list[str]]:
         """Gets the integer labels and string names for the confusion matrix axes."""
         # Get map from original seen index -> name
         seen_class_names_map = self._get_seen_class_names_map()

         # CM labels include -1 for Unknown, plus all original seen indices
         cm_axis_labels_int = [-1] + sorted(list(self.datamodule.original_seen_indices))

         # Get names: "Unknown" and names corresponding to the original seen indices
         cm_axis_labels_names = ["Unknown"] + [
             seen_class_names_map.get(lbl, f"Class_{lbl}") # Use map, fallback to generic name
             for lbl in cm_axis_labels_int if lbl != -1 # Iterate through sorted original seen indices
         ]
         return cm_axis_labels_int, cm_axis_labels_names

    def _map_preds_to_original(self, preds_mapped_batch: np.ndarray | torch.Tensor) -> np.ndarray:
        """Maps predicted indices (0..num_seen-1) back to original class indices."""
        original_seen_indices = self.datamodule.original_seen_indices
        if original_seen_indices is None:
            print("Warning: original_seen_indices not found in datamodule. Cannot map predictions back. Returning mapped indices.")
            return preds_mapped_batch.cpu().numpy() if isinstance(preds_mapped_batch, torch.Tensor) else np.array(preds_mapped_batch)

        # Ensure we have numpy arrays
        if isinstance(preds_mapped_batch, torch.Tensor):
            preds_mapped_np = preds_mapped_batch.cpu().numpy()
        elif not isinstance(preds_mapped_batch, np.ndarray):
            preds_mapped_np = np.array(preds_mapped_batch)
        else:
            preds_mapped_np = preds_mapped_batch

        if not isinstance(original_seen_indices, np.ndarray):
            original_seen_indices = np.array(original_seen_indices)

        # Create output array, initialized to -1 (unknown/invalid)
        preds_original_batch = np.full_like(preds_mapped_np, -1, dtype=int)

        # Find valid mapped indices (within the range 0 to num_seen_classes-1)
        valid_mask = (preds_mapped_np >= 0) & (preds_mapped_np < len(original_seen_indices))

        # Use valid mapped indices to look up the corresponding original index
        valid_mapped_indices = preds_mapped_np[valid_mask]
        preds_original_batch[valid_mask] = original_seen_indices[valid_mapped_indices]

        # Optional: Warn if some indices were invalid
        num_invalid = np.sum(~valid_mask)
        if num_invalid > 0:
            print(f"Warning: Found {num_invalid} invalid mapped prediction indices (e.g., < 0 or >= num_seen_classes) during mapping. These remain as -1.")

        return preds_original_batch

    # --- Visualization Methods ---

    def visualize(self, results: dict):
        """Generates and saves standard OSR visualizations."""
        print(f"[{self.__class__.__name__} Visualize] Generating plots...")
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Determine a base filename for plots for this method
        # Use the specific osr_method from args if evaluating one, else use class name
        method_name_for_file = self.args.osr_method if self.args.osr_method != 'all' else self.__class__.__name__.lower().replace("osr", "")
        # Include dataset and seen ratio for uniqueness
        base_filename = os.path.join(
            RESULTS_DIR,
            f"{method_name_for_file}_osr_{self.args.dataset}_{self.args.seen_class_ratio:.2f}"
        )

        if 'labels' not in results or len(results['labels']) == 0:
            print(f"[{self.__class__.__name__} Visualize] No data found in results to visualize.")
            return

        # Call common visualization helpers
        self._visualize_confusion_matrix(results, base_filename)
        self._visualize_roc_curve(results, base_filename)
        self._visualize_tsne(results, base_filename)

        # Subclasses can override this method to call super().visualize()
        # and then add their specific plots.

    def _visualize_confusion_matrix(self, results: dict, base_filename: str):
        """Generates and saves the confusion matrix plot."""
        if 'confusion_matrix' not in results or results['confusion_matrix'] is None:
            print("  Skipping confusion matrix plot (data not found in results).")
            return

        conf_matrix = results['confusion_matrix']
        cm_labels_int = results.get('confusion_matrix_labels', list(range(conf_matrix.shape[0])))
        cm_labels_names = results.get('confusion_matrix_names', [str(i) for i in cm_labels_int])

        # Ensure names match matrix dimensions
        if len(cm_labels_names) != conf_matrix.shape[0]:
             print(f"Warning: CM label names ({len(cm_labels_names)}) mismatch matrix dim ({conf_matrix.shape[0]}). Using generic labels.")
             cm_labels_names = [f"L{i}" for i in range(conf_matrix.shape[0])] # Fallback names

        # Get metrics for title
        f1_score_val = results.get('f1_score', float('nan'))
        fpr_tpr_val = results.get('fpr_at_tpr90', float('nan'))
        f1_str = f"Macro F1 (Known): {f1_score_val:.4f}" if pd.notna(f1_score_val) else "F1: N/A"
        fpr_str = f"FPR@TPR90: {fpr_tpr_val:.4f}" if pd.notna(fpr_tpr_val) else "FPR: N/A"
        method_display_name = self.__class__.__name__.replace("OSR", "") # Cleaner name for title

        try:
            plt.figure(figsize=(max(8, len(cm_labels_names) * 0.7), max(6, len(cm_labels_names) * 0.6)))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=cm_labels_names, yticklabels=cm_labels_names,
                        annot_kws={"size": 8}) # Adjust font size if needed
            plt.xlabel('Predicted Label', fontsize=10)
            plt.ylabel('True Label', fontsize=10)
            plt.title(f'Confusion Matrix ({method_display_name})\n{f1_str}, {fpr_str}', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(rotation=0, fontsize=9)
            plt.tight_layout()
            plot_filename = f"{base_filename}_confusion.png"
            plt.savefig(plot_filename)
            plt.close() # Close the figure to free memory
            print(f"  Confusion matrix plot saved to: {plot_filename}")
        except Exception as e:
            print(f"  Error generating confusion matrix plot: {e}")
            plt.close() # Ensure plot is closed even on error

    def _visualize_roc_curve(self, results: dict, base_filename: str):
         """Generates and saves the ROC curve plot for unknown detection."""
         if 'labels' not in results or 'scores_for_ranking' not in results:
             print("  Skipping ROC curve plot (missing labels or ranking scores).")
             return

         labels_np = np.array(results['labels'])
         scores = np.array(results['scores_for_ranking'])
         unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np) # True for unknown

         if len(labels_np) != len(scores):
              print(f"  Skipping ROC curve plot (label count {len(labels_np)} != score count {len(scores)}).")
              return
         if len(np.unique(unknown_labels_mask)) < 2:
              print("  Skipping ROC curve plot (only one class - known or unknown - present in labels).")
              return
         if not np.any(np.isfinite(scores)):
              print("  Skipping ROC curve plot (all scores are non-finite).")
              return

         # Filter out non-finite scores before calculating ROC
         finite_mask = np.isfinite(scores)
         if not finite_mask.all():
             print(f"  Warning: Filtering {np.sum(~finite_mask)} non-finite scores for ROC calculation.")
         valid_labels = unknown_labels_mask[finite_mask]
         valid_scores = scores[finite_mask]

         # Check again if filtering removed all samples of one class
         if len(np.unique(valid_labels)) < 2:
              print("  Skipping ROC curve plot (only one class remains after filtering non-finite scores).")
              return

         try:
             fpr, tpr, _ = roc_curve(valid_labels, valid_scores) # Use filtered data
             roc_auc_val = auc(fpr, tpr)

             plt.figure(figsize=(7, 6))
             plt.plot(fpr, tpr, lw=2.5, label=f'AUC = {roc_auc_val:.3f}', color='darkorange', alpha=0.8)
             plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', alpha=0.6) # Reference line

             # Plot FPR@TPR90 point if available and valid
             fpr90 = results.get('fpr_at_tpr90', float('nan'))
             if pd.notna(fpr90) and 0.0 <= fpr90 <= 1.0:
                 # Find the corresponding TPR (should be >= 0.9)
                 tpr90_indices = np.where(tpr >= 0.90)[0]
                 actual_tpr_at_fpr90 = tpr[tpr90_indices[0]] if len(tpr90_indices) > 0 else 0.9 # Use actual TPR achieved
                 plt.plot(fpr90, actual_tpr_at_fpr90, 'ro', markersize=8, label=f'FPR@TPR90 ({fpr90:.3f})', alpha=0.9)

             plt.xlim([-0.02, 1.02])
             plt.ylim([-0.02, 1.05])
             plt.xlabel('False Positive Rate (FPR)', fontsize=11)
             plt.ylabel('True Positive Rate (TPR) / Recall', fontsize=11)
             plt.title(f'ROC Curve - Unknown Detection ({self.__class__.__name__.replace("OSR", "")})', fontsize=13)
             plt.legend(loc="lower right", fontsize=10)
             plt.grid(True, linestyle=':', alpha=0.6)
             plt.tight_layout()
             plot_filename = f"{base_filename}_roc.png"
             plt.savefig(plot_filename)
             plt.close()
             print(f"  ROC curve plot saved to: {plot_filename}")
         except Exception as e:
             print(f"  Error generating ROC curve plot: {e}")
             plt.close()

    def _visualize_tsne(self, results: dict, base_filename: str):
        """Generates and saves a t-SNE plot of embeddings."""
        if 'embeddings' not in results or results['embeddings'] is None:
            print("  Skipping t-SNE plot (embeddings not found in results).")
            return

        features = results['embeddings']
        labels_np = np.array(results['labels'])

        if not isinstance(features, np.ndarray) or features.ndim != 2 or features.shape[0] == 0:
             print("  Skipping t-SNE plot (invalid embeddings format or empty).")
             return
        if features.shape[0] != len(labels_np):
             print(f"  Skipping t-SNE plot (embeddings count {features.shape[0]} != labels count {len(labels_np)}).")
             return
        if features.shape[1] <= 2:
             print(f"  Skipping t-SNE plot (embedding dimension {features.shape[1]} <= 2).")
             return
        if features.shape[0] < 50: # t-SNE performs poorly on very few samples
             print(f"  Skipping t-SNE plot (too few samples: {features.shape[0]} < 50).")
             return

        print("  Generating t-SNE plot (this may take a while for large datasets)...")
        unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)

        # Subsample if the dataset is too large for t-SNE
        n_samples = features.shape[0]
        if n_samples > DEFAULT_MAX_TSNE_SAMPLES:
            print(f"    Subsampling {DEFAULT_MAX_TSNE_SAMPLES} points out of {n_samples} for t-SNE.")
            indices = np.random.choice(n_samples, DEFAULT_MAX_TSNE_SAMPLES, replace=False)
            features_sub = features[indices]
            unknown_sub = unknown_labels_mask[indices]
            labels_sub = labels_np[indices] # Keep labels for potential coloring later
        else:
            features_sub = features
            unknown_sub = unknown_labels_mask
            labels_sub = labels_np

        try:
            # Determine metric based on whether features look normalized
            # Check norm of a few samples for robustness
            sample_norms = np.linalg.norm(features_sub[:min(10, features_sub.shape[0])], axis=1)
            is_normalized = np.allclose(sample_norms, 1.0, atol=1e-3)
            tsne_metric = 'cosine' if is_normalized else 'euclidean'
            print(f"    Using t-SNE metric: {tsne_metric} (features appear {'normalized' if is_normalized else 'not normalized'})")

            # Configure t-SNE
            perplexity_val = min(30, features_sub.shape[0] - 1) # Perplexity must be less than n_samples
            if perplexity_val <= 0:
                 print(f"    Skipping t-SNE: Not enough samples ({features_sub.shape[0]}) for perplexity > 0.")
                 return

            tsne = TSNE(n_components=2,
                        random_state=self.args.random_seed,
                        perplexity=perplexity_val,
                        n_iter=300, # Faster iteration count for quicker visualization
                        init='pca', # PCA initialization is generally faster and stabler
                        learning_rate='auto', # Recommended setting
                        metric=tsne_metric,
                        n_jobs=-1) # Use all available CPU cores

            reduced_feats = tsne.fit_transform(features_sub)

            # --- Plotting ---
            plt.figure(figsize=(10, 8))
            has_known = np.any(~unknown_sub)
            has_unknown = np.any(unknown_sub)

            # Plot known samples (blue)
            if has_known:
                plt.scatter(reduced_feats[~unknown_sub, 0], reduced_feats[~unknown_sub, 1],
                            c='cornflowerblue', alpha=0.5, s=10, label='Known') # Adjusted color/size

            # Plot unknown samples (red)
            if has_unknown:
                plt.scatter(reduced_feats[unknown_sub, 0], reduced_feats[unknown_sub, 1],
                            c='tomato', alpha=0.6, s=12, label='Unknown') # Adjusted color/size

            # --- Optional: Plot Centers (specifically for ADBOSR) ---
            centers_plotted = False
            if isinstance(self, ADBOSR) and hasattr(self.model, 'centers'):
                 try:
                      print("    Projecting ADB centers using the same t-SNE transformation...")
                      centers = self.model.centers.detach().cpu().numpy()
                      # Ensure centers are normalized if using cosine metric
                      if tsne_metric == 'cosine':
                          centers_norm = F.normalize(torch.from_numpy(centers), p=2, dim=-1).numpy()
                      else:
                          centers_norm = centers # Use raw centers if euclidean

                      # Transform centers using the *already fitted* t-SNE object
                      # Note: This is an approximation, as t-SNE is non-parametric.
                      # A more correct way involves fitting t-SNE on combined data,
                      # but that's slower and might distort the original embedding space.
                      # We'll stick to transforming centers for visualization simplicity.

                      # Need a way to apply the transformation. `transform` isn't standard in sklearn TSNE.
                      # Alternative: Fit t-SNE on combined data (features + centers)
                      combined_for_tsne = np.vstack([features_sub, centers_norm])
                      tsne_combined = TSNE(n_components=2, random_state=self.args.random_seed,
                                           perplexity=min(30, combined_for_tsne.shape[0]-1),
                                           n_iter=300, init='pca', learning_rate='auto',
                                           metric=tsne_metric, n_jobs=-1)
                      reduced_combined = tsne_combined.fit_transform(combined_for_tsne)
                      # Extract the transformed centers
                      reduced_centers = reduced_combined[len(features_sub):]

                      plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1],
                                  c='black', marker='X', s=120, edgecolors='white',
                                  linewidth=1.5, label='Class Centers (ADB)', zorder=3) # Plot on top
                      centers_plotted = True
                 except Exception as tsne_center_e:
                      print(f"    Warning: Could not project or plot ADB centers with t-SNE: {tsne_center_e}")

            # --- Final Touches ---
            plt.title(f't-SNE Visualization ({self.__class__.__name__.replace("OSR", "")} Embeddings)', fontsize=14)
            plt.xlabel("t-SNE Dimension 1", fontsize=11)
            plt.ylabel("t-SNE Dimension 2", fontsize=11)
            if has_known or has_unknown or centers_plotted:
                plt.legend(markerscale=1.5, fontsize=10)
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.tight_layout()
            plot_filename = f"{base_filename}_tsne.png"
            plt.savefig(plot_filename)
            plt.close()
            print(f"  t-SNE plot saved to: {plot_filename}")

        except ImportError:
            print("  Skipping t-SNE plot: scikit-learn (sklearn) is not installed or TSNE failed to import.")
        except Exception as e:
            print(f"  Error during t-SNE generation or plotting: {e}")
            import traceback; traceback.print_exc()
            plt.close() # Ensure plot is closed even on error


# --- OSR Algorithm Implementations ---

class ThresholdOSR(OSRAlgorithm):
    """OSR using a simple threshold on the maximum softmax probability."""
    def __init__(self, model, datamodule, args):
        super().__init__(model, datamodule, args)
        # Use provided threshold or default
        self.threshold = args.param_threshold if args.param_threshold is not None else 0.5
        print(f"[ThresholdOSR Init] Using softmax threshold: {self.threshold:.4f}")
        if not isinstance(self.model, RobertaClassifier):
             print(f"Warning: ThresholdOSR typically used with RobertaClassifier, but got {type(model).__name__}. Ensure model outputs standard logits.")

    def predict(self, dataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval().to(self.device)
        all_probs_list, all_preds_final_list, all_labels_original_list = [], [], []
        all_max_probs_list, all_embeddings_list = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (Threshold OSR)", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch.get('token_type_ids') # Optional
                if token_type_ids is not None: token_type_ids = token_type_ids.to(self.device)
                labels_orig = batch['label'].cpu().numpy() # Original labels (-1 or orig index)

                # Get model outputs (logits and embeddings)
                logits, embeddings = self.model(input_ids, attention_mask, token_type_ids)

                # Calculate softmax probabilities and find max probability + predicted class (mapped)
                probs = F.softmax(logits, dim=1)
                max_probs, preds_mapped = torch.max(probs, dim=1)

                # Convert to numpy for processing
                preds_mapped_cpu = preds_mapped.cpu().numpy()
                max_probs_cpu = max_probs.cpu().numpy()

                # Apply threshold: If max prob < threshold, predict unknown (-1)
                # Otherwise, map the predicted index back to the original class index
                final_batch_preds = np.full_like(preds_mapped_cpu, -1, dtype=int) # Default to unknown
                accept_mask = max_probs_cpu >= self.threshold # Samples above threshold

                if np.any(accept_mask):
                    # Map only the accepted predictions back to original indices
                    accepted_preds_mapped = preds_mapped_cpu[accept_mask]
                    mapped_original_preds = self._map_preds_to_original(accepted_preds_mapped)
                    final_batch_preds[accept_mask] = mapped_original_preds

                # Store results for the batch
                all_probs_list.append(probs.cpu().numpy())
                all_preds_final_list.extend(final_batch_preds)
                all_labels_original_list.extend(labels_orig)
                all_max_probs_list.append(max_probs_cpu)
                all_embeddings_list.append(embeddings.cpu().numpy())

        # Concatenate results from all batches
        all_probs = np.concatenate(all_probs_list) if all_probs_list else np.empty((0, self.num_known_classes))
        all_preds_final = np.array(all_preds_final_list)
        all_labels_original = np.array(all_labels_original_list)
        all_max_probs = np.concatenate(all_max_probs_list) if all_max_probs_list else np.array([])
        all_embeddings = np.concatenate(all_embeddings_list) if all_embeddings_list else np.empty((0, self.model.config.hidden_size))

        # Scores for ranking: Higher score = more likely unknown. Use negative max prob.
        scores_for_ranking = -all_max_probs

        # Return: raw scores (softmax probs), final predictions, original labels, ranking scores, embeddings
        return all_probs, all_preds_final, all_labels_original, scores_for_ranking, all_embeddings

    def visualize(self, results):
        """Adds confidence distribution plot to standard visualizations."""
        super().visualize(results) # Call common visualizations (CM, ROC, t-SNE)
        print(f"[{self.__class__.__name__} Visualize] Generating specific plots...")
        method_name_for_file = self.args.osr_method if self.args.osr_method != 'all' else self.__class__.__name__.lower().replace("osr","")
        base_filename = os.path.join(RESULTS_DIR, f"{method_name_for_file}_osr_{self.args.dataset}_{self.args.seen_class_ratio:.2f}")

        # Use negative ranking scores to get back max probabilities
        max_probs = -results.get('scores_for_ranking', np.array([]))

        if max_probs.size > 0:
             labels_np = np.array(results['labels'])
             unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)
             plot_data = pd.DataFrame({'Max Softmax Probability': max_probs, 'Is Known': ~unknown_labels_mask})

             try:
                 plt.figure(figsize=(8, 5))
                 sns.histplot(data=plot_data, x='Max Softmax Probability', hue='Is Known', kde=True,
                              stat='density', common_norm=False, bins=50, palette=['tomato', 'cornflowerblue'])
                 plt.axvline(self.threshold, color='black', linestyle='--', linewidth=1.5, label=f'Threshold = {self.threshold:.3f}')
                 plt.title('Confidence Distribution (Max Softmax Probability)', fontsize=13)
                 plt.xlabel('Max Softmax Probability', fontsize=11)
                 plt.ylabel('Density', fontsize=11)
                 plt.legend(title='Ground Truth', fontsize=10)
                 plt.grid(True, linestyle=':', alpha=0.5)
                 plt.tight_layout()
                 plot_filename = f"{base_filename}_confidence_dist.png"
                 plt.savefig(plot_filename)
                 plt.close()
                 print(f"  Confidence distribution plot saved to: {plot_filename}")
             except Exception as e:
                 print(f"  Error generating confidence distribution plot: {e}")
                 plt.close()
        else:
            print("  Skipping confidence distribution plot (no ranking scores found).")
        print(f"[{self.__class__.__name__} Visualize] Finished specific plots.")


class OpenMaxOSR(OSRAlgorithm):
    """OpenMax OSR algorithm implementation."""
    DEFAULT_TAIL_SIZE = 50
    DEFAULT_ALPHA = 10

    def __init__(self, model, datamodule, args):
        super().__init__(model, datamodule, args)
        self.tail_size = args.param_openmax_tailsize if args.param_openmax_tailsize is not None else self.DEFAULT_TAIL_SIZE
        self.alpha = args.param_openmax_alpha if args.param_openmax_alpha is not None else self.DEFAULT_ALPHA
        print(f"[OpenMaxOSR Init] Tail size: {self.tail_size}, Alpha: {self.alpha}, Known Classes: {self.num_known_classes}")
        if not isinstance(self.model, RobertaClassifier):
             print(f"Warning: OpenMaxOSR typically used with RobertaClassifier, but got {type(model).__name__}. Ensure model outputs standard logits and CLS embeddings.")

        # Placeholders for fitted models
        self.mav_dict: dict[int, np.ndarray] = {} # Mean Activation Vectors (MAVs) per class (mapped index)
        self.weibull_models: dict[int, tuple[float, float, float] | None] = {} # Weibull params (shape, loc, scale) or None if failed
        self.feat_dim = model.config.hidden_size # Dimension of embeddings

    def _fit_weibull_models(self, dataloader: DataLoader):
        """Fits class-specific Weibull models using correctly classified training samples."""
        print("[OpenMaxOSR Fit] Starting Weibull model fitting...")
        fit_start_time = time.time()
        self.model.eval().to(self.device)
        # Store embeddings (activation vectors) per class {mapped_class_idx: [embeddings]}
        embeddings_per_class = defaultdict(list)
        correctly_classified_count = 0

        print("[OpenMaxOSR Fit] Collecting embeddings from correctly classified training samples...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="OpenMax Fit: Collecting", leave=False):
                ids = batch['input_ids'].to(self.device)
                attn = batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids')
                if tok is not None: tok = tok.to(self.device)
                # Labels in training loader are already mapped (0..N-1)
                labels_mapped = batch["label"].to(self.device)

                logits, embeddings = self.model(ids, attn, tok)
                preds_mapped = torch.argmax(logits, dim=1)

                # Collect embeddings only for samples where prediction matches the true (mapped) label
                correct_mask = (preds_mapped == labels_mapped) & (labels_mapped >= 0)
                correctly_classified_count += correct_mask.sum().item()

                for i in torch.where(correct_mask)[0]: # Iterate over indices where correct
                    true_idx = labels_mapped[i].item()
                    embeddings_per_class[true_idx].append(embeddings[i].cpu().numpy())

        print(f"[OpenMaxOSR Fit] Collected embeddings for {len(embeddings_per_class)} classes from {correctly_classified_count} samples.")
        self.mav_dict.clear()
        self.weibull_models.clear()

        print("[OpenMaxOSR Fit] Calculating MAVs and fitting Weibull models...")
        for c_idx in tqdm(range(self.num_known_classes), desc="OpenMax Fit: Weibull Fitting", leave=False):
            if c_idx not in embeddings_per_class or len(embeddings_per_class[c_idx]) == 0:
                print(f"  Warning: No correctly classified samples found for class index {c_idx}. Cannot fit Weibull model.")
                self.mav_dict[c_idx] = np.zeros(self.feat_dim) # Placeholder MAV
                self.weibull_models[c_idx] = None # Mark as failed/unavailable
                continue

            # Calculate Mean Activation Vector (MAV)
            class_embeddings = np.stack(embeddings_per_class[c_idx])
            self.mav_dict[c_idx] = np.mean(class_embeddings, axis=0)

            # Calculate distances from MAV for Weibull fitting
            distances = np.linalg.norm(class_embeddings - self.mav_dict[c_idx], axis=1)
            distances_sorted = np.sort(distances)

            # Determine tail size dynamically
            current_tail_size = min(self.tail_size, len(distances_sorted))
            if current_tail_size < 2: # Need at least 2 points to fit Weibull
                print(f"  Warning: Insufficient tail size ({current_tail_size}) for class {c_idx} after collecting samples. Cannot fit Weibull.")
                self.weibull_models[c_idx] = None
                continue

            # Fit Weibull on the tail distances
            tail_distances = distances_sorted[-current_tail_size:]
            try:
                # Fit weibull_min distribution, fixing location (floc=0) as distances are non-negative
                shape, loc, scale = weibull_min.fit(tail_distances, floc=0)
                # Sanity check the fitted parameters
                if not np.isfinite([shape, scale]).all() or scale <= 1e-9 or shape <= 1e-9:
                    print(f"  Warning: Weibull fit resulted in invalid parameters (shape={shape:.2e}, scale={scale:.2e}) for class {c_idx}. Fit failed.")
                    self.weibull_models[c_idx] = None
                else:
                    self.weibull_models[c_idx] = (shape, loc, scale)
                    # print(f"  Class {c_idx}: Weibull fit OK (Shape={shape:.3f}, Scale={scale:.3f})") # Optional: Verbose logging
            except Exception as e:
                print(f"  Warning: Weibull fit exception for class {c_idx}: {e}. Fit failed.")
                self.weibull_models[c_idx] = None

        fit_duration = time.time() - fit_start_time
        print(f"[OpenMaxOSR Fit] Weibull fitting complete in {fit_duration:.2f}s. Models available for {len([m for m in self.weibull_models.values() if m is not None])}/{self.num_known_classes} classes.")

    def _compute_openmax_probabilities(self, embedding_av: np.ndarray, logits: np.ndarray) -> np.ndarray:
        """Computes OpenMax probabilities for a single sample."""
        num_known_classes_in_logits = len(logits)
        if num_known_classes_in_logits != self.num_known_classes:
             print(f"Warning: Logits dimension ({num_known_classes_in_logits}) differs from expected known classes ({self.num_known_classes}). Check model output.")
             # Attempt to proceed, but results might be unreliable

        # Check if MAVs and Weibull models are available
        if not self.mav_dict or not self.weibull_models:
            print("Error: MAVs or Weibull models not fitted. Returning uncalibrated softmax.")
            # Fallback to standard softmax + zero unknown probability
            exp_logits = np.exp(logits - np.max(logits))
            softmax_probs = exp_logits / np.sum(exp_logits)
            return np.append(softmax_probs, 0.0) # Append 0 for unknown score

        # Calculate distances from the sample embedding to each class MAV
        distances = np.full(self.num_known_classes, np.inf)
        for c_idx in range(self.num_known_classes):
            if c_idx in self.mav_dict:
                dist = np.linalg.norm(embedding_av - self.mav_dict[c_idx])
                distances[c_idx] = dist

        # Calculate Weibull CDF scores (probability of being an outlier for that class)
        weibull_cdf_scores = np.ones(self.num_known_classes) # Default to 1 (max outlier score)
        for c_idx in range(self.num_known_classes):
             weibull_params = self.weibull_models.get(c_idx)
             if weibull_params is not None and np.isfinite(distances[c_idx]):
                 shape, loc, scale = weibull_params
                 cdf = weibull_min.cdf(distances[c_idx], shape, loc=loc, scale=scale)
                 weibull_cdf_scores[c_idx] = cdf

        # Revise activation vector (logits) based on Weibull CDF scores
        revised_logits = logits.copy()
        # Sort logits in descending order to find top alpha classes
        sorted_indices = np.argsort(logits)[::-1]
        current_alpha = min(self.alpha, self.num_known_classes) # Ensure alpha isn't larger than num classes

        # Modulate logits of top-alpha classes by (1 - CDF score)
        for rank, c_idx in enumerate(sorted_indices):
            if rank < current_alpha:
                # Ensure index is valid before accessing cdf scores
                if 0 <= c_idx < len(weibull_cdf_scores):
                     revised_logits[c_idx] *= (1.0 - weibull_cdf_scores[c_idx])
                else:
                     print(f"Warning: Index {c_idx} out of bounds for CDF scores (len {len(weibull_cdf_scores)}). Skipping revision.")


        # Calculate the 'unknown' logit score
        unknown_logit_score = 0.0
        for rank, c_idx in enumerate(sorted_indices):
             if rank < current_alpha:
                 if 0 <= c_idx < len(weibull_cdf_scores) and 0 <= c_idx < len(logits):
                     unknown_logit_score += logits[c_idx] * weibull_cdf_scores[c_idx]
                 else:
                      print(f"Warning: Index {c_idx} out of bounds for CDF/logits during unknown score calculation. Skipping.")


        # Combine revised known logits with the new unknown logit
        final_logits_with_unknown = np.append(revised_logits, unknown_logit_score)

        # Compute final probabilities using softmax over the combined logits
        exp_final_logits = np.exp(final_logits_with_unknown - np.max(final_logits_with_unknown))
        openmax_probs = exp_final_logits / np.sum(exp_final_logits)

        return openmax_probs # Shape: (num_known_classes + 1,)

    def predict(self, dataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Fit Weibull models if not already done (e.g., first call or if fitting failed)
        # Check specifically if any valid models were fitted
        valid_weibull_models_exist = any(m is not None for m in self.weibull_models.values())
        if not self.mav_dict or not valid_weibull_models_exist:
            print("[OpenMaxOSR Predict] MAVs or Weibull models not fitted or incomplete. Fitting now...")
            # Use training dataloader for fitting
            try:
                train_loader = self.datamodule.train_dataloader()
                if train_loader is None: raise ValueError("Training dataloader is required for OpenMax fitting but is None.")
                self._fit_weibull_models(train_loader)
                valid_weibull_models_exist = any(m is not None for m in self.weibull_models.values())
                if not self.mav_dict or not valid_weibull_models_exist:
                    print("Error: OpenMax Weibull fitting failed or yielded no valid models. Predictions will use fallback softmax.")
            except Exception as fit_e:
                 print(f"Error during OpenMax fitting: {fit_e}. Predictions will use fallback softmax.")
                 self.mav_dict = {} # Ensure fallback is triggered in _compute_openmax_probabilities
                 self.weibull_models = {}

        self.model.eval().to(self.device)
        all_openmax_probs_list, all_preds_final_list, all_labels_original_list = [], [], []
        all_embeddings_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (OpenMax OSR)", leave=False):
                ids = batch['input_ids'].to(self.device)
                attn = batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids')
                if tok is not None: tok = tok.to(self.device)
                labels_orig = batch["label"].cpu().numpy() # Original labels

                # Get standard model outputs
                logits_batch_gpu, embeddings_batch_gpu = self.model(ids, attn, tok)
                logits_batch_cpu = logits_batch_gpu.cpu().numpy()
                embeddings_batch_cpu = embeddings_batch_gpu.cpu().numpy()
                all_embeddings_list.append(embeddings_batch_cpu) # Store embeddings

                # Process each sample in the batch
                batch_om_probs = []
                batch_final_preds = []
                for i in range(len(labels_orig)):
                    embedding_sample = embeddings_batch_cpu[i]
                    logits_sample = logits_batch_cpu[i]

                    # Compute OpenMax probabilities (includes unknown score at the end)
                    om_probs = self._compute_openmax_probabilities(embedding_sample, logits_sample)
                    batch_om_probs.append(om_probs)

                    # Determine final prediction
                    pred_idx_with_unknown = np.argmax(om_probs) # Index in the range [0, num_known_classes]

                    if pred_idx_with_unknown == self.num_known_classes: # Predicted as unknown
                        pred_final = -1
                    else: # Predicted as one of the known classes
                        # Map the predicted mapped index back to the original index
                        # Need to handle potential errors if index is somehow invalid
                        if 0 <= pred_idx_with_unknown < self.num_known_classes:
                            pred_final = self._map_preds_to_original([pred_idx_with_unknown])[0]
                        else:
                            print(f"Warning: Invalid OpenMax predicted known index {pred_idx_with_unknown}. Assigning -1.")
                            pred_final = -1
                    batch_final_preds.append(pred_final)

                all_openmax_probs_list.extend(batch_om_probs)
                all_preds_final_list.extend(batch_final_preds)
                all_labels_original_list.extend(labels_orig)

        # Concatenate results
        # Shape: (N, num_known_classes + 1)
        all_openmax_probs = np.array(all_openmax_probs_list) if all_openmax_probs_list else np.empty((0, self.num_known_classes + 1))
        all_preds_final = np.array(all_preds_final_list)
        all_labels_original = np.array(all_labels_original_list)
        all_embeddings = np.concatenate(all_embeddings_list) if all_embeddings_list else np.empty((0, self.feat_dim))

        # Scores for ranking: Use the probability of the 'unknown' class (last column)
        # Higher score = more likely unknown. Handle case where fallback occurred.
        if all_openmax_probs.shape[1] > self.num_known_classes:
            scores_for_ranking = all_openmax_probs[:, -1]
        else: # Fallback case where unknown prob wasn't added
             scores_for_ranking = np.zeros(len(all_labels_original))

        # Return: raw scores (OpenMax probs), final predictions, original labels, ranking scores, embeddings
        return all_openmax_probs, all_preds_final, all_labels_original, scores_for_ranking, all_embeddings

    def visualize(self, results):
        """Adds unknown probability distribution plot."""
        super().visualize(results) # Call common visualizations
        print(f"[{self.__class__.__name__} Visualize] Generating specific plots...")
        method_name_for_file = self.args.osr_method if self.args.osr_method != 'all' else self.__class__.__name__.lower().replace("osr","")
        base_filename = os.path.join(RESULTS_DIR, f"{method_name_for_file}_osr_{self.args.dataset}_{self.args.seen_class_ratio:.2f}")

        # Use the ranking score which is the unknown probability
        unknown_probs = results.get("scores_for_ranking")

        if unknown_probs is not None and len(unknown_probs) > 0:
            labels_np = np.array(results['labels'])
            unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)
            plot_data = pd.DataFrame({'Unknown Probability': unknown_probs, 'Is Known': ~unknown_labels_mask})

            try:
                plt.figure(figsize=(8, 5))
                sns.histplot(data=plot_data, x='Unknown Probability', hue='Is Known', kde=True,
                             stat='density', common_norm=False, bins=50, palette=['tomato', 'cornflowerblue'])
                plt.title('OpenMax Unknown Probability Distribution', fontsize=13)
                plt.xlabel('OpenMax Unknown Probability', fontsize=11)
                plt.ylabel('Density', fontsize=11)
                plt.legend(title='Ground Truth', fontsize=10)
                plt.grid(True, linestyle=':', alpha=0.5)
                plt.tight_layout()
                plot_filename = f"{base_filename}_unknown_prob_dist.png"
                plt.savefig(plot_filename)
                plt.close()
                print(f"  Unknown probability distribution plot saved to: {plot_filename}")
            except Exception as e:
                print(f"  Error generating unknown probability distribution plot: {e}")
                plt.close()
        else:
            print("  Skipping unknown probability distribution plot (no ranking scores found).")
        print(f"[{self.__class__.__name__} Visualize] Finished specific plots.")


class CROSROSR(OSRAlgorithm):
    """CROSR algorithm using reconstruction error and EVT."""
    DEFAULT_THRESHOLD_CDF = 0.9 # Default CDF threshold if not specified
    DEFAULT_TAIL_SIZE = 100

    def __init__(self, model, datamodule, args):
        if not isinstance(model, RobertaAutoencoder):
            raise TypeError(f"CROSR requires a RobertaAutoencoder model, but got {type(model).__name__}.")
        super().__init__(model, datamodule, args)
        # Use provided params or defaults
        self.threshold_cdf = args.param_crosr_reconstruction_threshold if args.param_crosr_reconstruction_threshold is not None else self.DEFAULT_THRESHOLD_CDF
        self.tail_size = args.param_crosr_tailsize if args.param_crosr_tailsize is not None else self.DEFAULT_TAIL_SIZE
        print(f"[CROSROSR Init] Threshold (CDF): {self.threshold_cdf:.4f}, EVT Tail Size: {self.tail_size}")
        self.weibull_model: tuple[float, float, float] | None = None # (shape, loc, scale) or None if failed

    def _fit_evt_model(self, dataloader: DataLoader):
        """Fits a Weibull EVT model on reconstruction errors from known training samples."""
        print("[CROSROSR Fit] Starting EVT model fitting on reconstruction errors...")
        fit_start_time = time.time()
        self.model.eval().to(self.device)
        recon_errors = []

        print("[CROSROSR Fit] Collecting reconstruction errors from known training samples...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="CROSR Fit: Collecting Errors", leave=False):
                ids = batch['input_ids'].to(self.device)
                attn = batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids')
                if tok is not None: tok = tok.to(self.device)
                labels_mapped = batch['label'].to(self.device) # Already mapped 0..N-1

                # Use only known samples (should be all in train loader)
                valid_mask = labels_mapped >= 0
                if not valid_mask.any(): continue

                # Get model outputs for valid samples
                _, cls_output, _, reconstructed = self.model(ids[valid_mask], attn[valid_mask], tok[valid_mask] if tok is not None else None)

                # Calculate L2 reconstruction error
                batch_errors = torch.norm(reconstructed - cls_output, p=2, dim=1).cpu().numpy()
                recon_errors.extend(batch_errors)

        if not recon_errors:
            print("Warning: No reconstruction errors collected for EVT fit. Cannot fit model.")
            self.weibull_model = None
            return

        errors_np = np.sort(np.array(recon_errors))
        # Determine dynamic tail size
        current_tail_size = min(self.tail_size, len(errors_np))
        if current_tail_size < 2:
            print(f"Warning: Insufficient tail size ({current_tail_size}) for EVT fit. Using default model.")
            # Fallback: Use mean error as scale, shape=1 (exponential)
            mean_error = np.mean(errors_np) if errors_np.size > 0 else 1.0
            self.weibull_model = (1.0, 0.0, mean_error)
            return

        # Fit Weibull on the tail errors
        tail_errors = errors_np[-current_tail_size:]
        print(f"[CROSROSR Fit] Fitting Weibull EVT model on tail of {len(tail_errors)} errors...")
        try:
            shape, loc, scale = weibull_min.fit(tail_errors, floc=0) # Fix location to 0
            if not np.isfinite([shape, scale]).all() or scale <= 1e-9 or shape <= 1e-9:
                raise ValueError(f"Invalid Weibull parameters: shape={shape:.2e}, scale={scale:.2e}")
            self.weibull_model = (shape, loc, scale)
            print(f"  CROSR Fitted Weibull: shape={shape:.4f}, loc={loc:.4f}, scale={scale:.4f}")
        except Exception as e:
            print(f"Warning: CROSR Weibull fit exception: {e}. Using fallback model.")
            mean_tail_error = np.mean(tail_errors) if len(tail_errors) > 0 else 1.0
            self.weibull_model = (1.0, 0.0, mean_tail_error) # Fallback

        fit_duration = time.time() - fit_start_time
        print(f"[CROSROSR Fit] EVT fitting complete in {fit_duration:.2f}s.")

    def predict(self, dataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Fit EVT model if not already done
        if self.weibull_model is None:
            print("[CROSROSR Predict] EVT model not fitted. Fitting now...")
            try:
                train_loader = self.datamodule.train_dataloader()
                if train_loader is None: raise ValueError("Training dataloader required for CROSR fitting.")
                self._fit_evt_model(train_loader)
                if self.weibull_model is None: # Check if fitting failed
                    print("Error: EVT fitting failed. Using default fallback model (shape=1, loc=0, scale=1).")
                    self.weibull_model = (1.0, 0.0, 1.0) # Default fallback
            except Exception as fit_e:
                 print(f"Error during CROSR fitting: {fit_e}. Using default fallback model.")
                 self.weibull_model = (1.0, 0.0, 1.0) # Default fallback

        self.model.eval().to(self.device)
        all_recon_errors_list, all_unknown_probs_list, all_preds_final_list = [], [], []
        all_labels_original_list, all_embeddings_list = [], []
        shape, loc, scale = self.weibull_model # Unpack fitted params

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (CROSR)", leave=False):
                ids = batch['input_ids'].to(self.device)
                attn = batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids')
                if tok is not None: tok = tok.to(self.device)
                labels_orig = batch['label'].cpu().numpy() # Original labels

                # Get model outputs (need recon errors)
                logits, cls_output, _, reconstructed = self.model(ids, attn, tok)
                preds_mapped = torch.argmax(logits, dim=1) # Standard prediction (mapped 0..N-1)

                # Calculate L2 reconstruction error for all samples in batch
                recon_errors_batch = torch.norm(reconstructed - cls_output, p=2, dim=1).cpu().numpy()

                # Calculate unknown probability using Weibull CDF
                # Ensure errors are non-negative for CDF calculation
                non_negative_errors = np.maximum(recon_errors_batch, 0)
                unknown_probs_batch = weibull_min.cdf(non_negative_errors, shape, loc=loc, scale=scale)

                # Apply threshold based on CDF score
                preds_mapped_cpu = preds_mapped.cpu().numpy()
                batch_preds_final = np.full_like(preds_mapped_cpu, -1, dtype=int) # Default unknown
                # Accept if CDF score (unknown prob) is *below* or equal to the threshold
                accept_mask = unknown_probs_batch <= self.threshold_cdf

                if np.any(accept_mask):
                    # Map accepted predictions back to original class indices
                    accepted_preds_mapped = preds_mapped_cpu[accept_mask]
                    mapped_original_preds = self._map_preds_to_original(accepted_preds_mapped)
                    batch_preds_final[accept_mask] = mapped_original_preds

                # Store results
                all_recon_errors_list.extend(recon_errors_batch)
                all_unknown_probs_list.extend(unknown_probs_batch)
                all_preds_final_list.extend(batch_preds_final)
                all_labels_original_list.extend(labels_orig)
                all_embeddings_list.append(cls_output.cpu().numpy()) # Store original CLS embeddings

        # Concatenate results
        all_recon_errors = np.array(all_recon_errors_list)
        all_unknown_probs = np.array(all_unknown_probs_list)
        all_preds_final = np.array(all_preds_final_list)
        all_labels_original = np.array(all_labels_original_list)
        all_embeddings = np.concatenate(all_embeddings_list) if all_embeddings_list else np.empty((0, self.model.config.hidden_size))

        # Scores for ranking: Use the unknown probability (CDF score). Higher = more unknown.
        scores_for_ranking = all_unknown_probs

        # Return: raw scores (recon errors), final predictions, original labels, ranking scores (CDF), embeddings
        return all_recon_errors, all_preds_final, all_labels_original, scores_for_ranking, all_embeddings

    def visualize(self, results):
        """Adds reconstruction error and unknown probability distribution plots."""
        super().visualize(results) # Call common visualizations
        print(f"[{self.__class__.__name__} Visualize] Generating specific plots...")
        method_name_for_file = self.args.osr_method if self.args.osr_method != 'all' else self.__class__.__name__.lower().replace("osr","")
        base_filename = os.path.join(RESULTS_DIR, f"{method_name_for_file}_osr_{self.args.dataset}_{self.args.seen_class_ratio:.2f}")

        # Get reconstruction errors (stored as 'raw_scores') and unknown probs (CDF, stored as 'scores_for_ranking')
        recon_errors = results.get('raw_scores')
        unknown_probs = results.get('scores_for_ranking')
        labels_np = np.array(results['labels'])
        unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)

        # Plot Reconstruction Error Distribution
        if recon_errors is not None and len(recon_errors) > 0:
            plot_data_err = pd.DataFrame({'Reconstruction Error': recon_errors, 'Is Known': ~unknown_labels_mask})
            try:
                plt.figure(figsize=(8, 5))
                sns.histplot(data=plot_data_err, x='Reconstruction Error', hue='Is Known', kde=True,
                             stat='density', common_norm=False, bins=50, palette=['tomato', 'cornflowerblue'])
                plt.title('CROSR Reconstruction Error Distribution', fontsize=13)
                plt.xlabel('L2 Reconstruction Error', fontsize=11)
                plt.ylabel('Density', fontsize=11)
                plt.legend(title='Ground Truth', fontsize=10)
                plt.grid(True, linestyle=':', alpha=0.5)
                plt.tight_layout()
                plot_filename_err = f"{base_filename}_recon_error_dist.png"
                plt.savefig(plot_filename_err)
                plt.close()
                print(f"  Reconstruction error distribution plot saved to: {plot_filename_err}")
            except Exception as e:
                print(f"  Error generating reconstruction error plot: {e}")
                plt.close()
        else:
            print("  Skipping reconstruction error distribution plot (no raw scores found).")

        # Plot Unknown Probability (CDF) Distribution
        if unknown_probs is not None and len(unknown_probs) > 0:
             plot_data_prob = pd.DataFrame({'Unknown Probability (CDF)': unknown_probs, 'Is Known': ~unknown_labels_mask})
             try:
                 plt.figure(figsize=(8, 5))
                 sns.histplot(data=plot_data_prob, x='Unknown Probability (CDF)', hue='Is Known', kde=True,
                              stat='density', common_norm=False, bins=50, palette=['tomato', 'cornflowerblue'])
                 plt.axvline(self.threshold_cdf, color='black', linestyle='--', linewidth=1.5, label=f'Threshold = {self.threshold_cdf:.3f}')
                 plt.title('CROSR Unknown Probability Distribution (Weibull CDF)', fontsize=13)
                 plt.xlabel('Weibull CDF of Reconstruction Error', fontsize=11)
                 plt.ylabel('Density', fontsize=11)
                 plt.legend(title='Ground Truth', fontsize=10)
                 plt.grid(True, linestyle=':', alpha=0.5)
                 plt.tight_layout()
                 plot_filename_prob = f"{base_filename}_unknown_prob_dist.png"
                 plt.savefig(plot_filename_prob)
                 plt.close()
                 print(f"  Unknown probability distribution plot saved to: {plot_filename_prob}")
             except Exception as e:
                 print(f"  Error generating unknown probability plot: {e}")
                 plt.close()
        else:
             print("  Skipping unknown probability distribution plot (no ranking scores found).")

        print(f"[{self.__class__.__name__} Visualize] Finished specific plots.")


class DOCOSR(OSRAlgorithm):
    """DOC algorithm using class-specific thresholds on sigmoid scores."""
    DEFAULT_K_SIGMA = 3.0 # Default k for threshold calculation

    def __init__(self, model, datamodule, args):
        if not isinstance(model, DOCRobertaClassifier):
            raise TypeError(f"DOC OSR requires a DOCRobertaClassifier model, but got {type(model).__name__}.")
        super().__init__(model, datamodule, args)
        self.k_sigma = args.param_doc_k if args.param_doc_k is not None else self.DEFAULT_K_SIGMA
        print(f"[DOCOSR Init] Using k-sigma: {self.k_sigma}")
        # Stores (mean, std_dev) of mirrored scores for each class
        self.gaussian_params: dict[int, tuple[float, float]] = {}
        # Stores calculated threshold (max(0.5, 1.0 - k*std)) for each class
        self.class_thresholds: dict[int, float] = {}

    def _fit_gaussian_models(self, dataloader: DataLoader):
        """Fits Gaussian models to mirrored sigmoid scores for each known class."""
        print("[DOCOSR Fit] Starting Gaussian model fitting for class thresholds...")
        fit_start_time = time.time()
        self.model.eval().to(self.device)
        # Store relevant sigmoid scores {mapped_class_idx: [scores]}
        scores_per_class = defaultdict(list)

        print("[DOCOSR Fit] Collecting relevant sigmoid scores from known training samples...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="DOC Fit: Collecting Scores", leave=False):
                ids = batch['input_ids'].to(self.device)
                attn = batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids')
                if tok is not None: tok = tok.to(self.device)
                labels_mapped = batch['label'].to(self.device) # Already mapped 0..N-1

                # Use only known samples
                valid_mask = labels_mapped >= 0
                if not valid_mask.any(): continue

                # Get sigmoid scores from DOC model
                logits, _ = self.model(ids[valid_mask], attn[valid_mask], tok[valid_mask] if tok is not None else None)
                sigmoid_scores = torch.sigmoid(logits)
                valid_labels = labels_mapped[valid_mask]

                # Collect the sigmoid score corresponding to the true class for each sample
                for i, true_mapped_idx in enumerate(valid_labels):
                     idx = true_mapped_idx.item()
                     # Ensure index is valid before accessing score
                     if 0 <= idx < sigmoid_scores.shape[1]:
                         scores_per_class[idx].append(sigmoid_scores[i, idx].item())
                     else:
                          print(f"Warning: Invalid class index {idx} encountered during score collection.")


        print(f"[DOCOSR Fit] Collected scores for {len(scores_per_class)} classes.")
        self.gaussian_params.clear()
        self.class_thresholds.clear()

        print("[DOCOSR Fit] Calculating Gaussian parameters and thresholds...")
        default_threshold = 0.5 # Fallback threshold
        for c_idx in tqdm(range(self.num_known_classes), desc="DOC Fit: Fitting Gaussians", leave=False):
            if c_idx not in scores_per_class or len(scores_per_class[c_idx]) < 2:
                print(f"  Warning: Insufficient scores ({len(scores_per_class.get(c_idx, []))}) for class {c_idx}. Using default threshold {default_threshold}.")
                self.gaussian_params[c_idx] = (0.5, 0.5) # Placeholder params
                self.class_thresholds[c_idx] = default_threshold
                continue

            # Filter scores (optional, e.g., remove very low scores if desired)
            # scores_np = np.array([s for s in scores_per_class[c_idx] if s > 0.1]) # Example filtering
            scores_np = np.array(scores_per_class[c_idx])
            if len(scores_np) < 2:
                 print(f"  Warning: Insufficient scores ({len(scores_np)}) after filtering for class {c_idx}. Using default threshold {default_threshold}.")
                 self.gaussian_params[c_idx] = (0.5, 0.5)
                 self.class_thresholds[c_idx] = default_threshold
                 continue

            # Mirror scores around 1.0 to approximate negative class scores
            mirrored_scores = 1.0 + (1.0 - scores_np)
            combined_scores = np.concatenate([scores_np, mirrored_scores])

            # Fit Gaussian (normal distribution) to the combined scores
            try:
                mean_combined, std_combined = norm.fit(combined_scores)
                # Ensure std deviation is positive and non-zero
                std_combined = max(std_combined, 1e-6)

                # Calculate threshold: max(0.5, 1.0 - k * sigma)
                threshold = max(0.5, 1.0 - self.k_sigma * std_combined)

                self.gaussian_params[c_idx] = (mean_combined, std_combined) # Store params of combined dist
                self.class_thresholds[c_idx] = threshold
                # print(f"  Class {c_idx}: Threshold={threshold:.4f} (Std={std_combined:.4f})") # Optional verbose
            except Exception as e:
                 print(f"  Warning: Gaussian fit exception for class {c_idx}: {e}. Using default threshold {default_threshold}.")
                 self.gaussian_params[c_idx] = (0.5, 0.5)
                 self.class_thresholds[c_idx] = default_threshold

        fit_duration = time.time() - fit_start_time
        print(f"[DOCOSR Fit] Gaussian fitting complete in {fit_duration:.2f}s. Thresholds calculated for {len(self.class_thresholds)} classes.")

    def predict(self, dataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Fit models if not already done
        if not self.class_thresholds:
            print("[DOCOSR Predict] Class thresholds not calculated. Fitting now...")
            try:
                train_loader = self.datamodule.train_dataloader()
                if train_loader is None: raise ValueError("Training dataloader required for DOC fitting.")
                self._fit_gaussian_models(train_loader)
                if not self.class_thresholds: # Check if fitting failed
                    print("Error: DOC Gaussian fitting failed. Using default threshold 0.5 for all classes.")
                    # Populate with default threshold if fitting failed entirely
                    self.class_thresholds = {i: 0.5 for i in range(self.num_known_classes)}
            except Exception as fit_e:
                 print(f"Error during DOC fitting: {fit_e}. Using default threshold 0.5.")
                 self.class_thresholds = {i: 0.5 for i in range(self.num_known_classes)}

        self.model.eval().to(self.device)
        all_sigmoid_scores_list, all_preds_final_list, all_labels_original_list = [], [], []
        all_max_scores_list, all_embeddings_list = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (DOC OSR)", leave=False):
                ids = batch['input_ids'].to(self.device)
                attn = batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids')
                if tok is not None: tok = tok.to(self.device)
                labels_orig = batch['label'].cpu().numpy() # Original labels

                # Get DOC model outputs
                logits, embeddings = self.model(ids, attn, tok)
                sigmoid_scores_batch = torch.sigmoid(logits) # Shape: (batch, num_known_classes)

                # Determine predicted class (mapped index) and max score for each sample
                max_scores_batch, preds_mapped = torch.max(sigmoid_scores_batch, dim=1)

                # Convert to numpy
                pred_indices_np = preds_mapped.cpu().numpy()
                max_scores_np = max_scores_batch.cpu().numpy()
                sigmoid_scores_np = sigmoid_scores_batch.cpu().numpy()

                # Apply class-specific thresholds
                batch_preds_final = np.full_like(pred_indices_np, -1, dtype=int) # Default unknown
                accept_mask = np.zeros_like(pred_indices_np, dtype=bool)
                for i in range(len(labels_orig)):
                     pred_mapped_idx = pred_indices_np[i]
                     # Get threshold for the predicted class, fallback to 0.5 if class missing
                     threshold = self.class_thresholds.get(pred_mapped_idx, 0.5)
                     # Accept if max score >= threshold
                     if max_scores_np[i] >= threshold:
                         accept_mask[i] = True

                # Map accepted predictions back to original class indices
                if np.any(accept_mask):
                    accepted_preds_mapped = pred_indices_np[accept_mask]
                    mapped_original_preds = self._map_preds_to_original(accepted_preds_mapped)
                    batch_preds_final[accept_mask] = mapped_original_preds

                # Store results
                all_sigmoid_scores_list.append(sigmoid_scores_np)
                all_max_scores_list.extend(max_scores_np)
                all_preds_final_list.extend(batch_preds_final)
                all_labels_original_list.extend(labels_orig)
                all_embeddings_list.append(embeddings.cpu().numpy())

        # Concatenate results
        all_sigmoid_scores = np.concatenate(all_sigmoid_scores_list) if all_sigmoid_scores_list else np.empty((0, self.num_known_classes))
        all_max_scores = np.array(all_max_scores_list)
        all_preds_final = np.array(all_preds_final_list)
        all_labels_original = np.array(all_labels_original_list)
        all_embeddings = np.concatenate(all_embeddings_list) if all_embeddings_list else np.empty((0, self.model.config.hidden_size))

        # Scores for ranking: Use negative max sigmoid score. Higher score = more likely unknown.
        scores_for_ranking = -all_max_scores

        # Return: raw scores (sigmoid), final predictions, original labels, ranking scores (-max sigmoid), embeddings
        return all_sigmoid_scores, all_preds_final, all_labels_original, scores_for_ranking, all_embeddings

    def visualize(self, results):
        """Adds max sigmoid score distribution plot."""
        super().visualize(results) # Call common visualizations
        print(f"[{self.__class__.__name__} Visualize] Generating specific plots...")
        method_name_for_file = self.args.osr_method if self.args.osr_method != 'all' else self.__class__.__name__.lower().replace("osr","")
        base_filename = os.path.join(RESULTS_DIR, f"{method_name_for_file}_osr_{self.args.dataset}_{self.args.seen_class_ratio:.2f}")

        # Get max scores (invert the ranking score)
        max_scores = -results.get('scores_for_ranking', np.array([]))

        if max_scores.size > 0:
            labels_np = np.array(results['labels'])
            unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)
            plot_data = pd.DataFrame({'Max Sigmoid Score': max_scores, 'Is Known': ~unknown_labels_mask})

            try:
                plt.figure(figsize=(8, 5))
                sns.histplot(data=plot_data, x='Max Sigmoid Score', hue='Is Known', kde=True,
                             stat='density', common_norm=False, bins=50, palette=['tomato', 'cornflowerblue'])

                # Indicate average threshold (useful overview, but remember it's class-specific)
                if self.class_thresholds:
                    avg_threshold = np.mean(list(self.class_thresholds.values()))
                    plt.axvline(avg_threshold, color='darkgreen', linestyle=':', linewidth=1.5, label=f'Avg Threshold ≈ {avg_threshold:.3f}')

                plt.title('DOC Max Sigmoid Score Distribution', fontsize=13)
                plt.xlabel('Max Sigmoid Score per Sample', fontsize=11)
                plt.ylabel('Density', fontsize=11)
                plt.legend(title='Ground Truth', fontsize=10)
                plt.grid(True, linestyle=':', alpha=0.5)
                plt.tight_layout()
                plot_filename = f"{base_filename}_max_score_dist.png"
                plt.savefig(plot_filename)
                plt.close()
                print(f"  Max sigmoid score distribution plot saved to: {plot_filename}")
            except Exception as e:
                print(f"  Error generating max score distribution plot: {e}")
                plt.close()
        else:
            print("  Skipping max score distribution plot (no ranking scores found).")

        # Optional: Could add Z-Score plot here if needed, using self.gaussian_params
        print(f"[{self.__class__.__name__} Visualize] Finished specific plots.")


class ADBOSR(OSRAlgorithm):
    """OSR algorithm using Adaptive Decision Boundaries (ADB)."""
    DEFAULT_DISTANCE_METRIC = 'cosine'

    def __init__(self, model, datamodule, args):
        if not isinstance(model, RobertaADB):
            raise TypeError(f"ADBOSR requires a RobertaADB model, but got {type(model).__name__}.")
        super().__init__(model, datamodule, args)
        # Use provided distance metric or default
        self.distance_metric = args.param_adb_distance if args.param_adb_distance else self.DEFAULT_DISTANCE_METRIC
        print(f"[ADBOSR Init] Using distance metric: {self.distance_metric}")
        if not hasattr(self.model, 'centers') or not hasattr(self.model, 'get_radii'):
             raise AttributeError("ADBOSR requires the model to have 'centers' attribute and 'get_radii' method.")

    def _compute_distances(self, features_norm: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Computes distances between normalized features and class centers."""
        # Ensure centers are normalized for cosine distance/similarity
        centers_norm = F.normalize(centers, p=2, dim=-1)

        if self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            similarity = torch.matmul(features_norm, centers_norm.t())
            similarity = torch.clamp(similarity, -1.0 + 1e-7, 1.0 - 1e-7) # Clamp for stability
            return 1.0 - similarity
        elif self.distance_metric == 'euclidean':
             # Calculate Euclidean distance between normalized features and normalized centers
             # Note: ADB paper primarily uses cosine distance with normalized features/centers
             return torch.cdist(features_norm, centers_norm, p=2)
        else:
            raise ValueError(f"Unknown distance metric specified for ADB: {self.distance_metric}")

    def predict(self, dataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval().to(self.device)
        all_features_list, all_distances_list, all_preds_final_list = [], [], []
        all_labels_original_list, all_min_distances_list = [], []

        # Get centers and radii once
        centers = self.model.centers.detach()
        radii = self.model.get_radii().detach() # Positive radii

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (ADB OSR)", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is not None: token_type_ids = token_type_ids.to(self.device)
                labels_orig = batch['label'].cpu().numpy() # Original labels

                # Get normalized features and similarity-based logits from ADB model
                logits, features_norm = self.model(input_ids, attention_mask, token_type_ids)
                # Prediction based on highest similarity (closest center in cosine space if normalized)
                preds_mapped = torch.argmax(logits, dim=1) # Mapped indices (0..N-1)

                # Compute distances from features to all centers
                distances_batch = self._compute_distances(features_norm, centers) # Shape: (batch, num_classes)
                # Find the minimum distance for each sample (distance to the *closest* center)
                min_distances_batch, _ = torch.min(distances_batch, dim=1)

                # Get the radius associated with the *predicted* class (closest center)
                closest_radii_batch = radii[preds_mapped]

                # Convert to numpy
                pred_indices_np = preds_mapped.cpu().numpy()
                min_distances_np = min_distances_batch.cpu().numpy()
                closest_radii_np = closest_radii_batch.cpu().numpy()
                distances_np = distances_batch.cpu().numpy() # All distances

                # Apply ADB threshold: Accept if min_distance <= radius_of_predicted_class
                batch_preds_final = np.full_like(pred_indices_np, -1, dtype=int) # Default unknown
                # If the distance to the closest center is within that center's radius, accept the prediction
                accept_mask = min_distances_np <= closest_radii_np

                # Map accepted predictions back to original class indices
                if np.any(accept_mask):
                    accepted_preds_mapped = pred_indices_np[accept_mask]
                    mapped_original_preds = self._map_preds_to_original(accepted_preds_mapped)
                    batch_preds_final[accept_mask] = mapped_original_preds

                # Store results
                all_features_list.append(features_norm.cpu().numpy()) # Normalized features
                all_distances_list.append(distances_np) # All distances to centers
                all_preds_final_list.extend(batch_preds_final)
                all_labels_original_list.extend(labels_orig)
                all_min_distances_list.extend(min_distances_np) # Min distance per sample

        # Concatenate results
        all_features = np.concatenate(all_features_list) if all_features_list else np.empty((0, self.model.feat_dim))
        all_distances = np.concatenate(all_distances_list) if all_distances_list else np.empty((0, self.num_known_classes))
        all_preds_final = np.array(all_preds_final_list)
        all_labels_original = np.array(all_labels_original_list)
        all_min_distances = np.array(all_min_distances_list)

        # Scores for ranking: Use the minimum distance to any center. Higher distance = more unknown.
        scores_for_ranking = all_min_distances

        # Return: raw scores (all distances), final predictions, original labels, ranking scores (min distance), embeddings (features)
        return all_distances, all_preds_final, all_labels_original, scores_for_ranking, all_features

    def visualize(self, results: dict):
        """Adds minimum distance distribution plot."""
        super().visualize(results) # Call common visualizations (CM, ROC, t-SNE - t-SNE will plot centers)
        print(f"[{self.__class__.__name__} Visualize] Generating specific plots...")
        method_name_for_file = self.args.osr_method if self.args.osr_method != 'all' else self.__class__.__name__.lower().replace("osr","")
        base_filename = os.path.join(RESULTS_DIR, f"{method_name_for_file}_osr_{self.args.dataset}_{self.args.seen_class_ratio:.2f}")

        # Use the ranking score which is the minimum distance
        min_distances = results.get('scores_for_ranking')

        if min_distances is not None and len(min_distances) > 0:
            labels_np = np.array(results['labels'])
            unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)
            plot_data = pd.DataFrame({'Min Distance': min_distances, 'Is Known': ~unknown_labels_mask})

            try:
                plt.figure(figsize=(8, 5))
                sns.histplot(data=plot_data, x='Min Distance', hue='Is Known', kde=True,
                             stat='density', common_norm=False, bins=50, palette=['tomato', 'cornflowerblue'])

                # Indicate average radius
                if hasattr(self.model, 'get_radii'):
                    mean_radius = self.model.get_radii().detach().mean().item()
                    plt.axvline(mean_radius, color='darkgreen', linestyle=':', linewidth=1.5, label=f'Avg Radius ≈ {mean_radius:.3f}')

                plt.title(f'ADB Minimum Distance Distribution ({self.distance_metric.capitalize()})', fontsize=13)
                plt.xlabel(f'Minimum {self.distance_metric.capitalize()} Distance to Any Center', fontsize=11)
                plt.ylabel('Density', fontsize=11)
                plt.legend(title='Ground Truth', fontsize=10)
                plt.grid(True, linestyle=':', alpha=0.5)
                plt.tight_layout()
                plot_filename = f"{base_filename}_min_distance_dist.png"
                plt.savefig(plot_filename)
                plt.close()
                print(f"  Minimum distance distribution plot saved to: {plot_filename}")
            except Exception as e:
                print(f"  Error generating min distance distribution plot: {e}")
                plt.close()
        else:
            print("  Skipping min distance distribution plot (no ranking scores found).")
        print(f"[{self.__class__.__name__} Visualize] Finished specific plots.")


# =============================================================================
# Training and Evaluation Orchestration Functions
# =============================================================================

class FinalizeLoggerCallback(Callback):
    """Callback to ensure logger is finalized properly."""
    def _finalize_logger(self, trainer, status: str):
        if trainer.logger:
            # Check for different logger types and call finalize if available
            if hasattr(trainer.logger, 'finalize'):
                trainer.logger.finalize(status)
                print(f"Logger ({type(trainer.logger).__name__}) finalized with status: {status}")
            elif hasattr(trainer.logger, 'save'): # CSVLogger might just need save
                 trainer.logger.save()
                 print(f"Logger ({type(trainer.logger).__name__}) saved.")
            # Add other logger finalization methods if needed

    def on_train_end(self, trainer, pl_module):
        self._finalize_logger(trainer, "success")

    def on_exception(self, trainer, pl_module, exception):
         # Also finalize on interruption/error
         print(f"Exception during training: {exception}")
         self._finalize_logger(trainer, "interrupted")

    def on_fit_end(self, trainer, pl_module):
         # Fallback in case on_train_end isn't reached but fit finishes
         # Check if logger still exists and might need finalizing
         if trainer.logger and hasattr(trainer.logger, 'finalize') and not getattr(trainer.logger, '_finalized', False):
              print("Finalizing logger in on_fit_end (fallback).")
              self._finalize_logger(trainer, "finished")


def _calculate_scheduler_steps(datamodule: DataModule, args: argparse.Namespace) -> tuple[int, int]:
    """Calculates total steps and warmup steps for the LR scheduler."""
    total_steps = 0
    warmup_steps = 0
    try:
        # Estimate steps based on training dataloader and epochs
        train_loader = datamodule.train_dataloader()
        if train_loader is not None and len(train_loader) > 0:
            train_batches = len(train_loader)
            total_steps = train_batches * args.epochs
            # Calculate warmup steps based on ratio, capped by max_warmup_steps
            warmup_steps = min(int(args.warmup_ratio * total_steps), args.max_warmup_steps)
            print(f"Scheduler steps calculated: Total={total_steps}, Warmup={warmup_steps} (Train batches={train_batches})")
        else:
             raise ValueError("Train dataloader is empty or None.")

    except Exception as e:
        print(f"Warning: Could not accurately determine scheduler steps: {e}. Using fallback values.")
        total_steps = DEFAULT_FALLBACK_TOTAL_STEPS
        warmup_steps = min(int(args.warmup_ratio * total_steps), args.max_warmup_steps)
        print(f"Using fallback scheduler steps: Total={total_steps}, Warmup={warmup_steps}")

    if total_steps <= 0:
        print(f"Warning: Calculated total_steps ({total_steps}) is not positive. Disabling scheduler.")
        return 0, 0 # Return 0 to indicate no scheduler

    return total_steps, warmup_steps

def _setup_trainer(model: pl.LightningModule, datamodule: DataModule, args: argparse.Namespace, run_identifier: str) -> tuple[pl.Trainer, str, str | None]:
    """Sets up the PyTorch Lightning Trainer and associated callbacks/loggers."""
    print(f"\n--- Setting up Trainer for Run: {run_identifier} ---")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.dataset}_{run_identifier}_{args.seen_class_ratio:.2f}_{timestamp}"

    # Define output directories
    output_dir = os.path.join(CHECKPOINTS_DIR, run_name)
    log_dir = os.path.join(LOGS_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print(f"  Checkpoints will be saved to: {output_dir}")
    print(f"  Logs will be saved to: {log_dir}")

    # Calculate scheduler steps and update model if needed
    total_steps, warmup_steps = _calculate_scheduler_steps(datamodule, args)
    if hasattr(model, 'total_steps'): model.total_steps = total_steps
    if hasattr(model, 'warmup_steps'): model.warmup_steps = warmup_steps

    # --- Callbacks ---
    # Monitor validation loss for checkpointing and early stopping
    monitor_metric = "val_loss"; monitor_mode = "min"
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=f"{model.__class__.__name__}-{{epoch:02d}}-{{{monitor_metric}:.4f}}",
        save_top_k=1,
        verbose=False, # Quieter output
        monitor=monitor_metric,
        mode=monitor_mode
    )
    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_delta,
        verbose=True,
        mode=monitor_mode
    )
    finalize_logger_callback = FinalizeLoggerCallback() # Ensures logger closes
    callbacks = [checkpoint_callback, early_stopping_callback, finalize_logger_callback]

    # --- Logger ---
    try:
        # Prefer TensorBoard if available
        logger = TensorBoardLogger(save_dir=LOGS_DIR, name=run_name, version="")
        print(f"  Using TensorBoardLogger in: {logger.log_dir}")
    except ImportError:
        print("  TensorBoard not available. Using CSVLogger.")
        logger = CSVLogger(save_dir=LOGS_DIR, name=run_name, version="")
        print(f"  Using CSVLogger in: {logger.log_dir}")

    # --- Trainer Configuration ---
    use_gpu = args.force_gpu or torch.cuda.is_available()
    precision = "16-mixed" if use_gpu else 32 # Mixed precision on GPU if available

    trainer_kwargs = {
        "max_epochs": args.epochs,
        "callbacks": callbacks,
        "logger": logger,
        "log_every_n_steps": 50, # Log roughly every 50 steps
        "precision": precision,
        "gradient_clip_val": args.gradient_clip_val,
        "deterministic": "warn", # Warn if non-deterministic operations are used
        "benchmark": False if args.random_seed else True, # Benchmark if seed not fixed
        "enable_progress_bar": True, # Show progress bar during training
        "num_sanity_val_steps": 0, # Disable sanity check loops for speed
    }

    if use_gpu:
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = [args.gpu_id]
        print(f"  Using GPU: {args.gpu_id} with precision: {precision}")
    else:
        print(f"  Using CPU with precision: {precision}")
        trainer_kwargs["accelerator"] = "cpu"
        trainer_kwargs["devices"] = 1 # Or os.cpu_count() if desired

    # Create Trainer instance
    trainer = pl.Trainer(**trainer_kwargs)

    return trainer, output_dir, checkpoint_callback # Return callback to get best path later

def train_model(model: pl.LightningModule, datamodule: DataModule, args: argparse.Namespace, run_identifier: str) -> str | None:
    """Trains a PyTorch Lightning model and returns the best checkpoint path."""
    trainer, output_dir, checkpoint_callback = _setup_trainer(model, datamodule, args, run_identifier)

    print(f"--- Starting Training ({run_identifier}) for {args.epochs} epochs ---")
    best_checkpoint_path = None
    try:
        trainer.fit(model, datamodule=datamodule)
        # Get the path to the best model saved by the callback
        best_checkpoint_path = checkpoint_callback.best_model_path
        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
             print(f"Training finished successfully. Best model saved at: {best_checkpoint_path}")
        else:
             print("Warning: Training finished, but best_model_path not found or invalid.")
             best_checkpoint_path = None # Ensure it's None if invalid

    except Exception as e:
        print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Error during training run '{run_identifier}': {e}")
        import traceback; traceback.print_exc()
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Attempt to retrieve the best checkpoint saved so far, even if training failed
        best_checkpoint_path = checkpoint_callback.best_model_path
        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            print(f"Attempting to use best checkpoint saved before error: {best_checkpoint_path}")
        else:
            print("Could not find a valid checkpoint after training error.")
            best_checkpoint_path = None

    # Final check for a valid checkpoint file
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        return best_checkpoint_path
    else:
        # If no best path, try to find *any* checkpoint in the directory as a last resort
        print(f"Warning: Best checkpoint path ('{best_checkpoint_path}') not valid. Searching for any '.ckpt' file in {output_dir}")
        try:
            existing_ckpts = [f for f in os.listdir(output_dir) if f.endswith('.ckpt')]
            if existing_ckpts:
                # Try to find one with epoch and val_loss in the name
                best_guess = next((os.path.join(output_dir, ckpt) for ckpt in sorted(existing_ckpts) if 'epoch' in ckpt and 'val_loss' in ckpt), None)
                if best_guess:
                    print(f"  Found a potential checkpoint based on naming: {best_guess}")
                    return best_guess
                else:
                    # Otherwise, return the last checkpoint created (usually highest epoch)
                    last_ckpt = os.path.join(output_dir, sorted(existing_ckpts)[-1])
                    print(f"  Found other checkpoints: {existing_ckpts}. Returning the last one: {last_ckpt}")
                    return last_ckpt
            else:
                 print(f"  No checkpoint files found in {output_dir}.")
                 return None # Truly no checkpoint found
        except FileNotFoundError:
             print(f"  Checkpoint directory {output_dir} not found.")
             return None
        except Exception as find_e:
             print(f"  Error while searching for fallback checkpoint: {find_e}")
             return None


# --- OSR Evaluation Wrappers ---

# Map OSR method names to required model classes and OSR algorithm classes
MODEL_CLASS_MAP = {
    'standard': RobertaClassifier, # Used by threshold, openmax
    'crosr': RobertaAutoencoder,
    'doc': DOCRobertaClassifier,
    'adb': RobertaADB
}
OSR_ALGORITHM_MAP = {
    'threshold': ThresholdOSR,
    'openmax': OpenMaxOSR,
    'crosr': CROSROSR,
    'doc': DOCOSR,
    'adb': ADBOSR
}
# Methods that require a specific model architecture different from the standard classifier
METHODS_NEEDING_SPECIAL_MODEL = ['crosr', 'doc', 'adb']
# Methods that inherently require retraining the model for hyperparameter tuning
# (because the parameters affect the training process itself)
METHODS_NEEDING_RETRAINING_PER_TRIAL = ['crosr', 'doc', 'adb']
# Methods that can potentially reuse a pre-trained standard classifier
METHODS_USING_STANDARD_MODEL = ['threshold', 'openmax']


def _initialize_model_for_eval(
    target_model_class: type[BaseRobertaModule],
    args_for_init: argparse.Namespace,
    num_classes: int
) -> BaseRobertaModule:
    """Initializes a model instance based on the target class and args."""
    print(f"    Initializing model: {target_model_class.__name__}")
    init_kwargs = {
        'model_name': args_for_init.model,
        'num_classes': num_classes,
        'weight_decay': args_for_init.weight_decay,
        'warmup_steps': args_for_init.max_warmup_steps, # Passed for potential scheduler use
        'total_steps': 0 # Will be calculated later if needed
    }
    # Add class-specific required arguments
    if target_model_class == RobertaClassifier:
        init_kwargs['learning_rate'] = args_for_init.lr
    elif target_model_class == RobertaAutoencoder:
        init_kwargs['learning_rate'] = args_for_init.lr
        init_kwargs['reconstruction_weight'] = args_for_init.param_crosr_recon_weight
        init_kwargs['latent_dim'] = 256 # Example: Make this configurable if needed
    elif target_model_class == DOCRobertaClassifier:
        init_kwargs['learning_rate'] = args_for_init.lr
    elif target_model_class == RobertaADB:
        init_kwargs['learning_rate'] = args_for_init.lr # Backbone LR
        init_kwargs['lr_adb'] = args_for_init.lr_adb # ADB params LR
        init_kwargs['param_adb_delta'] = args_for_init.param_adb_delta
        init_kwargs['param_adb_alpha'] = args_for_init.param_adb_alpha
        init_kwargs['adb_freeze_backbone'] = args_for_init.adb_freeze_backbone
    else:
        # Fallback for standard if somehow called directly
        init_kwargs['learning_rate'] = args_for_init.lr

    try:
        model_instance = target_model_class(**init_kwargs)
        print(f"    Model {target_model_class.__name__} initialized successfully.")
        return model_instance
    except Exception as e:
        print(f"    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"    Error initializing model {target_model_class.__name__}: {e}")
        import traceback; traceback.print_exc()
        print(f"    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise # Re-raise the exception to signal failure


def _cleanup_trial_artifacts(run_identifier: str, checkpoint_dir: str | None, args: argparse.Namespace):
    """Removes checkpoint and log directories associated with an Optuna trial."""
    print(f"    Cleaning up artifacts for trial run: {run_identifier}")
    # Remove checkpoint directory
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        try:
            shutil.rmtree(checkpoint_dir)
            print(f"      Removed trial checkpoint directory: {checkpoint_dir}")
        except OSError as e:
            print(f"      Warning: Could not remove trial checkpoint directory {checkpoint_dir}: {e}")

    # Construct log directory path based on naming convention
    log_dir_path = os.path.join(LOGS_DIR, f"{args.dataset}_{run_identifier}_{args.seen_class_ratio:.2f}_{'*'}") # Use wildcard for timestamp
    # Find matching log directories
    import glob
    potential_log_dirs = glob.glob(log_dir_path)
    for log_dir in potential_log_dirs:
         if os.path.exists(log_dir):
              try:
                  shutil.rmtree(log_dir)
                  print(f"      Removed trial log directory: {log_dir}")
              except OSError as e:
                  print(f"      Warning: Could not remove trial log directory {log_dir}: {e}")


def _handle_parameter_search(
    method_name: str,
    datamodule: DataModule,
    args: argparse.Namespace,
    osr_algorithm_class: type[OSRAlgorithm],
    base_model_for_eval_only: pl.LightningModule | None # Only used if not retraining
) -> tuple[dict, bool]:
    """Performs Optuna hyperparameter search."""
    print(f"\n--- Starting Optuna Search for {method_name.upper()} ---")
    print(f"  Optimizing metric: {args.tuning_metric} over {args.n_trials} trials.")
    tuner = OptunaHyperparameterTuner(method_name, datamodule, args)
    needs_final_training = False
    best_params = {}
    best_trial_results = {}

    target_model_class_name = method_name if method_name in METHODS_NEEDING_SPECIAL_MODEL else 'standard'
    target_model_class = MODEL_CLASS_MAP[target_model_class_name]

    if method_name in METHODS_NEEDING_RETRAINING_PER_TRIAL:
        print("  Tuning Mode: Retraining model from scratch for each trial.")
        needs_final_training = True # Always need final training after tuning

        def train_and_evaluate_trial(trial_args: argparse.Namespace, trial_num: int) -> tuple[dict, float]:
            """Objective function for Optuna: trains and evaluates a model."""
            trial_start_time = time.time()
            trial_run_id = f"trial_{method_name}_{trial_num}_{datetime.now().strftime('%H%M%S%f')}"
            print(f"\n  --- Starting Optuna Trial {trial_num}/{args.n_trials} (Run ID: {trial_run_id}) ---")
            print(f"    Params: { {k: v for k, v in vars(trial_args).items() if k.startswith('param_') or k in ['lr', 'lr_adb', 'weight_decay', 'adb_freeze_backbone']} }") # Log tuned params

            trial_model = None
            checkpoint_path = None
            trial_ckpt_dir = None
            results = {}
            score = -1e9 # Default score for failure

            try:
                # 1. Initialize Model for Trial
                trial_model = _initialize_model_for_eval(target_model_class, trial_args, datamodule.num_seen_classes)

                # 2. Train Model for Trial (potentially fewer epochs)
                tuning_epochs = max(3, args.epochs // 2) # Use fewer epochs for speed
                print(f"    Training trial model for {tuning_epochs} epochs...")
                trial_args_train = copy.deepcopy(trial_args)
                trial_args_train.epochs = tuning_epochs
                # Pass the unique trial ID for logging/checkpointing
                checkpoint_path = train_model(trial_model, datamodule, trial_args_train, run_identifier=trial_run_id)
                trial_ckpt_dir = os.path.dirname(checkpoint_path) if checkpoint_path else None # Get directory for cleanup

                if checkpoint_path is None or not os.path.exists(checkpoint_path):
                    print("    Trial Training Failed: No valid checkpoint found.")
                    raise RuntimeError("Trial training produced no checkpoint.") # Go to finally for cleanup

                # 3. Load Best Model from Trial Checkpoint
                print(f"    Loading best model from trial checkpoint: {checkpoint_path}")
                device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu')
                # Ensure loading uses the correct class
                trained_trial_model = target_model_class.load_from_checkpoint(checkpoint_path, map_location=device)

                # 4. Evaluate Trial Model
                print("    Evaluating trial model...")
                evaluator = osr_algorithm_class(trained_trial_model, datamodule, trial_args)
                # Use test dataloader for evaluation during tuning
                results = evaluator.evaluate(datamodule.test_dataloader())

                # 5. Get Score
                current_score = results.get(args.tuning_metric)
                if current_score is not None and np.isfinite(current_score):
                    score = float(current_score)
                else:
                    print(f"    Warning: Tuning metric '{args.tuning_metric}' not found or invalid in results. Using failure score.")
                    score = -1e9 # Penalize failure

            except Exception as e:
                print(f"    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"    Error during Optuna trial {trial_num}: {e}")
                import traceback; traceback.print_exc()
                print(f"    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                score = -1e9 # Ensure failure score is returned
                results = {'error': str(e)} # Store error in results

            finally:
                # 6. Cleanup Trial Artifacts (always run)
                _cleanup_trial_artifacts(trial_run_id, trial_ckpt_dir, args)
                trial_duration = time.time() - trial_start_time
                print(f"  --- Trial {trial_num} Completed in {trial_duration:.2f}s. Score ({args.tuning_metric}): {score:.4f if score > -1e8 else 'Fail'} ---")

            return results, score # Return results dict and score

        # Run the tuning process
        best_params, best_trial_results = tuner.tune(tuner._objective_with_retraining, train_and_evaluate_trial)

    else: # Tuning methods that don't require retraining (threshold, openmax)
        print("  Tuning Mode: Evaluating parameters using the pre-trained base model.")
        needs_final_training = False # No final training needed as base model is used

        if base_model_for_eval_only is None:
             raise ValueError("A pre-trained base model must be provided for evaluate-only tuning.")
        # Ensure the base model is of the correct type (standard classifier)
        if not isinstance(base_model_for_eval_only, MODEL_CLASS_MAP['standard']):
             print(f"Warning: Expected standard model for {method_name} tuning, but got {type(base_model_for_eval_only).__name__}. Results may be suboptimal.")
             # Proceed, but be aware the base model might not be ideal

        def evaluate_trial_no_retraining(trial_args: argparse.Namespace, trial_num: int) -> tuple[dict, float]:
            """Objective function: evaluates parameters without retraining."""
            trial_start_time = time.time()
            print(f"\n  --- Evaluating Optuna Trial {trial_num}/{args.n_trials} ---")
            print(f"    Params: { {k: v for k, v in vars(trial_args).items() if k.startswith('param_')} }") # Log tuned params

            results = {}
            score = -1e9
            try:
                # Evaluate using the provided base model and trial hyperparameters
                evaluator = osr_algorithm_class(base_model_for_eval_only, datamodule, trial_args)
                results = evaluator.evaluate(datamodule.test_dataloader())

                current_score = results.get(args.tuning_metric)
                if current_score is not None and np.isfinite(current_score):
                    score = float(current_score)
                else:
                    print(f"    Warning: Tuning metric '{args.tuning_metric}' not found or invalid. Using failure score.")
                    score = -1e9
            except Exception as e:
                print(f"    Error during evaluate-only trial {trial_num}: {e}")
                score = -1e9
                results = {'error': str(e)}

            trial_duration = time.time() - trial_start_time
            print(f"  --- Trial {trial_num} Evaluation Completed in {trial_duration:.2f}s. Score ({args.tuning_metric}): {score:.4f if score > -1e8 else 'Fail'} ---")
            return results, score

        best_params, best_trial_results = tuner.tune(tuner._objective_evaluate_only, evaluate_trial_no_retraining)

    print(f"\n--- Optuna Search for {method_name.upper()} Finished ---")
    if best_params:
        print("  Best parameters found:")
        for name, value in best_params.items(): print(f"    {name}: {value}")
        # Apply best parameters back to the main args namespace for the final run
        print("  Applying best parameters to args for final evaluation...")
        for name, value in best_params.items():
            if hasattr(args, name):
                setattr(args, name, value)
            else:
                print(f"    Warning: Best param '{name}' not found in args namespace.")
    else:
        print("  Optuna search did not find any successful trials or parameters.")
        # Decide how to proceed: use defaults, raise error? For now, defaults will be used later.

    return best_params, needs_final_training


def _handle_parameter_loading(method_name: str, args: argparse.Namespace) -> bool:
    """Loads best parameters from file or uses defaults. Determines if retraining is needed."""
    print(f"\n--- Loading Parameters for {method_name.upper()} (Parameter Search Disabled) ---")
    needs_final_training = False
    loaded_params = load_best_params(method_name, args.dataset, args.seen_class_ratio)

    if loaded_params:
        param_source = "Loaded from file"
    else:
        param_source = "Using default values"
        loaded_params = get_default_best_params(method_name) # Get defaults if file load failed

    print(f"  Source: {param_source}")
    print(f"  Applying parameters for final {method_name.upper()} evaluation:")
    if not loaded_params:
         print("    No specific parameters found or loaded.")
    else:
        for name, value in loaded_params.items():
            if hasattr(args, name):
                setattr(args, name, value)
                print(f"    {name}: {value}")
            else:
                # This indicates a mismatch between saved params and current args definition
                print(f"    Warning: Loaded/default parameter '{name}' does not exist in current args. Skipping.")

    # Check if the method requires a special model type. If so, retraining might be
    # necessary if the initially provided `base_model` isn't of the correct type.
    # This check happens *outside* this function, in _prepare_evaluation.
    # This function primarily handles loading/setting the parameter values in `args`.
    return needs_final_training # Currently always False, check happens later


def _prepare_evaluation(
    method_name: str,
    base_model: pl.LightningModule, # The initially trained model (usually standard classifier)
    datamodule: DataModule,
    args: argparse.Namespace,
    osr_algorithm_class: type[OSRAlgorithm]
) -> pl.LightningModule:
    """
    Handles parameter setup (tuning or loading) and potential model retraining
    required for a specific OSR method evaluation.

    Returns:
        The model instance (potentially retrained) ready for final evaluation.
    """
    print(f"\n===== Preparing for {method_name.upper()} OSR Evaluation =====")
    model_for_final_eval = None
    needs_final_training = False
    best_params_from_tuning = {}

    # Determine the required model class for this OSR method
    target_model_class_name = method_name if method_name in METHODS_NEEDING_SPECIAL_MODEL else 'standard'
    target_model_class = MODEL_CLASS_MAP[target_model_class_name]
    print(f"  Method '{method_name}' requires model type: {target_model_class.__name__}")

    # --- Step 1: Set Parameters (Tune or Load) ---
    if args.parameter_search and (args.osr_method == 'all' or method_name == args.osr_method):
        # Perform parameter search
        # Pass the base_model only if tuning doesn't involve retraining
        base_model_for_tuning = base_model if method_name not in METHODS_NEEDING_RETRAINING_PER_TRIAL else None
        best_params_from_tuning, needs_retraining_after_tuning = _handle_parameter_search(
            method_name, datamodule, args, osr_algorithm_class, base_model_for_tuning
        )
        needs_final_training = needs_retraining_after_tuning
    else:
        # Load parameters from file or use defaults
        _ = _handle_parameter_loading(method_name, args)
        # Check if retraining is needed due to model type mismatch *after* loading params
        if method_name in METHODS_NEEDING_SPECIAL_MODEL and not isinstance(base_model, target_model_class):
            print(f"  Warning: Model type mismatch detected after loading parameters.")
            print(f"           Required: {target_model_class.__name__}, Provided base: {type(base_model).__name__}.")
            print(f"           Retraining is necessary for method '{method_name}'.")
            needs_final_training = True
        else:
             needs_final_training = False # No type mismatch, no retraining needed based on this check

    # --- Step 2: Train Final Model (if necessary) ---
    if needs_final_training:
        print(f"\n--- Training Final Model for {method_name.upper()} ---")
        # Use the potentially updated args (from tuning/loading)
        final_model_instance = _initialize_model_for_eval(target_model_class, args, datamodule.num_seen_classes)

        # Train the final model using the full number of epochs
        final_args = copy.deepcopy(args) # Use current args state
        final_args.epochs = args.epochs # Ensure full epochs are used
        final_run_id = f"final_{method_name}" # Identifier for this final training run

        final_checkpoint_path = train_model(final_model_instance, datamodule, final_args, run_identifier=final_run_id)

        if final_checkpoint_path is None or not os.path.exists(final_checkpoint_path):
            raise RuntimeError(f"Failed to train or find checkpoint for the final {method_name.upper()} model.")

        print(f"--- Loading Final Trained Model for {method_name.upper()} ---")
        print(f"  Checkpoint: {final_checkpoint_path}")
        try:
            device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu')
            # Load using the correct target class
            model_for_final_eval = target_model_class.load_from_checkpoint(final_checkpoint_path, map_location=device)
            print(f"  Final model ({target_model_class.__name__}) loaded successfully.")
        except Exception as load_e:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Error loading final checkpoint '{final_checkpoint_path}': {load_e}")
            import traceback; traceback.print_exc()
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            raise RuntimeError(f"Failed to load final checkpoint for {method_name}") from load_e
    else:
         # Use the initially provided base model
         print(f"\n--- Using Pre-trained Base Model for {method_name.upper()} ---")
         # Verify the base model type is compatible *again* (important sanity check)
         if method_name in METHODS_NEEDING_SPECIAL_MODEL and not isinstance(base_model, target_model_class):
             # This should ideally be caught earlier, but double-check
             raise TypeError(f"Method {method_name} requires model {target_model_class.__name__}, but the provided base model is {type(base_model).__name__} and retraining was skipped.")
         print(f"  Model Type: {type(base_model).__name__} (compatible)")
         model_for_final_eval = base_model

    print(f"===== Preparation for {method_name.upper()} Complete =====")
    return model_for_final_eval


# --- Wrappers for each OSR method evaluation ---

def evaluate_threshold_osr(base_model, datamodule, args, all_results):
    method_name = 'threshold'
    osr_class = OSR_ALGORITHM_MAP[method_name]
    try:
        model_for_eval = _prepare_evaluation(method_name, base_model, datamodule, args, osr_class)
        print(f"\n--- Running Final {method_name.upper()} Evaluation ---")
        evaluator = osr_class(model_for_eval, datamodule, args)
        results = evaluator.evaluate(datamodule.test_dataloader())
        evaluator.visualize(results)
        all_results[method_name] = results
        return results
    except Exception as e:
        print(f"\n!!!!! Error during {method_name.upper()} evaluation pipeline: {e} !!!!!")
        import traceback; traceback.print_exc()
        all_results[method_name] = {"error": str(e)}
        return all_results[method_name]

def evaluate_openmax_osr(base_model, datamodule, args, all_results):
    method_name = 'openmax'
    osr_class = OSR_ALGORITHM_MAP[method_name]
    try:
        model_for_eval = _prepare_evaluation(method_name, base_model, datamodule, args, osr_class)
        print(f"\n--- Running Final {method_name.upper()} Evaluation ---")
        evaluator = osr_class(model_for_eval, datamodule, args)
        results = evaluator.evaluate(datamodule.test_dataloader())
        evaluator.visualize(results)
        all_results[method_name] = results
        return results
    except Exception as e:
        print(f"\n!!!!! Error during {method_name.upper()} evaluation pipeline: {e} !!!!!")
        import traceback; traceback.print_exc()
        all_results[method_name] = {"error": str(e)}
        return all_results[method_name]

def evaluate_crosr_osr(base_model, datamodule, args, all_results):
    method_name = 'crosr'
    osr_class = OSR_ALGORITHM_MAP[method_name]
    try:
        model_for_eval = _prepare_evaluation(method_name, base_model, datamodule, args, osr_class)
        print(f"\n--- Running Final {method_name.upper()} Evaluation ---")
        evaluator = osr_class(model_for_eval, datamodule, args)
        results = evaluator.evaluate(datamodule.test_dataloader())
        evaluator.visualize(results)
        all_results[method_name] = results
        return results
    except Exception as e:
        print(f"\n!!!!! Error during {method_name.upper()} evaluation pipeline: {e} !!!!!")
        import traceback; traceback.print_exc()
        all_results[method_name] = {"error": str(e)}
        return all_results[method_name]

def evaluate_doc_osr(base_model, datamodule, args, all_results):
    method_name = 'doc'
    osr_class = OSR_ALGORITHM_MAP[method_name]
    try:
        model_for_eval = _prepare_evaluation(method_name, base_model, datamodule, args, osr_class)
        print(f"\n--- Running Final {method_name.upper()} Evaluation ---")
        evaluator = osr_class(model_for_eval, datamodule, args)
        results = evaluator.evaluate(datamodule.test_dataloader())
        evaluator.visualize(results)
        all_results[method_name] = results
        return results
    except Exception as e:
        print(f"\n!!!!! Error during {method_name.upper()} evaluation pipeline: {e} !!!!!")
        import traceback; traceback.print_exc()
        all_results[method_name] = {"error": str(e)}
        return all_results[method_name]

def evaluate_adb_osr(base_model, datamodule, args, all_results):
    method_name = 'adb'
    osr_class = OSR_ALGORITHM_MAP[method_name]
    try:
        model_for_eval = _prepare_evaluation(method_name, base_model, datamodule, args, osr_class)
        print(f"\n--- Running Final {method_name.upper()} Evaluation ---")
        evaluator = osr_class(model_for_eval, datamodule, args)
        results = evaluator.evaluate(datamodule.test_dataloader())
        evaluator.visualize(results)
        all_results[method_name] = results
        return results
    except Exception as e:
        print(f"\n!!!!! Error during {method_name.upper()} evaluation pipeline: {e} !!!!!")
        import traceback; traceback.print_exc()
        all_results[method_name] = {"error": str(e)}
        return all_results[method_name]


# --- OSCR Curve Calculation and Visualization ---
def calculate_oscr_curve(results: dict, datamodule: DataModule) -> tuple[np.ndarray, np.ndarray]:
    """Calculates CCR vs FPR for the OSCR curve."""
    # Check required keys in results
    required_keys = ['predictions', 'labels', 'scores_for_ranking']
    if not all(key in results for key in required_keys):
        print("Warning: Missing required data for OSCR calculation (predictions, labels, or scores_for_ranking).")
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]) # Default line (no area)

    preds = np.array(results['predictions'])
    labels = np.array(results['labels'])
    scores_for_ranking = np.array(results['scores_for_ranking']) # Higher score = more unknown

    # Validate inputs
    if len(preds) != len(labels) or len(preds) != len(scores_for_ranking):
        print(f"Warning: Length mismatch in OSCR data (Preds: {len(preds)}, Labels: {len(labels)}, Scores: {len(scores_for_ranking)}). Cannot calculate OSCR.")
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    if len(preds) == 0:
        print("Warning: Empty data provided for OSCR calculation.")
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    # Filter out non-finite scores
    finite_score_mask = np.isfinite(scores_for_ranking)
    if not np.all(finite_score_mask):
        num_filtered = np.sum(~finite_score_mask)
        print(f"Warning: Filtering {num_filtered} non-finite scores for OSCR calculation.")
        preds = preds[finite_score_mask]
        labels = labels[finite_score_mask]
        scores_for_ranking = scores_for_ranking[finite_score_mask]
        if len(preds) == 0:
            print("Warning: No finite scores remain after filtering for OSCR.")
            return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    # Identify known/unknown based on ground truth labels
    unknown_labels_mask = datamodule._determine_unknown_labels(labels) # True if unknown (-1)
    known_mask = ~unknown_labels_mask

    # Calculate correctness for known samples and false positives for unknown samples
    # Correct known: Prediction matches label AND label is known
    is_correct_known = (preds == labels) & known_mask
    # False positive: Prediction is NOT unknown (-1) BUT label IS unknown
    is_false_positive = (preds != -1) & unknown_labels_mask

    # Count total known and unknown samples
    n_known = np.sum(known_mask)
    n_unknown = np.sum(unknown_labels_mask)

    if n_known == 0 or n_unknown == 0:
        print(f"Warning: No known ({n_known}) or no unknown ({n_unknown}) samples found after filtering. Cannot calculate OSCR curve.")
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    # Sort samples based on the 'unknownness' score (ascending order, so unknowns come later)
    sorted_indices = np.argsort(scores_for_ranking)
    sorted_correct_known = is_correct_known[sorted_indices]
    sorted_false_positive = is_false_positive[sorted_indices]

    # Calculate cumulative sums
    # CCR = Cumulative Correct Known / Total Known
    ccr_values = np.cumsum(sorted_correct_known) / n_known
    # FPR = Cumulative False Positives / Total Unknown
    fpr_values = np.cumsum(sorted_false_positive) / n_unknown

    # Add (0,0) point at the beginning for plotting and AUC calculation
    fpr_curve = np.insert(fpr_values, 0, 0.0)
    ccr_curve = np.insert(ccr_values, 0, 0.0)

    # Ensure curves end at (1, CCR_max) or potentially (FPR_max, 1) depending on score range
    # The current calculation correctly reflects the trade-off as threshold varies.

    return fpr_curve, ccr_curve


def visualize_oscr_curves(all_results: dict, datamodule: DataModule, args: argparse.Namespace):
    """Plots OSCR curves for comparing multiple OSR methods found in all_results."""
    print("\n--- Generating OSCR Comparison Curve ---")
    plt.figure(figsize=(9, 7)) # Slightly larger figure
    method_found = False
    plotted_methods = []

    # Sort methods alphabetically for consistent plot legend order
    sorted_methods = sorted(all_results.keys())

    for method in sorted_methods:
        results = all_results.get(method)
        # Check if results are valid and not an error entry
        if results and isinstance(results, dict) and 'error' not in results:
            print(f"  Calculating OSCR curve for: {method.upper()}")
            try:
                fpr, ccr = calculate_oscr_curve(results, datamodule)
                # Check if calculation returned valid data points
                if fpr is not None and ccr is not None and len(fpr) > 1 and len(ccr) > 1:
                    # Calculate Area Under the OSCR Curve using trapezoidal rule
                    oscr_auc = np.trapz(ccr, fpr)
                    plt.plot(fpr, ccr, lw=2.5, label=f'{method.upper()} (AUC = {oscr_auc:.3f})', alpha=0.85)
                    method_found = True
                    plotted_methods.append(method)
                    print(f"    {method.upper()} OSCR AUC: {oscr_auc:.4f}")
                else:
                    print(f"    Skipping OSCR plot for {method}: Insufficient data points returned from calculation.")
            except Exception as e:
                print(f"    Error calculating or plotting OSCR for {method}: {e}")
        elif results and isinstance(results, dict) and 'error' in results:
             print(f"  Skipping OSCR for {method.upper()} due to previous evaluation error: {results['error']}")
        else:
             print(f"  Skipping OSCR for {method.upper()}: Invalid or missing results.")


    if not method_found:
        print("No valid results found to plot OSCR comparison curve.")
        plt.close() # Close the empty figure
        return

    # Add reference line (ideal closed-set: 0 FPR, 1 CCR) - represented differently on OSCR
    # Often a diagonal line y=1-x is shown for reference, but its meaning is less direct here.
    # Let's just plot the axes clearly.
    # plt.plot([0, 1], [1, 0], color='grey', lw=1.5, linestyle='--', label='Reference (y=1-x)') # Optional reference

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('Correct Classification Rate (CCR)', fontsize=12)
    plt.title(f'OSCR Curves ({args.dataset}, Seen Ratio: {args.seen_class_ratio:.2f})', fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # Save the plot
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, f"oscr_comparison_{args.dataset}_{args.seen_class_ratio:.2f}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"--- OSCR comparison curve saved to: {save_path} ---")


# --- Result Saving and Summary ---

def _robust_json_converter(obj):
    """Handles various types for JSON serialization."""
    if isinstance(obj, (np.integer, int)): return int(obj)
    elif isinstance(obj, (np.floating, float)): return float(obj)
    elif isinstance(obj, (np.ndarray,)): return obj.tolist() # Convert numpy arrays
    elif isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat() # Convert dates/timestamps
    elif isinstance(obj, (torch.Tensor)): return obj.cpu().numpy().tolist() # Convert torch tensors
    elif isinstance(obj, set): return list(obj) # Convert sets
    elif isinstance(obj, pathlib.Path): return str(obj) # Convert Path objects
    elif isinstance(obj, (Callback, pl.Trainer, DataModule, argparse.Namespace)): return f"<{type(obj).__name__} object>" # Avoid serializing complex objects
    elif pd.isna(obj): return None # Handle pandas NaT, etc. -> null
    # Add more specific type handling if needed
    try:
        # Fallback for simple objects, but can fail
        return obj.__dict__
    except (AttributeError, TypeError):
        # Final fallback: represent as string
        return f"Unserializable object: {type(obj).__name__}"


def _save_results(all_results: dict, args: argparse.Namespace):
    """Saves evaluation results to JSON, with pickle fallback."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_suffix = "_tuned" if args.parameter_search else ""
    model_name_safe = args.model.replace('/', '_') # Make model name filename-safe
    base_filename = f"final_{model_name_safe}_{args.dataset}_{args.osr_method}_{args.seen_class_ratio:.2f}{results_suffix}"
    json_filename = os.path.join(RESULTS_DIR, f"{base_filename}.json")
    pickle_filename = os.path.join(RESULTS_DIR, f"{base_filename}_full.pkl")

    # Create a summary dictionary excluding large/complex objects
    summary_results = {}
    # Keys to exclude from the summary JSON for brevity/compatibility
    keys_to_exclude_from_summary = [
        'predictions', 'labels', 'raw_scores', 'scores_for_ranking',
        'embeddings', 'logits', 'sigmoid_scores', 'encoded', 'reconstructed',
        'recon_errors', 'features', 'distances', 'max_probs', 'unknown_probs',
        'max_scores', 'min_distances', 'z_scores', 'preds_mapped', 'labels_original',
        'confusion_matrix' # Exclude raw matrix, keep labels/names
    ]
    print("\n--- Saving Results ---")
    for method, res in all_results.items():
         if isinstance(res, dict):
             summary_results[method] = {k: v for k, v in res.items() if k not in keys_to_exclude_from_summary}
             # Optionally keep CM labels/names if needed in summary
             summary_results[method]['confusion_matrix_labels'] = res.get('confusion_matrix_labels')
             summary_results[method]['confusion_matrix_names'] = res.get('confusion_matrix_names')
         else:
             # Handle cases where result might not be a dict (e.g., just an error string)
             summary_results[method] = res

    # Attempt to save summary results as JSON
    try:
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False, default=_robust_json_converter)
        print(f"Consolidated results summary saved to: {json_filename}")
    except TypeError as json_e:
        print(f"\nWarning: Failed to save summary results as JSON due to serialization error: {json_e}")
        print("         Attempting to save full results (including large arrays) as pickle instead.")
        # Fallback to saving the *full* results dictionary as pickle
        try:
            with open(pickle_filename, 'wb') as pf:
                pickle.dump(all_results, pf)
            print(f"Full results saved as pickle: {pickle_filename}")
        except Exception as pickle_e:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Error saving full results as pickle ({pickle_filename}): {pickle_e}")
            import traceback; traceback.print_exc()
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    except Exception as e:
        print(f"\nUnexpected error saving summary results: {e}")


def _print_summary_table(all_results: dict, args: argparse.Namespace):
    """Prints a formatted summary table of key metrics to the console."""
    print("\n" + "="*110)
    print(f"{' ' * 40} Experiment Results Summary {' ' * 40}")
    print("="*110)

    # Metrics to display in the table and their display names
    metrics_to_display = ["accuracy", "auroc", "fpr_at_tpr90", "unknown_detection_rate", "f1_score"]
    metric_names_display = ["Acc(Known)", "AUROC", "FPR@TPR90", "UnkDetect", "F1(Known)"]

    # Filter out methods that resulted in an error
    methods_evaluated = sorted([m for m in all_results if isinstance(all_results.get(m), dict) and 'error' not in all_results[m]])

    if not methods_evaluated:
        print("No successful evaluation results to display in summary table.")
        print("="*110)
        return

    # Determine column width based on method names
    col_width = 16 # Adjust as needed
    header = "{:<20}".format("Metric") # Width for metric names
    for method in methods_evaluated:
        header += "{:<{}}".format(method.upper(), col_width)
    print(header)
    print("-" * len(header))

    # Print rows for each metric
    for i, metric_key in enumerate(metrics_to_display):
        # Highlight the tuning metric if parameter search was enabled
        metric_display_name = metric_names_display[i]
        if args.parameter_search and metric_key == args.tuning_metric:
            row = "* {:<18}".format(metric_display_name) # Add indicator and adjust spacing
        else:
            row = "  {:<18}".format(metric_display_name) # Add indent for alignment

        # Get value for each method
        for method in methods_evaluated:
            val = all_results[method].get(metric_key, "N/A")
            # Format the value
            try:
                # Format as float if possible, otherwise as string
                formatted_val = "{:<{}.4f}".format(float(val), col_width) if pd.notna(val) and isinstance(val, (float, int, np.number)) else "{:<{}}".format("NaN" if pd.isna(val) else str(val), col_width)
            except (TypeError, ValueError):
                formatted_val = "{:<{}}".format(str(val), col_width) # Fallback to string
            row += formatted_val
        print(row)

    if args.parameter_search:
        print("\n* Metric used for hyperparameter tuning.")
    print("=" * len(header))


# --- Main Evaluation Orchestrator ---
def evaluate_osr_main(initial_trained_model: pl.LightningModule, datamodule: DataModule, args: argparse.Namespace):
    """Runs evaluation for the selected OSR method(s)."""
    all_results = {} # Dictionary to store results for each method
    os.makedirs(RESULTS_DIR, exist_ok=True) # Ensure results directory exists

    if args.parameter_search:
        print("\n" + "="*70)
        print(f"{' ' * 15} Hyperparameter Tuning Mode (Optuna) Enabled {' ' * 15}")
        print("="*70)
        print(f" Tuning Metric: {args.tuning_metric}")
        print(f" Number of Trials: {args.n_trials}")
        print(f" Method(s) to Tune: {args.osr_method}")
        print("="*70 + "\n")

    # Map method names to their evaluation functions
    evaluation_function_map = {
        "threshold": evaluate_threshold_osr,
        "openmax": evaluate_openmax_osr,
        "crosr": evaluate_crosr_osr,
        "doc": evaluate_doc_osr,
        "adb": evaluate_adb_osr
    }

    # Determine which methods to run
    if args.osr_method == "all":
        methods_to_run = list(evaluation_function_map.keys())
    elif args.osr_method in evaluation_function_map:
        methods_to_run = [args.osr_method]
    else:
        print(f"Error: Unknown OSR method specified: '{args.osr_method}'. Choose from {list(evaluation_function_map.keys())} or 'all'.")
        return {} # Return empty results

    print(f"--- Starting OSR Evaluation for Method(s): {', '.join(m.upper() for m in methods_to_run)} ---")

    # Run evaluation for each selected method
    for method in methods_to_run:
        if method in evaluation_function_map:
            eval_func = evaluation_function_map[method]
            # The evaluation function handles preparation (tuning/loading, retraining) and evaluation
            eval_func(initial_trained_model, datamodule, args, all_results)
        # Error handling is now inside each evaluate_xxx function

    # --- Post-Evaluation Steps ---
    # Save consolidated results
    _save_results(all_results, args)

    # Print summary table
    _print_summary_table(all_results, args)

    # Generate OSCR comparison plot if multiple methods were evaluated successfully
    successful_methods = [m for m in all_results if isinstance(all_results.get(m), dict) and 'error' not in all_results[m]]
    if len(successful_methods) > 1:
        visualize_oscr_curves(all_results, datamodule, args)
    elif len(successful_methods) == 1:
         print("\nOnly one method evaluated successfully, skipping OSCR comparison plot.")
    else:
         print("\nNo methods evaluated successfully, skipping OSCR comparison plot.")


    print("\n--- OSR Evaluation Finished ---")
    return all_results


# =============================================================================
# Argument Parser and Main Execution Block
# =============================================================================
def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Open-Set Recognition Experiments with RoBERTa')

    # --- Dataset and Model ---
    parser.add_argument('-dataset', type=str, default='acm',
                        choices=['newsgroup20', 'bbc_news', 'trec', 'reuters8', 'acm', 'chemprot',
                                 'banking77', 'oos', 'stackoverflow', 'atis', 'snips',
                                 'financial_phrasebank', 'arxiv10', 'custom_syslog'],
                        help='Dataset to use.')
    parser.add_argument('-model', type=str, default='roberta-base',
                        help='Pre-trained RoBERTa model name from Hugging Face Hub.')

    # --- OSR Configuration ---
    parser.add_argument('-osr_method', type=str, default='all',
                        choices=['threshold', 'openmax', 'crosr', 'doc', 'adb', 'all'],
                        help='OSR method(s) to evaluate.')
    parser.add_argument('-seen_class_ratio', type=float, default=0.5,
                        help='Ratio of classes used as known/seen during setup (0.0 to 1.0).')

    # --- Training Hyperparameters ---
    parser.add_argument('-epochs', type=int, default=10, help='Maximum training epochs.')
    parser.add_argument('-batch_size', type=int, default=32, help='Batch size for training and evaluation.')
    parser.add_argument('-lr', type=float, default=2e-5, help='Learning rate for backbone/standard models.')
    parser.add_argument('-weight_decay', type=float, default=0.01, help='Weight decay for AdamW optimizer.')
    parser.add_argument('-warmup_ratio', type=float, default=0.1,
                        help='Ratio of total training steps used for linear learning rate warmup.')
    parser.add_argument('-max_warmup_steps', type=int, default=500,
                        help='Maximum number of warmup steps (overrides ratio if smaller).')
    parser.add_argument('-gradient_clip_val', type=float, default=1.0, help='Gradient clipping value.')
    parser.add_argument('-early_stopping_patience', type=int, default=3,
                        help='Patience (epochs) for early stopping based on validation loss.')
    parser.add_argument('-early_stopping_delta', type=float, default=0.001,
                        help='Minimum change in validation loss to qualify as improvement for early stopping.')

    # --- Data Splitting ---
    parser.add_argument('-train_ratio', type=float, default=0.7, help='Proportion of data for training.')
    parser.add_argument('-val_ratio', type=float, default=0.15, help='Proportion of data for validation.')
    parser.add_argument('-test_ratio', type=float, default=0.15, help='Proportion of data for testing.')

    # --- Environment and Reproducibility ---
    parser.add_argument('-random_seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('-force_gpu', action='store_true', help='Force GPU usage even if CUDA might report issues (use with caution).')
    parser.add_argument('-gpu_id', type=int, default=0, help='GPU ID to use if multiple GPUs are available.')

    # --- OSR Method Specific Parameters (Tunable) ---
    # ThresholdOSR
    parser.add_argument('-param_threshold', type=float, default=None, # Default handled in class if None
                        help='ThresholdOSR: Fixed softmax threshold.')
    # OpenMaxOSR
    parser.add_argument('-param_openmax_tailsize', type=int, default=None,
                        help='OpenMaxOSR: Tail size for Weibull fitting.')
    parser.add_argument('-param_openmax_alpha', type=int, default=None,
                        help='OpenMaxOSR: Number of top classes to revise logits for.')
    # CROSROSR
    parser.add_argument('-param_crosr_reconstruction_threshold', type=float, default=None,
                        help='CROSROSR: Weibull CDF threshold for unknown detection.')
    parser.add_argument('-param_crosr_tailsize', type=int, default=None,
                        help='CROSROSR: Tail size for EVT fitting on reconstruction errors.')
    parser.add_argument('-param_crosr_recon_weight', type=float, default=0.5, # Note: This affects training
                        help='CROSROSR: Weight of reconstruction loss during training.')
    # DOCOSR
    parser.add_argument('-param_doc_k', type=float, default=None,
                        help='DOCOSR: k-sigma value for calculating class thresholds.')
    # ADBOSR
    parser.add_argument('-lr_adb', type=float, default=5e-4, # Note: This affects training
                        help='ADBOSR: Learning rate specifically for ADB centers/radii.')
    parser.add_argument('-param_adb_distance', type=str, default='cosine', choices=['cosine', 'euclidean'],
                        help='ADBOSR: Distance metric used for calculation and thresholding.')
    parser.add_argument('-param_adb_delta', type=float, default=0.1, # Note: This affects training
                        help='ADBOSR: Margin delta used in the ADB loss function.')
    parser.add_argument('-param_adb_alpha', type=float, default=0.5, # Note: This affects training
                        help='ADBOSR: Weighting factor (alpha) for the ADB loss component.')
    # Use BooleanOptionalAction for clearer command line usage (--adb_freeze_backbone / --no-adb_freeze_backbone)
    parser.add_argument('--adb_freeze_backbone', action=argparse.BooleanOptionalAction, default=True, # Note: Affects training
                        help='ADBOSR: Freeze backbone weights during ADB training phase.')

    # --- Hyperparameter Tuning (Optuna) ---
    parser.add_argument('-parameter_search', action='store_true',
                        help='Enable Optuna hyperparameter search for the selected osr_method(s).')
    parser.add_argument('-tuning_metric', type=str, default='f1_score',
                        choices=['accuracy', 'auroc', 'f1_score', 'unknown_detection_rate', 'fpr_at_tpr90'],
                        help='Metric to optimize during hyperparameter search.')
    parser.add_argument('-n_trials', type=int, default=20,
                        help='Number of trials for Optuna hyperparameter search.')

    args = parser.parse_args()
    return args


def check_gpu():
    """Prints basic GPU diagnostics."""
    print("\n----- GPU Diagnostics -----")
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"Device Count: {torch.cuda.device_count()}")
        current_device_id = torch.cuda.current_device()
        print(f"Current Device ID: {current_device_id}")
        print(f"Current Device Name: {torch.cuda.get_device_name(current_device_id)}")
        # Optionally list all devices
        # for i in range(torch.cuda.device_count()):
        #     print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA Available: No")
    print("-------------------------\n")


def _setup_environment(args: argparse.Namespace):
    """Sets up the environment (seed, GPU check, prints args)."""
    print("\n----- Experiment Configuration -----")
    print(json.dumps(vars(args), indent=2))
    print("------------------------------------\n")
    check_gpu()
    print(f"Setting random seed for reproducibility: {args.random_seed}")
    pl.seed_everything(args.random_seed, workers=True) # Ensure worker seeding for DataLoaders

def _load_tokenizer(model_name: str) -> RobertaTokenizer:
    """Loads the RoBERTa tokenizer."""
    print(f"Loading tokenizer: {model_name}...")
    try:
        # Explicitly disable progress bars for cleaner logs
        tokenizer = RobertaTokenizer.from_pretrained(model_name, verbose=False)
        print("Tokenizer loaded successfully.")
        return tokenizer
    except OSError as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Error loading tokenizer '{model_name}': Model not found or network issue.")
        print(f"Ensure '{model_name}' is a valid model name on Hugging Face Hub and you have internet access.")
        print(f"Original Error: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit(1)
    except ValueError as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Error loading tokenizer '{model_name}': Invalid configuration or model name format.")
        print(f"Original Error: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit(1)
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"An unexpected error occurred while loading tokenizer '{model_name}': {e}")
        import traceback; traceback.print_exc()
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit(1)

def _setup_datamodule(args: argparse.Namespace, tokenizer) -> DataModule:
    """Initializes and sets up the DataModule."""
    print(f"\n--- Preparing DataModule: {args.dataset} ---")
    datamodule = DataModule(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seen_class_ratio=args.seen_class_ratio,
        random_seed=args.random_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_length=384, # Consider making this an arg
        data_dir=DATA_DIR,
        num_workers=NUM_DATALOADER_WORKERS
    )
    # Prepare data (download/extract if needed) - runs once
    datamodule.prepare_data()
    # Setup data (load, split, process) - runs on each process in DDP
    datamodule.setup(stage=None) # Setup for all stages initially

    num_model_classes = datamodule.num_seen_classes
    if num_model_classes is None or num_model_classes <= 0:
        raise ValueError(f"DataModule setup resulted in invalid number of seen classes: {num_model_classes}. Check data and split ratios.")
    print(f"DataModule setup complete. Model will be trained on {num_model_classes} known classes.")
    return datamodule

def _train_initial_model(datamodule: DataModule, args: argparse.Namespace) -> pl.LightningModule | None:
    """Trains the initial standard RobertaClassifier model."""
    print("\n--- Step 1: Training Initial Standard Base Model (RobertaClassifier) ---")
    initial_model_class = RobertaClassifier
    # Initialize using current args (learning rate, etc.)
    initial_model_instance = _initialize_model_for_eval(
        initial_model_class, args, datamodule.num_seen_classes
    )

    # Use a specific identifier for this initial training run
    train_args = copy.deepcopy(args) # Avoid modifying original args
    initial_run_id = "initial_standard"

    initial_checkpoint_path = train_model(initial_model_instance, datamodule, train_args, run_identifier=initial_run_id)

    if not initial_checkpoint_path or not os.path.exists(initial_checkpoint_path):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Error: Initial training failed to produce a valid checkpoint.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return None # Indicate failure

    print(f"--- Initial Model Training Complete. Checkpoint: {initial_checkpoint_path} ---")
    return initial_checkpoint_path


def _load_initial_model(checkpoint_path: str, args: argparse.Namespace) -> pl.LightningModule:
    """Loads the initially trained standard model from checkpoint."""
    print(f"\n--- Step 2: Loading Initially Trained Model ---")
    print(f"  Checkpoint: {checkpoint_path}")
    initial_model_class = RobertaClassifier # Assuming initial model is always standard
    try:
        device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu')
        # Load the model from the checkpoint
        loaded_standard_model = initial_model_class.load_from_checkpoint(checkpoint_path, map_location=device)
        print(f"Initial model ({initial_model_class.__name__}) loaded successfully onto {device}.")
        return loaded_standard_model
    except FileNotFoundError:
         print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print(f"Error: Checkpoint file not found at '{checkpoint_path}'. Cannot load initial model.")
         print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         sys.exit(1)
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Error loading initial model from checkpoint '{checkpoint_path}': {e}")
        import traceback; traceback.print_exc()
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit(1)


def main():
    """Main execution function."""
    start_time = time.time()
    args = parse_args()
    _setup_environment(args)
    tokenizer = _load_tokenizer(args.model)
    datamodule = _setup_datamodule(args, tokenizer)

    # --- Initial Model Training ---
    # Train a standard classifier first. This serves as the base for
    # methods like Threshold/OpenMax and as a starting point check for others.
    initial_checkpoint_path = _train_initial_model(datamodule, args)
    if initial_checkpoint_path is None:
        sys.exit(1) # Exit if initial training failed

    # --- Load the Trained Initial Model ---
    loaded_standard_model = _load_initial_model(initial_checkpoint_path, args)

    # --- OSR Evaluation ---
    print("\n--- Step 3: Evaluating OSR Algorithm(s) ---")
    # Pass the loaded standard model as the 'base_model'
    evaluate_osr_main(loaded_standard_model, datamodule, args)

    end_time = time.time()
    total_duration = end_time - start_time
    print("\n" + "="*50)
    print(f"Total Experiment Duration: {total_duration / 60:.2f} minutes")
    print("Experiment finished.")
    print("="*50)


if __name__ == "__main__":
    main()
