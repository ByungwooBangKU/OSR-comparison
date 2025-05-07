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
ORIGINAL_DATA_PATH = 'log_all_critical.csv' # 표준 모델 학습용 원본 ID 데이터 경로 (02_train_id_model_and_mask.py의 입력과 동일해야 함)
# MASKED_DATA_PATH 는 02_train_id_model_and_mask.py 의 출력 파일 경로를 사용해야 합니다.
# 예시: 'log_all_filtered_with_attention_masks.csv'
# 이 스크립트에서는 이 파일을 'INPUT_DATA_WITH_MASKS_PATH'로 명명합니다.
INPUT_DATA_WITH_MASKS_PATH = 'log_all_filtered_with_attention_masks.csv' # 수정됨: 이전 단계에서 마스크 정보가 추가된 파일
TRAINED_MODEL_PATH = "./roberta_base_known_classifier/best_model.ckpt" # 수정됨: 02_train_id_model_and_mask.py 에서 학습/저장한 모델 경로

TEXT_COLUMN = 'text'
CLASS_COLUMN = 'class' # 원본 데이터의 클래스 컬럼
MASKED_TEXT_COLUMN = 'masked_text_attention'
TOP_WORDS_COLUMN = 'top_attention_words'
EXCLUDE_CLASS_FOR_TRAINING = "unknown" # 모델 학습 시 제외했던 클래스 (LabelEncoder 구성 시 필요)

# 각 지표별 맞춤형 퍼센타일 및 모드 설정
METRIC_SETTINGS = {
    'removed_avg_attention': {'percentile': 90, 'mode': 'higher'},
    'top_k_avg_attention': {'percentile': 80, 'mode': 'higher'},
    'max_attention': {'percentile': 85, 'mode': 'higher'},
    'attention_entropy': {'percentile': 25, 'mode': 'lower'},
    'msp_difference': {'percentile': 90, 'mode': 'higher'} # 새로운 MSP 차이 지표 설정
}
# 순차 필터링 시퀀스 (MSP 차이도 포함 가능)
SEQUENTIAL_FILTERING_SEQUENCE = [
    ('removed_avg_attention', {'percentile': 90, 'mode': 'higher'}),
    ('msp_difference', {'percentile': 85, 'mode': 'higher'}), # 예시로 MSP 차이 추가
    ('attention_entropy', {'percentile': 30, 'mode': 'lower'})
]


# 출력 경로 설정
OUTPUT_DIR = 'oe_extraction_results_v2' # 결과 저장 디렉토리명 변경
LOG_DIR = os.path.join(OUTPUT_DIR, "lightning_logs_oe_extraction") # 로그 저장 경로 (모델 학습 안하므로 사실상 불필요)
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
OE_DATA_DIR = os.path.join(OUTPUT_DIR, "extracted_oe_datasets")

# 모델 및 데이터로딩 설정
MODEL_NAME = "roberta-base" # 로드할 모델의 기본 아키텍처 (TRAINED_MODEL_PATH와 일치해야 함)
MAX_LENGTH = 256
BATCH_SIZE = 64 # 추론 시 배치 크기
ACCELERATOR = "auto"
DEVICES = "auto"
PRECISION = "16-mixed" if torch.cuda.is_available() and ACCELERATOR != "cpu" else "32-true"
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1
RANDOM_STATE = 42
TOP_K_ATTENTION = 3

# 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True) # 비록 사용 안될 수 있지만 생성
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(OE_DATA_DIR, exist_ok=True)

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

def tokenize_nltk(text): # 이 함수는 현재 코드에서 직접 사용되지 않음
    if not text:
        return []
    try:
        return word_tokenize(text)
    except Exception:
        return text.split()

# --- 데이터 클래스 (TextDataset은 이전과 동일하게 유지 가능) ---
class TextDataset(TorchDataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128, labels: Optional[List[int]] = None):
        self.texts = texts
        # labels가 제공되지 않으면 더미 레이블 생성 (추론용)
        self.labels = labels if labels is not None else [-1] * len(texts)

        print(f"Tokenizing {len(texts)} texts for TextDataset...")
        valid_texts = [str(t) if pd.notna(t) else "" for t in texts]
        self.encodings = tokenizer(valid_texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        print("Tokenization complete for TextDataset.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# --- PyTorch Lightning Module (모델 로드용으로만 사용, 학습 기능은 필요 없음) ---
# 기존 LogClassifierPL을 사용하되, 학습 관련 부분은 무시됨
# from <your_previous_script> import LogClassifierPL # 만약 별도 파일에 있다면
# 여기서는 직접 정의 (학습 기능은 사용 안 함)
class LogClassifierPL(pl.LightningModule):
    def __init__(self, model_name, num_labels, label2id, id2label, learning_rate=2e-5,
                 use_weighted_loss=False, class_weights=None, use_lr_scheduler=False, warmup_steps=0,
                 confusion_matrix_dir=None): # confusion_matrix_dir 추가했지만, 추론만 하므로 실제 사용 안됨
        super().__init__()
        # save_hyperparameters()는 모델 로드 시 중요.
        # learning_rate 등 학습 파라미터는 로드 시점에 덮어써지므로, 로드 후 모델 상태에 영향 없음.
        self.save_hyperparameters()
        print(f"Initializing LogClassifierPL for loading: {model_name} for {num_labels} known classes.")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True, # 로드 시 중요
            output_attentions=True,       # 어텐션 추출용
            output_hidden_states=True   # 특징 추출용
        )
        # Loss, metrics, optimizer 등은 추론만 할 것이므로 정의하지 않거나 무시됨
        # 만약 load_from_checkpoint가 엄격하게 모든 것을 요구한다면 더미로 정의 필요

    def forward(self, batch, output_features=False, output_attentions=False):
        # 모델 추론을 위한 forward pass
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

    # training_step, validation_step, configure_optimizers 등은 모델 로드 후 추론만 하므로 필요 없음
    # 만약 load_from_checkpoint가 이들을 요구한다면 pass로 간단히 정의


# --- 단어 어텐션 스코어 추출 함수 (이전과 동일) ---
@torch.no_grad()
def get_word_attention_scores_pl(texts, model_pl, tokenizer, device, layer_idx=-1, head_idx=None, max_length=512):
    model_pl.eval() # Ensure model is in eval mode
    # model_pl.to(device) # 호출하는 쪽에서 device 설정

    word_attention_scores = [{} for _ in range(len(texts))]
    valid_indices, valid_texts = [], []
    for i, text in enumerate(texts):
        if isinstance(text, str) and text.strip():
            valid_indices.append(i); valid_texts.append(text)
    if not valid_texts: return word_attention_scores

    batch_size_attn = 32 # 어텐션 계산 시 메모리 사용량 고려
    for batch_start in range(0, len(valid_texts), batch_size_attn):
        batch_end = min(batch_start + batch_size_attn, len(valid_texts))
        current_batch_texts = valid_texts[batch_start:batch_end]
        current_batch_indices = valid_indices[batch_start:batch_end]

        inputs = tokenizer(current_batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length, return_offsets_mapping=True)
        offset_mappings_batch = inputs.pop('offset_mapping').cpu().numpy()
        input_ids_batch = inputs['input_ids'].cpu().numpy()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model_pl.model(**inputs, output_attentions=True) # Access HuggingFace model directly
        attentions_layer_batch = outputs.attentions[layer_idx].cpu().numpy() # (batch_size, num_heads, seq_len, seq_len)

        for i_in_batch, (text_idx, original_text_sample) in enumerate(zip(current_batch_indices, current_batch_texts)):
            attention_sample = attentions_layer_batch[i_in_batch] # (num_heads, seq_len, seq_len)
            offset_mapping_sample = offset_mappings_batch[i_in_batch]
            token_ids_sample = input_ids_batch[i_in_batch]

            # Aggregate heads (mean or specific head)
            if head_idx is not None:
                att_matrix = attention_sample[head_idx] # (seq_len, seq_len)
            else:
                att_matrix = np.mean(attention_sample, axis=0) # (seq_len, seq_len), mean over heads

            # CLS token's attention to other tokens
            cls_attentions_to_tokens = att_matrix[0, :] # (seq_len,)

            word_scores_for_sample = {}
            current_word_tokens_indices = []
            last_word_end_offset = 0

            # Reconstruct words and map attentions (RoBERTa/BERT specific subword handling)
            # This part is complex and needs to handle tokenizer specifics (e.g., "Ġ" for RoBERTa)
            processed_text_sample = preprocess_text_for_roberta(original_text_sample)
            tokens_from_tokenizer = tokenizer.convert_ids_to_tokens(token_ids_sample)

            # Simplified word reconstruction and scoring
            # A more robust solution would use offset_mapping to precisely map tokens to original words
            current_word_str = ""
            current_word_att_scores = []

            for j, token_id in enumerate(token_ids_sample):
                if token_id in tokenizer.all_special_ids: # Skip CLS, SEP, PAD
                    if current_word_str: # Finalize previous word
                        word_scores_for_sample[current_word_str.lower()] = np.mean(current_word_att_scores) if current_word_att_scores else 0
                        current_word_str, current_word_att_scores = "", []
                    continue

                token_str = tokens_from_tokenizer[j]
                # Basic RoBERTa subword handling (starts with 'Ġ' or is the first token)
                if token_str.startswith("Ġ") or not current_word_str:
                    if current_word_str: # Finalize previous word
                         word_scores_for_sample[current_word_str.lower()] = np.mean(current_word_att_scores) if current_word_att_scores else 0
                    current_word_str = token_str.lstrip("Ġ")
                    current_word_att_scores = [cls_attentions_to_tokens[j]]
                else: # Continuation of a word
                    current_word_str += token_str
                    current_word_att_scores.append(cls_attentions_to_tokens[j])

            if current_word_str: # Finalize last word
                word_scores_for_sample[current_word_str.lower()] = np.mean(current_word_att_scores) if current_word_att_scores else 0

            word_attention_scores[text_idx] = word_scores_for_sample
    return word_attention_scores


# --- 어텐션/특징 및 MSP 추출 함수 (개선) ---
@torch.no_grad()
def extract_metrics_and_features(
    model_pl: LogClassifierPL,
    texts_original: List[str],
    texts_masked: List[str],
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
    top_k_attn: int = 3
) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    model_pl.eval()
    model_pl.to(device)

    num_samples = len(texts_original)
    all_metrics = []
    all_features_masked = [] # 마스크된 텍스트에 대한 특징을 저장 (t-SNE용)

    print("Extracting metrics (including MSP difference) and features...")
    for i in tqdm(range(0, num_samples, batch_size), desc="Extracting Metrics"):
        batch_texts_original = texts_original[i:i+batch_size]
        batch_texts_masked = texts_masked[i:i+batch_size]

        # 1. Process Original Texts (for MSP_original)
        inputs_original = tokenizer(batch_texts_original, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        outputs_original = model_pl.model(**inputs_original) # Get logits from HuggingFace model
        probs_original = F.softmax(outputs_original.logits, dim=1)
        msp_original = torch.max(probs_original, dim=1)[0].cpu().numpy()

        # 2. Process Masked Texts (for MSP_masked, attention metrics, features)
        inputs_masked = tokenizer(batch_texts_masked, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        # 어텐션과 은닉 상태를 함께 얻기 위해 model_pl.forward 사용
        outputs_masked_obj = model_pl.forward(inputs_masked, output_features=True, output_attentions=True)
        
        logits_masked = outputs_masked_obj.logits
        probs_masked = F.softmax(logits_masked, dim=1)
        msp_masked = torch.max(probs_masked, dim=1)[0].cpu().numpy()

        attentions_batch = outputs_masked_obj.attentions[-1].cpu().numpy() # (batch, heads, seq, seq)
        features_batch = outputs_masked_obj.hidden_states[-1][:, 0, :].cpu().numpy() # CLS token features
        all_features_masked.extend(list(features_batch))

        # MSP 차이 계산
        msp_diff_batch = msp_original - msp_masked

        # 배치 내 각 샘플에 대한 어텐션 지표 계산
        for j in range(len(batch_texts_masked)):
            sample_metrics = {}
            sample_metrics['msp_original'] = msp_original[j]
            sample_metrics['msp_masked'] = msp_masked[j]
            sample_metrics['msp_difference'] = msp_diff_batch[j]

            # 어텐션 지표 계산 로직 (extract_attention_and_features 함수 내부 로직 참고)
            attn_sample = attentions_batch[j] # (num_heads, seq_len, seq_len)
            token_ids = inputs_masked["input_ids"][j].cpu().numpy()
            valid_token_indices = np.where(
                (token_ids != tokenizer.pad_token_id) &
                (token_ids != tokenizer.cls_token_id) &
                (token_ids != tokenizer.sep_token_id)
            )[0]

            if len(valid_token_indices) == 0:
                sample_metrics.update({'max_attention': 0, 'top_k_avg_attention': 0, 'attention_entropy': 0})
            else:
                cls_attentions = np.mean(attn_sample[:, 0, :], axis=0) # (seq_len,)
                valid_cls_attentions = cls_attentions[valid_token_indices]

                max_attention = np.max(valid_cls_attentions) if len(valid_cls_attentions) > 0 else 0
                k = min(top_k_attn, len(valid_cls_attentions))
                top_k_avg_attention = np.mean(np.sort(valid_cls_attentions)[-k:]) if k > 0 else 0
                attention_probs = F.softmax(torch.tensor(valid_cls_attentions), dim=0).numpy()
                attention_entropy = entropy(attention_probs) if len(attention_probs) > 0 else 0
                sample_metrics.update({
                    'max_attention': max_attention,
                    'top_k_avg_attention': top_k_avg_attention,
                    'attention_entropy': attention_entropy
                })
            all_metrics.append(sample_metrics)

    df_results = pd.DataFrame(all_metrics)

    del outputs_original, outputs_masked_obj, attentions_batch, features_batch, inputs_original, inputs_masked
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    return df_results, all_features_masked


# --- 시각화 함수 (plot_metric_distribution, plot_tsne 이전과 동일하게 사용 가능) ---
def plot_metric_distribution(scores: np.ndarray, metric_name: str, title: str, save_path: Optional[str] = None):
    if len(scores) == 0: print(f"No scores for {metric_name}. Skipping plot."); return
    plt.figure(figsize=(10, 6))
    if SNS_AVAILABLE: sns.histplot(scores, bins=50, kde=True, stat='density')
    else: plt.hist(scores, bins=50, density=True)
    plt.title(title); plt.xlabel(metric_name); plt.ylabel('Density'); plt.grid(alpha=0.3)
    if save_path: plt.savefig(save_path); print(f"{metric_name} dist plot saved: {save_path}"); plt.close()
    else: plt.show(); plt.close()

def plot_tsne(features, labels, title, save_path=None,
                  highlight_indices=None, highlight_label='OE Candidate',
                  class_names=None, seed=42, perplexity=30, n_iter=1000):
    if len(features) == 0: print("No features for t-SNE."); return
    print(f"Running t-SNE on {features.shape[0]} samples (perplexity={perplexity}, n_iter={n_iter})...")
    try:
        effective_perplexity = min(perplexity, features.shape[0] - 1)
        if effective_perplexity <= 0: print("Error: Not enough samples for t-SNE. Skipping plot."); return
        tsne = TSNE(n_components=2, random_state=seed, perplexity=effective_perplexity, n_iter=n_iter, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(features)
    except Exception as e: print(f"Error running t-SNE: {e}. Skipping plot."); return

    df_tsne = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
    df_tsne['label'] = labels; df_tsne['is_highlighted'] = False
    if highlight_indices is not None and len(highlight_indices) > 0:
        # Ensure highlight_indices are within bounds
        valid_highlight_indices = [idx for idx in highlight_indices if idx < len(df_tsne)]
        df_tsne.loc[valid_highlight_indices, 'is_highlighted'] = True

    plt.figure(figsize=(14, 10))
    unique_labels = sorted(df_tsne['label'].unique())
    # Ensure enough colors if many unique labels
    num_unique_labels = len(unique_labels)
    colors = plt.cm.get_cmap('tab20', max(20, num_unique_labels))(np.linspace(0, 1, num_unique_labels))


    for i, label_val in enumerate(unique_labels):
        subset = df_tsne[(df_tsne['label'] == label_val) & (~df_tsne['is_highlighted'])]
        if len(subset) > 0:
            class_name = class_names.get(label_val, f'Class {label_val}') if class_names else f'Class {label_val}'
            plt.scatter(subset['tsne1'], subset['tsne2'], color=colors[i % len(colors)], label=class_name, alpha=0.7, s=30)

    if highlight_indices is not None and len(highlight_indices) > 0:
        highlight_subset = df_tsne[df_tsne['is_highlighted']]
        if len(highlight_subset) > 0:
            plt.scatter(highlight_subset['tsne1'], highlight_subset['tsne2'],
                      color='red', marker='x', s=100, label=highlight_label, alpha=0.9, zorder=5)

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("t-SNE Dimension 1", fontsize=14); plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.grid(alpha=0.3, linestyle='--')
    legend = plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.tight_layout(rect=[0, 0, 0.83, 1]) # Adjust right boundary for legend
    if save_path: plt.savefig(save_path, dpi=300); print(f"t-SNE plot saved: {save_path}")
    else: plt.show()
    plt.close()


# --- 메인 실행 로직 ---
def main():
    set_seed(RANDOM_STATE)
    pl.seed_everything(RANDOM_STATE) # PyTorch Lightning 시드 설정

    # 1. 학습된 ID 분류 모델 로드
    print(f"--- 1. Loading Trained ID Classification Model from: {TRAINED_MODEL_PATH} ---")
    # 모델 로드를 위해 필요한 정보 (num_labels, label2id, id2label) 구성
    # 이는 02_train_id_model_and_mask.py에서 사용된 DataModule과 일치해야 함.
    # 간단하게는, 해당 정보를 저장했다가 로드하거나, 원본 데이터를 다시 로드하여 DataModule을 실행시켜 얻을 수 있음.
    # 여기서는 ORIGINAL_DATA_PATH 를 기반으로 DataModule을 실행하여 해당 정보를 얻는 것으로 가정.
    # (주의: DataModule 실행은 토큰화 등 시간이 걸릴 수 있음. 정보를 파일로 저장/로드하는 것이 더 효율적)

    # --- 임시 DataModule 실행으로 label 정보 얻기 ---
    # 이 부분은 실제로는 02_train_id_model_and_mask.py 실행 시 생성된 label_info.json 등을 로드하는 것으로 대체 가능
    print("Temporarily running DataModule to get label info (can be optimized by saving/loading label_info)...")
    temp_dm_for_label_info = LogDataModuleForKnownClasses( # oe_extractor.py의 DataModule과 동일하게
        file_path=ORIGINAL_DATA_PATH, text_col=TEXT_COLUMN, class_col=CLASS_COLUMN,
        exclude_class=EXCLUDE_CLASS_FOR_TRAINING, model_name=MODEL_NAME, batch_size=BATCH_SIZE, # batch_size 등은 실제 로딩에 영향 X
        min_samples_per_class=3, num_workers=1, random_state=RANDOM_STATE, use_weighted_loss=False
    )
    temp_dm_for_label_info.setup()
    num_labels_for_model = temp_dm_for_label_info.num_labels
    label2id_for_model = temp_dm_for_label_info.label2id
    id2label_for_model = temp_dm_for_label_info.id2label
    del temp_dm_for_label_info # 메모리 해제
    print(f"Label info for model loading: num_labels={num_labels_for_model}, label2id={label2id_for_model}")
    # --- 임시 DataModule 실행 완료 ---

    if not os.path.exists(TRAINED_MODEL_PATH):
        print(f"Error: Trained model not found at {TRAINED_MODEL_PATH}. Please run 02_train_id_model_and_mask.py first.")
        return

    try:
        # LogClassifierPL.load_from_checkpoint를 사용하여 모델 로드
        # hparams.yaml이 ckpt와 같은 디렉토리에 있다면 자동으로 하이퍼파라미터 로드
        # 없다면, __init__에 필요한 인자들을 명시적으로 전달해야 함
        trained_model_pl = LogClassifierPL.load_from_checkpoint(
            TRAINED_MODEL_PATH,
            # __init__ 에 정의된 모든 인자를 전달해야 할 수 있음.
            # save_hyperparameters()로 저장된 hparams가 ckpt와 함께 hparams.yaml로 있다면,
            # 아래 인자들은 자동으로 로드되거나, 필요시 덮어쓰기 가능
            model_name=MODEL_NAME,
            num_labels=num_labels_for_model,
            label2id=label2id_for_model,
            id2label=id2label_for_model
            # learning_rate 등 학습 관련 파라미터는 추론만 하므로 중요도 낮음
        )
        print("Trained ID model loaded successfully.")
    except Exception as e:
        print(f"Error loading model from checkpoint: {e}")
        print("Attempting to load with strict=False (may ignore some mismatches, use with caution)")
        try:
            trained_model_pl = LogClassifierPL.load_from_checkpoint(
                TRAINED_MODEL_PATH,
                strict=False, # 일부 파라미터 불일치 무시
                model_name=MODEL_NAME,
                num_labels=num_labels_for_model,
                label2id=label2id_for_model,
                id2label=id2label_for_model
            )
            print("Trained ID model loaded with strict=False.")
        except Exception as e2:
            print(f"Still failed to load model with strict=False: {e2}")
            return


    current_device = torch.device("cuda" if torch.cuda.is_available() and ACCELERATOR != "cpu" else "cpu")
    trained_model_pl.to(current_device)
    trained_model_pl.eval()
    # trained_model_pl.freeze() # freeze는 학습 방지용. eval()이면 충분.

    # Tokenizer는 모델 로드 후 model_pl.tokenizer로 접근하거나, 새로 생성
    # 여기서는 TRAINED_MODEL_PATH와 동일한 모델명의 tokenizer를 새로 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


    # 2. 입력 데이터 (마스크 정보 포함) 로드
    print(f"\n--- 2. Loading data with masks from: {INPUT_DATA_WITH_MASKS_PATH} ---")
    try:
        data_df = pd.read_csv(INPUT_DATA_WITH_MASKS_PATH)
        print(f"Data with masks loaded: {len(data_df)} samples")

        # 필수 컬럼 확인
        required_cols_input = [TEXT_COLUMN, MASKED_TEXT_COLUMN, TOP_WORDS_COLUMN, CLASS_COLUMN]
        if not all(col in data_df.columns for col in required_cols_input):
            missing_cols = [col for col in required_cols_input if col not in data_df.columns]
            print(f"Error: Missing required columns in {INPUT_DATA_WITH_MASKS_PATH}: {missing_cols}")
            return

        # top_words_column 파싱 (이전 스크립트에서 문자열 리스트로 저장된 경우)
        import ast
        def safe_literal_eval(val):
            try:
                if isinstance(val, str) and val.strip().startswith('['): return ast.literal_eval(val)
                elif isinstance(val, list): return val # 이미 리스트인 경우
                else: return []
            except: return []
        data_df[TOP_WORDS_COLUMN] = data_df[TOP_WORDS_COLUMN].apply(safe_literal_eval)
        print(f"'{TOP_WORDS_COLUMN}' parsed.")

        # NaN 값 처리 (텍스트 컬럼들)
        data_df[TEXT_COLUMN] = data_df[TEXT_COLUMN].fillna("").astype(str)
        data_df[MASKED_TEXT_COLUMN] = data_df[MASKED_TEXT_COLUMN].fillna("").astype(str)

    except FileNotFoundError:
        print(f"Error: Input data file not found: {INPUT_DATA_WITH_MASKS_PATH}")
        return
    except Exception as e:
        print(f"Error loading or processing input data: {e}")
        return

    # 3. 어텐션 지표, MSP 차이, 특징 벡터 추출
    print("\n--- 3. Extracting attention metrics, MSP difference, and features ---")
    # 주의: data_df가 매우 클 경우, 모든 텍스트를 메모리에 올리는 것이 부담될 수 있음.
    # 필요시 data_df를 chunk로 나누어 처리하는 로직 고려.
    metrics_df, all_features_masked = extract_metrics_and_features(
        trained_model_pl,
        data_df[TEXT_COLUMN].tolist(),
        data_df[MASKED_TEXT_COLUMN].tolist(),
        tokenizer,
        current_device,
        BATCH_SIZE,
        MAX_LENGTH,
        TOP_K_ATTENTION
    )

    if len(data_df) == len(metrics_df):
        data_df = pd.concat([data_df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)
        print("Metrics (including MSP diff) added to DataFrame.")
    else:
        print(f"Warning: Length mismatch. Original data: {len(data_df)}, Metrics: {len(metrics_df)}")
        print("Skipping concatenation of metrics. This might affect subsequent steps.")
        # 오류를 발생시키거나, 일부 샘플만 사용하도록 처리 필요

    # 4. 제거된 단어의 어텐션 계산
    print("\n--- 4. Calculating attention scores for removed words ---")
    if TOP_WORDS_COLUMN in data_df.columns and TEXT_COLUMN in data_df.columns:
        print("Calculating 'removed_avg_attention'...")
        word_attentions_list = get_word_attention_scores_pl(
            data_df[TEXT_COLUMN].tolist(),
            trained_model_pl, # 이미 로드된 모델 사용
            tokenizer,
            current_device,
            max_length=MAX_LENGTH
        )
        removed_attentions_scores = []
        for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Calculating Removed Avg Attn"):
            top_words_sample = row[TOP_WORDS_COLUMN]
            if isinstance(top_words_sample, list) and top_words_sample and idx < len(word_attentions_list):
                sentence_word_attns = word_attentions_list[idx]
                current_removed_scores = [sentence_word_attns.get(word.lower(), 0) for word in top_words_sample]
                removed_attentions_scores.append(np.mean(current_removed_scores) if current_removed_scores else 0)
            else:
                removed_attentions_scores.append(0)
        data_df['removed_avg_attention'] = removed_attentions_scores
        print("'removed_avg_attention' calculated and added.")
    else:
        print(f"Skipping 'removed_avg_attention' calculation: Missing '{TOP_WORDS_COLUMN}' or '{TEXT_COLUMN}'.")
        if 'removed_avg_attention' not in data_df.columns: # 안전장치
            data_df['removed_avg_attention'] = 0.0


    # 5. 어텐션 및 MSP 지표 분포 시각화
    print("\n--- 5. Visualizing metric distributions ---")
    # 기존 metric_columns에 msp_difference 추가
    all_metric_columns_for_vis = ['max_attention', 'top_k_avg_attention', 'attention_entropy', 'removed_avg_attention', 'msp_difference', 'msp_original', 'msp_masked']

    for metric in all_metric_columns_for_vis:
        if metric in data_df.columns and not data_df[metric].isnull().all():
            plot_metric_distribution(
                data_df[metric].dropna().values,
                metric,
                f'Distribution of {metric}',
                os.path.join(VIS_DIR, f'{metric}_distribution.png')
            )
        else:
            print(f"Skipping visualization for '{metric}' (column not found or all null).")


    # 6. OE 데이터셋 추출 (개별 지표 기준 + 순차 필터링)
    print("\n--- 6. Extracting OE datasets ---")

    # 6.1. 개별 지표 기준 OE 데이터셋 추출
    print("\n--- 6.1. Extracting OE datasets based on individual metrics ---")
    for metric_key, settings in METRIC_SETTINGS.items():
        if metric_key not in data_df.columns or MASKED_TEXT_COLUMN not in data_df.columns:
            print(f"Skipping OE extraction for '{metric_key}': Column '{metric_key}' or '{MASKED_TEXT_COLUMN}' not found.")
            continue

        scores = data_df[metric_key].values
        # NaN을 0으로 처리 (percentile 계산에 영향 최소화) 또는 dropna 후 인덱스 매칭 필요
        scores = np.nan_to_num(scores, nan=np.nanmedian(scores) if not np.all(np.isnan(scores)) else 0.0) # NaN을 중앙값 또는 0으로 대체

        filter_percentile = settings['percentile']
        filter_mode = settings['mode']
        selected_indices = []

        if len(scores) == 0 : # 점수 배열이 비어있는 경우 방지
             print(f"No valid scores for metric '{metric_key}'. Skipping.")
             continue

        if filter_mode == 'higher':
            threshold = np.percentile(scores, 100 - filter_percentile)
            selected_indices = np.where(scores >= threshold)[0]
            mode_desc = f"top{filter_percentile}pct"
            print(f"Filtering for '{metric_key}' >= {threshold:.4f} ({mode_desc})...")
        elif filter_mode == 'lower':
            threshold = np.percentile(scores, filter_percentile)
            selected_indices = np.where(scores <= threshold)[0]
            mode_desc = f"bottom{filter_percentile}pct"
            print(f"Filtering for '{metric_key}' <= {threshold:.4f} ({mode_desc})...")

        if len(selected_indices) > 0:
            oe_df_filtered_individual = data_df.iloc[selected_indices][[MASKED_TEXT_COLUMN]].copy()
            extended_cols_individual = [TEXT_COLUMN, MASKED_TEXT_COLUMN, TOP_WORDS_COLUMN, CLASS_COLUMN, metric_key] + list(METRIC_SETTINGS.keys())
            extended_cols_individual = [col for col in extended_cols_individual if col in data_df.columns] # 존재하는 컬럼만
            oe_df_extended_individual = data_df.iloc[selected_indices][extended_cols_individual].copy()

            oe_filename_individual = os.path.join(OE_DATA_DIR, f"oe_data_{metric_key}_{mode_desc}.csv")
            oe_extended_filename_individual = os.path.join(OE_DATA_DIR, f"oe_data_{metric_key}_{mode_desc}_extended.csv")

            try:
                oe_df_filtered_individual.to_csv(oe_filename_individual, index=False)
                print(f"  OE dataset ({len(oe_df_filtered_individual)} samples) saved: {oe_filename_individual}")
                oe_df_extended_individual.to_csv(oe_extended_filename_individual, index=False)
                print(f"  Extended OE dataset saved: {oe_extended_filename_individual}")
            except Exception as e:
                print(f"  Error saving OE dataset for '{metric_key}': {e}")
        else:
            print(f"  No samples selected for OE dataset based on '{metric_key}' {mode_desc}.")


    # 6.2. 순차 필터링 기반 OE 데이터셋 추출
    print("\n--- 6.2. Extracting OE dataset based on sequential filtering ---")
    selected_mask_sequential = np.ones(len(data_df), dtype=bool)
    filter_desc_parts = []

    for filter_step, (metric_seq, settings_seq) in enumerate(SEQUENTIAL_FILTERING_SEQUENCE):
        if metric_seq not in data_df.columns or MASKED_TEXT_COLUMN not in data_df.columns:
            print(f"Sequential filter step {filter_step+1} for '{metric_seq}' skipped: Column missing.")
            continue

        current_selection_df = data_df[selected_mask_sequential]
        if current_selection_df.empty:
            print(f"Sequential filter step {filter_step+1} for '{metric_seq}' skipped: No samples left from previous steps.")
            break

        scores_seq = current_selection_df[metric_seq].values
        scores_seq = np.nan_to_num(scores_seq, nan=np.nanmedian(scores_seq) if not np.all(np.isnan(scores_seq)) else 0.0)

        if len(scores_seq) == 0 : # 점수 배열이 비어있는 경우 방지
             print(f"No valid scores for metric '{metric_seq}' in sequential step. Skipping.")
             continue


        step_mask_on_selection = np.zeros(len(current_selection_df), dtype=bool)
        if settings_seq['mode'] == 'higher':
            threshold_seq = np.percentile(scores_seq, 100 - settings_seq['percentile'])
            step_mask_on_selection = scores_seq >= threshold_seq
            print(f"Sequential filter {filter_step+1}: '{metric_seq}' >= {threshold_seq:.4f} (top {settings_seq['percentile']}%)")
        else: # 'lower'
            threshold_seq = np.percentile(scores_seq, settings_seq['percentile'])
            step_mask_on_selection = scores_seq <= threshold_seq
            print(f"Sequential filter {filter_step+1}: '{metric_seq}' <= {threshold_seq:.4f} (bottom {settings_seq['percentile']}%)")

        # 전체 마스크 업데이트: 현재 선택된 것들 중에서 다시 필터링
        current_indices = np.where(selected_mask_sequential)[0]
        filtered_indices_within_selection = current_indices[step_mask_on_selection]

        selected_mask_sequential = np.zeros_like(selected_mask_sequential)
        if len(filtered_indices_within_selection) > 0:
            selected_mask_sequential[filtered_indices_within_selection] = True

        filter_desc_parts.append(f"{metric_seq}_{settings_seq['mode']}{settings_seq['percentile']}")
        print(f"  Samples remaining after step {filter_step+1}: {np.sum(selected_mask_sequential)}")
        if np.sum(selected_mask_sequential) == 0: break # 더 이상 샘플이 없으면 중단

    final_selected_indices_sequential = np.where(selected_mask_sequential)[0]
    if len(final_selected_indices_sequential) > 0:
        oe_df_sequential = data_df.iloc[final_selected_indices_sequential][[MASKED_TEXT_COLUMN]].copy()
        extended_cols_seq = [TEXT_COLUMN, MASKED_TEXT_COLUMN, TOP_WORDS_COLUMN, CLASS_COLUMN] + [m for m, _ in SEQUENTIAL_FILTERING_SEQUENCE if m in data_df.columns]
        extended_cols_seq = [col for col in extended_cols_seq if col in data_df.columns] # 존재하는 컬럼만
        oe_df_extended_sequential = data_df.iloc[final_selected_indices_sequential][extended_cols_seq].copy()

        filter_desc_str = "_then_".join(filter_desc_parts)
        oe_filename_sequential = os.path.join(OE_DATA_DIR, f"oe_data_sequential_{filter_desc_str}.csv")
        oe_extended_filename_sequential = os.path.join(OE_DATA_DIR, f"oe_data_sequential_{filter_desc_str}_extended.csv")

        try:
            oe_df_sequential.to_csv(oe_filename_sequential, index=False)
            print(f"Sequential OE dataset ({len(oe_df_sequential)} samples) saved: {oe_filename_sequential}")
            oe_df_extended_sequential.to_csv(oe_extended_filename_sequential, index=False)
            print(f"Extended sequential OE dataset saved: {oe_extended_filename_sequential}")
        except Exception as e:
            print(f"Error saving sequential OE dataset: {e}")
    else:
        print("No samples selected after sequential filtering.")


    # 7. t-SNE 시각화
    print("\n--- 7. t-SNE Visualization ---")
    if len(all_features_masked) == len(data_df):
        # t-SNE용 레이블 준비: Known (ID), Unknown (OOD), OE Candidate
        tsne_labels_base = []
        # label2id_for_model은 모델 로드 시 사용된 매핑
        unknown_class_value_lower = EXCLUDE_CLASS_FOR_TRAINING.lower() # 모델 학습 시 제외된 클래스

        for cls_val in data_df[CLASS_COLUMN]:
            cls_str_lower = str(cls_val).lower()
            if cls_str_lower == unknown_class_value_lower:
                tsne_labels_base.append(-1) # Unknown (실제 OOD)
            elif cls_str_lower in label2id_for_model:
                tsne_labels_base.append(label2id_for_model[cls_str_lower]) # Known ID
            else:
                tsne_labels_base.append(-2) # 그 외 (필터링되었거나 레이블 없는 경우 등)

        tsne_labels_base = np.array(tsne_labels_base)
        
        # id2label_for_model에 Unknown, Other 추가
        tsne_class_names_map = {**id2label_for_model, -1: 'OOD (Unknown)', -2: 'Other/Unlabeled'}


        # 7.1 개별 지표 기준 OE 후보 t-SNE
        for metric_key, settings in METRIC_SETTINGS.items():
            if metric_key not in data_df.columns: continue # 해당 지표가 계산 안됐으면 스킵
            
            scores_tsne = np.nan_to_num(data_df[metric_key].values, nan=np.nanmedian(data_df[metric_key].values) if not np.all(np.isnan(data_df[metric_key].values)) else 0.0)
            oe_candidate_indices_tsne = []
            
            if len(scores_tsne) == 0: continue

            if settings['mode'] == 'higher':
                threshold_tsne = np.percentile(scores_tsne, 100 - settings['percentile'])
                oe_candidate_indices_tsne = np.where(scores_tsne >= threshold_tsne)[0]
            else: # 'lower'
                threshold_tsne = np.percentile(scores_tsne, settings['percentile'])
                oe_candidate_indices_tsne = np.where(scores_tsne <= threshold_tsne)[0]

            if len(oe_candidate_indices_tsne) > 0:
                 plot_tsne(
                    features=np.array(all_features_masked),
                    labels=tsne_labels_base,
                    title=f't-SNE (Features from Masked Text)\nOE Candidates by {metric_key} ({settings["mode"]} {settings["percentile"]}%)',
                    save_path=os.path.join(VIS_DIR, f'tsne_oe_cand_{metric_key}_{settings["mode"]}{settings["percentile"]}pct.png'),
                    highlight_indices=oe_candidate_indices_tsne,
                    highlight_label=f'OE Candidate ({metric_key})',
                    class_names=tsne_class_names_map,
                    seed=RANDOM_STATE
                )
            else:
                print(f"No OE candidates for t-SNE based on {metric_key}. Skipping plot.")

        # 7.2 순차 필터링 기준 OE 후보 t-SNE
        if len(final_selected_indices_sequential) > 0:
            seq_filter_desc_for_plot = " + ".join([f"{m}_{s['mode']}{s['percentile']}" for m,s in SEQUENTIAL_FILTERING_SEQUENCE])
            plot_tsne(
                features=np.array(all_features_masked),
                labels=tsne_labels_base,
                title=f't-SNE (Features from Masked Text)\nOE Candidates by Sequential Filtering: {seq_filter_desc_for_plot}',
                save_path=os.path.join(VIS_DIR, f'tsne_oe_cand_sequential_{"_then_".join(filter_desc_parts)}.png'),
                highlight_indices=final_selected_indices_sequential,
                highlight_label=f'OE Candidate (Sequential)',
                class_names=tsne_class_names_map,
                seed=RANDOM_STATE
            )
        else:
             print("No OE candidates from sequential filtering for t-SNE. Skipping plot.")

    else:
        print(f"Skipping t-SNE: Feature vector count ({len(all_features_masked)}) and data_df count ({len(data_df)}) mismatch.")


    # 8. 결과 요약 및 저장 (필요시 추가적인 요약 정보 출력)
    print("\n--- 8. Results Summary ---")
    print(f"Extracted OE datasets are saved in: {OE_DATA_DIR}")
    print(f"Visualizations are saved in: {VIS_DIR}")

    # 어떤 기준으로 몇 개의 OE 샘플이 추출되었는지 요약
    print("\nSummary of extracted OE samples:")
    for fname in os.listdir(OE_DATA_DIR):
        if fname.endswith(".csv") and not fname.endswith("_extended.csv"):
            try:
                df_oe_summary = pd.read_csv(os.path.join(OE_DATA_DIR, fname))
                print(f"- {fname}: {len(df_oe_summary)} samples")
            except:
                print(f"- Could not read summary for {fname}")

    print("\nScript execution completed.")

if __name__ == '__main__':
    main()