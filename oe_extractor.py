import pandas as pd
import numpy as np
import re
from collections import defaultdict
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
from nltk.tokenize import word_tokenize
import string
import torch
import torch.optim as optim
import torch.nn as nn
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
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

import os
from tqdm.auto import tqdm
import gc
import math

# --- 설정값 ---
#FILE_PATH = 'log_all_critical.csv'
FILE_PATH = 'log_all.csv'
TEXT_COLUMN = 'text'
CLASS_COLUMN = 'class' # 범주 정보 컬럼
TRAIN_TEST_COLUMN = 'train/test' # 데이터 분할에 사용 안 함 (참고용으로만 남김)
EXCLUDE_CLASS_FOR_TRAINING = "unknown" # 학습에서 제외할 클래스
ATTENTION_TOP_PERCENT = 0.10
MIN_TOP_WORDS = 1
MASKED_OUTPUT_FILE_PATH_ATTN = 'log_all_critical_attention_masked_for_oe.csv' # 최종 출력 파일
MODEL_NAME = "roberta-base"
MODEL_SAVE_DIR = "./roberta_base_log_classifier_known_classes" # 모델 저장 경로
LOG_DIR = "./lightning_logs_roberta_base_known_classes" # 로그 저장 경로
CONFUSION_MATRIX_DIR = os.path.join(LOG_DIR, "confusion_matrices") # 검증 Confusion Matrix 저장
NUM_TRAIN_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MIN_SAMPLES_PER_CLASS_FOR_TRAIN_VAL = 3 # 학습/검증에 사용할 클래스의 최소 샘플 수
ACCELERATOR = "auto"
DEVICES = "auto"
PRECISION = "16-mixed" if torch.cuda.is_available() and ACCELERATOR != "cpu" else "32-true"
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1
LOG_EVERY_N_STEPS = 50
GRADIENT_CLIP_VAL = 1.0
USE_WEIGHTED_LOSS = True
USE_LR_SCHEDULER = True
RANDOM_STATE = 42

os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- 도우미 함수 (이전과 동일) ---
def preprocess_text_for_roberta(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\s+', ' ', text).strip(); return text

def tokenize_nltk(text):
    if not text: return []
    try: return word_tokenize(text)
    except Exception: return text.split()

def create_masked_sentence_per_sentence(original_text, sentence_important_words):
    if not isinstance(original_text, str): return ""
    if not sentence_important_words: return original_text
    processed_text = preprocess_text_for_roberta(original_text)
    tokens = tokenize_nltk(processed_text)
    important_set_lower = {word.lower() for word in sentence_important_words}
    masked_tokens = [word for word in tokens if word.lower() not in important_set_lower]
    masked_sentence = ' '.join(masked_tokens)
    if not masked_sentence: return "__EMPTY_MASKED__"
    return masked_sentence

# --- PyTorch Lightning DataModule (수정됨) ---
class LogDataModuleForKnownClasses(pl.LightningDataModule):
    # label_col, train_test_col 제거
    def __init__(self, file_path, text_col, class_col, exclude_class,
                 model_name, batch_size, min_samples_per_class=3, num_workers=1, random_state=42, use_weighted_loss=False):
        super().__init__()
        self.save_hyperparameters("file_path", "text_col", "class_col", "exclude_class",
                                  "model_name", "batch_size", "min_samples_per_class", "num_workers", "random_state", "use_weighted_loss")
        print(f"DataModule: Initializing tokenizer for {self.hparams.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.df_full = None # 전체 원본 데이터
        self.df_known_for_train_val = None # 학습/검증 분할 대상 데이터
        self.train_df_final = None
        self.val_df_final = None
        self.label2id = None; self.id2label = None; self.num_labels = None
        self.tokenized_train_val_datasets = None # 학습/검증용 토큰화 데이터
        self.class_weights = None

    def prepare_data(self): pass

    def setup(self, stage=None):
        if self.df_full is None:
            print(f"DataModule: Loading data from {self.hparams.file_path}")
            self.df_full = pd.read_csv(self.hparams.file_path)
            required_cols = [self.hparams.text_col, self.hparams.class_col] # train_test_col은 더 이상 필수 아님
            if not all(col in self.df_full.columns for col in required_cols): raise ValueError(f"CSV 누락 컬럼: {required_cols}")

            # 결측치 처리 (class_col 기준)
            self.df_full = self.df_full.dropna(subset=[self.hparams.class_col])
            self.df_full[self.hparams.class_col] = self.df_full[self.hparams.class_col].astype(str).str.lower()
            # 제외할 클래스명도 소문자로 변환
            exclude_class_lower = self.hparams.exclude_class.lower() if self.hparams.exclude_class else None

            # --- 학습/검증 데이터 준비 ---
            # 1. 특정 클래스 제외
            print(f"DataModule: Excluding class '{self.hparams.exclude_class}' for training/validation.")
            df_known = self.df_full[self.df_full[self.hparams.class_col] != self.hparams.exclude_class].copy()
            print(f"DataModule: Data size after excluding '{self.hparams.exclude_class}': {len(df_known)}")

            # 2. 레이블 매핑 (Known 클래스 기준)
            known_classes_str = sorted(df_known[self.hparams.class_col].unique())
            self.label2id = {label: i for i, label in enumerate(known_classes_str)}
            self.id2label = {i: label for label, i in self.label2id.items()}
            # !!! 중요: num_labels는 학습/검증에 사용되는 클래스 수 !!!
            self.num_labels = len(known_classes_str)
            print(f"\nDataModule - 레이블 매핑 완료 (Known Classes): {self.num_labels}개 클래스")
            print(f"Known Class Label to ID mapping: {self.label2id}")

            # 'label' 컬럼 생성 (Known 데이터에만)
            df_known['label'] = df_known[self.hparams.class_col].map(self.label2id)
            if df_known['label'].isnull().any(): # 혹시 모를 오류 처리
                print("경고: Known class -> label 매핑 실패. NaN 값 포함 행 제외.")
                df_known = df_known.dropna(subset=['label'])
            df_known['label'] = df_known['label'].astype(int)

            # 3. 최소 샘플 수 필터링 (Known 클래스 내에서)
            print(f"\nDataModule - Known 클래스 데이터 필터링 (클래스당 최소 {self.hparams.min_samples_per_class}개)...")
            label_counts_known = df_known['label'].value_counts()
            labels_to_keep = label_counts_known[label_counts_known >= self.hparams.min_samples_per_class].index
            self.df_known_for_train_val = df_known[df_known['label'].isin(labels_to_keep)].copy()
            print(f"DataModule - 필터링 후 학습/검증 대상 데이터: {len(self.df_known_for_train_val)} 행")
            if len(self.df_known_for_train_val) == 0: raise ValueError("필터링 후 학습/검증 대상 데이터 없음.")

            print("\n--- 학습/검증 대상 데이터 클래스 분포 ---")
            print(self.df_known_for_train_val['label'].map(self.id2label).value_counts())

            # 4. 클래스 가중치 계산 (필터링된 Known 데이터 기준)
            if self.hparams.use_weighted_loss:
                labels_for_weights = self.df_known_for_train_val['label'].values
                unique_labels_in_train_val = np.unique(labels_for_weights)
                try:
                    class_weights_array = compute_class_weight('balanced', classes=unique_labels_in_train_val, y=labels_for_weights)
                    # num_labels (Known 클래스 수) 에 맞춰 텐서 생성
                    self.class_weights = torch.ones(self.num_labels)
                    for i, label_idx in enumerate(unique_labels_in_train_val):
                        if label_idx < self.num_labels: self.class_weights[label_idx] = class_weights_array[i]
                    print(f"\nDataModule - 계산된 클래스 가중치: {self.class_weights}")
                except ValueError as e:
                    print(f"클래스 가중치 계산 오류: {e}. 가중치 없이 진행."); self.hparams.use_weighted_loss = False; self.class_weights = None

            # 5. 학습/검증 분할 (df_known_for_train_val 사용)
            print("\nDataModule - 학습/검증 데이터 분할...")
            try:
                self.train_df_final, self.val_df_final = train_test_split(
                    self.df_known_for_train_val, test_size=0.2,
                    random_state=self.hparams.random_state, stratify=self.df_known_for_train_val['label'])
            except ValueError:
                print("경고: Stratify 분할 실패. Stratify 없이 분할.")
                self.train_df_final, self.val_df_final = train_test_split(
                    self.df_known_for_train_val, test_size=0.2, random_state=self.hparams.random_state)
            print(f"DataModule - 최종 학습셋: {len(self.train_df_final)}, 최종 검증셋: {len(self.val_df_final)}")

            # 6. Hugging Face Dataset 생성 (학습/검증용)
            raw_train_val_datasets = DatasetDict({
                'train': Dataset.from_pandas(self.train_df_final),
                'validation': Dataset.from_pandas(self.val_df_final)
                # 'test' 는 여기서는 필요 없음
            })

            # 7. 토큰화 (학습/검증 데이터만)
            def tokenize_func(examples):
                return self.tokenizer([preprocess_text_for_roberta(text) for text in examples[self.hparams.text_col]],
                                      truncation=True, padding=False, max_length=self.tokenizer.model_max_length)
            print("\nDataModule - 학습/검증 데이터셋 토큰화 중...")
            self.tokenized_train_val_datasets = raw_train_val_datasets.map(tokenize_func, batched=True, num_proc=max(1, self.hparams.num_workers // 2),
                                                                        remove_columns=[col for col in raw_train_val_datasets['train'].column_names if col != 'label'])
            self.tokenized_train_val_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            print("DataModule - 토큰화 완료.")

    # DataLoader는 학습/검증용만 정의
    def train_dataloader(self):
        if self.tokenized_train_val_datasets is None: self.setup()
        return DataLoader(self.tokenized_train_val_datasets['train'], batch_size=self.hparams.batch_size, collate_fn=self.data_collator, num_workers=self.hparams.num_workers, shuffle=True, pin_memory=True, persistent_workers=self.hparams.num_workers > 0)

    def val_dataloader(self):
        if self.tokenized_train_val_datasets is None: self.setup()
        return DataLoader(self.tokenized_train_val_datasets['validation'], batch_size=self.hparams.batch_size, collate_fn=self.data_collator, num_workers=self.hparams.num_workers, pin_memory=True, persistent_workers=self.hparams.num_workers > 0)

    # test_dataloader는 필요 없음

    def get_full_dataframe(self):
         # 어텐션 계산 및 최종 저장을 위해 전체 원본 데이터 반환
         if self.df_full is None: self.setup()
         return self.df_full

# --- PyTorch Lightning Module (수정됨) ---
class LogClassifierPL(pl.LightningModule):
    # __init__에서 class_weights를 받도록 유지
    def __init__(self, model_name, num_labels, label2id, id2label, learning_rate=2e-5,
                 use_weighted_loss=False, class_weights=None, use_lr_scheduler=False, warmup_steps=0):
        super().__init__()
        self.save_hyperparameters("model_name", "num_labels", "label2id", "id2label", "learning_rate",
                                  "use_weighted_loss", "class_weights", "use_lr_scheduler", "warmup_steps")
        print(f"LightningModule: Initializing model {self.hparams.model_name} for {self.hparams.num_labels} known classes.")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hparams.model_name, num_labels=self.hparams.num_labels, # Known 클래스 수
            label2id=self.hparams.label2id, id2label=self.hparams.id2label, # Known 클래스 매핑
            ignore_mismatched_sizes=True
        )
        # Loss 함수 정의
        if self.hparams.use_weighted_loss and self.hparams.class_weights is not None:
            weights = torch.tensor(self.hparams.class_weights, dtype=torch.float)
            self.loss_fn = nn.CrossEntropyLoss(weight=weights)
            print("LightningModule: Using Weighted CrossEntropyLoss")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            print("LightningModule: Using Standard CrossEntropyLoss")

        # 메트릭 정의 (Known 클래스 수 기준)
        metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_labels),
            'f1_weighted': torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_labels, average='weighted'),
            'f1_macro': torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_labels, average='macro')
        })
        # Confusion Matrix도 Known 클래스 수 기준
        cm_metric = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_labels)
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        # test 관련 메트릭/CM 제거
        # self.test_metrics = metrics.clone(prefix='test_')
        self.val_cm = cm_metric.clone()
        # self.test_cm = cm_metric.clone()

    def setup(self, stage=None):
        if self.hparams.use_weighted_loss and isinstance(self.loss_fn, nn.CrossEntropyLoss) and self.loss_fn.weight is not None:
            self.loss_fn.weight = self.loss_fn.weight.to(self.device)
            print(f"LightningModule: Moved class weights to device {self.device}")

    def forward(self, batch):
        input_ids = batch.get('input_ids'); attention_mask = batch.get('attention_mask')
        if input_ids is None or attention_mask is None: raise ValueError("Batch missing 'input_ids' or 'attention_mask'")
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def _common_step(self, batch, batch_idx):
        labels = batch['labels']
        logits = self(batch)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

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

    # test_step 제거
    # def test_step(self, batch, batch_idx): ...

    def on_validation_epoch_end(self):
        try:
            val_cm_computed = self.val_cm.compute()
            print("\nValidation Confusion Matrix (Known Classes):")
            # 클래스 이름 가져오기 (Known 클래스 기준)
            known_class_names = list(self.hparams.id2label.values())
            cm_df = pd.DataFrame(val_cm_computed.cpu().numpy(), index=known_class_names, columns=known_class_names)
            print(cm_df)
            # 검증 CM 저장 (선택적)
            cm_filename = os.path.join(CONFUSION_MATRIX_DIR, f"validation_confusion_matrix_epoch_{self.current_epoch}.csv")
            cm_df.to_csv(cm_filename)
            print(f"Validation Confusion Matrix saved to: {cm_filename}")
        except Exception as e: print(f"Error computing/printing/saving validation confusion matrix: {e}")
        finally: self.val_cm.reset()

    # on_test_epoch_end 제거
    # def on_test_epoch_end(self): ...

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.use_lr_scheduler:
            # 스케줄러 설정
            if self.trainer and hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches > 0:
                 num_training_steps = self.trainer.estimated_stepping_batches
                 print(f"LightningModule: Estimated training steps for LR scheduler: {num_training_steps}")
                 scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=num_training_steps)
                 return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
            else: print("Warning: Could not estimate training steps for LR scheduler. Using optimizer only."); return optimizer
        else: return optimizer

# --- 어텐션 추출 함수 (이전과 동일) ---
def get_word_attention_scores_pl(batch_texts, model_pl: LogClassifierPL, tokenizer, device, layer=-1):
    # ... (이전 get_word_attention_scores_pl 함수 전체 복사) ...
    hf_model = model_pl.model; hf_model.eval()
    if not batch_texts: return []
    processed_batch = [preprocess_text_for_roberta(text) for text in batch_texts]
    inputs = tokenizer(processed_batch, return_tensors='pt', truncation=True, max_length=tokenizer.model_max_length, padding=True, return_offsets_mapping=True)
    offset_mappings = inputs.pop('offset_mapping').cpu().numpy()
    input_ids_batch = inputs['input_ids'].cpu().numpy()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    batch_word_scores = []
    with torch.no_grad():
        outputs = hf_model(**inputs, output_attentions=True)
        attentions_batch = outputs.attentions
    attention_layer_batch = attentions_batch[layer].cpu().numpy()
    for i in range(len(batch_texts)):
        attention_layer_sample = attention_layer_batch[i]; offset_mapping_sample = offset_mappings[i]
        input_ids_sample = input_ids_batch[i]; original_text_sample = processed_batch[i]
        attention_heads_mean = np.mean(attention_layer_sample, axis=0)
        token_attentions = attention_heads_mean[0, :]
        tokens = tokenizer.convert_ids_to_tokens(input_ids_sample)
        word_scores = defaultdict(list); current_word_indices = []; last_word_end_offset = 0
        for j, (token_id, offset) in enumerate(zip(input_ids_sample, offset_mapping_sample)):
            if offset[0] == offset[1] or token_id in tokenizer.all_special_ids: continue
            is_subword_or_continuation = (j > 0 and offset[0] == last_word_end_offset)
            if not is_subword_or_continuation and current_word_indices:
                start = offset_mapping_sample[current_word_indices[0]][0]; end = offset_mapping_sample[current_word_indices[-1]][1]
                try:
                    word = original_text_sample[start:end]; avg_score = np.mean(token_attentions[current_word_indices])
                    if word.strip(): word_scores[word.strip()].append(avg_score)
                except IndexError: pass
                current_word_indices = []
            current_word_indices.append(j); last_word_end_offset = offset[1]
        if current_word_indices:
            start = offset_mapping_sample[current_word_indices[0]][0]; end = offset_mapping_sample[current_word_indices[-1]][1]
            try:
                 word = original_text_sample[start:end]; avg_score = np.mean(token_attentions[current_word_indices])
                 if word.strip(): word_scores[word.strip()].append(avg_score)
            except IndexError: pass
        final_word_scores_sample = {word: np.mean(scores) for word, scores in word_scores.items()}
        batch_word_scores.append(final_word_scores_sample)
    del inputs, outputs, attentions_batch, attention_layer_batch
    return batch_word_scores

# --- 메인 실행 로직 ---
if __name__ == '__main__':
    pl.seed_everything(RANDOM_STATE)

    # 1. 데이터 모듈 초기화 및 설정
    print("--- 1. 데이터 모듈 설정 ---")
    log_data_module = LogDataModuleForKnownClasses( # 수정된 DataModule 사용
        file_path=FILE_PATH, text_col=TEXT_COLUMN, class_col=CLASS_COLUMN,
        exclude_class=EXCLUDE_CLASS_FOR_TRAINING, # 제외할 클래스 전달
        model_name=MODEL_NAME, batch_size=BATCH_SIZE,
        min_samples_per_class=MIN_SAMPLES_PER_CLASS_FOR_TRAIN_VAL, # 학습/검증용 최소 샘플 수
        num_workers=NUM_WORKERS, random_state=RANDOM_STATE,
        use_weighted_loss=USE_WEIGHTED_LOSS
    )
    log_data_module.setup() # 데이터 로드, 필터링, 분할, 토큰화, 가중치 계산 실행

    # 2. 모델 모듈 초기화 (Known 클래스 수 사용)
    print("\n--- 2. 모델 모듈 초기화 ---")
    model_module = LogClassifierPL(
        model_name=MODEL_NAME, num_labels=log_data_module.num_labels, # Known 클래스 수
        label2id=log_data_module.label2id, id2label=log_data_module.id2label, # Known 클래스 매핑
        learning_rate=LEARNING_RATE,
        use_weighted_loss=USE_WEIGHTED_LOSS,
        class_weights=log_data_module.class_weights,
        use_lr_scheduler=USE_LR_SCHEDULER,
        warmup_steps=0
    )

    # 3. 콜백 설정
    print("\n--- 3. 콜백 설정 ---")
    monitor_metric = 'val_f1_macro'
    monitor_mode = 'max'
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_SAVE_DIR, filename=f'best-model-{{epoch:02d}}-{{{monitor_metric}:.4f}}',
        save_top_k=1, monitor=monitor_metric, mode=monitor_mode
    )
    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric, patience=3, mode=monitor_mode, verbose=True
    )
    # progress_bar = TQDMProgressBar(refresh_rate=10)

    # 4. 로거 설정
    csv_logger = CSVLogger(save_dir=LOG_DIR, name="roberta_base_known_classifier")

    # 5. 트레이너 설정 및 학습
    print("\n--- 4. 트레이너 설정 및 학습 시작 ---")
    trainer = pl.Trainer(
        max_epochs=NUM_TRAIN_EPOCHS, accelerator=ACCELERATOR, devices=DEVICES,
        precision=PRECISION, logger=csv_logger,
        callbacks=[checkpoint_callback, early_stopping_callback], # progress_bar],
        deterministic=False, log_every_n_steps=LOG_EVERY_N_STEPS,
        gradient_clip_val=GRADIENT_CLIP_VAL
    )
    # trainer.fit 호출 시 datamodule 전달
    trainer.fit(model_module, datamodule=log_data_module)
    print("--- 모델 학습 완료 ---")

    # 6. 최적 모델 로드 (테스트 단계는 생략)
    print("\n--- 5. 최적 모델 로드 ---")
    best_model_path = checkpoint_callback.best_model_path
    print(f"최적 모델 경로: {best_model_path}")
    best_model_module = None
    if best_model_path and os.path.exists(best_model_path):
         print("최적 모델 로드 중...")
         # load_from_checkpoint 시 hparams 자동 로드됨
         # strict=False 추가하여 가중치 이름 불일치 시에도 로드 시도 (필요시)
         best_model_module = LogClassifierPL.load_from_checkpoint(best_model_path, strict=False)
         print("최적 모델 로드 완료.")
    else:
         print("경고: 최적 모델 경로를 찾을 수 없습니다. 마지막 상태 모델을 사용합니다.")
         best_model_module = model_module # 마지막 학습 상태 모델 사용

    # --- 7. 어텐션 추출 및 문장별 중요 단어 식별 ---
    print("\n--- 6. 어텐션 추출 및 문장별 중요 단어 식별 ---")
    if best_model_module is None:
        print("오류: 사용할 모델이 없어 어텐션 추출 및 마스킹을 건너<0xEB><0x9B><0x8D>니다.")
    else:
        attention_model_pl = best_model_module
        current_device = torch.device("cuda" if torch.cuda.is_available() and ACCELERATOR != "cpu" else "cpu")
        attention_model_pl.to(current_device)
        attention_model_pl.eval()
        attention_model_pl.freeze()

        # 어텐션 계산 대상: 전체 원본 데이터프레임 사용
        target_df = log_data_module.get_full_dataframe().reset_index(drop=True)
        print(f"어텐션 스코어 계산 대상 데이터: {len(target_df)}개 행")

        attention_batch_size = BATCH_SIZE
        num_batches = (len(target_df) + attention_batch_size - 1) // attention_batch_size
        tokenizer = log_data_module.tokenizer # DataModule에서 tokenizer 가져오기
        all_sentence_top_words = [None] * len(target_df)

        print("문장별 어텐션 스코어 계산 및 상위 단어 추출 중...")
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
        for i in tqdm(range(num_batches), desc="Calculating Sentence Attentions"):
            start_idx = i * attention_batch_size
            end_idx = min((i + 1) * attention_batch_size, len(target_df))
            # DataFrame에서 직접 텍스트 추출
            batch_texts = target_df.iloc[start_idx:end_idx][TEXT_COLUMN].tolist()
            if not batch_texts: continue
            try:
                batch_scores_list = get_word_attention_scores_pl(batch_texts, attention_model_pl, tokenizer, current_device)
                for j, sentence_scores in enumerate(batch_scores_list):
                    if not sentence_scores: all_sentence_top_words[start_idx + j] = []; continue
                    sorted_sentence_words = sorted(sentence_scores.items(), key=lambda item: item[1], reverse=True)
                    num_words_in_sentence = len(sorted_sentence_words)
                    n_top = max(MIN_TOP_WORDS, math.ceil(num_words_in_sentence * ATTENTION_TOP_PERCENT))
                    # 불용어 제외 로직 추가 (선택적)
                    top_n_words = [word for word, score in sorted_sentence_words[:n_top] if word.lower() not in ['__arg__', '__num__', '__id__', '__addr__', '__path__', '__netif__', '__version__', '__user__', 'a', 'an', 'the', 'is', 'was', 'on', 'in', 'at', 'to', 'of', 'for', 'and', 'or', 'but', 'error', 'failed', 'failure', 'critical', 'warning', 'device', 'system', 'detected', 'has', 'been', 'not', 'are', 'with', 'due', 'because', 'than', 'its', 'from', 'this', 'that', 'will', 'be']]
                    # 만약 불용어 제외 후 단어가 없으면 원래 상위 단어 유지 (선택적)
                    if not top_n_words and sorted_sentence_words:
                        top_n_words = [word for word, score in sorted_sentence_words[:n_top]]

                    all_sentence_top_words[start_idx + j] = top_n_words
            except RuntimeError as e:
                 if "CUDA out of memory" in str(e): print(f"\n배치 {i+1}/{num_batches} CUDA 메모리 부족!"); torch.cuda.empty_cache(); gc.collect(); continue
                 else: print(f"\n배치 {i+1}/{num_batches} 런타임 오류: {e}"); continue
            except Exception as e: print(f"\n배치 {i+1}/{num_batches} 예외 발생: {e}"); continue
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

        # target_df는 전체 데이터프레임이므로 여기에 컬럼 추가
        target_df['top_attention_words'] = all_sentence_top_words
        print("\n문장별 상위 어텐션 단어 추출 완료.")

        print("\n문장별 상위 어텐션 단어 예시:")
        sample_indices = target_df.sample(min(5, len(target_df))).index
        for i in sample_indices:
            print("-" * 20); print(f"Original: {target_df.loc[i, TEXT_COLUMN]}")
            print(f"Top Words: {target_df.loc[i, 'top_attention_words']}")
        del attention_model_pl
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
        print("\n어텐션 계산 모델 메모리 해제 완료.")

        # --- 8. 마스크된 텍스트 생성 및 저장 ---
        print("\n--- 7. 마스크된 텍스트 생성 및 저장 ---")
        print("마스크된 텍스트 생성 중...")
        # target_df (전체 데이터 + top_words)에 마스킹 적용
        target_df['masked_text_attention'] = target_df.apply(
            lambda row: create_masked_sentence_per_sentence(row[TEXT_COLUMN], row['top_attention_words'] if isinstance(row['top_attention_words'], list) else []),
            axis=1
        )
        print("\n마스크된 텍스트 예시:")
        sample_indices = target_df.sample(min(10, len(target_df))).index
        for i in sample_indices:
            print("-" * 20); print(f"Class: {target_df.loc[i, CLASS_COLUMN]}");
            print(f"Original: {target_df.loc[i, TEXT_COLUMN]}");
            if 'top_attention_words' in target_df.columns: print(f"Top Words: {target_df.loc[i, 'top_attention_words']}")
            print(f"Masked (Attn): {target_df.loc[i, 'masked_text_attention']}")

        print(f"\n최종 결과 저장 중: {MASKED_OUTPUT_FILE_PATH_ATTN}...")
        try:
            # 저장할 컬럼 목록 정의 ('label' 컬럼은 내부용이었으므로 제외 가능, CLASS_COLUMN 유지)
            columns_to_save = [TEXT_COLUMN, CLASS_COLUMN, TRAIN_TEST_COLUMN] # 원본 train/test 정보 유지
            if 'top_attention_words' in target_df.columns: columns_to_save.append('top_attention_words')
            columns_to_save.append('masked_text_attention')
            # 원본 CSV의 다른 컬럼들도 필요하면 여기에 추가 (예: severity 등)
            # original_cols = ['train/test', 'log type', 'severity', 'Reference', 'label_4class', 'label_node', 'label_job', 'label_total']
            # for col in original_cols:
            #     if col in target_df.columns and col not in columns_to_save:
            #         columns_to_save.append(col)

            df_to_save = target_df[columns_to_save]
            df_to_save.to_csv(MASKED_OUTPUT_FILE_PATH_ATTN, index=False, encoding='utf-8-sig')
            print("파일 저장 성공.")
        except KeyError as e:
            print(f"파일 저장 오류: 컬럼 찾기 실패 - {e}. 저장 시도 컬럼: {columns_to_save}")
            print("사용 가능 컬럼:", target_df.columns.tolist())
        except Exception as e: print(f"파일 저장 중 예상치 못한 오류: {e}")

    print("\n스크립트 실행 완료.")