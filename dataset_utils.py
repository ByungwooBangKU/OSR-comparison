# -*- coding: utf-8 -*-
"""
데이터셋 다운로드 및 전처리 유틸리티 모듈
(Dataset download and preprocessing utility module)
"""
import os
import re
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import requests # download_file 함수를 위해 추가
import shutil
from tqdm import tqdm # download_file 함수를 위해 추가
from datasets import load_dataset, ClassLabel # Hugging Face datasets를 위해 추가
from sklearn.datasets import fetch_20newsgroups # newsgroup20 용도
from sklearn.preprocessing import LabelEncoder # custom_syslog, 50_class_reviews(예시) 용도

# --- 상수 ---
DATA_DIR = "data" # 다운로드된 데이터의 기본 디렉토리

# --- 유틸리티 함수 ---
def ensure_dir(directory):
    """디렉토리가 존재하지 않으면 생성합니다."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_file(url, file_path, description=None):
    """파일을 다운로드하고 진행 상황을 표시합니다."""
    try:
        if not description: description = os.path.basename(file_path)
        dirname = os.path.dirname(file_path)
        if dirname: ensure_dir(dirname)

        if os.path.exists(file_path):
            # print(f"'{os.path.basename(file_path)}' 파일이 이미 존재합니다. 다운로드를 건너뜁니다.") # 로그 간소화
            return True

        print(f"'{description}' 다운로드 중... ({url})")
        # User-Agent 추가 (일부 서버에서 차단 방지)
        response = requests.get(url, stream=True, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        total_size = int(response.headers.get('content-length', 0))
        # 진행률 표시를 위한 블록 크기 (1MB)
        block_size = 1024 * 1024

        with open(file_path, 'wb') as f, tqdm(
                desc=description, total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                f.write(data)

        # 다운로드 완료 후 크기 검증 (선택 사항)
        if total_size != 0 and bar.n != total_size:
             print(f"경고: 다운로드된 파일 크기 불일치! ({bar.n}/{total_size} bytes)")
             # 필요 시 파일 삭제 로직 추가 가능
             # os.remove(file_path)
             # return False
        print(f"'{os.path.basename(file_path)}' 다운로드 완료.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"다운로드 중 네트워크 오류 발생 ({url}): {e}")
        if os.path.exists(file_path): os.remove(file_path) # 실패 시 불완전 파일 삭제
        return False
    except Exception as e:
        print(f"다운로드 중 예상치 못한 오류 발생 ({url}): {e}")
        if os.path.exists(file_path): os.remove(file_path)
        return False

def load_csv_universal(file_path, text_col_candidates, label_col_candidates):
    """다양한 컬럼 이름을 가진 CSV 파일을 유연하게 로드합니다."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV 파일 없음: {file_path}")

    try:
        # 여러 인코딩 시도 (UTF-8 기본, 실패 시 latin1)
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            print(f"  경고: UTF-8 디코딩 실패. latin1으로 재시도 중... ({os.path.basename(file_path)})")
            df = pd.read_csv(file_path, encoding='latin1')
    except Exception as e:
        raise ValueError(f"CSV 파일 로딩 오류 ({file_path}): {e}") from e

    # 후보군 중에서 실제 존재하는 텍스트/레이블 컬럼 찾기
    text_col = next((col for col in text_col_candidates if col in df.columns), None)
    label_col = next((col for col in label_col_candidates if col in df.columns), None)

    if not text_col: raise ValueError(f"텍스트 열을 찾을 수 없음 (후보: {text_col_candidates}) in {file_path}")
    if not label_col: raise ValueError(f"레이블 열을 찾을 수 없음 (후보: {label_col_candidates}) in {file_path}")

    # print(f"'{os.path.basename(file_path)}' 로드 완료. Text: '{text_col}', Label: '{label_col}'.") # 로그 간소화
    # 텍스트 컬럼을 문자열로 변환하고 결측치 처리
    texts = df[text_col].fillna('').astype(str).tolist()
    # 레이블 컬럼도 문자열로 변환 (인코딩 전 일관성 유지)
    classes = df[label_col].fillna('unknown').astype(str).tolist()
    return texts, classes


# --- 데이터셋별 다운로드 및 준비 함수 ---

# 1. ACM (변경 없음)
def download_acm_dataset(output_dir=DATA_DIR):
    """Downloads ACM dataset, extracts, and saves as CSV."""
    dataset_name = "acm"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    zip_url = "https://zenodo.org/records/7555249/files/acm.zip"
    zip_path = os.path.join(dataset_dir, "acm.zip")
    csv_path = os.path.join(dataset_dir, "acm.csv")
    txt_file_name = "texts.txt"
    score_file_name = "score.txt"

    if os.path.exists(csv_path):
        # print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path

    if not os.path.exists(zip_path):
        if not download_file(zip_url, zip_path, "ACM ZIP"):
            raise ConnectionError("ACM ZIP 다운로드 실패.")

    print(f"'{os.path.basename(zip_path)}' 압축 해제 및 처리 중...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            members = z.namelist()
            txt_member = next((m for m in members if m.endswith(txt_file_name)), None)
            score_member = next((m for m in members if m.endswith(score_file_name)), None)
            if not txt_member or not score_member:
                raise FileNotFoundError(f"ZIP 파일 내에서 '{txt_file_name}' 또는 '{score_file_name}'을 찾을 수 없습니다.")
            texts = z.read(txt_member).decode("utf-8").splitlines()
            labels = z.read(score_member).decode("utf-8").splitlines()
        pd.DataFrame({"text": texts, "label": labels}).to_csv(csv_path, index=False, encoding="utf-8")
        print(f"ACM 데이터가 '{csv_path}'에 저장되었습니다.")
        # os.remove(zip_path) # 필요 시 압축 파일 삭제
        return csv_path
    except zipfile.BadZipFile:
        print(f"오류: '{zip_path}' 파일이 손상되었습니다. 삭제 후 다시 시도하세요.")
        if os.path.exists(zip_path): os.remove(zip_path)
        raise
    except Exception as e:
        print(f"ACM ZIP 압축 해제 또는 처리 중 오류: {e}")
        raise

def prepare_acm_dataset(data_dir=DATA_DIR):
    """ACM 데이터셋 준비: 다운로드, 처리, 레이블 매핑."""
    print("ACM 데이터셋 준비 중...")
    try:
        csv_path = download_acm_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text'], ['label'])
    except Exception as e:
        print(f"ACM 데이터셋 준비 실패: {e}")
        return [], [], []
    class_mapping = {
        '0': 'artificial intelligence', '1': 'computer networks', '2': 'computer security',
        '3': 'database', '4': 'distributed systems', '5': 'graphics & vision',
        '6': 'human‑computer interaction', '7': 'information retrieval', '8': 'operating systems',
        '9': 'programming languages', '10': 'software engineering'
    }
    mapped_classes = [class_mapping.get(cls, f"Unknown_{cls}") for cls in classes]
    unique_classes = sorted(list(set(mapped_classes)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[c] for c in mapped_classes]
    print(f"ACM 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes

# 2. Reuters-8 (변경 없음)
R8_LABEL2TOPIC = { 0: "acq", 1: "crude", 2: "earn", 3: "grain", 4: "interest", 5: "money-fx", 6: "ship", 7: "trade" }

def download_reuters8_dataset(output_dir=DATA_DIR):
    """Hugging Face에서 Reuters-8을 다운로드하고 CSV로 저장합니다."""
    dataset_name = "reuters8"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "reuters8.csv")
    if os.path.exists(csv_path):
        # print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path
    try:
        print("Reuters-8 데이터셋 다운로드 중 (Hugging Face dxgp/R8)...")
        # cache_dir = os.path.join(dataset_dir, "hf_cache") # 캐시 위치 지정 가능
        ds = load_dataset("dxgp/R8", split="train+test") # train/test 합침
        df = ds.to_pandas()
        df["topic"] = df["label"].map(R8_LABEL2TOPIC).astype(str) # 숫자 레이블을 문자열로
        df[["text", "topic"]].to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Reuters-8 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 Reuters-8 다운로드 실패: {e}")
        raise ConnectionError("Reuters-8 다운로드 실패.")

def prepare_reuters8_dataset(data_dir=DATA_DIR):
    """Reuters-8 데이터셋을 준비합니다."""
    print("Reuters-8 데이터셋 준비 중...")
    try:
        csv_path = download_reuters8_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text'], ['topic'])
    except Exception as e:
        print(f"Reuters-8 데이터셋 준비 실패: {e}")
        return [], [], []
    unique_classes = sorted(list(set(classes)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"Reuters-8 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes

# 3. ChemProt (변경 없음)
def download_chemprot_dataset(output_dir=DATA_DIR):
    """Hugging Face에서 ChemProt를 다운로드하고 CSV로 저장합니다."""
    dataset_name = "chemprot"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "chemprot.csv")
    if os.path.exists(csv_path):
        # print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path
    try:
        print("ChemProt 데이터셋 다운로드 중 (Hugging Face AdaptLLM/ChemProt)...")
        # cache_dir = os.path.join(dataset_dir, "hf_cache")
        ds_dict = load_dataset("AdaptLLM/ChemProt")
        dfs = [ds_dict[split].to_pandas() for split in ds_dict.keys()]
        df = pd.concat(dfs, ignore_index=True)
        if "label" not in df.columns: raise ValueError("'label' column not found.")
        feat = ds_dict[list(ds_dict.keys())[0]].features["label"]
        if isinstance(feat, ClassLabel): df["relation"] = df["label"].apply(feat.int2str)
        else: df["relation"] = df["label"].astype(str)
        if "text" not in df.columns: raise ValueError("'text' column not found.")
        df[["text", "relation"]].to_csv(csv_path, index=False, encoding="utf-8")
        print(f"ChemProt 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 ChemProt 다운로드 실패: {e}")
        raise ConnectionError("ChemProt 다운로드 실패.")

def prepare_chemprot_dataset(data_dir=DATA_DIR):
    """ChemProt 데이터셋을 준비합니다."""
    print("ChemProt 데이터셋 준비 중...")
    try:
        csv_path = download_chemprot_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text'], ['relation'])
    except Exception as e:
        print(f"ChemProt 데이터셋 준비 실패: {e}")
        return [], [], []
    unique_classes = sorted(list(set(classes)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"ChemProt 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes

# 4. BBC News (변경 없음)
def prepare_bbc_news_dataset(data_dir=DATA_DIR):
    """BBC News 데이터셋을 다운로드, 압축 해제 및 준비합니다."""
    print("BBC News 데이터셋 준비 중...")
    dataset_name = "bbc_news"
    dataset_dir = os.path.join(data_dir, dataset_name)
    ensure_dir(dataset_dir)
    url = 'http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip'
    zip_path = os.path.join(dataset_dir, 'bbc-fulltext.zip')
    extracted_base_dir = os.path.join(dataset_dir, 'bbc')

    if not download_file(url, zip_path, "BBC News ZIP"):
         print("BBC News 다운로드 실패.")
         return [], [], []

    if not os.path.exists(extracted_base_dir) or not os.listdir(extracted_base_dir):
        print(f"압축 파일 해제 중... {zip_path} -> {dataset_dir}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            print(f"압축 해제 완료. 내용 확인 중...")
            potential_nested_dir = os.path.join(dataset_dir, 'bbc-fulltext', 'bbc')
            if os.path.exists(potential_nested_dir) and not os.path.exists(extracted_base_dir):
                 print(f"  중첩된 디렉토리 구조 감지됨. 이동 중...")
                 shutil.move(potential_nested_dir, extracted_base_dir)
                 intermediate_parent = os.path.join(dataset_dir, 'bbc-fulltext')
                 if os.path.exists(intermediate_parent) and not os.listdir(intermediate_parent):
                      try: os.rmdir(intermediate_parent)
                      except OSError: pass
            elif not os.path.exists(extracted_base_dir):
                 if os.path.exists(os.path.join(dataset_dir, 'bbc')): pass
                 else:
                      print(f"오류: 압축 해제 후 예상 디렉토리 없음: {extracted_base_dir}")
                      print(f"  '{dataset_dir}' 내용: {os.listdir(dataset_dir)}")
                      return [], [], []
        except zipfile.BadZipFile:
             print(f"오류: '{zip_path}' 파일 손상됨. 삭제 후 재시도.")
             if os.path.exists(zip_path): os.remove(zip_path)
             return [], [], []
        except Exception as e: print(f"압축 해제 중 오류: {e}"); return [], [], []
    # else: print(f"이미 압축 해제된 디렉토리 사용: {extracted_base_dir}")

    categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
    texts, labels = [], []
    print("데이터 로딩 중...")
    for i, category in enumerate(categories):
        category_dir = os.path.join(extracted_base_dir, category)
        if not os.path.isdir(category_dir):
            print(f"경고: 카테고리 디렉토리 없음: {category_dir}")
            continue
        try:
            for filename in os.listdir(category_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(category_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='latin1', errors='ignore') as file:
                            text = re.sub(r'\s+', ' ', file.read()).strip()
                            if text: texts.append(text); labels.append(i)
                    except Exception as e: print(f"\n파일 읽기 오류 ({filename}): {e}")
        except Exception as e: print(f"\n디렉토리 리스팅 오류 ({category_dir}): {e}")

    if not texts: print("경고: BBC News 텍스트 로드 실패!"); return [], [], []
    print(f"BBC News 데이터셋 로딩 완료: {len(texts)} 샘플, {len(categories)} 클래스")
    return texts, labels, categories

# 5. TREC (변경 없음)
def prepare_trec_dataset(data_dir=DATA_DIR):
    """TREC 데이터셋을 다운로드하고 준비합니다 (6개 coarse 레이블 사용)."""
    print("TREC 데이터셋 준비 중...")
    dataset_name = "trec"
    dataset_dir = os.path.join(data_dir, dataset_name)
    ensure_dir(dataset_dir)
    train_url = 'https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label'
    test_url = 'https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label'
    train_path = os.path.join(dataset_dir, 'train_trec.txt')
    test_path = os.path.join(dataset_dir, 'test_trec.txt')

    if not download_file(train_url, train_path, "TREC Train"): return ([], []), ([], []), []
    if not download_file(test_url, test_path, "TREC Test"): return ([], []), ([], []), []

    def load_trec_file(file_path):
        texts, labels_str = [], []
        valid_coarse_labels = {'ABBR', 'DESC', 'ENTY', 'HUM', 'LOC', 'NUM'}
        try:
            encodings_to_try = ['latin1', 'utf-8']
            content = None
            for enc in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=enc) as file: content = file.readlines()
                    # print(f"  TREC 파일 '{os.path.basename(file_path)}' 로드 성공 (인코딩: {enc})")
                    break
                except UnicodeDecodeError: continue
            if content is None:
                print(f"  경고: TREC 파일 '{os.path.basename(file_path)}' 디코딩 실패. 건너뜁니다.")
                return [], []
            for line in content:
                line = line.strip()
                if not line: continue
                parts = line.split(' ', 1)
                if len(parts) != 2: continue
                label_part, question = parts
                coarse_label = label_part.split(':', 1)[0]
                if coarse_label in valid_coarse_labels:
                    texts.append(question)
                    labels_str.append(coarse_label)
        except Exception as e: print(f"TREC 파일 로딩 오류 ({file_path}): {e}")
        return texts, labels_str

    train_texts, train_labels_str = load_trec_file(train_path)
    test_texts, test_labels_str = load_trec_file(test_path)
    if not train_texts or not test_texts: print("오류: TREC 로딩 실패."); return [], [], []

    all_labels_str = train_labels_str + test_labels_str
    unique_labels = sorted(list(set(all_labels_str)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    train_labels = [label_to_id[label] for label in train_labels_str]
    test_labels = [label_to_id[label] for label in test_labels_str]
    all_texts = train_texts + test_texts
    all_labels = train_labels + test_labels
    print(f"TREC 로딩 완료: 총 {len(all_texts)} 샘플, 클래스={unique_labels}")
    return all_texts, all_labels, unique_labels

# 6. Banking77 (변경 없음)
def download_banking77_dataset(output_dir=DATA_DIR):
    """Banking77 데이터셋을 다운로드하고 CSV로 저장합니다 (HF 우선, GitHub 대체)."""
    dataset_name = "banking77"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "banking77.csv")
    if os.path.exists(csv_path):
        # print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path

    hf_url = "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/banking77.csv"
    github_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"

    if download_file(hf_url, csv_path, "Banking77 (HF)"):
        try:
            df = pd.read_csv(csv_path)
            text_col = next((c for c in ['text', 'query'] if c in df.columns), None)
            label_col = next((c for c in ['category', 'label'] if c in df.columns), None)
            if text_col and label_col:
                 df_processed = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'category'})
                 df_processed.to_csv(csv_path, index=False, encoding='utf-8')
                 print(f"Banking77 (HF) 데이터가 '{csv_path}'에 저장되었습니다.")
                 return csv_path
            else:
                 print("HF Banking77 CSV 컬럼 불일치, GitHub 대체 시도 중...")
                 os.remove(csv_path)
        except Exception as e:
             print(f"다운로드된 Banking77 (HF) 처리 오류: {e}. 대체 시도 중...")
             if os.path.exists(csv_path): os.remove(csv_path)

    print("GitHub 대체 URL 시도 중...")
    if download_file(github_url, csv_path, "Banking77 (GitHub Fallback)"):
         try:
            df = pd.read_csv(csv_path)
            text_col = next((c for c in ['text', 'query'] if c in df.columns), None)
            label_col = next((c for c in ['category', 'label'] if c in df.columns), None)
            if text_col and label_col:
                 df_processed = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'category'})
                 df_processed.to_csv(csv_path, index=False, encoding='utf-8')
                 print(f"Banking77 (GitHub) 데이터가 '{csv_path}'에 저장되었습니다.")
                 return csv_path
            else: raise ValueError("GitHub 대체 파일에서 텍스트 또는 레이블 열을 찾을 수 없습니다.")
         except Exception as e:
              print(f"다운로드된 Banking77 (GitHub) 처리 오류: {e}")
              if os.path.exists(csv_path): os.remove(csv_path)
    raise ConnectionError("Banking77 다운로드 실패.")

def prepare_banking77_dataset(data_dir=DATA_DIR):
    """Banking77 데이터셋을 준비합니다."""
    print("Banking77 데이터셋 준비 중...")
    try:
        csv_path = download_banking77_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text', 'query'], ['category', 'label'])
    except Exception as e:
        print(f"Banking77 데이터셋 준비 실패: {e}")
        return [], [], []
    unique_classes = sorted(list(set(classes)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"Banking77 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes

# 7. OOS (CLINC150) (변경 없음)
def download_oos_dataset(output_dir=DATA_DIR, config="plus"):
    """Hugging Face에서 OOS (CLINC150)를 다운로드하고 CSV로 저장합니다 ('plus' 설정 사용)."""
    dataset_name = "oos_clinc150"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, f"oos_{config}.csv")
    if os.path.exists(csv_path):
        # print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path
    try:
        print(f"OOS (CLINC150) 데이터셋 다운로드 중 (Hugging Face clinc_oos/{config})...")
        # cache_dir = os.path.join(dataset_dir, "hf_cache")
        try:
             ds = load_dataset("clinc_oos", config)
             splits = list(ds.keys())
             # print(f"  사용 가능한 분할: {splits}")
             df = pd.concat([ds[split].to_pandas() for split in splits], ignore_index=True)
        except ValueError:
             print(f"구성 '{config}'을(를) 찾을 수 없습니다. 대체 구성 시도 중...")
             # Add more fallbacks if needed (e.g., 'imbalanced')
             ds = load_dataset("clinc_oos") # Load default config
             splits = list(ds.keys())
             df = pd.concat([ds[split].to_pandas() for split in splits], ignore_index=True)

        label_col_found = None
        if "intent" in df.columns: label_col_found = "intent"
        elif "label" in df.columns:
             feat = ds[splits[0]].features["label"]
             if isinstance(feat, ClassLabel): df["intent"] = df["label"].apply(feat.int2str)
             else: df["intent"] = df["label"].astype(str)
             label_col_found = "intent"
        if not label_col_found: raise ValueError("레이블/의도 열을 찾을 수 없습니다.")

        text_col_found = None
        if "text" in df.columns: text_col_found = "text"
        else:
             text_col_cand = next((c for c in ["sentence", "utterance", "query"] if c in df.columns), None)
             if text_col_cand: df.rename(columns={text_col_cand: "text"}, inplace=True); text_col_found = "text"
        if not text_col_found: raise ValueError("텍스트 열을 찾을 수 없습니다.")

        df[["text", "intent"]].to_csv(csv_path, index=False, encoding="utf-8")
        print(f"OOS (CLINC150) 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 OOS (CLINC150) 다운로드 실패: {e}")
        raise ConnectionError("OOS (CLINC150) 다운로드 실패.")

def prepare_oos_dataset(data_dir=DATA_DIR):
    """OOS (CLINC150) 데이터셋을 준비합니다."""
    print("OOS (CLINC150) 데이터셋 준비 중...")
    try:
        csv_path = download_oos_dataset(data_dir, config="plus")
        texts, classes = load_csv_universal(csv_path, ['text', 'sentence', 'utterance', 'query'], ['intent', 'label'])
    except Exception as e:
        print(f"OOS (CLINC150) 데이터셋 준비 실패: {e}")
        return [], [], []
    # OOS 레이블('oos')을 포함하여 모든 클래스를 로드합니다.
    # 실제 OSR 실험에서는 DataModule 등에서 'oos' 레이블을 -1 등으로 매핑해야 합니다.
    unique_classes = sorted(list(set(classes)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"OOS (CLINC150) 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스 (포함 가능성: 'oos')")
    return texts, labels, unique_classes


# 8. StackOverflow (클래스 수 기본값 20으로 수정)
def download_stackoverflow_dataset(output_dir=DATA_DIR):
    """Hugging Face에서 StackOverflow 데이터셋을 다운로드하고 CSV로 저장합니다."""
    dataset_name = "stackoverflow"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "stackoverflow.csv")
    if os.path.exists(csv_path):
        # print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path
    try:
        print("StackOverflow 데이터셋 다운로드 중 (Hugging Face)...")
        # cache_dir = os.path.join(dataset_dir, "hf_cache")
        df = None
        try: # 더 표준적인 데이터셋 먼저 시도
             print("  c17hawke/stackoverflow-dataset 시도 중...")
             ds = load_dataset("c17hawke/stackoverflow-dataset", split="train")
             print("  c17hawke/stackoverflow-dataset 로드됨.")
             df = ds.to_pandas()
             df['text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')
             df['text'] = df['text'].str.strip()
             if 'tags' not in df.columns: raise ValueError("c17hawke에서 태그 열 누락.")
        except Exception as e1:
             print(f"  c17hawke/stackoverflow 로딩 실패 ({e1}), 대체 소스 시도 불가.")
             raise ConnectionError("StackOverflow 다운로드 실패.")

        if df is None or 'text' not in df.columns or 'tags' not in df.columns:
             raise ValueError("StackOverflow 데이터 처리 실패: 텍스트 또는 태그 열 없음.")

        # 태그를 문자열로 변환 (리스트/문자열 처리)
        if not df.empty and isinstance(df['tags'].iloc[0], list):
            df['tags'] = df['tags'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else str(x))
        else:
            df['tags'] = df['tags'].astype(str)

        df = df[['text', 'tags']].dropna(subset=['text']) # 관련 열만 남기고 텍스트 없는 행 제거
        df = df[df['text'].str.len() > 10] # 매우 짧은 텍스트 제거

        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"StackOverflow 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 StackOverflow 다운로드 실패: {e}")
        raise ConnectionError("StackOverflow 다운로드 실패.")

def prepare_stackoverflow_dataset(data_dir=DATA_DIR, max_classes=20): # 기본 클래스 수 20으로 변경
    """StackOverflow 데이터셋 준비 (클래스 수 제한 가능)."""
    print(f"StackOverflow 데이터셋 준비 중 (최대 {max_classes} 클래스)...")
    try:
        csv_path = download_stackoverflow_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text', 'title'], ['tags'])
    except Exception as e:
        print(f"StackOverflow 데이터셋 준비 실패: {e}")
        return [], [], []

    # 태그 조합이 너무 많으면 빈도 기반 상위 N개만 사용
    unique_classes_full = sorted(list(set(classes)))
    if len(unique_classes_full) > max_classes:
         print(f"  경고: StackOverflow 태그 조합이 너무 많음 ({len(unique_classes_full)}). 상위 {max_classes}개 태그만 사용.")
         tag_counts = pd.Series(classes).value_counts()
         top_tags = tag_counts.nlargest(max_classes).index.tolist()
         mask = pd.Series(classes).isin(top_tags)
         texts = [t for i, t in enumerate(texts) if mask.iloc[i]]
         classes = [c for i, c in enumerate(classes) if mask.iloc[i]]
         unique_classes = sorted(top_tags)
         print(f"  상위 태그 필터링 후 샘플 수: {len(texts)}")
    else:
         unique_classes = unique_classes_full

    if not texts: print("경고: 필터링 후 StackOverflow 데이터 없음."); return [], [], []
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"StackOverflow 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes

# 9. ATIS (변경 없음)
def download_atis_dataset(output_dir=DATA_DIR):
    """Hugging Face에서 ATIS 데이터셋을 다운로드하고 CSV로 저장합니다."""
    dataset_name = "atis"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "atis.csv")
    if os.path.exists(csv_path):
        # print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path
    try:
        print("ATIS 데이터셋 다운로드 중 (Hugging Face tuetschek/atis)...")
        # cache_dir = os.path.join(dataset_dir, "hf_cache")
        ds = load_dataset("tuetschek/atis")
        df = pd.concat([ds[split].to_pandas() for split in ds.keys()], ignore_index=True)
        text_col = next((c for c in ['text', 'sentence', 'query'] if c in df.columns), None)
        label_col_int = 'label' if 'label' in df.columns else None
        label_col_str = 'intent' if 'intent' in df.columns else None
        if not text_col: raise ValueError("텍스트 열 없음.")
        if not label_col_int and not label_col_str: raise ValueError("레이블/의도 열 없음.")
        if label_col_str: df['intent_norm'] = df[label_col_str]
        elif label_col_int:
            feat = ds[list(ds.keys())[0]].features[label_col_int]
            if isinstance(feat, ClassLabel): df["intent_norm"] = df[label_col_int].apply(feat.int2str)
            else: df["intent_norm"] = df[label_col_int].astype(str)
        df_processed = df[[text_col, 'intent_norm']].rename(columns={text_col: 'text', 'intent_norm': 'intent'})
        df_processed.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"ATIS 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 ATIS 다운로드 실패: {e}")
        raise ConnectionError("ATIS 다운로드 실패.")

def prepare_atis_dataset(data_dir=DATA_DIR):
    """ATIS 데이터셋을 준비합니다."""
    print("ATIS 데이터셋 준비 중...")
    try:
        csv_path = download_atis_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text', 'sentence'], ['intent', 'label'])
    except Exception as e:
        print(f"ATIS 데이터셋 준비 실패: {e}")
        return [], [], []
    unique_classes = sorted(list(set(classes)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"ATIS 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes

# 10. SNIPS (변경 없음)
def download_snips_dataset(output_dir=DATA_DIR):
    """Hugging Face에서 SNIPS 데이터셋을 다운로드하고 CSV로 저장합니다."""
    dataset_name = "snips"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "snips.csv")
    if os.path.exists(csv_path):
        # print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path
    try:
        print("SNIPS 데이터셋 다운로드 중 (Hugging Face snips_built_in_intents)...")
        # cache_dir = os.path.join(dataset_dir, "hf_cache")
        ds = load_dataset("snips_built_in_intents")
        df = pd.concat([ds[split].to_pandas() for split in ds.keys()], ignore_index=True)
        text_col = next((c for c in ['text', 'sentence', 'query'] if c in df.columns), None)
        label_col_int = 'label' if 'label' in df.columns else None
        label_col_str = 'intent' if 'intent' in df.columns else None
        if not text_col: raise ValueError("텍스트 열 없음.")
        if not label_col_int and not label_col_str: raise ValueError("레이블/의도 열 없음.")
        if label_col_str: df['intent_norm'] = df[label_col_str]
        elif label_col_int:
            feat = ds[list(ds.keys())[0]].features[label_col_int]
            if isinstance(feat, ClassLabel): df["intent_norm"] = df[label_col_int].apply(feat.int2str)
            else: df["intent_norm"] = df[label_col_int].astype(str)
        df_processed = df[[text_col, 'intent_norm']].rename(columns={text_col: 'text', 'intent_norm': 'intent'})
        df_processed.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"SNIPS 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 SNIPS 다운로드 실패: {e}")
        raise ConnectionError("SNIPS 다운로드 실패.")

def prepare_snips_dataset(data_dir=DATA_DIR):
    """SNIPS 데이터셋을 준비합니다."""
    print("SNIPS 데이터셋 준비 중...")
    try:
        csv_path = download_snips_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text'], ['intent', 'label'])
    except Exception as e:
        print(f"SNIPS 데이터셋 준비 실패: {e}")
        return [], [], []
    unique_classes = sorted(list(set(classes)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"SNIPS 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes

# 11. Financial PhraseBank (변경 없음)
def download_financial_phrasebank_dataset(output_dir=DATA_DIR):
    """Hugging Face에서 Financial PhraseBank를 다운로드하고 CSV로 저장합니다."""
    dataset_name = "financial_phrasebank"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "financial_phrasebank.csv")
    if os.path.exists(csv_path):
        # print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path
    try:
        print("Financial PhraseBank 데이터셋 다운로드 중 (Hugging Face takala/financial_phrasebank)...")
        # cache_dir = os.path.join(dataset_dir, "hf_cache")
        ds = load_dataset("takala/financial_phrasebank", "sentences_allagree", split="train")
        df = ds.to_pandas()
        text_col = next((c for c in ['text', 'sentence'] if c in df.columns), None)
        label_col_int = 'label' if 'label' in df.columns else None
        label_col_str = 'sentiment' if 'sentiment' in df.columns else None
        if not text_col: raise ValueError("텍스트/문장 열 없음.")
        if not label_col_int and not label_col_str: raise ValueError("감성/레이블 열 없음.")
        if label_col_str: df['sentiment_norm'] = df[label_col_str]
        elif label_col_int:
            label_map = {0: 'negative', 1: 'neutral', 2: 'positive'} # 숫자 -> 문자열 매핑
            df['sentiment_norm'] = df[label_col_int].map(label_map).fillna('unknown')
        df_processed = df[[text_col, 'sentiment_norm']].rename(columns={text_col: 'text', 'sentiment_norm': 'sentiment'})
        df_processed.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Financial PhraseBank 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 Financial PhraseBank 다운로드 실패: {e}")
        raise ConnectionError("Financial PhraseBank 다운로드 실패.")

def prepare_financial_phrasebank_dataset(data_dir=DATA_DIR):
    """Financial PhraseBank 데이터셋을 준비합니다."""
    print("Financial PhraseBank 데이터셋 준비 중...")
    try:
        csv_path = download_financial_phrasebank_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text', 'sentence'], ['sentiment', 'label'])
    except Exception as e:
        print(f"Financial PhraseBank 데이터셋 준비 실패: {e}")
        return [], [], []
    unique_classes = sorted(list(set(classes))) # 'negative', 'neutral', 'positive'
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"Financial PhraseBank 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes

# 12. ArXiv-10 (변경 없음)
def download_arxiv10_dataset(output_dir=DATA_DIR):
    """ArXiv-10 데이터셋을 다운로드, 압축 해제, 정규화하고 CSV로 저장합니다."""
    dataset_name = "arxiv10"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    zip_url = "https://github.com/ashfarhangi/Protoformer/raw/main/data/ArXiv-10.zip"
    zip_path = os.path.join(dataset_dir, "arxiv10.zip")
    csv_path = os.path.join(dataset_dir, "arxiv10.csv") # 최종 목표 CSV

    if os.path.exists(csv_path):
        # print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path

    if not os.path.exists(zip_path):
        if not download_file(zip_url, zip_path, "ArXiv-10 ZIP"):
            raise ConnectionError("ArXiv-10 ZIP 다운로드 실패.")

    print(f"'{os.path.basename(zip_path)}' 압축 해제 및 처리 중...")
    temp_extract_path = os.path.join(dataset_dir, "temp_extract")
    ensure_dir(temp_extract_path)
    try:
        with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(temp_extract_path)
        # print(f"  '{temp_extract_path}'에 압축 해제 완료.")
        found_csv_path = None
        target_csv_name = 'arxiv-10.csv'
        for root, _, files in os.walk(temp_extract_path):
            for file in files:
                if file.lower() == target_csv_name: found_csv_path = os.path.join(root, file); break
            if found_csv_path: break
        if not found_csv_path:
            print(f"  '{target_csv_name}'을(를) 찾지 못했습니다. 폴더 내 다른 CSV 검색 중...")
            for root, _, files in os.walk(temp_extract_path):
                 for file in files:
                      if file.lower().endswith('.csv'): found_csv_path = os.path.join(root, file); break
                 if found_csv_path: break
        if not found_csv_path: raise FileNotFoundError("압축 해제된 폴더에서 ArXiv-10 CSV 파일을 찾을 수 없습니다.")
        # print(f"  CSV 파일 처리 중: {found_csv_path}")
        df = pd.read_csv(found_csv_path)
        text_col = next((c for c in ['abstract', 'text', 'title'] if c in df.columns), None)
        label_col = next((c for c in ['category', 'label', 'class'] if c in df.columns), None)
        if not text_col: raise ValueError("ArXiv-10 CSV에서 텍스트 열(abstract/text/title)을 찾을 수 없습니다.")
        if not label_col: raise ValueError("ArXiv-10 CSV에서 카테고리 열(category/label/class)을 찾을 수 없습니다.")
        # print(f"  정규화된 열 사용: Text='{text_col}', Label='{label_col}'")
        df_processed = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'category'})
        df_processed.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"ArXiv-10 데이터가 '{csv_path}'에 최종 저장되었습니다.")
        try: shutil.rmtree(temp_extract_path)
        except Exception as e_rm: print(f"경고: 임시 추출 폴더 삭제 실패 {temp_extract_path}: {e_rm}")
        # os.remove(zip_path)
        return csv_path
    except zipfile.BadZipFile:
        print(f"오류: '{zip_path}' 파일이 손상되었습니다. 삭제 후 다시 시도하세요.")
        if os.path.exists(zip_path): os.remove(zip_path)
        if os.path.exists(temp_extract_path): shutil.rmtree(temp_extract_path)
        raise
    except Exception as e:
        print(f"ArXiv-10 처리 오류: {e}")
        if os.path.exists(temp_extract_path): shutil.rmtree(temp_extract_path)
        raise

def prepare_arxiv10_dataset(data_dir=DATA_DIR):
    """ArXiv-10 데이터셋을 준비합니다."""
    print("ArXiv-10 데이터셋 준비 중...")
    try:
        csv_path = download_arxiv10_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text', 'abstract', 'title'], ['category', 'label', 'class'])
    except Exception as e:
        print(f"ArXiv-10 데이터셋 준비 실패: {e}")
        return [], [], []
    unique_classes = sorted(list(set(classes)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"ArXiv-10 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes

# 13. NewsGroup20 (변경 없음)
def prepare_newsgroup20_dataset(data_dir=DATA_DIR):
    """sklearn의 fetcher를 사용하여 20 Newsgroups 데이터셋을 준비합니다."""
    print("20 Newsgroups 데이터셋 준비 중 (sklearn 사용)...")
    try:
        # data_home 인자를 사용하여 캐시 위치 제어 가능
        # data_home_path = os.path.join(data_dir, "newsgroup20", "sklearn_cache")
        # ensure_dir(data_home_path)
        data_all = fetch_20newsgroups(
            subset='all', remove=('headers', 'footers', 'quotes'),
            random_state=42 # 필요 시 재현성을 위한 상태 고정
            # data_home=data_home_path
        )
        texts = data_all.data
        labels = data_all.target # 0-19 정수 레이블
        class_names = data_all.target_names # 실제 클래스 이름
        print(f"20 Newsgroups 로딩 완료: {len(texts)} 샘플, {len(class_names)} 클래스")
        return texts, labels, class_names
    except Exception as e:
        print(f"20 Newsgroups 로딩 실패: {e}")
        return [], [], []

# 14. Custom Syslog (변경 없음)
def download_custom_syslog_dataset(output_dir=DATA_DIR):
    """사용자 정의 syslog 데이터셋 파일의 존재 여부만 확인합니다."""
    dataset_name = "custom_syslog"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "custom_syslog.csv") # 예상 파일명
    if os.path.exists(csv_path):
        print(f"Custom Syslog 데이터셋 파일 '{csv_path}'이(가) 존재합니다.")
        return csv_path
    else:
        error_message = (
            f"오류: Custom Syslog 데이터셋 파일을 찾을 수 없습니다.\n"
            f"'{csv_path}' 경로에 'custom_syslog.csv' 파일을 위치시켜 주세요.\n"
            f"이 파일은 텍스트 열('text', 'message' 등)과 클래스 레이블 열('class', 'label' 등)을 가져야 합니다."
        )
        raise FileNotFoundError(error_message)

def prepare_custom_syslog_dataset(data_dir=DATA_DIR):
    """사용자 정의 syslog 데이터셋을 준비합니다."""
    print("Custom Syslog 데이터셋 준비 중...")
    try:
        csv_path = download_custom_syslog_dataset(data_dir)
        texts, classes_str = load_csv_universal(
            csv_path,
            text_col_candidates=['text', 'message', 'log', 'content'],
            label_col_candidates=['class', 'label', 'category', 'event', 'template']
        )
    except Exception as e:
        print(f"Custom Syslog 데이터셋 준비 실패: {e}")
        return [], [], []
    if not texts: print("경고: Custom Syslog 텍스트 데이터가 비어 있습니다."); return [], [], []
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(classes_str)
    class_names = label_encoder.classes_.tolist()
    print(f"Custom Syslog 데이터셋 준비 완료: {len(texts)} 샘플, {len(class_names)} 클래스")
    # print(f"  클래스 예시: {class_names[:10]}{'...' if len(class_names) > 10 else ''}")
    return texts, labels_encoded, class_names

# --- 신규 추가된 데이터셋 함수 ---

# 15. 50-class reviews (Placeholder)
def prepare_50_class_reviews_dataset(data_dir=DATA_DIR):
    """
    50-class reviews 데이터셋 준비 (Placeholder).
    이 데이터셋은 특정 논문(예: DOC)에서 사용된 것으로, 표준 허브에서 직접
    다운로드하기 어렵습니다. 해당 논문의 리소스나 별도의 전처리 스크립트를 통해
    데이터를 'data/50_class_reviews/reviews.csv' 와 같은 형태로 준비해야 합니다.
    CSV 파일은 텍스트 열과 클래스(카테고리) 레이블 열을 포함해야 합니다.
    """
    print("50-class reviews 데이터셋 준비 시도 중 (사용자 제공 필요)...")
    dataset_name = "50_class_reviews"
    dataset_dir = os.path.join(data_dir, dataset_name)
    ensure_dir(dataset_dir)
    # 사용자가 준비해야 할 예상 CSV 파일 경로
    csv_path = os.path.join(dataset_dir, "reviews.csv") # 예시 파일명

    if not os.path.exists(csv_path):
        error_message = (
            f"오류: 50-class reviews 데이터셋 파일을 찾을 수 없습니다.\n"
            f"이 데이터셋은 표준 다운로드를 지원하지 않습니다.\n"
            f"관련 논문(DOC 등)의 지침에 따라 데이터를 '{csv_path}' 경로에 준비해주세요.\n"
            f"CSV 파일은 텍스트 열과 50개의 클래스 레이블 열을 포함해야 합니다."
        )
        print(error_message)
        # FileNotFoundError(error_message) # 오류 발생 대신 빈 리스트 반환
        return [], [], []

    print(f"'{csv_path}' 파일 로딩 시도...")
    try:
        # 실제 파일이 있다면 로드 시도 (컬럼 이름은 유연하게)
        texts, classes_str = load_csv_universal(
            csv_path,
            text_col_candidates=['text', 'review', 'content'],
            label_col_candidates=['class', 'label', 'category', 'topic']
        )
    except Exception as e:
        print(f"50-class reviews 데이터셋 로딩 실패 ({csv_path}): {e}")
        return [], [], []

    if not texts: print("경고: 50-class reviews 텍스트 데이터 로드 실패."); return [], [], []

    # 레이블 인코딩
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(classes_str)
    class_names = label_encoder.classes_.tolist()

    if len(class_names) != 50:
        print(f"경고: 로드된 50-class reviews 데이터셋의 클래스 수가 50개가 아닙니다 ({len(class_names)}개).")

    print(f"50-class reviews 데이터셋 준비 완료: {len(texts)} 샘플, {len(class_names)} 클래스")
    return texts, labels_encoded, class_names

# 16. SST-5 (Stanford Sentiment Treebank - 5 classes)
def download_sst5_dataset(output_dir=DATA_DIR):
    """Hugging Face에서 SST-5 (SetFit/sst5) 데이터셋을 다운로드하고 CSV로 저장합니다."""
    dataset_name = "sst5"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "sst5.csv")
    if os.path.exists(csv_path):
        # print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path
    try:
        print("SST-5 데이터셋 다운로드 중 (Hugging Face SetFit/sst5)...")
        # cache_dir = os.path.join(dataset_dir, "hf_cache")
        ds = load_dataset("SetFit/sst5")
        df = pd.concat([ds[split].to_pandas() for split in ds.keys()], ignore_index=True)

        # 컬럼 표준화
        text_col = next((c for c in ['text', 'sentence'] if c in df.columns), None)
        label_col_int = 'label' if 'label' in df.columns else None
        label_col_str = 'label_text' if 'label_text' in df.columns else None # SetFit/sst5 에는 label_text 존재

        if not text_col: raise ValueError("텍스트 열 없음.")
        if not label_col_int and not label_col_str: raise ValueError("레이블 열 없음.")

        if label_col_str:
            df['sentiment'] = df[label_col_str]
        elif label_col_int: # label_text가 없을 경우 대비
             # SST-5의 정수 레이블은 0(매우 부정) ~ 4(매우 긍정)
             label_map = {0: 'very negative', 1: 'negative', 2: 'neutral', 3: 'positive', 4: 'very positive'}
             df['sentiment'] = df[label_col_int].map(label_map).fillna('unknown')

        df_processed = df[[text_col, 'sentiment']].rename(columns={text_col: 'text', 'sentiment': 'sentiment'})
        df_processed.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"SST-5 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 SST-5 다운로드 실패: {e}")
        raise ConnectionError("SST-5 다운로드 실패.")

def prepare_sst5_dataset(data_dir=DATA_DIR):
    """SST-5 데이터셋을 준비합니다."""
    print("SST-5 데이터셋 준비 중...")
    try:
        csv_path = download_sst5_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text', 'sentence'], ['sentiment', 'label', 'label_text'])
    except Exception as e:
        print(f"SST-5 데이터셋 준비 실패: {e}")
        return [], [], []
    unique_classes = sorted(list(set(classes)))
    if len(unique_classes) != 5:
        print(f"경고: SST-5 데이터셋의 클래스 수가 5개가 아닙니다 ({len(unique_classes)}개). 레이블: {unique_classes}")
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"SST-5 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes

# 17. DBpedia-14
def download_dbpedia14_dataset(output_dir=DATA_DIR):
    """Hugging Face에서 DBpedia-14 데이터셋을 다운로드하고 CSV로 저장합니다."""
    dataset_name = "dbpedia14"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "dbpedia14.csv")
    if os.path.exists(csv_path):
        # print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path
    try:
        print("DBpedia-14 데이터셋 다운로드 중 (Hugging Face dbpedia_14)...")
        # cache_dir = os.path.join(dataset_dir, "hf_cache")
        ds = load_dataset("dbpedia_14")
        df = pd.concat([ds[split].to_pandas() for split in ds.keys()], ignore_index=True)

        # 컬럼 표준화
        # content 컬럼이 주로 사용됨 (title은 매우 짧음)
        text_col = 'content' if 'content' in df.columns else None
        label_col_int = 'label' if 'label' in df.columns else None

        if not text_col: raise ValueError("텍스트(content) 열 없음.")
        if not label_col_int: raise ValueError("레이블 열 없음.")

        # 숫자 레이블을 문자열로 변환
        feat = ds[list(ds.keys())[0]].features[label_col_int]
        if isinstance(feat, ClassLabel): df["category"] = df[label_col_int].apply(feat.int2str)
        else: df["category"] = df[label_col_int].astype(str)

        df_processed = df[[text_col, 'category']].rename(columns={text_col: 'text', 'category': 'category'})
        # 텍스트 클리닝 (DBpedia는 이스케이프 문자가 많음)
        # df_processed['text'] = df_processed['text'].str.replace('\\\\', '', regex=False).str.replace('\\"', '"', regex=False).str.strip()

        df_processed.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"DBpedia-14 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 DBpedia-14 다운로드 실패: {e}")
        raise ConnectionError("DBpedia-14 다운로드 실패.")

def prepare_dbpedia14_dataset(data_dir=DATA_DIR):
    """DBpedia-14 데이터셋을 준비합니다."""
    print("DBpedia-14 데이터셋 준비 중...")
    try:
        csv_path = download_dbpedia14_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text', 'content'], ['category', 'label'])
    except Exception as e:
        print(f"DBpedia-14 데이터셋 준비 실패: {e}")
        return [], [], []
    unique_classes = sorted(list(set(classes)))
    if len(unique_classes) != 14:
        print(f"경고: DBpedia-14 데이터셋의 클래스 수가 14개가 아닙니다 ({len(unique_classes)}개).")
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"DBpedia-14 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes


# --- 데이터셋 준비 함수 매핑 (선택 사항) ---
# 다른 스크립트에서 쉽게 호출하기 위한 딕셔너리
PREPARE_FUNCTIONS = {
    "acm": prepare_acm_dataset,
    "reuters8": prepare_reuters8_dataset,
    "chemprot": prepare_chemprot_dataset,
    "bbc_news": prepare_bbc_news_dataset,
    "trec": prepare_trec_dataset,
    "banking77": prepare_banking77_dataset,
    "oos": prepare_oos_dataset, # OOS (CLINC150)
    "stackoverflow": prepare_stackoverflow_dataset,
    "atis": prepare_atis_dataset,
    "snips": prepare_snips_dataset,
    "financial_phrasebank": prepare_financial_phrasebank_dataset,
    "arxiv10": prepare_arxiv10_dataset,
    "newsgroup20": prepare_newsgroup20_dataset,
    "custom_syslog": prepare_custom_syslog_dataset,
    "50_class_reviews": prepare_50_class_reviews_dataset, # Placeholder
    "sst5": prepare_sst5_dataset, # SST-5 추가
    "dbpedia14": prepare_dbpedia14_dataset, # DBpedia-14 추가
}

def get_dataset(dataset_name, data_dir=DATA_DIR, **kwargs):
    """주어진 이름의 데이터셋을 준비하고 결과를 반환합니다."""
    if dataset_name not in PREPARE_FUNCTIONS:
        raise ValueError(f"알 수 없는 데이터셋 이름: {dataset_name}. 사용 가능: {list(PREPARE_FUNCTIONS.keys())}")
    prepare_func = PREPARE_FUNCTIONS[dataset_name]
    # 특정 데이터셋에 추가 인자 전달 (예: StackOverflow의 max_classes)
    return prepare_func(data_dir=data_dir, **kwargs)

# --- 스크립트 직접 실행 시 예시 (테스트용) ---
if __name__ == "__main__":
    print("데이터셋 준비 유틸리티 테스트...")
    # 테스트하려는 데이터셋 이름 목록
    # datasets_to_test = ["banking77", "oos", "stackoverflow", "newsgroup20", "sst5", "dbpedia14", "trec"]
    datasets_to_test = ["sst5", "dbpedia14"] # 예시: 새로 추가된 것만 테스트
    results = {}
    for name in datasets_to_test:
        print(f"\n--- {name.upper()} 테스트 시작 ---")
        try:
            # StackOverflow의 경우 max_classes 전달 예시
            if name == "stackoverflow":
                texts, labels, classes = get_dataset(name, max_classes=20)
            else:
                texts, labels, classes = get_dataset(name)

            if texts: # 데이터가 성공적으로 로드되었는지 확인
                results[name] = {
                    "num_samples": len(texts),
                    "num_classes": len(classes),
                    "class_names_example": classes[:5] # 클래스 이름 예시
                }
                print(f"--- {name.upper()} 테스트 성공 ---")
            else:
                print(f"--- {name.upper()} 테스트 실패 (데이터 없음) ---")
                results[name] = "Failed (No data)"
        except Exception as e:
            print(f"--- {name.upper()} 테스트 중 오류 발생 ---")
            print(e)
            results[name] = f"Failed ({type(e).__name__})"

    print("\n--- 테스트 결과 요약 ---")
    for name, result in results.items():
        print(f"{name}: {result}")