import os
import cv2
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================================================
# [설정] 기본 경로 설정
# =========================================================
DEFAULT_CSV_PATH = "../data/dev/dev.csv"
DEFAULT_SAVE_DIR = "../data/dev/images"

def _create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

def _load_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return None
    return pd.read_csv(csv_path)

def _download_image(url):
    """URL -> OpenCV Image 변환"""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return None

def _save_image(image, save_dir, file_name):
    """이미지 저장"""
    if image is None: return False
    try:
        save_path = os.path.join(save_dir, file_name)
        cv2.imwrite(save_path, image)
        return True
    except Exception:
        return False


def run_downloader(csv_path=DEFAULT_CSV_PATH, save_dir=DEFAULT_SAVE_DIR):
    """
    CSV 경로와 저장 폴더만 주면, 폴더 생성 -> 파일 로드 -> 다운로드 -> 저장
    """
    print(f"Starting Downloader...")

    _create_directory(save_dir)
    df = _load_csv(csv_path)

    if df is None or df.empty:
        print("No data to process.")
        return

    print(f"Processing {len(df)} images from '{csv_path}'")
    success = 0
    fail = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        img = _download_image(row['img_url'])
        is_saved = _save_image(img, save_dir, f"{row['id']}.jpg")

        if is_saved:
            success += 1
        else:
            fail += 1

    print("\n" + "=" * 30)
    print(f"Done!")
    print(f"   - Saved to : {save_dir}")
    print(f"   - Success  : {success}")
    print(f"   - Failed   : {fail}")
    print("=" * 30)

# =========================================================
# [추가] 하단 전처리 부분
# =========================================================

if __name__ == "__main__":
    # image 파일 저장 시 사용
    # run_downloader()
