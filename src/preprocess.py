import os
import cv2
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob

class ImageDownloader:
    """
    역할: CSV 파일에서 URL을 읽어 이미지를 다운로드하고 data 폴더 내에 저장
    """
    def __init__(self, csv_path, save_dir):
        self.csv_path = csv_path
        self.save_dir = save_dir
        # self.create_directory(self.save_dir)

    def create_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"[Info] Created directory: {path}")

    def load_csv(self):
        if not os.path.exists(self.csv_path):
            print(f"[Error] CSV file not found at {self.csv_path}")
            return None
        return pd.read_csv(self.csv_path)

    def download_image(self, url):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
            return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def save_image(self, image, file_name):
        if image is None: return False
        try:
            save_path = os.path.join(self.save_dir, file_name)
            cv2.imwrite(save_path, image)
            return True
        except Exception:
            return False

    def run(self):
        print(f"Starting Downloader...")
        df = self._load_csv()

        if df is None or df.empty:
            print("No data to process.")
            return

        print(f"Processing {len(df)} images from '{self.csv_path}'")
        success = 0
        fail = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
            img = self._download_image(row['img_url'])
            # ID를 파일명으로 사용
            is_saved = self._save_image(img, f"{row['id']}.jpg")

            if is_saved:
                success += 1
            else:
                fail += 1

        print("\n" + "=" * 30)
        print(f"Download Finished!")
        print(f"   - Saved to : {self.save_dir}")
        print(f"   - Success  : {success}")
        print(f"   - Failed   : {fail}")
        print("=" * 30)

class ImagePreprocessor:
    """
    역할: 이미지 전처리 수행
        - CLAHE
        - *Canny Edge
    """

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), canny_low=50, canny_high=150):
        # CLAHE 설정
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        # Canny 설정
        self.canny_low = canny_low
        self.canny_high = canny_high

    def apply_clahe(self, image):
        """
        컬러 이미지에 CLAHE 적용 (LAB 색상 공간 사용)
        """
        if image is None: return None

        try:
            # BGR -> LAB 변환
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # L 채널(밝기)에 CLAHE 적용
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            l_clahe = clahe.apply(l)

            # 병합 및 BGR 복귀
            lab_merged = cv2.merge((l_clahe, a, b))
            return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
        except Exception as e:
            print(f"CLAHE Error: {e}")
            return image

    def apply_canny(self, image):
        """
        이미지에 Canny Edge Detection 적용
        """
        if image is None: return None

        # 컬러면 흑백 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        return cv2.Canny(blurred, self.canny_low, self.canny_high)

    def apply_canny_overlay(self, image1, image2, color=(0, 0, 255)):
        """
        [NEW] 원본(혹은 CLAHE) 이미지 위에 Canny Edge를 빨간색으로 합성하여 반환
        """
        if image1 is None or image2 is None: return None

        # 1. Edge 추출
        edges = self.apply_canny(image1)

        # 2. Edge 마스크를 컬러로 변환
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # 3. Edge 부분에 색상 적용 (기본: Red)
        # BGR 순서이므로 (0, 0, 255)가 빨간색
        edges_colored[edges > 0] = color

        # 4. 이미지 합성 (원본 100% + 엣지 100%)
        overlay_img = cv2.addWeighted(image2, 1.0, edges_colored, 1.0, 0)

        return overlay_img

class ImageVisualizer:
    def __init__(self, image_dir):
        self.image_dir = image_dir

    def show_samples(self, num_samples=3):
        # 이미지 검색
        image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")) +
                             glob.glob(os.path.join(self.image_dir, "*.png")))

        if not image_paths:
            print(f"❌ 이미지가 없습니다: {self.image_dir}")
            return

        print(f"📊 Visualizing {num_samples} samples (Original -> CLAHE -> Edge -> Overlay)...")

        # 4개 컬럼: [원본] [CLAHE] [Edge] [CLAHE+Overlay]
        plt.figure(figsize=(20, 5 * num_samples))

        for i in range(min(num_samples, len(image_paths))):
            path = image_paths[i]
            original = cv2.imread(path)
            if original is None: continue

            # ---------------------------
            # [Step 1] CLAHE 적용
            # ---------------------------
            clahe_img = ImagePreprocessor().apply_clahe(original)


            # ---------------------------
            # [Step 3] Overlay (CLAHE 위에 엣지 얹기)
            # ---------------------------
            overlay = ImagePreprocessor().apply_canny_overlay(original, clahe_img, color=(0, 0, 255))  # 빨간색

            # === Plotting ===

            # 1. Original
            plt.subplot(num_samples, 3, i * 3 + 1)
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            plt.title(f"1. Original\n{os.path.basename(path)}")
            plt.axis('off')

            # 2. CLAHE
            plt.subplot(num_samples, 3, i * 3 + 2)
            plt.imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))
            plt.title("2. CLAHE (Enhanced)")
            plt.axis('off')

            # 4. Overlay (CLAHE + Edges)
            plt.subplot(num_samples, 3, i * 3 + 3)
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title("3. Overlay (CLAHE + Red Edge)")
            plt.axis('off')

        plt.tight_layout()
        plt.show()


# =========================================================
# 실행부
# =========================================================
if __name__ == "__main__":
    SAVE_DIR = "../data/dev/images"

    if os.path.exists(SAVE_DIR):
        # 1. 프로세서 설정
        # clip_limit를 높이면 대비가 더 강해짐 (보통 2.0 ~ 4.0 사용)
        processor = ImagePreprocessor(
            clip_limit=3.0,
            tile_grid_size=(8, 8),
            canny_low=50,
            canny_high=150
        )

        # 2. 시각화 실행
        visualizer = ImageVisualizer(image_dir=SAVE_DIR)
        visualizer.show_samples(num_samples=3)
    else:
        print(f"경로를 찾을 수 없습니다: {SAVE_DIR}")