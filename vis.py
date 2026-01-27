import pandas as pd
import cv2
import matplotlib.pyplot as plt
from src import config
from src.utils import ImageHandler

# ==========================================
# 설정
# ==========================================
TARGET_ID = "DEV_014"  # CSV에서 찾을 ID
FIG_SIZE = (20, 8)  # 그래프 크기
GAMMA_VAL = 1.2  # 핸들러 설정값


def visualize_from_csv(target_id):
    # ---------------------------------------------------------
    # 1. CSV 파일 읽기 (config.DEV_INPUT_PATH 사용)
    # ---------------------------------------------------------
    csv_path = config.DEV_INPUT_PATH
    print(f"📂 Reading CSV file: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ CSV Read Error: {e}")
        return

    # 필요한 컬럼 확인
    if "id" not in df.columns or "img_url" not in df.columns:
        print(f"❌ Error: CSV must have 'id' and 'img_url' columns. Found: {df.columns.tolist()}")
        return

    # ---------------------------------------------------------
    # 2. TARGET_ID에 해당하는 URL 찾기
    # ---------------------------------------------------------
    target_row = df[df["id"] == target_id]

    if target_row.empty:
        print(f"❌ Error: ID '{target_id}' not found in CSV.")
        return

    # 첫 번째 매칭되는 행의 URL 가져오기
    target_url = target_row.iloc[0]["img_url"]
    print(f"🎯 Found ID: {target_id}")
    print(f"🔗 Target URL: {target_url}")

    # ---------------------------------------------------------
    # 3. 이미지 다운로드 (ImageHandler 사용)
    # ---------------------------------------------------------
    handler = ImageHandler(gamma=GAMMA_VAL)
    print("⬇️  Downloading image...")

    raw_img = handler.download_image(target_url)

    if raw_img is None:
        print("❌ Image download failed.")
        return

    # ==========================================
    # 4. 단계별 전처리 수행 (시각화용 분해)
    # ==========================================
    steps = []

    # Step 0: 원본
    steps.append(("0. Original", raw_img, False))

    # Step 1: Denoise
    denoised = handler._apply_denoise(raw_img)
    steps.append(("1. Denoise (Blur)", denoised, False))

    # Step 2: Gamma
    gamma_img = handler._apply_gamma(denoised)
    steps.append((f"2. Gamma (g={GAMMA_VAL})", gamma_img, False))

    # Step 3: Sharpening
    sharpened = handler._apply_sharpen(gamma_img)
    steps.append(("3. Sharpening", sharpened, False))

    # Step 4: CLAHE
    clahe_img = handler._apply_clahe(sharpened)
    steps.append(("4. CLAHE (Contrast)", clahe_img, False))

    # Step 5: Canny Edges
    edges = handler._apply_canny(sharpened)
    steps.append(("5. Canny Edges (Mask)", edges, True))

    # Step 6: Final Overlay
    final_img = handler._apply_canny_overlay(base_image=sharpened, overlay_target=clahe_img, color=(0, 0, 255))
    steps.append(("6. Final Result (Overlay)", final_img, False))

    # ==========================================
    # 5. Plotting
    # ==========================================
    plt.figure(figsize=FIG_SIZE)
    rows, cols = 1, 7

    for i, (title, img, is_gray) in enumerate(steps):
        plt.subplot(rows, cols, i + 1)
        plt.title(title, fontsize=13)

        if is_gray:
            plt.imshow(img, cmap='gray')
        else:
            # OpenCV (BGR) -> Matplotlib (RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)

        plt.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


if __name__ == "__main__":
    visualize_from_csv(TARGET_ID)