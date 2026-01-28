import os
import cv2
import pandas as pd
import time
from src import config, agent, utils
from src.utils import ImageHandler  # ImageHandler 임포트

LABEL_NORMAL = config.LABEL_NORMAL
TEST_INPUT_PATH = config.TEST_INPUT_PATH
DEV_INPUT_PATH = config.DEV_INPUT_PATH
GITHUB_BASE_URL = config.GITHUB_BASE_URL
TEST_OUTPUT_PATH = config.TEST_OUTPUT_PATH

# ImageHandler 초기화
handler = ImageHandler()

df = pd.read_csv(TEST_INPUT_PATH)
df_anchor = pd.read_csv(DEV_INPUT_PATH)
if "id" not in df.columns or "img_url" not in df.columns:
    raise ValueError(f"columns: {df.columns.tolist()}")

# -------------------------------------------------------------------------
# 1. 정상 앵커(Anchor) 이미지 준비 (미리 다운로드 -> 전처리 -> Base64 변환)
# -------------------------------------------------------------------------
ANCHOR_IDS = {"DEV_000", "DEV_002", "DEV_003", "DEV_004", "DEV_006", "DEV_010", "DEV_011", "DEV_012", "DEV_013",
              "DEV_015", "DEV_018", "DEV_019"}
anchor_urls_raw = df_anchor[df_anchor["id"].isin(ANCHOR_IDS)]["img_url"].tolist()

anchor_b64s = []
print("Initializing Anchor Images (Download & Preprocess & Base64 Encode)...")

for idx, a_url in enumerate(anchor_urls_raw):
    # 다운로드 (메모리)
    raw_img = handler.download_image(a_url)
    if raw_img is not None:
        b64_str = utils.encode_image_to_base64(raw_img)
        if b64_str:
            anchor_b64s.append(b64_str)

if not anchor_b64s:
    print("WARNING: No anchor images loaded.")
else:
    print(f"Loaded {len(anchor_b64s)} anchor images as Base64.")

# -------------------------------------------------------------------------
# 2. 메인 루프 (대상 이미지 다운로드 -> 전처리 -> Base64 -> Agent 호출)
# -------------------------------------------------------------------------
results = []
n = len(df)

for i, row in df.iterrows():
    _id = row["id"]
    img_url = row["img_url"]

    try:
        # [Step 1] 다운로드
        raw_img = handler.download_image(img_url)
        if raw_img is None:
            raise ValueError("Image download failed")

        # [Step 2] 전처리 (CLAHE + Canny)
        prep_img = handler.preprocess_image(raw_img)

        # [Step 3] Base64 인코딩 (파일 저장 안 함)
        raw_b64 = utils.encode_image_to_base64(raw_img)  # 원본
        prep_b64 = utils.encode_image_to_base64(prep_img)  # 전처리

        input_images = [raw_b64, prep_b64]

        # [Step 4] Agent 호출
        # 주의: agent.classify_agent가 첫 번째 인자로 URL 대신 Base64 문자열을 받도록
        #       agent 쪽 코드도 대응이 필요합니다. (또는 LLM이 Data URI를 URL로 인식하므로 그대로 전달)
        label, obs, dec = agent.classify_agent(
            input_images,  # URL 대신 Base64 전달
            anchor_urls=anchor_b64s,  # 앵커도 Base64 리스트 전달
            dbg_id=_id
        )

        results.append({
            "id": _id,
            "img_url": img_url,
            "label": label,
            "why": dec.why,
            "score": dec.score,
            "uncertain": dec.uncertain,
        })
        print(f"[{i + 1}/{n}] id={_id} -> {label} | {dec.why}")

    except Exception as e:
        print(f"[{i + 1}/{n}] id={_id} ERROR -> fallback 0 | {e}")
        results.append({"id": _id, "img_url": img_url, "label": LABEL_NORMAL, "why": f"ERROR: {e}", "score": -1,
                        "uncertain": True})

    time.sleep(1)

# -------------------------------------------------------------------------
# 3. 재검토 (Validate) - 재검토 시에도 Base64 앵커 사용
# -------------------------------------------------------------------------
# agent.validate_and_maybe_rerun 내부에서도 Base64를 처리할 수 있어야 합니다.
results = agent.validate_and_maybe_rerun(results, anchor_urls=anchor_b64s)

# 제출 파일
out_df = pd.DataFrame([{"id": r["id"], "label": r["label"]} for r in results], columns=["id", "label"])
out_df.to_csv(TEST_OUTPUT_PATH, index=False)
print(f"\n✅ Saved: {TEST_OUTPUT_PATH}")
print(out_df.head())