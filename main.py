import os

import cv2
import pandas as pd
import time
from src import config, agent

LABEL_NORMAL = config.LABEL_NORMAL
DEV_INPUT_PATH = config.DEV_INPUT_PATH
DEV_OUTPUT_PATH = config.DEV_OUTPUT_PATH

# 사전 전처리된 PNG 이미지가 저장된 디렉토리
PREP_DIR = os.path.join(config.BASE_DIR, "data", "dev", "preprocessed_images")

df = pd.read_csv(DEV_INPUT_PATH)
if "id" not in df.columns or "img_url" not in df.columns:
    raise ValueError(f"columns: {df.columns.tolist()}")

# 정상 앵커 URL 수집:
ANCHOR_IDS = {"DEV_000", "DEV_002", "DEV_003", "DEV_004", "DEV_006", "DEV_010", "DEV_011", "DEV_012", "DEV_013",
              "DEV_015", "DEV_018", "DEV_019"}
anchor_urls = df[df["id"].isin(ANCHOR_IDS)]["img_url"].tolist()
if len(anchor_urls) == 0:
    # 앵커가 없으면 그냥 빈 리스트로 진행
    anchor_urls = []

results = []
n = len(df)

for i, row in df.iterrows():
    _id = row["id"]
    img_url = row["img_url"]

    try:
        # 로컬에 미리 전처리된 PNG 사용 (예: data/dev/preprocessed_images/DEV_000.png)
        img_path = os.path.join(PREP_DIR, f"{_id}.png")

        if os.path.exists(img_path):
            preprocessed_img = cv2.imread(img_path)
        else:
            preprocessed_img = None

        if preprocessed_img is None:
            # 로컬 파일이 없거나 로드 실패 시, 기존처럼 URL을 직접 Agent에 전달
            final_input = img_url
            print(f"Local image not found or failed to load for {_id}, using URL fallback.")
        else:
            final_input = preprocessed_img

        label, obs, dec = agent.classify_agent(final_input, anchor_urls=anchor_urls, dbg_id=_id)
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
        results.append({"id": _id, "img_url": img_url, "label": LABEL_NORMAL, "why": "ERROR_FALLBACK", "score": -1,
                        "uncertain": True})

    time.sleep(0.2)

# 전체 분포 검증 후 uncertain만 추가 재검토
results = agent.validate_and_maybe_rerun(results, anchor_urls=anchor_urls)

# 제출 파일
out_df = pd.DataFrame([{"id": r["id"], "label": r["label"]} for r in results], columns=["id", "label"])
out_df.to_csv(DEV_OUTPUT_PATH, index=False)
print(f"\n✅ Saved: {DEV_OUTPUT_PATH}")
print(out_df.head())