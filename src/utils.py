import os
import re
import json
from typing import Optional, Tuple

import cv2
import numpy as np
import requests

from src import config

MODEL = config.MODEL
BRIDGE_URL = config.BRIDGE_URL
HEADERS = config.HEADERS
OBS_ITEMS = config.OBS_ITEMS

# =========================
# LLM 호출/파싱 유틸
# =========================
def post_chat(messages, timeout=90) -> str:
    payload = {"model": MODEL, "messages": messages, "stream": False}
    r = requests.post(BRIDGE_URL, headers=HEADERS, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"status={r.status_code}, body={r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"].strip()

def safe_json_extract(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"JSON parse failed: {s[:200]}")

# =========================
# 프롬프트 생성
# =========================
def build_prompt(mode: str = "normal") -> str:
    """
    mode:
      - normal: 일반 관찰
      - strict: 애매하면 false 쪽으로
      - anchor_compare: 정상 앵커와 비교(재검토용)
    """
    schema = {
        k: {"value": False, "confidence": 0.0, "reason": ""}
        for k, _ in OBS_ITEMS
    }
    header = (
        "아래 항목을 이미지에서 관찰해 JSON만 출력해.\n"
        "각 항목은 value(true/false), confidence(0~1), reason(짧게)로 채워.\n"
        "형식은 반드시 아래 스키마와 동일해야 한다.\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n"
    )

    criteria = "\n".join([f"- {k}: {desc}" for k, desc in OBS_ITEMS])

    if mode == "strict":
        rule = (
            "\n판단 기준:\n"
            "- 매우 보수적으로 판단한다. 애매하면 value=false.\n"
            "- confidence는 확신 정도(0~1). 확신 없으면 0.3 이하로 둔다.\n"
        )
    elif mode == "anchor_compare":
        rule = (
            "\n판단 기준:\n"
            "- 두 번째/세 번째로 제공되는 이미지는 '정상 앵커'다.\n"
            "- 첫 번째 이미지(검사 대상)가 앵커와 구조적으로 다른지 비교해서 판단한다.\n"
            "- 애매하면 value=false.\n"
        )
    else:
        rule = (
            "\n판단 기준:\n"
            "- 아주 명확할 때만 value=true. 애매하면 false.\n"
            "- confidence는 확신 정도(0~1).\n"
        )

    return header + rule + "\n관찰 항목 설명:\n" + criteria


class ImageHandler:
    """
    역할:
      - URL로부터 이미지를 다운로드
      - 원본 이미지를 저장
      - CLAHE + Canny Edge Overlay 전처리 수행
      - 전처리 이미지를 `data/dev/preprocessed_images`에 PNG로 저장
        (파일명 예: DEV_000.png)
    """

    def __init__(
        self,
        raw_save_dir: Optional[str] = None,
        preprocessed_save_dir: Optional[str] = None,
        clip_limit: float = config.CLIP_LIMIT,
        tile_grid_size: Tuple[int, int] = config.TILE_GRID_SIZE,
        canny_low: int = config.CANNY_LOW,
        canny_high: int = config.CANNY_HIGH,
    ):
        base_dir = config.BASE_DIR

        # 원본 이미지 저장 경로 (기본: 기존 DEV_SAVE_DIR)
        self.raw_save_dir = raw_save_dir or config.DEV_SAVE_DIR

        # 전처리 이미지 저장 경로 (기본: BASE_DIR/data/dev/preprocessed_images)
        self.preprocessed_save_dir = (
            preprocessed_save_dir
            or os.path.join(base_dir, "data", "dev", "preprocessed_images")
        )

        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.canny_low = canny_low
        self.canny_high = canny_high

        # 디렉토리 생성
        self._create_directory(self.raw_save_dir)
        self._create_directory(self.preprocessed_save_dir)

    # ==============
    # 내부 유틸
    # ==============
    def _create_directory(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    # ==============
    # 다운로드 / 저장
    # ==============
    def download_image(self, url: str) -> Optional[np.ndarray]:
        """URL에서 이미지를 BGR(OpenCV) 포맷으로 로드"""
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
            return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def save_raw_image(self, image: np.ndarray, image_id: str) -> Optional[str]:
        """원본 이미지를 PNG로 저장 (예: DEV_000.png)"""
        if image is None:
            return None
        try:
            file_name = f"{image_id}.png"
            save_path = os.path.join(self.raw_save_dir, file_name)
            cv2.imwrite(save_path, image)
            return save_path
        except Exception:
            return None

    def save_preprocessed_image(self, image: np.ndarray, image_id: str) -> Optional[str]:
        """전처리 이미지를 PNG로 저장 (예: DEV_000.png)"""
        if image is None:
            return None
        try:
            file_name = f"{image_id}.png"
            save_path = os.path.join(self.preprocessed_save_dir, file_name)
            cv2.imwrite(save_path, image)
            return save_path
        except Exception:
            return None

    # ==============
    # 전처리 (CLAHE + Canny Edge Overlay)
    # ==============
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """컬러 이미지에 CLAHE 적용 (LAB 색상 공간 사용)"""
        if image is None:
            return image

        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
            )
            l_clahe = clahe.apply(l)

            lab_merged = cv2.merge((l_clahe, a, b))
            return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
        except Exception:
            # 문제가 생기면 원본을 그대로 사용
            return image

    def _apply_canny(self, image: np.ndarray) -> np.ndarray:
        """이미지에 Canny Edge Detection 적용"""
        if image is None:
            return image

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        return edges

    def _apply_canny_overlay(
        self, base_image: np.ndarray, clahe_image: np.ndarray, color=(0, 0, 255)
    ) -> np.ndarray:
        """
        CLAHE 전처리된 이미지 위에 Canny Edge를 컬러로 얹어서 반환
        (기본: 빨간색 Edge Overlay)
        """
        if base_image is None or clahe_image is None:
            return clahe_image

        edges = self._apply_canny(base_image)
        if edges is None:
            return clahe_image

        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_colored[edges > 0] = color

        overlay_img = cv2.addWeighted(clahe_image, 1.0, edges_colored, 1.0, 0)
        return overlay_img

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        외부에서 호출용:
          - CLAHE 적용
          - Canny Edge Overlay 적용
        """
        clahe_img = self._apply_clahe(image)
        overlay = self._apply_canny_overlay(image, clahe_img, color=(0, 0, 255))
        return overlay

    # ==============
    # 전체 파이프라인
    # ==============
    def process_from_url(self, url: str, image_id: str) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
        """
        URL로부터:
          1) 이미지 다운로드
          2) 원본 PNG 저장 (DEV_000.png)
          3) Canny Edge Overlay 전처리
          4) 전처리 PNG 저장 (DEV_000.png, data/dev/preprocessed_images)
        반환값:
          (전처리이미지, 원본경로, 전처리경로)
        """
        img_raw = self.download_image(url)
        if img_raw is None:
            return None, None, None

        raw_path = self.save_raw_image(img_raw, image_id)
        preprocessed_img = self.preprocess_image(img_raw)
        preprocessed_path = self.save_preprocessed_image(preprocessed_img, image_id)

        return preprocessed_img, raw_path, preprocessed_path


# if __name__ == "__main__":
#     """
#     간단 일괄 테스트용:
#       - DEV CSV 전체 row(id, img_url)를 순회
#       - ImageHandler로 다운로드/전처리 수행
#       - 원본/전처리 이미지를 모두 PNG로 저장
#     """
#     import pandas as pd
#
#     csv_path = config.DEV_INPUT_PATH
#     df = pd.read_csv(csv_path)
#     if "id" not in df.columns or "img_url" not in df.columns:
#         raise ValueError(f"columns: {df.columns.tolist()}")
#
#     handler = ImageHandler()
#
#     n = len(df)
#     print(f"[ImageHandler 테스트] 총 {n}개 이미지 처리 시작 (원본/전처리 PNG 저장)")
#
#     for i, row in df.iterrows():
#         _id = row["id"]
#         img_url = row["img_url"]
#
#         pre_img, raw_p, pre_p = handler.process_from_url(img_url, _id)
#
#         if pre_img is None:
#             print(f"[{i + 1}/{n}] {_id} - 실패 (다운로드 오류)")
#         else:
#             print(f"[{i + 1}/{n}] {_id} - OK | raw={raw_p}, prep={pre_p}")
#
#     print("✅ ImageHandler 일괄 테스트 완료")