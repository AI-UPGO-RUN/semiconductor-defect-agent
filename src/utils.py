import os
import re
import json
import base64  # 추가됨
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
# Base64 인코딩 유틸 (추가됨)
# =========================
def encode_image_to_base64(image: np.ndarray, ext: str = ".png") -> str:
    """
    OpenCV 이미지를 Base64 문자열로 변환 (Data URI Scheme 포함)
    예: data:image/png;base64,iVBORw0KGgo...
    """
    if image is None:
        return ""

    try:
        # 이미지를 메모리 버퍼로 인코딩
        success, buffer = cv2.imencode(ext, image)
        if not success:
            return ""

        # 버퍼를 Base64 문자열로 변환
        b64_str = base64.b64encode(buffer).decode("utf-8")

        # MIME 타입 결정 (확장자에서 . 제거)
        mime_type = f"image/{ext.replace('.', '')}"
        return f"data:{mime_type};base64,{b64_str}"
    except Exception as e:
        print(f"Base64 encoding error: {e}")
        return ""


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
      - URL로부터 이미지를 다운로드 (메모리)
      - 전처리 (CLAHE + Canny) 수행 (메모리)
      - 필요 시 저장도 가능하지만, 주 목적은 메모리 상 처리 지원
    """

    def __init__(
            self,
            raw_save_dir: Optional[str] = None,
            preprocessed_save_dir: Optional[str] = None,
            clip_limit: float = config.CLIP_LIMIT,
            tile_grid_size: Tuple[int, int] = config.TILE_GRID_SIZE,
            canny_low: int = config.CANNY_LOW,
            canny_high: int = config.CANNY_HIGH,
            gamma: float = 1.2,  # [NEW] 감마 값 (1.0보다 크면 어두운 곳 밝게)
    ):
        base_dir = config.BASE_DIR
        self.raw_save_dir = raw_save_dir or config.DEV_SAVE_DIR
        self.preprocessed_save_dir = preprocessed_save_dir or os.path.join(base_dir, "data", "dev",
                                                                           "preprocessed_images")

        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.gamma = gamma

        # self._create_directory(self.raw_save_dir)
        # self._create_directory(self.preprocessed_save_dir)

    def _create_directory(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def download_image(self, url: str) -> Optional[np.ndarray]:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
            return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except Exception:
            return None

    # (저장 메서드는 필요 시 사용, 로직상 유지)
    def save_raw_image(self, image: np.ndarray, image_id: str) -> Optional[str]:
        if image is None: return None
        try:
            file_name = f"{image_id}.png"
            save_path = os.path.join(self.raw_save_dir, file_name)
            cv2.imwrite(save_path, image)
            return save_path
        except Exception:
            return None

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        if image is None: return image
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            l_clahe = clahe.apply(l)
            lab_merged = cv2.merge((l_clahe, a, b))
            return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
        except Exception:
            return image

    def _apply_canny(self, image: np.ndarray) -> np.ndarray:
        if image is None: return image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        return edges

    # [NEW] 1. 노이즈 제거 (Gaussian Blur)
    # 미세한 노이즈를 날려서 Canny 에지가 너무 지저분해지는 것을 방지
    def _apply_denoise(self, image: np.ndarray) -> np.ndarray:
        if image is None: return image
        return cv2.GaussianBlur(image, (3, 3), 0)
    # [NEW] 2. 감마 보정 (Gamma Correction)
    # 어두운 부분을 밝게 하여 숨겨진 결함 드러내기
    def _apply_gamma(self, image: np.ndarray) -> np.ndarray:
        if image is None: return image
        inv_gamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    # [NEW] 3. 선명화 (Sharpening)
    # 흐릿한 스크래치를 또렷하게 만듦
    def _apply_sharpen(self, image: np.ndarray) -> np.ndarray:
        if image is None: return image
        # 중심 픽셀 강조 커널
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def _apply_canny_overlay(self, base_image: np.ndarray, overlay_target: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
        if base_image is None or overlay_target is None: return overlay_target
        edges = self._apply_canny(base_image)
        if edges is None: return overlay_target
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_colored[edges > 0] = color
        overlay_img = cv2.addWeighted(overlay_target, 1.0, edges_colored, 1.0, 0)
        return overlay_img

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        순서:
        1. Denoise (노이즈 제거)
        2. Gamma Correction (밝기 보정)
        3. Sharpen (선명화 - 스크래치 강조)
        4. CLAHE (대비 극대화)
        5. Canny Overlay (윤곽선 붉은색 표시)
        """
        # 1. 노이즈 제거
        denoised = self._apply_denoise(image)

        # 2. 감마 보정
        gamma_img = self._apply_gamma(denoised)

        # 3. 선명화 (Sharpness)
        sharpened = self._apply_sharpen(gamma_img)

        # 4. CLAHE (Local Contrast)
        clahe_img = self._apply_clahe(sharpened)

        # 5. Canny Overlay
        # 에지 추출은 노이즈가 적은 'denoised'나 'gamma_img' 기반으로 하는 게 깔끔할 수 있음
        # 배경은 보기가 가장 좋은 'clahe_img'를 사용
        final_img = self._apply_canny_overlay(base_image=sharpened, overlay_target=clahe_img, color=(0, 0, 255))

        return final_img