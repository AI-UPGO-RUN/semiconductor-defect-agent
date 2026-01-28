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
        "Observe the image and output ONLY JSON.\n"
        "For each item, fill in value (true/false), confidence (0–1), and reason (brief).\n"
        "The output format MUST exactly match the schema below.\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n"
    )

    # [Core] Image Analysis Guide
    img_guide = (
        "\n[Image Analysis Guide: Connectivity-Focused Inspection]\n"
        "You are inspecting the electrical connectivity of a semiconductor device.\n"
        "Visual shape is secondary — **connectivity is the top priority**.\n\n"
        "- Ignore the orientation, rotation, or tilt of the transistor body.\n"
        "Evaluate defects ONLY based on whether each lead clearly enters a blue highlighted hole."


        "**1. Bent / Short Judgment (Continuity Check)**\n"
        "- **Normal (Pass)**: Even if a lead is bent, curved, or irregular,\n"
        "  it is considered normal **if a continuous red trace is visible from the device body\n"
        "  into the inside of a highlighted hole without interruption**.\n"
        "  In this case, classify `lead_defect=false`.\n"
        "- **Defective (Fail)**:\n"
        "  (a) The lead does not reach a highlighted hole and is interrupted midway (Short).\n"
        "  (b) The lead connects to an incorrect location (side hole, empty space, or outside the target region) (Misalignment).\n\n"

        "**2. Lead Counting Rule (50% Rule)**\n"
        "- **Count as present**: Even if the outline is faint or partially missing,\n"
        "  count the lead if **50% or more of its total length is visible**.\n"
        "- **Count as missing**: Reduce `visible_lead_count` ONLY if\n"
        "  the lead is completely detached from the body or **more than 50% is missing**.\n\n"

        "**3. Critical Notes**\n"
        "- Accurately identify the background holes.\n"
        "- The key question is whether the **tip of each lead clearly enters the highlighted circular hole region**.\n"
        "- A lead is considered connected ONLY if its endpoint is clearly inside the blue highlighted circular hole.\n"
        "Touching the edge or stopping just outside the circle is NOT sufficient."
        "- Ignore the rotation or orientation of the transistor package body."
        "- Focus on the DIRECTION of each lead:"
        "each lead must extend downward toward the bottom (6 o'clock direction)\n"
        "and clearly enter a blue highlighted hole.\n"
        "- If a lead points sideways or upward and cannot reach a hole,it must be considered defective."
    )

    criteria = "\n".join([f"- {k}: {desc}" for k, desc in OBS_ITEMS])

    if mode == "strict":
        rule = (
            "\nJudgment Rules:\n"
            "- Judge very conservatively. If uncertain, set value=false.\n"
            "- confidence represents certainty (0–1). If unsure, keep it ≤ 0.3.\n"
        )
    elif mode == "anchor_compare":
        rule = (
            "\nJudgment Rules:\n"
            "- The second provided image is a NORMAL anchor reference.\n"
            "- Compare the first image (inspection target) against the anchor for structural differences.\n"
            "- If uncertain, set value=false.\n"
        )
    else:
        rule = (
            "\nJudgment Rules:\n"
            "- Set value=true ONLY when the condition is very clear.\n"
            "- If ambiguous, set value=false.\n"
            "- confidence represents certainty (0–1).\n"
        )

    return header + img_guide + rule + "\nObservation Item Descriptions:\n" + criteria


def build_bottom_center_hole_mask(hole_ref_img: np.ndarray) -> np.ndarray:
    """
    Test_008.png 기준:
    - 모든 hole 검출
    - 가장 하단 row 선택
    - x 기준 중앙 3개 hole만 mask 생성
    """
    gray = cv2.cvtColor(hole_ref_img, cv2.COLOR_BGR2GRAY)

    # hole은 어두움
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # contour = hole 후보
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    holes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        holes.append((cx, cy, cnt))

    if not holes:
        return np.zeros_like(gray)

    # 🔽 가장 아래 row 선택
    max_y = max(h[1] for h in holes)
    row_thresh = 10
    bottom_row = [h for h in holes if abs(h[1] - max_y) < row_thresh]

    # 🔽 x 기준 정렬 → 중앙 3개
    bottom_row = sorted(bottom_row, key=lambda x: x[0])
    mid = len(bottom_row) // 2
    selected = bottom_row[mid - 1: mid + 2]

    # mask 생성
    mask = np.zeros_like(gray)
    for _, _, cnt in selected:
        cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)

    return mask


def overlay_holes(
    image: np.ndarray,
    hole_mask: np.ndarray,
    color=(255, 0, 0),  # 파랑 (BGR)
    alpha=0.6
) -> np.ndarray:
    overlay = image.copy()
    color_layer = np.zeros_like(image)
    color_layer[:] = color

    mask_3ch = cv2.cvtColor(hole_mask, cv2.COLOR_GRAY2BGR)

    return np.where(
        mask_3ch > 0,
        cv2.addWeighted(image, 1 - alpha, color_layer, alpha, 0),
        image
    )


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
        hole_ref = cv2.imread("data/Test_008.png")
        if hole_ref is None:
            raise RuntimeError("Failed to load Test_008.png for hole reference")

        self.bottom_center_hole_mask = build_bottom_center_hole_mask(hole_ref)

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
        denoised = self._apply_denoise(image)
        gamma_img = self._apply_gamma(denoised)
        sharpened = self._apply_sharpen(gamma_img)
        clahe_img = self._apply_clahe(sharpened)

        # 🔵 하단 중앙 3개 hole만 강조
        hole_overlayed = overlay_holes(
            clahe_img,
            self.bottom_center_hole_mask,
            color=(255, 0, 0),
            alpha=0.6
        )

        # 🔴 리드 윤곽 강조
        final_img = self._apply_canny_overlay(
            base_image=sharpened,
            overlay_target=hole_overlayed,
            color=(0, 0, 255)
        )

        return final_img

if __name__ == "__main__":
    import cv2
    from src.utils import ImageHandler

    # ImageHandler 초기화 (이미 수정된 utils.py 기준)
    handler = ImageHandler()

    # 테스트할 이미지 경로 or URL
    img_path = "../data/TEST_007.png"   # ← 아무 DEV 이미지 하나
    # img_url = "https://..."           # URL 테스트 시

    # --- 로드 ---
    raw_img = cv2.imread(img_path)
    # raw_img = handler.download_image(img_url)

    if raw_img is None:
        raise RuntimeError("이미지 로드 실패")

    # --- 전처리 ---
    prep_img = handler.preprocess_image(raw_img)

    # --- 시각화 ---
    cv2.imshow("RAW", raw_img)
    cv2.imshow("PREPROCESSED (Bottom-Center Holes Highlighted)", prep_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
