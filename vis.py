import cv2
import numpy as np

from src.utils import ImageHandler, overlay_holes
from src import config


def show_hole_mask(mask: np.ndarray) -> np.ndarray:
    """
    hole mask 단독 시각화 (회색 → 컬러)
    """
    colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    colored[mask > 0] = (255, 0, 0)  # 파랑
    return colored


if __name__ == "__main__":
    # =========================
    # ImageHandler 초기화
    # =========================
    handler = ImageHandler()

    # =========================
    # 테스트 이미지
    # =========================
    img_path = "data/TEST_007.png"   # ← 원하는 이미지
    raw_img = cv2.imread(img_path)

    if raw_img is None:
        raise RuntimeError("이미지 로드 실패")

    # =========================
    # hole mask 확인
    # =========================
    hole_mask = handler.bottom_center_hole_mask

    hole_mask_vis = show_hole_mask(hole_mask)

    # =========================
    # hole overlay 확인
    # =========================
    hole_overlay = overlay_holes(
        raw_img,
        hole_mask,
        color=(255, 0, 0),   # 파랑
        alpha=0.6
    )

    # =========================
    # 전체 preprocess 결과
    # =========================
    preprocessed = handler.preprocess_image(raw_img)

    # =========================
    # 시각화
    # =========================
    cv2.imshow("RAW IMAGE", raw_img)
    cv2.imshow("BOTTOM-CENTER 3 HOLE MASK", hole_mask_vis)
    cv2.imshow("RAW + HOLE OVERLAY", hole_overlay)
    cv2.imshow("FINAL PREPROCESSED", preprocessed)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
