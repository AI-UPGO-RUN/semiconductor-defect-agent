# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("LUXIA_API_KEY")

# 모델 변경 시 변경 필요
BRIDGE_URL = "https://bridge.luxiacloud.com/llm/openai/chat/completions/gpt-4o-mini/create"
MODEL = "llm"

# 경로 설정
# - BASE_DIR: 현재 디렉토리
# - DEV_CSV_PATH: DEV용 INPUT CSV 경로
# - DEV_SAVE_DIR: DEV용 IMAGE SAVE 디렉토리
# - DEV_OUTPUT_PATH: DEV용 OUTPUT CSV 경로
# - DEV_DEBUG_DIR: DEV용 DEBUG LOG 디렉토리
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV_INPUT_PATH = os.path.join(BASE_DIR, "data", "dev", "dev.csv")
DEV_SAVE_DIR = os.path.join(BASE_DIR, "data", "dev", "images")
DEV_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "dev", "submission.csv")
DEV_DEBUG_DIR = os.path.join(BASE_DIR, "logs", "dev")
GITHUB_BASE_URL = "https://raw.githubusercontent.com/AI-UPGO-RUN/semiconductor-defect-agent/main/data/dev/preprocessed_images"

# 헤더 정보
HEADERS = {"apikey": API_KEY, "Content-Type": "application/json"}

# 전처리 정보
CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)
CANNY_LOW = 50
CANNY_HIGH = 150

# 라벨 정보
LABEL_NORMAL = 0
LABEL_ABNORMAL = 1

# 관찰 항목
OBS_ITEMS = [
    ("lead_missing_or_short", "리드가 0~2개이거나, 리드가 정상 대비 길이가 짧거나 소실됨"),
    ("lead_severe_bend_or_cross", "리드가 휘거나 벌어져 비정상 형태거나, 리드끼리 교차/접촉했거나, 구멍 안에 정상적으로 결합되지 않은 리드가 존재함"),
    ("lead_asymmetry", "각 리드 간격 또는 각도가 비대칭(단독 불량 근거로 사용 금지)"),
    ("device_tilt_or_rotation", "소자 본체가 정상 소자 대비 기울거나 회전됨"),
    ("misalignment_severe", "소자 중심이 뚜렷하게 벗어남(단독 불량 근거 금지)"),
    ("package_damage", "소자에 깨짐/뜯김/크랙 등 외곽 손상이 존재"),
]
KEYS = [k for k, _ in OBS_ITEMS]

# 가중치
WEIGHTS = {
    "lead_missing_or_short": 5.0,
    "lead_severe_bend_or_cross": 4.0,
    "lead_asymmetry": 1.0,
    "device_tilt_or_rotation": 2.0,
    "misalignment_severe": 1.5,
    "package_damage": 4.0,
}
HARD_ABNORMAL_KEYS = {"lead_missing_or_short", "lead_severe_bend_or_cross", "package_damage"}

# 시스템 프롬프트
SYSTEM = (
    "너는 반도체 소자 검사 이미지 분석기다.\n"
    "불량인 소자는 반드시 걸러낸다.\n"
    "기준을 명확하게 파악하고 판단에 활용한다.\n"
    "불량인 소자를 걸러낼 때에는 불량 이유를 명확하고 정확하게 특정한다.\n"
    "반드시 요청한 JSON만 출력한다. 다른 텍스트는 절대 출력하지 않는다.\n"
)

STRUCT_PROMPT = """
    첫 번째 이미지를 보고 아래 JSON만 출력해.
    - visible_lead_count: 눈에 보이는 정상적인 리드 개수(0~3 정수)
    - rotation_severity: 0(거의 수직) / 1(약간 기울) / 2(명확히 기울)
    - severe_bend: 리드가 과도하게 휘었거나 교차/접촉하면 true
    - occluded: 가림/조명/해상도 문제로 확신이 어려우면 true
    
    JSON 형식:
    {
      "visible_lead_count": 3,
      "rotation_severity": 0,
      "severe_bend": false,
      "occluded": false
    }   
"""

# 디버그 여부
DEBUG = True
