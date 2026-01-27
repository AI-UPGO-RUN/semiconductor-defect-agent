# src/config.py
import os
from dotenv import load_dotenv

# .env 파일 로드 (현재 위치 기준 상위 폴더나 같은 위치 탐색)
load_dotenv()

# 1. 보안 설정 (환경 변수에서 가져옴)
API_KEY = os.getenv("SALTLUX_API_KEY")
BRIDGE_URL = os.getenv("BRIDGE_URL")
MODEL = "gpt-4o-mini-2024-07-18"

# 2. 경로 설정
# (상대 경로 문제 방지를 위해 절대 경로로 변환하는 테크닉)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dev.csv")
SAVE_DIR = os.path.join(BASE_DIR, "data", "images")
OUT_PATH = os.path.join(BASE_DIR, "submission.csv")

# 3. 모델 파라미터
OBS_ITEMS = [
    ("package_damage", "패키지 손상"),
    # ...
]