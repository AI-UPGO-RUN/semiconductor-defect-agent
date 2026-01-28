import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("LUXIA_API_KEY")

# =====================================================
# Model / API
# =====================================================
BRIDGE_URL = "https://bridge.luxiacloud.com/llm/openai/chat/completions/gpt-4o-mini/create"
MODEL = "llm"

# =====================================================
# Paths
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV_INPUT_PATH = os.path.join(BASE_DIR, "data", "dev", "dev.csv")
TEST_INPUT_PATH = os.path.join(BASE_DIR, "data", "test", "test.csv")
DEV_SAVE_DIR = os.path.join(BASE_DIR, "data", "dev", "images")
DEV_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "dev", "submission.csv")
TEST_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "test", "submission.csv")
DEV_DEBUG_DIR = os.path.join(BASE_DIR, "logs", "dev")

GITHUB_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "AI-UPGO-RUN/semiconductor-defect-agent/"
    "main/data/dev/preprocessed_images"
)

# =====================================================
# Headers
# =====================================================
HEADERS = {"apikey": API_KEY, "Content-Type": "application/json"}

# =====================================================
# Preprocessing Parameters
# =====================================================
CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)
CANNY_LOW = 50
CANNY_HIGH = 150

# =====================================================
# Labels
# =====================================================
LABEL_NORMAL = 0
LABEL_ABNORMAL = 1   # IMPORTANT: 1 = ABNORMAL

# =====================================================
# Observation Items (Hole-aware)
# =====================================================
OBS_ITEMS = [
    (
        "lead_missing_or_short",
        "One or more leads are missing, truncated, or do not reach the "
        "highlighted bottom-center hole region (0–2 valid leads visible)."
    ),
    (
        "lead_severe_bend_or_cross",
        "Any lead is severely bent, spread abnormally, crossed or touching "
        "another lead, or fails to be vertically inserted into one of the "
        "three highlighted bottom-center holes."
    ),
    (
        "lead_asymmetry",
        "Spacing or angle between leads is asymmetric "
        "(must NOT be used as the sole reason for abnormality)."
    ),
    (
        "device_tilt_or_rotation",
        "The device body is tilted or rotated compared to a normal reference."
    ),
    (
        "misalignment_severe",
        "The device center is clearly misaligned "
        "(must NOT be used as the sole reason for abnormality)."
    ),
    (
        "package_damage",
        "Cracks, chips, or visible damage exist on the device package."
    ),
]

KEYS = [k for k, _ in OBS_ITEMS]

# =====================================================
# Weights (F1-score optimized)
# =====================================================
WEIGHTS = {
    "lead_missing_or_short": 6.0,       # strongest signal
    "lead_severe_bend_or_cross": 5.0,   # includes hole insertion failure
    "lead_asymmetry": 0.5,
    "device_tilt_or_rotation": 1.5,
    "misalignment_severe": 0.5,
    "package_damage": 4.0,
}

# Immediate abnormal triggers
HARD_ABNORMAL_KEYS = {
    "lead_missing_or_short",
    "lead_severe_bend_or_cross",
}

# =====================================================
# System Prompt (Critical for F1)
# =====================================================
SYSTEM = (
    "You are an image inspection AI for semiconductor devices.\n"
    "Your primary goal is to detect abnormal (defective) devices.\n"
    "False negatives are unacceptable: if a defect is suspected, "
    "classify the device as abnormal.\n"
    "The provided image is a preprocessed inspection image.\n"
    "Only the three highlighted holes at the bottom-center of the image "
    "are valid lead insertion targets.\n"
    "Each lead must form a continuous, unbroken connection from the "
    "device body into one of these highlighted holes.\n"
    "If a lead does not clearly enter a highlighted hole, it must be "
    "considered defective.\n"
    "When judging abnormality, identify the reason clearly and precisely.\n"
    "Output ONLY the requested JSON. Do NOT output any other text.\n"
    "The device is a transistor consisting of one black package and "
    "three silver leads.\n"
    "The image may contain NO transistor device at all.\n"
    "If no transistor body and no valid leads are visible,"
    "the device must be classified as abnormal."
    "Each lead must extend downward toward the bottom-center holes"
    "(i.e., approximately the 6 o'clock direction)."
    "The orientation of the package body itself is irrelevant."
)

# =====================================================
# Structured Output Prompt
# =====================================================
STRUCT_PROMPT = """
Look at the first image and output ONLY the following JSON.

- visible_lead_count:
  The number of leads that form a continuous connection from the device body
  into the highlighted bottom-center holes (integer 0–3).

- rotation_severity:
  0 = nearly vertical,
  1 = slightly tilted,
  2 = clearly tilted or rotated.

- severe_bend:
  true if any lead is severely bent, crossed, touching another lead,
  or fails to enter a highlighted hole in a vertical and stable manner.

- occluded:
  true if occlusion, lighting, or resolution issues prevent confident judgment.

JSON format:
{
  "visible_lead_count": 3,
  "rotation_severity": 0,
  "severe_bend": false,
  "occluded": false
}
"""

# =====================================================
# Debug
# =====================================================
DEV_DEBUG = True
