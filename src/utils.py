import re
import requests
from src import config
import json

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