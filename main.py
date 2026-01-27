import re
import json
import time
import random
import requests
import pandas as pd
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional


# =========================
# 설정
# =========================
API_KEY = ""
BRIDGE_URL = "https://bridge.luxiacloud.com/llm/openai/chat/completions/gpt-4o-mini/create"
MODEL = "gpt-4o-mini-2024-07-18"

TEST_CSV_PATH = "./dev.csv"
OUT_PATH = "./submission.csv"

HEADERS = {"apikey": API_KEY, "Content-Type": "application/json"}

LABEL_NORMAL = 0
LABEL_ABNORMAL = 1

DEBUG = True
DEBUG_DIR = "./debug_logs"
os.makedirs(DEBUG_DIR, exist_ok=True)

# =========================
# 관찰 항목
# =========================
OBS_ITEMS = [
    ("lead_missing_or_short", "리드가 0~2개이거나, 리드가 정상 대비 길이가 짧거나 소실됨"),
    ("lead_severe_bend_or_cross", "리드가 휘거나 벌어져 비정상 형태거나, 리드끼리 교차/접촉했거나, 구멍 안에 정상적으로 결합되지 않은 리드가 존재함"),
    ("lead_asymmetry", "각 리드 간격 또는 각도가 비대칭(단독 불량 근거로 사용 금지)"),
    ("device_tilt_or_rotation", "소자 본체가 정상 소자 대비 기울거나 회전됨"),
    ("misalignment_severe", "소자 중심이 뚜렷하게 벗어남(단독 불량 근거 금지)"),
    ("package_damage", "소자에 깨짐/뜯김/크랙 등 외곽 손상이 존재"),
]
KEYS = [k for k, _ in OBS_ITEMS]

# =========================
# 가중치/룰
# =========================
WEIGHTS = {
    "lead_missing_or_short": 5.0,
    "lead_severe_bend_or_cross": 4.0,
    "lead_asymmetry": 1.0,
    "device_tilt_or_rotation": 2.0,
    "misalignment_severe": 1.5, 
    "package_damage": 4.0,
}

HARD_ABNORMAL_KEYS = {"lead_missing_or_short", "lead_severe_bend_or_cross", "package_damage"}

# 불량 사유가 여럿인 경우
def combo_abnormal(obs: Dict[str, Any]) -> bool:
    # tilt + (misalignment or package_damage)면 불량 가능성 높음 (DEV_014류)
    return bool(obs["device_tilt_or_rotation"]["value"]) and (
        bool(obs["misalignment_severe"]["value"])
    )

# =========================
# 시스템 프롬프트
# =========================
SYSTEM = (
    "너는 반도체 소자 검사 이미지 분석기다.\n"
    "불량인 소자는 반드시 걸러낸다.\n"
    "기준을 명확하게 파악하고 판단에 활용한다.\n"
    "불량인 소자를 걸러낼 때에는 불량 이유를 명확하고 정확하게 특정한다.\n"
    "반드시 요청한 JSON만 출력한다. 다른 텍스트는 절대 출력하지 않는다.\n"
)

# =========================
# LLM 호출/파싱 유틸
# =========================
def _post_chat(messages, timeout=90) -> str:
    payload = {"model": MODEL, "messages": messages, "stream": False}
    r = requests.post(BRIDGE_URL, headers=HEADERS, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"status={r.status_code}, body={r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"].strip()

def _safe_json_extract(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"JSON parse failed: {s[:200]}")
    

# =========================
# 프롬프트: value + confidence + reason 스키마
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

PROMPT_NORMAL = build_prompt("normal")
PROMPT_STRICT = build_prompt("strict")
PROMPT_ANCHOR = build_prompt("anchor_compare")
# =========================
# 구조 관찰(객관식)
# =========================
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

def observe_struct(img_url: str, dbg_id: Optional[str] = None) -> Dict[str, Any]:
    raw = _post_chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": STRUCT_PROMPT},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]},
    ])
    parsed = _safe_json_extract(raw)
    out = {
        "visible_lead_count": int(parsed.get("visible_lead_count", 3)),
        "rotation_severity": int(parsed.get("rotation_severity", 0)),
        "severe_bend": bool(parsed.get("severe_bend", False)),
        "occluded": bool(parsed.get("occluded", False)),
    }
    if DEBUG and dbg_id:
        with open(os.path.join(DEBUG_DIR, f"{dbg_id}_struct.json"), "w", encoding="utf-8") as f:
            json.dump({"id": dbg_id, "raw_text": raw[:4000], "parsed": parsed, "struct": out},
                      f, ensure_ascii=False, indent=2)
    return out


# =========================
# Observe: 앵커 이미지 함께 넣기
# =========================
def _normalize_item(v) -> Dict[str, Any]:
    """
    LLM이 여러 형태로 주는 출력을 최대한 흡수:
    - bool -> {"value": bool, "confidence": 0.7, "reason": ""}
    - {"value": true} -> confidence 없으면 0.6 기본
    - {"confidence": 0.8}만 있으면 value는 confidence>=0.55로 추정
    """
    if isinstance(v, bool):
        return {"value": v, "confidence": 0.7 if v else 0.0, "reason": ""}
    if isinstance(v, (int, float)):
        c = float(v)
        return {"value": c >= 0.55, "confidence": max(0.0, min(1.0, c)), "reason": ""}
    if isinstance(v, dict):
        value = v.get("value", None)
        conf = v.get("confidence", None)
        reason = v.get("reason", "") or v.get("rationale", "") or ""

        # value 보정
        if isinstance(value, str):
            value = value.strip().lower() in ("true", "1", "yes", "y")
        if value is None:
            # value가 없으면 confidence로 추정
            try:
                cc = float(conf) if conf is not None else 0.0
            except Exception:
                cc = 0.0
            value = cc >= 0.55

        # confidence 보정
        try:
            conf = float(conf) if conf is not None else (0.6 if value else 0.0)
        except Exception:
            conf = 0.6 if value else 0.0

        return {
            "value": bool(value),
            "confidence": max(0.0, min(1.0, conf)),
            "reason": str(reason)[:160],
        }

    # default
    return {"value": False, "confidence": 0.0, "reason": ""}


def observe(img_url: str, mode: str = "normal", anchor_urls: Optional[List[str]] = None, dbg_id: Optional[str] = None) -> Dict[str, Any]:
    if mode == "strict":
        prompt = PROMPT_STRICT
    elif mode == "anchor_compare":
        prompt = PROMPT_ANCHOR
    else:
        prompt = PROMPT_NORMAL

    content_blocks = [{"type": "text", "text": prompt}]
    content_blocks.append({"type": "image_url", "image_url": {"url": img_url}})

    if anchor_urls:
        for u in anchor_urls[:2]:
            content_blocks.append({"type": "image_url", "image_url": {"url": u}})

    raw_text = _post_chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content_blocks},
    ])

    parsed = None
    try:
        parsed = _safe_json_extract(raw_text)
    except Exception:
        parsed = {"_parse_error": True}

    # 정규화
    out = {}
    if isinstance(parsed, dict):
        for k in KEYS:
            out[k] = _normalize_item(parsed.get(k, None))
    else:
        for k in KEYS:
            out[k] = {"value": False, "confidence": 0.0, "reason": ""}

    # DEBUG
    if DEBUG and dbg_id:
        dump = {
            "id": dbg_id,
            "mode": mode,
            "img_url": img_url,
            "anchor_urls": (anchor_urls[:2] if anchor_urls else []),
            "raw_text": raw_text[:4000],
            "parsed": parsed,
            "normalized": out,
        }
        with open(os.path.join(DEBUG_DIR, f"{dbg_id}_{mode}.json"), "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False, indent=2)

    return out

def is_template_zero(obs: Dict[str, Any]) -> bool:
    return all((not obs[k]["value"]) and (obs[k]["confidence"] == 0.0) and (obs[k]["reason"] == "") for k in KEYS)



# =========================
# Decide: 하드룰 + 조합룰 + 스코어
# =========================
@dataclass
class Decision:
    label: int
    score: float
    uncertain: bool
    why: str

def decide(obs: Dict[str, Any]) -> Decision:
    # 1) Hard abnormal
    for k in HARD_ABNORMAL_KEYS:
        if obs[k]["value"] and obs[k]["confidence"] >= 0.55:
            return Decision(LABEL_ABNORMAL, 999.0, False, f"HARD:{k}")

    # 2) Combo abnormal
    if combo_abnormal(obs):
        tilt_c = obs["device_tilt_or_rotation"]["confidence"]
        m_c = obs["misalignment_severe"]["confidence"]
        p_c = obs["package_damage"]["confidence"]
        if tilt_c >= 0.5 and max(m_c, p_c) >= 0.5:
            return Decision(LABEL_ABNORMAL, 50.0, False, "COMBO:tilt+(misalign|damage)")

    # 3) Weighted score (value * confidence * weight)
    score = 0.0
    for k in KEYS:
        if obs[k]["value"]:
            score += WEIGHTS.get(k, 1.0) * obs[k]["confidence"]


    # 임계치(튜닝 포인트)
    # - dev에서 너무 보수면 1.8~2.2 사이로 조절
    TH = 2.2
    label = LABEL_ABNORMAL if score >= TH else LABEL_NORMAL

    # 불확실성: 점수 애매 / 핵심 항목 confidence 낮음
    uncertain = (abs(score - TH) <= 0.6) or any(
        obs[k]["value"] and obs[k]["confidence"] < 0.55 for k in HARD_ABNORMAL_KEYS
    )
    why = f"SCORE:{score:.2f} TH:{TH}"
    return Decision(label, score, uncertain, why)

def decide_with_struct(obs: Dict[str, Any], st: Dict[str, Any]) -> Decision:
    # 리드가 3개가 아닌 경우 하드 불량
    if (st["visible_lead_count"] <= 2) and (not st["occluded"]):
        return Decision(LABEL_ABNORMAL, 999.0, False, "HARD:visible_lead_count<=2")

    # 회전/기울기(>=2) + 심한 휨이면 하드 불량
    if st["rotation_severity"] >= 2 and st["severe_bend"]:
        return Decision(LABEL_ABNORMAL, 999.0, False, "HARD:rotation>=2+severe_bend")

    # 너무 심한 회전(3)은 단독으로도 불량
    if st["rotation_severity"] >= 3 and (not st["occluded"]):
        return Decision(LABEL_ABNORMAL, 999.0, False, "HARD:rotation>=3")

    # 그 외는 기존 decide 사용
    return decide(obs)


# =========================
# Review Policy 재검토
# =========================
def need_review(dec: Decision, obs: Dict[str, Any]) -> bool:
    # 1) decide에서 uncertain
    if dec.uncertain:
        return True
    # 2) 핵심 결함이 value=true인데 confidence가 낮음
    for k in HARD_ABNORMAL_KEYS:
        if obs[k]["value"] and obs[k]["confidence"] < 0.65:
            return True
    # 3) tilt/misalignment 같이 경계 케이스
    if obs["device_tilt_or_rotation"]["value"] and obs["device_tilt_or_rotation"]["confidence"] < 0.6:
        return True
    return False

# =========================
# Classify Agent: normal -> strict -> anchor_compare
# =========================
def classify_agent(img_url: str, anchor_urls: List[str], dbg_id: str, max_retries=3) -> Tuple[int, Dict[str, Any], Decision]:
    for attempt in range(max_retries):
        try:
            st = observe_struct(img_url, dbg_id=dbg_id)
            if DEBUG:
                print(f"[DBG] {dbg_id} struct={st}")

            obs1 = observe(img_url, mode="normal", dbg_id=dbg_id)
            

            if is_template_zero(obs1):
                obs1b = None
            try:
                obs1b = observe(img_url, mode="normal", dbg_id=dbg_id)
            except Exception:
                obs1b = None

            if obs1b is not None and (not is_template_zero(obs1b)):
                obs1 = obs1b

            dec1 = decide_with_struct(obs1, st)


            if not need_review(dec1, obs1):
                return dec1.label, obs1, dec1

            obs2 = observe(img_url, mode="strict", dbg_id=dbg_id)
            dec2 = decide_with_struct(obs2, st)

            if dec2.label == LABEL_ABNORMAL and (dec2.why.startswith("HARD") or dec2.why.startswith("COMBO")):
                return LABEL_ABNORMAL, obs2, dec2

            obs3 = observe(img_url, mode="anchor_compare", anchor_urls=anchor_urls[:2], dbg_id=dbg_id)
            dec3 = decide_with_struct(obs3, st)

            candidates = [(dec1, obs1), (dec2, obs2), (dec3, obs3)]
            for d, o in candidates:
                if d.why.startswith("HARD") or d.why.startswith("COMBO"):
                    return LABEL_ABNORMAL, o, d

            best_d, best_o = max(candidates, key=lambda x: x[0].score)
            return best_d.label, best_o, best_d

        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep((0.5 * (2 ** attempt)) + random.uniform(0, 0.2))


# =========================
# Validator: 전체 분포 sanity check 후 재검토
# =========================
def validate_and_maybe_rerun(rows: List[Dict[str, Any]], anchor_urls: List[str]) -> List[Dict[str, Any]]:
    labels = [r["label"] for r in rows]
    n = len(labels)
    abnormal_rate = sum(labels) / max(1, n)

    # 극단 분포면, uncertain했던 샘플만 앵커 재검토
    extreme = (abnormal_rate < 0.05) or (abnormal_rate > 0.95) or (sum(labels) in (0, n))
    if not extreme:
        return rows

    new_rows = []
    for r in rows:
        if r.get("uncertain", False):
            # 앵커 비교만 재수행
            obs = observe(r["img_url"], mode="anchor_compare", anchor_urls=anchor_urls[:2])
            dec = decide(obs)
            r["label"] = dec.label
            r["why"] = f"VALIDATE_RERUN->{dec.why}"
            r["uncertain"] = dec.uncertain
        new_rows.append(r)
    return new_rows

# =========================
# main
# =========================
def main():
    df = pd.read_csv(TEST_CSV_PATH)
    if "id" not in df.columns or "img_url" not in df.columns:
        raise ValueError(f"columns: {df.columns.tolist()}")

    # 정상 앵커 URL 수집:
    ANCHOR_IDS = {"DEV_000", "DEV_002", "DEV_003", "DEV_004", "DEV_006", "DEV_010", "DEV_011", "DEV_012", "DEV_013", "DEV_015", "DEV_018", "DEV_019"}
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
            label, obs, dec = classify_agent(img_url, anchor_urls=anchor_urls, dbg_id=_id)
            results.append({
                "id": _id,
                "img_url": img_url,
                "label": label,
                "why": dec.why,
                "score": dec.score,
                "uncertain": dec.uncertain,
            })
            print(f"[{i+1}/{n}] id={_id} -> {label} | {dec.why}")

        except Exception as e:
            print(f"[{i+1}/{n}] id={_id} ERROR -> fallback 0 | {e}")
            results.append({"id": _id, "img_url": img_url, "label": LABEL_NORMAL, "why": "ERROR_FALLBACK", "score": -1, "uncertain": True})

        time.sleep(0.2)

    # 전체 분포 검증 후 uncertain만 추가 재검토
    results = validate_and_maybe_rerun(results, anchor_urls=anchor_urls)

    # 제출 파일
    out_df = pd.DataFrame([{"id": r["id"], "label": r["label"]} for r in results], columns=["id", "label"])
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\n✅ Saved: {OUT_PATH}")
    print(out_df.head())

if __name__ == "__main__":
    main()