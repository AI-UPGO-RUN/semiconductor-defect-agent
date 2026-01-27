import os
import json
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from src import config, utils
import time
import random

PROMPT_NORMAL = utils.build_prompt("normal")
PROMPT_STRICT = utils.build_prompt("strict")
PROMPT_ANCHOR = utils.build_prompt("anchor_compare")
SYSTEM = config.SYSTEM
STRUCT_PROMPT = config.STRUCT_PROMPT
KEYS = config.KEYS
HARD_ABNORMAL_KEYS = config.HARD_ABNORMAL_KEYS
LABEL_ABNORMAL = config.LABEL_ABNORMAL
LABEL_NORMAL = config.LABEL_NORMAL
WEIGHTS = config.WEIGHTS
DEBUG_DIR = config.DEV_DEBUG_DIR
DEBUG = config.DEV_DEBUG

if DEBUG:
    # 예: debug/20231025_143000/ 처럼 한 폴더에 다 모임
    SESSION_TIMESTAMP = time.strftime("%y%m%d_%H%M")
    CURRENT_DEBUG_DIR = os.path.join(DEBUG_DIR, SESSION_TIMESTAMP)
    os.makedirs(CURRENT_DEBUG_DIR, exist_ok=True)
    print(f"[DEBUG] Logs will be saved to: {CURRENT_DEBUG_DIR}")
else:
    CURRENT_DEBUG_DIR = None

def observe_struct(img_url: str, dbg_id: Optional[str] = None) -> Dict[str, Any]:
    raw = utils.post_chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": STRUCT_PROMPT},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]},
    ])
    parsed = utils.safe_json_extract(raw)
    out = {
        "visible_lead_count": int(parsed.get("visible_lead_count", 3)),
        "rotation_severity": int(parsed.get("rotation_severity", 0)),
        "severe_bend": bool(parsed.get("severe_bend", False)),
        "occluded": bool(parsed.get("occluded", False)),
    }
    if DEBUG and dbg_id and CURRENT_DEBUG_DIR:
        save_path = os.path.join(CURRENT_DEBUG_DIR, f"{dbg_id}_struct.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"id": dbg_id, "raw_text": raw[:4000], "parsed": parsed, "struct": out},
                      f, ensure_ascii=False, indent=2)
        return out

def observe(img_input: Any, mode: str = "normal", anchor_urls: Optional[List[str]] = None, dbg_id: Optional[str] = None) -> Dict[str, Any]:
    if mode == "strict":
        prompt = PROMPT_STRICT
    elif mode == "anchor_compare":
        prompt = PROMPT_ANCHOR
    else:
        prompt = PROMPT_NORMAL

    content_blocks = [{"type": "text", "text": prompt}]

    if isinstance(img_input, list):
        for url in img_input:
            content_blocks.append({"type": "image_url", "image_url": {"url": url}})
    else:
        content_blocks.append({"type": "image_url", "image_url": {"url": img_input}})
    if anchor_urls:
        for u in anchor_urls[:2]:
            content_blocks.append({"type": "image_url", "image_url": {"url": u}})

    raw_text = utils.post_chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content_blocks},
    ])

    parsed = None
    try:
        parsed = utils.safe_json_extract(raw_text)
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
    if DEBUG and dbg_id and CURRENT_DEBUG_DIR:
        dump = {
            "id": dbg_id,
            "mode": mode,
            # "img_url": img_input[0],  # base64가 너무 길면 생략 가능
            "anchor_urls": (anchor_urls[:2] if anchor_urls else []),
            "raw_text": raw_text[:4000],
            "parsed": parsed,
            "normalized": out,
        }
        # 모드별로 파일 저장
        save_path = os.path.join(CURRENT_DEBUG_DIR, f"{dbg_id}_{mode}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False, indent=2)

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

# def observe(img_url: str, mode: str = "normal", anchor_urls: Optional[List[str]] = None, dbg_id: Optional[str] = None) -> Dict[str, Any]:
#     if mode == "strict":
#         prompt = PROMPT_STRICT
#     elif mode == "anchor_compare":
#         prompt = PROMPT_ANCHOR
#     else:
#         prompt = PROMPT_NORMAL
#
#     content_blocks = [{"type": "text", "text": prompt}]
#     content_blocks.append({"type": "image_url", "image_url": {"url": img_url}})
#
#     if anchor_urls:
#         for u in anchor_urls[:2]:
#             content_blocks.append({"type": "image_url", "image_url": {"url": u}})
#
#     raw_text = utils.post_chat([
#         {"role": "system", "content": SYSTEM},
#         {"role": "user", "content": content_blocks},
#     ])
#
#     parsed = None
#     try:
#         parsed = utils.safe_json_extract(raw_text)
#     except Exception:
#         parsed = {"_parse_error": True}
#
#     # 정규화
#     out = {}
#     if isinstance(parsed, dict):
#         for k in KEYS:
#             out[k] = _normalize_item(parsed.get(k, None))
#     else:
#         for k in KEYS:
#             out[k] = {"value": False, "confidence": 0.0, "reason": ""}
#
#     # DEBUG
#     if DEBUG and dbg_id:
#         dump = {
#             "id": dbg_id,
#             "mode": mode,
#             "img_url": img_url,
#             "anchor_urls": (anchor_urls[:2] if anchor_urls else []),
#             "raw_text": raw_text[:4000],
#             "parsed": parsed,
#             "normalized": out,
#         }
#         with open(os.path.join(DEBUG_DIR, f"{dbg_id}_{mode}.json"), "w", encoding="utf-8") as f:
#             json.dump(dump, f, ensure_ascii=False, indent=2)
#
#     return out

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
    # if combo_abnormal(obs):
    #     tilt_c = obs["device_tilt_or_rotation"]["confidence"]
    #     m_c = obs["misalignment_severe"]["confidence"]
    #     p_c = obs["package_damage"]["confidence"]
    #     if tilt_c >= 0.5 and max(m_c, p_c) >= 0.5:
    #         return Decision(LABEL_ABNORMAL, 50.0, False, "COMBO:tilt+(misalign|damage)")

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
def classify_agent(img_input: Any, anchor_urls: List[str], dbg_id: str, max_retries=3) -> Tuple[
    int, Dict[str, Any], Decision]:
    for attempt in range(max_retries):
        try:
            # 구조적 관찰은 원본만
            st = observe_struct(img_input[0], dbg_id=dbg_id)
            if DEBUG:
                print(f"[DBG] {dbg_id} struct={st}")

            # 결함 관찰은 [원본 + 전처리]
            obs1 = observe(img_input, mode="normal", dbg_id=dbg_id)

            if is_template_zero(obs1):
                obs1b = None
            try:
                obs1b = observe(img_input, mode="normal", dbg_id=dbg_id)
            except Exception:
                obs1b = None

            if obs1b is not None and (not is_template_zero(obs1b)):
                obs1 = obs1b

            dec1 = decide_with_struct(obs1, st)

            if not need_review(dec1, obs1):
                return dec1.label, obs1, dec1

            obs2 = observe(img_input, mode="strict", dbg_id=dbg_id)
            dec2 = decide_with_struct(obs2, st)

            if dec2.label == LABEL_ABNORMAL and (dec2.why.startswith("HARD") or dec2.why.startswith("COMBO")):
                return LABEL_ABNORMAL, obs2, dec2

            obs3 = observe(img_input[0], mode="anchor_compare", anchor_urls=anchor_urls[:2], dbg_id=dbg_id)
            dec3 = decide_with_struct(obs3, st)

            candidates = [(dec1, obs1), (dec2, obs2), (dec3, obs3)]
            for d, o in candidates:
                if d.why.startswith("HARD") or d.why.startswith("COMBO"):
                    return LABEL_ABNORMAL, o, d

            best_d, best_o = max(candidates, key=lambda x: x[0].score)
            return best_d.label, best_o, best_d

        except Exception as e:
            print(f"[ERR] classify_agent dbg_id={dbg_id} attempt={attempt}")
            print(e)
            import traceback
            traceback.print_exc()
            if attempt == max_retries - 1:
                raise
            time.sleep((0.5 * (2 ** attempt)) + random.uniform(0, 0.2))


# =========================
# Validator: 전체 분포 sanity check 후 재검토
# 여기서는 url 가져오고 다른건 base64니까 검토가 필요함
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
