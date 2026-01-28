import os
import json
import time
import random
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from src import config, utils

# =========================
# Config
# =========================
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

TH_SCORE = 2.2
HARD_BOOST = 2.5
STRUCT_BOOST = 1.5

# =========================
# Debug dir
# =========================
if DEBUG:
    SESSION_TIMESTAMP = time.strftime("%y%m%d_%H%M")
    CURRENT_DEBUG_DIR = os.path.join(DEBUG_DIR, SESSION_TIMESTAMP)
    os.makedirs(CURRENT_DEBUG_DIR, exist_ok=True)
else:
    CURRENT_DEBUG_DIR = None

# =========================
# Utils
# =========================
def _normalize_item(v) -> Dict[str, Any]:
    if isinstance(v, bool):
        return {"value": v, "confidence": 0.7 if v else 0.1, "reason": ""}

    if isinstance(v, (int, float)):
        c = float(v)
        return {
            "value": c >= 0.5,
            "confidence": max(0.0, min(1.0, c)),
            "reason": ""
        }

    if isinstance(v, dict):
        value = v.get("value")
        conf = v.get("confidence")
        reason = v.get("reason", "") or v.get("rationale", "")

        if isinstance(value, str):
            value = value.lower() in ("true", "1", "yes")

        if conf is None:
            conf = 0.65 if value else 0.15

        try:
            conf = float(conf)
        except Exception:
            conf = 0.65 if value else 0.15

        if value is None:
            value = conf >= 0.55

        return {
            "value": bool(value),
            "confidence": max(0.0, min(1.0, conf)),
            "reason": reason[:160]
        }

    return {"value": False, "confidence": 0.0, "reason": ""}


def is_template_zero(obs: Dict[str, Any]) -> bool:
    return all(obs[k]["confidence"] < 0.15 for k in KEYS)

# =========================
# Observe
# =========================
def observe_struct(img_url: str, dbg_id=None) -> Dict[str, Any]:
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
        with open(os.path.join(CURRENT_DEBUG_DIR, f"{dbg_id}_struct.json"), "w") as f:
            json.dump(out, f, indent=2)

    return out


def observe(img_input, mode="normal", anchor_urls=None, dbg_id=None) -> Dict[str, Any]:
    prompt = {
        "normal": PROMPT_NORMAL,
        "strict": PROMPT_STRICT,
        "anchor_compare": PROMPT_ANCHOR
    }.get(mode, PROMPT_NORMAL)

    content = [{"type": "text", "text": prompt}]

    if isinstance(img_input, list):
        for u in img_input:
            content.append({"type": "image_url", "image_url": {"url": u}})
    else:
        content.append({"type": "image_url", "image_url": {"url": img_input}})

    if anchor_urls:
        for u in anchor_urls[:2]:
            content.append({"type": "image_url", "image_url": {"url": u}})

    raw = utils.post_chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ])

    parsed = utils.safe_json_extract(raw)
    obs = {k: _normalize_item(parsed.get(k)) for k in KEYS}

    if DEBUG and dbg_id and CURRENT_DEBUG_DIR:
        with open(os.path.join(CURRENT_DEBUG_DIR, f"{dbg_id}_{mode}.json"), "w") as f:
            json.dump(obs, f, indent=2)

    return obs

# =========================
# Decision
# =========================
@dataclass
class Decision:
    label: int
    score: float
    uncertain: bool
    why: str


def compute_score(obs: Dict[str, Any]) -> Tuple[float, List[str]]:
    score = 0.0
    reasons = []

    for k in KEYS:
        if obs[k]["value"]:
            s = WEIGHTS.get(k, 1.0) * obs[k]["confidence"]
            score += s
            reasons.append(f"{k}:{s:.2f}")

    for k in HARD_ABNORMAL_KEYS:
        if obs[k]["value"]:
            boost = HARD_BOOST * obs[k]["confidence"]
            score += boost
            reasons.append(f"HARD_BOOST:{k}:{boost:.2f}")

    return score, reasons


def apply_struct_score(score: float, st: Dict[str, Any], reasons: List[str]) -> float:
    if st["visible_lead_count"] <= 2 and not st["occluded"]:
        score += STRUCT_BOOST
        reasons.append("STRUCT:lead<=2")

    if st["rotation_severity"] >= 2:
        score += 0.8 * st["rotation_severity"]
        reasons.append(f"STRUCT:rotation={st['rotation_severity']}")

    if st["severe_bend"]:
        score += 1.0
        reasons.append("STRUCT:severe_bend")

    return score


def decide(obs: Dict[str, Any], st: Dict[str, Any]) -> Decision:
    score, reasons = compute_score(obs)
    score = apply_struct_score(score, st, reasons)

    label = LABEL_ABNORMAL if score >= TH_SCORE else LABEL_NORMAL
    uncertain = abs(score - TH_SCORE) < 0.5

    return Decision(
        label=label,
        score=score,
        uncertain=uncertain,
        why=" | ".join(reasons)
    )

# =========================
# Agent
# =========================
def classify_agent(img_input, anchor_urls, dbg_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            st = observe_struct(img_input[0], dbg_id)

            obs1 = observe(img_input, "normal", dbg_id=dbg_id)
            if is_template_zero(obs1):
                obs1 = observe(img_input, "strict", dbg_id=dbg_id)

            dec1 = decide(obs1, st)
            if not dec1.uncertain:
                return dec1.label, obs1, dec1

            obs2 = observe(img_input[0], "anchor_compare", anchor_urls, dbg_id)
            dec2 = decide(obs2, st)

            return (
                dec1.label if dec1.score >= dec2.score else dec2.label,
                obs1 if dec1.score >= dec2.score else obs2,
                dec1 if dec1.score >= dec2.score else dec2
            )

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.5 * (2 ** attempt))

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
