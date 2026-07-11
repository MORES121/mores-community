#!/usr/bin/env python3
"""
bash example:
```
python3 evaluator.py ./result.jsonl ./verify.csv ./outputs
```

Evaluate ASR predictions against verify.csv reference.

Inputs:
- pred_jsonl: prediction file with audio_path and pred_text (default: result.jsonl)
- verify_csv: reference file with audio_path and ref_text (default: verify.csv)

Rules:
1. audio_path must strictly match one-to-one between prediction and reference
2. Convert pred_text from traditional to simplified Chinese before scoring
3. Compare ref_text and pred_text after removing all punctuation
4. A sample passes if edit distance between ref_text and pred_text is <= 2 characters

Outputs:
- metrics.json
- top_confusions.csv
- error_examples.json
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Tuple


PUNCT_REGEX = re.compile(
    r"[，。！？；：、“”‘’（）()【】《》〈〉—…·,\.!\?;:\"'`\[\]\{\}\s]"
)

CHAR_TOLERANCE = 2

_OPENCC = None


def get_opencc():
    global _OPENCC
    if _OPENCC is None:
        try:
            from opencc import OpenCC
        except ImportError as exc:
            raise ImportError(
                "opencc is required for traditional-to-simplified conversion. "
                "Install with: pip install opencc-python-reimplemented"
            ) from exc
        _OPENCC = OpenCC("t2s")
    return _OPENCC


def to_simplified(text: str) -> str:
    if not text:
        return text
    return get_opencc().convert(text)


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_pred_jsonl = os.path.join(script_dir, "result.jsonl")
    default_verify_csv = os.path.join(script_dir, "verify.csv")
    default_report_dir = os.path.join(script_dir, "outputs")

    parser = argparse.ArgumentParser(
        description="Evaluate ASR predictions against verify.csv.",
        epilog="Example: python3 evaluator.py ./result.jsonl ./verify.csv ./outputs",
    )
    parser.add_argument(
        "pred_jsonl",
        nargs="?",
        default=default_pred_jsonl,
        help=f"Prediction jsonl path. Default: {default_pred_jsonl}",
    )
    parser.add_argument(
        "verify_csv",
        nargs="?",
        default=default_verify_csv,
        help=f"Reference csv path. Default: {default_verify_csv}",
    )
    parser.add_argument(
        "report_dir",
        nargs="?",
        default=default_report_dir,
        help=f"Report directory. Default: {default_report_dir}",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = re.sub(r"^\[(RAW|NORM)\]\s*", "", text or "")
    text = re.sub(PUNCT_REGEX, "", text)
    return text.strip()


def load_verify_csv(path: str) -> List[Tuple[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty csv: {path}")

        audio_col = None
        ref_col = None
        for name in reader.fieldnames:
            lower = name.strip().lower()
            if lower in ("audio_path", "audio:file"):
                audio_col = name
            elif lower in ("ref_text", "text:label"):
                ref_col = name

        if not audio_col or not ref_col:
            raise ValueError(
                f"verify.csv must contain audio_path and ref_text columns, got: {reader.fieldnames}"
            )

        for line_no, row in enumerate(reader, start=2):
            audio = (row.get(audio_col) or "").strip()
            ref_text = row.get(ref_col) or ""
            if not audio:
                raise ValueError(f"Missing audio_path at {path}:{line_no}")
            rows.append((audio, ref_text))

    if not rows:
        raise ValueError(f"No reference rows found in {path}")
    return rows


def load_pred_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid json at {path}:{line_no}: {exc}") from exc
    return rows


def align_predictions(
    ref_rows: List[Tuple[str, str]],
    pred_rows: List[Dict],
) -> Tuple[List[str], List[str], List[str]]:
    ref_map = {audio: ref for audio, ref in ref_rows}
    ref_audios = [audio for audio, _ in ref_rows]

    if len(pred_rows) != len(ref_rows):
        raise ValueError(
            f"Row count mismatch: predictions={len(pred_rows)}, references={len(ref_rows)}"
        )

    pred_by_audio: Dict[str, str] = {}
    pred_order: List[str] = []
    for idx, row in enumerate(pred_rows, start=1):
        audio = row.get("audio_path")
        if audio is None:
            audio = row.get("audio") or row.get("utt_id")
        if not audio:
            raise ValueError(f"Missing audio_path in prediction row {idx}")

        audio = str(audio).strip()
        if audio in pred_by_audio:
            raise ValueError(f"Duplicate audio_path in predictions: {audio}")

        pred_text = row.get("pred_text")
        if pred_text is None:
            raise ValueError(f"Missing pred_text for audio_path: {audio}")

        pred_by_audio[audio] = to_simplified(str(pred_text))
        pred_order.append(audio)

    missing_in_pred = [audio for audio in ref_audios if audio not in pred_by_audio]
    extra_in_pred = [audio for audio in pred_order if audio not in ref_map]
    if missing_in_pred:
        preview = ", ".join(missing_in_pred[:5])
        raise ValueError(
            f"Predictions missing {len(missing_in_pred)} reference audio_path(s): {preview}"
        )
    if extra_in_pred:
        preview = ", ".join(extra_in_pred[:5])
        raise ValueError(
            f"Predictions contain {len(extra_in_pred)} unknown audio_path(s): {preview}"
        )

    if pred_order != ref_audios:
        first_mismatch = next(
            (i for i, (pred_audio, ref_audio) in enumerate(zip(pred_order, ref_audios))
             if pred_audio != ref_audio),
            None,
        )
        if first_mismatch is not None:
            raise ValueError(
                "audio_path order mismatch at "
                f"line {first_mismatch + 1}: "
                f"pred={pred_order[first_mismatch]!r}, ref={ref_audios[first_mismatch]!r}"
            )

    audios = ref_audios
    refs = [normalize_text(ref_map[audio]) for audio in audios]
    preds = [normalize_text(pred_by_audio[audio]) for audio in audios]
    return audios, refs, preds


def levenshtein_ops(ref: List[str], hyp: List[str]) -> Tuple[int, List[Tuple[str, str]]]:
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    bt = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
        bt[i][0] = "D"
    for j in range(1, n + 1):
        dp[0][j] = j
        bt[0][j] = "I"

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost_sub = dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1)
            cost_del = dp[i - 1][j] + 1
            cost_ins = dp[i][j - 1] + 1
            best = min(cost_sub, cost_del, cost_ins)
            dp[i][j] = best
            if best == cost_sub:
                bt[i][j] = "M" if ref[i - 1] == hyp[j - 1] else "S"
            elif best == cost_del:
                bt[i][j] = "D"
            else:
                bt[i][j] = "I"

    i, j = m, n
    pairs = []
    while i > 0 or j > 0:
        op = bt[i][j]
        if op in ("M", "S"):
            pairs.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif op == "D":
            pairs.append((ref[i - 1], "<del>"))
            i -= 1
        elif op == "I":
            pairs.append(("<ins>", hyp[j - 1]))
            j -= 1
        else:
            break
    pairs.reverse()
    return dp[m][n], pairs


def edit_distance(ref: str, pred: str) -> int:
    dist, _ = levenshtein_ops(list(ref), list(pred))
    return dist


def is_match(ref: str, pred: str, tolerance: int = CHAR_TOLERANCE) -> bool:
    return edit_distance(ref, pred) <= tolerance


def char_error_rate(preds: List[str], refs: List[str]) -> float:
    total_dist = 0
    total_chars = 0
    for pred, ref in zip(preds, refs):
        dist, _ = levenshtein_ops(list(ref), list(pred))
        total_dist += dist
        total_chars += max(1, len(ref))
    return total_dist / total_chars if total_chars else 0.0


def sentence_accuracy(preds: List[str], refs: List[str], tolerance: int = CHAR_TOLERANCE) -> float:
    if not refs:
        return 0.0
    return sum(int(is_match(r, p, tolerance)) for p, r in zip(preds, refs)) / len(refs)


def main():
    args = parse_args()
    os.makedirs(args.report_dir, exist_ok=True)

    ref_rows = load_verify_csv(args.verify_csv)
    pred_rows = load_pred_jsonl(args.pred_jsonl)
    if not pred_rows:
        raise ValueError("Empty prediction jsonl")

    audios, refs, preds = align_predictions(ref_rows, pred_rows)

    cer = char_error_rate(preds, refs)
    sent_acc = sentence_accuracy(preds, refs)

    sub_counter = Counter()
    ins_counter = Counter()
    del_counter = Counter()
    example_errors = []
    for audio, pred, ref in zip(audios, preds, refs):
        _, ops = levenshtein_ops(list(ref), list(pred))
        local_has_error = False
        for a, b in ops:
            if a == "<ins>":
                ins_counter[b] += 1
                local_has_error = True
            elif b == "<del>":
                del_counter[a] += 1
                local_has_error = True
            elif a != b:
                sub_counter[(a, b)] += 1
                local_has_error = True
        if not is_match(ref, pred):
            example_errors.append({
                "audio_path": audio,
                "reference": ref,
                "prediction": pred,
                "edit_distance": edit_distance(ref, pred),
            })

    metrics = {
        "num_samples": len(refs),
        "cer": cer,
        "char_accuracy_approx": 1.0 - cer,
        "sentence_accuracy": sent_acc,
        "sentence_accuracy_percent": f"{round(sent_acc * 100, 1):g}%",
        "char_tolerance": CHAR_TOLERANCE,
        "pred_jsonl": os.path.abspath(args.pred_jsonl),
        "verify_csv": os.path.abspath(args.verify_csv),
        "ignore_punctuation": True,
        "pred_text_to_simplified": True,
    }

    with open(os.path.join(args.report_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.report_dir, "top_confusions.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "reference", "prediction", "count"])
        for (a, b), c in sub_counter.most_common(50):
            writer.writerow(["substitution", a, b, c])
        for a, c in del_counter.most_common(30):
            writer.writerow(["deletion", a, "<del>", c])
        for b, c in ins_counter.most_common(30):
            writer.writerow(["insertion", "<ins>", b, c])

    with open(os.path.join(args.report_dir, "error_examples.json"), "w", encoding="utf-8") as f:
        json.dump(example_errors, f, ensure_ascii=False, indent=2)

    score = sent_acc * 100.0
    with open(os.path.join(args.report_dir, "score.txt"), "w", encoding="utf-8") as f:
        f.write(f"{score:g}\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved -> {args.report_dir}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
