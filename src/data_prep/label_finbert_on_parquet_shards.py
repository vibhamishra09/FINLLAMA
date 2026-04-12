from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"
INPUT_SHARDS_DIR = Path("./financial_news_shards")
OUTPUT_SHARDS_DIR = Path("./financial_news_shards_labeled")

# Keep batch sizes moderate; this pipeline uses encoder-only models and truncation to 512.
BATCH_SIZE = 32
MAX_LENGTH = 512


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _infer_sentiment_and_confidence(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
    id2label: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run FinBERT in batches and return:
      sentiment_codes: int array mapping negative=0, neutral=1, positive=2
      confidences: float array = max softmax probability per example
    """
    # FinBERT predicts 3 classes; we map them into codes:
    # negative -> 0, neutral -> 1, positive -> 2
    label_to_code = {"negative": 0, "neutral": 1, "positive": 2}

    sentiment_codes: list[int] = []
    confidences: list[float] = []

    n = len(texts)
    total_batches = (n + batch_size - 1) // batch_size if n else 0
    # Keep long CPU runs from looking "hung" in terminals/logs.
    log_every = max(1, min(50, max(1, total_batches // 20)))

    batch_times = []
    for batch_idx, start in enumerate(range(0, n, batch_size), start=1):
        batch_texts = texts[start : start + batch_size]

        if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches:
            processed = min(start + len(batch_texts), n)
            print(
                f"    Inference progress: batch {batch_idx}/{total_batches} "
                f"({processed:,}/{n:,} rows)",
                flush=True,
            )

        batch_start_time = time.time()
        inputs = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # [batch, 3]
            probs = F.softmax(logits, dim=-1)

            pred_indices = torch.argmax(probs, dim=-1).detach().cpu().numpy()
            batch_conf = torch.max(probs, dim=-1).values.detach().cpu().numpy().astype(float)

        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        print(f"      Batch {batch_idx}/{total_batches} took {batch_time:.2f} sec", flush=True)

        # Convert predictions to required numeric encoding.
        for idx in pred_indices.tolist():
            if id2label:
                # id2label can be int->str or str->str depending on HF serialization.
                label = id2label.get(int(idx), None) or id2label.get(str(int(idx)), None)
                if label is None:
                    code = int(idx)
                else:
                    code = label_to_code.get(str(label).lower(), int(idx))
            else:
                # Fallback: assume HF label index order equals negative/neutral/positive.
                code = int(idx)

            # Ensure code is one of (0,1,2).
            if code not in (0, 1, 2):
                code = 1

            sentiment_codes.append(int(code))

        confidences.extend([float(x) for x in batch_conf.tolist()])

    print(f"    Mean batch time: {np.mean(batch_times):.2f} sec, Median: {np.median(batch_times):.2f} sec", flush=True)
    return (
        np.asarray(sentiment_codes, dtype=np.int64),
        np.asarray(confidences, dtype=np.float32),
    )


def _extract_shard_number(filename: str) -> int:
    """
    Extract the numeric shard id from strings like shard_0001.parquet.
    Used only for logging / sorting.
    """
    m = re.search(r"shard_(\d+)\.parquet$", filename)
    return int(m.group(1)) if m else -1


def label_shards_with_finbert(
    input_shards_dir: Path = INPUT_SHARDS_DIR,
    output_shards_dir: Path = OUTPUT_SHARDS_DIR,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    finbert_model_name: str = FINBERT_MODEL_NAME,
    shard_number: int | None = None,
    max_rows: int | None = None,
    shard_range_start: int | None = None,
    shard_range_end: int | None = None,
) -> None:
    output_shards_dir.mkdir(parents=True, exist_ok=True)

    if not input_shards_dir.exists():
        raise FileNotFoundError(f"Input shards directory not found: {input_shards_dir.resolve()}")

    parquet_files = sorted(input_shards_dir.glob("shard_*.parquet"), key=lambda p: _extract_shard_number(p.name))
    if not parquet_files:
        raise FileNotFoundError(f"No shard_*.parquet files found in {input_shards_dir.resolve()}")

    # Filter by shard_number or range if specified
    if shard_number is not None:
        parquet_files = [p for p in parquet_files if _extract_shard_number(p.name) == shard_number]
        if not parquet_files:
            raise FileNotFoundError(
                f"No shard_XXXX.parquet found for shard_number={shard_number} in {input_shards_dir.resolve()}"
            )
    elif shard_range_start is not None or shard_range_end is not None:
        start = shard_range_start if shard_range_start is not None else float('-inf')
        end = shard_range_end if shard_range_end is not None else float('inf')
        parquet_files = [p for p in parquet_files if start <= _extract_shard_number(p.name) <= end]
        if not parquet_files:
            raise FileNotFoundError(
                f"No shard_XXXX.parquet files found in range {start} to {end} in {input_shards_dir.resolve()}"
            )

    device = _get_device()
    print(f"FinBERT model: {finbert_model_name}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Input shards: {len(parquet_files)} file(s)", flush=True)
    print(f"Output shards dir: {output_shards_dir.resolve()}", flush=True)
    print(flush=True)

    print(f"Loading tokenizer + model from {finbert_model_name}...", flush=True)
    # In some environments, AutoTokenizer may fail when trying to instantiate a backend
    # "fast" tokenizer. Fall back to the known compatible BERT tokenizer class.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            finbert_model_name,
            use_fast=False,
            local_files_only=False,
        )
    except Exception as e:
        print(
            f"  [WARN] AutoTokenizer failed ({e.__class__.__name__}: {e}). Falling back to BertTokenizer...",
            flush=True,
        )
        from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained(
            finbert_model_name,
            local_files_only=False,
        )
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            finbert_model_name,
            local_files_only=False,
            use_safetensors=False,
        )
    except Exception as e:
        print(
            f"  [WARN] AutoModelForSequenceClassification failed ({e.__class__.__name__}: {e}). Falling back to BertForSequenceClassification...",
            flush=True,
        )
        from transformers import BertForSequenceClassification

        model = BertForSequenceClassification.from_pretrained(
            finbert_model_name,
            local_files_only=False,
            use_safetensors=False,
        )
    model.to(device)
    model.eval()

    id2label = getattr(model.config, "id2label", None)
    if id2label:
        # Normalize to int->str where possible.
        normalized_id2label = {}
        for k, v in id2label.items():
            try:
                normalized_id2label[int(k)] = str(v)
            except Exception:
                normalized_id2label[int(str(k))] = str(v)
        id2label = normalized_id2label
    else:
        id2label = {}

    for shard_idx, shard_path in enumerate(parquet_files, start=1):
        shard_start_time = time.time()
        shard_basename = shard_path.name  # e.g., shard_0001.parquet
        shard_number = _extract_shard_number(shard_basename)
        print(
            f"[{shard_idx}/{len(parquet_files)}] Processing {shard_basename} (shard_number={shard_number})",
            flush=True,
        )

        df = pd.read_parquet(str(shard_path))
        if "text" not in df.columns:
            raise KeyError(f"Missing required 'text' column in {shard_path.resolve()}. Found columns: {list(df.columns)}")

        if max_rows is not None:
            df = df.iloc[:max_rows].copy()

        # Keep text as string; NaNs become empty strings.
        texts = df["text"].fillna("").astype(str).tolist()

        n_records = len(texts)
        print(f"  Records: {n_records:,}", flush=True)

        sentiment, confidence = _infer_sentiment_and_confidence(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            id2label=id2label,
        )

        if len(sentiment) != n_records or len(confidence) != n_records:
            raise RuntimeError(
                f"Inconsistent output sizes for {shard_basename}: "
                f"sentiment={len(sentiment)}, confidence={len(confidence)}, expected={n_records}"
            )

        df_out = df.copy()
        df_out["sentiment"] = sentiment
        df_out["confidence"] = confidence

        out_path = output_shards_dir / shard_basename
        df_out.to_parquet(str(out_path), index=False)
        print(f"  Saved labeled shard to: {out_path.resolve()}", flush=True)
        shard_time = time.time() - shard_start_time
        print(f"  Shard {shard_basename} processed in {shard_time:.2f} sec ({shard_time/60:.2f} min)", flush=True)
        print(flush=True)


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Label FinBERT sentiment into Parquet shards.")
    parser.add_argument("--input-shards-dir", type=str, default=str(INPUT_SHARDS_DIR))
    parser.add_argument("--output-shards-dir", type=str, default=str(OUTPUT_SHARDS_DIR))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--finbert-model-name", type=str, default=FINBERT_MODEL_NAME)
    parser.add_argument("--shard-number", type=int, default=None, help="Process only shard_XXXX.parquet with this numeric XXXX.")
    parser.add_argument("--shard-range-start", type=int, default=None, help="Start of shard number range (inclusive).")
    parser.add_argument("--shard-range-end", type=int, default=None, help="End of shard number range (inclusive).")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows per shard (e.g., first 5000 rows).")

    args = parser.parse_args()
    label_shards_with_finbert(
        input_shards_dir=Path(args.input_shards_dir),
        output_shards_dir=Path(args.output_shards_dir),
        batch_size=args.batch_size,
        max_length=args.max_length,
        finbert_model_name=args.finbert_model_name,
        shard_number=args.shard_number,
        max_rows=args.max_rows,
        shard_range_start=args.shard_range_start,
        shard_range_end=args.shard_range_end,
    )
    total_time = time.time() - start_time
    print(f"\nTotal script runtime: {total_time:.2f} seconds ({total_time/60:.2f} min)", flush=True)

