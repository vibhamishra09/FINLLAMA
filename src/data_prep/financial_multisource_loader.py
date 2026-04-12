import argparse
import hashlib
import os
from pathlib import Path

from datasets import load_dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass


def load_and_process_financial_news(
    shard_size: int = 50000,
    output_dir: str = "financial_news_shards",
    max_shards: int | None = None,
    skip_shards: int = 0,
):
    """
    Stream the multisource dataset and write Parquet shards.

    Sharding basis (see module docstring / README):
    - Records are taken in Hugging Face streaming order (per split, order from the iterator).
    - Only rows with non-empty text are kept; text is lowercased and stripped.
    - Global MD5 dedup on normalized text across the entire run (all splits).
    - Each shard is the next ``shard_size`` *unique accepted* rows in that order, except the
      last shard which may be smaller (or ``skip_shards`` / ``max_shards`` may truncate the run).

    ``skip_shards`` still consumes the stream from the beginning (dedup state preserved) but
    discards the first ``skip_shards`` full buffers without writing files, so you can resume
    "later" shard indices only by re-streaming from the start — not random access.
    """
    print("=" * 80)
    print("FINANCIAL NEWS MULTISOURCE DATASET LOADER")
    print("=" * 80)
    print()
    print("Loading financial news dataset from HuggingFace...")
    print()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("ERROR: No Hugging Face token found.")
        print("Set HF_TOKEN in .env at the repo root or export HUGGING_FACE_HUB_TOKEN.")
        return

    try:
        ds = load_dataset(
            "Brianferrell787/financial-news-multisource",
            streaming=True,
            token=hf_token,
        )
    except Exception as e:
        print("WARNING: AUTHENTICATION OR ACCESS ERROR")
        print(f"Error: {e}")
        print()
        print("This dataset is gated on HuggingFace and requires authentication.")
        print()
        print("SETUP INSTRUCTIONS:")
        print("-" * 80)
        print("1. Accept the dataset license:")
        print("   Visit: https://huggingface.co/datasets/Brianferrell787/financial-news-multisource")
        print("   Click 'Agree and access dataset'")
        print()
        print("2. Set HF_TOKEN (see .env at repo root) or: huggingface-cli login")
        print("-" * 80)
        return

    if skip_shards < 0 or (max_shards is not None and max_shards < 1):
        print("ERROR: skip_shards must be >= 0 and max_shards must be >= 1 when set.")
        return

    seen_hashes = set()
    processed_records = []
    skip_remaining = skip_shards
    shards_saved = 0
    next_shard_index = 0
    total_processed = 0

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out.resolve()}")
    print(f"Shard size (unique rows per shard): {shard_size:,}")
    if skip_shards:
        print(f"Skip first {skip_shards} full shard(s) (stream still read; dedup global).")
    if max_shards is not None:
        print(f"Stop after saving at most {max_shards} shard file(s).")
    print()
    print("Processing dataset: filtering, deduplicating, and normalizing...")

    stop_all = False

    def flush_full_buffer():
        nonlocal processed_records, skip_remaining, shards_saved, next_shard_index, stop_all
        if skip_remaining > 0:
            skip_remaining -= 1
            processed_records = []
            print(f"  Skipped full shard ({skip_shards - skip_remaining}/{skip_shards} skips used).")
            return
        if max_shards is not None and shards_saved >= max_shards:
            stop_all = True
            return
        save_shard(processed_records, next_shard_index, out)
        next_shard_index += 1
        shards_saved += 1
        processed_records = []
        if max_shards is not None and shards_saved >= max_shards:
            print(f"\nReached --max-shards={max_shards}; stopping.")
            stop_all = True

    for split_name, split_data in ds.items():
        if stop_all:
            break
        print(f"Processing split: {split_name}")

        for record in split_data:
            if stop_all:
                break

            text = record.get("text") or record.get("content") or record.get("body")

            if not text or (isinstance(text, str) and text.strip() == ""):
                continue

            date = record.get("date") or record.get("timestamp") or record.get("published_at") or "unknown"

            text_normalized = text.lower().strip()
            text_hash = hashlib.md5(text_normalized.encode()).hexdigest()

            if text_hash in seen_hashes:
                continue

            seen_hashes.add(text_hash)

            processed_records.append({"date": str(date), "text": text_normalized})

            total_processed += 1

            if total_processed % 10000 == 0:
                print(f"  Progress: {total_processed:,} unique records accepted...")

            if len(processed_records) >= shard_size:
                flush_full_buffer()

    if not stop_all and processed_records and skip_remaining == 0:
        if max_shards is None or shards_saved < max_shards:
            save_shard(processed_records, next_shard_index, out)
            shards_saved += 1

    print(f"\nProcessing complete!")
    print(f"Shards written this run: {shards_saved}")
    print(f"Unique records after deduplication (this run): {len(seen_hashes):,}")
    print(f"Shard size: {shard_size:,} samples per full shard")


def save_shard(records, shard_number: int, shard_dir: Path):
    """Save a batch of records to Parquet format."""
    from datasets import Dataset

    dataset = Dataset.from_dict(
        {
            "date": [r["date"] for r in records],
            "text": [r["text"] for r in records],
        }
    )

    shard_path = shard_dir / f"shard_{shard_number:04d}.parquet"
    dataset.to_parquet(str(shard_path))
    print(f"Saved shard {shard_number}: {shard_path} ({len(records)} records)")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Stream financial-news-multisource into Parquet shards (dedup, lowercase)."
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=50000,
        help="Number of unique accepted rows per shard (default: 50000).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="financial_news_shards",
        help="Directory for shard_*.parquet (default: financial_news_shards).",
    )
    p.add_argument(
        "--max-shards",
        type=int,
        default=None,
        metavar="N",
        help="Stop after writing N shards (after --skip-shards). Useful for a small sample run.",
    )
    p.add_argument(
        "--skip-shards",
        type=int,
        default=0,
        metavar="K",
        help=(
            "Discard the first K full shards without writing (stream still read from the start; "
            "global dedup unchanged). Output numbering starts at shard_0000 for the first saved file."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    load_and_process_financial_news(
        shard_size=args.shard_size,
        output_dir=args.output_dir,
        max_shards=args.max_shards,
        skip_shards=args.skip_shards,
    )
