import re
import math
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- Configuration ---
# Paths are set relative to the src/data_prep/ directory
INPUT_DIR = Path("/content/drive/MyDrive/financial_news_shards_labeled") 
OUTPUT_DIR = Path("/content/drive/MyDrive/llm_training_data")

TARGET_TOTAL_ROWS = 15000
NUM_TIME_ERAS = 5  

def _extract_shard_number(filename: str) -> int:
    """Extracts the integer number from filenames like 'shard_0015.parquet'"""
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else -1

def auto_squeeze_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    parquet_files = list(INPUT_DIR.glob("*.parquet"))
    if not parquet_files:
        print(f"Error: No labeled parquet files found in {INPUT_DIR.resolve()}")
        # Fallback check in case the old folder names are still being used
        print("Please verify if your data folders are named '02_labeled_shards' or 'financial_news_shards_labeled'")
        return
        
    parquet_files.sort(key=lambda p: _extract_shard_number(p.name))
    total_files = len(parquet_files)
    print(f"Found {total_files} labeled shards. Auto-grouping into {NUM_TIME_ERAS} Time Eras...")

    chunk_size = math.ceil(total_files / NUM_TIME_ERAS)
    eras = [parquet_files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
    
    rows_per_era = TARGET_TOTAL_ROWS // len(eras)
    sampled_dfs = []

    for era_idx, era_files in enumerate(eras, start=1):
        if not era_files:
            continue
            
        rows_per_file = rows_per_era // len(era_files)
        
        start_shard = _extract_shard_number(era_files[0].name)
        end_shard = _extract_shard_number(era_files[-1].name)
        
        print(f"  Era {era_idx} (Shards {start_shard}-{end_shard}): Found {len(era_files)} files. Pulling ~{rows_per_file} rows each.")
        
        for file in era_files:
            df = pd.read_parquet(file)
            n_to_sample = min(rows_per_file, len(df))
            
            df_sampled = df.sample(n=n_to_sample, random_state=42)
            sampled_dfs.append(df_sampled)

    print("\nCombining and shuffling the auto-balanced data...")
    master_df = pd.concat(sampled_dfs, ignore_index=True)
    master_df = master_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print(f"Successfully extracted {len(master_df):,} total rows.")

    train_df, temp_df = train_test_split(master_df, test_size=0.20, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    train_path = OUTPUT_DIR / "train_llm.parquet"
    val_path = OUTPUT_DIR / "val_llm.parquet"
    test_path = OUTPUT_DIR / "test_llm.parquet"
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"\n✅ Success! LLM-ready datasets saved to: {OUTPUT_DIR.resolve()}")
    print(f"  Train:      {len(train_df):,} rows")
    print(f"  Validation: {len(val_df):,} rows")
    print(f"  Test:       {len(test_df):,} rows")

if __name__ == "__main__":
    auto_squeeze_data()