import random
import shutil
from pathlib import Path

# Base directory = project root (QueryMinds-main)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Where all processed JSONs are (from Milestone 1)
SOURCE_DIR = BASE_DIR / "data" / "parsed_tokens"

# Where we put our chosen 30 JSON files
TARGET_DIR = BASE_DIR / "data" / "selected_30"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Collect all JSON files
all_files = sorted(SOURCE_DIR.glob("*.json"))

print(f"Found {len(all_files)} files in {SOURCE_DIR}")

if len(all_files) < 30:
    raise ValueError("Not enough files to choose 30 from!")

# For reproducibility (so you and your teammates can match)
random.seed(42)

# Randomly pick 30 files
selected = random.sample(all_files, 30)

print("Selected files:")
for f in selected:
    print("  ", f.name)

# Copy them
for src in selected:
    dst = TARGET_DIR / src.name
    shutil.copy2(src, dst)
    print(f"Copied {src.name} -> {dst}")

print("\nDone. 30 files are now in data/selected_30")
