#!/usr/bin/env python3
"""
CALLNGRAM Dataset Fetcher
Downloads primary (AppTek Call-Center Dialogues) and secondary (Bitext Customer Support) datasets.
"""

import os
import sys
import json
import urllib.request
import urllib.error
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw_transcripts"
APPTEK_DIR = DATA_DIR / "apptek"
BITEXT_FILE = DATA_DIR / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

# Hugging Face URLs
BITEXT_URL = "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

APPTEK_BASE_URL = "https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues/resolve/main/test"
APPTEK_ACCENTS = [
    "en-AU", "en-CA", "en-CN", "en-GB", "en-GB_SCT", "en-GB_WLS",
    "en-IE", "en-IN", "en-MX", "en-SG", "en-US_Aave", "en-US_General",
    "en-US_Southern", "en-ZA"
]


def download_file(url: str, dest: Path, desc: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[OK] {desc} already exists ({dest.stat().st_size:,} bytes).")
        return
    print(f"Downloading {desc} from {url}...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (CALLNGRAM Pipeline/1.0)"}
    )
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)
    print(f"[SUCCESS] Downloaded {desc} ({dest.stat().st_size:,} bytes).")


def fetch_bitext():
    print("--- Fetching Bitext Customer Support Dataset ---")
    download_file(BITEXT_URL, BITEXT_FILE, "Bitext Customer Support 27K CSV")


def fetch_apptek():
    print("--- Fetching AppTek Call-Center Dialogues Metadata ---")
    APPTEK_DIR.mkdir(parents=True, exist_ok=True)
    total_records = 0
    domains = set()
    accents = set()
    customer_count = 0
    agent_count = 0

    all_records = []
    for accent in APPTEK_ACCENTS:
        url = f"{APPTEK_BASE_URL}/{accent}/metadata.jsonl"
        accent_file = APPTEK_DIR / f"{accent}_metadata.jsonl"
        download_file(url, accent_file, f"AppTek {accent} metadata")

        # Parse records
        with open(accent_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                all_records.append(rec)
                total_records += 1
                domains.add(rec.get("domain", ""))
                accents.add(rec.get("accent", accent))
                role = rec.get("role", "").lower()
                if role == "customer":
                    customer_count += 1
                elif role == "agent":
                    agent_count += 1

    # Save consolidated raw jsonl and csv
    consolidated_jsonl = APPTEK_DIR / "apptek_dialogues_consolidated.jsonl"
    with open(consolidated_jsonl, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    print(f"\n[AppTek Audit]")
    print(f"Total transcript records: {total_records}")
    print(f"Unique domains: {len(domains)} -> {sorted(list(domains))}")
    print(f"Accent groups: {len(accents)} -> {sorted(list(accents))}")
    print(f"Customer records: {customer_count}")
    print(f"Agent records: {agent_count}")


if __name__ == "__main__":
    fetch_bitext()
    fetch_apptek()
    print("\nDataset preparation finished successfully!")
