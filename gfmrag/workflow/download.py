#!/usr/bin/env python
# build_multihop_raw.py
# Build GFM-RAG raw data from HotpotQA, MuSiQue, and 2WikiMultiHopQA.
# Outputs:
#   .data/<name>/raw/dataset_corpus.json
#   .data/<name>/raw/train.json
#   .data/<name>/raw/test.json
#
# Default sampling: 20k train per dataset (up to available), 2k test per dataset.

import argparse, json, os, random, re
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Optional, Any

try:
    from datasets import load_dataset
    from datasets.utils.file_utils import cached_path
    import datasets
except Exception as e:
    raise SystemExit("pip install datasets") from e

random.seed(7)

# --- add near the other manual helpers ---
from typing import List, Dict

def _manual_load_voidful_2wiki(split: str) -> List[dict]:
    """
    Download raw JSON arrays from voidful/2WikiMultihopQA and return a list[dict].
    Splits: train.json, dev.json, test.json.
    """
    from huggingface_hub import hf_hub_download
    split_map = {"validation": "dev", "valid": "dev", "dev": "dev",
                 "train": "train", "test": "test"}
    s = split_map.get(split, split)
    fname = {"train": "train.json", "dev": "dev.json", "test": "test.json"}.get(s, "dev.json")
    path = hf_hub_download(repo_id="voidful/2WikiMultihopQA", filename=fname, repo_type="dataset")

    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)  # file is a single JSON array
    # No schema enforcement here; builder handles multiple shapes.
    # Just guarantee dict type (some sources store keys as strings already).
    out = []
    for ex in data:
        if isinstance(ex, dict):
            out.append(ex)
    return out

# ---------- utils ----------

def clean_ws(s: str) -> str:
    s = (s or "").replace("\t"," ").replace("\r"," ").replace("\n"," ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def join_sentences(ss: Iterable[str]) -> str:
    return clean_ws(" ".join([clean_ws(x) for x in ss if x]))

# NEW: sanitize any trailing comma before closing brace/bracket
def _sanitize_trailing_comma_file(path: str) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        # find the last closing } or ]
        m = re.search(r'([}\]])\s*$', data, re.S)
        if not m:
            return
        end_pos = m.start(1)
        # walk back to find previous non-space
        i = end_pos - 1
        while i >= 0 and data[i].isspace():
            i -= 1
        if i >= 0 and data[i] == ",":
            # drop the trailing comma
            new_data = data[:i] + data[i+1:]
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_data)
    except Exception:
        # best-effort; do not fail the build
        pass

def clear_dataset_cache(dataset_name: str, config_name: str = None) -> None:
    """Clear corrupted dataset cache to force re-download."""
    try:
        import shutil
        # Candidate cache roots: env vars or common defaults
        cache_locations = []
        # honor HF env vars if present
        if os.environ.get("HF_DATASETS_CACHE"):
            cache_locations.append(os.environ.get("HF_DATASETS_CACHE"))
        if os.environ.get("HF_HOME"):
            cache_locations.append(os.path.join(os.environ.get("HF_HOME"), "datasets"))
        if os.environ.get("HF_HUB_CACHE"):
            cache_locations.append(os.environ.get("HF_HUB_CACHE"))
        # common defaults
        cache_locations.extend([
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets"),
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
        ])
        
        for cache_dir in cache_locations:
            if not cache_dir or not os.path.exists(cache_dir):
                continue
                
            # Try multiple cache naming patterns
            cache_patterns = []
            if config_name:
                cache_patterns.extend([
                    f"{dataset_name}___{config_name}",
                    f"{dataset_name}--{config_name}",
                    dataset_name
                ])
            else:
                cache_patterns.append(dataset_name)
            
            for pattern in cache_patterns:
                dataset_cache = os.path.join(cache_dir, pattern)
                if os.path.exists(dataset_cache):
                    print(f"Clearing corrupted cache: {dataset_cache}")
                    shutil.rmtree(dataset_cache, ignore_errors=True)
                    
            # Also clear any cache directories that contain the dataset name (safe heuristic)
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                if os.path.isdir(item_path) and dataset_name.replace("/", "_") in item:
                    print(f"Clearing related cache: {item_path}")
                    shutil.rmtree(item_path, ignore_errors=True)
                        
    except Exception as e:
        print(f"Warning: Could not clear cache: {e}")

def load_dataset_with_retry(dataset_name: str, config_name: str = None, split: str = None, max_retries: int = 0):
    return list(load_dataset(dataset_name, config_name, split=split, trust_remote_code=True))

# ---------- HotpotQA (distractor) ----------
# HF: "hotpot_qa", config="distractor"; has 'context' and 'supporting_facts'.
# Paper: Yang et al., 2018.  :contentReference[oaicite:1]{index=1}

def load_hotpot(split: str) -> List[dict]:
    # Try the robust loader first; if that fails, attempt a fresh temporary cache dir
    try:
        return load_dataset_with_retry("hotpot_qa", "distractor", split)
    except Exception as e:
        print("Primary hotpot load failed, attempting fresh temporary cache download...")
        import tempfile, shutil
        tmpdir = tempfile.mkdtemp(prefix="hf_hotpot_")
        try:
            # Force redownload into a clean cache dir to avoid reading corrupt parquet files
            try:
                ds = load_dataset("hotpot_qa", "distractor", split=split, cache_dir=tmpdir, download_mode=datasets.DownloadMode.FORCE_REDOWNLOAD)
            except Exception:
                # fallback if enum not available
                ds = load_dataset("hotpot_qa", "distractor", split=split, cache_dir=tmpdir)
            return list(ds)
        finally:
            # best-effort cleanup of temporary cache
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass


# ---------- MuSiQue ----------
# HF: "dgslibisey/MuSiQue" (JSONL). Fields vary in literature.
# Viewer shows 'paragraphs': [{'title','paragraph_text','is_supporting'}, ...] and
# either top-level {'question','answer'} or nested QA items. :contentReference[oaicite:2]{index=2}

def load_musique(split: str) -> List[dict]:
    # Use default builder; common splits: "train", "validation" (aka "dev"), "test"
    # Some repos name dev as "dev". Try fallback chain.
    try:
        return load_dataset_with_retry("dgslibisey/MuSiQue", split=split)
    except Exception:
        # fallback aliases
        alias = {"validation":"dev", "dev":"validation"}.get(split, split)
        return load_dataset_with_retry("dgslibisey/MuSiQue", split=alias)

# ---------- 2WikiMultiHopQA ----------
# Safer mirror: "cmriat/2wikimultihopqa" (schema previewed).  :contentReference[oaicite:3]{index=3}
# Fields: 'question','answer','supporting_facts': {'title':[...],'sent_id':[...]}
#         'context': {'title':[...], 'content': [[sentences...], ...]}

# Add module-level cache for manual 2Wiki loads
_TWIKI_MANUAL_CACHE: Dict[str, List[dict]] = {}

def _manual_load_2wiki_from_hub(repo_id: str) -> List[dict]:
    """Download and parse queries.jsonl (or alternatives) from a dataset repo on the HF hub.
       Returns a list of dicts (parsed JSONL). Caches the result."""
    if repo_id in _TWIKI_MANUAL_CACHE:
        return _TWIKI_MANUAL_CACHE[repo_id]

    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise RuntimeError("Please pip install huggingface_hub to use manual 2Wiki fallback") from e

    # try several likely filenames; ensure repo_type="dataset"
    filenames = ["queries.jsonl", "corpus.jsonl", "train.jsonl", "validation.jsonl", "data.jsonl"]
    fname = None
    last_exc = None
    for fn in filenames:
        try:
            fname = hf_hub_download(repo_id=repo_id, filename=fn, repo_type="dataset")
            break
        except Exception as e:
            last_exc = e
            continue
    if not fname:
        # surface a clearer error if none found
        raise RuntimeError(f"No known jsonl found in {repo_id}: last error: {last_exc}") from last_exc

    data: List[dict] = []
    import json
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except Exception:
                # skip malformed lines
                continue

    _TWIKI_MANUAL_CACHE[repo_id] = data
    return data

# --- replace your load_2wiki with this robust version ---
def load_2wiki(split: str) -> List[dict]:
    """
    Prefer raw JSON from voidful/2WikiMultihopQA (has train/dev/test).
    Fallbacks: cmriat (validation-only), then thinkall manual JSONL.
    Avoids datasets.load_dataset to skip Arrow/Pandas type inference on mixed schemas.
    """
    # 1) voidful raw files
    # try:
    return _manual_load_voidful_2wiki(split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", required=True, help="Name under .data/<out-name>/raw")
    ap.add_argument("--root", default=".data", help="Base path for outputs")
    # legacy knobs kept for compatibility but not used for sharding
    ap.add_argument("--hotpot-train", type=int, default=20000)
    ap.add_argument("--musique-train", type=int, default=20000)
    ap.add_argument("--twiki-train", type=int, default=20000)
    ap.add_argument("--hotpot-test", type=int, default=1000)
    ap.add_argument("--musique-test", type=int, default=1000)
    ap.add_argument("--twiki-test", type=int, default=1000)
    ap.add_argument("--shuffle", action="store_true")
    # NEW: shard controls
    ap.add_argument("--train-target-per-dataset", type=int, default=20000)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--test-size", type=int, default=1000)
    args = ap.parse_args()

    out_raw = os.path.join(args.root, args.out_name, "raw")
    os.makedirs(out_raw, exist_ok=True)

    # Load splits
    hp_train = load_hotpot("train")
    hp_dev   = load_hotpot("validation")
    mq_train = load_musique("train")
    mq_dev   = load_musique("validation")
    tw_train = load_2wiki("train")
    try:
        tw_dev = load_2wiki("validation")
    except Exception:
        tw_dev = load_2wiki("dev")

if __name__ == "__main__":
    main()
