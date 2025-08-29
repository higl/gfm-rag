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
    # """Load dataset with retry logic and cache clearing on corruption."""
    # import time
    # # use the datasets DownloadMode enum when forcing redownload
    # try:
    #     dl_force = datasets.DownloadMode.FORCE_REDOWNLOAD
    # except Exception:
    #     dl_force = None

    # for attempt in range(max_retries + 1):
    #     try:
    #         print(f"Loading dataset {dataset_name} (attempt {attempt + 1}/{max_retries + 1})")
    #         # Force redownload on retry attempts
    #         if attempt > 0 and dl_force is not None:
    #             if config_name:
    #                 ds = load_dataset(dataset_name, config_name, split=split, download_mode=dl_force, trust_remote_code=True)
    #             else:
    #                 ds = load_dataset(dataset_name, split=split, download_mode=dl_force, trust_remote_code=True)
    #         else:
    #             if config_name:
    #                 ds = load_dataset(dataset_name, config_name, split=split, trust_remote_code=True)
    #             else:
    #                 ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    #         return list(ds)
    #     except Exception as e:
    #         error_msg = str(e).lower()
    #         is_corruption_error = any(keyword in error_msg for keyword in [
    #             "histogram size mismatch",
    #             "dataset generation error", 
    #             "arrow",
    #             "parquet",
    #             "corrupt",
    #             "invalid",
    #             "checksum"
    #         ])
            
    #         if attempt < max_retries and is_corruption_error:
    #             print(f"Dataset loading failed (attempt {attempt + 1}): {e}")
    #             print(f"Detected corruption error, clearing cache and retrying...")
    #             clear_dataset_cache(dataset_name, config_name)
    #             time.sleep(2)  # Brief delay before retry
    #         else:
    #             print(f"Failed to load dataset {dataset_name} after {max_retries + 1} attempts")
    #             raise e

def add_doc(title: str, text: str,
            global_docs: Dict[str,str],
            title_map: Dict[str,str]) -> str:
    """
    Ensure uniqueness of doc keys in dataset_corpus.json.
    Keeps a stable 'Title ## 000' suffixing scheme per *base title*.
    Returns the final doc_id used (key in corpus).
    """
    base = clean_ws(title)
    if not base:
        base = "UNTITLED"
    if base not in title_map:
        candidate = f"{base} ## 000"
        if candidate in global_docs:
            idx = 1
            while True:
                candidate = f"{base} ## {idx:03d}"
                if candidate not in global_docs:
                    break
                idx += 1
        title_map[base] = candidate
        # Only write when we have non-empty text; otherwise defer
        if clean_ws(text):
            global_docs[candidate] = clean_ws(text)
        return candidate
    doc_id = title_map[base]
    # If new non-empty text differs, fork a new suffix
    if clean_ws(text) and global_docs.get(doc_id, "") != clean_ws(text):
        idx = 1
        while True:
            candidate = f"{base} ## {idx:03d}"
            if candidate not in global_docs:
                global_docs[candidate] = clean_ws(text)
                return candidate
            idx += 1
    # Otherwise reuse; fill text if missing/not set yet
    if clean_ws(text) and not global_docs.get(doc_id):
        global_docs[doc_id] = clean_ws(text)
    return doc_id

# NEW: provide sampling helper used by main()
def sample_list(items: List[Any], k: int) -> List[Any]:
    if k <= 0:
        return []
    if len(items) <= k:
        return list(items)
    return random.sample(items, k)

# Helper: pull first non-empty text from various shapes
def _first_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return clean_ws(value)
    if isinstance(value, dict):
        # common keys for nested answers
        for k in ("text", "answer", "value", "label"):
            v = value.get(k)
            s = clean_ws(v or "")
            if s:
                return s
        # fallback: any stringy value
        for v in value.values():
            s = _first_text(v)
            if s:
                return s
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            s = _first_text(item)
            if s:
                return s
        return ""
    # other types
    return clean_ws(str(value))

def _pick_field(d: dict, keys: List[str]) -> str:
    for k in keys:
        if k in d:
            s = _first_text(d.get(k))
            if s:
                return s
    return ""

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

def build_hotpot_docs_and_examples(examples: List[dict],
                                   global_docs: Dict[str,str]) -> Tuple[List[dict], Dict[str,str]]:
    title_map: Dict[str,str] = {}   # base title -> doc_id
    built: List[dict] = []
    # Build docs from 'context'
    for ex in examples:
        ctx = ex.get("context") or []
        # handle dict-of-lists: {'title': [...], 'sentences': [[...], ...]}
        if isinstance(ctx, dict):
            titles = ctx.get("title") or []
            blocks = ctx.get("sentences") or ctx.get("content") or []
            for i, t in enumerate(titles):
                sents = blocks[i] if i < len(blocks) else []
                text = join_sentences(sents or [])
                add_doc(t, text, global_docs, title_map)
            continue
        # Fallback: list/tuple/dict blocks
        for block in ctx:
            title = ""
            sents = []
            if isinstance(block, (list, tuple)):
                if len(block) >= 2:
                    title = block[0]; sents = block[1]
                elif len(block) == 1:
                    title = block[0]; sents = []
                else:
                    continue
            elif isinstance(block, dict):
                title = block.get("title") or block.get("doc_title") or ""
                sents = block.get("sentences") or block.get("sents") or block.get("content") or block.get("paragraph_text") or []
            else:
                continue
            text = join_sentences(sents or [])
            add_doc(title, text, global_docs, title_map)

    for ex in examples:
        qid = str(ex.get("_id") or ex.get("id"))
        q = clean_ws(ex.get("question",""))
        a = clean_ws(ex.get("answer",""))
        sfs = ex.get("supporting_facts") or []
        sup_titles = []
        seen = set()
        # Prefer dict-of-lists directly when present
        if isinstance(sfs, dict):
            for t in (sfs.get("title") or []):
                t = clean_ws(t)
                if not t or t in seen:
                    continue
                doc_id = title_map.get(t)
                if doc_id:
                    sup_titles.append(doc_id)
                    seen.add(t)
        else:
            for pair in sfs:
                if isinstance(pair, (list,tuple)) and pair:
                    t = clean_ws(pair[0])
                elif isinstance(pair, dict):
                    t = clean_ws(pair.get("title",""))
                else:
                    t = ""
                if not t or t in seen:
                    continue
                doc_id = title_map.get(t)
                if doc_id:
                    sup_titles.append(doc_id)
                    seen.add(t)
        built.append({"id": qid, "question": q, "answer": a, "supporting_facts": sup_titles})
    return built, title_map

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

def build_musique_docs_and_examples(examples: List[dict],
                                    global_docs: Dict[str,str]) -> Tuple[List[dict], Dict[str,str]]:
    title_map: Dict[str,str] = {}
    built: List[dict] = []

    # First pass: add all paragraphs to corpus
    for ex in examples:
        paras = ex.get("paragraphs") or ex.get("contexts") or []
        for p in paras:
            title = p.get("title") or ""
            text = p.get("paragraph_text") or p.get("text") or ""
            if title and text:
                add_doc(title, text, global_docs, title_map)

    # Build examples
    for ex in examples:
        q = clean_ws(ex.get("question") or ex.get("qas",[{}])[0].get("question",""))
        a = clean_ws(ex.get("answer") or (ex.get("answers") or [""])[0])
        qid = str(ex.get("id") or ex.get("qas",[{}])[0].get("id") or "")
        sup_titles: List[str] = []

        # Preferred: explicit booleans
        paras = ex.get("paragraphs") or []
        if paras and any("is_supporting" in p for p in paras):
            for p in paras:
                if p.get("is_supporting"):
                    t = clean_ws(p.get("title",""))
                    if t:
                        sup_titles.append(title_map.get(t) or add_doc(t, p.get("paragraph_text",""), global_docs, title_map))

        # Alternative: indices
        if not sup_titles:
            idxs = ex.get("paragraph_support_idx")
            if isinstance(idxs, list) and paras:
                for i in idxs:
                    if isinstance(i,int) and 0 <= i < len(paras):
                        t = clean_ws(paras[i].get("title",""))
                        if t:
                            sup_titles.append(title_map.get(t) or add_doc(t, paras[i].get("paragraph_text",""), global_docs, title_map))

        # Final fallback: unique titles from paragraphs marked supporting in nested QA
        if not sup_titles and "qas" in ex:
            for qa in (ex.get("qas") or []):
                ps = qa.get("paragraph_support_idx")
                if isinstance(ps, list) and paras:
                    for i in ps:
                        if isinstance(i,int) and 0 <= i < len(paras):
                            t = clean_ws(paras[i].get("title",""))
                            if t:
                                sup_titles.append(title_map.get(t) or add_doc(t, paras[i].get("paragraph_text",""), global_docs, title_map))
                if not q:
                    q = clean_ws(qa.get("question",""))
                if not a:
                    a = clean_ws(qa.get("answer","") or (qa.get("answers") or [""])[0])
                if not qid:
                    qid = str(qa.get("id",""))

        built.append({"id": qid, "question": q, "answer": a, "supporting_facts": list(dict.fromkeys(sup_titles))})
    return built, title_map

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

def _split_manual_2wiki(data: List[dict], split: str) -> List[dict]:
    """Deterministic split when the JSONL doesn't include explicit split tags.
       If entries contain a 'split' or 'set' key, honor it. Otherwise use index%10==0 -> validation."""
    if not data:
        return []

    target = "validation" if split in ("validation","dev","valid") else split

    # If entries include explicit split-like keys, filter by them
    for key in ("split","set","subset"):
        if any(isinstance(d.get(key), str) for d in data):
            return [d for d in data if (d.get(key) or "").lower() == target]

    # fallback deterministic partition: every 10th example -> validation
    if target in ("validation","dev"):
        return [d for i,d in enumerate(data) if (i % 10) == 0]
    # train = the rest
    return [d for i,d in enumerate(data) if (i % 10) != 0]

# def load_2wiki(split: str) -> List[dict]:
#     # Prefer cmriat mirror. Fall back to thinkall or manual queries.jsonl if rows lack usable QA+context.
#     def _extract_nonempty_qa(ex: dict) -> Tuple[str, str]:
#         q = _pick_field(ex, ["question", "query", "q", "question_text"])
#         a = _pick_field(ex, ["answer", "answers", "a", "final_answer", "answer_text"])
#         return q, a

#     def _extract_context_blocks_from_ex(ex: dict) -> List[Tuple[str, List[str]]]:
#         # primary: nested dict under 'context'
#         ctx = ex.get("context")
#         titles = []; contents = []
#         if isinstance(ctx, dict):
#             titles = ctx.get("title") or []
#             contents = ctx.get("content") or ctx.get("sentences") or []
#         else:
#             # flattened variants (rare, but be robust)
#             flat_titles = ex.get("context.title") or ex.get("context_titles")
#             flat_contents = ex.get("context.content") or ex.get("context_sentences") or ex.get("context_contents")
#             if isinstance(flat_titles, list) and isinstance(flat_contents, list):
#                 titles, contents = flat_titles, flat_contents
#         blocks: List[Tuple[str, List[str]]] = []
#         for i, t in enumerate(titles or []):
#             sents = contents[i] if i < len(contents) else []
#             if t and sents:
#                 blocks.append((t, sents))
#         return blocks

#     def _looks_like_full_examples(items: List[dict]) -> bool:
#         if not items:
#             return False
#         n = min(50, len(items))
#         ok = 0
#         for ex in items[:n]:
#             q, a = _extract_nonempty_qa(ex)
#             ctx_blocks = _extract_context_blocks_from_ex(ex)
#             if q and a and ctx_blocks:
#                 ok += 1
#         # Require a reasonable portion to truly contain QA + context
#         return (ok / max(1, n)) >= 0.2

#     # Try cmriat
#     try:
#         ds = load_dataset_with_retry("cmriat/2wikimultihopqa", split=split)
#         if _looks_like_full_examples(ds):
#             return ds
#         print("2Wiki cmriat split lacks non-empty QA+context; trying thinkall...")
#     except Exception as e_cmriat:
#         print(f"cmriat loader failed: {e_cmriat}. Trying thinkall...")

#     # Try thinkall
#     try:
#         ds = load_dataset_with_retry("thinkall/2WikiMultiHopQA", split=split)
#         if _looks_like_full_examples(ds):
#             return ds
#         print("2Wiki thinkall split lacks non-empty QA+context; trying manual queries.jsonl...")
#     except Exception as e_think:
#         print(f"thinkall loader failed: {e_think}. Trying manual queries.jsonl...")

#     # Manual queries.jsonl from hub (thinkall first, then cmriat)
#     last_err = None
#     for repo in ("thinkall/2WikiMultiHopQA", "cmriat/2wikimultihopqa"):
#         try:
#             all_items = _manual_load_2wiki_from_hub(repo)
#             shard = _split_manual_2wiki(all_items, split)
#             return shard
#         except Exception as e_manual:
#             last_err = e_manual
#             continue
#     if last_err:
#         raise last_err
#     return []

# def build_2wiki_docs_and_examples(examples: List[dict],
#                                   global_docs: Dict[str,str]) -> Tuple[List[dict], Dict[str,str]]:
#     title_map: Dict[str,str] = {}
#     built: List[dict] = []

#     def _extract_context_blocks(ex: dict) -> List[Tuple[str, List[str]]]:
#         blocks: List[Tuple[str, List[str]]] = []
#         ctx = ex.get("context")
#         if isinstance(ctx, dict):
#             titles = ctx.get("title") or []
#             contents = ctx.get("content") or ctx.get("sentences") or []
#             for i, t in enumerate(titles):
#                 sents = contents[i] if i < len(contents) else []
#                 if t and sents:
#                     blocks.append((t, sents))
#         else:
#             # flattened fallbacks
#             flat_titles = ex.get("context.title") or ex.get("context_titles")
#             flat_contents = ex.get("context.content") or ex.get("context_sentences") or ex.get("context_contents")
#             if isinstance(flat_titles, list) and isinstance(flat_contents, list):
#                 for i, t in enumerate(flat_titles):
#                     sents = flat_contents[i] if i < len(flat_contents) else []
#                     if t and sents:
#                         blocks.append((t, sents))
#         return blocks

#     # Add docs from context
#     for ex in examples:
#         for t, sents in _extract_context_blocks(ex):
#             add_doc(t, join_sentences(sents), global_docs, title_map)

#     # Build examples
#     for ex in examples:
#         qid = str(ex.get("id") or ex.get("_id") or "")
#         q = _pick_field(ex, ["question", "query", "q", "question_text"])
#         a = _pick_field(ex, ["answer", "answers", "a", "final_answer", "answer_text"])

#         # supporting_facts: nested or flattened
#         sup = ex.get("supporting_facts")
#         if sup is None:
#             # flattened fallback
#             titles = ex.get("supporting_facts.title")
#             sids = ex.get("supporting_facts.sent_id") or ex.get("supporting_facts.sent_ids")
#             if isinstance(titles, list):
#                 sup = {"title": titles, "sent_id": sids or []}
#         sup = sup or {}
#         sup_titles: List[str] = []
#         seen: set = set()
#         if isinstance(sup, dict):
#             for t in (sup.get("title") or []):
#                 t = clean_ws(t)
#                 if not t or t in seen:
#                     continue
#                 doc_id = title_map.get(t)
#                 if doc_id:
#                     sup_titles.append(doc_id)
#                     seen.add(t)
#         elif isinstance(sup, list):
#             for item in sup:
#                 t = ""
#                 if isinstance(item, (list, tuple)) and item:
#                     t = clean_ws(item[0])
#                 elif isinstance(item, dict):
#                     t = clean_ws(item.get("title",""))
#                 if not t or t in seen:
#                     continue
#                 doc_id = title_map.get(t)
#                 if doc_id:
#                     sup_titles.append(doc_id)
#                     seen.add(t)

#         built.append({"id": qid, "question": q, "answer": a, "supporting_facts": list(dict.fromkeys(sup_titles))})
#     return built, title_map

# def load_2wiki(split: str) -> List[dict]:
#     """
#     Prefer cmriat/2wikimultihopqa. That repo only ships a 'validation' split.
#     If asked for 'train', fall back to 'validation' deterministically.
#     """
#     target_split = split
#     try:
#         # cmriat has only 'validation'; viewer columns: id, question, golden_answers, metadata{context, supporting_facts}
#         # https://huggingface.co/datasets/cmriat/2wikimultihopqa
#         return load_dataset_with_retry("cmriat/2wikimultihopqa", split=target_split)
#     except Exception:
#         if split == "train":
#             # use validation as the training pool for this source
#             return load_dataset_with_retry("cmriat/2wikimultihopqa", split="validation")
#         # fallbacks: thinkall or manual JSONL if you still want them
#         try:
#             return load_dataset_with_retry("thinkall/2WikiMultihopQA", split=split)
#         except Exception:
#             return []

# def load_2wiki(split: str) -> List[dict]:
#     """
#     Prefer xanhho/2WikiMultihopQA which ships train/dev/test.
#     Map 'validation' -> 'dev'. Fallbacks: cmriat (validation-only), then thinkall.
#     """
#     split_map = {"validation": "dev", "valid": "dev", "dev": "dev",
#                  "train": "train", "test": "test"}
#     x_split = split_map.get(split, split)
#     try:
#         return load_dataset_with_retry("xanhho/2WikiMultihopQA", split=x_split)
#     except Exception as e_x:
#         print(f"xanhho load failed: {e_x}; trying cmriat...")
#         try:
#             # cmriat only has validation; use it for any requested split
#             return load_dataset_with_retry("cmriat/2wikimultihopqa", split="validation")
#         except Exception as e_c:
#             print(f"cmriat load failed: {e_c}; trying thinkall...")
#             try:
#                 return load_dataset_with_retry("thinkall/2WikiMultihopQA", split=split)
#             except Exception as e_t:
#                 print(f"thinkall load failed: {e_t}")
#                 return []

# # --- loader ---
# def load_2wiki(split: str) -> List[dict]:
#     # map common aliases
#     split_map = {"validation": "dev", "valid": "dev", "dev": "dev", "train": "train", "test": "test"}
#     s = split_map.get(split, split)
#     return load_dataset("voidful/2WikiMultihopQA", split=s)
#     # # 1) voidful: train/dev/test as JSON, no remote code
#     # try:
#     #     return load_dataset_with_retry("voidful/2WikiMultihopQA", split=s)
#     # except Exception as e_void:
#     #     print(f"voidful load failed: {e_void}; trying cmriat validation-only mirror...")
#     # # 2) cmriat: validation-only with metadata wrapper
#     # try:
#     #     return load_dataset_with_retry("cmriat/2wikimultihopqa", split="validation")
#     # except Exception as e_cmriat:
#     #     print(f"cmriat load failed: {e_cmriat}; trying thinkall manual fallback...")
#     # # 3) thinkall manual JSONL as last resort (your existing helper)
#     # try:
#     #     all_items = _manual_load_2wiki_from_hub("thinkall/2WikiMultihopQA")
#     #     return _split_manual_2wiki(all_items, s)
#     # except Exception as e_think:
#     #     print(f"thinkall manual fallback failed: {e_think}")
#     #     return []


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
    # except Exception as e_void:
    #     print(f"voidful raw load failed: {e_void}; trying cmriat validation-only...")

    # # 2) cmriat validation-only mirror (Parquet auto-convert)
    # try:
    #     return load_dataset_with_retry("cmriat/2wikimultihopqa", split="validation")
    # except Exception as e_cmriat:
    #     print(f"cmriat load failed: {e_cmriat}; trying thinkall manual queries.jsonl...")

    # # 3) thinkall manual JSONL (last resort)
    # try:
    #     all_items = _manual_load_2wiki_from_hub("thinkall/2WikiMultihopQA")
    #     return _split_manual_2wiki(all_items, split)
    # except Exception as e_think:
    #     print(f"thinkall manual fallback failed: {e_think}")
    #     return []


# def build_2wiki_docs_and_examples(
#     examples: List[dict],
#     global_docs: Dict[str, str],
# ) -> Tuple[List[dict], Dict[str, str]]:
#     """
#     cmriat schema:
#       - id: str
#       - question: str
#       - golden_answers: list[str]
#       - metadata: dict with:
#           * supporting_facts: {'title': [...], 'sent_id': [...]}
#           * context: {'title': [...], 'content': [[sentences...], ...]}
#     Build corpus from context; map supporting_facts titles to doc_ids.
#     """
#     title_map: Dict[str, str] = {}
#     built: List[dict] = []

#     def _get_meta(ex: dict) -> dict:
#         return ex.get("metadata") or {}

#     def _extract_ctx(ex: dict) -> list[tuple[str, list[str]]]:
#         meta = _get_meta(ex)
#         ctx = meta.get("context") or ex.get("context") or {}
#         titles = (ctx.get("title") or []) if isinstance(ctx, dict) else []
#         contents = (ctx.get("content") or ctx.get("sentences") or []) if isinstance(ctx, dict) else []
#         out = []
#         for i, t in enumerate(titles):
#             if not t:
#                 continue
#             sents = contents[i] if i < len(contents) else []
#             if sents:
#                 out.append((t, sents))
#         return out

#     # 1) Build corpus
#     for ex in examples:
#         for t, sents in _extract_ctx(ex):
#             add_doc(t, join_sentences(sents), global_docs, title_map)

#     # 2) Build QA rows
#     for ex in examples:
#         meta = _get_meta(ex)

#         qid = str(ex.get("id") or ex.get("_id") or "")
#         q = clean_ws(ex.get("question") or meta.get("question") or "")

#         # golden_answers can be a list; take the first non-empty string
#         ga = ex.get("golden_answers") or meta.get("golden_answers") or ex.get("answers") or meta.get("answers") or []
#         a = _first_text(ga)

#         sup = meta.get("supporting_facts") or ex.get("supporting_facts") or {}
#         sup_titles: list[str] = []
#         seen: set[str] = set()

#         if isinstance(sup, dict):
#             for t in (sup.get("title") or []):
#                 t = clean_ws(t)
#                 if not t or t in seen:
#                     continue
#                 doc_id = title_map.get(t)
#                 if doc_id:
#                     sup_titles.append(doc_id)
#                     seen.add(t)
#         elif isinstance(sup, list):
#             for item in sup:
#                 t = ""
#                 if isinstance(item, (list, tuple)) and item:
#                     t = clean_ws(item[0])
#                 elif isinstance(item, dict):
#                     t = clean_ws(item.get("title", ""))
#                 if not t or t in seen:
#                     continue
#                 doc_id = title_map.get(t)
#                 if doc_id:
#                     sup_titles.append(doc_id)
#                     seen.add(t)

#         built.append({
#             "id": qid,
#             "question": q,
#             "answer": a,
#             "supporting_facts": list(dict.fromkeys(sup_titles)),
#         })

#     return built, title_map

# def build_2wiki_docs_and_examples(examples: List[dict],
#                                   global_docs: Dict[str,str]) -> Tuple[List[dict], Dict[str,str]]:
#     title_map: Dict[str,str] = {}
#     built: List[dict] = []

#     def _ctx_blocks(ex: dict) -> List[Tuple[str, List[str]]]:
#         # xanhho: context is List[{"title":str, "content": List[str]}]
#         ctx = ex.get("context")
#         out = []
#         if isinstance(ctx, list):
#             for blk in ctx:
#                 if isinstance(blk, dict):
#                     t = clean_ws(blk.get("title",""))
#                     sents = blk.get("content") or blk.get("sentences") or []
#                     if t and isinstance(sents, list) and sents:
#                         out.append((t, sents))
#             return out
#         # cmriat: metadata.context is dict-of-lists
#         meta = ex.get("metadata") or {}
#         ctx = meta.get("context")
#         if isinstance(ctx, dict):
#             titles = ctx.get("title") or []
#             contents = ctx.get("content") or ctx.get("sentences") or []
#             for i, t in enumerate(titles):
#                 sents = contents[i] if i < len(contents) else []
#                 if t and sents:
#                     out.append((t, sents))
#         return out

#     # 1) Corpus
#     for ex in examples:
#         for t, sents in _ctx_blocks(ex):
#             add_doc(t, join_sentences(sents), global_docs, title_map)

#     # 2) QA rows
#     for ex in examples:
#         qid = str(ex.get("id") or ex.get("_id") or "")
#         meta = ex.get("metadata") or {}
#         q = clean_ws(ex.get("question") or meta.get("question") or "")
#         # xanhho: 'answer' is a string; cmriat: 'golden_answers' is list[str]
#         a = clean_ws(ex.get("answer") or _first_text(ex.get("golden_answers") or meta.get("golden_answers") or ""))

#         sup = ex.get("supporting_facts") or meta.get("supporting_facts") or {}
#         sup_titles: List[str] = []
#         if isinstance(sup, list):  # xanhho
#             for item in sup:
#                 if isinstance(item, dict):
#                     t = clean_ws(item.get("title",""))
#                     if t and t in title_map:
#                         sup_titles.append(title_map[t])
#         elif isinstance(sup, dict):  # cmriat
#             for t in (sup.get("title") or []):
#                 t = clean_ws(t)
#                 if t and t in title_map:
#                     sup_titles.append(title_map[t])

#         built.append({
#             "id": qid,
#             "question": q,
#             "answer": a,
#             "supporting_facts": list(dict.fromkeys(sup_titles)),
#         })

#     return built, title_map

# --- builder robust to voidful/xanhho/cmriat ---
def build_2wiki_docs_and_examples(examples: List[dict],
                                  global_docs: Dict[str,str]) -> Tuple[List[dict], Dict[str,str]]:
    title_map: Dict[str,str] = {}
    built: List[dict] = []

    def _ctx_blocks(ex: dict) -> List[Tuple[str, List[str]]]:
        # voidful: context is list of [title, [sentences...]]
        ctx = ex.get("context")
        out: List[Tuple[str, List[str]]] = []
        if isinstance(ctx, list):
            for blk in ctx:
                t, sents = "", []
                if isinstance(blk, dict):                 # xanhho style
                    t = clean_ws(blk.get("title",""))
                    sents = blk.get("content") or blk.get("sentences") or []
                elif isinstance(blk, (list, tuple)):      # voidful style
                    if len(blk) >= 1: t = clean_ws(blk[0])
                    if len(blk) >= 2 and isinstance(blk[1], list): sents = blk[1]
                if t and sents:
                    out.append((t, sents))
            return out
        # cmriat: context in metadata dict-of-lists
        meta = ex.get("metadata") or {}
        cdict = meta.get("context") or {}
        if isinstance(cdict, dict):
            titles = cdict.get("title") or []
            contents = cdict.get("content") or cdict.get("sentences") or []
            for i, t in enumerate(titles):
                sents = contents[i] if i < len(contents) else []
                if t and sents:
                    out.append((clean_ws(t), sents))
        return out

    # 1) corpus
    for ex in examples:
        for t, sents in _ctx_blocks(ex):
            add_doc(t, join_sentences(sents), global_docs, title_map)

    # 2) qa rows
    for ex in examples:
        meta = ex.get("metadata") or {}
        qid = str(ex.get("id") or ex.get("_id") or "")
        q = clean_ws(ex.get("question") or meta.get("question") or "")
        # answers: voidful/xanhho: 'answer' string; cmriat: 'golden_answers' list
        a = clean_ws(ex.get("answer") or _first_text(ex.get("golden_answers") or meta.get("golden_answers") or ""))

        sup = ex.get("supporting_facts") or meta.get("supporting_facts") or {}
        sup_titles: List[str] = []
        seen: set[str] = set()
        if isinstance(sup, dict):  # cmriat dict-of-lists
            for t in (sup.get("title") or []):
                t = clean_ws(t)
                if t and t not in seen and t in title_map:
                    sup_titles.append(title_map[t]); seen.add(t)
        elif isinstance(sup, list):  # voidful/xanhho list
            for item in sup:
                t = ""
                if isinstance(item, dict): t = clean_ws(item.get("title",""))
                elif isinstance(item, (list, tuple)) and item: t = clean_ws(item[0])
                if t and t not in seen and t in title_map:
                    sup_titles.append(title_map[t]); seen.add(t)

        built.append({"id": qid, "question": q, "answer": a, "supporting_facts": sup_titles})

    return built, title_map


# ---------- build combined ----------

def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)
    _sanitize_trailing_comma_file(path)

# NEW: helpers to select and shard examples deterministically
def _chunk_balanced(lst: List[Any], n: int) -> List[List[Any]]:
    if n <= 0:
        return [lst]
    k, m = divmod(len(lst), n)
    # first m chunks have size k+1, the rest size k
    chunks = []
    start = 0
    for i in range(n):
        size = k + 1 if i < m else k
        if size <= 0:
            continue
        chunks.append(lst[start:start+size])
        start += size
    return [c for c in chunks if c]

def _select_and_shard(train_pool: List[dict], total_target: int, shards: int, do_shuffle: bool) -> List[List[dict]]:
    pool = list(train_pool)
    if do_shuffle:
        random.shuffle(pool)
    if total_target > 0 and len(pool) > total_target:
        pool = random.sample(pool, total_target)
    return _chunk_balanced(pool, shards)

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

    # Shard per dataset: sample up to target and partition into num_shards
    datasets_cfg = [
        ("hotpotqa", hp_train, hp_dev, build_hotpot_docs_and_examples),
        ("musique", mq_train, mq_dev, build_musique_docs_and_examples),
        ("2wikimultihopqa", tw_train, tw_dev, build_2wiki_docs_and_examples),
    ]

    DATA_DIR = "data"  # stage1 expects ./data/<name>/raw

    def write_dataset(out_name: str, corpus_obj: Dict[str,str], train_rows: Optional[List[dict]], test_rows: Optional[List[dict]]) -> None:
        base = os.path.join(DATA_DIR, out_name, "raw")
        os.makedirs(base, exist_ok=True)
        write_json(os.path.join(base, "dataset_corpus.json"), corpus_obj)
        if train_rows is not None:
            write_json(os.path.join(base, "train.json"), train_rows)
        if test_rows is not None:
            write_json(os.path.join(base, "test.json"), test_rows)

    total_docs = 0
    total_train_qs = 0
    total_test_qs = 0
    total_shards = 0

    # Build test sets (per dataset, independent corpus)
    for ds_key, _, dev_pool, builder in datasets_cfg:
        test_samples = sample_list(list(dev_pool), args.test_size)
        test_corpus: Dict[str,str] = {}
        _, _ = builder(test_samples, test_corpus)
        out_name = f"{'2wikimultihopqa' if ds_key=='2wikimultihopqa' else ds_key}_test"
        write_dataset(out_name, test_corpus, None, test_samples)
        total_test_qs += len(test_samples)
        total_docs += len(test_corpus)

    # Build training shards: per-shard corpus and train.json
    for ds_key, train_pool, _, builder in datasets_cfg:
        shards = _select_and_shard(list(train_pool), args.train_target_per_dataset, args.num_shards, args.shuffle)
        for i, shard_examples in enumerate(shards):
            shard_corpus: Dict[str,str] = {}
            built_rows, _ = builder(shard_examples, shard_corpus)
            out_name = f"{'2wikimultihopqa' if ds_key=='2wikimultihopqa' else ds_key}_train{i}"
            write_dataset(out_name, shard_corpus, built_rows, None)
            total_docs += len(shard_corpus)
            total_train_qs += len(built_rows)
            total_shards += 1

    print(f"Built {total_shards} training shards across datasets (target {args.num_shards} per dataset)")
    print(f"Total shard docs written: {total_docs}")
    print(f"Total train QA pairs: {total_train_qs}")
    print(f"Total test QA pairs: {total_test_qs}")

if __name__ == "__main__":
    main()
