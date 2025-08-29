#!/usr/bin/env python
# build_and_index_gfmrag.py
# End-to-end pipeline:
#  1) Acquire raw text from flexible sources (Wikipedia, arXiv, URLs list, HF datasets, local files)
#  2) Write raw/dataset_corpus.json
#  3) Extract triples with REBEL and build document→entities map
#  4) Write processed/stage1/{kg.txt, document2entities.json}
#
# Usage:
#   python build_and_index_gfmrag.py \
#       --data-root /path/to/DATASET_NAME \
#       --config /path/to/config.json \
#       --device cuda:0 \
#       --rebel-model Babelscape/rebel-large
#
# Minimal config.json example:
# {
#   "sources": [
#     {"type": "wikipedia", "titles": ["Graph neural network","Knowledge graph"], "lang": "en"},
#     {"type": "arxiv", "query": "graph RAG OR knowledge graph retrieval", "max_results": 50, "download_pdf": false},
#     {"type": "urls_file", "path": "/abs/path/urls.txt"},
#     {"type": "hf_dataset", "name": "ag_news", "split": "train", "text_field": "text", "limit": 2000},
#     {"type": "local_dir", "path": "/abs/path/docs", "glob": "**/*.txt"}
#   ],
#   "filters": {"lang": "en", "min_chars": 800, "max_chars": 200000}
# }

import argparse
import json
import os
import re
import sys
import io
import hashlib
from typing import List, Tuple, Dict, Set, Iterable, Optional

# --------------------
# Optional deps imports (handled gracefully if missing)
# --------------------
try:
    import requests
except Exception:
    requests = None

# Wikipedia
_WIKI_API = None
try:
    import wikipediaapi  # pip install wikipedia-api
    _WIKI_API = wikipediaapi.Wikipedia
except Exception:
    pass

# arXiv
_ARXIV = None
try:
    import arxiv  # pip install arxiv
    _ARXIV = arxiv
except Exception:
    pass

# HTML→text
_TRAFILATURA = None
try:
    import trafilatura  # pip install trafilatura
    _TRAFILATURA = trafilatura
except Exception:
    pass

# HTML parsing fallback
_BS4 = None
try:
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
    _BS4 = BeautifulSoup
except Exception:
    pass

# PDFs
_PYPDF = None
try:
    import pypdf  # pip install pypdf
    _PYPDF = pypdf
except Exception:
    pass

# HF datasets
_DATASETS = None
try:
    from datasets import load_dataset  # pip install datasets
    _DATASETS = load_dataset
except Exception:
    pass

# Lang detect
_LANGDETECT = None
try:
    from langdetect import detect  # pip install langdetect
    _LANGDETECT = detect
except Exception:
    pass

# REBEL
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    torch = None

# spaCy
try:
    import spacy
    _SPACY = None
except Exception:
    spacy = None
    _SPACY = None

# NLI verifier
try:
    from transformers import AutoModelForSequenceClassification as HFClsModel
except Exception:
    HFClsModel = None

NLI_MODEL_ID = "microsoft/deberta-large-mnli"
_NLI_TOKENIZER = None
_NLI_MODEL = None

def _load_nli():
    global _NLI_TOKENIZER, _NLI_MODEL
    if HFClsModel is None:
        return None, None
    if _NLI_MODEL is None:
        _NLI_TOKENIZER = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
        _NLI_MODEL = HFClsModel.from_pretrained(NLI_MODEL_ID)
    return _NLI_TOKENIZER, _NLI_MODEL

def triple_to_hypothesis(h: str, r: str, t: str) -> str:
    templates = {
        "instance of": f"{h} is an instance of {t}.",
        "subclass of": f"{h} is a subclass of {t}.",
        "owned by": f"{h} is owned by {t}.",
        "maintained by": f"{h} is maintained by {t}.",
        "part of": f"{h} is part of {t}.",
        "used by": f"{h} is used by {t}.",
        "uses": f"{h} uses {t}.",
        "announced in": f"{h} was announced in {t}.",
        "acquired by": f"{h} was acquired by {t}.",
        "integrates": f"{h} integrates {t}.",
        "supports": f"{h} supports {t}.",
        "validates": f"{h} validates {t}.",
        "queries": f"{h} queries {t}.",
        "published as": f"{h} was published as {t}.",
    }
    return templates.get(r, f"{h} {r} {t}.")

def nli_entails(premise: str, hypothesis: str, tokenizer, model, device: Optional[str]) -> float:
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    if device:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits[0], dim=-1).tolist()  # [contradiction, neutral, entailment]
    return float(probs[2])

def best_support_sentence(h: str, t: str, sentences: List[str]) -> Optional[str]:
    hl, tl = h.lower(), t.lower()
    cands = [s for s in sentences if hl in s.lower() and tl in s.lower()]
    if not cands:
        cands = [s for s in sentences if hl in s.lower() or tl in s.lower()]
    return max(cands, key=len) if cands else None

# --------------------
# Text utilities and canonicalization
# --------------------

PRONOUNS = {
    "i","you","he","she","it","we","they","me","him","her","us","them",
    "my","your","his","her","its","our","their","mine","yours","hers","ours","theirs",
}

def clean_ws(s: str) -> str:
    s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def text_quality_ok(s: str, lang: Optional[str], min_chars: int, max_chars: int) -> bool:
    if not s or len(s) < min_chars or len(s) > max_chars:
        return False
    if lang and _LANGDETECT is not None:
        try:
            detected = _LANGDETECT(s[:2000])
            if detected.lower() != lang.lower():
                return False
        except Exception:
            pass
    bad_ratio = sum(1 for c in s if c in "{}[]<>|") / max(1, len(s))
    if bad_ratio > 0.05:
        return False
    return True

def canonical_entity(s: str) -> str:
    s = clean_ws(s.strip("\"'` "))
    s = re.sub(r"[，,.;:!?]+$", "", s)
    s = re.sub(r"^\((.*?)\)$", r"\1", s).strip()
    if s.lower() in PRONOUNS:
        return ""
    if len(s) <= 2 or s.isupper():
        return s
    words = s.split()
    if any(w[:1].isupper() for w in words):
        return s
    return " ".join(w.capitalize() if len(w) > 2 else w for w in words)

def canonical_relation(s: str) -> str:
    s = clean_ws(s).lower()
    s = re.sub(r"^(is|was|are|were|be|been|being)\s+", "", s)
    return s if len(s) >= 2 else ""

# --------------------
# Robust spaCy sentence splitter (no duplicate sentencizer)
# --------------------

def _get_spacy_nlp():
    global _SPACY
    if spacy is None:
        return None
    if _SPACY is None:
        try:
            _SPACY = spacy.load("en_core_web_sm")
        except Exception:
            _SPACY = spacy.blank("en")
        # Ensure sentence segmentation exists, but don't add twice
        names = set(_SPACY.pipe_names)
        if ("senter" not in names) and ("parser" not in names) and ("sentencizer" not in names):
            _SPACY.add_pipe("sentencizer")
    return _SPACY

def split_sentences_spacy(text: str) -> List[str]:
    if spacy is None:
        return re.split(r"(?<=[.!?])\s+", text.strip())
    nlp = _get_spacy_nlp()
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]

# --------------------
# REBEL parsing / extraction
# --------------------

def parse_rebel_output(text: str) -> List[Tuple[str, str, str]]:
    """
    REBEL linearization:
      <triplet> SUBJECT <subj> OBJECT <obj> RELATION
    """
    if not text:
        return []
    toks = text.replace("<s>", " ").replace("</s>", " ").replace("<pad>", " ").split()
    triples: List[Tuple[str, str, str]] = []
    subj, obj, rel = "", "", ""
    state = None  # "subject" | "object" | "relation"

    def flush():
        nonlocal subj, obj, rel
        hs, os, rs = subj.strip(), obj.strip(), rel.strip()
        if hs and os and rs:
            h = canonical_entity(hs)
            t = canonical_entity(os)
            r = canonical_relation(rs)
            if h and t and r and h != t and h.lower() not in PRONOUNS and t.lower() not in PRONOUNS:
                if len(h) <= 200 and len(t) <= 200 and len(r) <= 120:
                    triples.append((h, r, t))
        subj, obj, rel = "", "", ""

    for tok in toks:
        if tok == "<triplet>":
            flush()
            state = "subject"
        elif tok == "<subj>":
            state = "object"
        elif tok == "<obj>":
            state = "relation"
        else:
            if state == "subject":
                subj += (" " if subj else "") + tok
            elif state == "object":
                obj += (" " if obj else "") + tok
            elif state == "relation":
                rel += (" " if rel else "") + tok
    flush()
    return triples

def chunk_sentences(text: str, max_chars: int = 1200) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if cur_len + len(s) + 1 > max_chars and cur:
            chunks.append(" ".join(cur))
            cur, cur_len = [s], len(s)
        else:
            cur.append(s)
            cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def rebel_extract_triples_sentence_level(
    text: str, tokenizer, model, device: Optional[str], max_new_tokens: int, num_beams: int
) -> List[Tuple[str,str,str]]:
    triples: Set[Tuple[str,str,str]] = set()
    for sent in split_sentences_spacy(text):
        if len(sent) < 40:
            continue
        inputs = tokenizer(sent, return_tensors="pt", truncation=True)
        if device:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=max_new_tokens, num_beams=num_beams,
                length_penalty=0.0, early_stopping=True
            )
        for seq in tokenizer.batch_decode(gen, skip_special_tokens=False):
            for h,r,t in parse_rebel_output(seq):
                triples.add((h,r,t))
    return list(triples)

def rebel_extract_triples(
    text: str, tokenizer, model, device: Optional[str], max_new_tokens: int, num_beams: int
) -> List[Tuple[str, str, str]]:
    triples: Set[Tuple[str, str, str]] = set()
    for chunk in chunk_sentences(text):
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True)
        if device:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=max_new_tokens, num_beams=num_beams,
                length_penalty=0.0, early_stopping=True
            )
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=False)
        for seq in decoded:
            triples.update(parse_rebel_output(seq))
    return list(triples)

# --------------------
# Relation normalization
# --------------------

CANON_REL_MAP = {
    "instance of": {"is a", "is an", "isa", "type of", "kind of"},
    "subclass of": {"subclass of", "is a subclass of", "is a type of"},
    "owned by": {"owned by", "maintained by", "operated by"},
    "maintained by": {"maintained by", "run by"},
    "part of": {"part of", "component of", "subset of"},
    "used by": {"used by", "adopted by"},
    "uses": {"uses", "use", "using", "based on"},
    "announced in": {"announced in", "launched in", "introduced in", "appeared in"},
    "acquired by": {"acquired by", "bought by"},
    "integrates": {"integrates", "combines", "links"},
    "supports": {"supports", "implements"},
    "validates": {"validates", "constrains"},
    "queries": {"queries", "retrieves from"},
    "published as": {"published as", "published in"},
}
CANON_ALLOW = set(CANON_REL_MAP.keys())

def normalize_relation_label(r: str) -> str:
    r = canonical_relation(r)
    for canon, syns in CANON_REL_MAP.items():
        if r == canon or r in syns:
            return canon
    return ""

# --------------------
# Fallback NER-only entities
# --------------------

def spacy_fallback_entities(text: str) -> List[str]:
    if spacy is None:
        return []
    nlp = _get_spacy_nlp()
    doc = nlp(text)
    ents = set()
    for e in doc.ents:
        ce = canonical_entity(e.text)
        if ce:
            ents.add(ce)
    return sorted(ents)

# --------------------
# Write KG
# --------------------

def write_kg(kg_path: str, triples: Iterable[Tuple[str, str, str]]) -> None:
    with open(kg_path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h.replace('\t',' ').strip()}\t{r.replace('\t',' ').strip()}\t{t.replace('\t',' ').strip()}\n")

# --------------------
# Ingestors
# --------------------

def ingest_wikipedia(spec: dict, filters: dict, sink: Dict[str, str], seen: Set[str]) -> None:
    titles = spec.get("titles") or []
    lang = spec.get("lang", "en")
    if not _WIKI_API:
        raise RuntimeError("Missing dependency: wikipedia-api")
    api = _WIKI_API(language=lang, extract_format=wikipediaapi.ExtractFormat.WIKI)
    for title in titles:
        page = api.page(title)
        if not page.exists():
            continue
        text = clean_ws(page.text)
        if not text_quality_ok(text, filters.get("lang"), filters.get("min_chars", 0), filters.get("max_chars", 10**9)):
            continue
        doc_id = f"wikipedia:{lang}:{title.replace(' ', '_')}"
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        sink[doc_id] = text

def _download(url: str) -> Optional[str]:
    if requests is None:
        return None
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        return r.text
    except Exception:
        return None

def _html_to_text(html: str) -> str:
    if _TRAFILATURA:
        try:
            extracted = _TRAFILATURA.extract(html, include_comments=False, include_tables=False)
            if extracted:
                return clean_ws(extracted)
        except Exception:
            pass
    if _BS4:
        try:
            soup = _BS4(html, "html.parser")
            for s in soup(["script","style","noscript"]):
                s.decompose()
            return clean_ws(soup.get_text(" "))
        except Exception:
            pass
    return ""

def ingest_urls_file(spec: dict, filters: dict, sink: Dict[str, str], seen: Set[str]) -> None:
    path = spec.get("path")
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if not url:
                continue
            html = _download(url)
            if not html:
                continue
            text = _html_to_text(html)
            if not text_quality_ok(text, filters.get("lang"), filters.get("min_chars", 0), filters.get("max_chars", 10**9)):
                continue
            doc_id = f"url:{url}"
            h = hashlib.md5(text.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            sink[doc_id] = text

def ingest_arxiv(spec: dict, filters: dict, sink: Dict[str, str], seen: Set[str]) -> None:
    if _ARXIV is None:
        raise RuntimeError("Missing dependency: arxiv")
    query = spec.get("query", "")
    max_results = int(spec.get("max_results", 50))
    download_pdf = bool(spec.get("download_pdf", False))
    search = _ARXIV.Search(query=query, max_results=max_results, sort_by=_ARXIV.SortCriterion.Relevance)
    for r in search.results():
        doc_id = f"arxiv:{r.get_short_id()}"
        text_parts = [r.title or "", r.summary or ""]
        text = clean_ws("\n\n".join([p for p in text_parts if p]))
        if download_pdf and _PYPDF is not None and requests is not None and r.pdf_url:
            try:
                pdf_resp = requests.get(r.pdf_url, timeout=25)
                if pdf_resp.status_code == 200:
                    with io.BytesIO(pdf_resp.content) as bio:
                        reader = _PYPDF.PdfReader(bio)
                        pages = min(len(reader.pages), 6)
                        pdf_text = " ".join(reader.pages[i].extract_text() or "" for i in range(pages))
                        text = clean_ws(text + "\n\n" + pdf_text)
            except Exception:
                pass
        if not text_quality_ok(text, filters.get("lang"), filters.get("min_chars", 0), filters.get("max_chars", 10**9)):
            continue
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        sink[doc_id] = text

def ingest_hf_dataset(spec: dict, filters: dict, sink: Dict[str, str], seen: Set[str]) -> None:
    if _DATASETS is None:
        raise RuntimeError("Missing dependency: datasets")
    name = spec["name"]
    split = spec.get("split", "train")
    config = spec.get("config", None)
    txt_field = spec.get("text_field", "text")
    limit = spec.get("limit", None)
    ds = _DATASETS(name, config, split=split, trust_remote_code=True) if config else _DATASETS(name, split=split, trust_remote_code=True)
    count = 0
    for i, ex in enumerate(ds):
        if txt_field not in ex:
            continue
        text = clean_ws(str(ex[txt_field]))
        if not text_quality_ok(text, filters.get("lang"), filters.get("min_chars", 0), filters.get("max_chars", 10**9)):
            continue
        doc_id = f"hf:{name}:{split}:{i}"
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        sink[doc_id] = text
        count += 1
        if limit and count >= int(limit):
            break

def _load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_pdf(path: str) -> str:
    if _PYPDF is None:
        return ""
    try:
        reader = _PYPDF.PdfReader(path)
        pages = min(len(reader.pages), 20)
        return " ".join(reader.pages[i].extract_text() or "" for i in range(pages))
    except Exception:
        return ""

def _strip_html_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        return _html_to_text(html)
    except Exception:
        return ""

def ingest_local_dir(spec: dict, filters: dict, sink: Dict[str, str], seen: Set[str]) -> None:
    import glob
    base = spec["path"]
    pattern = spec.get("glob", "**/*")
    paths = glob.glob(os.path.join(base, pattern), recursive=True)
    for p in paths:
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(p)[1].lower()
        text = ""
        if ext in {".txt", ".md"}:
            text = _load_text_file(p)
        elif ext in {".html", ".htm"}:
            text = _strip_html_file(p)
        elif ext in {".pdf"}:
            text = _read_pdf(p)
        else:
            continue
        text = clean_ws(text)
        if not text_quality_ok(text, filters.get("lang"), filters.get("min_chars", 0), filters.get("max_chars", 10**9)):
            continue
        doc_id = f"local:{os.path.relpath(p, base)}"
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        sink[doc_id] = text

# Extend canonical map with inverses and synonyms
CANON_REL_MAP.update({
    "owned by": {"owner of","owned by","maintained by","operated by","run by"},
    "part of": {"part of","has part"},
    "uses": {"uses","use","used by","adopted by"},
    "announced in": {"announced in","inception","introduced in","launched in","appeared in"},
})
INVERSE_NEEDS_SWAP = {"owner of","used by","has part"}

def normalize_and_orient(h: str, r: str, t: str) -> tuple[str,str,str] | None:
    r0 = canonical_relation(r)
    canon = None
    for c, syns in CANON_REL_MAP.items():
        if r0 == c or r0 in syns:
            canon = c
            break
    if not canon:
        return None  # drop non-canonical like 'facet of', 'studied by', etc.
    if r0 in INVERSE_NEEDS_SWAP:
        h, t = t, h  # flip direction
    if canon == "uses":
        canon = "uses"
    return h, canon, t


INGESTORS = {
    "wikipedia": ingest_wikipedia,
    "arxiv": ingest_arxiv,
    "urls_file": ingest_urls_file,
    "hf_dataset": ingest_hf_dataset,
    "local_dir": ingest_local_dir,
}

# --------------------
# Main
# --------------------

def main():
    ap = argparse.ArgumentParser(description="Acquire raw text from flexible sources, then build GFM-RAG stage-1 KG files.")
    ap.add_argument("--data-root", required=True, help="Output dataset root directory to create/populate.")
    ap.add_argument("--config", required=True, help="JSON config with 'sources' and optional 'filters'.")
    ap.add_argument("--rebel-model", default="Babelscape/rebel-large", help="HF model id for REBEL.")
    ap.add_argument("--device", default=None, help="e.g., cuda:0 or cpu. Auto if omitted.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--beam-size", type=int, default=4)
    ap.add_argument("--write-jsonl", action="store_true", help="Write processed/stage1/triples.jsonl for auditing.")
    ap.add_argument("--nli-verify", action="store_true", help="Filter triples with MNLI entailment.")
    ap.add_argument("--nli-threshold", type=float, default=0.8, help="Entailment threshold.")
    ap.add_argument("--strict-relations", action="store_true", help="Keep only triples whose relations map to the canonical schema.")
    args = ap.parse_args()

    raw_dir = os.path.join(args.data_root, "raw")
    out_dir = os.path.join(args.data_root, "processed", "stage1")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sources = cfg.get("sources", [])
    filters = cfg.get("filters", {})
    filters.setdefault("lang", None)
    filters.setdefault("min_chars", 400)
    filters.setdefault("max_chars", 250000)

    corpus: Dict[str, str] = {}
    seen_hashes: Set[str] = set()
    for spec in sources:
        t = spec.get("type")
        if t not in INGESTORS:
            raise ValueError(f"Unknown source type: {t}")
        INGESTORS[t](spec, filters, corpus, seen_hashes)

    corpus_path = os.path.join(raw_dir, "dataset_corpus.json")
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    if AutoTokenizer is None or AutoModelForSeq2SeqLM is None or torch is None:
        raise RuntimeError("Install transformers and torch for REBEL.")
    tokenizer = AutoTokenizer.from_pretrained(args.rebel_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.rebel_model)
    device = args.device
    if device is None:
        device = "cuda:0" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
    model = model.to(device)
    model.eval()

    nli_tok = None
    nli_mod = None
    if args.nli_verify:
        nli_tok, nli_mod = _load_nli()
        if nli_mod is not None:
            nli_mod = nli_mod.to(device).eval()

    extractor = rebel_extract_triples_sentence_level

    all_triples: Set[Tuple[str, str, str]] = set()
    doc2ents: Dict[str, Set[str]] = {}
    jsonl_path = os.path.join(out_dir, "triples.jsonl") if args.write_jsonl else None
    jf = open(jsonl_path, "w", encoding="utf-8") if jsonl_path else None

    for doc_id, text in corpus.items():
        sentences = split_sentences_spacy(text)
        triples = extractor(
            text, tokenizer=tokenizer, model=model, device=device,
            max_new_tokens=args.max_new_tokens, num_beams=args.beam_size
        )

        cleaned = []
        # for h, r, t in triples:
        #     r_norm = normalize_relation_label(r) if args.strict_relations else canonical_relation(r)
        #     if args.strict_relations and not r_norm:
        #         continue
        #     if r_norm:
        #         r = r_norm
        #     if args.nli_verify and nli_tok and nli_mod:
        #         sup = best_support_sentence(h, t, sentences)
        #         if not sup:
        #             continue
        #         score = nli_entails(sup, triple_to_hypothesis(h, r, t), nli_tok, nli_mod, device)
        #         if score < args.nli_threshold:
        #             continue
        #     cleaned.append((h, r, t))
        for h, r, t in triples:
            if args.strict_relations:
                norm = normalize_and_orient(h, r, t)
                if norm is None:
                    continue
                h, r, t = norm
            else:
                r = canonical_relation(r)

            if args.nli_verify and nli_tok and nli_mod:
                sup = best_support_sentence(h, t, sentences)
                if not sup:
                    continue
                score = nli_entails(sup, triple_to_hypothesis(h, r, t), nli_tok, nli_mod, device)
                if score < args.nli_threshold:
                    continue

            cleaned.append((h, r, t))


        uniq: Set[Tuple[str, str, str]] = set()
        for h, r, t in cleaned:
            if (t, r, h) in uniq:
                continue
            uniq.add((h, r, t))

        ents = {h for h, _, t in uniq} | {t for _, _, t in uniq}
        for tri in uniq:
            all_triples.add(tri)
        if not ents:
            ents.update(spacy_fallback_entities(text))
        doc2ents[doc_id] = ents

        if jf:
            jf.write(json.dumps({
                "doc_id": doc_id,
                "triples": [{"h": h, "r": r, "t": t} for (h, r, t) in sorted(uniq)],
                "entities": sorted(ents)
            }, ensure_ascii=False) + "\n")

    if jf:
        jf.close()

    kg_path = os.path.join(out_dir, "kg.txt")
    write_kg(kg_path, sorted(all_triples))
    with open(os.path.join(out_dir, "document2entities.json"), "w", encoding="utf-8") as f:
        json.dump({k: sorted(v) for k, v in doc2ents.items()}, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {corpus_path}")
    print(f"Wrote: {kg_path}")
    print(f"Wrote: {os.path.join(out_dir, 'document2entities.json')}")

if __name__ == "__main__":
    main()

# !/usr/bin/env python
# build_and_index_gfmrag_kggen.py
# End-to-end pipeline:
#  1) Acquire raw text from flexible sources (Wikipedia, arXiv, URLs list, HF datasets, local files)
#  2) Write raw/dataset_corpus.json
#  3) Extract triples with KGGen (LM+clustering), normalize + verify (optional NLI)
#  4) Write processed/stage1/{kg.txt, document2entities.json, triples.jsonl(optional)}

# import argparse, json, os, re, io, hashlib, sys
# from typing import List, Tuple, Dict, Set, Iterable, Optional

# # --------------------
# # Optional deps
# # --------------------
# try:
#     import requests
# except Exception:
#     requests = None

# # Wikipedia
# _WIKI_API = None
# try:
#     import wikipediaapi  # pip install wikipedia-api
#     _WIKI_API = wikipediaapi.Wikipedia
# except Exception:
#     pass

# # arXiv
# _ARXIV = None
# try:
#     import arxiv  # pip install arxiv
#     _ARXIV = arxiv
# except Exception:
#     pass

# # HTML→text
# _TRAFILATURA = None
# try:
#     import trafilatura  # pip install trafilatura
#     _TRAFILATURA = trafilatura
# except Exception:
#     pass

# # HTML parsing fallback
# _BS4 = None
# try:
#     from bs4 import BeautifulSoup  # pip install beautifulsoup4
#     _BS4 = BeautifulSoup
# except Exception:
#     pass

# # PDFs
# _PYPDF = None
# try:
#     import pypdf  # pip install pypdf
#     _PYPDF = pypdf
# except Exception:
#     pass

# # HF datasets
# _DATASETS = None
# try:
#     from datasets import load_dataset  # pip install datasets
#     _DATASETS = load_dataset
# except Exception:
#     pass

# # Lang detect
# _LANGDETECT = None
# try:
#     from langdetect import detect  # pip install langdetect
#     _LANGDETECT = detect
# except Exception:
#     pass

# # spaCy splitter + NER fallback
# try:
#     import spacy
#     _SPACY = None
# except Exception:
#     spacy = None
#     _SPACY = None

# # NLI verifier
# try:
#     import torch
#     from transformers import AutoTokenizer, AutoModelForSequenceClassification as HFClsModel
# except Exception:
#     torch = None
#     AutoTokenizer = None
#     HFClsModel = None

# # KGGen
# try:
#     from kg_gen import KGGen  # pip install kg-gen
# except Exception:
#     KGGen = None

# # --------------------
# # NLI
# # --------------------
# # NLI_MODEL_ID = "microsoft/deberta-v3-large"
# _NLI_TOKENIZER = None
# _NLI_MODEL = None

# def _load_nli(device: Optional[str], model_id: str):
#     global _NLI_TOKENIZER, _NLI_MODEL
#     if HFClsModel is None or AutoTokenizer is None or torch is None:
#         return None, None
#     if _NLI_MODEL is not None:
#         return _NLI_TOKENIZER, _NLI_MODEL

#     tok_kwargs = {}
#     # Force slow tokenizer for DeBERTa to avoid tiktoken converter path.
#     if "deberta" in model_id.lower():
#         tok_kwargs["use_fast"] = False
#     try:
#         _NLI_TOKENIZER = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
#         _NLI_MODEL = HFClsModel.from_pretrained(model_id)
#     except Exception:
#         # Fallback to roberta-large-mnli which does not require sentencepiece/tiktoken conversion.
#         fallback_id = "roberta-large-mnli"
#         _NLI_TOKENIZER = AutoTokenizer.from_pretrained(fallback_id)
#         _NLI_MODEL = HFClsModel.from_pretrained(fallback_id)
#         model_id = fallback_id

#     if device:
#         _NLI_MODEL = _NLI_MODEL.to(device)
#     _NLI_MODEL.eval()
#     return _NLI_TOKENIZER, _NLI_MODEL


# def triple_to_hypothesis(h: str, r: str, t: str) -> str:
#     templates = {
#         "instance of": f"{h} is an instance of {t}.",
#         "subclass of": f"{h} is a subclass of {t}.",
#         "owned by": f"{h} is owned by {t}.",
#         "maintained by": f"{h} is maintained by {t}.",
#         "part of": f"{h} is part of {t}.",
#         "uses": f"{h} uses {t}.",
#         "announced in": f"{h} was announced in {t}.",
#         "acquired by": f"{h} was acquired by {t}.",
#         "integrates": f"{h} integrates {t}.",
#         "supports": f"{h} supports {t}.",
#         "validates": f"{h} validates {t}.",
#         "queries": f"{h} queries {t}.",
#         "published as": f"{h} was published as {t}.",
#         "licensed under": f"{h} is licensed under {t}.",
#         "alias of": f"{h} is also known as {t}.",
#         "used for": f"{h} is used for {t}.",
#         "based on": f"{h} is based on {t}.",
#     }
#     return templates.get(r, f"{h} {r} {t}.")

# def nli_entails(premise: str, hypothesis: str, tokenizer, model, device: Optional[str]) -> float:
#     inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
#     if device:
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     probs = torch.softmax(logits[0], dim=-1).tolist()  # [contradiction, neutral, entailment]
#     return float(probs[2])

# def best_support_sentence(h: str, t: str, sentences: List[str]) -> Optional[str]:
#     hl, tl = h.lower(), t.lower()
#     cands = [s for s in sentences if hl in s.lower() and tl in s.lower()]
#     if not cands:
#         cands = [s for s in sentences if hl in s.lower() or tl in s.lower()]
#     return max(cands, key=len) if cands else None

# # --------------------
# # Text utils
# # --------------------
# PRONOUNS = {
#     "i","you","he","she","it","we","they","me","him","her","us","them",
#     "my","your","his","her","its","our","their","mine","yours","hers","ours","theirs",
# }

# def clean_ws(s: str) -> str:
#     s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
#     s = re.sub(r"\s+", " ", s).strip()
#     return s

# def text_quality_ok(s: str, lang: Optional[str], min_chars: int, max_chars: int) -> bool:
#     if not s or len(s) < min_chars or len(s) > max_chars:
#         return False
#     if lang and _LANGDETECT is not None:
#         try:
#             detected = _LANGDETECT(s[:2000])
#             if detected.lower() != lang.lower():
#                 return False
#         except Exception:
#             pass
#     bad_ratio = sum(1 for c in s if c in "{}[]<>|") / max(1, len(s))
#     if bad_ratio > 0.05:
#         return False
#     return True

# def canonical_entity(s: str) -> str:
#     s = clean_ws(s.strip("\"'` "))
#     s = re.sub(r"[，,.;:!?]+$", "", s)
#     s = re.sub(r"^\((.*?)\)$", r"\1", s).strip()
#     if s.lower() in PRONOUNS:
#         return ""
#     if len(s) <= 2 or s.isupper():
#         return s
#     words = s.split()
#     if any(w[:1].isupper() for w in words):
#         return s
#     return " ".join(w.capitalize() if len(w) > 2 else w for w in words)

# def canonical_relation(s: str) -> str:
#     s = clean_ws(s).lower()
#     s = re.sub(r"^(is|was|are|were|be|been|being)\s+", "", s)
#     return s if len(s) >= 2 else ""

# # --------------------
# # spaCy splitter/NER fallback
# # --------------------
# def _get_spacy_nlp():
#     global _SPACY
#     if spacy is None:
#         return None
#     if _SPACY is None:
#         try:
#             _SPACY = spacy.load("en_core_web_sm")
#         except Exception:
#             _SPACY = spacy.blank("en")
#         names = set(_SPACY.pipe_names)
#         if ("senter" not in names) and ("parser" not in names) and ("sentencizer" not in names):
#             _SPACY.add_pipe("sentencizer")
#     return _SPACY

# def split_sentences_spacy(text: str) -> List[str]:
#     if spacy is None:
#         return re.split(r"(?<=[.!?])\s+", text.strip())
#     nlp = _get_spacy_nlp()
#     doc = nlp(text)
#     return [s.text.strip() for s in doc.sents if s.text.strip()]

# def spacy_fallback_entities(text: str) -> List[str]:
#     if spacy is None:
#         return []
#     nlp = _get_spacy_nlp()
#     doc = nlp(text)
#     ents = set()
#     for e in doc.ents:
#         ce = canonical_entity(e.text)
#         if ce:
#             ents.add(ce)
#     return sorted(ents)

# # --------------------
# # Relation normalization and inversion
# # --------------------
# CANON_REL_MAP = {
#     "instance of": {"is a", "is an", "isa", "type of", "kind of", "is type of"},
#     "subclass of": {"subclass of", "is a subclass of", "is a type of", "is subset of"},
#     "owned by": {"owned by", "maintained by", "operated by", "run by"},
#     "maintained by": {"maintained by", "run by"},
#     "part of": {"part of", "component of", "subset of", "has part"},
#     "uses": {"uses", "use", "using", "based on", "used by", "adopted by"},
#     "announced in": {"announced in", "launched in", "introduced in", "appeared in", "inception"},
#     "acquired by": {"acquired by", "bought by"},
#     "integrates": {"integrates", "combines", "links"},
#     "supports": {"supports", "implements"},
#     "validates": {"validates", "constrains"},
#     "queries": {"queries", "retrieves from"},
#     "published as": {"published as", "published in"},
#     "licensed under": {"licensed under", "license", "licensed"},
#     "alias of": {"alias of", "aka", "also known as", "same as"},
#     "used for": {"used for", "serves", "applied to"},
#     "based on": {"based on"},
# }
# INVERSE_NEEDS_SWAP = {"owner of", "used by", "has part"}

# def normalize_and_orient(h: str, r: str, t: str) -> Optional[Tuple[str,str,str]]:
#     r0 = canonical_relation(r)
#     canon = None
#     for c, syns in CANON_REL_MAP.items():
#         if r0 == c or r0 in syns:
#             canon = c
#             break
#     if not canon:
#         return None
#     if r0 in INVERSE_NEEDS_SWAP:
#         h, t = t, h
#     return h, canon, t

# # --------------------
# # Write KG
# # --------------------
# def write_kg(kg_path: str, triples: Iterable[Tuple[str, str, str]]) -> None:
#     with open(kg_path, "w", encoding="utf-8") as f:
#         for h, r, t in triples:
#             f.write(f"{h.replace('\t',' ').strip()}\t{r.replace('\t',' ').strip()}\t{t.replace('\t',' ').strip()}\n")

# # --------------------
# # Ingestors
# # --------------------
# def ingest_wikipedia(spec: dict, filters: dict, sink: Dict[str, str], seen: Set[str]) -> None:
#     titles = spec.get("titles") or []
#     lang = spec.get("lang", "en")
#     if not _WIKI_API:
#         raise RuntimeError("Missing dependency: wikipedia-api")
#     api = _WIKI_API(language=lang, extract_format=wikipediaapi.ExtractFormat.WIKI)
#     for title in titles:
#         page = api.page(title)
#         if not page.exists():
#             continue
#         text = clean_ws(page.text)
#         if not text_quality_ok(text, filters.get("lang"), filters.get("min_chars", 0), filters.get("max_chars", 10**9)):
#             continue
#         doc_id = f"wikipedia:{lang}:{title.replace(' ', '_')}"
#         h = hashlib.md5(text.encode("utf-8")).hexdigest()
#         if h in seen:
#             continue
#         seen.add(h)
#         sink[doc_id] = text

# def _download(url: str) -> Optional[str]:
#     if requests is None:
#         return None
#     try:
#         r = requests.get(url, timeout=20)
#         if r.status_code != 200:
#             return None
#         return r.text
#     except Exception:
#         return None

# def _html_to_text(html: str) -> str:
#     if _TRAFILATURA:
#         try:
#             extracted = _TRAFILATURA.extract(html, include_comments=False, include_tables=False)
#             if extracted:
#                 return clean_ws(extracted)
#         except Exception:
#             pass
#     if _BS4:
#         try:
#             soup = _BS4(html, "html.parser")
#             for s in soup(["script","style","noscript"]):
#                 s.decompose()
#             return clean_ws(soup.get_text(" "))
#         except Exception:
#             pass
#     return ""

# def ingest_urls_file(spec: dict, filters: dict, sink: Dict[str, str], seen: Set[str]) -> None:
#     path = spec.get("path")
#     if not path or not os.path.exists(path):
#         return
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             url = line.strip()
#             if not url:
#                 continue
#             html = _download(url)
#             if not html:
#                 continue
#             text = _html_to_text(html)
#             if not text_quality_ok(text, filters.get("lang"), filters.get("min_chars", 0), filters.get("max_chars", 10**9)):
#                 continue
#             doc_id = f"url:{url}"
#             h = hashlib.md5(text.encode("utf-8")).hexdigest()
#             if h in seen:
#                 continue
#             seen.add(h)
#             sink[doc_id] = text

# def ingest_arxiv(spec: dict, filters: dict, sink: Dict[str, str], seen: Set[str]) -> None:
#     if _ARXIV is None:
#         raise RuntimeError("Missing dependency: arxiv")
#     query = spec.get("query", "")
#     max_results = int(spec.get("max_results", 50))
#     download_pdf = bool(spec.get("download_pdf", False))
#     search = _ARXIV.Search(query=query, max_results=max_results, sort_by=_ARXIV.SortCriterion.Relevance)
#     for r in search.results():
#         doc_id = f"arxiv:{r.get_short_id()}"
#         text_parts = [r.title or "", r.summary or ""]
#         text = clean_ws("\n\n".join([p for p in text_parts if p]))
#         if download_pdf and _PYPDF is not None and requests is not None and r.pdf_url:
#             try:
#                 pdf_resp = requests.get(r.pdf_url, timeout=25)
#                 if pdf_resp.status_code == 200:
#                     with io.BytesIO(pdf_resp.content) as bio:
#                         reader = _PYPDF.PdfReader(bio)
#                         pages = min(len(reader.pages), 6)
#                         pdf_text = " ".join(reader.pages[i].extract_text() or "" for i in range(pages))
#                         text = clean_ws(text + "\n\n" + pdf_text)
#             except Exception:
#                 pass
#         if not text_quality_ok(text, filters.get("lang"), filters.get("min_chars", 0), filters.get("max_chars", 10**9)):
#             continue
#         h = hashlib.md5(text.encode("utf-8")).hexdigest()
#         if h in seen:
#             continue
#         seen.add(h)
#         sink[doc_id] = text

# def ingest_hf_dataset(spec: dict, filters: dict, sink: Dict[str, str], seen: Set[str]) -> None:
#     if _DATASETS is None:
#         raise RuntimeError("Missing dependency: datasets")
#     name = spec["name"]
#     split = spec.get("split", "train")
#     config = spec.get("config", None)
#     txt_field = spec.get("text_field", "text")
#     limit = spec.get("limit", None)
#     ds = _DATASETS(name, config, split=split) if config else _DATASETS(name, split=split)
#     count = 0
#     for i, ex in enumerate(ds):
#         if txt_field not in ex:
#             continue
#         text = clean_ws(str(ex[txt_field]))
#         if not text_quality_ok(text, filters.get("lang"), filters.get("min_chars", 0), filters.get("max_chars", 10**9)):
#             continue
#         doc_id = f"hf:{name}:{split}:{i}"
#         h = hashlib.md5(text.encode("utf-8")).hexdigest()
#         if h in seen:
#             continue
#         seen.add(h)
#         sink[doc_id] = text
#         count += 1
#         if limit and count >= int(limit):
#             break

# def _load_text_file(path: str) -> str:
#     with open(path, "r", encoding="utf-8", errors="ignore") as f:
#         return f.read()

# def _read_pdf(path: str) -> str:
#     if _PYPDF is None:
#         return ""
#     try:
#         reader = _PYPDF.PdfReader(path)
#         pages = min(len(reader.pages), 20)
#         return " ".join(reader.pages[i].extract_text() or "" for i in range(pages))
#     except Exception:
#         return ""

# def _strip_html_file(path: str) -> str:
#     try:
#         with open(path, "r", encoding="utf-8", errors="ignore") as f:
#             html = f.read()
#         return _html_to_text(html)
#     except Exception:
#         return ""

# def ingest_local_dir(spec: dict, filters: dict, sink: Dict[str, str], seen: Set[str]) -> None:
#     import glob
#     base = spec["path"]
#     pattern = spec.get("glob", "**/*")
#     paths = glob.glob(os.path.join(base, pattern), recursive=True)
#     for p in paths:
#         if not os.path.isfile(p):
#             continue
#         ext = os.path.splitext(p)[1].lower()
#         text = ""
#         if ext in {".txt", ".md"}:
#             text = _load_text_file(p)
#         elif ext in {".html", ".htm"}:
#             text = _strip_html_file(p)
#         elif ext in {".pdf"}:
#             text = _read_pdf(p)
#         else:
#             continue
#         text = clean_ws(text)
#         if not text_quality_ok(text, filters.get("lang"), filters.get("min_chars", 0), filters.get("max_chars", 10**9)):
#             continue
#         doc_id = f"local:{os.path.relpath(p, base)}"
#         h = hashlib.md5(text.encode("utf-8")).hexdigest()
#         if h in seen:
#             continue
#         seen.add(h)
#         sink[doc_id] = text

# INGESTORS = {
#     "wikipedia": ingest_wikipedia,
#     "arxiv": ingest_arxiv,
#     "urls_file": ingest_urls_file,
#     "hf_dataset": ingest_hf_dataset,
#     "local_dir": ingest_local_dir,
# }

# # --------------------
# # KGGen extraction
# # --------------------
# def kggen_extract(text: str, kg: "KGGen", chunk_size: Optional[int], do_cluster: bool, context: Optional[str]) -> List[Tuple[str,str,str]]:
#     """
#     Returns list of (h, r, t) from KGGen graph object.
#     KGGen API: KGGen.generate(input_data, chunk_size=?, cluster=?, context=?). Relations live in graph.relations. Entities in graph.entities.  (README).  # noqa
#     """
#     graph = kg.generate(input_data=text, chunk_size=chunk_size, cluster=do_cluster, context=context or "")
#     triples = []
#     # Robustly read relations whether set of tuples or list of dicts
#     rels = getattr(graph, "relations", None) or graph.get("relations", None) if isinstance(graph, dict) else None
#     if rels is None:
#         # fallback: try edges if provided
#         rels = getattr(graph, "edges", None) or graph.get("edges", None) if isinstance(graph, dict) else None
#     if isinstance(rels, dict):
#         # sometimes { (h,r,t): score } or similar
#         rels = list(rels.keys())
#     for item in rels or []:
#         if isinstance(item, (tuple, list)) and len(item) == 3:
#             h, r, t = item
#         elif isinstance(item, dict) and {"head","relation","tail"} <= set(item.keys()):
#             h, r, t = item["head"], item["relation"], item["tail"]
#         else:
#             continue
#         h = canonical_entity(str(h))
#         r = canonical_relation(str(r))
#         t = canonical_entity(str(t))
#         if h and r and t and h != t and h.lower() not in PRONOUNS and t.lower() not in PRONOUNS:
#             triples.append((h, r, t))
#     return triples

# # --------------------
# # Main
# # --------------------
# def main():
#     ap = argparse.ArgumentParser(description="Acquire raw text, extract KG with KGGen, write GFM-RAG stage-1 files.")
#     ap.add_argument("--data-root", default="./data/kggen_smoke")
#     ap.add_argument("--config", default="./configs/smoke_real_kg.json", help="JSON with 'sources' and optional 'filters' + optional 'context' string.")
#     # KGGen params
#     ap.add_argument("--kggen-model", default="openai/gpt-4o", help="LiteLLM model string, e.g., 'ollama_chat/llama3.1:70b' or 'openai/gpt-4o'.")
#     ap.add_argument("--kggen-api-key", default=None, help="Optional; else taken from env for chosen provider.")
#     ap.add_argument("--kggen-base-url", default=None, help="Optional custom base_url for LiteLLM providers (e.g., local gateway).")
#     ap.add_argument("--kggen-temperature", type=float, default=0.0)
#     ap.add_argument("--kggen-chunk-size", type=int, default=5000)
#     ap.add_argument("--kggen-cluster", action="store_true")
#     # Filters and verification
#     ap.add_argument("--strict-relations", action="store_true", help="Whitelist + orientation flip.")
#     ap.add_argument("--nli-verify", action="store_true", help="MNLI gate.")
#     ap.add_argument("--nli-threshold", type=float, default=0.95)
#     ap.add_argument("--device", default=None)
#     ap.add_argument("--write-jsonl", action="store_true")
#     ap.add_argument("--nli-model", default="microsoft/deberta-v3-large",
#                 help="HF NLI model id. Use 'roberta-large-mnli' or 'microsoft/deberta-v3-large-mnli'.")

#     args = ap.parse_args()

#     if KGGen is None:
#         raise RuntimeError("Install kg-gen (pip install kg-gen).")

#     raw_dir = os.path.join(args.data_root, "raw")
#     out_dir = os.path.join(args.data_root, "processed", "stage1")
#     os.makedirs(raw_dir, exist_ok=True)
#     os.makedirs(out_dir, exist_ok=True)

#     with open(args.config, "r", encoding="utf-8") as f:
#         cfg = json.load(f)
#     sources = cfg.get("sources", [])
#     filters = cfg.get("filters", {})
#     context = cfg.get("context", "")
#     filters.setdefault("lang", None)
#     filters.setdefault("min_chars", 400)
#     filters.setdefault("max_chars", 250000)

#     corpus: Dict[str, str] = {}
#     seen_hashes: Set[str] = set()
#     for spec in sources:
#         t = spec.get("type")
#         if t not in INGESTORS:
#             raise ValueError(f"Unknown source type: {t}")
#         INGESTORS[t](spec, filters, corpus, seen_hashes)

#     corpus_path = os.path.join(raw_dir, "dataset_corpus.json")
#     with open(corpus_path, "w", encoding="utf-8") as f:
#         json.dump(corpus, f, ensure_ascii=False, indent=2)

#     # KGGen init
#     # README documents model routing via LiteLLM, chunking and clustering flags, and optional base_url. :contentReference[oaicite:0]{index=0}
#     kg_kwargs = dict(model=args.kggen-model if hasattr(args, "kggen-model") else args.kggen_model,
#                      temperature=args.kggen_temperature)
#     # Python identifiers cannot contain '-', fallback:
#     if "kggen-model" in vars(args):
#         kg_model = getattr(args, "kggen-model")
#     else:
#         kg_model = args.kggen_model
#     kg_kwargs = dict(model=kg_model, temperature=args.kggen_temperature)
#     if args.kggen_api_key:
#         kg_kwargs["api_key"] = args.kggen_api_key
#     if args.kggen_base_url:
#         kg_kwargs["base_url"] = args.kggen_base_url  # supported in tests/custom base example. :contentReference[oaicite:1]{index=1}
#     kg = KGGen(**kg_kwargs)

#     # NLI
#     device = args.device or ("cuda:0" if (torch and hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu")
#     nli_tok = nli_mod = None
#     if args.nli_verify:
#         # keep for global visibility if other functions read it
#         global NLI_MODEL_ID
#         NLI_MODEL_ID = args.nli_model
#         nli_tok, nli_mod = _load_nli(device, args.nli_model)

#     # Extract
#     all_triples: Set[Tuple[str, str, str]] = set()
#     doc2ents: Dict[str, Set[str]] = {}
#     jsonl_path = os.path.join(out_dir, "triples.jsonl") if args.write_jsonl else None
#     jf = open(jsonl_path, "w", encoding="utf-8") if jsonl_path else None

#     for doc_id, text in corpus.items():
#         sentences = split_sentences_spacy(text)
#         triples = kggen_extract(text, kg, args.kggen_chunk_size, args.kggen_cluster, context)

#         # normalize + verify
#         cleaned: List[Tuple[str,str,str]] = []
#         for h, r, t in triples:
#             if args.strict_relations:
#                 norm = normalize_and_orient(h, r, t)
#                 if norm is None:
#                     continue
#                 h, r, t = norm
#             else:
#                 r = canonical_relation(r)

#             if args.nli_verify and nli_tok and nli_mod:
#                 sup = best_support_sentence(h, t, sentences)
#                 if not sup:
#                     continue
#                 score = nli_entails(sup, triple_to_hypothesis(h, r, t), nli_tok, nli_mod, device)
#                 if score < args.nli_threshold:
#                     continue

#             cleaned.append((h, r, t))

#         # inverse duplicate prune
#         uniq: Set[Tuple[str, str, str]] = set()
#         for h, r, t in cleaned:
#             if (t, r, h) in uniq:
#                 continue
#             uniq.add((h, r, t))

#         ents = {h for h, _, t in uniq} | {t for _, _, t in uniq}
#         if not ents:
#             ents.update(spacy_fallback_entities(text))
#         doc2ents[doc_id] = ents
#         for tri in uniq:
#             all_triples.add(tri)

#         if jf:
#             jf.write(json.dumps({
#                 "doc_id": doc_id,
#                 "triples": [{"h": h, "r": r, "t": t} for (h, r, t) in sorted(uniq)],
#                 "entities": sorted(ents)
#             }, ensure_ascii=False) + "\n")

#     if jf:
#         jf.close()

#     # write outputs
#     out_dir_stage1 = out_dir
#     os.makedirs(out_dir_stage1, exist_ok=True)
#     kg_path = os.path.join(out_dir_stage1, "kg.txt")
#     write_kg(kg_path, sorted(all_triples))
#     with open(os.path.join(out_dir_stage1, "document2entities.json"), "w", encoding="utf-8") as f:
#         json.dump({k: sorted(v) for k, v in doc2ents.items()}, f, ensure_ascii=False, indent=2)

#     print(f"Wrote: {corpus_path}")
#     print(f"Wrote: {kg_path}")
#     print(f"Wrote: {os.path.join(out_dir_stage1, 'document2entities.json')}")

# if __name__ == "__main__":
#     main()
