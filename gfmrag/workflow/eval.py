# eval_kg.py
import argparse, json, os, re, random, math
from collections import Counter, defaultdict

# Optional deps
try:
    import spacy
except Exception:
    spacy = None
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

ALLOW = {
    "instance of","subclass of","uses","used by","part of","owned by","maintained by",
    "announced in","acquired by","integrates","supports","validates","queries","published as"
}
# Direction normalizer: map passive to active canonical
def normalize_rel(h, r, t):
    rl = r.strip().lower()
    if rl in {"has part"}:
        return (t, "part of", h)
    if rl in {"used by","adopted by"}:
        return (t, "uses", h)
    if rl in {"inception","launched in","introduced in","appeared in"}:
        return (h, "announced in", t)
    return (h, rl, t)

def load_sent_splitter():
    if spacy is None:
        return None
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
    if ("senter" not in nlp.pipe_names) and ("parser" not in nlp.pipe_names) and ("sentencizer" not in nlp.pipe_names):
        nlp.add_pipe("sentencizer")
    return nlp

def best_support_sentence(h, t, sents):
    hl, tl = h.lower(), t.lower()
    cands = [s for s in sents if hl in s.lower() and tl in s.lower()]
    if not cands:
        cands = [s for s in sents if hl in s.lower() or tl in s.lower()]
    return max(cands, key=len) if cands else None

def load_nli(device):
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
        return None, None
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    mdl = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(device).eval()
    return tok, mdl

def nli_entails(premise, hyp, tok, mdl, device):
    if tok is None or mdl is None:
        return float("nan")
    with torch.no_grad():
        inputs = tok(premise, hyp, return_tensors="pt", truncation=True, max_length=512).to(device)
        probs = torch.softmax(mdl(**inputs).logits[0], dim=-1).tolist()
    return float(probs[2])  # entailment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--nli", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    raw_path = os.path.join(args.data_root, "raw", "dataset_corpus.json")
    triples_path = os.path.join(args.data_root, "processed", "stage1", "triples.jsonl")

    with open(raw_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Collect triples
    all_tri = []
    with open(triples_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            doc_id = rec["doc_id"]
            text = corpus.get(doc_id, "")
            tri = [(doc_id, t["h"], t["r"].lower(), t["t"]) for t in rec["triples"]]
            all_tri.extend(tri)

    nlp = load_sent_splitter()
    device = args.device or ("cuda:0" if (torch and torch.cuda.is_available()) else "cpu")
    tok, mdl = load_nli(device) if args.nli else (None, None)

    # Metrics
    rel_raw = Counter([r for _,_,r,_ in all_tri])
    schema_ok = 0
    oriented = 0
    total = len(all_tri)
    nli_pass = 0
    sampled_rows = []

    # Precompute sentence splits
    sent_cache = {}
    def get_sents(doc_id):
        if doc_id in sent_cache:
            return sent_cache[doc_id]
        text = corpus.get(doc_id, "")
        if nlp:
            sents = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
        else:
            sents = re.split(r"(?<=[.!?])\s+", text.strip())
        sent_cache[doc_id] = sents
        return sents

    for doc_id, h, r, t in all_tri:
        H,R,T = normalize_rel(h, r, t)
        if R in {"has part","used by","inception"}:
            oriented += 1
        schema_ok += int(R in ALLOW)
        if args.nli:
            sents = get_sents(doc_id)
            sup = best_support_sentence(H, T, sents) or ""
            hyp = f"{H} {R} {T}."
            score = nli_entails(sup, hyp, tok, mdl, device)
            nli_pass += int(score >= args.threshold)
            if len(sampled_rows) < args.sample:
                sampled_rows.append({
                    "doc_id": doc_id, "h": H, "r": R, "t": T,
                    "support": sup[:400], "nli": round(score, 3)
                })

    print(f"Triples: {total}")
    print(f"Schema-compliant: {schema_ok} ({schema_ok/total:.2%})")
    if args.nli:
        print(f"NLI≥{args.threshold}: {nli_pass} ({nli_pass/total:.2%})")
    print("Top raw relations:", rel_raw.most_common(10))
    print(f"Oriented (passive→active) normalized: {oriented}")

    if sampled_rows:
        print("\nSAMPLE:")
        for r in sampled_rows:
            print(json.dumps(r, ensure_ascii=False))

if __name__ == "__main__":
    main()
