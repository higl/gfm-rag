import json
import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple

def chunk_text(text: str, max_words: int = 1200, overlap: int = 120) -> List[str]:
    """Chunk text into overlapping segments."""
    words = text.split()
    chunks = []
    i = 0
    
    while i < len(words):
        end_idx = min(len(words), i + max_words)
        chunk = " ".join(words[i:end_idx])
        chunks.append(chunk)
        
        # Move start position with overlap
        i = max(end_idx - overlap, end_idx)
        if end_idx >= len(words):
            break
    
    return chunks

def build_corpus(raw_documents: List[Tuple[str, str]], output_dir: str) -> None:
    """Build corpus with chunked documents."""
    os.makedirs(output_dir, exist_ok=True)
    
    corpus = {}
    
    for title, text in raw_documents:
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Chunk the document
        chunks = chunk_text(text)
        
        for chunk_idx, chunk in enumerate(chunks):
            # Create unique key for each chunk
            chunk_key = f"{title} ## {chunk_idx:03d}"
            # Prepend title to chunk content
            chunk_content = f"{title}\n{chunk}"
            corpus[chunk_key] = chunk_content
    
    # Save corpus
    corpus_path = os.path.join(output_dir, "dataset_corpus.json")
    with open(corpus_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    
    print(f"Built corpus with {len(corpus)} chunks at {corpus_path}")

def load_sample_data() -> List[Tuple[str, str]]:
    """Load sample documents - replace with your data loader."""
    # Example: HotpotQA-style data
    sample_docs = [
        (
            "Los Angeles Rams Logo History",
            "The Los Angeles Rams were the first NFL team to have a logo painted on their helmets. "
            "The original design was created by Fred Gehrke, who played halfback for the team from "
            "1945 to 1947. Gehrke, who had studied art, painted the now-famous ram's horn design on "
            "the team's helmets before the 1948 season. The design became iconic and has remained "
            "largely unchanged since its introduction. Gehrke later served as general manager of "
            "the Denver Broncos from 1977 to 1981."
        ),
        (
            "Chicago Cardinals History",
            "The Chicago Cardinals were a professional American football team based in Chicago. "
            "Founded in 1898, they were one of the oldest franchises in what would become the NFL. "
            "Many players moved between teams in the early days, including Fred Gehrke who played "
            "for the Chicago Cardinals before joining the Los Angeles Rams."
        )
    ]
    return sample_docs

def main():
    parser = argparse.ArgumentParser(description="Build raw corpus for KG construction")
    parser.add_argument("--data_name", default="hotpotqa_train_example", help="Dataset name")
    parser.add_argument("--data_root", default="data", help="Data root directory")
    parser.add_argument("--max_words", type=int, default=1200, help="Max words per chunk")
    parser.add_argument("--overlap", type=int, default=120, help="Overlap between chunks")
    
    args = parser.parse_args()
    
    # Load your documents here - replace load_sample_data() with your data loader
    documents = load_sample_data()
    
    # Build output directory
    output_dir = os.path.join(args.data_root, args.data_name, "raw")
    
    # Build corpus
    build_corpus(documents, output_dir)

if __name__ == "__main__":
    main()
