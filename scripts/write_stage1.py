import os
import json
import argparse
from pathlib import Path
from gfmrag.kg_construction.utils import KG_DELIMITER

def write_stage1_files(tmp_dir: str, output_dir: str) -> None:
    """Convert passage_info.json to stage1 format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load passage info
    passage_info_path = os.path.join(tmp_dir, "passage_info.json")
    with open(passage_info_path, 'r', encoding='utf-8') as f:
        rows = json.load(f)
    
    print(f"Processing {len(rows)} passages...")
    
    # Write kg.txt - one triple per line
    kg_path = os.path.join(output_dir, "kg.txt")
    triple_count = 0
    with open(kg_path, 'w', encoding='utf-8') as f:
        for row in rows:
            triples = row.get("clean_triples", [])
            for head, relation, tail in triples:
                f.write(f"{head}{KG_DELIMITER}{relation}{KG_DELIMITER}{tail}\n")
                triple_count += 1
    
    print(f"Wrote {triple_count} triples to {kg_path}")
    
    # Write document2entities.json
    doc2entities = {}
    for row in rows:
        title = row["title"]
        entities = row.get("entities", [])
        # Remove duplicates and sort
        unique_entities = sorted(list(set(entities)))
        doc2entities[title] = unique_entities
    
    doc2entities_path = os.path.join(output_dir, "document2entities.json")
    with open(doc2entities_path, 'w', encoding='utf-8') as f:
        json.dump(doc2entities, f, ensure_ascii=False, indent=2)
    
    print(f"Wrote {len(doc2entities)} documents to {doc2entities_path}")
    
    # Create minimal train/test.json files if they don't exist
    for split in ("train", "test"):
        split_path = os.path.join(output_dir, f"{split}.json")
        if not os.path.exists(split_path):
            with open(split_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
            print(f"Created placeholder {split_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert KG constructor output to stage1 format")
    parser.add_argument("--tmp_dir", required=True, help="Temporary directory with passage_info.json")
    parser.add_argument("--out_dir", required=True, help="Output directory for stage1 files")
    
    args = parser.parse_args()
    
    if not os.path.exists(os.path.join(args.tmp_dir, "passage_info.json")):
        print(f"Error: passage_info.json not found in {args.tmp_dir}")
        return
    
    write_stage1_files(args.tmp_dir, args.out_dir)
    print("Stage1 files written successfully!")

if __name__ == "__main__":
    main()
