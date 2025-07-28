import os, json
import numpy as np
from extract_text import extract_blocks
from nlp_analyzer import is_heading, model, extract_pdf_structure
from sklearn.metrics.pairwise import cosine_distances

def build_outline(pdf_path):
    blocks = extract_blocks(pdf_path)
    texts = [b["text"] for b in blocks]
    if not texts:
        return []

    embs = model.encode(texts)
    mean_emb = np.mean(embs, axis=0, keepdims=True)
    dists = cosine_distances(embs, mean_emb).flatten()

    outline = []
    first_heading_found = False

    for i, block in enumerate(blocks):
        text = block["text"]
        dist = dists[i]
        if is_heading(text, dist_from_mean=dist):
            level = "H1" if not first_heading_found else "H2"
            outline.append({
                "level": level,
                "text": text,
                "page": block["page"]
            })
            first_heading_found = True

    return outline

if __name__ == "__main__":
    input_dir = "input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            print(f"Processing: {filename}")
            pdf_path = os.path.join(input_dir, filename)
            outline = build_outline(pdf_path)
            structure = extract_pdf_structure(pdf_path)
            result = {
                "title": os.path.basename(pdf_path).replace(".pdf", ""),
                "outline": outline,
                "structure": structure
            }
            out_path = os.path.join(output_dir, filename.replace(".pdf", ".json"))
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"Saved output to: {out_path}")