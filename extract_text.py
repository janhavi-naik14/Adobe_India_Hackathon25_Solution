import fitz  # PyMuPDF

def extract_blocks(pdf_path, margin=50, max_block_width=480):
    """
    Extracts text blocks from a PDF, skipping headers, footers, and wide blocks.
    Returns a list of dicts with text, size, font, page, and bbox.
    """
    doc = fitz.open(pdf_path)
    all_blocks = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    y0, y1 = span["bbox"][1], span["bbox"][3]
                    block_width = span["bbox"][2] - span["bbox"][0]
                    # Skip header/footer and full-width table blocks
                    if y0 < margin or y1 > page.rect.height - margin:
                        continue
                    if block_width > max_block_width:
                        continue

                    all_blocks.append({
                        "text": text,
                        "size": span.get("size"),
                        "font": span.get("font"),
                        "page": page_num,
                        "bbox": span.get("bbox")
                    })
    return all_blocks