import pdfplumber
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# Load a real transformer model (make sure to install sentence-transformers)
model = SentenceTransformer("all-MiniLM-L6-v2")

def is_heading(text, dist_from_mean=None, font_size=None, threshold=0.25):
    """
    Heuristic: A heading is detected if the distance from mean is above threshold
    and/or font size is large. You can adjust logic as needed.
    """
    if font_size is not None and font_size > 14:
        return True
    if dist_from_mean is not None and dist_from_mean > threshold:
        return True
    return False

def extract_pdf_structure(pdf_path):
    result = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extract text blocks
            for block in page.extract_words():
                result.append({"type": "text", "content": block["text"], "page": page_num + 1})

            # Extract tables
            for table in page.extract_tables():
                result.append({"type": "table", "content": table, "page": page_num + 1})

    # Extract images and links using PyMuPDF
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Images
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            result.append({"type": "image", "content": base_image, "page": page_num + 1})
        # Links
        for link in page.get_links():
            result.append({"type": "link", "content": link, "page": page_num + 1})

    return result