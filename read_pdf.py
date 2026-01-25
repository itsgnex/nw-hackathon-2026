# read_pdf.py
from pypdf import PdfReader
import tempfile
import os

def extract_text_from_pdf_file(file_storage) -> str:
    """
    Extracts all text content from an uploaded PDF.
    Since we're receiving a FileStorage object from Flask, we save it locally 
    before parsing, then clean up the temp file.
    """
    temp_path = None
    text = ""

    try:
        # 1. Create a safe temporary file to store the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            temp_path = tmp.name
            file_storage.save(temp_path)

        # 2. Parse the PDF and loop through all available pages
        reader = PdfReader(temp_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                # Build the full text string
                text += page_text + "\n"

    finally:
        # 3. Always delete the temp file to avoid cluttering the server
        if temp_path and os.getenv("KEEP_TEMP_FILES") != "1": 
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return text.strip()
