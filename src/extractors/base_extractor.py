import os
import fitz  # PyMuPDF
import docx

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(file_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text()
    return text

def extract_text_from_word(file_path):
    """Extract text from a Word file."""
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text(directory_path):
    """Extract text from all files in a directory."""
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.endswith(".pdf"):
            documents.append(extract_text_from_pdf(file_path))
        elif filename.endswith(".docx"):
            documents.append(extract_text_from_word(file_path))
    return documents
