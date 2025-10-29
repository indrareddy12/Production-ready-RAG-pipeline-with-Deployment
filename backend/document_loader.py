import PyPDF2
import docx
import io

def extract_text(file_bytes, filename):
    """Extract text from PDFs, DOCX, and TXT files."""
    text = ""

    if filename.endswith(".pdf"):
        # Extract text from PDF
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

    elif filename.endswith(".docx"):
        # Extract text from DOCX
        doc = docx.Document(io.BytesIO(file_bytes)) 
        for para in doc.paragraphs:
            text += para.text + "\n"

    elif filename.endswith(".txt"):
        # Extract text from TXT
        text = file_bytes.decode("utf-8")

    return text