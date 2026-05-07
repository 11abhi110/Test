from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
from pathlib import Path
import re
from collections import Counter
import spacy
from langdetect import detect

nlp = spacy.load("en_core_web_sm")


# --------------------------------------------------
# File parsing (DO NOT truncate here)
# --------------------------------------------------
def parse_file(file):
    suffix = Path(file.name).suffix.lower()

    try:
        if suffix == ".pdf":
            reader = PdfReader(file)
            text = ""

            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

        elif suffix == ".docx":
            doc = Document(file)
            text = "\n".join(p.text for p in doc.paragraphs)

        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(file)
            text = df.to_string(index=False)

        else:
            text = file.getvalue().decode("utf-8", errors="ignore")

        if not text.strip():
            return ""

    except Exception as e:
        return ""

    return text   # ✅ FULL DOCUMENT (no truncation)


# --------------------------------------------------
# Chunking
# --------------------------------------------------
def chunk_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# --------------------------------------------------
# FULL METADATA EXTRACTION
# --------------------------------------------------
def extract_all_metadata(pdf_file, file_size_kb=0):
    reader = PdfReader(pdf_file)
    pages = reader.pages
    meta = reader.metadata or {}

    metadata = {
        "title": meta.get("/Title"),
        "authors": [],
        "primary_person": None,
        "organizations": [],
        "creation_date": meta.get("/CreationDate"),
        "page_count": len(pages),
        "sections": [],
        "summary": "",
        "introduction": "",
        "keywords": [],
        "people_mentions": [],
        "organization_mentions": [],
        "language": "unknown",
        "is_scanned": True,
        "file_format": "PDF",
        "file_size_kb": file_size_kb,
    }

    # -------- First page --------
    first_page_text = pages[0].extract_text() or ""
    lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]

    if lines:
        metadata["title"] = metadata["title"] or lines[0]
        metadata["introduction"] = " ".join(lines[:5])
        metadata["summary"] = metadata["introduction"]

    try:
        metadata["language"] = detect(first_page_text)
    except Exception:
        pass

    # -------- Full text sample --------
    full_text = ""
    for page in pages[:50]:
        t = page.extract_text()
        if t:
            full_text += t + "\n"

    if full_text.strip():
        metadata["is_scanned"] = False

    # -------- Section detection --------
    for line in full_text.split("\n"):
        if line.isupper() and 4 < len(line) < 80:
            metadata["sections"].append(line.title())

    metadata["sections"] = list(set(metadata["sections"]))

    # -------- Named entities --------
    doc = nlp(full_text[:40000])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            metadata["people_mentions"].append(ent.text)
        if ent.label_ == "ORG":
            metadata["organization_mentions"].append(ent.text)

    metadata["people_mentions"] = list(set(metadata["people_mentions"]))
    metadata["organization_mentions"] = list(set(metadata["organization_mentions"]))

    if metadata["people_mentions"]:
        metadata["primary_person"] = metadata["people_mentions"][0]

    # -------- Keywords --------
    words = re.findall(r"\b[a-zA-Z]{5,}\b", full_text.lower())
    freq = Counter(words)
    metadata["keywords"] = [w for w, _ in freq.most_common(15)]

    return metadata
