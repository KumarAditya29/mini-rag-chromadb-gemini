from pathlib import Path
import re
import chromadb
from pypdf import PdfReader

DATA_DIR = Path("data")
DB_DIR = "chroma_db"
COLLECTION = "documents"


def chunk_text(text, size=500, overlap=50):
    words = re.findall(r"\S+", text)
    chunks = []

    step = size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + size])

        if len(chunk) >= 50:
            chunks.append(chunk)

    return chunks


client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(COLLECTION)

ids = []
documents = []
metadatas = []

for path in DATA_DIR.glob("*.pdf"):
    reader = PdfReader(path)

    for page_number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""

        for chunk_number, chunk in enumerate(chunk_text(text)):
            ids.append(f"{path.stem}-p{page_number}-c{chunk_number}")
            documents.append(chunk)
            metadatas.append({
                "source": path.name,
                "page": page_number,
                "chunk": chunk_number
            })

collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print(f"Files processed: {len(list(DATA_DIR.glob('*.pdf')))}")
print(f"Total chunks in Chroma: {collection.count()}")
print("Persistent database: chroma_db/")
