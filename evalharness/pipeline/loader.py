"""Load documents from data/docs/ and split them into identifiable chunks.

The doc_id is the critical piece: every chunk gets a stable, human-readable
id like "hr_policy.pdf::c3". The golden dataset references these ids, so
retrieval scoring depends on them staying consistent across runs.
"""

from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    doc_id: str      # e.g. "hr_policy.pdf::c3"
    source: str      # original filename
    text: str
    index: int       # position within the source document


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


READERS = {
    ".pdf": _read_pdf,
    ".txt": _read_text,
    ".md": _read_text,
}


def load_chunks(
    docs_dir: str | Path = "data/docs",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Read every supported file in docs_dir and return chunked text.

    chunk_size and chunk_overlap come from the config file — varying them
    and re-running is one of the main experiments this harness enables.
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"No such directory: {docs_dir.resolve()}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for path in sorted(docs_dir.iterdir()):
        reader = READERS.get(path.suffix.lower())
        if reader is None:
            continue

        raw = reader(path).strip()
        if not raw:
            print(f"  warning: {path.name} produced no text, skipping")
            continue

        for i, piece in enumerate(splitter.split_text(raw)):
            chunks.append(
                Chunk(
                    doc_id=f"{path.name}::c{i}",
                    source=path.name,
                    text=piece,
                    index=i,
                )
            )

    if not chunks:
        raise ValueError(f"No readable documents found in {docs_dir.resolve()}")

    return chunks


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {len(set(c.source for c in chunks))} files")
    for c in chunks[:3]:
        print(f"\n[{c.doc_id}]  {c.text[:120]}...")