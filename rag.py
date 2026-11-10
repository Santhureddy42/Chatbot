"""Load project documents and build an in-memory FAISS vector index."""

import json
import logging
import os
from pathlib import Path
from threading import Lock
from uuid import uuid4

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError
from pypdf import PdfReader

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_DIR / "documents"
VECTOR_STORE_DIR = Path(
    os.getenv("VECTOR_STORE_DIR", str(PROJECT_DIR / "vector_store"))
).expanduser()
VECTOR_STORE_MANIFEST = VECTOR_STORE_DIR / "current.json"
EMBEDDING_MODEL = "gemini-embedding-001"
RAG_DEMO_MODE = os.getenv("RAG_DEMO_MODE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEMO_DOCUMENTS = frozenset(
    {
        "library_events.txt",
        "library_faq.pdf",
        "library_information.txt",
        "library_rules.txt",
        "sample.txt",
    }
)

# Load the API key from the .env file next to this module.
load_dotenv(PROJECT_DIR / ".env")

# Keep the index and its chunk metadata available for repeated queries.
_vector_index: faiss.IndexFlatIP | None = None
_indexed_chunks: list[dict[str, str]] = []
_index_lock = Lock()


def _load_persisted_vector_store() -> tuple[faiss.IndexFlatIP, list[dict[str, str]]] | None:
    """Load a saved FAISS index and its matching chunk metadata, if available."""
    if not VECTOR_STORE_MANIFEST.is_file():
        return None

    print(
        f"Loading saved FAISS index and metadata from {VECTOR_STORE_DIR}...",
        flush=True,
    )
    with VECTOR_STORE_MANIFEST.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("The vector-store manifest has an unsupported format.")

    index_name = manifest.get("index_file")
    metadata_name = manifest.get("metadata_file")
    if (
        not isinstance(index_name, str)
        or Path(index_name).name != index_name
        or not isinstance(metadata_name, str)
        or Path(metadata_name).name != metadata_name
    ):
        raise ValueError("The vector-store manifest contains invalid file names.")

    index_path = VECTOR_STORE_DIR / index_name
    metadata_path = VECTOR_STORE_DIR / metadata_name
    index = faiss.read_index(str(index_path))
    if not isinstance(index, faiss.IndexFlatIP):
        raise ValueError("The saved FAISS index is not an IndexFlatIP index.")

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    if not isinstance(metadata, dict) or metadata.get("version") != 1:
        raise ValueError("The saved vector-store metadata has an unsupported format.")

    chunks = metadata.get("chunks")
    if not isinstance(chunks, list) or not all(
        isinstance(chunk, dict)
        and isinstance(chunk.get("text"), str)
        and isinstance(chunk.get("source"), str)
        for chunk in chunks
    ):
        raise ValueError("The saved vector-store chunk metadata is invalid.")
    if index.ntotal != len(chunks):
        raise ValueError(
            "The saved FAISS index and metadata have different vector counts."
        )
    if index.ntotal == 0 or index.d <= 0:
        raise ValueError("The saved FAISS index is empty or invalid.")

    logger.info("Loaded %s vectors from the persistent FAISS store.", index.ntotal)
    print(f"FAISS vectors loaded: {index.ntotal}", flush=True)
    print(f"Metadata chunks loaded: {len(chunks)}", flush=True)
    return index, chunks


def _save_vector_store(
    index: faiss.IndexFlatIP,
    chunks: list[dict[str, str]],
) -> None:
    """Persist an index/metadata snapshot and atomically publish its manifest."""
    if index.ntotal != len(chunks):
        raise RuntimeError("Cannot save an index and metadata with different sizes.")

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    generation = uuid4().hex
    index_name = f"index-{generation}.faiss"
    metadata_name = f"metadata-{generation}.json"
    index_path = VECTOR_STORE_DIR / index_name
    metadata_path = VECTOR_STORE_DIR / metadata_name
    manifest_temp_path = VECTOR_STORE_DIR / f"current-{generation}.tmp"

    try:
        # Write both generation files before publishing either as the current store.
        faiss.write_index(index, str(index_path))
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump({"version": 1, "chunks": chunks}, metadata_file, ensure_ascii=False)
            metadata_file.flush()
            os.fsync(metadata_file.fileno())

        with manifest_temp_path.open("w", encoding="utf-8") as manifest_file:
            json.dump(
                {
                    "version": 1,
                    "index_file": index_name,
                    "metadata_file": metadata_name,
                },
                manifest_file,
            )
            manifest_file.flush()
            os.fsync(manifest_file.fileno())

        # The manifest is the commit point: readers see the old or the new pair.
        os.replace(manifest_temp_path, VECTOR_STORE_MANIFEST)
        print(
            f"Saved FAISS index and {len(chunks)} metadata chunks to {VECTOR_STORE_DIR}.",
            flush=True,
        )
    except Exception as error:
        for temporary_path in (index_path, metadata_path, manifest_temp_path):
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Could not remove incomplete vector-store file %s.",
                    temporary_path.name,
                )
        raise RuntimeError("Could not save the persistent FAISS vector store.") from error


def load_documents() -> list[dict[str, str]]:
    """Read non-empty TXT and PDF documents from the documents folder."""
    documents = []

    if not DOCUMENTS_DIR.exists():
        logger.warning("Documents directory does not exist yet: %s", DOCUMENTS_DIR)
        return documents

    for file_path in sorted(DOCUMENTS_DIR.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in {".txt", ".pdf"}:
            continue
        if RAG_DEMO_MODE and file_path.name not in DEMO_DOCUMENTS:
            logger.info("Skipping non-demo document: %s", file_path.name)
            continue

        logger.info("Loading document from disk: %s", file_path.name)
        text = _extract_document_text(file_path)
        text = " ".join(text.split())
        if not text:
            logger.warning("Skipping empty document: %s", file_path.name)
            continue

        documents.append({"text": text, "source": file_path.name})

    logger.info("Loaded %s readable documents from %s.", len(documents), DOCUMENTS_DIR)
    return documents


def _extract_document_text(file_path: Path) -> str:
    """Extract text from one supported TXT or PDF file."""
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("UTF-8 decode failed for %s; retrying with latin-1 fallback.", file_path.name)
            text = file_path.read_text(encoding="latin-1")
        logger.info("Extracted %s plain-text characters from %s.", len(text), file_path.name)
        return text
    if suffix == ".pdf":
        try:
            reader = PdfReader(file_path)
            pages = []
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                pages.append(page_text)
                logger.debug("Extracted PDF page %s from %s (%s chars).", page_number, file_path.name, len(page_text))
            text = "\n".join(pages)
            logger.info("Extracted %s plain-text characters from PDF %s.", len(text), file_path.name)
            return text
        except Exception as error:
            logger.error("Failed to extract text from PDF %s (%s).", file_path.name, type(error).__name__)
            raise ValueError(f"Could not read the PDF file: {file_path.name}.") from error
    raise ValueError("Only .txt and .pdf documents can be indexed.")


def split_documents(
    documents: list[dict[str, str]],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict[str, str]]:
    """Split document text into overlapping chunks and retain each source name."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and smaller than chunk_size.")

    chunks = []
    step = chunk_size - overlap

    for document in documents:
        text = document["text"].strip()
        source = document["source"]

        for start in range(0, len(text), step):
            chunk_text = text[start : start + chunk_size].strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "source": source})
            if start + chunk_size >= len(text):
                break

    return chunks


def create_embeddings(chunks: list[dict[str, str]]) -> np.ndarray:
    """Generate and L2-normalize one Gemini embedding for each text chunk."""
    if not chunks:
        raise ValueError("Cannot create embeddings because there are no text chunks.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from the environment.")

    client = genai.Client(api_key=api_key)
    vectors = []
    logger.info("Creating embeddings for %s chunks using %s.", len(chunks), EMBEDDING_MODEL)

    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            logger.warning("Skipping empty chunk for source %s.", chunk.get("source"))
            continue
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config={"task_type": "RETRIEVAL_DOCUMENT"},
            )
        except APIError as error:
            logger.error("Gemini embedding request failed for %s with status %s.", chunk.get("source"), getattr(error, "code", "unknown"))
            raise RuntimeError(f"Gemini could not create document embeddings for {chunk.get('source', 'unknown')}.") from error

        if not response.embeddings or response.embeddings[0].values is None:
            raise RuntimeError(
                f"Gemini returned no embedding for document {chunk['source']}."
            )

        vector = np.asarray(response.embeddings[0].values, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if vector.ndim != 1 or not np.isfinite(vector).all() or norm == 0:
            raise RuntimeError(
                f"Gemini returned an invalid embedding for document {chunk['source']}."
            )

        vectors.append(vector / norm)

    if not vectors:
        raise RuntimeError("No valid embeddings were produced from the supplied chunks.")

    try:
        matrix = np.vstack(vectors).astype(np.float32, copy=False)
        logger.info("Generated embedding matrix with shape %s.", matrix.shape)
        return matrix
    except ValueError as error:
        raise RuntimeError("Gemini returned embeddings with inconsistent dimensions.") from error


def build_vector_store() -> tuple[faiss.IndexFlatIP, list[dict[str, str]]]:
    """Load the saved index or build and persist it from the documents folder."""
    global _vector_index, _indexed_chunks

    with _index_lock:
        if _vector_index is not None:
            return _vector_index, _indexed_chunks

        try:
            persisted_store = _load_persisted_vector_store()
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
            logger.warning(
                "Could not load the persistent vector store (%s); rebuilding it.",
                type(error).__name__,
            )
            print(
                "Saved FAISS store could not be loaded; rebuilding from documents/.",
                flush=True,
            )
            persisted_store = None

        if persisted_store is not None:
            persisted_index, persisted_chunks = persisted_store
            if persisted_index.ntotal != len(persisted_chunks):
                logger.warning(
                    "Persisted FAISS store is out of sync (%s vectors, %s chunks); rebuilding it.",
                    persisted_index.ntotal,
                    len(persisted_chunks),
                )
                persisted_store = None
            else:
                _vector_index, _indexed_chunks = persisted_store
                logger.info("Loaded persisted FAISS store with %s vectors.", _vector_index.ntotal)
                return _vector_index, _indexed_chunks

        if not VECTOR_STORE_MANIFEST.exists():
            print(
                "No saved FAISS index found; building a new index from documents/.",
                flush=True,
            )
        documents = load_documents()
        if not documents:
            raise ValueError(
                f"No non-empty TXT or PDF documents found in {DOCUMENTS_DIR}."
            )

        chunks = split_documents(documents)
        if not chunks:
            raise ValueError("No text chunks were created from the documents.")

        embeddings = create_embeddings(chunks)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        _save_vector_store(index, chunks)

        _vector_index = index
        _indexed_chunks = chunks
        logger.info("Built and saved %s document vectors.", index.ntotal)
        print(
            f"Built a new FAISS index with {index.ntotal} vectors.",
            flush=True,
        )
        return _vector_index, _indexed_chunks


def _build_store_from_documents(
    documents: list[dict[str, str]],
) -> tuple[faiss.IndexFlatIP, list[dict[str, str]]]:
    """Build and persist an initial store from a specific document list."""
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks were created from the documents.")

    embeddings = create_embeddings(chunks)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    _save_vector_store(index, chunks)
    print(f"Built a new FAISS index with {index.ntotal} vectors.", flush=True)
    return index, chunks


def add_document_to_vector_store(file_path: str | Path) -> int:
    """Add a TXT or PDF file to the cached FAISS store and return chunks added.

    A source filename already present in the store is treated as a duplicate
    and skipped. If the project documents have not been indexed yet, they are
    indexed first so this function preserves the existing document store.
    """
    global _vector_index, _indexed_chunks

    document_path = Path(file_path).expanduser().resolve()
    if not document_path.is_file():
        raise FileNotFoundError(f"Document file does not exist: {document_path}")
    if document_path.suffix.lower() not in {".txt", ".pdf"}:
        raise ValueError("Only .txt and .pdf documents can be indexed.")

    # Initialize from persistence or the existing documents before adding the
    # uploaded file. The file may already be in documents/ because Flask saves it
    # before calling this function, so exclude it from a cold-start rebuild.
    if _vector_index is None and DOCUMENTS_DIR.is_dir():
        with _index_lock:
            if _vector_index is None:
                try:
                    persisted_store = _load_persisted_vector_store()
                except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
                    logger.warning(
                        "Could not load the persistent vector store (%s); rebuilding it.",
                        type(error).__name__,
                    )
                    persisted_store = None

                if persisted_store is not None:
                    _vector_index, _indexed_chunks = persisted_store
                else:
                    documents = load_documents()
                    existing_documents = [
                        document
                        for document in documents
                        if (DOCUMENTS_DIR / document["source"]).resolve()
                        != document_path
                    ]
                    if existing_documents:
                        _vector_index, _indexed_chunks = _build_store_from_documents(
                            existing_documents
                        )

    with _index_lock:
        source = document_path.name
        if any(chunk["source"] == source for chunk in _indexed_chunks):
            logger.info("Skipping already indexed document: %s", source)
            return 0

        text = _extract_document_text(document_path).strip()
        if not text:
            raise ValueError(f"Document contains no extractable text: {source}")

        chunks = split_documents([{"text": text, "source": source}])
        if not chunks:
            raise ValueError(f"No text chunks could be created for: {source}")

        embeddings = create_embeddings(chunks)
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Gemini returned {len(embeddings)} embeddings for {len(chunks)} chunks."
            )

        # Build an updated copy first, so a failed FAISS operation cannot leave
        # the live index and its chunk metadata out of sync.
        if _vector_index is None:
            updated_index = faiss.IndexFlatIP(embeddings.shape[1])
        else:
            if _vector_index.ntotal != len(_indexed_chunks):
                raise RuntimeError("The FAISS index and chunk metadata are out of sync.")
            if _vector_index.d != embeddings.shape[1]:
                raise ValueError(
                    "New document embedding dimensions do not match the existing "
                    f"FAISS index ({embeddings.shape[1]} != {_vector_index.d})."
                )
            cloned_index = faiss.clone_index(_vector_index)
            if not isinstance(cloned_index, faiss.IndexFlatIP):
                raise RuntimeError("Could not safely clone the FAISS inner-product index.")
            updated_index = cloned_index

        updated_index.add(embeddings)
        updated_chunks = [*_indexed_chunks, *chunks]
        if updated_index.ntotal != len(updated_chunks):
            raise RuntimeError("The updated FAISS index and chunk metadata are out of sync.")

        # Persist the complete snapshot before publishing it to the live cache.
        _save_vector_store(updated_index, updated_chunks)
        _vector_index = updated_index
        _indexed_chunks = updated_chunks
        logger.info("Indexed %s chunks from %s.", len(chunks), source)
        return len(chunks)


def retrieve_relevant_chunks(
    question: str,
    top_k: int = 3,
) -> list[dict[str, str | float]]:
    """Find the most similar stored chunks for a question.

    The document index is built at most once per Python process. The query
    embedding is generated for each question, but document embeddings are reused.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    # Build once when this process has not created or loaded an index yet.
    index = _vector_index
    chunks = _indexed_chunks
    if index is None or index.ntotal == 0:
        index, chunks = build_vector_store()
    if index.ntotal == 0 or not chunks:
        raise RuntimeError("The FAISS index is empty or has no chunk metadata.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from the environment.")

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=question.strip(),
            config={"task_type": "RETRIEVAL_QUERY"},
        )
    except APIError as error:
        logger.error("Gemini query embedding failed with status %s.", error.code)
        raise RuntimeError("Gemini could not create an embedding for the question.") from error

    if not response.embeddings or response.embeddings[0].values is None:
        raise RuntimeError("Gemini returned no embedding for the question.")

    query_vector = np.asarray(response.embeddings[0].values, dtype=np.float32)
    norm = np.linalg.norm(query_vector)
    if (
        query_vector.ndim != 1
        or not np.isfinite(query_vector).all()
        or norm == 0
    ):
        raise RuntimeError("Gemini returned an invalid embedding for the question.")
    if query_vector.size != index.d:
        raise ValueError(
            "Question embedding dimension does not match the FAISS index "
            f"({query_vector.size} != {index.d})."
        )

    # Normalize the query so inner-product scores represent cosine similarity.
    query_vector = (query_vector / norm).reshape(1, -1)
    result_count = min(top_k, index.ntotal)
    scores, chunk_indices = index.search(query_vector, result_count)

    results = []
    for score, chunk_index in zip(scores[0], chunk_indices[0]):
        if chunk_index < 0:
            continue

        chunk = chunks[int(chunk_index)]
        results.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "score": float(score),
            }
        )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    vector_index, indexed_chunks = build_vector_store()
    print(f"Indexed {vector_index.ntotal} chunks from {len(indexed_chunks)} documents.")
