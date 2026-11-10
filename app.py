import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from google import genai
from google.genai.errors import APIError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from rag import (
    add_document_to_vector_store,
    build_vector_store,
    retrieve_relevant_chunks,
)

DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024

# Load secrets from this project's .env file without putting them in frontend code.
load_dotenv(Path(__file__).resolve().with_name(".env"))
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Create the Gemini client only when a key is configured, so the Flask app can
# still start and return a clear error if setup is incomplete.
gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

# Create the Flask application.
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
app.config["ENABLE_DOCUMENT_UPLOADS"] = os.getenv(
    "ENABLE_DOCUMENT_UPLOADS", "true"
).strip().lower() in {"1", "true", "yes", "on"}
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://",
)

vector_store_ready = False


def initialize_vector_store() -> None:
    """Load or build the persistent FAISS store for this serving process."""
    global vector_store_ready

    print("Flask startup: initializing the FAISS vector store...", flush=True)
    try:
        vector_index, indexed_chunks = build_vector_store()
        vector_store_ready = vector_index.ntotal > 0 and bool(indexed_chunks)
        if vector_store_ready:
            print(
                "Flask startup: FAISS is ready with "
                f"{vector_index.ntotal} vectors and {len(indexed_chunks)} metadata chunks.",
                flush=True,
            )
        else:
            app.logger.error("The FAISS vector store contains no document chunks.")
            print("Flask startup: the FAISS vector store is empty.", flush=True)
    except Exception as error:
        # Keep Flask available so setup problems are reported through the API.
        app.logger.error(
            "FAISS vector store initialization failed (%s): %s",
            type(error).__name__,
            error,
        )
        print(
            "Flask startup: FAISS initialization failed "
            f"({type(error).__name__}: {error}).",
            flush=True,
        )


# Load the saved vector store once when this application process starts.
initialize_vector_store()


@app.errorhandler(RequestEntityTooLarge)
def handle_upload_too_large(error):
    """Return a JSON error when the request exceeds the configured size limit."""
    app.logger.warning("Rejected oversized request (%s).", type(error).__name__)
    return jsonify(
        {
            "success": False,
            "error": "The upload is too large. Maximum request size is 10 MB.",
        }
    ), 413


@app.errorhandler(429)
def handle_rate_limit_exceeded(error):
    """Return a friendly response without exposing limiter internals."""
    return jsonify(
        {"error": "Too many chat requests. Please wait a minute and try again."}
    ), 429


# Display the chatbot page.
@app.get("/")
def home():
    return render_template(
        "index.html",
        uploads_enabled=app.config["ENABLE_DOCUMENT_UPLOADS"],
    )


@app.get("/health")
def health():
    """Report whether the app can currently serve Gemini-backed chat requests."""
    if not vector_store_ready or gemini_client is None:
        return jsonify({"status": "unavailable"}), 503
    return jsonify({"status": "ok"})


@app.post("/api/upload")
def upload_document():
    """Save and index one PDF or TXT document without overwriting existing files."""
    if not app.config["ENABLE_DOCUMENT_UPLOADS"]:
        return jsonify(
            {
                "success": False,
                "error": "Document uploads are disabled.",
            }
        ), 403

    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify(
            {
                "success": False,
                "error": "Choose a PDF or TXT file to upload.",
            }
        ), 400

    filename = secure_filename(uploaded_file.filename)
    if not filename:
        return jsonify(
            {
                "success": False,
                "error": "The uploaded filename is not valid.",
            }
        ), 400

    if Path(filename).suffix.lower() not in {".pdf", ".txt"}:
        return jsonify(
            {
                "success": False,
                "error": "Only PDF and TXT files are supported.",
            }
        ), 400

    destination = DOCUMENTS_DIR / filename
    created_file = False
    try:
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        app.logger.info("Saving uploaded file %s (%s bytes).", filename, uploaded_file.content_length)

        # Exclusive creation prevents race conditions and accidental overwrite.
        with destination.open("xb") as saved_file:
            created_file = True
            shutil.copyfileobj(uploaded_file.stream, saved_file)
    except FileExistsError:
        app.logger.warning("Upload rejected because %s already exists.", filename)
        return jsonify(
            {
                "success": False,
                "error": "A file with that name already exists. Rename it and try again.",
            }
        ), 409
    except OSError as error:
        if created_file:
            try:
                destination.unlink()
            except OSError as cleanup_error:
                app.logger.error(
                    "Could not remove incomplete upload (%s).",
                    type(cleanup_error).__name__,
                )
        app.logger.error("Could not save uploaded document (%s).", type(error).__name__)
        return jsonify(
            {
                "success": False,
                "error": "The uploaded file could not be saved.",
            }
        ), 500

    try:
        # Reuse RAG indexing; this adds vectors to the existing in-memory store.
        app.logger.info("Indexing uploaded file %s.", filename)
        chunks_indexed = add_document_to_vector_store(destination)
    except Exception as error:
        # Do not keep a file that failed to enter the index.
        try:
            destination.unlink(missing_ok=True)
        except OSError as cleanup_error:
            app.logger.error(
                "Could not remove failed upload (%s).",
                type(cleanup_error).__name__,
            )
        app.logger.error("Uploaded document indexing failed for %s (%s): %s", filename, type(error).__name__, error)
        return jsonify(
            {
                "success": False,
                "error": "The file was saved but could not be indexed. It may be empty, unreadable, or contain no extractable text.",
            }
        ), 502

    if chunks_indexed == 0:
        destination.unlink(missing_ok=True)
        app.logger.info("Upload skipped because %s was already indexed.", filename)
        return jsonify(
            {
                "success": False,
                "error": "This document is already indexed.",
            }
        ), 409

    global vector_store_ready
    vector_store_ready = True
    app.logger.info("Upload successful for %s: %s chunks indexed.", filename, chunks_indexed)
    return jsonify(
        {
            "success": True,
            "filename": filename,
            "chunks_indexed": chunks_indexed,
            "message": f"Uploaded and indexed {filename} successfully.",
        }
    ), 201


# Accept a question and return a document-grounded or general-knowledge answer.
@app.post("/api/chat")
@limiter.limit("10 per minute")
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400

    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Message must not be empty."}), 400

    if not vector_store_ready:
        return jsonify({"error": "The document index is unavailable."}), 503

    if gemini_client is None:
        return jsonify({"error": "The Gemini API is not configured."}), 503

    # Retrieve relevant chunks from the existing in-memory FAISS index.
    try:
        retrieved_chunks = retrieve_relevant_chunks(message.strip(), top_k=3)
    except (RuntimeError, ValueError) as error:
        app.logger.error("Document retrieval failed (%s).", type(error).__name__)
        return jsonify({"error": "Unable to search the provided documents."}), 502

    # Keep source names alongside the context and return them to the frontend.
    sources = []
    for chunk in retrieved_chunks:
        source = chunk.get("source")
        if isinstance(source, str) and source.strip():
            sources.append(source.strip())
    sources = list(dict.fromkeys(sources))
    if not retrieved_chunks:
        return jsonify(
            {
                "response": "I couldn't find this information in the provided documents.",
                "sources": sources,
            }
        )

    context = "\n\n".join(
        f"Source: {chunk['source']}\n{chunk['text']}"
        for chunk in retrieved_chunks
    )
    prompt = f"""Answer the user's question using only the document context below.
Do not use outside knowledge or invent information. If the answer is not available
in the context, respond exactly: "I couldn't find this information in the provided documents."
Give a clear and simple answer.

<document_context>
{context}
</document_context>

Question: {message.strip()}
"""

    # Generate an answer grounded only in the retrieved document context.
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
        )
    except APIError as error:
        app.logger.error("Gemini API request failed with status %s.", error.code)
        return jsonify({"error": "Unable to get a response from Gemini."}), 502

    answer = response.text
    if not answer:
        return jsonify({"error": "Gemini returned an empty response."}), 502

    return jsonify({"response": answer, "sources": sources})


# Start Flask's development server when this file is run directly.
if __name__ == "__main__":
    debug_enabled = os.getenv("FLASK_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=debug_enabled,
    )