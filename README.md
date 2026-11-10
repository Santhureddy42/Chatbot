# Gemini RAG Chatbot

A Flask chatbot that retrieves relevant chunks from PDF and TXT documents with
FAISS and uses the Gemini API to generate document-grounded answers.

## Run locally

1. Install Python and create/activate a virtual environment.
2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Set `GEMINI_API_KEY` in an ignored local `.env` file, or in your shell
   environment. Never commit the key.
4. Start the development server:

   ```powershell
   python app.py
   ```

Local document uploads remain enabled by default. Set
`ENABLE_DOCUMENT_UPLOADS=false` to turn them off locally.
The development server binds to `127.0.0.1` and runs with debug mode off by
default. Set `FLASK_DEBUG=true` only when debugging locally.

## Deploy on Render

1. Before connecting a repository, review the files you intend to publish.
   `.gitignore` excludes files in `documents/` by default and allows only
   `library_events.txt`, `library_faq.pdf`, `library_information.txt`,
   `library_rules.txt`, and `sample.txt`. The resumes, government report, and
   any other documents stay excluded; do not force-add them.
2. Make sure the application source, frontend, requirements, README,
   `render.yaml`, and the five public demo documents are included in the Git
   repository. Do not include `.env` or `vector_store/`.
3. In Render, create a Blueprint from that repository and use `render.yaml`.
   The Blueprint installs `requirements.txt`, runs one Gunicorn worker bound to
   `0.0.0.0:$PORT`, and checks `/health`.
4. Set `GEMINI_API_KEY` when prompted by Render. It is declared with
   `sync: false`, so its value is entered in Render's environment settings and
   is not stored in this repository.
5. The public configuration enables `RAG_DEMO_MODE` and disables
   `ENABLE_DOCUMENT_UPLOADS`. The upload form is hidden and the upload API
   rejects requests. The chat API allows 10 requests per minute per client IP;
   this in-memory limit is shared by the single configured Gunicorn worker and
   resets when the process restarts. Do not enable public uploads without adding
   suitable authentication and durable storage.

Render builds the demo FAISS index at startup from the allowlisted documents,
using the configured Gemini key. Its index path is separate from the local
`vector_store/`, which is ignored by Git and is not changed by this deployment
configuration. The Render demo path is ephemeral; because uploads are disabled,
the index can be recreated from the demo documents after a restart. Public
Render services may sleep on free plans, and the first request after a sleep
can take longer.
