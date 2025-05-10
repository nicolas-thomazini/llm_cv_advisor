# 🧠 RAG FAQ Assistant — Jupyter Notebook

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using:

✅ minsearch.py for keyword & semantic indexing

✅ Elasticsearch for document storage & search

✅ Mistral API for LLM-based responses

✅ A Jupyter Notebook to orchestrate everything interactively

## 📁 Project Structure

```bash
.
├── notebooks/
│   └── rag_faq_pipeline.ipynb       # Jupyter notebook (main logic)
├── data/
│   └── documents.json               # JSON file with FAQs
├── minsearch.py                     # Lightweight vector search engine
├── run_notebook.sh                  # (Optional) script to auto-launch the notebook
├── requirements.txt                 # Python dependencies
```

## 🚀 How to Run

1.  Install Requirements
    Make sure you’re using a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
2.  Start Elasticsearch (Docker required)

    ```bash
    sudo docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    elasticsearch:9.0.1
    ```

    ⚠️ Version 9.0+ is not supported with elasticsearch-py yet.

3.  Run the Notebook
    You can open the notebook manually:

    ```bash
    jupyter notebook notebooks/rag_faq_pipeline.ipynb
    ```

    Or automatically with the script:

    ```bash
    bash run_notebook.sh
    ```

## 📌 What This Notebook Does

1. Loads and prepares FAQ documents from documents.json

2. Indexes them using minsearch and Elasticsearch

3. Implements a hybrid RAG search combining:

4. BM25-style search (via Elasticsearch)

5. Embedding-like search (via minsearch)

6. Sends the combined context to an LLM (Mistral API)

7. Returns answers grounded in the FAQ database

## 🔑 Notes

- Replace the Mistral API key inside the notebook.

- You must have Docker installed and running.

- Make sure the Elasticsearch container is accessible via localhost:9200.
