# News Research Tool

Streamlit app that loads news article URLs, builds a FAISS vector store, and answers questions using Meta Llama 3 via Hugging Face.

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your Hugging Face API token:

   ```bash
   copy .env.example .env
   ```

   Get a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Run

```bash
streamlit run main.py
```

1. Enter up to 3 news article URLs in the sidebar.
2. Click **Process URLs** to load, split, embed, and save the vector store.
3. Ask a question in the main input to query the articles.

## Generated files

- `faiss_store_hfai.pkl` — created after processing URLs; ignored by git.
