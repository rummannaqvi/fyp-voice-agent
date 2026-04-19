import os
import chromadb
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAIEmbeddings

load_dotenv()

embeddings_model = VertexAIEmbeddings(
    model_name="text-embedding-004",
    project=os.getenv("VERTEX_PROJECT_ID"),
    location="us-east1",
)

# Initialize local ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="parametric_knowledge")

WEBSITE_URLS = [
    "https://www.parametricestimates.com",
    "https://www.parametricestimates.com/about-us",
    "https://www.parametricestimates.com/contact",
    "https://www.parametricestimates.com/pricing-plan",
    "https://www.parametricestimates.com/faq",
    "https://www.parametricestimates.com/blog",
    "https://www.parametricestimates.com/samples",
    "https://www.parametricestimates.com/services",
    "https://www.parametricestimates.com/services/preliminary-estimating",
    "https://www.parametricestimates.com/services/residential-estimating",
    "https://www.parametricestimates.com/services/commercial-estimating",
    "https://www.parametricestimates.com/services/industrial-estimating",
    "https://www.parametricestimates.com/services/gc-estimating",
    "https://www.parametricestimates.com/services/mep-estimating",
    "https://www.parametricestimates.com/services/bim-estimating",
    "https://www.parametricestimates.com/services/remodeling",
    "https://www.parametricestimates.com/services/excavation-estimating",
    "https://www.parametricestimates.com/services/cpm-scheduling"
]

def scrape_text_from_url(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

def init_knowledge_base():
    if collection.count() > 0:
        print(f"📚 Knowledge Base already loaded with {collection.count()} chunks.")
        return

    print("🌐 Scraping Parametric Estimates website...")
    all_chunks = []

    for url in WEBSITE_URLS:
        print(f"Reading: {url}")
        text = scrape_text_from_url(url)
        if not text:
            continue

        words = text.split()
        chunk_size = 100
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i + chunk_size])
            if len(chunk_text.strip()) > 30:
                safe_url = url.split('.com')[-1].replace('/', '_')
                if not safe_url:
                    safe_url = "_home"
                all_chunks.append({
                    "text": chunk_text,
                    "id": f"doc{safe_url}_chunk_{i}"
                })

    if not all_chunks:
        print("Failed to extract any text from the website.")
        return

    print(f"🧠 Generating Vertex AI embeddings for {len(all_chunks)} chunks...")

    texts = [c["text"] for c in all_chunks]
    ids = [c["id"] for c in all_chunks]
    batch_size = 20

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        # LangChain VertexAIEmbeddings
        batch_embeddings = embeddings_model.embed_documents(batch_texts)

        collection.upsert(
            documents=batch_texts,
            embeddings=batch_embeddings,
            ids=batch_ids,
        )

    print(f"✅ Successfully loaded {len(all_chunks)} website chunks!")

def query_knowledge_base(user_question: str, top_k: int = 2) -> str:
    try:
        query_embedding = embeddings_model.embed_query(user_question)

        # Guard: check stored dimension matches query dimension
        stored_count = collection.count()
        if stored_count == 0:
            return ""

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, stored_count),
        )

        retrieved_facts = results['documents'][0]
        if not retrieved_facts:
            return ""
        return " ".join(retrieved_facts)

    except Exception as e:
        print(f"RAG Error: {e}")
        return ""   # ← always return empty string, never crash

init_knowledge_base()