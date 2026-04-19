import os
import chromadb
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60.0)

# Initialize local ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="parametric_knowledge")

# Your sitemap URLs
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
    """Fetches a web page and extracts clean text."""
    try:
        # Pretend to be a normal web browser so the site doesn't block us
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Strip out the code we don't need to read (headers, footers, scripts)
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
            
        # Get the visible text
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

def init_knowledge_base():
    """Scrapes the website, chunks the text, and saves it to ChromaDB."""
    # Check if we already have data to avoid scraping every single time you restart the server
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
            
        # Chop the webpage into chunks of ~100 words so the LLM can digest them easily
        words = text.split()
        chunk_size = 100 
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Only save chunks that actually have a decent amount of text
            if len(chunk_text.strip()) > 30:
                # Create a unique ID for this chunk based on the URL
                safe_url = url.split('.com')[-1].replace('/', '_')
                if not safe_url: safe_url = "_home"
                
                all_chunks.append({
                    "text": chunk_text,
                    "id": f"doc{safe_url}_chunk_{i}"
                })

    if not all_chunks:
        print("Failed to extract any text from the website.")
        return

    print(f"🧠 Generating vector embeddings for {len(all_chunks)} chunks...")
    
    # Process the embeddings in batches of 100 to respect OpenAI's rate limits
    texts = [c["text"] for c in all_chunks]
    ids = [c["id"] for c in all_chunks]
    batch_size = 20 
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        # Send text to OpenAI to get the mathematical meaning (embedding)
        response = client.embeddings.create(
            input=batch_texts,
            model="text-embedding-3-small"
        )
        embeddings = [data.embedding for data in response.data]
        
        # Save to your local ChromaDB
        collection.upsert(
            documents=batch_texts,
            embeddings=embeddings,
            ids=batch_ids
        )
        
    print(f"✅ Successfully loaded {len(all_chunks)} website chunks into the brain!")

def query_knowledge_base(user_question: str, top_k: int = 2) -> str:
    """Searches the database for the most relevant facts to the user's question."""
    try:
        response = client.embeddings.create(
            input=user_question,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieved_facts = results['documents'][0]
        if not retrieved_facts:
            return ""
            
        return " ".join(retrieved_facts)
    except Exception as e:
        print(f"RAG Error: {e}")
        return ""

# Fire up the scraper when the server starts
init_knowledge_base()