import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
POLICY_FILE = os.path.join(DATA_DIR, "company_policy.txt")
QDRANT_INDEX = os.path.join(DATA_DIR, "qdrant_index")

# ── Step 1: Load document ───────
# Analogy: scanning a physical document into the computer
def load_documents(file_path:str) -> list[Document]:
    """
    Reads a text file from disk and returns it as
    a list of LangChain Document objects.

    Why list? Because some loaders return multiple documents
    e.g. PDFLoader returns one Document per page
    TextLoader returns one Document for the whole file
    """
    loader= TextLoader(file_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s)")

    return documents

# ── Step 2: Split into chunks ─────────────────────────────────────
# Analogy: cutting a book into individual pages
def split_documents(documents: list[Document]) -> list[Document]:
    """
    Splits large documents into smaller chunks.

    chunk_size=500
    → each chunk is maximum 500 characters
    → why 500? balance between context and precision
      too large  → retrieves too much irrelevant text
      too small  → loses context, misses meaning

    chunk_overlap=50
    → last 50 characters of chunk 1 repeat at start of chunk 2
    → why? so sentences split across boundaries still make sense
    → like Venn diagram — small overlap between circles

    separators=["\n\n", "\n", " ", ""]
    → tries to split at paragraph break first (\n\n)
    → then line break (\n)
    → then space (between words)
    → last resort: anywhere ("")
    → this preserves natural document structure
    """

    splitters = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]

    )
    chunks = splitters.split_documents(documents)

    print(f"Split into {len(chunks)} chunks")

    if chunks:
        print(f"\nFirst chunk preview:")
        print(f"Content: {chunks[0].page_content[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")


    return chunks

# ── Step 3: Create embeddings ─────────────────────────────────────
# Analogy: the fingerprint machine
# converts text to numbers so similar meaning = similar numbers

def create_embeddings():
    """
    Initialises the Voyage AI embedding model.

    What it does:
    → connects to Voyage AI API
    → when you call embed_query("refund policy")
      it returns [0.2, 0.8, 0.1, ...] — 512 numbers
    → similar text = similar numbers = close in vector space

    model="voyage-3-lite"
    → Voyage AI's lightweight model
    → produces 512-dimensional vectors
    → free tier available
    → fast and accurate enough for our demo
    → alternative: "voyage-3" → higher quality, costs more
    """
    print("Loading vouage ai embeddings")

    embeddings = VoyageAIEmbeddings(
        voyage_api_key = os.getenv("VOYAGE_API_KEY"),
        model = "voyage-3-lite"
    )
    print("embeddings ready")
    return embeddings

# ── Step 4: Store in Qdrant ───────────────────────────────────────
# Analogy: the filing cabinet organised by fingerprint

def create_vector_store(
        chunks:list[Document],
        embeddings,
        persist_path: str= QDRANT_INDEX
)-> QdrantVectorStore:
    """
    Converts every chunk into a vector and stores in Qdrant.

    What happens internally:
    1. For each chunk, calls Voyage AI to get its vector
    2. Stores (vector, original text, metadata) in Qdrant
    3. Saves everything to disk at persist_path

    QdrantClient(path=persist_path)
    → path= means embedded mode — runs inside your Python process
    → saves to disk automatically
    → no separate server needed
    → alternative: QdrantClient(url="http://localhost:6333")
      → runs as separate server process (production mode)

    client.recreate_collection(...)
    → creates a fresh collection (like CREATE TABLE in SQL)
    → recreate = drop if exists + create new
    → collection_name = "company_docs" (like a table name)

    VectorParams(size=512, distance=Distance.COSINE)
    → size=512 must match voyage-3-lite output dimensions
    → Distance.COSINE measures angle between vectors
      perfect for text — direction matters more than magnitude
    """
    print("Creating Qdrant vector store...")
    print("(Fingerprinting every chunk — takes ~30 seconds)")

    client = QdrantClient(path=persist_path)
    client.recreate_collection(
        collection_name="company_docs",
        vectors_config=VectorParams(
            size=512,
            distance = Distance.COSINE
        )

    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="company_docs",
        embedding=embeddings
    )

    vector_store.add_documents(chunks)
    print(f"Qdrant vector store saved to {persist_path}")

    return vector_store

def load_vector_store(
        embeddings,
        persist_path:str = QDRANT_INDEX

)-> QdrantVectorStore:
    """
    Loads existing Qdrant index from disk.
    Like opening the filing cabinet we already built.
    No re-embedding needed — vectors already stored.
    """
    client = QdrantClient(path=persist_path)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="company_docs",
        embeddings=embeddings
    )
    print(f"Qdrant vector store loaded from {persist_path}")
    return vector_store

# ── Step 5: Search ────────────────────────────────────────────────
# Analogy: finding the most relevant pages for a question

def search(
        vector_store: QdrantVectorStore,
        query:str,
        k: int=3
)-> list[Document]:
    """
    Searches vector store for most relevant chunks.

    What happens internally:
    1. Converts query to a vector using Voyage AI
    2. Compares query vector against all stored vectors
    3. Returns k most similar chunks

    similarity_search
    → finds chunks whose vectors are closest to query vector
    → closest = most similar meaning
    → NOT keyword matching — semantic meaning matching

    k=3
    → return top 3 most relevant chunks
    → why 3? enough context without overwhelming Claude
    → in production you'd tune this based on chunk size
    """

    results = vector_store.similarity_search(query,k=k)
    return results


# ── Build full pipeline ───────────────────────────────────────────

def build_rag_pipeline(force_rebuild: bool=False) -> tuple:
    """
    Builds or loads the complete RAG pipeline.

    force_rebuild=True  → always rebuild from scratch
                          use when document changes
    force_rebuild=False → load existing if available
                          faster — skips re-embedding
    """

    embeddings = create_embeddings()
    index_path = QDRANT_INDEX

    if os.path.exists(index_path) and not force_rebuild:
        print("Found existing index - loading from disk")
        vector_store = load_vector_store(embeddings, index_path)
    else:
        print("Building new index")
        documents = load_documents(POLICY_FILE)
        chunks = split_documents(documents)
        vector_store= create_vector_store(
            chunks,
            embeddings,
            index_path
        )

    return vector_store,embeddings


if __name__ == "__main__":
    vector_store, embeddings = build_rag_pipeline(force_rebuild=False)
    print("\n" + "="*50)
    print("Testing RAG search")
    print("="*50)

    test_questions = [
        "What is the refund policy?",
        "How much does the Premium plan cost?",
        "What are the support hours for Basic plan?",
        "Does TechCorp sell customer data?"
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        results = search(vector_store,question,k=2)
        print(f"Found {len(results)} relevant chunks:")

        for i,doc in enumerate(results):
            print(f"  Chunk {i+1}: {doc.page_content[:150]}...")








