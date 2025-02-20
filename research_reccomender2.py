print("1️⃣ Importing libraries in research_recommender2.py...")  # Debug step 1
import arxiv
import pandas as pd
import numpy as np
import faiss
import nltk
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

print("2️⃣ Downloading NLTK resources...")  # Debug step 2
nltk.download('wordnet')


class ArxivFetcher:
    """
    Fetches research papers from Arxiv API.
    Implements retry mechanism for reliability.
    """
    print("3️⃣ Initializing ArxivFetcher class...")  # Debug step 3

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch(self, query="cat:cs.LG", max_results=100):
        print(f"🔍 Fetching {max_results} papers for query: {query}")  # Debug step 4
        client = arxiv.Client(page_size=100, delay_seconds=3)
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
        results = list(client.results(search))

        print(f"📄 Total papers fetched: {len(results)}")  # Debug step 5
        if len(results) == 0:
            print("⚠️ No papers retrieved! Check query or network connection.")

        df = pd.DataFrame([{
            'id': result.entry_id.split('/')[-1],
            'title': result.title,
            'abstract': result.summary,
            'authors': [a.name for a in result.authors],
            'published': result.published.date(),
            'source': 'arxiv'
        } for result in results])

        print(f"✅ Successfully converted {len(df)} papers into DataFrame.")  # Debug step 6
        return df


class EmbeddingSystem:
    """
    Embedding System to encode research papers and perform similarity search.
    Uses FAISS for fast indexing.
    """
    print("4️⃣ Initializing EmbeddingSystem class...")  # Debug step 7

    def __init__(self):
        print("🔵 Loading sentence transformer model...")  # Debug step 8
        self.model = SentenceTransformer('all-mpnet-base-v2')  # Load embedding model
        self.index = faiss.IndexFlatIP(768)  # FAISS index for similarity search
        self.metadata = pd.DataFrame()

    def generate_embeddings(self, texts: list) -> np.ndarray:
        print(f"🔵 Generating embeddings for {len(texts)} texts...")  # Debug step 9
        embeddings = self.model.encode(texts).astype('float32')
        print(f"✅ Embeddings generated: {embeddings.shape}")  # Debug step 10
        return embeddings

    def process_papers(self, df: pd.DataFrame):
        print("🔵 Processing papers for FAISS indexing...")  # Debug step 11
        if df.empty:
            print("⚠️ No papers to process. Skipping FAISS indexing.")
            return

        df['clean_text'] = df['title'] + ' [SEP] ' + df['abstract']  # Limit abstract to 1000 chars
        embeddings = self.generate_embeddings(df['clean_text'].tolist())  # Convert text to embeddings

        print("🔵 Normalizing embeddings for FAISS...")  # Debug step 12
        faiss.normalize_L2(embeddings)  # Normalize embeddings for cosine similarity
        self.index.add(embeddings)  # Add embeddings to FAISS
        print(f"✅ FAISS index now contains {self.index.ntotal} embeddings.")  # Debug step 13

        self.metadata = df.copy()
        self.metadata['embedding'] = list(embeddings)  # Store embeddings in metadata
        print("✅ Metadata stored with embeddings.")  # Debug step 14

    def recommend(self, text=None, paper_id=None, k=5):
        print("🔵 Getting recommendations...")  # Debug step 15
        if text:
            print(f"🟡 Searching recommendations for input text: {text[:50]}...")  # Show first 50 chars
            query = self.model.encode([text]).astype('float32')
        elif paper_id:
            print(f"🟡 Searching recommendations for paper ID: {paper_id}")  # Debug step 16
            query = np.array([self.metadata[self.metadata['id'] == paper_id]['embedding'].values[0]])

        print("🔵 Normalizing query embedding for FAISS search...")  # Debug step 17
        faiss.normalize_L2(query)  # Normalize query embedding
        distances, indices = self.index.search(query, k)  # Perform similarity search

        print(f"✅ Top {k} recommendations found.")  # Debug step 18
        recommended_papers = self.metadata.iloc[indices[0]].assign(similarity=1 - distances[0])

        print("📄 Recommended Papers:")
        for i, row in recommended_papers.iterrows():
            print(f"   {i + 1}. {row['title']} (Similarity: {row['similarity']:.2f})")  # Debug step 19

        return recommended_papers
