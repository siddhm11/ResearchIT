import arxiv
import pandas as pd
import numpy as np
import faiss
import nltk
import re
import logging
import datetime
import json
import requests
from bs4 import BeautifulSoup
import time
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Union, Tuple, Optional, Any
from multiprocessing import Pool, cpu_count
from fuzzywuzzy import fuzz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('research_recommender')

# Download required NLTK resources

nltk_resources = ["punkt", "stopwords", "wordnet"]
for resource in nltk_resources:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)



class TextPreprocessor:
    """
    Advanced text preprocessing for research papers.
    Implements efficient cleaning, tokenization, and normalization.
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Add research-specific stopwords
        research_stopwords = {'et', 'al', 'fig', 'figure', 'table', 'eq', 'equation', 'ref'}
        self.stop_words.update(research_stopwords)
        self.lemmatizer = WordNetLemmatizer()
        # Cache for lemmatized words to avoid redundant processing
        self.lemma_cache = {}
        
    def clean_text(self, text: str) -> str:
        """Clean text by removing special characters and normalizing whitespace"""
        if not text:
            return ""
            
        # Replace line breaks and tabs with spaces
        text = re.sub(r'[\n\t\r]', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove LaTeX equations (often between $ symbols)
        text = re.sub(r'\$+[^$]+\$+', ' equation ', text)
        
        # Remove citations like [1], [2,3], etc.
        text = re.sub(r'\[\d+(,\s*\d+)*\]', '', text)
        
        # Remove redundant spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def cached_lemmatize(self, word: str) -> str:
        """Lemmatize with caching for performance"""
        if word not in self.lemma_cache:
            self.lemma_cache[word] = self.lemmatizer.lemmatize(word)
        return self.lemma_cache[word]
    
    def process_text(self, text: str, lemmatize: bool = True) -> str:
        """Process text with tokenization, stopword removal, and optional lemmatization"""
        if not text:
            return ""
            
        # Clean the text first
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        filtered_tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words]
        
        # Apply lemmatization if requested
        if lemmatize:
            # Only lemmatize tokens that are likely nouns or informative terms
            # This selective approach balances precision and performance
            if len(filtered_tokens) > 100:  # For long texts, be selective
                processed_tokens = []
                for token in filtered_tokens:
                    if len(token) > 3:  # Focus on longer words that are more likely to be significant
                        processed_tokens.append(self.cached_lemmatize(token))
                    else:
                        processed_tokens.append(token)
                return ' '.join(processed_tokens)
            else:
                return ' '.join([self.cached_lemmatize(t) for t in filtered_tokens])
        
        return ' '.join(filtered_tokens)
    
    def batch_process(self, texts: List[str], lemmatize: bool = True, n_jobs: int = None) -> List[str]:
        """Process multiple texts in parallel"""
        if not n_jobs:
            n_jobs = max(1, cpu_count() - 1)  # Use all cores except one by default
            
        # For small batches, don't use multiprocessing overhead
        if len(texts) < 10:
            return [self.process_text(text, lemmatize) for text in texts]
            
        with Pool(n_jobs) as pool:
            return pool.starmap(self.process_text, [(text, lemmatize) for text in texts])


class CitationFetcher:
    """
    Fetches citation information for papers to determine their impact and quality.
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Cache to avoid repeated requests
        self.citation_cache = {}
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_citation_count(self, paper_id: str, title: str = None) -> dict:
        """
        Get citation count and other metrics for a paper
        
        Args:
            paper_id: arXiv ID of the paper
            title: Title of the paper (for fallback search)
            
        Returns:
            Dictionary with citation metrics
        """
        # Check cache first
        if paper_id in self.citation_cache:
            return self.citation_cache[paper_id]
            
        metrics = {
            'citation_count': 0,
            'h_index': 0,
            'journal_impact': 0.0,
            'last_updated': datetime.datetime.now().isoformat()
        }
        
        # Try to get citation info from Semantic Scholar
        try:
            url = f"https://api.semanticscholar.org/v1/paper/arXiv:{paper_id}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                metrics['citation_count'] = data.get('citationCount', 0)
                # Store the influentialCitationCount as a bonus metric if available
                if 'influentialCitationCount' in data:
                    metrics['influential_citations'] = data['influentialCitationCount']
                    
                # Store venue information if available
                if 'venue' in data and data['venue']:
                    metrics['venue'] = data['venue']
                    
                # Store in cache
                self.citation_cache[paper_id] = metrics
                logger.info(f"Retrieved citation data for {paper_id}: {metrics['citation_count']} citations")
                return metrics
        except Exception as e:
            logger.warning(f"Error fetching citation data from Semantic Scholar for {paper_id}: {e}")
            
        # Fallback to Google Scholar if title is provided
        if title:
            try:
                # Wait to avoid rate limiting
                time.sleep(2)
                
                # Search Google Scholar using the paper title
                search_query = title.replace(' ', '+')
                url = f"https://scholar.google.com/scholar?q={search_query}"
                
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for citation information
                    for result in soup.select('.gs_ri'):
                        result_title = result.select_one('.gs_rt')
                        if result_title and self._title_similarity(title, result_title.text.strip()) > 0.8:
                            # Found a matching paper, extract citation count
                            citation_info = result.select_one('.gs_fl')
                            if citation_info:
                                citation_text = citation_info.text
                                citation_match = re.search(r'Cited by (\d+)', citation_text)
                                if citation_match:
                                    metrics['citation_count'] = int(citation_match.group(1))
                                    logger.info(f"Retrieved citation data from Google Scholar for {paper_id}: {metrics['citation_count']} citations")
                                    break
            except Exception as e:
                logger.warning(f"Error fetching citation data from Google Scholar for {paper_id}: {e}")
        
        # Store in cache
        self.citation_cache[paper_id] = metrics
        return metrics
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two paper titles"""
        # Simple implementation using character-level similarity
        title1 = title1.lower()
        title2 = title2.lower()
        
        # Remove common prefixes/suffixes and special characters
        for prefix in ['the ', 'a ', 'an ']:
            if title1.startswith(prefix):
                title1 = title1[len(prefix):]
            if title2.startswith(prefix):
                title2 = title2[len(prefix):]
                
        title1 = re.sub(r'[^\w\s]', '', title1)
        title2 = re.sub(r'[^\w\s]', '', title2)
        
        # Check if one is substring of another
        if title1 in title2 or title2 in title1:
            return 0.9
            
        # Calculate Jaccard similarity of words
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_author_impact(self, authors: List[str]) -> dict:
        """
        Get impact metrics for paper authors
        
        Args:
            authors: List of author names
            
        Returns:
            Dictionary with author impact metrics
        """
        # This is a simplified implementation
        # In a production system, you would query academic databases
        
        impact_metrics = {
            'max_h_index': 0,
            'avg_h_index': 0,
            'total_citations': 0
        }
        
        # Real implementation would query academic APIs or databases
        # For example: Semantic Scholar, Google Scholar, or ORCID
        
        return impact_metrics


class PaperQualityAssessor:
    """
    Assesses the quality of research papers based on various metrics.
    """
    
    def __init__(self, citation_fetcher: CitationFetcher = None):
        self.citation_fetcher = citation_fetcher or CitationFetcher()
        # For normalization of scores across different metrics
        self.max_citation_count = 1000  # Will be updated as we process papers
        
    def assess_paper_quality(self, paper: dict) -> float:
        """
        Calculate a quality score for a paper
        
        Args:
            paper: Paper metadata dictionary
            
        Returns:
            Quality score between 0 and 1
        """
        # Get citation metrics
        citation_metrics = self.citation_fetcher.get_citation_count(
            paper_id=paper['id'],
            title=paper['title']
        )
        
        # Basic quality score components
        citation_score = self._normalize_citation_count(citation_metrics['citation_count'])
        recency_score = self._calculate_recency_score(paper['published'])
        
        # Consider venue quality if available
        venue_score = 0.0
        if 'venue' in citation_metrics:
            venue_score = self._assess_venue_quality(citation_metrics['venue'])
            
        # Consider author impact
        author_score = 0.0
        if 'authors' in paper:
            author_metrics = self.citation_fetcher.get_author_impact(paper['authors'])
            author_score = min(author_metrics.get('max_h_index', 0) / 50, 1.0) * 0.5
            
        # Analysis of paper content
        content_score = self._analyze_paper_content(paper)
        
        # Calculate weighted final score
        # Weights can be adjusted based on importance of different factors
        weights = {
            'citation': 0.4,      # Citations strongly indicate quality
            'recency': 0.2,       # Recent papers might be more relevant
            'venue': 0.1,         # Published venue indicates peer review
            'author': 0.1,        # Author reputation matters
            'content': 0.2        # Content analysis for inherent quality
        }
        
        quality_score = (
            weights['citation'] * citation_score +
            weights['recency'] * recency_score +
            weights['venue'] * venue_score +
            weights['author'] * author_score +
            weights['content'] * content_score
        )
        
        # Ensure score is between 0 and 1
        quality_score = max(0.0, min(1.0, quality_score))
        
        logger.info(f"Quality score for {paper['id']}: {quality_score:.2f}")
        
        return quality_score
    
    def _normalize_citation_count(self, citation_count: int) -> float:
        """Normalize citation count to a score between 0 and 1"""
        # Update maximum citation count seen so far for better normalization
        self.max_citation_count = max(self.max_citation_count, citation_count)
        
        # Use logarithmic scaling to handle papers with vastly different citation counts
        if citation_count == 0:
            return 0.0
        log_citations = np.log1p(citation_count)
        log_max = np.log1p(self.max_citation_count)
        
        return log_citations / log_max
    
    def _calculate_recency_score(self, published_date: Union[str, datetime.date]) -> float:
        """Calculate recency score (newer papers score higher)"""
        if isinstance(published_date, str):
            try:
                published_date = datetime.datetime.strptime(published_date, "%Y-%m-%d").date()
            except ValueError:
                published_date = datetime.datetime.now().date()
                
        days_old = (datetime.datetime.now().date() - published_date).days
        
        # Papers less than 6 months old get high scores
        if days_old < 180:
            return 0.8 + (180 - days_old) / 900  # Max 1.0 for very recent papers
        
        # Papers between 6 months and 3 years get moderate scores
        elif days_old < 1095:
            return 0.4 + (1095 - days_old) / 2300
            
        # Papers older than 3 years get lower scores
        else:
            return max(0.1, 0.4 - (days_old - 1095) / 10000)
    
    def _assess_venue_quality(self, venue: str) -> float:
        """Assess quality of publication venue"""
        # This is a simplified implementation
        # In a real system, you would have a database of journal/conference rankings
        
        # Check for top-tier venues (examples from CS/ML)
        top_venues = {
            'NeurIPS': 1.0,
            'ICML': 1.0,
            'ICLR': 1.0,
            'CVPR': 1.0,
            'ECCV': 0.95,
            'ACL': 0.95,
            'EMNLP': 0.9,
            'JMLR': 0.95,
            'TPAMI': 0.95,
            'Nature': 1.0,
            'Science': 1.0,
            'Cell': 0.95
        }
        
        # Check if venue matches or contains a top venue name
        for top_venue, score in top_venues.items():
            if top_venue.lower() == venue.lower() or top_venue.lower() in venue.lower():
                return score
                
        # Default score for unknown venues
        return 0.3
    
    def _analyze_paper_content(self, paper: dict) -> float:
        """Analyze paper content for quality indicators"""
        score = 0.5  # Default score
        
        # Check for presence of abstract
        if 'abstract' in paper and paper['abstract'] and len(paper['abstract']) > 100:
            # Abstract length and complexity can indicate paper quality
            abstract_length = len(paper['abstract'])
            if abstract_length > 1500:
                score += 0.15
            elif abstract_length > 800:
                score += 0.1
            elif abstract_length > 400:
                score += 0.05
                
            # Check for key quality indicators in abstract
            quality_indicators = [
                'novel', 'state-of-the-art', 'state of the art', 'sota',
                'outperform', 'improve', 'contribution', 'breakthrough',
                'demonstrate', 'experiment', 'dataset'
            ]
            
            abstract_lower = paper['abstract'].lower()
            indicator_count = sum(1 for indicator in quality_indicators if indicator in abstract_lower)
            score += min(indicator_count * 0.02, 0.1)
            
            # Penalize for vague language (might indicate lower quality)
            vague_terms = ['may', 'might', 'could', 'possibly', 'perhaps', 'potential']
            vague_count = sum(1 for term in vague_terms if f" {term} " in f" {abstract_lower} ")
            score -= min(vague_count * 0.02, 0.1)
            
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))


class ArxivFetcher:
    """
    Fetches research papers from Arxiv API with enhanced filtering.
    Implements retry mechanism for reliability.
    """
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch(self, query: str = "cat:cs.LG", max_results: int = 100, date_start: Optional[str] = None, date_end: Optional[str] = None, sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance) -> pd.DataFrame:
        logger.info(f"Fetching {max_results} papers for query: {query}")

        # Initialize client with conservative rate-limiting
        client = arxiv.Client(page_size=100, delay_seconds=3)
        search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_by)

        try:
            results = list(client.results(search))
            logger.info(f"Total papers fetched: {len(results)}")

            if len(results) == 0:
                logger.warning("⚠️ No papers retrieved! Check query or network connection.")
                return pd.DataFrame()

            papers = []
            for result in results:
                paper = {
                    'id': result.entry_id.split('/')[-1],
                    'title': result.title,
                    'abstract': result.summary,
                    'authors': [a.name for a in result.authors],
                    'primary_category': result.primary_category,
                    'categories': result.categories,
                    'published': result.published.date(),
                    'pdf_url': result.pdf_url,
                    'source': 'arxiv'
                }
                papers.append(paper)

            df = pd.DataFrame(papers)
            logger.info(f"✅ Successfully converted {len(df)} papers into DataFrame.")
            return df

        except Exception as e:
            logger.error(f"❌ Error fetching papers: {e}")
            return pd.DataFrame()

    def search_by_keywords(self, 
                          keywords: List[str], 
                          categories: List[str] = None,
                          max_results: int = 100,
                          date_start: Optional[str] = None, 
                          date_end: Optional[str] = None) -> pd.DataFrame:
        """
        Search for papers by keywords and categories
        
        Args:
            keywords: List of keywords to search for
            categories: List of arXiv categories (e.g. ['cs.LG', 'cs.AI'])
            max_results: Maximum number of papers to fetch
            date_start: Start date in format YYYY-MM-DD
            date_end: End date in format YYYY-MM-DD
            
        Returns:
            DataFrame with search results
        """
        # Build keyword part of query
        keyword_query = " AND ".join([f"all:{kw}" for kw in keywords])
        
        # Add categories if specified
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            query = f"({keyword_query}) AND ({cat_query})"
        else:
            query = keyword_query
            
        return self.fetch(
            query=query,
            max_results=max_results,
            date_start=date_start,
            date_end=date_end,
            sort_by=arxiv.SortCriterion.Relevance
        )
    
    def search_seminal_papers(self, topic: str, max_results: int = 10) -> pd.DataFrame:
        """
        Search for potentially seminal papers on a topic
        
        Args:
            topic: Research topic to find seminal papers for
            max_results: Maximum number of papers to fetch
            
        Returns:
            DataFrame with search results
        """
        # Explicitly search for survey papers and reviews which often cite seminal works
        query = f'all:"{topic}" AND (all:"survey" OR all:"review" OR all:"overview")'
        
        # Sort by submission date to get established papers (older papers tend to be more cited)
        # This is a better proxy than using Relevance which may not correlate with citations
        return self.fetch(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

class EmbeddingSystem:
    """
    Embedding System to encode research papers and perform similarity search.
    Uses FAISS for fast indexing and optimized vector representations.
    """
    
    #claude added this code to limit the size of the raw_embeddings cache to prevent memory issues
    
    def limit_embedding_cache(self, max_size: int = 10000):
        """
        Limit the size of the raw_embeddings cache to prevent memory issues
        
        Args:
            max_size: Maximum number of embeddings to keep in cache
        """
        if len(self.raw_embeddings) > max_size:
            logger.info(f"Trimming embedding cache from {len(self.raw_embeddings)} to {max_size} entries")
            # Keep only the most recent max_size entries
            # Convert to list of (key, value) tuples, sort by keys or values if needed
            items = list(self.raw_embeddings.items())
            # For simplicity, we'll just keep the last max_size items
            # In practice, you might want a more sophisticated strategy
            self.raw_embeddings = dict(items[-max_size:])
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        logger.info(f"Loading sentence transformer model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded with embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
        # Create two indices:
        # 1. A flat index for accurate reconstruction
        self.flat_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # 2. IVF index for faster search with minimal accuracy loss
        self.quantizer = faiss.IndexFlatIP(self.embedding_dim)
        # Use 4x sqrt(n) clusters for better balance of speed and accuracy
        # We'll initialize with 100 clusters and retrain as needed
        self.index = faiss.IndexIVFFlat(self.quantizer, self.embedding_dim, 100, faiss.METRIC_INNER_PRODUCT)
        self.index_trained = False
        
        # Store embeddings and their mapping to papers
        self.metadata = pd.DataFrame()
        # Store raw embeddings for direct lookup
        self.raw_embeddings = {}  # Map paper_id to embedding vector
        
        self.preprocessor = TextPreprocessor()
        
    def generate_embeddings(self, texts: list) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        # Process in batches for memory efficiency
        batch_size = 64  # Increased from 32 for MiniLM
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False).astype('float32')
            all_embeddings.append(batch_embeddings)
            
        embeddings = np.vstack(all_embeddings)
        logger.info(f"Embeddings generated: {embeddings.shape}")
        return embeddings
    
    def prepare_text_for_embedding(self, title: str, abstract: str) -> str:
        """Ensure title and abstract are processed even if missing."""
        clean_title = self.preprocessor.clean_text(title) if title else "Untitled Paper"
        clean_abstract = self.preprocessor.clean_text(abstract) if abstract else "No abstract available"
        return f"{clean_title} [SEP] {clean_title} [SEP] {clean_abstract}"

    
    def process_papers(self, df: pd.DataFrame, preprocess: bool = True) -> None:
        if df.empty:
            logger.warning("⚠️ No papers to process, skipping FAISS indexing.")
            return

        if "title" not in df.columns or "abstract" not in df.columns:
            logger.error("❌ Missing required columns ('title' or 'abstract') in DataFrame.")
            return

        # Ensure clean_text column exists
        df['clean_text'] = df.apply(lambda row: self.prepare_text_for_embedding(row['title'], row.get('abstract', "")), axis=1)

        if df['clean_text'].isnull().all():
            logger.error("❌ All clean_text values are empty! Stopping FAISS processing.")
            return

        logger.info(f"✅ Processing {len(df)} papers into FAISS index.")
        
        # Generate embeddings
        embeddings = self.generate_embeddings(df['clean_text'].tolist())

        # Train IVF index if necessary
        if not self.index_trained or self.index.ntotal < 1000:
            n_clusters = min(4 * int(np.sqrt(len(df) + self.index.ntotal)), 256)
            n_clusters = max(n_clusters, 100)

            logger.info(f"Training IVF index with {n_clusters} clusters...")

            self.index = faiss.IndexIVFFlat(self.quantizer, self.embedding_dim, n_clusters, faiss.METRIC_INNER_PRODUCT)

            if len(embeddings) < n_clusters:
                logger.warning(f"Not enough vectors to train {n_clusters} clusters. Using simple FAISS index.")
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Fallback to Flat index
                self.index_trained = True
            else:
                self.index.train(embeddings)
                self.index_trained = True

        # Add embeddings to FAISS index
        if self.index_trained:
            self.index.add(embeddings)
            logger.info(f"📌 FAISS index now contains {self.index.ntotal} embeddings.")
        else:
            logger.warning("Index not trained. Cannot add vectors.")
            return

        # Store metadata
        paper_df = df.copy()
        current_size = len(self.metadata)
        paper_df['embedding_idx'] = list(range(current_size, current_size + len(df)))

        if self.metadata.empty:
            self.metadata = paper_df
        else:
            self.metadata = pd.concat([self.metadata, paper_df], ignore_index=True)

        # Preprocess text if required
        if preprocess:
            logger.info("Adding preprocessed text for improved recommendations...")
            processed_text_map = {}

            if len(df) > 50:
                processed_texts = self.preprocessor.batch_process(df['clean_text'].tolist(), lemmatize=True)
                for i, (_, row) in enumerate(df.iterrows()):
                    processed_text_map[row['id']] = processed_texts[i]
            else:
                for _, row in df.iterrows():
                    processed_text = self.preprocessor.process_text(row['clean_text'], lemmatize=True)
                    processed_text_map[row['id']] = processed_text

            # Apply processed text mapping
            self.metadata.loc[self.metadata['id'].isin(processed_text_map.keys()), 'processed_text'] = \
                self.metadata.loc[self.metadata['id'].isin(processed_text_map.keys()), 'id'].map(processed_text_map)

        logger.info("✅ Metadata stored with embeddings.")

    def get_paper_embedding(self, paper_id: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a specific paper ID
        
        Args:
            paper_id: ID of the paper
            
        Returns:
            Numpy array containing the embedding vector or None if not found
        """
        # First try direct lookup from stored raw embeddings (fastest)
        if paper_id in self.raw_embeddings:
            return self.raw_embeddings[paper_id].reshape(1, -1)
        
        # Fallback to metadata lookup and flat index reconstruction
        paper_data = self.metadata[self.metadata['id'] == paper_id]
        if paper_data.empty:
            logger.warning(f"Paper ID {paper_id} not found in index.")
            return None
        
        # Get embedding index for this paper
        if 'embedding_idx' not in paper_data.columns:
            logger.warning(f"No embedding index for paper ID {paper_id}")
            return None
            
        embedding_idx = int(paper_data['embedding_idx'].iloc[0])
        
        # Reconstruct embedding from flat index (reliable)
        try:
            vector = np.zeros((1, self.embedding_dim), dtype='float32')
            vector[0] = self.flat_index.reconstruct(embedding_idx)
            return vector
        except Exception as e:
            logger.error(f"Error reconstructing embedding for paper ID {paper_id}: {e}")
            return None

    def recommend(self, 
              text: str = None, 
              paper_id: str = None,
              user_preferences: np.ndarray = None,
              k: int = 5, 
              filter_criteria: Dict = None,
              nprobe: int = 10,
              quality_assessor: Optional['PaperQualityAssessor'] = None) -> pd.DataFrame:
        """
        Get recommendations based on text, paper_id, or user preferences
        
        Args:
            text: Input text to find similar papers
            paper_id: ID of paper to find similar papers to
            user_preferences: Pre-computed user preference vector
            k: Number of recommendations to return
            filter_criteria: Dictionary of metadata filters to apply
            nprobe: Number of clusters to probe in IVF index (higher = more accurate but slower)
            quality_assessor: PaperQualityAssessor instance for quality scoring
            
        Returns:
            DataFrame with recommended papers and similarity scores
        """
        if not self.index_trained or self.index.ntotal == 0:
            logger.warning("Index not trained or empty. Cannot recommend papers.")
            return pd.DataFrame()

        if hasattr(self.index, 'nprobe'):  # Set nprobe only if using IVF index
            self.index.nprobe = nprobe

        query_vector = None
        matched_titles = []
        
        # 🔹 Case 1: Text query
        if text:
            logger.info(f"Generating embedding for query text: {text[:50]}...")
            clean_text = self.preprocessor.clean_text(text).lower()

            # 🔹 Perform fuzzy matching with titles
            matched_titles = [
                paper for paper in self.metadata.to_dict(orient="records")
                if fuzz.ratio(clean_text, paper["title"].lower()) >= 60  # Adjust threshold as needed
            ]

            # Sort fuzzy matches by highest similarity
            matched_titles.sort(key=lambda x: fuzz.ratio(clean_text, x["title"].lower()), reverse=True)

            if matched_titles:
                logger.info(f"Found {len(matched_titles)} fuzzy matches for '{text}'")

            # 🔹 Encode query text into embedding vector
            query_vector = self.model.encode([clean_text])[0].astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_vector)

        # 🔹 Case 2: Paper ID query
        elif paper_id:
            logger.info(f"Finding papers similar to paper_id: {paper_id}")
            query_vector = self.get_paper_embedding(paper_id)
            
            if query_vector is None:
                logger.warning(f"Could not retrieve embedding for paper ID {paper_id}")
                return pd.DataFrame()

        # 🔹 Case 3: User Preferences
        elif user_preferences is not None:
            logger.info("Using provided user preference vector for recommendations")
            query_vector = user_preferences.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_vector)

        else:
            logger.error("No query provided. Please provide text, paper_id, or user_preferences.")
            return pd.DataFrame()
        
        # 🔹 Perform FAISS similarity search
        num_results = min(2 * k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, num_results)

        # 🔹 Convert FAISS results into DataFrame
        results_df = pd.DataFrame({
            'embedding_idx': indices[0],
            'similarity_score': scores[0]
        })

        # 🔹 Merge FAISS results with metadata
        results_with_metadata = []
        for _, row in results_df.iterrows():
            embedding_idx = int(row['embedding_idx'])
            metadata_matches = self.metadata[self.metadata['embedding_idx'] == embedding_idx]

            if not metadata_matches.empty:
                paper_data = metadata_matches.iloc[0].to_dict()
                paper_data['similarity_score'] = row['similarity_score']
                results_with_metadata.append(paper_data)

        if not results_with_metadata:
            logger.warning("No metadata matches found for search results.")
            return pd.DataFrame()

        results_df = pd.DataFrame(results_with_metadata)

        # 🔹 Merge fuzzy title matches **with FAISS results**, removing duplicates
        if matched_titles:
            fuzzy_df = pd.DataFrame(matched_titles)
            results_df = pd.concat([fuzzy_df, results_df]).drop_duplicates(subset=['id']).reset_index(drop=True)

        # 🔹 Ensure at least `k` results by using fallback recommendations
        if len(results_df) < k:
            logger.warning("Not enough results, fetching fallback recommendations.")
            fallback_papers = self.get_fallback_recommendations(k - len(results_df))
            results_df = pd.concat([results_df, fallback_papers]).reset_index(drop=True)

        # 🔹 Remove the query paper from results if searching by paper_id
        if paper_id:
            results_df = results_df[results_df['id'] != paper_id]

        # 🔹 Apply filters if provided
        if filter_criteria:
            for column, value in filter_criteria.items():
                if column in results_df.columns:
                    if isinstance(value, list):
                        results_df = results_df[results_df[column].isin(value)]
                    elif isinstance(value, dict) and 'min' in value and 'max' in value:
                        results_df = results_df[(results_df[column] >= value['min']) & (results_df[column] <= value['max'])]
                    elif isinstance(value, dict) and 'after' in value:
                        results_df = results_df[results_df[column] >= value['after']]
                    elif isinstance(value, dict) and 'before' in value:
                        results_df = results_df[results_df[column] <= value['before']]
                    else:
                        results_df = results_df[results_df[column] == value]

        # 🔹 Apply quality assessment if enabled
        if quality_assessor and not results_df.empty:
            logger.info("Applying quality assessment to recommendations...")
            results_df['quality_score'] = results_df.apply(lambda paper: quality_assessor.assess_paper_quality(paper), axis=1)
            
            similarity_weight = 0.7  # 70% weight for similarity
            quality_weight = 0.3  # 30% weight for quality
            
            results_df['combined_score'] = (similarity_weight * results_df['similarity_score']) + (quality_weight * results_df['quality_score'])
            results_df = results_df.sort_values('combined_score', ascending=False)
        else:
            results_df = results_df.sort_values('similarity_score', ascending=False)

        # 🔹 Normalize similarity score to percentage
        results_df['similarity_percent'] = (results_df['similarity_score'] * 100).round(2)

        return results_df.head(k)

    
     
class ResearchRecommender:
    """
    Main class for the research paper recommendation system.
    Integrates fetching, preprocessing, embedding, citation analysis, and quality assessment.
    """
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.fetcher = ArxivFetcher()
        self.embedding_system = EmbeddingSystem(model_name=embedding_model)
        self.preprocessor = TextPreprocessor()
        self.citation_fetcher = CitationFetcher()
        self.quality_assessor = PaperQualityAssessor(citation_fetcher=self.citation_fetcher)
        
    def fetch_and_index(self, query: str = None, keywords: List[str] = None, categories: List[str] = None, max_results: int = 100):
        """Improved search function with fallback queries"""
        
        if keywords:
            query = " OR ".join([f'all:"{kw}"' for kw in keywords])
            
        if categories:
            cat_query = " OR ".join([f'cat:{cat}' for cat in categories])
            query = f"({query}) AND ({cat_query})"

        logger.info(f"Fetching papers with query: {query}")

        papers = self.fetcher.fetch(query=query, max_results=max_results)
        
        if papers.empty:
            logger.warning("⚠️ No papers found! Expanding search...")
            # Retry with broader query
            query = query.replace('"', '')  # Remove strict quotes
            papers = self.fetcher.fetch(query=query, max_results=max_results)

        if papers.empty:
            logger.error("❌ Still no results. Try different keywords/categories.")
            return pd.DataFrame()
        
        self.embedding_system.process_papers(papers)
        return papers

    
    def fetch_and_index(self, query: str = None, keywords: List[str] = None, categories: List[str] = None, 
                    max_results: int = 100, date_start: Optional[str] = None, date_end: Optional[str] = None):
        """Improved search function with optional date filtering"""

        if keywords:
            query = " OR ".join([f'all:"{kw}"' for kw in keywords])

        if categories:
            cat_query = " OR ".join([f'cat:{cat}' for cat in categories])
            query = f"({query}) AND ({cat_query})"

        logger.info(f"Fetching papers with query: {query}")

        # Now passing date_start and date_end correctly to fetch()
        papers = self.fetcher.fetch(query=query, max_results=max_results, date_start=date_start, date_end=date_end)

        if papers.empty:
            logger.warning("⚠️ No papers found! Expanding search...")
            query = query.replace('"', '')  # Remove strict quotes
            papers = self.fetcher.fetch(query=query, max_results=max_results, date_start=date_start, date_end=date_end)

        if papers.empty:
            logger.error("❌ Still no results. Try different keywords/categories.")
            return pd.DataFrame()

        self.embedding_system.process_papers(papers)
        return papers

        
    def find_seminal_papers(self, topic: str, max_results: int = 10) -> pd.DataFrame:
        """Find potentially seminal/highly-cited papers on a topic"""
        papers = self.fetcher.search_seminal_papers(topic, max_results)
        
        if not papers.empty:
            self.embedding_system.process_papers(papers)
            
        return papers
    
    def recommend(self, 
                 text: str = None, 
                 paper_id: str = None,
                 user_preferences: np.ndarray = None,
                 k: int = 5,
                 min_date: str = None,
                 max_date: str = None,
                 quality_aware: bool = True,
                 nprobe: int = 10) -> pd.DataFrame:
        """
        Get paper recommendations
        
        Args:
            text: Query text
            paper_id: Query paper ID
            user_preferences: User preference vector
            k: Number of recommendations
            min_date: Minimum date (YYYY-MM-DD)
            max_date: Maximum date (YYYY-MM-DD)
            quality_aware: Whether to use quality assessment in ranking
            nprobe: Number of clusters to probe in IVF index
            
        Returns:
            DataFrame with recommendations
        """
        # Prepare filter criteria if dates specified
        filter_criteria = None
        if min_date or max_date:
            filter_criteria = {
                'published': {
                    'min': min_date,
                    'max': max_date
                }
            }
            
        # Get recommendations
        return self.embedding_system.recommend(
            text=text,
            paper_id=paper_id,
            user_preferences=user_preferences,
            k=k,
            filter_criteria=filter_criteria,
            nprobe=nprobe,
            quality_assessor=self.quality_assessor if quality_aware else None
        )
    
    def assess_paper_quality(self, paper_id: str) -> float:
        """
        Assess the quality of a specific paper
        
        Args:
            paper_id: ID of the paper to assess
            
        Returns:
            Quality score between 0 and 1
        """
        paper_data = self.embedding_system.metadata[self.embedding_system.metadata['id'] == paper_id]
        if paper_data.empty:
            logger.warning(f"Paper ID {paper_id} not found in index.")
            return 0.0
            
        return self.quality_assessor.assess_paper_quality(paper_data.iloc[0])
    
    def get_citation_info(self, paper_id: str) -> dict:
        """
        Get citation information for a paper
        
        Args:
            paper_id: ID of the paper
            
        Returns:
            Dictionary with citation metrics
        """
        paper_data = self.embedding_system.metadata[self.embedding_system.metadata['id'] == paper_id]
        if paper_data.empty:
            logger.warning(f"Paper ID {paper_id} not found in index.")
            return {}
            
        return self.citation_fetcher.get_citation_count(
            paper_id=paper_id,
            title=paper_data['title'].iloc[0]
        )
    
    def save_index(self, filepath: str) -> None:
        """Save the FAISS index to disk"""
        if self.embedding_system.index.ntotal > 0:
            logger.info(f"Saving index with {self.embedding_system.index.ntotal} vectors to {filepath}")
            faiss.write_index(self.embedding_system.index, filepath)
            self.embedding_system.metadata.to_pickle(f"{filepath}_metadata.pkl")
            
            # Save citation cache
            with open(f"{filepath}_citation_cache.json", 'w') as f:
                json.dump(self.citation_fetcher.citation_cache, f)
                
            logger.info("Index, metadata and citation cache saved successfully")
        else:
            logger.warning("Cannot save empty index")
    
    def load_index(self, filepath: str) -> None:
        """Load a FAISS index from disk"""
        try:
            logger.info(f"Loading index from {filepath}")
            self.embedding_system.index = faiss.read_index(filepath)
            self.embedding_system.metadata = pd.read_pickle(f"{filepath}_metadata.pkl")
            
            # Load citation cache if exists
            try:
                with open(f"{filepath}_citation_cache.json", 'r') as f:
                    self.citation_fetcher.citation_cache = json.load(f)
            except FileNotFoundError:
                logger.warning("Citation cache not found, starting with empty cache")
                
            # Set index as trained
            if self.embedding_system.index.ntotal > 0:
                self.embedding_system.index_trained = True
                
            logger.info(f"Successfully loaded index with {self.embedding_system.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize the recommender
    recommender = ResearchRecommender()
    
    # Get recent papers on transformers
    papers = recommender.fetch_and_index(
        query="all:transformer AND all:attention", 
        max_results=50,
        date_start="2023-01-01",
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    # Also get the original "Attention is All You Need" paper
    seminal_papers = recommender.find_seminal_papers("attention transformer", max_results=5)
    
    # Get recommendations based on a query
    recommendations = recommender.recommend(
        text="Efficient transformer models for natural language processing", 
        k=5
    )
    
    # Print results
    print(f"\nTop recommendations:")
    for i, row in recommendations.iterrows():
        print(f"{i+1}. {row['title']} (Similarity: {row['similarity_score']:.2f})")
