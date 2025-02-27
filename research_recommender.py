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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('research_recommender')

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.warning(f"NLTK resource download issue: {e}")


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
    def fetch(self, 
              query: str = "cat:cs.LG", 
              max_results: int = 100, 
              date_start: Optional[str] = None, 
              date_end: Optional[str] = None,
              sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance) -> pd.DataFrame:
        """
        Fetch papers from arXiv with flexible querying options
        
        Args:
            query: arXiv query string (can include categories, keywords, etc.)
            max_results: maximum number of papers to fetch
            date_start: start date in format YYYY-MM-DD
            date_end: end date in format YYYY-MM-DD
            sort_by: sorting criterion (Relevance, SubmittedDate, LastUpdatedDate)
            
        Returns:
            DataFrame containing paper metadata
        """
        logger.info(f"Fetching {max_results} papers for query: {query}")
        
        # Build date filter if provided
        if date_start or date_end:
            date_filter = []
            if date_start:
                date_filter.append(f"submittedDate:[{date_start}000000 TO 999999999999]")
            if date_end:
                date_filter.append(f"submittedDate:[00000000000 TO {date_end}235959]")
            
            # Combine with the main query
            if "AND" in query or "OR" in query:
                query = f"({query}) AND {' AND '.join(date_filter)}"
            else:
                query = f"{query} AND {' AND '.join(date_filter)}"
                
        logger.info(f"Final query: {query}")
        
        # Initialize client with conservative rate-limiting
        client = arxiv.Client(page_size=100, delay_seconds=3)
        search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_by)
        
        try:
            results = list(client.results(search))
            logger.info(f"Total papers fetched: {len(results)}")
            
            if len(results) == 0:
                logger.warning("No papers retrieved! Check query or network connection.")
                return pd.DataFrame()
                
            # Process results into DataFrame
            papers = []
            for result in results:
                # Extract paper data
                paper = {
                    'id': result.entry_id.split('/')[-1],
                    'title': result.title,
                    'abstract': result.summary,
                    'authors': [a.name for a in result.authors],
                    'primary_category': result.primary_category,
                    'categories': result.categories,
                    'published': result.published.date(),
                    'updated': result.updated.date() if hasattr(result, 'updated') else None,
                    'doi': result.doi if hasattr(result, 'doi') else None,
                    'journal_ref': result.journal_ref if hasattr(result, 'journal_ref') else None,
                    'pdf_url': result.pdf_url,
                    'source': 'arxiv'
                }
                papers.append(paper)
                
            df = pd.DataFrame(papers)
            
            # Process text in a separate step to avoid slowing down API fetching
            logger.info(f"Successfully converted {len(df)} papers into DataFrame.")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching papers: {e}")
            raise
    
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
        # Search for highly cited papers (more citations = higher relevance in arXiv)
        query = f'all:"{topic}" AND all:"survey"'
        
        # No date filters, to ensure we get older seminal papers
        return self.fetch(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )


class EmbeddingSystem:
    """
    Embedding System to encode research papers and perform similarity search.
    Uses FAISS for fast indexing and optimized vector representations.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        logger.info(f"Loading sentence transformer model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded with embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
        # Use IVF index for faster search with minimal accuracy loss
        self.quantizer = faiss.IndexFlatIP(self.embedding_dim)
        # Use 4x sqrt(n) clusters for better balance of speed and accuracy
        # We'll initialize with 100 clusters and retrain as needed
        self.index = faiss.IndexIVFFlat(self.quantizer, self.embedding_dim, 100, faiss.METRIC_INNER_PRODUCT)
        self.index_trained = False
        
        self.metadata = pd.DataFrame()
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
        """
        Prepare paper text for embedding by combining title and abstract
        with special handling for better semantic representation
        """
        # Clean the title and abstract separately
        clean_title = self.preprocessor.clean_text(title)
        clean_abstract = self.preprocessor.clean_text(abstract)
        
        # Give more weight to the title by repeating it
        return f"{clean_title} [SEP] {clean_title} [SEP] {clean_abstract}"
    
    def process_papers(self, df: pd.DataFrame, preprocess: bool = True) -> None:
        """
        Process papers for FAISS indexing with optimized text preparation
        
        Args:
            df: DataFrame containing papers
            preprocess: Whether to apply text preprocessing
        """
        logger.info("Processing papers for FAISS indexing...")
        if df.empty:
            logger.warning("No papers to process. Skipping FAISS indexing.")
            return
            
        # Prepare text for embedding - combining title and abstract with special handling
        df['clean_text'] = df.apply(
            lambda row: self.prepare_text_for_embedding(row['title'], row['abstract']), 
            axis=1
        )
        
        # Generate embeddings
        embeddings = self.generate_embeddings(df['clean_text'].tolist())
        
        # Normalize embeddings for cosine similarity
        logger.info("Normalizing embeddings for FAISS...")
        faiss.normalize_L2(embeddings)
        
        # Train index if not trained or if we have a lot of new data
        if not self.index_trained or self.index.ntotal < 1000:
            # Determine optimal number of clusters based on data size
            n_clusters = min(4 * int(np.sqrt(len(df) + self.index.ntotal)), 256)
            n_clusters = max(n_clusters, 100)  # At least 100 clusters
            
            logger.info(f"Training IVF index with {n_clusters} clusters...")
            
            # Create new index with adjusted clusters
            self.index = faiss.IndexIVFFlat(self.quantizer, self.embedding_dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
            
            # Training requires at least n_clusters vectors
            if len(embeddings) < n_clusters:
                logger.warning(f"Not enough vectors to train {n_clusters} clusters. Using existing index.")
            else:
                self.index.train(embeddings)
                self.index_trained = True
        
        # Add to index
        self.index.add(embeddings)
        logger.info(f"FAISS index now contains {self.index.ntotal} embeddings.")
        
        # Store metadata
        if self.metadata.empty:
            self.metadata = df.copy()
        else:
            # Concatenate new metadata with existing
            # First, save the current embedding indices
            if 'embedding_idx' in self.metadata.columns:
                old_embeddings_count = self.index.ntotal - len(df)
                # Update existing indices if needed
                self.metadata['embedding_idx'] = list(range(old_embeddings_count))
                
                # Create indices for new data
                new_df = df.copy()
                new_df['embedding_idx'] = list(range(old_embeddings_count, self.index.ntotal))
                
                # Combine
                self.metadata = pd.concat([self.metadata, new_df], ignore_index=True)
            else:
                # First time adding embedding indices
                self.metadata['embedding_idx'] = list(range(self.index.ntotal))
        
        # Store normalized clean text for better recommendations
        if preprocess:
            logger.info("Adding preprocessed text for improved recommendations...")
            # Use parallel processing for preprocessing when dealing with many papers
            if len(df) > 50:
                processed_texts = self.preprocessor.batch_process(
                    df['clean_text'].tolist(),
                    lemmatize=True
                )
                for i, (_, row) in enumerate(df.iterrows()):
                    idx = self.metadata[self.metadata['id'] == row['id']].index
                    if len(idx) > 0:
                        self.metadata.at[idx[0], 'processed_text'] = processed_texts[i]
            else:
                for _, row in df.iterrows():
                    processed_text = self.preprocessor.process_text(row['clean_text'], lemmatize=True)
                    idx = self.metadata[self.metadata['id'] == row['id']].index
                    if len(idx) > 0:
                        self.metadata.at[idx[0], 'processed_text'] = processed_text
                
        logger.info("Metadata stored with embeddings.")
        
    # Complete the recommend method in EmbeddingSystem class
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
        user_preferences: User preference embedding vector
        k: Number of recommendations to return
        filter_criteria: Dictionary with metadata filters
        nprobe: Number of clusters to probe (higher = more accurate but slower)
        quality_assessor: Optional quality assessor for scoring papers
        
    Returns:
        DataFrame with recommendations
    """
    if not self.index_trained or self.index.ntotal == 0:
        logger.warning("Index not trained or empty. Cannot provide recommendations.")
        return pd.DataFrame()
        
    # Set nprobe for search - balance between speed and accuracy
    self.index.nprobe = nprobe
    
    # Get query vector
    query_vector = None
    
    if text is not None:
        # Clean and process the query text
        clean_text = self.preprocessor.clean_text(text)
        query_vector = self.model.encode([clean_text])[0].reshape(1, -1).astype('float32')
        
    elif paper_id is not None:
        # Find paper in metadata and use its embedding
        paper_idx = self.metadata[self.metadata['id'] == paper_id].index
        if len(paper_idx) == 0:
            logger.warning(f"Paper ID {paper_id} not found in metadata.")
            return pd.DataFrame()
            
        embedding_idx = self.metadata.loc[paper_idx[0], 'embedding_idx']
        
        # Get the embedding from the FAISS index
        query_vector = np.zeros((1, self.embedding_dim), dtype='float32')
        self.index.reconstruct(int(embedding_idx), query_vector[0])
        
    elif user_preferences is not None:
        # Use user preferences directly
        query_vector = user_preferences.reshape(1, -1).astype('float32')
        
    else:
        logger.error("No query provided. Must provide text, paper_id, or user_preferences.")
        return pd.DataFrame()
    
    # Normalize query vector for cosine similarity
    faiss.normalize_L2(query_vector)
    
    # Search for similar papers
    distances, indices = self.index.search(query_vector, k * 2)  # Get 2x results for filtering
    
    # Convert to dataframe for easier filtering
    results = pd.DataFrame({
        'embedding_idx': indices[0],
        'similarity_score': distances[0]
    })
    
    # Join with metadata
    results = results.merge(
        self.metadata, 
        left_on='embedding_idx', 
        right_on='embedding_idx', 
        how='inner'
    )
    
    # Apply filters if provided
    if filter_criteria:
        for field, value in filter_criteria.items():
            if field in results.columns:
                if isinstance(value, list):
                    results = results[results[field].isin(value)]
                elif isinstance(value, dict) and ('min' in value or 'max' in value):
                    if 'min' in value:
                        results = results[results[field] >= value['min']]
                    if 'max' in value:
                        results = results[results[field] <= value['max']]
                else:
                    results = results[results[field] == value]
    
    # Apply quality assessment if provided
    if quality_assessor:
        logger.info("Applying quality assessment to recommendations...")
        quality_scores = []
        
        for _, row in results.iterrows():
            quality_score = quality_assessor.assess_paper_quality(row.to_dict())
            quality_scores.append(quality_score)
            
        results['quality_score'] = quality_scores
        
        # Combined score: weighted combination of similarity and quality
        # Adjust the weights to prioritize similarity or quality
        similarity_weight = 0.7
        quality_weight = 0.3
        
        results['combined_score'] = (
            similarity_weight * results['similarity_score'] + 
            quality_weight * results['quality_score']
        )
        
        # Sort by combined score
        results = results.sort_values('combined_score', ascending=False)
    else:
        # Sort by similarity score only
        results = results.sort_values('similarity_score', ascending=False)
    
    # Return top k after all filtering
    return results.head(k)


# Add new class for user preference tracking
class UserPreferenceTracker:
    """
    Tracks and updates user preferences based on interactions with papers.
    Uses an evolving embedding to represent user interests.
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.users = {}  # Dictionary to store user preferences
        
    def initialize_user(self, user_id: str) -> None:
        """Initialize a new user with default preferences"""
        if user_id not in self.users:
            self.users[user_id] = {
                'preference_vector': np.zeros(self.embedding_dim, dtype='float32'),
                'interaction_count': 0,
                'liked_papers': set(),
                'disliked_papers': set(),
                'viewed_papers': set(),
                'categories_of_interest': {}
            }
            logger.info(f"Initialized new user: {user_id}")
    
    def update_preferences(self, 
                          user_id: str, 
                          paper_embedding: np.ndarray,
                          interaction_type: str = 'view',
                          paper_id: str = None,
                          paper_categories: List[str] = None) -> None:
        """
        Update user preferences based on interaction with a paper
        
        Args:
            user_id: User identifier
            paper_embedding: Embedding vector of the paper
            interaction_type: Type of interaction ('view', 'like', 'dislike', 'save')
            paper_id: ID of the paper
            paper_categories: Categories of the paper
        """
        if user_id not in self.users:
            self.initialize_user(user_id)
            
        user = self.users[user_id]
        
        # Define weights for different types of interactions
        weights = {
            'view': 0.1,
            'like': 0.5,
            'dislike': -0.3,
            'save': 0.7,
            'cite': 0.8
        }
        
        # Update interaction count
        user['interaction_count'] += 1
        
        # Track paper interaction
        if paper_id:
            if interaction_type == 'like':
                user['liked_papers'].add(paper_id)
                # Remove from disliked if previously disliked
                user['disliked_papers'].discard(paper_id)
            elif interaction_type == 'dislike':
                user['disliked_papers'].add(paper_id)
                # Remove from liked if previously liked
                user['liked_papers'].discard(paper_id)
            elif interaction_type in ['view', 'save', 'cite']:
                user['viewed_papers'].add(paper_id)
        
        # Track category interests
        if paper_categories:
            for category in paper_categories:
                user['categories_of_interest'][category] = user['categories_of_interest'].get(category, 0) + weights.get(interaction_type, 0.1)
        
        # Early return if no embedding to update
        if paper_embedding is None:
            return
            
        # Ensure the embedding is normalized
        paper_embedding_norm = paper_embedding.copy()
        faiss.normalize_L2(paper_embedding_norm.reshape(1, -1))
        paper_embedding_norm = paper_embedding_norm.flatten()
        
        # Calculate adaptive learning rate - decay over time but maintain responsiveness
        learn_rate = 1.0 / (1.0 + 0.1 * user['interaction_count'])
        
        # Apply interaction weight
        weight = weights.get(interaction_type, 0.1)
        update = weight * learn_rate * paper_embedding_norm
        
        # Update preference vector
        user['preference_vector'] += update
        
        # Normalize preference vector
        pref_norm = np.linalg.norm(user['preference_vector'])
        if pref_norm > 0:
            user['preference_vector'] /= pref_norm
            
        logger.info(f"Updated preferences for user {user_id} based on {interaction_type} interaction")
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """Get user preference information"""
        if user_id not in self.users:
            logger.warning(f"User {user_id} not found. Initializing with default preferences.")
            self.initialize_user(user_id)
            
        return self.users[user_id]
    
    def get_preference_vector(self, user_id: str) -> np.ndarray:
        """Get user preference embedding vector for recommendations"""
        if user_id not in self.users:
            logger.warning(f"User {user_id} not found. Initializing with default preferences.")
            self.initialize_user(user_id)
            
        return self.users[user_id]['preference_vector']
    
    def save_preferences(self, filepath: str) -> None:
        """Save user preferences to file"""
        data_to_save = {}
        
        # Convert numpy arrays to lists for JSON serialization
        for user_id, user_data in self.users.items():
            data_to_save[user_id] = {
                'preference_vector': user_data['preference_vector'].tolist(),
                'interaction_count': user_data['interaction_count'],
                'liked_papers': list(user_data['liked_papers']),
                'disliked_papers': list(user_data['disliked_papers']),
                'viewed_papers': list(user_data['viewed_papers']),
                'categories_of_interest': user_data['categories_of_interest']
            }
        
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f)
        
        logger.info(f"Saved user preferences to {filepath}")
    
    def load_preferences(self, filepath: str) -> None:
        """Load user preferences from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            for user_id, user_data in data.items():
                self.users[user_id] = {
                    'preference_vector': np.array(user_data['preference_vector'], dtype='float32'),
                    'interaction_count': user_data['interaction_count'],
                    'liked_papers': set(user_data['liked_papers']),
                    'disliked_papers': set(user_data['disliked_papers']),
                    'viewed_papers': set(user_data['viewed_papers']),
                    'categories_of_interest': user_data['categories_of_interest']
                }
                
            logger.info(f"Loaded preferences for {len(self.users)} users")
        except Exception as e:
            logger.error(f"Error loading user preferences: {e}")


# Create a more efficient RecommenderSystem class that ties everything together
class ResearchRecommender:
    """
    Complete research paper recommendation system with quality assessment and user preferences.
    """
    
    def __init__(self, 
                use_faster_model: bool = True,
                load_existing: bool = False,
                model_path: str = "./models",
                user_prefs_path: str = "./user_preferences.json"):
        """
        Initialize the recommendation system
        
        Args:
            use_faster_model: Whether to use a faster, more efficient model
            load_existing: Whether to load existing index and user preferences
            model_path: Path to save/load models
            user_prefs_path: Path to save/load user preferences
        """
        self.model_path = model_path
        self.user_prefs_path = user_prefs_path
        
        # Select model: faster vs more accurate
        if use_faster_model:
            # MiniLM is ~2x faster than MPNet with minimal quality loss
            model_name = 'all-MiniLM-L6-v2'
        else:
            # MPNet has better quality but is slower
            model_name = 'all-mpnet-base-v2'
            
        logger.info(f"Initializing with model: {model_name}")
        
        # Initialize components
        self.embedding_system = EmbeddingSystem(model_name=model_name)
        self.citation_fetcher = CitationFetcher()
        self.quality_assessor = PaperQualityAssessor(self.citation_fetcher)
        self.arxiv_fetcher = ArxivFetcher()
        self.user_tracker = UserPreferenceTracker(
            embedding_dim=self.embedding_system.embedding_dim
        )
        
        # Load existing data if requested
        if load_existing:
            self._load_state()
    
    def _load_state(self) -> None:
        """Load existing index and user preferences"""
        try:
            # Load FAISS index if exists
            index_path = f"{self.model_path}/paper_index.faiss"
            metadata_path = f"{self.model_path}/paper_metadata.pkl"
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                self.embedding_system.index = faiss.read_index(index_path)
                self.embedding_system.metadata = pd.read_pickle(metadata_path)
                self.embedding_system.index_trained = True
                logger.info(f"Loaded existing index with {self.embedding_system.index.ntotal} papers")
            
            # Load user preferences if exists
            if os.path.exists(self.user_prefs_path):
                self.user_tracker.load_preferences(self.user_prefs_path)
        except Exception as e:
            logger.error(f"Error loading existing state: {e}")
    
    def save_state(self) -> None:
        """Save current index and user preferences"""
        try:
            # Ensure directory exists
            os.makedirs(self.model_path, exist_ok=True)
            
            # Save FAISS index
            index_path = f"{self.model_path}/paper_index.faiss"
            metadata_path = f"{self.model_path}/paper_metadata.pkl"
            
            faiss.write_index(self.embedding_system.index, index_path)
            self.embedding_system.metadata.to_pickle(metadata_path)
            
            # Save user preferences
            self.user_tracker.save_preferences(self.user_prefs_path)
            
            logger.info("System state saved successfully")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def fetch_papers(self, 
                    query: str = None,
                    keywords: List[str] = None,
                    categories: List[str] = None,
                    max_results: int = 100,
                    date_start: str = None,
                    date_end: str = None) -> pd.DataFrame:
        """
        Fetch papers based on query or keywords
        
        Args:
            query: Direct arXiv query
            keywords: List of keywords to search for
            categories: List of arXiv categories
            max_results: Maximum number of papers to fetch
            date_start: Start date for papers
            date_end: End date for papers
            
        Returns:
            DataFrame with papers
        """
        if query:
            papers = self.arxiv_fetcher.fetch(
                query=query,
                max_results=max_results,
                date_start=date_start,
                date_end=date_end
            )
        elif keywords:
            papers = self.arxiv_fetcher.search_by_keywords(
                keywords=keywords,
                categories=categories,
                max_results=max_results,
                date_start=date_start,
                date_end=date_end
            )
        else:
            logger.error("Must provide either query or keywords")
            return pd.DataFrame()
            
        # Process papers
        if not papers.empty:
            self.embedding_system.process_papers(papers)
            
        return papers
    
    def assess_paper_quality(self, paper_id: str, paper_data: dict = None) -> float:
        """
        Assess the quality of a paper
        
        Args:
            paper_id: ID of the paper
            paper_data: Paper data if available, otherwise fetched from metadata
            
        Returns:
            Quality score between 0 and 1
        """
        if paper_data is None:
            # Try to get from metadata
            paper_idx = self.embedding_system.metadata[self.embedding_system.metadata['id'] == paper_id].index
            if len(paper_idx) == 0:
                logger.warning(f"Paper ID {paper_id} not found in metadata.")
                return 0.5
            
            paper_data = self.embedding_system.metadata.loc[paper_idx[0]].to_dict()
            
        return self.quality_assessor.assess_paper_quality(paper_data)
    
    def get_recommendations(self,
                           text: str = None,
                           paper_id: str = None,
                           user_id: str = None,
                           k: int = 5,
                           filter_criteria: Dict = None,
                           include_quality: bool = True) -> pd.DataFrame:
        """
        Get personalized paper recommendations
        
        Args:
            text: Query text
            paper_id: Paper ID to find similar papers
            user_id: User ID for personalized recommendations
            k: Number of recommendations
            filter_criteria: Filtering criteria
            include_quality: Whether to include quality assessment
            
        Returns:
            DataFrame with recommendations
        """
        user_preferences = None
        
        # If user_id is provided, get personalized recommendations
        if user_id:
            user_preferences = self.user_tracker.get_preference_vector(user_id)
            
            # If user has no preferences yet, fall back to text/paper_id
            if np.all(user_preferences == 0):
                user_preferences = None
                logger.info(f"User {user_id} has no preferences yet, falling back to content-based")
        
        # Get recommendations
        recommendations = self.embedding_system.recommend(
            text=text,
            paper_id=paper_id,
            user_preferences=user_preferences,
            k=k,
            filter_criteria=filter_criteria,
            quality_assessor=self.quality_assessor if include_quality else None
        )
        
        return recommendations
    
    def record_user_interaction(self,
                               user_id: str,
                               paper_id: str,
                               interaction_type: str = 'view') -> None:
        """
        Record user interaction with a paper and update preferences
        
        Args:
            user_id: User ID
            paper_id: Paper ID
            interaction_type: Type of interaction ('view', 'like', 'dislike', 'save', 'cite')
        """
        # Get paper data and embedding
        paper_idx = self.embedding_system.metadata[self.embedding_system.metadata['id'] == paper_id].index
        if len(paper_idx) == 0:
            logger.warning(f"Paper ID {paper_id} not found in metadata.")
            return
            
        paper_data = self.embedding_system.metadata.loc[paper_idx[0]]
        embedding_idx = paper_data['embedding_idx']
        
        # Get embedding from FAISS index
        paper_embedding = np.zeros((1, self.embedding_system.embedding_dim), dtype='float32')
        self.embedding_system.index.reconstruct(int(embedding_idx), paper_embedding[0])
        
        # Update user preferences
        self.user_tracker.update_preferences(
            user_id=user_id,
            paper_embedding=paper_embedding[0],
            interaction_type=interaction_type,
            paper_id=paper_id,
            paper_categories=paper_data.get('categories', [])
        )
        
        logger.info(f"Recorded {interaction_type} interaction for user {user_id} on paper {paper_id}")
    
    def get_top_cited_papers(self, query: str, k: int = 5) -> pd.DataFrame:
        """
        Get top cited papers for a query
        
        Args:
            query: Search query
            k: Number of papers to return
            
        Returns:
            DataFrame with top cited papers
        """
        # Fetch papers
        papers = self.arxiv_fetcher.fetch(
            query=query,
            max_results=k * 3,  # Fetch more for filtering
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        if papers.empty:
            return pd.DataFrame()
        
        # Get citation counts
        citation_counts = []
        for _, paper in papers.iterrows():
            citation_data = self.citation_fetcher.get_citation_count(
                paper_id=paper['id'],
                title=paper['title']
            )
            citation_counts.append(citation_data['citation_count'])
        
        # Add citation counts to papers
        papers['citation_count'] = citation_counts
        
        # Sort by citation count and return top k
        return papers.sort_values('citation_count', ascending=False).head(k)
    
    def get_user_insights(self, user_id: str) -> Dict:
        """
        Get insights about user preferences
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with user insights
        """
        if user_id not in self.user_tracker.users:
            logger.warning(f"User {user_id} not found.")
            return {}
        
        user_data = self.user_tracker.get_user_preferences(user_id)
        
        # Get top categories
        categories = sorted(
            user_data['categories_of_interest'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get similar papers to user preferences
        similar_papers = []
        if not np.all(user_data['preference_vector'] == 0):
            recommendations = self.embedding_system.recommend(
                user_preferences=user_data['preference_vector'].reshape(1, -1),
                k=5
            )
            if not recommendations.empty:
                similar_papers = recommendations[['id', 'title', 'similarity_score']].to_dict(orient='records')
        
        return {
            'interaction_count': user_data['interaction_count'],
            'liked_papers_count': len(user_data['liked_papers']),
            'disliked_papers_count': len(user_data['disliked_papers']),
            'viewed_papers_count': len(user_data['viewed_papers']),
            'top_categories': categories[:5],
            'similar_papers': similar_papers
        }


# Example usage
def main():
    # Initialize recommender
    recommender = ResearchRecommender(
        use_faster_model=True,  # Use faster model for better performance
        load_existing=True      # Load existing data if available
    )
    
    # Fetch papers on a topic
    papers = recommender.fetch_papers(
        keywords=["large language models", "transformer"],
        categories=["cs.CL", "cs.LG"],
        max_results=50,
        date_start="2023-01-01"
    )
    
    # Create a user and record interactions
    user_id = "user123"
    
    # Simulate some user interactions
    for _, paper in papers.head(5).iterrows():
        recommender.record_user_interaction(
            user_id=user_id,
            paper_id=paper['id'],
            interaction_type='view'
        )
    
    # Record a like for the first paper
    if not papers.empty:
        recommender.record_user_interaction(
            user_id=user_id,
            paper_id=papers.iloc[0]['id'],
            interaction_type='like'
        )
    
    # Get personalized recommendations
    recommendations = recommender.get_recommendations(
        user_id=user_id,
        k=5,
        include_quality=True
    )
    
    # Print recommendations
    print("\nPersonalized Recommendations:")
    for _, rec in recommendations.iterrows():
        print(f"Title: {rec['title']}")
        print(f"Combined Score: {rec['combined_score']:.4f}")
        print(f"Quality Score: {rec['quality_score']:.4f}")
        print(f"Similarity: {rec['similarity_score']:.4f}")
        print("---")
    
    # Save state for future use
    recommender.save_state()


if __name__ == "__main__":
    main()