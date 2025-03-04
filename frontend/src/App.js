import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

// Helper function to generate dummy authors for demo purposes - moved outside components to be reusable
const generateDummyAuthors = () => {
  const firstNames = ['John', 'Jane', 'Alex', 'Sarah', 'Michael', 'Emily', 'David', 'Sophia', 'Wei', 'Li', 'Yann', 'Fei-Fei', 'Geoffrey', 'Yoshua', 'Ian', 'Andrej', 'Clement', 'Samy', 'Ilya', 'Aidan'];
  const lastNames = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis', 'Chen', 'Wang', 'Zhang', 'LeCun', 'Li', 'Hinton', 'Bengio', 'Goodfellow', 'Karpathy', 'Delangue', 'Bengio', 'Sutskever', 'Gomez'];
  const count = Math.floor(Math.random() * 3) + 1;
  
  const authors = [];
  for (let i = 0; i < count; i++) {
    const firstName = firstNames[Math.floor(Math.random() * firstNames.length)];
    const lastName = lastNames[Math.floor(Math.random() * lastNames.length)];
    authors.push(`${firstName} ${lastName}`);
  }
  
  return authors;
};

// Helper function to generate dummy papers - also moved outside
const generateDummyPapers = (count) => {
  const papers = [];
  const titles = [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "GPT-3: Language Models are Few-Shot Learners",
    "Deep Residual Learning for Image Recognition",
    "DeepSeek: Scaling Large Language Models for Wide Access",
    "Gemini: A Family of Highly Capable Multimodal Models",
    "LLaMA: Open and Efficient Foundation Language Models",
    "Stable Diffusion: High-Resolution Image Synthesis with Latent Diffusion Models",
    "DALL·E 2: Hierarchical Text-Conditional Image Generation with CLIP Latents",
    "Whisper: Robust Speech Recognition via Large-Scale Weak Supervision",
    "AlphaGo: Mastering the Game of Go with Deep Neural Networks and Tree Search",
    "CLIP: Learning Transferable Visual Models from Natural Language Supervision",
    "Transformers: A New Architecture for Neural Machine Translation",
    "MuZero: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model",
    "Diffusion Models Beat GANs on Image Synthesis",
    "DALL·E 3: Improving Image Generation with Better Captions",
    "Mixtral of Experts: Sparse Mixture-of-Experts Architecture",
    "Claude 2: A Large Language Model with Constitutional AI",
    "ImageNet Classification with Deep Convolutional Neural Networks",
    "Large Language Models Can Self-Improve"
  ];
  
  const abstracts = [
    "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
    "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
    "We demonstrate that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. GPT-3, an autoregressive language model with 175 billion parameters, achieves strong performance on many NLP tasks.",
    "Deep neural networks are becoming deeper, with state-of-the-art networks evolving from having just a few layers to now having over a hundred. We present residual learning framework to ease the training of deep networks. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs.",
    "We present DeepSeek, a suite of large language models (LLMs) designed to enhance wide access to AI capabilities. Trained on a diverse corpus of code and text, DeepSeek models demonstrate exceptional performance across reasoning, coding, and specialized domain tasks while maintaining computational efficiency.",
    "We introduce Gemini, a family of multimodal models capable of processing and reasoning across text, images, audio, and video. These models demonstrate state-of-the-art performance on multimodal benchmarks and excel at complex tasks requiring reasoning across different modalities.",
    "We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We demonstrate that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. We train a 65B-parameter LLaMA model that outperforms GPT-3.",
    "We present a latent text-to-image diffusion model capable of generating photorealistic images given any text input, applying unCLIP to produce reliable and high-resolution image synthesis. The model's transformer backbone encodes text into a latent space that conditions the diffusion process.",
    "We present a neural network that generates images from text descriptions. DALL·E 2 combines a CLIP image encoder with an autoregressive transformer, enabling unprecedented photorealism and revealing new capabilities in image manipulation and generation from natural language.",
    "We present Whisper, a robust speech recognition system trained on 680,000 hours of multilingual and multitask data collected from the web. We show that the use of such a large and diverse dataset leads to improved robustness to accents, background noise, and technical language.",
    "The game of Go has long been viewed as the most challenging of classic games for artificial intelligence. This complexity is due to the difficulty in evaluating board positions and moves. Here we introduce a new approach that uses value networks to evaluate board positions and policy networks to select moves.",
    "State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations.",
    "We present a new architecture for neural machine translation that replaces recurrent neural networks with self-attention mechanisms. Our model achieves new state-of-the-art BLEU scores on English-to-German and English-to-French translation tasks, while being more parallelizable and requiring significantly less time to train.",
    "Constructing agents with planning capabilities has long been one of the main challenges in the pursuit of artificial intelligence. For the first time, a single algorithm can achieve superhuman performance in a wide range of challenging planning domains without any domain-specific modifications.",
    "We show that diffusion models can achieve image sample quality superior to the current state of the art in generative models. We find that diffusion models are practical to sample from and can be made more efficient with architectural and training methodology improvements.",
    "We present DALL·E 3, a text-to-image system that generates more accurate images by creating detailed captions from user prompts. By first expanding user prompts into detailed captions, DALL·E 3 produces images that better match user intent while improving accuracy of details like spatial relationships and counting.",
    "We introduce Mixtral 8x7B, a sparse mixture of experts model (SMoE) with 46.7B parameters but the same computational cost as a 13B model. We report state-of-the-art results for a decoder-only model in the 10-30B FLOPs range, outperforming Llama 2 70B in reasoning, mathematics, and code generation.",
    "Claude 2 represents a significant advancement in AI assistant capabilities, featuring improved reasoning, coding, and multilingual fluency. Our model was trained with Constitutional AI, a method to align language models with human intent through supervised learning from human feedback.",
    "We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art.",
    "We explore a method for improving language models by having them generate their own training data, which we use to create a new model that outperforms the original one. We demonstrate that language models can improve themselves by bootstrapping off their own generations."
  ];

  for (let i = 0; i < count; i++) {
    const randomIndex = Math.floor(Math.random() * titles.length);
    const authors = generateDummyAuthors();
    
    // Calculate popularity-based citation count (earlier papers in the array are more popular/foundational)
    const citationBase = 15000 - (randomIndex * 500);
    const citations = Math.max(500, Math.floor(citationBase + Math.random() * 3000));
    
    // More realistic publication dates for seminal papers
    const currentYear = new Date().getFullYear();
    const yearSpan = 12; // Papers from last 12 years
    const yearsAgo = Math.floor(Math.random() * yearSpan);
    const pubDate = new Date(currentYear - yearsAgo, Math.floor(Math.random() * 12), Math.floor(Math.random() * 28) + 1);
    
    papers.push({
      id: `dummy-${i}-${Date.now()}`, // Ensure unique IDs
      title: titles[randomIndex % titles.length],
      abstract: abstracts[randomIndex % abstracts.length],
      authors,
      published: pubDate.toISOString(),
      categories: ["cs.LG", "cs.AI", "cs.CL"].slice(0, Math.floor(Math.random() * 3) + 1),
      citations: citations,
      pdf_url: "#",
      similarity: Math.random() * 0.3 + 0.7, // Random similarity score between 0.7 and 1.0
      quality_score: Math.random() * 0.3 + 0.7 // Random quality score between 0.7 and 1.0
    });
  }
  
  // Sort by citations to make popular papers appear first
  return papers.sort((a, b) => b.citations - a.citations);
};

// Tabs component for navigation
const Tabs = ({ activeTab, setActiveTab }) => {
  return (
    <div className="tabs">
      <button 
        className={activeTab === 'search' ? 'active' : ''} 
        onClick={() => setActiveTab('search')}
      >
        Search Papers
      </button>
      <button 
        className={activeTab === 'recommendations' ? 'active' : ''} 
        onClick={() => setActiveTab('recommendations')}
      >
        Get Recommendations
      </button>
      <button 
        className={activeTab === 'seminal' ? 'active' : ''} 
        onClick={() => setActiveTab('seminal')}
      >
        Find Seminal Papers
      </button>
      <button 
        className={activeTab === 'compare' ? 'active' : ''} 
        onClick={() => setActiveTab('compare')}
      >
        Compare Papers
      </button>
    </div>
  );
};

// Paper Detail Component for viewing a single paper
const PaperDetail = ({ paper, onClose, onAddToCompare }) => {
  if (!paper) return null;
  
  return (
    <div className="paper-detail-overlay">
      <div className="paper-detail-container">
        <div className="paper-detail-header">
          <h2>{paper.title}</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>
        
        <div className="paper-detail-content">
          <div className="paper-meta-full">
            <div className="meta-row">
              <span className="meta-label">Published:</span> 
              <span>{new Date(paper.published).toLocaleDateString()}</span>
            </div>
            <div className="meta-row">
              <span className="meta-label">Authors:</span> 
              <span>{paper.authors?.join(', ') || 'Unknown'}</span>
            </div>
            <div className="meta-row">
              <span className="meta-label">Categories:</span> 
              <span>{paper.categories?.join(', ') || 'Uncategorized'}</span>
            </div>
            <div className="meta-row">
              <span className="meta-label">Citations:</span> 
              <span>{paper.citations || 'Unknown'}</span>
            </div>
          </div>
          
          <div className="paper-abstract-full">
            <h3>Abstract</h3>
            <p>{paper.abstract}</p>
          </div>
          
          <div className="paper-actions-full">
            <a href={paper.pdf_url} target="_blank" rel="noopener noreferrer" className="paper-button primary">
              View PDF
            </a>
            <button className="paper-button secondary" onClick={() => onAddToCompare(paper)}>
              Add to Comparison
            </button>
            <button className="paper-button secondary">
              Save to Library
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Scrollable Research View Component
const ScrollableResearchView = ({ papers, onViewPaper, onAddToCompare }) => {
  return (
    <div className="scrollable-research-view">
      {papers.map((paper) => (
        <div key={paper.id} className="paper-card-horizontal">
          <div className="paper-info">
            <h3 className="paper-title" onClick={() => onViewPaper(paper)}>{paper.title}</h3>
            <div className="paper-meta-compact">
              <span className="meta-item">{new Date(paper.published).toLocaleDateString()}</span>
              <span className="meta-item">
                {paper.authors?.slice(0, 3).join(', ') || 'Unknown'}
                {paper.authors?.length > 3 ? ', et al.' : ''}
              </span>
              <span className="meta-item">
                <span className="meta-label">Similarity:</span> 
                <span> {paper.similarity ? (paper.similarity * 100).toFixed(2) + "%" : "N/A"}</span>
              </span>
            </div>
            <p className="paper-abstract-preview">{paper.abstract.substring(0, 150)}...</p>
          </div>
          <div className="paper-actions-compact">
            <button className="paper-action-icon" onClick={() => onViewPaper(paper)} title="View Details">
              <i className="icon-view"></i>
            </button>
            <button className="paper-action-icon" onClick={() => onAddToCompare(paper)} title="Add to Compare">
              <i className="icon-compare"></i>
            </button>
          </div>
        </div>
      ))}
    </div>
  );
};


// Enhanced Search component
const Search = ({ onAddToCompare }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [categories, setCategories] = useState(["cs.LG", "cs.CL"]);
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [viewMode, setViewMode] = useState('scroll'); // Default to scroll view

  // Generate dummy data for 20 papers on initial load
  React.useEffect(() => {
    const fetchInitialPapers = async () => {
      try {
        // Try to fetch from API first
        const response = await axios.post('http://localhost:8000/search', { 
          query: '"machine learning"', // Default query for initial load
          categories
        }).catch(() => ({ data: [] })); // If API fails, use empty array
        
        let papersData = response.data;
        
        // If API returned less than 20 papers or failed, generate dummy data to make up the difference
        if (papersData.length < 20) {
          const dummyPapers = generateDummyPapers(20 - papersData.length);
          papersData = [...papersData, ...dummyPapers];
        }
        
        // Sort by published date
        const sorted = papersData.sort((a, b) => 
          new Date(b.published) - new Date(a.published)
        );
        
        setResults(sorted);
        setLoading(false);
      } catch (error) {
        console.error("Error loading initial papers:", error);
        // Even if there's an error, show dummy data
        const dummyPapers = generateDummyPapers(20);
        setResults(dummyPapers);
        setLoading(false);
      }
    };
    
    fetchInitialPapers();
  }, [categories]); // generateDummyPapers is now outside the component, so no need to include it

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    setLoading(true);
    setError('');
    try {
      const response = await axios.post('http://localhost:8000/search', { 
        query: `"${query}"`,
        categories
      }).catch(() => {
        // If API fails, generate dummy search results
        return { data: generateDummySearchResults(query, 12) };
      });
      
      // Add dummy data for authors and citations since they're not in the original API response
      const enhancedResults = response.data.map(paper => ({
        ...paper,
        authors: paper.authors || generateDummyAuthors(),
        citations: paper.citations || Math.floor(Math.random() * 1000)
      }));
      
      const sorted = enhancedResults.sort((a, b) => 
        new Date(b.published) - new Date(a.published)
      );
      setResults(sorted);
    } catch (error) {
      setError('Failed to fetch papers. Please try different keywords.');
      // If search fails, still show some results
      setResults(generateDummySearchResults(query, 8));
    } finally {
      setLoading(false);
    }
  };

  // Generate dummy search results based on query
  const generateDummySearchResults = (query, count) => {
    const papers = [];
    const searchTerm = query.toLowerCase();
    
    for (let i = 0; i < count; i++) {
      const authors = generateDummyAuthors();
      
      papers.push({
        id: `search-${i}-${Date.now()}`,
        title: `Recent Advances in ${searchTerm.charAt(0).toUpperCase() + searchTerm.slice(1)} Research`,
        abstract: `This paper presents a comprehensive survey of recent developments in the field of ${searchTerm}. We analyze the current state-of-the-art methods, identify key challenges, and propose future research directions.`,
        authors,
        published: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString(), // Random date in last year
        categories: ["cs.LG", "cs.AI"].slice(0, Math.floor(Math.random() * 2) + 1),
        citations: Math.floor(Math.random() * 1000),
        pdf_url: "#"
      });
    }
    
    return papers;
  };

  const categoryOptions = [
    { value: "cs.LG", label: "Machine Learning" },
    { value: "cs.CL", label: "Computational Linguistics" },
    { value: "cs.AI", label: "Artificial Intelligence" },
    { value: "cs.CV", label: "Computer Vision" },
    { value: "cs.RO", label: "Robotics" }
  ];

  const handleCategoryChange = (category) => {
    if (categories.includes(category)) {
      setCategories(categories.filter(c => c !== category));
    } else {
      setCategories([...categories, category]);
    }
  };

  const handleViewPaper = (paper) => {
    setSelectedPaper(paper);
  };

  const handleClosePaperDetail = () => {
    setSelectedPaper(null);
  };

  return (
    <div className="content-section">
      <div className="section-header">
        <h2>Search Research Papers</h2>
        <p>Find the latest papers in your field of interest</p>
      </div>

      <form onSubmit={handleSearch} className="search-form">
        <div className="search-input-container">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter paper title or keywords"
            className="search-input"
          />
          <button type="submit" disabled={loading} className="search-button">
            {loading ? 
              <span className="loading-spinner"></span> : 
              <span>Search</span>
            }
          </button>
        </div>

        <div className="search-options">
          <div className="category-filters">
            <p>Filter by categories:</p>
            <div className="category-options">
              {categoryOptions.map(option => (
                <label key={option.value} className="category-checkbox">
                  <input
                    type="checkbox"
                    checked={categories.includes(option.value)}
                    onChange={() => handleCategoryChange(option.value)}
                  />
                  {option.label}
                </label>
              ))}
            </div>
          </div>
          
          <div className="view-toggle">
            <button 
              type="button"
              className={`view-toggle-button ${viewMode === 'grid' ? 'active' : ''}`}
              onClick={() => setViewMode('grid')}
            >
              Grid View
            </button>
            <button 
              type="button"
              className={`view-toggle-button ${viewMode === 'scroll' ? 'active' : ''}`}
              onClick={() => setViewMode('scroll')}
            >
              Scrollable View
            </button>
          </div>
        </div>
      </form>
      
      {error && <div className="error-message">{error}</div>}
      
      <div className="results-container">
        {results.length > 0 && (
          <div className="results-header">
            <h3>Found {results.length} papers</h3>
          </div>
        )}

        {results.length > 0 && viewMode === 'grid' ? (
          <div className="results-grid">
            {results.map((paper) => (
              <div key={paper.id} className="paper-card">
                <h3 className="paper-title" onClick={() => handleViewPaper(paper)}>{paper.title}</h3>
                <div className="paper-meta">
                  <div className="meta-item">
                    <span className="meta-label">Published:</span> 
                    <span>{new Date(paper.published).toLocaleDateString()}</span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Authors:</span> 
                    <span>{paper.authors.slice(0, 2).join(', ')}{paper.authors.length > 2 ? ', et al.' : ''}</span>
                  </div>
                </div>
                <p className="paper-abstract">{paper.abstract.substring(0, 180)}...</p>
                <div className="paper-actions">
                  <button className="paper-button secondary" onClick={() => handleViewPaper(paper)}>
                    View Details
                  </button>
                  <button className="paper-button secondary" onClick={() => onAddToCompare(paper)}>
                    Compare
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : results.length > 0 ? (
          <ScrollableResearchView 
            papers={results} 
            onViewPaper={handleViewPaper}
            onAddToCompare={onAddToCompare}
          />
        ) : !loading && (
          <div className="empty-state">
            <p>Enter keywords above to search for research papers</p>
          </div>
        )}

        {loading && (
          <div className="loading-container">
            <div className="loading-spinner large"></div>
            <p>Searching for papers...</p>
          </div>
        )}
      </div>

      {selectedPaper && (
        <PaperDetail 
          paper={selectedPaper} 
          onClose={handleClosePaperDetail}
          onAddToCompare={onAddToCompare}
        />
      )}
    </div>
  );
};

// Enhanced Recommendations component
const Recommendations = ({ onAddToCompare }) => {
  const [text, setText] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedPaper, setSelectedPaper] = useState(null);

  const handleRecommend = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    
    setLoading(true);
    setError('');
    try {
      const response = await axios.post('http://localhost:8000/recommend', { text });
      
      // Add dummy data for demo purposes
      const enhancedResults = response.data.map(paper => ({
        ...paper,
        authors: paper.authors || generateDummyAuthors(),
        published: paper.published || new Date().toISOString(),
        abstract: paper.abstract || 'Abstract not available for this paper.',
        categories: paper.categories || ['cs.AI'],
        citations: paper.citations || Math.floor(Math.random() * 1000),
        pdf_url: paper.pdf_url || '#'
      }));
      
      setRecommendations(enhancedResults);
    } catch (error) {
      setError('Error getting recommendations. Please try again.');
      // If the API fails, generate dummy recommendations based on the input text
      const dummyRecommendations = generateDummyRecommendations(text, 10);
      setRecommendations(dummyRecommendations);
    } finally {
      setLoading(false);
    }
  };

  // Generate dummy recommendations based on input text
  const generateDummyRecommendations = (text, count) => {
    const papers = [];
    // Extract keywords from input text
    const words = text.toLowerCase().split(/\W+/);
    const keywords = words.filter(word => 
      word.length > 4 && !['about', 'these', 'their', 'there', 'which', 'would', 'should'].includes(word)
    ).slice(0, 5);
    
    if (keywords.length === 0) keywords.push('research');
    
    for (let i = 0; i < count; i++) {
      const keyword = keywords[i % keywords.length];
      const similarity = 0.95 - (i * 0.05); // Decreasing similarity
      
      papers.push({
        id: `rec-${i}-${Date.now()}`,
        title: `Advanced ${keyword.charAt(0).toUpperCase() + keyword.slice(1)} Techniques for Modern Research`,
        authors: generateDummyAuthors(),
        abstract: `This research explores innovative approaches to ${keyword} in the context of modern computational methods. We present novel algorithms that improve upon existing techniques and demonstrate their effectiveness through extensive experiments.`,
        published: new Date(Date.now() - Math.random() * 2 * 365 * 24 * 60 * 60 * 1000).toISOString(),
        categories: ['cs.AI', 'cs.LG'],
        citations: Math.floor(Math.random() * 800) + 10,
        pdf_url: '#',
        similarity: similarity > 0.5 ? similarity : 0.5 + Math.random() * 0.3
      });
    }
    
    return papers;
  };

  const handleViewPaper = (paper) => {
    setSelectedPaper(paper);
  };

  const handleClosePaperDetail = () => {
    setSelectedPaper(null);
  };

  return (
    <div className="content-section">
      <div className="section-header">
        <h2>Get Paper Recommendations</h2>
        <p>Paste an abstract or research idea to find related papers</p>
      </div>

      <form onSubmit={handleRecommend} className="recommendation-form">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter an abstract, research question, or text to get related papers"
          className="recommendation-textarea"
        />
        <button type="submit" disabled={loading} className="recommendation-button">
          {loading ? 'Finding papers...' : 'Get Recommendations'}
        </button>
      </form>

      {error && <div className="error-message">{error}</div>}

      {recommendations.length > 0 && (
        <div className="recommendations-results">
          <h3>Recommended Papers</h3>
          <div className="recommendations-list">
            {recommendations.map((paper) => (
              <div key={paper.id} className="recommendation-card">
                <div className="recommendation-content">
                  <h4 onClick={() => handleViewPaper(paper)}>{paper.title}</h4>
                  <p className="recommendation-authors">
                    {paper.authors.join(', ')}
                  </p>
                  <div className="similarity-score">
                    <div className="similarity-bar" style={{ width: `${Math.min(paper.similarity * 100, 100)}%` }}></div>
                    <span className="similarity-text">{(paper.similarity * 100).toFixed(0)}% Match</span>
                  </div>
                </div>
                <div className="recommendation-actions">
                  <button className="mini-button" onClick={() => handleViewPaper(paper)}>View</button>
                  <button className="mini-button" onClick={() => onAddToCompare(paper)}>Compare</button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {loading && (
        <div className="loading-container">
          <div className="loading-spinner large"></div>
          <p>Finding relevant papers...</p>
        </div>
      )}

      {selectedPaper && (
        <PaperDetail 
          paper={selectedPaper} 
          onClose={handleClosePaperDetail}
          onAddToCompare={onAddToCompare}
        />
      )}
    </div>
  );
};

// Enhanced SeminalPapers component
// Enhanced SeminalPapers component
const SeminalPapers = ({ onAddToCompare }) => {
    const [topic, setTopic] = useState('');
    const [seminalPapers, setSeminalPapers] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [selectedPaper, setSelectedPaper] = useState(null);
  
    const handleSeminalSearch = async (e) => {
      e.preventDefault();
      if (!topic.trim()) return;
      
      setLoading(true);
      setError('');
      try {
        const response = await axios.post('http://localhost:8000/seminal-papers', { topic });
        
        const enhancedResults = response.data.map(paper => ({
          ...paper,
          authors: paper.authors || generateDummyAuthors(),
          published: paper.published || new Date(Date.now() - Math.random() * 5 * 365 * 24 * 60 * 60 * 1000).toISOString(),
          abstract: paper.abstract || `This seminal paper established foundational concepts in ${topic}.`,
          categories: paper.categories || ['cs.AI'],
          citations: paper.citations || Math.floor(Math.random() * 10000),
          pdf_url: paper.pdf_url || '#',
          seminal_score: 0.9 + Math.random() * 0.1 // Random high seminal score
        }));
        
        setSeminalPapers(enhancedResults);
      } catch (error) {
        setError('Error finding seminal papers. Using demo data.');
        setSeminalPapers(generateDummySeminalPapers(topic, 8));
      } finally {
        setLoading(false);
      }
    };
  
    const generateDummySeminalPapers = (topic, count) => {
      return Array.from({ length: count }, (_, i) => ({
        id: `seminal-${i}-${Date.now()}`,
        title: `Seminal Work on ${topic.charAt(0).toUpperCase() + topic.slice(1)} ${i + 1}`,
        authors: generateDummyAuthors(),
        abstract: `This groundbreaking paper introduced fundamental concepts in ${topic} that shaped the field.`,
        published: new Date(2010 - i, 0).toISOString(), // Earlier papers are more seminal
        categories: ['cs.AI', 'cs.LG'],
        citations: 15000 - (i * 1000) + Math.floor(Math.random() * 5000), // Earlier papers have more citations
        pdf_url: '#',
        seminal_score: 0.95 - (i * 0.02)
      }));
    };
  
    const handleViewPaper = (paper) => {
      setSelectedPaper(paper);
    };
  
    const handleClosePaperDetail = () => {
      setSelectedPaper(null);
    };
  
    return (
      <div className="content-section">
        <div className="section-header">
          <h2>Discover Seminal Papers</h2>
          <p>Identify foundational works in any research area</p>
        </div>
  
        <form onSubmit={handleSeminalSearch} className="seminal-form">
          <input
            type="text"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="Enter research topic or domain"
            className="seminal-input"
          />
          <button type="submit" disabled={loading} className="seminal-button">
            {loading ? 'Searching...' : 'Find Foundational Works'}
          </button>
        </form>
  
        {error && <div className="error-message">{error}</div>}
  
        {loading ? (
          <div className="loading-container">
            <div className="loading-spinner large"></div>
            <p>Discovering seminal papers...</p>
          </div>
        ) : (
          <div className="seminal-results">
            {seminalPapers.map(paper => (
              <div key={paper.id} className="seminal-card">
                <div className="seminal-header">
                  <h3 onClick={() => handleViewPaper(paper)}>{paper.title}</h3>
                  <span className="seminal-year">
                    {new Date(paper.published).getFullYear()}
                  </span>
                </div>
                <div className="seminal-metrics">
                  <div className="metric">
                    <span className="metric-label">Citations:</span>
                    <span className="metric-value">{paper.citations.toLocaleString()}+</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Influence Score:</span>
                    <span className="metric-value">
                      {(paper.seminal_score * 100).toFixed(0)}/100
                    </span>
                  </div>
                </div>
                <p className="seminal-authors">
                  {paper.authors.join(', ')}
                </p>
                <div className="seminal-actions">
                  <button 
                    className="paper-button primary"
                    onClick={() => handleViewPaper(paper)}
                  >
                    View Details
                  </button>
                  <button 
                    className="paper-button secondary"
                    onClick={() => onAddToCompare(paper)}
                  >
                    Add to Comparison
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
  
        {selectedPaper && (
          <PaperDetail 
            paper={selectedPaper} 
            onClose={handleClosePaperDetail}
            onAddToCompare={onAddToCompare}
          />
        )}
      </div>
    );
  };
  
  // Complete Compare component 
  const Compare = ({ papers, onRemovePaper }) => {
    return (
      <div className="content-section">
        <div className="section-header">
          <h2>Compare Research Papers</h2>
          <p>Side-by-side comparison of selected papers</p>
        </div>
  
        {papers.length === 0 ? (
          <div className="empty-state">
            <p>Add papers from other sections to compare</p>
          </div>
        ) : (
          <div className="comparison-grid">
            <div className="comparison-header">
              <div className="header-spacer"></div>
              {papers.map((paper) => (
                <div key={paper.id} className="paper-header">
                  <h3>{paper.title}</h3>
                  <button className="remove-button" onClick={() => onRemovePaper(paper.id)}>
                    Remove
                  </button>
                </div>
              ))}
            </div>
  
            <div className="comparison-row">
              <div className="comparison-label">Authors</div>
              {papers.map(paper => (
                <div key={paper.id} className="comparison-item">
                  {paper.authors.join(', ')}
                </div>
              ))}
            </div>
  
            <div className="comparison-row">
              <div className="comparison-label">Published</div>
              {papers.map(paper => (
                <div key={paper.id} className="comparison-item">
                  {new Date(paper.published).toLocaleDateString()}
                </div>
              ))}
            </div>
  
            <div className="comparison-row">
              <div className="comparison-label">Categories</div>
              {papers.map(paper => (
                <div key={paper.id} className="comparison-item">
                  {paper.categories?.join(', ') || 'Uncategorized'}
                </div>
              ))}
            </div>
  
            <div className="comparison-row abstract-row">
              <div className="comparison-label">Abstract</div>
              {papers.map(paper => (
                <div key={paper.id} className="comparison-item">
                  {paper.abstract.substring(0, 200)}...
                </div>
              ))}
            </div>
  
            <div className="comparison-row">
              <div className="comparison-label">Citations</div>
              {papers.map(paper => (
                <div key={paper.id} className="comparison-item">
                  {paper.citations.toLocaleString()}
                </div>
              ))}
            </div>
  
            <div className="comparison-row">
              <div className="comparison-label">Actions</div>
              {papers.map(paper => (
                <div key={paper.id} className="comparison-item">
                  <a href={paper.pdf_url} target="_blank" rel="noopener noreferrer" className="mini-button">
                    View PDF
                  </a>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };
  
  // Updated App component
  const App = () => {
    const [activeTab, setActiveTab] = useState('search');
    const [comparedPapers, setComparedPapers] = useState([]);
  
    const handleAddToCompare = (paper) => {
      if (!comparedPapers.find(p => p.id === paper.id)) {
        setComparedPapers([...comparedPapers, paper]);
      }
    };
  
    const handleRemovePaper = (paperId) => {
      setComparedPapers(comparedPapers.filter(paper => paper.id !== paperId));
    };
  
    return (
      <div className="App">
        <header className="app-header">
          <h1>Research Papers Explorer</h1>
          {comparedPapers.length > 0 && (
            <div className="comparison-badge" onClick={() => setActiveTab('compare')}>
              {comparedPapers.length} paper{comparedPapers.length > 1 ? 's' : ''} in comparison
            </div>
          )}
        </header>
        
        <Tabs activeTab={activeTab} setActiveTab={setActiveTab} />
        
        {activeTab === 'search' && (
          <Search onAddToCompare={handleAddToCompare} />
        )}
        
        {activeTab === 'recommendations' && (
          <Recommendations onAddToCompare={handleAddToCompare} />
        )}
        
        {activeTab === 'seminal' && (
          <SeminalPapers onAddToCompare={handleAddToCompare} />
        )}
        
        {activeTab === 'compare' && (
          <Compare papers={comparedPapers} onRemovePaper={handleRemovePaper} />
        )}
      </div>
    );
  };
  
  export default App;