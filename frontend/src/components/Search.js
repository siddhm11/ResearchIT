// Add missing imports at the top
import React, { useState } from 'react';
import axios from 'axios';

function Search() {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
  
    const handleSearch = async (e) => {
      e.preventDefault();
      setLoading(true);
      setError('');
      try {
        const response = await axios.post('http://localhost:8000/search', { 
          query: `"${query}"`,  // Add quotes for exact match
          categories: ["cs.LG", "cs.CL"]  // Default to ML/NLP categories
        });
        
        // Sort by publication date
        const sorted = response.data.sort((a, b) => 
          new Date(b.published) - new Date(a.published)
        );
        setResults(sorted);
      } catch (error) {
        setError('Failed to fetch papers. Please try different keywords.');
      } finally {
        setLoading(false);
      }
    };
  
    return (
      <div className="search-section">
        <h2>Search Research Papers</h2>
        <form onSubmit={handleSearch}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter paper title (e.g. 'attention is all you need')"
          />
          <button type="submit" disabled={loading}>
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>
        
        {error && <div className="error">{error}</div>}
        
        <div className="results">
          {results.map((paper) => (
            <div key={paper.id} className="paper-card">
              <h3>{paper.title}</h3>
              <div className="paper-meta">
                <span>Published: {new Date(paper.published).toLocaleDateString()}</span>
                <span>Categories: {paper.categories?.join(', ')}</span>
              </div>
              <p className="abstract">{paper.abstract.substring(0, 200)}...</p>
              <a href={paper.pdf_url} target="_blank" rel="noopener noreferrer">
                View PDF
              </a>
            </div>
          ))}
        </div>
      </div>
    );
  }
  
export default Search;