import React, { useState } from 'react';
import axios from 'axios';

function SeminalPapers() {
  const [topic, setTopic] = useState('');
  const [seminalPapers, setSeminalPapers] = useState([]);

  const handleSeminalSearch = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:8000/seminal-papers', { topic });
      setSeminalPapers(response.data);
    } catch (error) {
      console.error('Error finding seminal papers:', error);
    }
  };

  return (
    <div>
      <h2>Find Seminal Papers</h2>
      <form onSubmit={handleSeminalSearch}>
        <input
          type="text"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="Enter research topic"
        />
        <button type="submit">Find Seminal Papers</button>
      </form>
      <ul>
        {seminalPapers.map((paper) => (
          <li key={paper.id}>{paper.title} (Quality Score: {paper.quality_score.toFixed(2)})</li>
        ))}
      </ul>
    </div>
  );
}

export default SeminalPapers;
