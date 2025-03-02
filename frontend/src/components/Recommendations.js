import React, { useState } from 'react';
import axios from 'axios';

function Recommendations() {
  const [text, setText] = useState('');
  const [recommendations, setRecommendations] = useState([]);

  const handleRecommend = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:8000/recommend', { text });
      setRecommendations(response.data);
    } catch (error) {
      console.error('Error getting recommendations:', error);
    }
  };

  return (
    <div>
      <h2>Get Recommendations</h2>
      <form onSubmit={handleRecommend}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text for recommendations"
        />
        <button type="submit">Get Recommendations</button>
      </form>
      <ul>
        {recommendations.map((paper) => (
          <li key={paper.id}>{paper.title} (Similarity: {paper.similarity.toFixed(2)})</li>
        ))}
      </ul>
    </div>
  );
}

export default Recommendations;
