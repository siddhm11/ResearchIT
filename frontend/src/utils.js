// In utils.js or shared constants
const TRANSFORMER_AUTHORS = [
    'Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar',
    'Jakob Uszkoreit', 'Llion Jones', 'Aidan Gomez',
    'Lukasz Kaiser', 'Illia Polosukhin'
  ];
  
  const firstNames = [...TRANSFORMER_AUTHORS.map(a => a.split(' ')[0])];
  const lastNames = [...TRANSFORMER_AUTHORS.map(a => a.split(' ')[1])];
  