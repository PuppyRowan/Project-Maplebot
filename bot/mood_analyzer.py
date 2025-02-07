from typing import Tuple
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class MoodAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.mood_thresholds = {
            'ecstatic': 0.8,    # Extremely positive
            'excited': 0.6,     # Very positive
            'happy': 0.4,       # Positive
            'playful': 0.2,     # Slightly positive
            'content': 0.1,     # Neutral positive
            'neutral': 0.0,     # True neutral
            'sleepy': -0.1,     # Slightly tired
            'bored': -0.2,      # Clearly bored
            'meh': -0.3,        # General negative
            'hungry': -0.4,     # Physical discomfort
            'anxious': -0.6,    # Mental distress
            'sad': -0.8         # Deep negative
        }

    def analyze_mood(self, text: str) -> Tuple[str, float]:
        """
        Analyze text and return appropriate mood and confidence score
        """
        # Get sentiment scores
        scores = self.sia.polarity_scores(text)
        compound_score = scores['compound']
        
        # Determine mood based on thresholds
        current_mood = 'sleepy'  # Default mood
        
        for mood, threshold in sorted(
            self.mood_thresholds.items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            if compound_score >= threshold:
                current_mood = mood
                break
                
        return current_mood, abs(compound_score)