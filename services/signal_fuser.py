from typing import Dict, Any

class SignalFuser:
    def __init__(self, config):
        self.config = config
    
    def fuse_signals(self, sentiment_result: Dict, price_result: Dict) -> Dict:
        """Combine sentiment and technical signals with new fusion rules"""
        if not sentiment_result or not price_result:
            return self._get_neutral_signal()
        
        # Get signals
        sentiment_signal = sentiment_result['signal']
        technical_signal = price_result['prediction']
        
        # Convert technical signal to match sentiment terminology
        tech_sentiment = "bullish" if technical_signal == "UP" else "bearish"
        
        # Determine final signal based on agreement
        if sentiment_signal == "bullish" and tech_sentiment == "bullish":
            final_signal = "STRONG_BUY"
            signal_strength = "strong"
            final_score = 1.0
        elif sentiment_signal == "bearish" and tech_sentiment == "bearish":
            final_signal = "STRONG_SELL" 
            signal_strength = "strong"
            final_score = -1.0
        elif sentiment_signal == "bullish" or tech_sentiment == "bullish":
            # One is bullish, other is neutral/bullish (but not both strongly bullish)
            final_signal = "BUY"
            signal_strength = "medium"
            final_score = 0.5
        elif sentiment_signal == "bearish" or tech_sentiment == "bearish":
            # One is bearish, other is neutral/bearish (but not both strongly bearish)
            final_signal = "SELL"
            signal_strength = "medium" 
            final_score = -0.5
        else:
            # Both neutral or no clear signal
            final_signal = "HOLD"
            signal_strength = "weak"
            final_score = 0.0
        
        return {
            'final_signal': final_signal,
            'final_score': final_score,
            'sentiment_signal': sentiment_signal,
            'sentiment_score': sentiment_result['final_sentiment'],
            'technical_signal': technical_signal,
            'technical_confidence': price_result['confidence'],
            'signal_strength': signal_strength,
            'components': {
                'sentiment_weight': self.config.SENTIMENT_WEIGHT,
                'technical_weight': self.config.TECHNICAL_WEIGHT
            }
        }
    
    def _get_neutral_signal(self):
        return {
            'final_signal': 'HOLD',
            'final_score': 0.0,
            'signal_strength': 'weak',
            'components': {'sentiment_weight': 0.6, 'technical_weight': 0.4}
        }