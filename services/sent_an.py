import pandas as pd
from datetime import datetime
import numpy as np
from typing import Dict, List
import re
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class OptimizedSentimentAnalyzer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        self._initialize_essential_patterns()
    
    def _initialize_essential_patterns(self):
        """Only essential patterns for faster loading"""
        self.POSITIVE_TERMS = {
            "order": 0.35, "contract": 0.35, "profit": 0.40, "dividend": 0.35,
            "bonus": 0.45, "buyback": 0.40, "acquisition": 0.35, "approval": 0.25,
            "secured": 0.30, "won": 0.35, "awarded": 0.35, "crore": 0.25
        }
        
        self.NEGATIVE_TERMS = {
            "net loss": -0.45, "investigation": -0.50, "penalty": -0.40,
            "shutdown": -0.45, "default": -0.55
        }
        
        self.NEUTRAL_TERMS = {
            "loss of shares", "transmission of shares", "board meeting", "agm"
        }
    
    def _ensure_model_loaded(self):
        """Lazy load model only when needed"""
        if not self._model_loaded:
            try:
                self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
                self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
                self.model.eval()
                self._model_loaded = True
            except Exception as e:
                print(f"❌ FinBERT load failed: {e}")
    
    def analyze_sentiment(self, text: str) -> float:
        self._ensure_model_loaded()
        if self.model is None or not text:
            return 0.0
            
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                                  truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            return probabilities[0][1].item() - probabilities[0][2].item()
        except:
            return 0.0
    
    def analyze_news_sentiment(self, announcements: List[Dict]) -> Dict:
        if not announcements:
            return self._get_neutral_response()
        
        scores = []
        for announcement in announcements:
            text = f"{announcement.get('headline', '')} {announcement.get('full_content', '')}".lower()
            base_sentiment = self.analyze_sentiment(text)
            
            # Fast keyword boosting
            boost = 0.0
            for term, value in self.POSITIVE_TERMS.items():
                if term in text and not any(neutral in text for neutral in self.NEUTRAL_TERMS):
                    boost += value
            for term, value in self.NEGATIVE_TERMS.items():
                if term in text:
                    boost += value
            
            time_weight = self._calculate_time_decay(announcement['date'])
            final_sentiment = max(-1.0, min(1.0, base_sentiment + boost))
            scores.append(final_sentiment * time_weight)
        
        final_sentiment = np.mean(scores) if scores else 0.0
        signal = "bullish" if final_sentiment >= self.config.POSITIVE_THRESHOLD else \
                 "bearish" if final_sentiment <= self.config.NEGATIVE_THRESHOLD else "neutral"
        
        return {
            'final_sentiment': round(final_sentiment, 4),
            'signal': signal,
            'total_news': len(announcements),
            'confidence': 'high' if abs(final_sentiment) > 0.3 else 'medium' if abs(final_sentiment) > 0.15 else 'low'
        }
    
    def _calculate_time_decay(self, news_date: str) -> float:
        try:
            current_date = datetime.now()
            for fmt in ['%d-%b-%Y', '%Y-%m-%d']:
                try:
                    news_dt = datetime.strptime(news_date, fmt)
                    days_diff = (current_date - news_dt).days
                    return max(np.exp(-self.config.DECAY_RATE * days_diff), 0.01)
                except:
                    continue
            return 0.5
        except:
            return 0.5
    
    def _get_neutral_response(self):
        return {
            'final_sentiment': 0.0, 'signal': 'neutral', 'total_news': 0, 'confidence': 'low'
        }