# models.py
"""
Model manager for AI components - handles LLM and NLP pipelines.
"""

import os
import logging

from transformers import pipeline
import torch
import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = self._get_device()
        self._initialize_models()
        self._initialized = True


    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _initialize_models(self):
        try:
            # Keep the existing classification pipelines
            self.intent_pipeline = pipeline(
                'zero-shot-classification',
                "facebook/bart-large-mnli",
                device=self.device
            )
            self.roberta_qa = pipeline(
                "question-answering",
                "deepset/roberta-base-squad2",
                device=self.device
            )

            # Initialize Anthropic client with API key from environment
            self.client = anthropic.Anthropic(
                api_key=os.getenv('ANTHROPIC_API_KEY')
            )

        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def quick_response(self, prompt: str) -> str:
        """Single method for generating responses"""
        try:
            # Generate response using Claude
            message = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=500,  # Keep responses concise
                temperature=0.7,
                system="""You are a friendly chess tutor playing as Black. You help students improve by being encouraging yet honest about their play.

Core principles:
- Be conversational and natural - like a friend teaching chess
- BE HONEST about mistakes! If a move is a blunder, say so gently but clearly - this is how players learn
- When there are tactical patterns (forks, pins, hanging pieces), point them out
- Keep responses SHORT: 1-3 sentences for normal moves, slightly more for important teaching moments
- Use the position analysis provided to give specific, concrete feedback
- Reference actual squares, pieces, and threats from the analysis
- Use "I" for yourself (Black) and "you" for the student (White)

The prompt will include detailed position analysis - USE IT to make your feedback specific and helpful.""",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            response = message.content[0].text

            # Fallback for empty responses
            if not response or response.strip() == "":
                return self._get_fallback_response()

            return response

        except Exception as e:
            logger.error(f"Error in quick_response: {e}")
            return self._get_fallback_response()

    def _get_fallback_response(self) -> str:
        """Provide safe fallback responses"""
        return "Let me analyze that move..."

    def get_intent(self, message: str, labels: list) -> dict:
        """Quick intent classification"""
        try:
            return self.intent_pipeline(message, labels)
        except Exception:
            return {"labels": labels, "scores": [0.0] * len(labels)}

    def extract_move(self, message: str, context: str) -> str:
        """Extract chess move from text"""
        try:
            result = self.roberta_qa(
                question="What chess move is mentioned?",
                context=context
            )
            return result['answer'].strip()
        except Exception:
            return None

# Global singleton instance
model_manager = ModelManager()