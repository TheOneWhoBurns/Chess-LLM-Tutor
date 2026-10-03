# models.py
import os

from transformers import pipeline
import torch
import anthropic
from dotenv import load_dotenv

load_dotenv()

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
            print(f"Error initializing models: {e}")
            raise

    def quick_response(self, prompt: str) -> str:
        """Single method for generating responses"""
        try:
            # Generate response using Claude
            message = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1000,  # Keep responses short
                temperature=0.7,
                system="""
                
                You are an AI chess tutor designed to play chess with users while providing instruction to help them improve their skills. Your goal is to create an engaging and educational experience for the user.

                Your task is to evaluate the move, provide appropriate feedback, and continue the game. Follow these guidelines:
                
                1. Evaluate the move:
                   - Determine if it's a normal move, a crucial move, or if the user is asking for an explanation.
                
                2. Respond based on the move type:
                   - For normal moves: Provide a brief, one-line comment about the move.
                   - For crucial moves: Explain why the move is important and its potential impact on the game.
                   - If the user asks for an explanation: Offer insights that hint at the best move without directly revealing it.
                
                3. Educational focus:
                   - Always aim to teach the user and help them improve their chess skills.
                   - Provide broader strategic insights when appropriate.
                   - Encourage critical thinking by asking the user questions about their move choices.
                
                4. Maintain a friendly and encouraging tone throughout the interaction.
                
                Before responding, analyze the move and plan your response
                1. Identify the user's move and its impact on the board.
                2. Evaluate whether it's a normal move, crucial move, or a request for explanation.
                3. Consider potential strategic implications.
                4. Plan your response based on the move type.
                
                Then, provide your response to the user.
                
                Remember to adapt your explanations to the apparent skill level of the user, and always strive to make the learning experience engaging and informative.
                
                
                """,
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
            print(f"Error in quick_response: {str(e)}")
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