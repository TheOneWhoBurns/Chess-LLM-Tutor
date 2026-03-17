# views.py
"""
Django views for the chess tutor application.
"""

import json
import logging
import os

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .intent import intent_classifier
from .ChessLogic import ChessLogicUnit

logger = logging.getLogger(__name__)

# Initialize chess logic with project directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
chess_logic = ChessLogicUnit(project_dir)


def chat_view(request):
    """Render the main chess tutor interface"""
    return render(request, 'chat.html')


@csrf_exempt  # Note: CSRF exempt for local development. Frontend sends token but this simplifies testing.
def send_message(request):
    """
    Handle incoming messages from the chess tutor interface.

    Accepts POST requests with JSON body: {"message": "user message"}
    Returns JSON with response, status, moves, and FEN position.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=400)

    try:
        data = json.loads(request.body)
        message = data.get('message', '').strip()

        if not message:
            return JsonResponse({
                'response': "Please enter a move or message.",
                'status': 'error',
                'moves': chess_logic.get_move_history(),
                'fen': chess_logic.get_current_position()
            })

        # Update intent classifier's board state to match current game
        intent_classifier.update_board(chess_logic.board)

        # Classify the user's intent
        intent_result = intent_classifier.classify(message)
        intent_result["message"] = message

        # Process through chess logic
        response = chess_logic.handle_message(intent_result)

        return JsonResponse({
            'response': response["message"],
            'status': response["status"],
            'moves': response["moves"],
            'fen': chess_logic.get_current_position(),
            'player_stats': response.get("player_stats", chess_logic.get_player_stats())
        })

    except json.JSONDecodeError:
        logger.error("Invalid JSON in request body")
        return JsonResponse({
            'response': "Invalid request format.",
            'status': 'error',
            'moves': chess_logic.get_move_history(),
            'fen': chess_logic.get_current_position()
        })
    except Exception as e:
        logger.error(f"Error in send_message: {e}", exc_info=True)
        return JsonResponse({
            'response': "Something went wrong. Please try again.",
            'status': 'error',
            'moves': chess_logic.get_move_history(),
            'fen': chess_logic.get_current_position()
        })
