"""Mistake lessons independent of live play. Repeat checks measure retention only."""
from datetime import timedelta
import hashlib

import chess
from django.utils import timezone

from .models import PracticeAttempt, PracticeEvaluation, PracticeLesson
from .opponent import MATE, REPLY_NODES, engine_room

CONFIRM_NODES = 600000
MIN_LOSS = 100
ACCEPTABLE_LOSS = 35
INTERVALS = (1, 3, 7, 14, 30)


def source_key(board, played):
    return hashlib.sha256((board.fen() + " " + played.uci()).encode()).hexdigest()


def acceptable(best, row):
    if best['mate'] is not None and best['mate'] > 0:
        return row['mate'] is not None and row['mate'] > 0
    if row['mate'] is not None and row['mate'] < 0:
        return False
    return best['score'] - row['score'] <= ACCEPTABLE_LOSS


def create_lesson(profile, board, played, source, *, known_loss=None, room=None):
    """Screen, then independently search all legal choices with a larger budget."""
    if not board.is_valid() or board.is_game_over() or played not in board.legal_moves:
        return None
    key = source_key(board, played)
    if PracticeEvaluation.objects.filter(profile=profile, source_key=key).exists() or \
            PracticeLesson.objects.filter(profile=profile, source_key=key).exists():
        return None
    room = room or engine_room()
    with room.turn():
        if known_loss is None:
            screen = room.stockfish.rank(board, count=board.legal_moves.count(), nodes=REPLY_NODES)
            known_loss = screen[0]['score'] - next(r['score'] for r in screen if r['move'] == played)
        if known_loss < MIN_LOSS:
            PracticeEvaluation.objects.get_or_create(profile=profile, source_key=key)
            return None
        rows = room.stockfish.rank(board, count=board.legal_moves.count(), nodes=CONFIRM_NODES)
    best = rows[0]
    original = next(row for row in rows if row['move'] == played)
    # Avoid teaching arbitrary survival choices in already forced losses.
    if best['score'] <= -MATE + 1000 or acceptable(best, original):
        PracticeEvaluation.objects.get_or_create(profile=profile, source_key=key)
        return None
    loss = max(0, best['score'] - original['score'])
    if loss < MIN_LOSS:
        PracticeEvaluation.objects.get_or_create(profile=profile, source_key=key)
        return None
    evidence = {
        'fen': board.fen(), 'root_fen': board.root().fen(),
        'history': [m.uci() for m in board.move_stack],
        'source': source, 'original': played.uci(), 'loss_cp': min(loss, 2000),
        'engine': room.stockfish.name, 'nodes_budget': CONFIRM_NODES,
        'depth': min(row['depth'] for row in rows),
        'accepted': [row['move'].uci() for row in rows if acceptable(best, row)],
        'choices': {row['move'].uci(): {
            'san': board.san(row['move']), 'score_cp': row['score'],
            'mate': row['mate'], 'pv': row.get('pv', [row['move'].uci()]),
        } for row in rows},
        'best': best['move'].uci(),
    }
    lesson, created = PracticeLesson.objects.get_or_create(
        profile=profile, source_key=key, defaults={'evidence': evidence})
    PracticeEvaluation.objects.get_or_create(profile=profile, source_key=key)
    return lesson if created else None


def extract_saved_mistakes(profile):
    """Up to three lessons per request from recent recorded local mistakes."""
    state = profile.state
    games = [{'game': state.get('games', 1), 'moves': state.get('moves', []),
              'events': state.get('events', [])}, *state.get('archive', [])]
    created, examined = 0, 0
    checked = set(profile.practice_evaluations.values_list('source_key', flat=True))
    checked.update(profile.practice_lessons.values_list('source_key', flat=True))
    for game in games:
        by_position = {event.get('fen_before'): event for event in game.get('events', [])
                       if event.get('loss_cp', 0) >= MIN_LOSS}
        board = chess.Board()
        candidates = []
        for uci in game.get('moves', []):
            move = board.parse_uci(uci)
            event = by_position.get(board.fen())
            if event and event.get('move_uci') == uci:
                if source_key(board, move) not in checked:
                    candidates.append((event['loss_cp'], board.copy(), move))
            board.push(move)
        for loss, position, move in sorted(candidates, key=lambda item: -item[0]):
            examined += 1
            created += create_lesson(profile, position, move,
                f"Local game {game['game']} · move {position.fullmove_number}", known_loss=loss) is not None
            checked.add(source_key(position, move))
            if created >= 3 or examined >= 8:
                return {'created': created, 'examined': examined}
    return {'created': created, 'examined': examined}


def line_positions(fen, moves):
    board = chess.Board(fen)
    positions = [{'fen': fen, 'move': 'Start'}]
    for uci in moves:
        move = board.parse_uci(uci)
        san = board.san(move)
        board.push(move)
        positions.append({'fen': board.fen(), 'move': san})
    return positions


def attempt_public(attempt):
    data = attempt.lesson.evidence
    board = chess.Board(data['fen'])
    hints = []
    if attempt.hint_level:
        best = chess.Move.from_uci(data['best'])
        hints.append(f"Consider your {chess.piece_name(board.piece_type_at(best.from_square))} on "
                     f"{chess.square_name(best.from_square)}. Check the opponent's strongest reply.")
        if attempt.hint_level >= 2:
            hints.append(f"Try moving it to {chess.square_name(best.to_square)}.")
    result = {
        'id': str(attempt.pk), 'revision': attempt.revision, 'fen': data['fen'],
        'color': 'white' if board.turn else 'black', 'source': data['source'],
        'legal_moves': [move.uci() for move in board.legal_moves],
        'hints': hints, 'is_review': attempt.is_review,
        'started_at': attempt.started_at.isoformat(), 'finished': attempt.finished_at is not None,
    }
    if attempt.finished_at:
        result['result'] = attempt.result
    return result


def finish_attempt(attempt, move_text, reasoning, *, reveal=False):
    evidence = attempt.lesson.evidence
    board = chess.Board(evidence['fen'])
    move = None
    if not reveal:
        clean = move_text.strip().replace('0-0', 'O-O')
        try:
            move = board.parse_san(clean)
        except ValueError:
            move = board.parse_uci(clean)
        if move not in board.legal_moves:
            raise ValueError('Choose a legal move.')
    correct = move is not None and move.uci() in evidence['accepted']
    unaided = bool(correct and attempt.hint_level == 0)
    now = timezone.now()
    lesson = attempt.lesson
    lesson.stage = min(lesson.stage + 1, len(INTERVALS)) if unaided else 0
    days = INTERVALS[max(0, lesson.stage - 1)]
    lesson.due_at = now + timedelta(days=days)
    lesson.save(update_fields=['stage', 'due_at'])
    best = evidence['choices'][evidence['best']]
    original = evidence['choices'][evidence['original']]
    chosen = evidence['choices'][move.uci()] if move else best
    # Explain verifiable moves, not an inferred psychological diagnosis.
    attempt.result = {
        'correct': bool(correct), 'unaided': unaided, 'revealed': reveal,
        'played': board.san(move) if move else None,
        'original': original['san'], 'best': best['san'],
        'accepted': [evidence['choices'][uci]['san'] for uci in evidence['accepted']],
        'loss_cp': evidence['loss_cp'], 'engine': evidence['engine'], 'depth': evidence['depth'],
        'best_line': line_positions(evidence['fen'], best['pv']),
        'original_line': line_positions(evidence['fen'], original['pv']),
        'your_line': line_positions(evidence['fen'], chosen['pv']),
        'due_at': lesson.due_at.isoformat(), 'interval_days': days,
        'elapsed_seconds': max(0, round((now - attempt.started_at).total_seconds())),
        'reasoning': reasoning,
    }
    attempt.reasoning = reasoning
    attempt.finished_at = now
    attempt.save(update_fields=['reasoning', 'finished_at', 'result'])


def summary(profile):
    lessons = profile.practice_lessons.all()
    checks = PracticeAttempt.objects.filter(lesson__profile=profile, finished_at__isnull=False)
    delayed = checks.filter(is_review=True)
    next_due = lessons.filter(due_at__gt=timezone.now()).order_by('due_at').first()
    return {
        'lessons': lessons.count(), 'due': lessons.filter(due_at__lte=timezone.now()).count(),
        'completed': checks.count(), 'delayed_checks': delayed.count(),
        'delayed_unaided': sum(bool(a.result.get('unaided')) for a in delayed),
        'next_due': next_due.due_at.isoformat() if next_due else None,
    }


def export_practice(profile):
    return [{
        'id': str(lesson.pk), 'source_key': lesson.source_key, 'evidence': lesson.evidence,
        'due_at': lesson.due_at.isoformat(), 'stage': lesson.stage,
        'attempts': [{
            'id': str(a.pk), 'hint_level': a.hint_level, 'is_review': a.is_review,
            'started_at': a.started_at.isoformat(),
            'finished_at': a.finished_at.isoformat() if a.finished_at else None,
            'result': a.result,
        } for a in lesson.attempts.all()],
    } for lesson in profile.practice_lessons.prefetch_related('attempts')]
