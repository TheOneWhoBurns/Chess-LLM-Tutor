"""NEMESIS: strong search plus expected replies from a personal human policy.

Stockfish supplies chess values. Frozen Maia supplies human move priors. A
separate neural adapter learns the player's actual choices. There is no toy
engine fallback, and no position-category heuristic is presented as a diagnosis.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager
import math
import os
from pathlib import Path
import shutil
import threading

import chess
import chess.engine

from .maia_policy import MaiaPolicy, MaiaUnavailable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STOCKFISH = ROOT / '.runtime/stockfish/stockfish/stockfish-macos-universal'
MATE = 100000
MAX_COST = 65
REGRET_CAP = 2000
ROOT_NODES = 80000
REPLY_NODES = 120000


class EngineUnavailable(RuntimeError):
    pass


class Stockfish:
    def __init__(self, path=None):
        self.path = str(path or os.environ.get('NEMESIS_STOCKFISH') or shutil.which('stockfish') or DEFAULT_STOCKFISH)
        self.engine = None
        self.name = 'Stockfish 19'
        self.timeout = 20.0

    def _start(self):
        if self.engine is not None:
            return self.engine
        if not Path(self.path).is_file():
            raise EngineUnavailable('Stockfish is missing. Run python scripts/setup_engines.py or set NEMESIS_STOCKFISH.')
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.path, timeout=self.timeout)
            self.name = self.engine.id.get('name', 'Stockfish')
            if 'Stockfish' not in self.name:
                self.close()
                raise EngineUnavailable('NEMESIS_STOCKFISH must point to a Stockfish executable.')
            self.engine.configure({'Threads': 2, 'Hash': 64})
        except (OSError, chess.engine.EngineError, TimeoutError) as exc:
            self.close()
            raise EngineUnavailable('Stockfish could not start. Check NEMESIS_STOCKFISH and the engine installation.') from exc
        return self.engine

    def rank(self, board, count=6, nodes=ROOT_NODES):
        if board.is_game_over():
            return []
        expected = min(count, board.legal_moves.count())
        timer = None
        timed_out = threading.Event()
        try:
            engine = self._start()

            def expire():
                # python-chess disables its analyse() deadline for nodes-only
                # limits. Closing the transport kills a hung child and releases
                # the waiting coroutine without extending the deadline for quit.
                timed_out.set()
                engine.close()

            timer = threading.Timer(self.timeout, expire)
            timer.daemon = True
            timer.start()
            analysis = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=expected,
                                      info=chess.engine.INFO_SCORE | chess.engine.INFO_PV | chess.engine.INFO_BASIC)
            timer.cancel()
            timer.join()
            if timed_out.is_set():
                raise TimeoutError('Stockfish analysis exceeded its deadline')
            rows = []
            for info in analysis:
                if not info.get('pv') or info['pv'][0] not in board.legal_moves or 'score' not in info:
                    raise EngineUnavailable('Stockfish returned an incomplete analysis. The move was not saved.')
                score = info['score'].pov(board.turn)
                rows.append({'move': info['pv'][0], 'score': score.score(mate_score=MATE),
                             'mate': score.mate(), 'depth': info.get('depth', 0),
                             'nodes': info.get('nodes', 0)})
            if len(rows) != expected or len({r['move'] for r in rows}) != expected:
                raise EngineUnavailable('Stockfish did not evaluate the required legal moves. Please retry.')
            return sorted(rows, key=lambda r: (-r['score'], r['move'].uci()))
        except (OSError, chess.engine.EngineError, TimeoutError) as exc:
            if timed_out.is_set():
                self.engine = None
                engine.close()
                raise EngineUnavailable(
                    f'Stockfish analysis timed out after {self.timeout:g} seconds. Your board and training data are unchanged.'
                ) from exc
            self.close()
            raise EngineUnavailable('Stockfish analysis failed. Your board and training data are unchanged.') from exc
        finally:
            if timer is not None:
                timer.cancel()
                timer.join()

    def close(self):
        engine, self.engine = self.engine, None
        if engine:
            try:
                engine.quit()
            except (chess.engine.EngineError, TimeoutError):
                engine.close()


class EngineRoom:
    def __init__(self, stockfish=None, maia=None):
        self.stockfish = stockfish or Stockfish()
        self.maia = maia or MaiaPolicy()
        self.lock = threading.RLock()

    @contextmanager
    def turn(self):
        with self.lock:
            yield self

    def status(self):
        problems = []
        if not Path(self.stockfish.path).is_file():
            problems.append('Stockfish has not been installed.')
        if not self.maia.weights_path.is_file():
            problems.append('Maia weights have not been installed.')
        if not shutil.which(self.maia.engine_path) and not Path(self.maia.engine_path).is_file():
            problems.append('LC0 has not been installed.')
        return {'ready': not problems, 'engine': self.stockfish.name,
                'model': 'Maia 1500 + personal policy',
                'engine_error': ' '.join(problems) or None}

    def close(self):
        with self.lock:
            self.stockfish.close()
            self.maia.close()

    def observe(self, board, move, policy):
        """Log the prediction made BEFORE learning; evaluate from the human POV."""
        prior = self.maia.probabilities(board)
        ranks = self.stockfish.rank(board, count=board.legal_moves.count(), nodes=REPLY_NODES)
        best = ranks[0]
        played = next(row for row in ranks if row['move'] == move)
        raw_loss = max(0, best['score'] - played['score'])
        diagnostics = policy.learn(board, prior, move.uci())
        diagnostics['predicted_move'] = board.san(chess.Move.from_uci(diagnostics['top_prediction']))
        diagnostics['prior_prediction'] = board.san(chess.Move.from_uci(max(prior, key=prior.get)))
        return {**diagnostics, 'loss_cp': min(REGRET_CAP, raw_loss), 'raw_loss_cp': raw_loss,
                'alternative': board.san(best['move']), 'engine': self.stockfish.name,
                'depth': min(r['depth'] for r in ranks), 'nodes': max(r['nodes'] for r in ranks),
                'quality': 'Blunder' if raw_loss >= 200 else 'Mistake' if raw_loss >= 100 else 'Sound',
                'fen_before': board.fen(), 'move_uci': move.uci(),
                'prior_hit': max(prior, key=prior.get) == move.uci(),
                'personal_hit': diagnostics['top_prediction'] == move.uci()}

    def _candidate(self, board, candidate, policy):
        next_board = board.copy()
        san = next_board.san(candidate['move'])
        next_board.push(candidate['move'])
        report = {'move': san, 'uci': candidate['move'].uci(), 'engine_score_cp': candidate['score'],
                  'reply_score_cp': candidate['score'],
                  'expected_regret_cp': 0., 'prior_expected_regret_cp': 0., 'replies': [],
                  'depth': candidate['depth'], 'nodes': candidate['nodes']}
        if next_board.is_game_over():
            report['reply_score_cp'] = MATE - 1 if next_board.is_checkmate() else 0
            return report
        prior = self.maia.probabilities(next_board)
        personal = policy.distribution(next_board, prior)
        replies = self.stockfish.rank(next_board, count=next_board.legal_moves.count(), nodes=REPLY_NODES)
        best = replies[0]['score']
        # This later search sees the HUMAN to move. Negate its best reply to
        # compare the candidate from NEMESIS's perspective. A refutation found
        # here supersedes the earlier, smaller root search for reranking.
        report['reply_score_cp'] = -best
        for reply in replies:
            uci = reply['move'].uci()
            loss = min(REGRET_CAP, max(0, best - reply['score']))
            report['replies'].append({'move': next_board.san(reply['move']), 'uci': uci,
                                      'prior_probability': prior[uci], 'personal_probability': personal[uci],
                                      'loss_cp': loss})
        report['expected_regret_cp'] = math.fsum(r['personal_probability'] * r['loss_cp'] for r in report['replies'])
        report['prior_expected_regret_cp'] = math.fsum(r['prior_probability'] * r['loss_cp'] for r in report['replies'])
        report['depth'] = min(r['depth'] for r in replies)
        report['nodes'] += max(r['nodes'] for r in replies)
        report['replies'].sort(key=lambda r: -r['personal_probability'])
        return report

    def choose(self, board, policy, mode='adaptive'):
        ranks = self.stockfish.rank(board)
        if not ranks:
            raise ValueError('There is no opponent move in a finished game.')
        baseline = ranks[0]
        pool = [row for row in ranks if row['score'] >= baseline['score'] - MAX_COST]
        # A mate-band score is a forced-search result, never a trade for expected regret.
        locked = abs(baseline['score']) >= MATE - 1000
        if locked or mode == 'baseline':
            pool = [baseline]
        reports = [self._candidate(board, row, policy) for row in pool]
        refreshed_best = max(report['reply_score_cp'] for report in reports)
        refreshed_mate = abs(refreshed_best) >= MATE - 1000
        for report in reports:
            report['engine_cost_cp'] = baseline['score'] - report['engine_score_cp']
            report['reply_cost_cp'] = refreshed_best - report['reply_score_cp']
            report['eligible'] = True
            report['guard_reason'] = None
            if not locked and mode != 'baseline':
                if report['reply_score_cp'] <= -MATE + 1000 and refreshed_best > -MATE + 1000:
                    report['eligible'] = False
                    report['guard_reason'] = 'reply_search_found_losing_mate'
                elif refreshed_mate and report['reply_score_cp'] != refreshed_best:
                    report['eligible'] = False
                    report['guard_reason'] = 'preserve_best_reply_search_mate'
                elif report['reply_cost_cp'] > MAX_COST:
                    report['eligible'] = False
                    report['guard_reason'] = 'reply_cost_exceeds_limit'
        eligible = [report for report in reports if report['eligible']]
        prior_choice = max(eligible, key=lambda r: r['reply_score_cp'] + r['prior_expected_regret_cp'])
        personal_choice = max(eligible, key=lambda r: r['reply_score_cp'] + r['expected_regret_cp'])
        choice = reports[0] if locked or mode == 'baseline' else personal_choice
        for report in reports:
            report['selected'] = report is choice
        decision = {key: value for key, value in choice.items() if key != 'uci'}
        decision.update(baseline_move=board.san(baseline['move']), prior_move=prior_choice['move'],
                        personal_changed=policy.samples > 0 and choice['uci'] != prior_choice['uci'],
                        engine_changed=choice['uci'] != baseline['move'].uci(),
                        candidates=[{k:v for k,v in r.items() if k not in ('replies','uci')} for r in reports],
                        reply_baseline_move=max(reports, key=lambda r: r['reply_score_cp'])['move'],
                        engine=self.stockfish.name, policy=self.maia.info['name'],
                        sample_count=policy.samples, cost_limit_cp=MAX_COST, regret_cap_cp=REGRET_CAP,
                        root_nodes=max(r['nodes'] for r in ranks), root_depth=min(r['depth'] for r in ranks))
        return chess.Move.from_uci(choice['uci']), decision


_ROOM = EngineRoom()
atexit.register(_ROOM.close)


def engine_room():
    return _ROOM
