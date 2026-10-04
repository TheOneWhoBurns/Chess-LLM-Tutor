(() => {
  'use strict';
  const $id = id => document.getElementById(id);
  const pieceNames = {p:'pawn', n:'knight', b:'bishop', r:'rook', q:'queen', k:'king'};
  let attempt = null, stats = null, busy = false, selected = null, lineIndex = 0;
  let lastDrop = {square:null, at:0};
  const board = Chessboard('practice-board', {
    position:'start', draggable:true, pieceTheme:'/static/img/chesspieces/wikipedia/{piece}.png',
    onDragStart: source => {
      if (busy || !attempt || attempt.finished) return false;
      return attempt.legal_moves.some(move => move.startsWith(source));
    },
    onDrop: (source, target) => {
      lastDrop = {square:target, at:Date.now()};
      if (source === target) selected = selected === source ? null : source;
      else chooseMove(source, target);
      highlight();
      return 'snapback';
    },
    onSnapbackEnd: () => highlight(),
    onMoveEnd: () => highlight()
  });

  function highlight() {
    const pieces = board.position();
    document.querySelectorAll('#practice-board .square-55d63').forEach(square => {
      square.classList.toggle('practice-selected', square.dataset.square === selected);
      const piece = pieces[square.dataset.square];
      square.setAttribute('aria-label', square.dataset.square + (piece
        ? ` ${piece[0] === 'w' ? 'white' : 'black'} ${pieceNames[piece[1].toLowerCase()]}` : ' empty'));
    });
  }

  function chooseMove(from, to) {
    if (!attempt || busy || attempt.finished) return;
    const base = from + to;
    const moves = attempt.legal_moves.filter(move => move.startsWith(base));
    if (!moves.length) return;
    const uci = moves.find(move => move.length === 4 || move[4] === $id('practice-promotion').value);
    if (!uci) return;
    const game = new Chess(attempt.fen);
    const result = game.move({from, to, promotion:uci[4]});
    $id('practice-move').value = result ? result.san : uci;
    selected = null;
    highlight();
    $id('practice-reason').focus();
  }

  $id('practice-board').addEventListener('click', event => {
    const square = event.target.closest('[data-square]');
    if (!square || !attempt || attempt.finished || busy) return;
    const destination = square.dataset.square;
    if (destination === lastDrop.square && Date.now() - lastDrop.at < 200) return;
    if (selected && attempt.legal_moves.some(move => move.startsWith(selected + destination))) {
      chooseMove(selected, destination);
    } else {
      selected = attempt.legal_moves.some(move => move.startsWith(destination)) ? destination : null;
      highlight();
    }
  });

  function updateStats() {
    if (!stats) return;
    $id('due-count').textContent = stats.due;
    $id('lesson-count').textContent = stats.lessons;
    $id('retention-count').textContent = `${stats.delayed_unaided}/${stats.delayed_checks}`;
    $id('start-practice').textContent = stats.due ? 'Start practice' : 'No lessons due';
    if (!stats.lessons) {
      $id('empty-title').textContent = 'Start with a real decision.';
      $id('empty-copy').textContent = 'Find lessons from mistakes in your saved local games. Each position gets a fresh engine check before it enters your practice queue.';
    } else if (!stats.due) {
      $id('empty-title').textContent = 'Give it time to stick.';
      $id('empty-copy').textContent = stats.next_due
        ? `Your next check is due ${new Date(stats.next_due).toLocaleString()}. Come back without reviewing the answer first.`
        : 'You have finished the available lessons.';
    } else {
      $id('empty-title').textContent = 'Your next decision is ready.';
      $id('empty-copy').textContent = 'Take a moment to work out a move before asking for a hint. Your first answer is recorded.';
    }
  }

  function controls() {
    document.querySelectorAll('main button, main input, main textarea, main select').forEach(el => { el.disabled = busy; });
    $id('start-practice').disabled = busy || !stats || !stats.due;
    $id('get-hint').disabled = busy || !attempt || attempt.hints.length >= 2;
    if (attempt && attempt.finished) {
      const line = attempt.result[$id('line-choice').value];
      $id('line-back').disabled = busy || lineIndex === 0;
      $id('line-forward').disabled = busy || lineIndex >= line.length - 1;
    }
    $id('find-lessons').textContent = busy ? 'Working…' : 'Find lessons in saved games';
  }

  function renderLine() {
    if (!attempt || !attempt.finished) return;
    const line = attempt.result[$id('line-choice').value];
    lineIndex = Math.max(0, Math.min(lineIndex, line.length - 1));
    board.position(line[lineIndex].fen, false);
    highlight();
    $id('position-color').textContent = `${line[lineIndex].fen.split(' ')[1] === 'w' ? 'White' : 'Black'} to move${lineIndex ? ' · Replay' : ''}`;
    $id('line-moves').textContent = line.slice(1).map(row => row.move).join('  ');
    $id('line-position').textContent = `${lineIndex ? line[lineIndex].move : 'Starting position'} · ${lineIndex} of ${line.length - 1} half-moves`;
    controls();
  }

  function renderAttempt(next) {
    const changed = !attempt || !next || next.id !== attempt.id;
    const justFinished = next && next.finished && (!attempt || !attempt.finished || changed);
    attempt = next;
    $id('practice-empty').hidden = !!attempt;
    $id('lesson-active').hidden = !attempt || attempt.finished;
    $id('lesson-result').hidden = !attempt || !attempt.finished;
    $id('continuation').hidden = !attempt || !attempt.finished;
    if (changed) {
      selected = null;
      $id('practice-move').value = '';
      $id('practice-reason').value = '';
    }
    if (!attempt) {
      board.position('start', false);
      $id('position-color').textContent = 'Your next decision';
      $id('position-source').textContent = 'Positions from your saved games';
      $id('attempt-kind').textContent = 'Practice';
      return;
    }
    if (changed) board.orientation(attempt.color);
    $id('practice-promotion').hidden = !attempt.legal_moves.some(move => move.length === 5);
    board.position(attempt.fen, false);
    highlight();
    $id('position-color').textContent = `${attempt.color === 'white' ? 'White' : 'Black'} to move`;
    $id('attempt-kind').textContent = attempt.is_review ? 'Delayed retention check' : 'First retry';
    $id('position-source').textContent = attempt.source;
    $id('practice-hints').replaceChildren();
    attempt.hints.forEach(hint => {
      const p = document.createElement('p');
      p.textContent = hint;
      $id('practice-hints').appendChild(p);
    });
    $id('practice-hints').hidden = !attempt.hints.length;
    if (attempt.finished) {
      const r = attempt.result;
      $id('result-label').textContent = r.unaided ? 'SOLVED WITHOUT HINTS' : r.correct ? 'SOLVED WITH HELP' : 'A POSITION TO REVISIT';
      $id('result-title').textContent = r.correct ? 'A sound decision.' : r.revealed ? 'Study the continuation.' : 'Look one reply further.';
      $id('result-copy').textContent = r.correct
        ? `${r.played} is within the lesson's accepted engine range. ${r.unaided ? 'Now check the continuation you expected.' : 'Try it without help when it returns.'}`
        : `${r.played ? r.played + ' falls outside the accepted range. ' : ''}Compare the continuations to see what changes.`;
      $id('original-move').textContent = r.original;
      $id('sound-moves').textContent = r.accepted.join(', ');
      $id('result-reason').hidden = !r.reasoning;
      $id('result-reason').textContent = `Your expectation: ${r.reasoning}`;
      $id('next-review').textContent = `Returns ${new Date(r.due_at).toLocaleString()} · ${r.interval_days} day${r.interval_days === 1 ? '' : 's'}`;
      $id('engine-evidence').textContent = `Checked with ${r.engine}, minimum reported depth ${r.depth}. Finite engine analysis can change. Your explanation is saved for reflection, not automatically graded.`;
      if (justFinished) { lineIndex = 0; $id('line-choice').value = 'best_line'; }
      renderLine();
    }
  }

  async function request(action, extra = {}) {
    if (busy) return;
    busy = true;
    $id('practice-error').hidden = true;
    if (action === 'extract') $id('practice-status').textContent = 'Checking saved mistakes with Stockfish…';
    controls();
    try {
      const options = action ? {
        method:'POST', headers:{'Content-Type':'application/json',
          'X-CSRFToken':document.querySelector('[name=csrfmiddlewaretoken]').value},
        body:JSON.stringify({action, id:attempt?.id, revision:attempt?.revision, ...extra})
      } : {};
      const response = await fetch('/api/practice/', options);
      const data = await response.json();
      if (data.summary) { stats = data.summary; updateStats(); }
      if (Object.hasOwn(data, 'attempt')) renderAttempt(data.attempt);
      if (!response.ok) throw new Error(data.error || 'Practice could not complete. Please retry.');
      $id('practice-status').textContent = data.extracted
        ? data.extracted.created
          ? `${data.extracted.created} new lesson${data.extracted.created === 1 ? '' : 's'} ready. Find lessons again to check more saved mistakes.`
          : 'No new confirmed mistakes found in the checked positions. Play a local game, or import lessons from your downloaded history.'
        : 'Work at your own pace. Your first answer and any hints are recorded.';
    } catch (error) {
      $id('practice-error').textContent = error.message || 'Connection lost. Refresh to resume your saved attempt.';
      $id('practice-error').hidden = false;
      $id('practice-status').textContent = 'Your saved game is unchanged.';
    } finally {
      busy = false;
      controls();
    }
  }

  $id('find-lessons').addEventListener('click', () => request('extract'));
  $id('start-practice').addEventListener('click', () => request('start'));
  $id('next-lesson').addEventListener('click', () => request('start'));
  $id('get-hint').addEventListener('click', () => request('hint'));
  $id('reveal-answer').addEventListener('click', () => request('reveal', {reasoning:$id('practice-reason').value}));
  $id('practice-form').addEventListener('submit', event => {
    event.preventDefault();
    request('submit', {move:$id('practice-move').value, reasoning:$id('practice-reason').value});
  });
  $id('practice-flip').addEventListener('click', () => { board.flip(); highlight(); });
  $id('line-choice').addEventListener('change', () => { lineIndex = 0; renderLine(); });
  $id('line-back').addEventListener('click', () => { lineIndex--; renderLine(); });
  $id('line-forward').addEventListener('click', () => { lineIndex++; renderLine(); });
  window.addEventListener('resize', () => { board.resize(); highlight(); });
  request();
})();
