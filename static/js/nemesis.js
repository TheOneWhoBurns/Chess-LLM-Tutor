/* Server-authoritative chess state. All model and engine values come from the API. */
(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const pieceNames = {p:'pawn', n:'knight', b:'bishop', r:'rook', q:'queen', k:'king'};
  let state = null, board = null, busy = false, connected = false, selected = null;
  let pendingPromotion = null, confirmation = null, lastDrag = 0, retryTimer = null;
  let chat = {messages:[], ready:false, model:null};
  let chatLoading = true, chatLoadActive = false, chatBusy = false, chatFailure = '', chatRetryMode = 'load';
  let pendingChat = null, renderedChat = null;
  const number = (value, digits = 0) => value === null || value === undefined || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(digits);
  const probability = value => value === null || value === undefined ? '—' : `${number(Number(value) * 100, 1)}%`;
  const cp = value => value === null || value === undefined ? '—' : `${number(value)} cp`;
  const count = value => value === null || value === undefined || !Number.isFinite(Number(value)) ? '—' : Number(value).toLocaleString('en-US', {maximumFractionDigits:0});
  const node = (tag, className, text) => {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text !== undefined) item.textContent = text;
    return item;
  };
  const put = (id, value) => {byId(id).textContent = value;};
  const legalMoves = () => Array.isArray(state?.legal_moves) ? state.legal_moves : [];
  const available = () => !!state && connected && state.ready === true && !state.engine_error;
  const canPlay = () => available() && !busy && !state.game_over && state.turn !== 'black' && !pendingPromotion;

  function notice(message, reconnect = false) {
    put('notice-copy', message || '');
    byId('notice').hidden = !message;
    byId('reconnect').hidden = !reconnect;
  }

  function setBusy(value) {
    busy = value;
    document.body.classList.toggle('busy', value);
    byId('board').setAttribute('aria-busy', String(value));
    renderStatus();
  }

  function renderStatus() {
    renderChatControls();
    const ready = available();
    byId('engine-status').classList.toggle('ready', ready);
    byId('engine-status').classList.toggle('error', (!connected && !busy) || !!state?.engine_error);
    put('engine-name', !connected ? busy ? 'Connecting…' : 'Disconnected' : state?.engine || 'Loading engine');
    byId('engine-status').title = state?.engine_error || (state ? `${state.engine} · ${state.model}` : 'Connecting to the local server');
    const locked = busy || !ready;
    ['new-game', 'mode-label'].forEach(id => {byId(id).disabled = locked;});
    byId('forget-profile').disabled = busy || !state;
    ['move-input', 'resign', 'claim-draw'].forEach(id => {byId(id).disabled = locked || !!state?.game_over;});
    byId('move-form').querySelector('button').disabled = locked || !!state?.game_over;
    if (!state) {
      put('turn-status', busy ? 'Loading your position…' : 'Connect to load your game');
      put('opponent-status', busy ? 'Connecting to engine' : 'Engine unavailable');
      put('model-status', 'Waiting for engine');
      return;
    }
    put('player-name', state.profile?.username || state.profile?.training?.username || 'You');
    put('mode-label', state.mode === 'baseline' ? 'Baseline' : 'Adaptive');
    if (!ready) {
      put('turn-status', !connected ? 'Connection interrupted' : 'Engine unavailable');
      put('opponent-status', state.engine_error ? 'Engine needs attention' : 'Waiting for engine');
      put('model-status', !connected ? 'Disconnected' : 'Engine unavailable');
    } else if (busy) {
      put('turn-status', 'NEMESIS is thinking…');
      put('opponent-status', state.mode === 'baseline' ? 'Searching with Stockfish' : 'Evaluating continuations');
      put('model-status', 'Calculating…');
    } else {
      const result = state.result === '1-0' ? 'You win' : state.result === '0-1' ? 'NEMESIS wins' : 'Draw';
      put('turn-status', state.game_over ? result : state.in_check ? 'Your move · In check' : state.turn === 'black' ? 'Black to move' : 'Your move');
      const samples = state.profile?.samples || 0;
      const liveSamples = state.profile?.live_samples ?? (state.profile?.training ? 0 : samples);
      put('opponent-status', state.game_over ? 'Game complete' : state.mode === 'baseline' ? 'Stockfish · Best move' : samples ? 'Maia + your personal policy' : 'Maia 1500 · No personal history');
      put('model-status', state.mode === 'baseline' ? 'Baseline · Model still learning' : state.profile?.training ? `History fitted · ${count(liveSamples)} live choices` : samples ? `${count(samples)} observed move${samples === 1 ? '' : 's'}` : 'Maia prior · 0 observed moves');
    }
    byId('game-result').hidden = !state.game_over;
    put('game-result', state.result === '*' ? '' : state.result || '');
  }

  async function readResponse(response) {
    try {return await response.json();} catch (_) {throw new Error(`The server returned an unreadable response (${response.status}).`);}
  }

  function coachName(model = chat.model) {
    return model === 'gpt-6-astra' ? 'Astra' : model || 'Coach';
  }

  function renderChatControls() {
    const canSend = chat.ready && !chatLoading && !chatBusy && !busy && !!state && connected;
    byId('chat-input').disabled = chatBusy;
    byId('chat-send').disabled = !canSend || !byId('chat-input').value.trim();
    byId('chat-form').setAttribute('aria-busy', String(chatBusy));
    byId('chat-thinking').hidden = !chatBusy;
    byId('chat-retry').disabled = chatLoading || chatBusy || busy;
    document.querySelectorAll('[data-chat-prompt]').forEach(button => {button.disabled = chatBusy;});
    byId('chat-model').classList.toggle('ready', chat.ready && !chatLoading && !chatFailure);
    put('chat-model', chatLoading ? 'Connecting…' : !chat.ready ? `${coachName()} · Unavailable` : chatFailure ? `${coachName()} · Check connection` : coachName());
    byId('chat-model').title = chat.model || 'Connecting to the configured coaching model';
    if (state) {
      const move = Math.floor((state.moves?.length || 0) / 2) + 1;
      put('chat-context', `Game ${state.profile?.games || 1} · ${state.game_over ? 'Game complete' : `Move ${move} · ${state.turn === 'black' ? 'Black' : 'White'} to move`}`);
    }
    const error = chatFailure || (!chatLoading && !chat.ready ? 'The coaching connection is unavailable.' : '');
    byId('chat-error').hidden = !error;
    put('chat-error-copy', error);
    byId('chat-retry').hidden = chatRetryMode === 'none';
    put('chat-retry', chatRetryMode === 'send' ? 'Retry message' : 'Check connection');
  }

  function renderChat(scroll = false) {
    const history = byId('chat-messages');
    const messages = Array.isArray(chat.messages) ? chat.messages.filter(message => ['user', 'assistant'].includes(message.role) && typeof message.content === 'string') : [];
    const key = JSON.stringify([messages, chatBusy ? pendingChat?.message : null, chat.model]);
    if (key !== renderedChat) {
      const nearBottom = history.scrollHeight - history.scrollTop - history.clientHeight < 60;
      const oldScroll = history.scrollTop;
      history.replaceChildren();
      const add = (message, pending = false) => {
        const entry = node('article', `chat-message ${message.role}${pending ? ' pending' : ''}`);
        const heading = node('div', 'chat-message-heading');
        heading.append(node('strong', '', message.role === 'user' ? 'You' : coachName(message.model || chat.model)));
        if (message.context_label || pending) heading.append(node('span', 'chat-message-context', pending ? 'Sending…' : message.context_label));
        entry.append(heading, node('div', 'chat-message-content', message.content));
        history.append(entry);
      };
      messages.forEach(message => add(message));
      if (chatBusy && pendingChat) add({role:'user', content:pendingChat.message}, true);
      if (!messages.length && !chatBusy) {
        const empty = node('div', 'chat-empty');
        empty.append(node('h3', '', 'Talk through your game.'), node('p', '', 'Ask about a mistake, NEMESIS’s last move, or what to practice next.'));
        history.append(empty);
      }
      renderedChat = key;
      history.scrollTop = scroll || nearBottom ? history.scrollHeight : oldScroll;
    } else if (scroll) history.scrollTop = history.scrollHeight;
    renderChatControls();
  }

  async function loadChat() {
    if (chatBusy || chatLoadActive) return;
    chatLoadActive = true;
    chatLoading = true;
    renderChatControls();
    try {
      const response = await fetch('/api/chat/', {cache:'no-store', signal:AbortSignal.timeout(20000)});
      const result = await readResponse(response);
      if (Array.isArray(result.messages)) chat.messages = result.messages;
      if (typeof result.model === 'string') chat.model = result.model;
      chat.ready = response.ok && result.ready === true;
      chatFailure = result.error || (response.ok ? '' : `Cannot load the coach (${response.status}).`);
      chatRetryMode = 'load';
    } catch (error) {
      chat.ready = false;
      chatFailure = error.name === 'TimeoutError' ? 'The coaching connection timed out. Your game is still available.' : error.message || 'Cannot connect to the coach. Your game is still available.';
      chatRetryMode = 'load';
    } finally {
      chatLoading = false;
      chatLoadActive = false;
      renderChat(true);
    }
  }

  async function sendChat(retry = false) {
    const message = retry && pendingChat ? pendingChat.message : byId('chat-input').value.trim();
    if (!message || !chat.ready || chatLoading || chatBusy || busy || !state || !connected) return;
    if (!pendingChat || pendingChat.message !== message) {
      pendingChat = {message, revision:state.revision, request_id:crypto.randomUUID()};
    }
    const request = pendingChat;
    const restoreFocus = byId('chat-panel').contains(document.activeElement);
    chatBusy = true;
    chatFailure = '';
    renderChat(true);
    try {
      const response = await fetch('/api/chat/', {
        method:'POST', headers:{'Content-Type':'application/json', 'X-CSRFToken':document.querySelector('[name=csrfmiddlewaretoken]').value},
        body:JSON.stringify(request), signal:AbortSignal.timeout(150000)
      });
      const result = await readResponse(response);
      if (Array.isArray(result.messages)) chat.messages = result.messages;
      if (typeof result.model === 'string') chat.model = result.model;
      if (typeof result.ready === 'boolean') chat.ready = result.ready;
      if (!response.ok) {
        chatFailure = result.error || `The coach could not reply (${response.status}).`;
        chatRetryMode = chat.ready ? 'send' : 'load';
        if (response.status === 409 && result.error_code === 'chat_busy') {
          chatFailure += ' Retry in a moment to check for the reply.';
          chatRetryMode = 'send';
        } else if (response.status === 409) {
          pendingChat = null;
          chatRetryMode = 'none';
          chatFailure += ' Your draft is kept. Please send it again.';
          if (!busy) {
            try {await loadState();} catch (_) {notice('Reconnect to refresh the board before sending your message.', true);}
          }
        }
        return;
      }
      if (!Array.isArray(result.messages)) throw new Error('The server did not return the conversation. Retry to check your saved reply.');
      pendingChat = null;
      byId('chat-input').value = '';
      chatRetryMode = 'load';
      if (!busy && result.revision !== undefined && result.revision !== state.revision) {
        try {await loadState();} catch (_) {notice('Reconnect to refresh the saved board position.', true);}
      }
    } catch (error) {
      chatFailure = error.name === 'TimeoutError' ? 'The coach is taking longer than expected. Retry to check for your reply; your message will not be sent twice.' : error.message || 'The coaching connection was interrupted. Your draft is kept.';
      chatRetryMode = 'send';
    } finally {
      chatBusy = false;
      renderChat(true);
      if (restoreFocus && !byId('chat-panel').hidden && (document.activeElement === document.body || byId('chat-panel').contains(document.activeElement))) byId('chat-input').focus({preventScroll:true});
    }
  }

  async function loadState() {
    const response = await fetch('/api/state/', {cache:'no-store', signal:AbortSignal.timeout(120000)});
    const result = await readResponse(response);
    if (result.state) state = result.state;
    else if (result.fen) state = result;
    connected = true;
    if (state) render();
    if (!response.ok) throw new Error(result.error || 'The engine is not available yet. Reconnect when it is ready.');
    if (!result.fen) throw new Error('The server did not return a game position.');
    if (state.engine_error) notice(state.engine_error, true);
    else if (!state.ready) notice('The engine is still starting. Reconnect to check its status.', true);
    return state;
  }

  async function reconnect() {
    if (busy) return;
    clearTimeout(retryTimer);
    notice('Connecting to your saved game…');
    setBusy(true);
    try {await loadState(); if (available()) notice('');}
    catch (error) {connected = false; notice(error.message || 'Cannot connect to the local server.', true);}
    finally {setBusy(false);}
  }

  async function act(action, extra = {}) {
    if (busy || !state || (action !== 'forget' && !available())) return;
    const restoreMoveFocus = action === 'move' && byId('move-form').contains(document.activeElement);
    notice('');
    selected = null;
    setBusy(true);
    highlightSquares();
    try {
      const response = await fetch('/api/action/', {
        method:'POST', headers:{'Content-Type':'application/json', 'X-CSRFToken':document.querySelector('[name=csrfmiddlewaretoken]').value},
        body:JSON.stringify({action, revision:state.revision, ...extra}), signal:AbortSignal.timeout(120000)
      });
      const result = await readResponse(response);
      if (!response.ok) {
        if (result.state) {state = result.state; render();}
        throw new Error(result.error || `The move could not be saved (${response.status}).`);
      }
      if (!result.fen) throw new Error('The server did not return the updated position.');
      state = result;
      connected = true;
      render();
      if (action === 'move') byId('move-input').value = '';
      if (state.engine_error) notice(state.engine_error, true);
    } catch (error) {
      const message = error.name === 'TimeoutError' ? 'The engine took too long to respond. Checking your saved position…' : error.message || 'Connection interrupted.';
      notice(message);
      try {await loadState(); if (available()) notice(message);}
      catch (_) {connected = false; notice(`${message} Reconnect to check your saved position.`, true);}
    } finally {
      setBusy(false);
      highlightSquares();
      if (restoreMoveFocus && canPlay()) byId('move-input').focus({preventScroll:true});
    }
  }

  function play(from, to) {
    if (!canPlay()) return;
    const moves = legalMoves().filter(move => move.startsWith(from + to));
    if (!moves.length) return;
    if (moves.some(move => move.length === 5)) {
      pendingPromotion = from + to;
      byId('promotion-dialog').showModal();
    } else act('move', {move:moves[0]});
  }

  function highlightSquares() {
    document.querySelectorAll('#board [data-square]').forEach(square => {
      const name = square.dataset.square;
      square.classList.remove('last-square', 'selected-square', 'legal-square', 'check-square');
      if (typeof state?.last_move === 'string' && [state.last_move.slice(0, 2), state.last_move.slice(2, 4)].includes(name)) square.classList.add('last-square');
      if (selected === name) square.classList.add('selected-square');
      if (selected && legalMoves().some(move => move.startsWith(selected + name))) square.classList.add('legal-square');
      const code = square.querySelector('img')?.getAttribute('data-piece');
      if (state?.in_check && code === (state.turn === 'black' ? 'bK' : 'wK')) square.classList.add('check-square');
      square.setAttribute('aria-label', name + (code ? ` ${code[0] === 'w' ? 'white' : 'black'} ${pieceNames[code[1].toLowerCase()]}` : ' empty'));
    });
  }

  function table(headers, rows, className = '') {
    const result = node('table', `data-table ${className}`);
    const head = node('thead');
    const labels = node('tr');
    headers.forEach(label => {const th = node('th', '', label); th.scope = 'col'; labels.append(th);});
    head.append(labels);
    const body = node('tbody');
    rows.forEach(row => {
      const tr = node('tr', row.selected ? 'selected-row' : '');
      if (row.title) tr.title = row.title;
      row.values.forEach((value, index) => tr.append(node('td', row.classes?.[index], value)));
      body.append(tr);
    });
    result.append(head, body);
    return result;
  }

  function renderMoves() {
    const history = byId('move-history');
    const moves = state.moves || [];
    const nearBottom = history.scrollHeight - history.scrollTop - history.clientHeight < 45;
    history.replaceChildren();
    put('move-count', `Game ${state.profile?.games || 1}`);
    for (let index = 0; index < moves.length; index += 2) {
      const pair = node('div', 'move-pair');
      pair.append(node('small', '', `${index / 2 + 1}.`));
      pair.append(node('span', index === moves.length - 1 ? 'current-move' : '', moves[index]));
      pair.append(node('span', index + 1 === moves.length - 1 ? 'current-move' : '', moves[index + 1] || ''));
      history.append(pair);
    }
    if (!moves.length) history.append(node('p', 'empty-moves', 'White to move.'));
    if (nearBottom || busy) history.scrollTop = history.scrollHeight;
  }

  function renderDecision() {
    const decision = state.decision;
    byId('analysis-empty').hidden = !!decision;
    byId('decision-detail').hidden = !decision;
    put('model-name', state.model || 'Maia 1500 + personal policy');
    if (!decision) return;
    put('decision-move', decision.move || '—');
    put('baseline-move', decision.baseline_move || '—');
    put('engine-cost', cp(decision.engine_cost_cp));
    put('expected-regret', cp(decision.expected_regret_cp));
    put('search-detail', decision.depth ? `Depth ${decision.depth}` : decision.nodes ? `${Number(decision.nodes).toLocaleString()} nodes` : '');
    const personal = Number(state.profile?.samples) > 0;
    let label = 'Maia prior';
    let description = 'Selected using Maia’s predicted replies. There is no personal history yet.';
    if (state.mode === 'baseline') {
      label = 'Stockfish baseline';
      description = 'Plays Stockfish’s best move. Reply predictions are recorded for comparison; the personal model continues learning.';
    } else if (personal && decision.personal_changed) {
      label = 'Personal choice';
      description = `Your personal policy changed the choice from ${decision.prior_move || 'the Maia-only continuation'} to ${decision.move}.`;
    } else if (personal) {
      label = decision.engine_changed ? 'Model choice' : 'Engine choice';
      description = 'Your personal policy and the Maia prior selected the same continuation.';
    }
    if (state.mode !== 'baseline' && !decision.engine_changed) description += ' It also matches Stockfish’s choice.';
    put('decision-label', label);
    put('decision-description', description);
    const replies = Array.isArray(decision.replies) ? [...decision.replies].sort((a, b) => (b.personal_probability ?? b.prior_probability ?? 0) - (a.personal_probability ?? a.prior_probability ?? 0)) : [];
    const replyTable = rows => table(['Move', 'Maia prior', 'Personal', 'Loss'], rows.map(reply => ({
      values:[reply.move, probability(reply.prior_probability), probability(reply.personal_probability), cp(reply.loss_cp)],
      classes:['', '', 'personal-probability', '']
    })));
    byId('replies-table').replaceChildren(replies.length ? replyTable(replies.slice(0, 5)) : node('p', 'empty-table', state.game_over ? 'The game has ended. No replies remain.' : 'No reply estimates available.'));
    byId('remaining-replies').hidden = replies.length <= 5;
    put('remaining-reply-count', replies.length > 5 ? `${replies.length - 5} more` : '');
    byId('remaining-replies-table').replaceChildren(...(replies.length > 5 ? [replyTable(replies.slice(5))] : []));
    const candidates = Array.isArray(decision.candidates) ? decision.candidates : [];
    put('candidate-benchmark', decision.reply_baseline_move ? `Best after reply search: ${decision.reply_baseline_move}` : 'Reply-search values were not recorded for this saved decision.');
    if (!candidates.length) byId('candidates-table').replaceChildren(node('p', 'empty-table', 'No candidate estimates available.'));
    else {
      const candidateTable = table(['Move', 'Reply score', 'Cost', 'Expected loss'], candidates.map(candidate => ({
        values:[candidate.move, cp(candidate.reply_score_cp), cp(candidate.reply_cost_cp), cp(candidate.expected_regret_cp)], selected:candidate.selected,
        title:`Initial Stockfish score: ${cp(candidate.engine_score_cp)}. Initial search cost: ${cp(candidate.engine_cost_cp)}. Maia prior expected reply loss: ${cp(candidate.prior_expected_regret_cp)}.`
      })));
      const guards = {
        reply_search_found_losing_mate:'Excluded: reply search found a losing mate.',
        preserve_best_reply_search_mate:'Excluded: reply search found a better mate result.',
        reply_cost_exceeds_limit:`Excluded: more than ${cp(decision.cost_limit_cp ?? 65)} behind the best reply-search score.`
      };
      [...candidateTable.querySelector('tbody').children].forEach((row, index) => {
        const candidate = candidates[index];
        row.classList.toggle('excluded-row', candidate.eligible === false);
        row.children[0].append(node('span', 'candidate-status', candidate.eligible === false ? 'Excluded' : candidate.eligible === true ? candidate.selected ? 'Played' : 'Eligible' : 'No guard data'));
        if (candidate.eligible === false) {
          const reason = node('tr', 'candidate-guard-row');
          const cell = node('td', 'candidate-guard', guards[candidate.guard_reason] || 'Excluded by the reply-search guard.');
          cell.colSpan = 4;
          reason.append(cell);
          row.after(reason);
        }
      });
      byId('candidates-table').replaceChildren(candidateTable);
    }
  }

  function renderProfile() {
    const profile = state.profile || {};
    const samples = Number(profile.samples) || 0;
    const training = profile.training;
    const liveSamples = Number(profile.live_samples ?? (training ? 0 : samples)) || 0;
    const completed = Number(profile.completed) || 0;
    put('profile-state', training ? 'History fitted' : samples ? 'Personalized' : 'Maia prior');
    put('profile-description', training ? 'Your personal policy has learned from your Chess.com history. Each new live move continues its training.' : samples ? 'A personal adapter learns from your observed move choices. Predictions below are recorded before each move is added to its training history.' : 'No personal history yet. Predictions start with the pretrained Maia 1500 model.');
    put('profile-samples', count(samples));
    put('profile-storage', profile.username ? 'Your personal profile is saved locally and shared across browsers on this installation.' : 'One personal profile, linked to this browser and saved on this server.');
    put('profile-live-samples', count(liveSamples));
    put('profile-games', `${count(completed)} completed game${completed === 1 ? '' : 's'}`);
    put('profile-loss', cp(profile.mean_loss));
    put('prior-log-loss', number(profile.prior_log_loss, 3));
    put('personal-log-loss', number(profile.personal_log_loss, 3));
    put('prior-hits', liveSamples ? `${count(profile.prior_hits || 0)} / ${count(liveSamples)}` : '—');
    put('personal-hits', liveSamples ? `${count(profile.personal_hits || 0)} / ${count(liveSamples)}` : '—');
    byId('imported-history').hidden = !training;
    if (training) {
      put('imported-account', training.username || profile.username || 'Imported account');
      put('imported-fit', `Final fit · ${count(training.games_total)} games · ${count(training.total_positions)} move choices`);
      put('holdout-scope', `${count(training.test_games)} later games (${count(training.test_positions)} choices), held out from the first ${count(training.train_games)} games (${count(training.train_positions)} choices).`);
      put('history-prior-log-loss', number(training.prior_log_loss, 3));
      put('history-personal-log-loss', number(training.personal_log_loss, 3));
      put('history-prior-accuracy', probability(training.prior_accuracy));
      put('history-personal-accuracy', probability(training.personal_accuracy));
      const priorLoss = Number(training.prior_log_loss);
      const personalLoss = Number(training.personal_log_loss);
      const difference = personalLoss - priorLoss;
      put('holdout-result', training.prior_log_loss == null || training.personal_log_loss == null || !Number.isFinite(difference) ? 'Held-out scores are unavailable.' : number(priorLoss, 3) === number(personalLoss, 3) ? 'Held-out log loss matches at the displayed precision.' : `Personal log loss is ${difference < 0 ? 'lower' : 'higher'} on the held-out games.`);
    }
    const events = Array.isArray(state.events) ? state.events : [];
    put('observation-count', events.length ? 'This game' : '');
    const observations = byId('observations-table');
    if (!events.length) observations.replaceChildren(node('p', 'empty-table', 'Play a move to record the first observation.'));
    else {
      const wrapper = node('div', 'observations-scroll');
      wrapper.append(table(['Played', 'Maia prior', 'Personal', 'Loss'], events.map(event => ({
        values:[`${event.number}. ${event.player}`, probability(event.prior_probability), probability(event.personal_probability), cp(event.loss_cp)],
        classes:['', '', 'personal-probability', ''],
        title:`Log loss — Maia: ${number(event.prior_log_loss, 3)}; personal: ${number(event.personal_log_loss, 3)}. Stockfish alternative: ${event.alternative || '—'}.`
      }))));
      observations.replaceChildren(wrapper);
    }
    const archive = Array.isArray(state.archive) ? state.archive : [];
    put('archive-count', archive.length || '');
    const list = byId('archive-list');
    list.replaceChildren();
    if (!archive.length) list.append(node('p', 'empty-table', 'Finished and archived games appear here.'));
    archive.forEach(game => {
      const row = node('div', 'archive-row');
      row.append(node('strong', '', `Game ${game.game}`), node('span', '', `${game.mode === 'baseline' ? 'Baseline' : 'Adaptive'} · ${Math.ceil((game.moves?.length || 0) / 2)} moves`), node('span', '', game.result === '*' ? 'Unfinished' : game.result || '—'));
      list.append(row);
    });
  }

  function render() {
    if (!state?.fen) return;
    board.position(state.fen, false);
    renderStatus();
    renderMoves();
    renderDecision();
    renderProfile();
    highlightSquares();
  }

  function openNewGame() {
    if (!available() || busy) return;
    document.querySelector(`input[name=mode][value="${state.mode === 'baseline' ? 'baseline' : 'adaptive'}"]`).checked = true;
    byId('new-game-dialog').showModal();
  }

  function openConfirmation(title, copy, action, label) {
    put('confirm-title', title);
    put('confirm-copy', copy);
    put('confirm-action', label);
    confirmation = action;
    byId('confirm-dialog').showModal();
  }

  function switchTab(name, focus = false) {
    document.querySelectorAll('[data-tab]').forEach(button => {
      const active = button.dataset.tab === name;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', String(active));
      button.tabIndex = active ? 0 : -1;
      byId(`${button.dataset.tab}-panel`).hidden = !active;
      if (active && focus) button.focus();
    });
    if (name === 'chat') renderChat(true);
  }

  board = Chessboard('board', {
    position:'start', draggable:true, pieceTheme:'/static/img/chesspieces/wikipedia/{piece}.png',
    moveSpeed:140, snapbackSpeed:100,
    onDragStart:(_, piece) => canPlay() && piece.startsWith('w'),
    onDrop:(from, to) => {
      lastDrag = Date.now();
      if (from === to) selected = selected === from ? null : from;
      else {selected = null; if (to !== 'offboard') play(from, to);}
      highlightSquares();
      return 'snapback';
    },
    onSnapbackEnd:() => {if (state) board.position(state.fen, false); highlightSquares();}
  });
  byId('board').addEventListener('click', event => {
    if (!canPlay() || Date.now() - lastDrag < 180) return;
    const square = event.target.closest('[data-square]')?.dataset.square;
    if (!square) return;
    if (selected && legalMoves().some(move => move.startsWith(selected + square))) {play(selected, square); selected = null;}
    else selected = selected === square ? null : legalMoves().some(move => move.startsWith(square)) ? square : null;
    highlightSquares();
  });
  byId('move-form').addEventListener('submit', event => {event.preventDefault(); if (canPlay()) act('move', {move:byId('move-input').value.trim()});});
  byId('chat-form').addEventListener('submit', event => {event.preventDefault(); sendChat();});
  byId('chat-input').addEventListener('input', () => {
    if (pendingChat && pendingChat.message !== byId('chat-input').value.trim()) {
      pendingChat = null;
      if (chatRetryMode === 'send') chatRetryMode = 'none';
    }
    renderChatControls();
  });
  byId('chat-input').addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendChat();
    }
  });
  byId('chat-retry').addEventListener('click', () => chatRetryMode === 'send' ? sendChat(true) : loadChat());
  document.querySelectorAll('[data-chat-prompt]').forEach(button => button.addEventListener('click', () => {
    byId('chat-input').value = button.dataset.chatPrompt;
    pendingChat = null;
    if (chatRetryMode === 'send') chatRetryMode = 'none';
    renderChatControls();
    byId('chat-input').focus({preventScroll:true});
  }));
  byId('flip-board').addEventListener('click', () => {
    board.flip();
    const flipped = board.orientation() === 'black';
    const frame = document.querySelector('.board-frame');
    const opponent = document.querySelector('.opponent-row');
    const human = document.querySelector('.human-row');
    document.querySelector('.game-column').classList.toggle('flipped', flipped);
    frame.insertAdjacentElement('beforebegin', flipped ? human : opponent);
    frame.insertAdjacentElement('afterend', flipped ? opponent : human);
    selected = null;
    highlightSquares();
  });
  byId('new-game').addEventListener('click', openNewGame);
  byId('mode-label').addEventListener('click', openNewGame);
  byId('confirm-new-game').addEventListener('click', () => {byId('new-game-dialog').close(); act('new_game', {mode:document.querySelector('input[name=mode]:checked').value});});
  byId('resign').addEventListener('click', () => openConfirmation('Resign this game?', 'This game will be recorded as a loss. Your player model is kept.', 'resign', 'Resign'));
  byId('forget-profile').addEventListener('click', () => openConfirmation('Reset your player model?', 'This clears the active model, saved games and current game. Predictions start again from Maia. Downloaded Chess.com archives and training files remain on disk.', 'forget', 'Reset model'));
  byId('claim-draw').addEventListener('click', () => act('claim_draw'));
  byId('confirm-cancel').addEventListener('click', () => byId('confirm-dialog').close());
  byId('confirm-action').addEventListener('click', () => {byId('confirm-dialog').close(); act(confirmation);});
  document.querySelectorAll('[data-promotion]').forEach(button => button.addEventListener('click', () => {
    if (!pendingPromotion) return;
    const move = pendingPromotion + button.dataset.promotion;
    pendingPromotion = null;
    byId('promotion-dialog').close();
    act('move', {move});
  }));
  byId('promotion-dialog').addEventListener('close', () => {pendingPromotion = null; selected = null; if (state) board.position(state.fen, false); highlightSquares();});
  document.querySelectorAll('[data-tab]').forEach(button => {
    button.addEventListener('click', () => switchTab(button.dataset.tab));
    button.addEventListener('keydown', event => {
      if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
        event.preventDefault();
        const names = ['chat', 'analysis', 'player'];
        const index = names.indexOf(button.dataset.tab);
        switchTab(event.key === 'Home' ? names[0] : event.key === 'End' ? names[names.length - 1] : names[(index + (event.key === 'ArrowRight' ? 1 : names.length - 1)) % names.length], true);
      }
    });
  });
  byId('reconnect').addEventListener('click', reconnect);
  window.addEventListener('online', () => {if (!connected) reconnect(); if (!chat.ready && !chatLoading && !chatBusy) loadChat();});
  let resizeFrame;
  window.addEventListener('resize', () => {cancelAnimationFrame(resizeFrame); resizeFrame = requestAnimationFrame(() => {board.resize(); highlightSquares();});});
  setBusy(true);
  loadState().catch(error => {
    connected = false;
    notice(error.message || 'Cannot connect to the local server.', true);
    retryTimer = setTimeout(reconnect, 6000);
  }).finally(() => {setBusy(false); loadChat();});
})();
