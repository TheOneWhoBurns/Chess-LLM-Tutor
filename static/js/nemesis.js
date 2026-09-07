/* NEMESIS client: the server is authoritative for moves, learning, and legal state. */
(() => {
  'use strict';
  const $id = id => document.getElementById(id);
  let state = null, board = null, busy = false, selected = null, pendingPromotion = null;
  let lastDrag = 0, confirmation = null;
  const pieceNames = {p:'pawn', n:'knight', b:'bishop', r:'rook', q:'queen', k:'king'};
  const headings = {
    arena: ['The arena', 'Your habits. ', 'My advantage.', 'Every move tells me something. Let’s find out what yours reveal.'],
    profile: ['Your profile', 'A rival that ', 'remembers you.', 'Your patterns, gathered one decision at a time.'],
    research: ['Field notes', 'The learning ', 'behind the game.', 'An observable experiment in personalized opposition.']
  };
  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function notice(message) {
    $id('notice').textContent = message || '';
    $id('notice').hidden = !message;
  }
  function setBusy(value) {
    busy = value;
    document.body.classList.toggle('busy', value);
    document.querySelectorAll('#move-form button, #move-input, #new-game, #resign, #claim-draw, #forget-profile').forEach(button => {button.disabled = value;});
    if (value) {
      $id('turn-status').textContent = 'NEMESIS is thinking…';
      $id('opponent-status').textContent = 'Reading your position';
    } else if (state) renderStatus();
  }
  async function loadState() {
    const response = await fetch('/api/state/', {cache:'no-store'});
    if (!response.ok) throw new Error('Could not load your saved game. Refresh to reconnect.');
    state = await response.json();
    render();
  }
  async function act(action, extra = {}) {
    if (busy || !state) return;
    notice('');
    selected = null;
    setBusy(true);
    try {
      const response = await fetch('/api/action/', {
        method: 'POST', headers: {'Content-Type':'application/json', 'X-CSRFToken':document.querySelector('[name=csrfmiddlewaretoken]').value},
        body:JSON.stringify({action, revision:state.revision, ...extra}),
        signal:AbortSignal.timeout(30000)
      });
      const result = await response.json();
      if (!response.ok) {
        if (result.state) {state = result.state; render();}
        throw new Error(result.error || 'The request failed. Please refresh and try again.');
      }
      state = result;
      render();
      if (action === 'move') $id('move-input').value = '';
    } catch (error) {
      notice(error.message || 'Connection interrupted. Reconnecting to your saved position…');
      try {await loadState();} catch (_) {notice('Connection lost. Refresh to reconnect to your saved game.');}
    } finally {setBusy(false); highlightSquares();}
  }
  function play(from, to) {
    if (!state || busy || state.game_over || pendingPromotion) return;
    const moves = state.legal_moves.filter(move => move.startsWith(from + to));
    if (!moves.length) return;
    if (moves.some(move => move.length === 5)) {
      pendingPromotion = from + to;
      $id('promotion-dialog').showModal();
    } else act('move', {move:moves[0]});
  }
  function highlightSquares() {
    document.querySelectorAll('#board [data-square]').forEach(square => {
      const name = square.dataset.square;
      square.classList.remove('last-square','selected-square','legal-square');
      if (state?.last_move && [state.last_move.slice(0,2),state.last_move.slice(2,4)].includes(name)) square.classList.add('last-square');
      if (selected === name) square.classList.add('selected-square');
      if (selected && state.legal_moves.some(move => move.startsWith(selected + name))) square.classList.add('legal-square');
      const img = square.querySelector('img');
      const code = img?.getAttribute('data-piece');
      square.setAttribute('aria-label', name + (code ? ` ${code[0] === 'w' ? 'white' : 'black'} ${pieceNames[code[1].toLowerCase()]}` : ' empty'));
    });
  }
  function renderStatus() {
    let status = 'Your move · You play white';
    if (state.game_over) status = state.result === '1-0' ? 'You won. NEMESIS remembers.' : state.result === '0-1' ? 'NEMESIS wins. A lesson for next time.' : 'A draw. There’s always another encounter.';
    else if (state.in_check) status = 'You are in check · Protect your king';
    $id('turn-status').textContent = status;
    $id('opponent-status').textContent = state.game_over ? 'Encounter complete' : state.mode === 'baseline' ? 'Baseline opponent · Learning in the background' : state.profile.samples < 8 ? 'Observing your play' : 'Adapting to your patterns';
    $id('move-input').disabled = state.game_over || busy;
    $id('move-form').querySelector('button').disabled = state.game_over || busy;
    $id('resign').disabled = state.game_over || busy;
    $id('claim-draw').disabled = state.game_over || busy;
  }
  function render() {
    board.position(state.fen, false);
    renderStatus();
    $id('game-number').textContent = String(state.profile.games).padStart(3, '0');
    $id('mode-label').textContent = state.mode.toUpperCase() + ' MODE';
    $id('engine-name').textContent = state.engine;
    $id('learning-phase').textContent = state.profile.phase.toUpperCase();
    const count = state.profile.samples;
    $id('sample-label').textContent = `${count} move${count === 1 ? '' : 's'} observed`;
    $id('sample-goal').textContent = count < 8 ? `${8-count} to begin adapting` : `${state.profile.influence}% model weight`;
    $id('sample-progress').style.width = Math.min(100, count / 40 * 100) + '%';
    $id('mind-title').textContent = state.mode === 'baseline' ? 'Observing. Holding back.' : count < 8 ? 'First, I get to know you.' : state.profile.target ? `A pattern: ${state.profile.target.toLowerCase()}.` : 'Your habits are taking shape.';
    $id('mind-copy').textContent = state.mode === 'baseline' ? 'I still learn from your decisions. This encounter uses ordinary search so you can compare.' : count < 8 ? 'Play your opening. I’m looking for patterns, not making assumptions.' : 'My player model now influences move selection. More games will help test these early impressions.';
    const list = $id('weakness-list');
    list.replaceChildren();
    state.profile.weaknesses.forEach(row => {
      const flagged = row.supported && row.errors > 0;
      const item = element('div','weakness-item' + (flagged ? ' flagged' : ''));
      const line = element('div');
      line.append(element('span','',row.name), element('span','weakness-value',row.supported ? `${row.errors}/${row.positions} errors` : row.positions ? `${row.positions}/5 observed` : 'UNOBSERVED'));
      const track = element('div','weakness-track');
      const fill = element('span');
      fill.style.width = row.supported ? row.rate+'%' : '0%';
      track.append(fill); item.append(line, track); list.append(item);
    });
    const last = state.events.at(-1);
    if (last) {
      const explanation = last.loss_cp >= 100 ? `${last.player} gave up an estimated ${(last.loss_cp/100).toFixed(1)} pawns against ${last.alternative}. That position joins my training memory.` : `${last.player} stayed close to the search’s best move. I learn from the positions you handle well, too.`;
      $id('latest-insight').textContent = explanation + (last.adapted ? ' My reply was changed by your learned profile.' : '');
      $id('insight-meta').textContent = `MOVE ${last.number} · ${last.quality.toUpperCase()} · SEARCH ESTIMATE`;
    } else {
      $id('latest-insight').textContent = 'The board is set. Make your first move and the learning starts.';
      $id('insight-meta').textContent = 'WAITING FOR YOUR MOVE';
    }
    $id('move-count').textContent = `${state.moves.length} PLIES`;
    const history = $id('move-history');
    history.replaceChildren();
    for (let index=0; index<state.moves.length; index+=2) {
      const pair = element('div','move-pair');
      pair.append(element('small','',`${index/2+1}.`),element('span','',state.moves[index]),element('span','',state.moves[index+1] || '…'));
      history.append(pair);
    }
    if (!state.moves.length) history.append(element('p','empty-state','A blank board of possibilities. Your story starts with the first move.'));
    history.scrollLeft = history.scrollWidth;
    $id('profile-samples').textContent = count;
    $id('profile-games').textContent = state.profile.completed;
    $id('profile-loss').textContent = state.profile.mean_loss ?? '—';
    renderTable(); renderArchive(); highlightSquares();
  }
  function renderTable() {
    const table = element('table','profile-table');
    const head = element('thead'); const tr = element('tr');
    ['CONTEXT','POSITIONS','ERRORS','MEAN LOSS','EVIDENCE'].forEach(label=>tr.append(element('th','',label)));
    head.append(tr); table.append(head);
    const body = element('tbody');
    state.profile.weaknesses.forEach(row=>{
      const line = element('tr');
      [row.name,row.positions,row.errors,row.positions ? `${row.mean_loss} cp` : '—',row.supported ? 'Observed' : 'Gathering'].forEach(value=>line.append(element('td','',value)));
      body.append(line);
    });
    table.append(body); $id('profile-table').replaceChildren(table);
  }
  function renderArchive() {
    const list = $id('archive-list'); list.replaceChildren();
    if (!state.archive.length) list.append(element('p','empty-state','Previous encounters will appear here when you start a new game.'));
    state.archive.forEach(game=>{
      const row = element('div','archive-row');
      row.append(element('strong','',`Encounter ${String(game.game).padStart(3,'0')}`),element('span','',`${game.mode} · ${game.moves.length} plies`),element('span','',game.result === '*' ? 'Unfinished' : game.result));
      list.append(row);
    });
  }
  function navigate(page) {
    if (!headings[page]) page = 'arena';
    document.querySelectorAll('.page').forEach(section=>{section.hidden = section.id !== `${page}-page`;});
    document.querySelectorAll('[data-page]').forEach(button=>{
      button.classList.toggle('active',button.dataset.page === page);
      if (button.dataset.page === page) button.setAttribute('aria-current','page'); else button.removeAttribute('aria-current');
    });
    const [label,title,emphasis,subtitle] = headings[page];
    $id('page-label').textContent = label;
    $id('page-title').replaceChildren(document.createTextNode(title),element('em','',emphasis));
    $id('page-subtitle').textContent = subtitle;
    if (page === 'arena') {board.resize(); highlightSquares();}
    history.replaceState(null,'','#'+page);
  }
  function confirmAction(title, copy, action) {
    $id('confirm-title').textContent = title; $id('confirm-copy').textContent = copy;
    confirmation = action; $id('confirm-dialog').showModal();
  }
  board = Chessboard('board', {
    position:'start', draggable:true, pieceTheme:'/static/img/chesspieces/wikipedia/{piece}.png',
    onDragStart:(_, piece) => !!state && !busy && !state.game_over && !pendingPromotion && piece.startsWith('w'),
    onDrop:(from,to) => {
      lastDrag = Date.now();
      if (from === to) selected = selected === from ? null : from;
      else {selected = null; if(to !== 'offboard') play(from,to);}
      highlightSquares();
      return 'snapback';
    },
    onSnapbackEnd:()=>{if(state) board.position(state.fen,false); highlightSquares();}
  });
  $id('board').addEventListener('click', event=>{
    if(!state || busy || state.game_over || pendingPromotion || Date.now()-lastDrag < 180) return;
    const square = event.target.closest('[data-square]')?.dataset.square;
    if(!square) return;
    if(selected && state.legal_moves.some(move=>move.startsWith(selected+square))) {play(selected,square); selected=null;}
    else selected = selected === square ? null : state.legal_moves.some(move=>move.startsWith(square)) ? square : null;
    highlightSquares();
  });
  $id('move-form').addEventListener('submit',event=>{event.preventDefault(); act('move',{move:$id('move-input').value.trim()});});
  $id('flip-board').addEventListener('click',()=>{board.flip(); highlightSquares();});
  $id('new-game').addEventListener('click',()=>{document.querySelector(`input[name=mode][value=${state.mode}]`).checked=true; $id('new-game-dialog').showModal();});
  $id('confirm-new-game').addEventListener('click',()=>{$id('new-game-dialog').close(); act('new_game',{mode:document.querySelector('input[name=mode]:checked').value});});
  $id('resign').addEventListener('click',()=>confirmAction('End this encounter?', 'NEMESIS wins this game. Your training memory will be kept.', 'resign'));
  $id('forget-profile').addEventListener('click',()=>confirmAction('A clean slate?', 'This deletes your saved learning, game history, and current encounter. It cannot be undone.', 'forget'));
  $id('claim-draw').addEventListener('click',()=>act('claim_draw'));
  $id('confirm-cancel').addEventListener('click',()=>$id('confirm-dialog').close());
  $id('confirm-action').addEventListener('click',()=>{$id('confirm-dialog').close(); act(confirmation);});
  document.querySelectorAll('[data-promotion]').forEach(button=>button.addEventListener('click',()=>{
    const move = pendingPromotion + button.dataset.promotion; pendingPromotion=null; $id('promotion-dialog').close(); act('move',{move});
  }));
  $id('promotion-dialog').addEventListener('close',()=>{pendingPromotion=null; if(state) board.position(state.fen,false);});
  document.querySelectorAll('[data-page]').forEach(button=>button.addEventListener('click',()=>navigate(button.dataset.page)));
  window.addEventListener('resize',()=>{if(!$id('arena-page').hidden) {board.resize(); highlightSquares();}});
  window.addEventListener('hashchange',()=>navigate(location.hash.slice(1)));
  navigate(location.hash.slice(1));
  setBusy(true);
  loadState().catch(error=>notice(error.message)).finally(()=>setBusy(false));
})();
