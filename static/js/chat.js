document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const chatMessages = document.getElementById('chat-messages');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const historyContent = document.querySelector('.history-content');

    // Stats elements
    const playerEloEl = document.getElementById('player-elo');
    const maiaLevelEl = document.getElementById('maia-level');
    const playerRecordEl = document.getElementById('player-record');

    // Track previous elo for animation
    let previousElo = 1200;

    function scrollToBottom(element) {
        if (element) {
            element.scrollTop = element.scrollHeight;
        }
    }

    const chatObserver = new MutationObserver(() => {
        scrollToBottom(chatMessages);
    });

    chatObserver.observe(chatMessages, {
        childList: true,
        subtree: true
    });

    function addMessage(sender, text) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message';
        // Use textContent for sender to prevent XSS, and sanitize text
        const senderSpan = document.createElement('strong');
        senderSpan.textContent = sender + ': ';
        messageElement.appendChild(senderSpan);
        messageElement.appendChild(document.createTextNode(text));
        chatMessages.appendChild(messageElement);
    }

    function updatePlayerStats(stats) {
        if (!stats) return;

        // Update Elo with animation
        if (stats.estimated_elo !== undefined && playerEloEl) {
            const newElo = stats.estimated_elo;
            playerEloEl.textContent = newElo;

            // Add animation class based on elo change
            if (newElo > previousElo) {
                playerEloEl.classList.remove('elo-down');
                playerEloEl.classList.add('elo-up');
                setTimeout(() => playerEloEl.classList.remove('elo-up'), 500);
            } else if (newElo < previousElo) {
                playerEloEl.classList.remove('elo-up');
                playerEloEl.classList.add('elo-down');
                setTimeout(() => playerEloEl.classList.remove('elo-down'), 500);
            }
            previousElo = newElo;
        }

        // Update Maia level
        if (stats.current_maia_level !== undefined && maiaLevelEl) {
            maiaLevelEl.textContent = stats.current_maia_level || '—';
        }

        // Update record (W-L-D)
        if (playerRecordEl) {
            const wins = stats.wins || 0;
            const losses = stats.losses || 0;
            const draws = stats.draws || 0;
            playerRecordEl.textContent = `${wins}-${losses}-${draws}`;
        }
    }

    async function sendMessage(message) {
        try {
            addMessage('You', message);
            userInput.value = '';
            userInput.focus();

            const response = await fetch('/send_message/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrftoken'),
                },
                body: JSON.stringify({ message }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // Update player stats if present
            if (data.player_stats) {
                updatePlayerStats(data.player_stats);
            }

            if (data.status === 'success') {
                addMessage('Tutor', data.response);
                const serverEvent = new CustomEvent('serverResponse', { detail: data });
                window.dispatchEvent(serverEvent);
            } else if (data.status === 'ignore') {
                // Silent ignore for same-square moves, etc.
                return;
            } else if (data.status === 'error') {
                addMessage('Tutor', data.response || 'Something went wrong.');
                const serverEvent = new CustomEvent('serverResponse', { detail: data });
                window.dispatchEvent(serverEvent);
            } else {
                throw new Error('Server returned unexpected status');
            }

        } catch (error) {
            addMessage('System', 'An error occurred. Please try again.');
        }
    }

    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const message = userInput.value.trim();
        if (!message) return;
        sendMessage(message);
    });

    // Listen for chess moves with simplified data
    window.addEventListener('chessMoveToChat', (e) => {
        sendMessage(e.detail); // Now receiving just the move string
    });

    window.addEventListener('chessReset', () => {
        chatMessages.innerHTML = '';
        sendMessage('new game');
    });

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    scrollToBottom(chatMessages);
    scrollToBottom(historyContent);
});
