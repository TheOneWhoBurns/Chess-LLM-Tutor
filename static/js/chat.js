document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const chatMessages = document.getElementById('chat-messages');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const historyContent = document.querySelector('.history-content');

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