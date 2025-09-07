console.log('Chatbot script loaded');

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded');
    
    // Chat UI elements
    const chatButton = document.getElementById('chat-button');
    const chatContainer = document.getElementById('chat-container');
    const chatClose = document.getElementById('chat-close');
    const chatMessages = document.getElementById('chat-messages');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    let chatOpened = false;
    
    // Debug log all elements
    console.log('Chat button:', chatButton);
    console.log('Chat container:', chatContainer);
    console.log('Chat close button:', chatClose);
    console.log('Chat messages:', chatMessages);
    console.log('Chat form:', chatForm);
    console.log('User input:', userInput);
    console.log('Send button:', sendButton);
    
    // Toggle chat window and show welcome message on first open
    if (chatButton) {
        chatButton.addEventListener('click', function(e) {
            console.log('Chat button clicked');
            console.log('Current chat container display:', window.getComputedStyle(chatContainer).display);
            
            // Toggle display with animation
            if (chatContainer.classList.contains('d-none')) {
                chatContainer.classList.remove('d-none');
                chatContainer.style.display = 'flex';
                userInput.focus();
                if (!chatOpened) {
                    showWelcomeMessage();
                    chatOpened = true;
                }
            } else {
                chatContainer.style.animation = 'fadeOut 0.3s';
                setTimeout(() => {
                    chatContainer.classList.add('d-none');
                    chatContainer.style.animation = '';
                }, 300);
            }
            e.stopPropagation();
        });
    } else {
        console.error('Chat button not found!');
    }
    
    // Close chat window
    chatClose.addEventListener('click', function(e) {
        e.stopPropagation();
        chatContainer.classList.add('d-none');
    });
    
    // Send message
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const message = userInput.value.trim();
        if (message) {
            addMessage('user', message);
            sendMessageToServer(message);
            userInput.value = '';
        }
    });
    
    // Add a message to the chat
    function addMessage(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = text;
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Send message to the server
    async function sendMessageToServer(message) {
        // Show typing indicator
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'message bot-message';
        typingIndicator.id = 'typing-indicator';
        typingIndicator.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
        chatMessages.appendChild(typingIndicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });
            
            const data = await response.json();
            
            // Remove typing indicator
            const typingEl = document.getElementById('typing-indicator');
            if (typingEl) {
                typingEl.remove();
            }
            
            if (data.status === 'success') {
                addMessage('bot', data.response);
            } else {
                addMessage('bot', 'Sorry, I encountered an error. Please try again later.');
                console.error('Chat error:', data.error);
            }
            
        } catch (error) {
            console.error('Error:', error);
            const typingEl = document.getElementById('typing-indicator');
            if (typingEl) {
                typingEl.remove();
            }
            addMessage('bot', 'Sorry, I am having trouble connecting to the server. Please try again later.');
        }
    }
    
    // Allow sending message with Enter key (but allow Shift+Enter for new lines)
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });
    
    // Add welcome message
    function showWelcomeMessage() {
        const welcomeMessage = "Hello! I'm your academic assistant. How can I help you today?";
        setTimeout(() => {
            addMessage('bot', welcomeMessage);
        }, 500);
    }
    
    // Show welcome message
    function showWelcomeMessage() {
        const welcomeMessage = "Hello! I'm your academic assistant. How can I help you today?";
        addMessage('bot', welcomeMessage);
    }
});
