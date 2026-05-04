(function () {
  if (window.RadgnarackAssistWidget) return;

  var currentScript = document.currentScript || (function () {
    var scripts = document.getElementsByTagName('script');
    return scripts[scripts.length - 1];
  })();

  var scriptUrl = new URL(currentScript.src, window.location.href);
  var apiBase = currentScript.getAttribute('data-api-base') || scriptUrl.origin;
  var cssUrl = new URL('./widget.css', scriptUrl.href).toString();
  var widgetWidth = currentScript.getAttribute('data-width') || '360px';
  var widgetBottom = currentScript.getAttribute('data-bottom') || '24px';
  var widgetRight = currentScript.getAttribute('data-right') || '24px';

  function makeConversationId() {
    if (window.crypto && window.crypto.randomUUID) {
      return window.crypto.randomUUID();
    }
    return 'ra-' + Date.now() + '-' + Math.random().toString(16).slice(2);
  }

  var conversationId = makeConversationId();

  function injectStyles(shadowRoot) {
    var link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = cssUrl;
    shadowRoot.appendChild(link);
  }

  function boot() {
    var host = document.createElement('div');
    document.body.appendChild(host);

    var shadowRoot = host.attachShadow({ mode: 'open' });
    injectStyles(shadowRoot);

    var root = document.createElement('div');
    root.className = 'ra-widget-root';
    root.style.setProperty('--ra-widget-width', widgetWidth);
    root.style.setProperty('--ra-widget-bottom', widgetBottom);
    root.style.setProperty('--ra-widget-right', widgetRight);
    root.innerHTML = [
      '<div class="ra-widget-panel" aria-hidden="true">',
      '  <div class="ra-widget-header">Radgnarack Assist</div>',
      '  <div class="ra-widget-messages"></div>',
      '  <form class="ra-widget-input-row">',
      '    <input class="ra-widget-input" type="text" maxlength="2000" placeholder="Ask a question..." />',
      '    <button class="ra-widget-send" type="submit">Send</button>',
      '  </form>',
      '</div>',
      '<button class="ra-widget-toggle" type="button" aria-label="Open chat">💬</button>'
    ].join('');
    shadowRoot.appendChild(root);

    var panel = root.querySelector('.ra-widget-panel');
    var toggle = root.querySelector('.ra-widget-toggle');
    var form = root.querySelector('.ra-widget-input-row');
    var input = root.querySelector('.ra-widget-input');
    var sendButton = root.querySelector('.ra-widget-send');
    var messages = root.querySelector('.ra-widget-messages');
    var open = false;
    var sending = false;

    function setOpen(nextOpen) {
      open = nextOpen;
      panel.classList.toggle('ra-widget-open', open);
      panel.setAttribute('aria-hidden', String(!open));
      toggle.setAttribute('aria-label', open ? 'Close chat' : 'Open chat');
      toggle.textContent = open ? '×' : '💬';
      if (open) {
        input.focus();
      }
    }

    function setSending(nextSending) {
      sending = nextSending;
      input.disabled = nextSending;
      sendButton.disabled = nextSending;
    }

    function scrollMessages() {
      messages.scrollTop = messages.scrollHeight;
    }

    function addMessage(text, type) {
      var item = document.createElement('div');
      item.className = 'ra-widget-message ra-widget-message-' + type;
      item.textContent = text;
      messages.appendChild(item);
      scrollMessages();
      return item;
    }

    async function sendMessage(question) {
      addMessage(question, 'user');
      input.value = '';
      setSending(true);
      var pending = addMessage('Thinking...', 'system');

      try {
        var response = await fetch(apiBase.replace(/\/$/, '') + '/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            question: question,
            conversation_id: conversationId
          })
        });

        var data = await response.json().catch(function () {
          return {};
        });

        pending.remove();

        if (!response.ok) {
          throw new Error(data.detail || 'Request failed.');
        }

        addMessage(data.answer || 'No response received.', 'assistant');
      } catch (error) {
        pending.remove();
        addMessage(error && error.message ? error.message : 'Unable to reach the assistant.', 'system');
      } finally {
        setSending(false);
        input.focus();
      }
    }

    toggle.addEventListener('click', function () {
      setOpen(!open);
    });

    form.addEventListener('submit', function (event) {
      event.preventDefault();
      if (sending) return;

      var question = input.value.trim();
      if (!question) return;

      sendMessage(question);
    });

    addMessage('Hi — how can I help?', 'assistant');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

  window.RadgnarackAssistWidget = {
    getConversationId: function () {
      return conversationId;
    }
  };
})();
