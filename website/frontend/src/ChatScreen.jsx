import React, { useEffect, useRef, useState } from 'react';

const API_URL = import.meta.env.VITE_API_URL || '';
const MAX_CHARS = 1000;

const SUGGESTIONS = [
  'How can I check a 500 rupee note by hand?',
  'What does the authenticity grade mean?',
  'I think I received a fake note. What should I do?',
];

const VERDICT_LABEL = { REAL: 'GENUINE', FAKE: 'COUNTERFEIT SUSPECTED', NOT_NOTE: 'NOT A NOTE' };

const STATUS_ATTEMPTS = 3;
const STATUS_RETRY_DELAY_MS = 2500;

async function fetchChatEnabled() {
  for (let attempt = 0; attempt < STATUS_ATTEMPTS; attempt += 1) {
    try {
      const res = await fetch(`${API_URL}/api/chat/status`);
      if (res.ok) return Boolean((await res.json()).enabled);
    } catch {
      // the free server may still be waking up; try again
    }
    if (attempt < STATUS_ATTEMPTS - 1) {
      await new Promise((resolve) => setTimeout(resolve, STATUS_RETRY_DELAY_MS));
    }
  }
  return null;
}

export default function ChatScreen({ scan }) {
  const [status, setStatus] = useState('loading');
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const endRef = useRef(null);

  const checkStatus = async () => {
    setStatus('loading');
    const enabled = await fetchChatEnabled();
    setStatus(enabled === null ? 'unreachable' : enabled ? 'on' : 'off');
  };

  useEffect(() => {
    checkStatus();
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' });
  }, [messages, pending, error]);

  const request = async (history) => {
    setPending(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: history, ...(scan ? { scan } : {}) }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Something went wrong.');
      setMessages([...history, { role: 'assistant', content: data.reply }]);
    } catch (e) {
      setError(e.message === 'Failed to fetch' ? 'Cannot reach the server.' : e.message);
    }
    setPending(false);
  };

  const send = (text) => {
    const content = text.trim();
    if (!content || pending) return;
    const history = [...messages, { role: 'user', content }];
    setMessages(history);
    setDraft('');
    request(history);
  };

  if (status === 'loading') {
    return <p className="hint">CONNECTING TO HELP DESK...</p>;
  }

  if (status === 'unreachable') {
    return (
      <div className="chat chat-offline">
        <p className="prompt">CANNOT REACH HELP DESK</p>
        <p className="hint">The server may be waking up. Please try again in a moment.</p>
        <div className="softkeys">
          <button className="key" onClick={checkStatus}>&gt; TRY AGAIN</button>
        </div>
      </div>
    );
  }

  if (status === 'off') {
    return (
      <div className="chat chat-offline">
        <p className="prompt">HELP DESK OFFLINE</p>
        <p className="hint">The assistant is not available right now. Scanning still works.</p>
      </div>
    );
  }

  const suggestions = scan ? ['Explain my scan result', ...SUGGESTIONS.slice(0, 2)] : SUGGESTIONS;

  return (
    <div className="chat">
      <div className="chat-log" role="log" aria-live="polite">
        {messages.length === 0 && (
          <div className="chat-empty">
            <p className="prompt">HELP DESK</p>
            <p className="hint">Ask about banknote security features or your scan result.</p>
            {scan && (
              <p className="hint">
                LAST SCAN: {VERDICT_LABEL[scan.verdict]} ({scan.confidence}%)
              </p>
            )}
            <div className="softkeys">
              {suggestions.map((s) => (
                <button key={s} className="key small" onClick={() => send(s)}>
                  &gt; {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role}`}>
            <span className="who">{m.role === 'user' ? 'YOU' : 'HELP'}</span>
            <span className="text">{m.content}</span>
          </div>
        ))}
        {pending && <div className="chat-msg assistant"><span className="who">HELP</span><span className="text">...</span></div>}
        {error && (
          <div className="chat-error">
            <span>{error}</span>
            {messages.length > 0 && (
              <button className="key small" onClick={() => request(messages)}>&gt; RETRY</button>
            )}
          </div>
        )}
        <div ref={endRef} />
      </div>
      <form
        className="chat-form"
        onSubmit={(e) => { e.preventDefault(); send(draft); }}
      >
        <input
          value={draft}
          maxLength={MAX_CHARS}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Type your question"
          aria-label="Your question"
        />
        <button className="key small" type="submit" disabled={pending || !draft.trim()}>&gt; SEND</button>
      </form>
    </div>
  );
}
