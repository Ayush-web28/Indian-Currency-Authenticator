import React, { useState } from 'react';

const API_URL = import.meta.env.VITE_API_URL || '';

const LABELS = {
  REAL: 'GENUINE NOTE',
  FAKE: 'COUNTERFEIT',
  NOT_NOTE: 'NOT A NOTE',
};

export default function FeedbackPrompt({ predicted, confidence, quality }) {
  const [step, setStep] = useState('ask');
  const [error, setError] = useState(null);

  const send = async (actual) => {
    setStep('sending');
    setError(null);
    try {
      const res = await fetch(`${API_URL}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ predicted, actual, confidence, quality }),
      });
      if (!res.ok) throw new Error();
      setStep('done');
    } catch {
      setError('Could not send feedback. Please try again.');
      setStep('ask');
    }
  };

  if (step === 'done') {
    return <p className="feedback feedback-done">THANK YOU - FEEDBACK RECORDED</p>;
  }

  if (step === 'wrong') {
    const options = Object.keys(LABELS).filter((k) => k !== predicted);
    return (
      <div className="feedback">
        <p>WHAT IS IT ACTUALLY?</p>
        <div className="softkeys">
          {options.map((o) => (
            <button key={o} className="key small" onClick={() => send(o)}>
              &gt; {LABELS[o]}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="feedback">
      <p>WAS THIS RESULT CORRECT?</p>
      {error && <p className="feedback-error">{error}</p>}
      <div className="softkeys">
        <button className="key small" disabled={step === 'sending'} onClick={() => send(predicted)}>
          &gt; YES
        </button>
        <button className="key small" disabled={step === 'sending'} onClick={() => setStep('wrong')}>
          &gt; NO
        </button>
      </div>
    </div>
  );
}
