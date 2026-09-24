import React, { useEffect, useRef, useState } from 'react';
import BatchScreen from './BatchScreen.jsx';
import ChatScreen from './ChatScreen.jsx';
import QualityNote from './QualityNote.jsx';
import FeedbackPrompt from './FeedbackPrompt.jsx';
import Heatmap from './Heatmap.jsx';

const API_URL = import.meta.env.VITE_API_URL || '';

const SCAN_STEPS = [
  'READING NOTE IMAGE',
  'STAGE 1: CURRENCY CHECK',
  'STAGE 2: AUTHENTICITY CHECK',
  'COMPUTING GRADE',
];

function Stars({ count }) {
  return (
    <span className="stars" aria-label={`${count} of 5 stars`}>
      {[1, 2, 3, 4, 5].map((n) => (
        <span key={n} className={n <= count ? 'on' : 'off'}>★</span>
      ))}
    </span>
  );
}

function IdleScreen({ onFile }) {
  const inputRef = useRef(null);
  const [drag, setDrag] = useState(false);

  const pick = (file) => file && file.type.startsWith('image/') && onFile(file);

  return (
    <div
      className={`slot-area ${drag ? 'drag' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}
    >
      <p className="prompt">PLEASE INSERT NOTE</p>
      <div className="slot"><div className="slot-glow" /></div>
      <p className="hint">Drop a note image here, or use a button below</p>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => pick(e.target.files[0])}
      />
      <div className="softkeys">
        <button className="key" onClick={() => inputRef.current.click()}>
          &gt; UPLOAD IMAGE
        </button>
        <button
          className="key"
          onClick={() => {
            inputRef.current.setAttribute('capture', 'environment');
            inputRef.current.click();
            inputRef.current.removeAttribute('capture');
          }}
        >
          &gt; USE CAMERA
        </button>
      </div>
    </div>
  );
}

function ScanScreen({ preview, step }) {
  return (
    <div className="scan">
      <div className="note-frame">
        <img src={preview} alt="Inserted note" />
        <div className="scanline" />
      </div>
      <ul className="steps">
        {SCAN_STEPS.map((s, i) => (
          <li key={s} className={i < step ? 'done' : i === step ? 'active' : ''}>
            {i < step ? '[OK]' : i === step ? '[..]' : '[  ]'} {s}
          </li>
        ))}
      </ul>
    </div>
  );
}

function toScan(result) {
  if (!result) return null;
  const { stage1, stage2, quality } = result;
  if (!stage2) return { verdict: 'NOT_NOTE', confidence: stage1.confidence, quality: quality?.rating };
  return {
    verdict: stage2.classification,
    confidence: stage2.confidence,
    grade: stage2.grade,
    risk: stage2.risk,
    decision_strength: stage2.decision_strength,
    quality: quality?.rating,
  };
}

function ResultScreen({ result, preview, onReset, onAsk }) {
  const [showWhy, setShowWhy] = useState(false);
  const { stage1, stage2, explanation } = result;

  if (!stage2) {
    return (
      <div className="result neutral">
        <img className="thumb" src={preview} alt="Inserted note" />
        <h2>NOT A CURRENCY NOTE</h2>
        <p className="sub">Stage 1 confidence: {stage1.confidence}% not currency</p>
        <p className="hint">Please insert a clear image of an Indian banknote.</p>
        <QualityNote quality={result.quality} />
        <FeedbackPrompt predicted="NOT_NOTE" confidence={stage1.confidence} quality={result.quality?.rating} />
        <div className="softkeys">
          <button className="key" onClick={onReset}>&gt; TRY ANOTHER NOTE</button>
          <button className="key" onClick={onAsk}>&gt; ASK ABOUT THIS</button>
        </div>
      </div>
    );
  }

  const isReal = stage2.classification === 'REAL';
  return (
    <div className={`result ${isReal ? 'real' : 'fake'}`}>
      {showWhy && explanation ? (
        <Heatmap src={preview} grid={explanation.grid} target={explanation.target} />
      ) : (
        <img className="thumb" src={preview} alt="Inserted note" />
      )}
      {showWhy && explanation && (
        <div className="why">
          <p>
            {explanation.target === 'REAL' ? 'GREEN' : 'RED'} AREAS PUSHED THE MODEL TOWARD{' '}
            {explanation.target === 'REAL' ? 'GENUINE' : 'COUNTERFEIT'}
          </p>
          <p className="why-note">Shows where the model looked, not why. It cannot name specific security features.</p>
          {explanation.edge_share >= 0.65 && (
            <p className="why-warn">
              Most of the attention is on the edges of the photo, so the background may be influencing this result.
            </p>
          )}
        </div>
      )}
      <h2>{isReal ? 'GENUINE NOTE' : 'COUNTERFEIT SUSPECTED'}</h2>
      <p className="sub">Confidence {stage2.confidence}%</p>

      <div className="meter" role="img" aria-label={`${stage2.percent_real}% real`}>
        <div className="meter-real" style={{ width: `${stage2.percent_real}%` }} />
        <div className="meter-fake" style={{ width: `${stage2.percent_fake}%` }} />
      </div>
      <div className="meter-legend">
        <span>REAL {stage2.percent_real}%</span>
        <span>FAKE {stage2.percent_fake}%</span>
      </div>

      <div className="receipt">
        <div><span>AUTHENTICITY GRADE</span><b className="grade">{stage2.grade}</b></div>
        <div><span>RISK LEVEL</span><b className={`risk ${stage2.risk.toLowerCase()}`}>{stage2.risk}</b></div>
        <div><span>RATING</span><Stars count={stage2.stars} /></div>
        <div><span>DECISION STRENGTH</span><b className={`strength ${stage2.decision_strength.toLowerCase()}`}>{stage2.decision_strength}</b></div>
        <div><span>CURRENCY CHECK</span><b>{stage1.confidence}%</b></div>
      </div>

      <QualityNote quality={result.quality} />
      <FeedbackPrompt predicted={stage2.classification} confidence={stage2.confidence} quality={result.quality?.rating} />

      <div className="softkeys">
        <button className="key" onClick={onReset}>&gt; CHECK ANOTHER NOTE</button>
        {explanation && (
          <button className="key" onClick={() => setShowWhy((v) => !v)}>
            &gt; {showWhy ? 'HIDE WHY' : 'SHOW WHY'}
          </button>
        )}
        <button className="key" onClick={onAsk}>&gt; ASK ABOUT THIS RESULT</button>
      </div>
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState('single');
  const [phase, setPhase] = useState('idle');
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [step, setStep] = useState(0);

  useEffect(() => {
    if (phase !== 'scanning') return undefined;
    setStep(0);
    const id = setInterval(() => setStep((s) => Math.min(s + 1, SCAN_STEPS.length - 1)), 700);
    return () => clearInterval(id);
  }, [phase]);

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  const reset = () => {
    setPhase('idle');
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const scan = async (file) => {
    setPreview(URL.createObjectURL(file));
    setError(null);
    setPhase('scanning');
    const minDelay = new Promise((r) => setTimeout(r, SCAN_STEPS.length * 700));
    try {
      const body = new FormData();
      body.append('file', file);
      const [res] = await Promise.all([fetch(`${API_URL}/api/detect`, { method: 'POST', body }), minDelay]);
      if (!res.ok) throw new Error((await res.json()).detail || `Server error ${res.status}`);
      setResult(await res.json());
      setPhase('result');
    } catch (e) {
      setError(e.message === 'Failed to fetch' ? 'Cannot reach the detection server.' : e.message);
      setPhase('error');
    }
  };

  return (
    <main className="atm">
      <div className="bezel">
        <header className="topbar">
          <span className="brand">INDIAN CURRENCY AUTHENTICATOR</span>
          <span className="led" />
        </header>
        <nav className="modes" aria-label="Mode">
          <button className={mode === 'single' ? 'active' : ''} onClick={() => setMode('single')}>SINGLE NOTE</button>
          <button className={mode === 'batch' ? 'active' : ''} onClick={() => setMode('batch')}>BATCH</button>
          <button className={mode === 'help' ? 'active' : ''} onClick={() => setMode('help')}>HELP</button>
        </nav>
        <section className="display">
          {mode === 'batch' && <BatchScreen />}
          {mode === 'help' && <ChatScreen scan={toScan(result)} />}
          {mode === 'single' && phase === 'idle' && <IdleScreen onFile={scan} />}
          {mode === 'single' && phase === 'scanning' && <ScanScreen preview={preview} step={step} />}
          {mode === 'single' && phase === 'result' && <ResultScreen result={result} preview={preview} onReset={reset} onAsk={() => setMode('help')} />}
          {mode === 'single' && phase === 'error' && (
            <div className="result fake">
              <h2>SERVICE UNAVAILABLE</h2>
              <p className="sub">{error}</p>
              <div className="softkeys">
                <button className="key" onClick={reset}>&gt; RETURN</button>
              </div>
            </div>
          )}
        </section>
        <footer className="tray">
          <div className="tray-slot" />
          <span>NOTE RETURN</span>
        </footer>
      </div>
    </main>
  );
}
