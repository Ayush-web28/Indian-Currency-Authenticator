import React, { useRef, useState } from 'react';

const API_URL = import.meta.env.VITE_API_URL || '';
const MAX_FILES = 20;

function verdict(row) {
  if (row.error) return { label: 'ERROR', cls: 'neutral', detail: row.error };
  const retake = row.quality?.retake ? ' · RETAKE' : '';
  if (!row.stage2) return { label: 'NOT NOTE', cls: 'neutral', detail: `${row.stage1.confidence}%${retake}` };
  const isReal = row.stage2.classification === 'REAL';
  return {
    label: isReal ? 'REAL' : 'FAKE',
    cls: isReal ? 'real' : 'fake',
    detail: `${row.stage2.confidence}% · ${row.stage2.grade}${retake}`,
  };
}

export default function BatchScreen() {
  const inputRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [phase, setPhase] = useState('select');
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [drag, setDrag] = useState(false);

  const add = (list) => {
    const images = [...list].filter((f) => f.type.startsWith('image/'));
    setFiles((prev) => [...prev, ...images].slice(0, MAX_FILES));
  };

  const reset = () => {
    setFiles([]);
    setData(null);
    setError(null);
    setPhase('select');
  };

  const run = async () => {
    setPhase('scanning');
    setError(null);
    try {
      const body = new FormData();
      files.forEach((f) => body.append('files', f));
      const res = await fetch(`${API_URL}/api/batch`, { method: 'POST', body });
      if (!res.ok) throw new Error((await res.json()).detail || `Server error ${res.status}`);
      setData(await res.json());
      setPhase('results');
    } catch (e) {
      setError(e.message === 'Failed to fetch' ? 'Cannot reach the detection server.' : e.message);
      setPhase('select');
    }
  };

  if (phase === 'scanning') {
    return (
      <div className="batch batch-busy">
        <p className="prompt">SCANNING {files.length} NOTES</p>
        <div className="slot"><div className="slot-glow" /></div>
        <p className="hint">This can take a few seconds per note</p>
      </div>
    );
  }

  if (phase === 'results') {
    const { summary, results } = data;
    return (
      <div className="batch">
        <h2 className="batch-title">BATCH REPORT</h2>
        <div className="batch-summary">
          <span className="real">REAL {summary.real}</span>
          <span className="fake">FAKE {summary.fake}</span>
          <span className="neutral">OTHER {summary.not_currency + summary.errors}</span>
        </div>
        <ul className="batch-rows">
          {results.map((row, i) => {
            const v = verdict(row);
            return (
              <li key={`${row.filename}-${i}`} className={v.cls}>
                <span className="name">{row.filename}</span>
                <span className="tag">{v.label}</span>
                <span className="detail">{v.detail}</span>
              </li>
            );
          })}
        </ul>
        <div className="softkeys">
          <button className="key" onClick={reset}>&gt; NEW BATCH</button>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`batch ${drag ? 'drag' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); add(e.dataTransfer.files); }}
    >
      <p className="prompt">INSERT MULTIPLE NOTES</p>
      <p className="hint">Up to {MAX_FILES} images per batch</p>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={(e) => { add(e.target.files); e.target.value = ''; }}
      />
      {error && <p className="batch-error">{error}</p>}
      {files.length > 0 && (
        <ul className="batch-files">
          {files.map((f, i) => (
            <li key={`${f.name}-${i}`}>
              <span className="name">{f.name}</span>
              <button
                className="remove"
                aria-label={`Remove ${f.name}`}
                onClick={() => setFiles((prev) => prev.filter((_, j) => j !== i))}
              >
                x
              </button>
            </li>
          ))}
        </ul>
      )}
      <div className="softkeys">
        <button className="key" onClick={() => inputRef.current.click()}>&gt; ADD IMAGES</button>
        {files.length > 0 && (
          <button className="key" onClick={run}>&gt; SCAN {files.length} NOTE{files.length > 1 ? 'S' : ''}</button>
        )}
      </div>
    </div>
  );
}
