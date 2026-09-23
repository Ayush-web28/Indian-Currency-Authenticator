import React from 'react';

export default function QualityNote({ quality }) {
  if (!quality) return null;
  const cls = quality.rating.toLowerCase();
  return (
    <div className={`quality ${cls}`}>
      <div className="quality-head">
        <span>IMAGE QUALITY</span>
        <b>{quality.rating}</b>
      </div>
      {quality.retake && (
        <p className="quality-retake">RETAKE RECOMMENDED - result may be unreliable</p>
      )}
      {quality.tips.length > 0 && (
        <ul>
          {quality.tips.map((tip) => (
            <li key={tip}>{tip}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
