import React, { useEffect, useRef } from 'react';

const OUTPUT_SIZE = 112;
const COLORS = { FAKE: [255, 80, 70], REAL: [109, 255, 176] };

export default function Heatmap({ src, grid, target }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const size = grid.length;
    const tiny = document.createElement('canvas');
    tiny.width = size;
    tiny.height = size;
    const tinyCtx = tiny.getContext('2d');
    const pixels = tinyCtx.createImageData(size, size);
    const [r, g, b] = COLORS[target] || COLORS.FAKE;

    grid.forEach((row, y) => {
      row.forEach((value, x) => {
        const i = (y * size + x) * 4;
        pixels.data[i] = r;
        pixels.data[i + 1] = g;
        pixels.data[i + 2] = b;
        pixels.data[i + 3] = Math.round(value * 200);
      });
    });
    tinyCtx.putImageData(pixels, 0, 0);

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(tiny, 0, 0, canvas.width, canvas.height);
  }, [grid, target]);

  return (
    <div className="heatmap">
      <img src={src} alt="Scanned note" />
      <canvas ref={canvasRef} width={OUTPUT_SIZE} height={OUTPUT_SIZE} aria-hidden="true" />
    </div>
  );
}
