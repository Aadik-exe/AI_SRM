import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  Satellite, UploadCloud, Eye, Activity, Navigation,
  Download, FileText, CheckCircle, ZoomIn, ZoomOut, RotateCcw,
  AlertCircle, Image, Cpu, RefreshCw, Info, MapPin, BarChart2
} from 'lucide-react';
import Globe from '@/components/ui/globe';
const API = 'https://5f4b2688a6eeba.lhr.life';

// ============================================================================
// API HELPERS
// ============================================================================
async function runDemo() {
  const r = await fetch(`${API}/api/process-demo`, { method: 'POST' });
  if (!r.ok) throw new Error(`Server error: ${r.status}`);
  return r.json();
}

const MAX_UPLOAD_PX = 512;

async function resizeToBlob(file) {
  if (file.name.match(/\.tiff?$/i)) return file;
  return new Promise((resolve) => {
    const img = new window.Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      const { naturalWidth: w, naturalHeight: h } = img;
      const scale = Math.min(1, MAX_UPLOAD_PX / Math.max(w, h));
      const canvas = document.createElement('canvas');
      canvas.width = Math.round(w * scale);
      canvas.height = Math.round(h * scale);
      canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => resolve(new File([blob], file.name, { type: 'image/png' })), 'image/png');
    };
    img.onerror = () => { URL.revokeObjectURL(url); resolve(file); };
    img.src = url;
  });
}

async function uploadFile(file, onProgress) {
  const resized = await resizeToBlob(file);
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    fd.append('file', resized);
    const xhr = new XMLHttpRequest();
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(Math.round(e.loaded / e.total * 30));
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
      else reject(new Error(`Upload failed: ${xhr.status} — ${xhr.responseText}`));
    };
    xhr.onerror = () => reject(new Error('Network error'));
    xhr.open('POST', `${API}/api/upload`);
    xhr.send(fd);
  });
}

// ============================================================================
// HEADER
// ============================================================================
const Header = ({ onReset }) => (
  <header className="header-float sticky top-0 z-50">
    <div className="max-w-6xl mx-auto px-5 h-14 flex items-center justify-between">
      <div className="flex items-center gap-3">
        {/* Logo mark */}
        <div style={{
          width: 36, height: 36,
          background: 'var(--accent)',
          borderRadius: 10,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 3px 0 #1a4033, 0 4px 12px rgba(45,106,79,0.25)'
        }}>
          <Satellite style={{ width: 18, height: 18, color: '#fff' }} />
        </div>
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--ink)', letterSpacing: '-0.01em', lineHeight: 1.2 }}>
            Satellite AI
          </div>
          <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--warm-dark)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            Super Resolution Mapping
          </div>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '4px 12px', borderRadius: 999,
          background: 'var(--accent-pale)',
          border: '1px solid var(--accent-light)',
          fontSize: 11, fontWeight: 600, color: 'var(--accent)',
          letterSpacing: '0.04em'
        }}>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent-2)', display: 'inline-block', boxShadow: '0 0 0 2px var(--accent-light)' }} className="animate-pulse" />
          EDSR-LITE ONLINE
        </div>
        {onReset && (
          <button onClick={onReset} className="btn-secondary" style={{ padding: '5px 12px', fontSize: 12 }}>
            <RefreshCw style={{ width: 12, height: 12 }} /> New Image
          </button>
        )}
      </div>
    </div>
  </header>
);

// ============================================================================
// UPLOAD PANEL — 3D hero section
// ============================================================================
const UploadPanel = ({ onDemo, onFile }) => {
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef();

  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  }, [onFile]);

  return (
    <div style={{ maxWidth: 680, margin: '0 auto' }}>
      {/* Hero label */}
      <div style={{ textAlign: 'center', marginBottom: 36 }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '4px 14px', borderRadius: 999, background: 'var(--blue-pale)', border: '1px solid var(--blue-light)', fontSize: 11, fontWeight: 600, color: 'var(--blue)', letterSpacing: '0.06em', marginBottom: 16 }}>
          <Cpu style={{ width: 11, height: 11 }} /> SIH 2024 · Problem Statement 26142 · NTRO
        </div>
        <h1 style={{ fontSize: 36, fontWeight: 700, color: 'var(--ink)', letterSpacing: '-0.03em', lineHeight: 1.15, marginBottom: 12 }}>
          Deep Learning<br />
          <span style={{ color: 'var(--accent)' }}>Super-Resolution</span> Mapping
        </h1>
        <p style={{ fontSize: 15, color: 'var(--warm-dark)', maxWidth: 440, margin: '0 auto', lineHeight: 1.65 }}>
          Upload any satellite image or GeoTIFF. Our EDSR-Lite AI pipeline upscales it 3× and extracts geospatial intelligence automatically.
        </p>
      </div>

      {/* Drop zone with 3D styling */}
      <div
        className={`drop-zone${dragOver ? ' active' : ''}`}
        style={{ padding: '48px 32px', textAlign: 'center', marginBottom: 16 }}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileRef.current?.click()}
      >
        <input ref={fileRef} type="file" accept=".tif,.tiff,.png,.jpg,.jpeg" style={{ display: 'none' }} onChange={(e) => { const f = e.target.files[0]; if (f) onFile(f); }} />

        {/* Floating icon with perspective rings */}
        <div style={{ position: 'relative', width: 72, height: 72, margin: '0 auto 20px', perspective: 300 }}>
          <div className="animate-float" style={{
            width: 72, height: 72, borderRadius: 20,
            background: dragOver ? 'var(--accent-pale)' : '#fff',
            border: '1.5px solid var(--stone)',
            boxShadow: dragOver ? '0 8px 0 var(--accent-light), var(--shadow-lg)' : '0 6px 0 var(--stone), var(--shadow-md)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'all 0.2s'
          }}>
            <UploadCloud style={{ width: 30, height: 30, color: dragOver ? 'var(--accent)' : 'var(--warm-mid)' }} />
          </div>
        </div>

        <p style={{ fontSize: 16, fontWeight: 600, color: 'var(--ink)', marginBottom: 6 }}>
          {dragOver ? 'Drop to process' : 'Drop your satellite image here'}
        </p>
        <p style={{ fontSize: 13, color: 'var(--warm-dark)', marginBottom: 16 }}>
          Click to browse · GeoTIFF, TIFF, PNG, JPG supported
        </p>
        <div style={{ display: 'flex', gap: 6, justifyContent: 'center', flexWrap: 'wrap' }}>
          {['.tif (GeoTIFF)', '.tiff', '.png', '.jpg'].map(t => (
            <span key={t} className="badge badge-stone">{t}</span>
          ))}
        </div>
      </div>

      {/* OR divider */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, margin: '18px 0' }}>
        <div className="divider" style={{ flex: 1 }} />
        <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--warm-mid)', letterSpacing: '0.06em' }}>OR</span>
        <div className="divider" style={{ flex: 1 }} />
      </div>

      {/* Demo button */}
      <button onClick={onDemo} className="btn-primary" style={{ width: '100%', padding: '13px 24px', fontSize: 14.5, borderRadius: 12, justifyContent: 'center' }}>
        <Image style={{ width: 16, height: 16 }} />
        Run with Built-in Sentinel-2 Demo Scene
      </button>

      {/* Info note */}
      <div style={{ display: 'flex', gap: 10, padding: '12px 16px', marginTop: 14, borderRadius: 10, background: 'var(--blue-pale)', border: '1px solid var(--blue-light)' }}>
        <Info style={{ width: 14, height: 14, color: 'var(--blue)', flexShrink: 0, marginTop: 2 }} />
        <p style={{ fontSize: 12, color: 'var(--blue)', lineHeight: 1.6, margin: 0 }}>
          Multi-band GeoTIFFs supported — bands 1–3 used as RGB. PSNR/SSIM/SAM require a reference image; for custom uploads these show N/A.
        </p>
      </div>
    </div>
  );
};

// ============================================================================
// PROGRESS BAR
// ============================================================================
const ProgressBar = ({ progress, status, filename }) => (
  <div style={{ maxWidth: 520, margin: '80px auto 0' }}>
    <div className="card animate-fade-in" style={{ padding: 32 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 24 }}>
        <div style={{ width: 44, height: 44, borderRadius: 12, background: 'var(--accent-pale)', border: '1px solid var(--accent-light)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
          <Cpu style={{ width: 20, height: 20, color: 'var(--accent)' }} className="animate-spin" />
        </div>
        <div>
          <p style={{ fontSize: 15, fontWeight: 600, color: 'var(--ink)', marginBottom: 2 }}>{status}</p>
          {filename && <p style={{ fontSize: 11, color: 'var(--warm-dark)', fontFamily: 'DM Mono, monospace' }}>{filename}</p>}
        </div>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8, fontSize: 11, color: 'var(--warm-mid)', fontFamily: 'DM Mono, monospace' }}>
        <span>EDSR-Lite · Apple MPS</span>
        <span>{progress}%</span>
      </div>
    </div>
  </div>
);

// ============================================================================
// ERROR CARD
// ============================================================================
const ErrorCard = ({ message, onRetry }) => (
  <div style={{ maxWidth: 480, margin: '80px auto 0', textAlign: 'center' }}>
    <div className="card animate-fade-in" style={{ padding: 40 }}>
      <div style={{ width: 52, height: 52, borderRadius: 14, background: 'var(--red-light)', border: '1px solid #fecaca', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
        <AlertCircle style={{ width: 24, height: 24, color: 'var(--red)' }} />
      </div>
      <p style={{ fontSize: 16, fontWeight: 600, color: 'var(--ink)', marginBottom: 8 }}>Processing Failed</p>
      <p style={{ fontSize: 13, color: 'var(--warm-dark)', marginBottom: 24, lineHeight: 1.6 }}>{message}</p>
      <button onClick={onRetry} className="btn-primary">Try Again</button>
    </div>
  </div>
);

// ============================================================================
// COMPARISON VIEWER — 3D framed slider
// ============================================================================
const ComparisonViewer = ({ results }) => {
  const [pos, setPos] = useState(50);
  const [dragging, setDragging] = useState(false);
  const [zoom, setZoom] = useState(1);
  const ref = useRef();

  const getX = useCallback((e) => {
    if (!ref.current) return 50;
    const rect = ref.current.getBoundingClientRect();
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    return Math.max(0, Math.min(100, (cx - rect.left) / rect.width * 100));
  }, []);

  const onMove = useCallback((e) => { if (dragging) { e.preventDefault(); setPos(getX(e)); } }, [dragging, getX]);
  const onUp = useCallback(() => setDragging(false), []);

  useEffect(() => {
    if (dragging) {
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
      window.addEventListener('touchmove', onMove, { passive: false });
      window.addEventListener('touchend', onUp);
      return () => {
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', onUp);
        window.removeEventListener('touchmove', onMove);
        window.removeEventListener('touchend', onUp);
      };
    }
  }, [dragging, onMove, onUp]);

  const { input, output } = results.images;
  const { dimensions: dim } = results;

  return (
    <div className="card animate-fade-in" style={{ padding: 20, gridColumn: '1 / -1' }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
        <div>
          <p style={{ fontSize: 13, fontWeight: 700, color: 'var(--ink)', letterSpacing: '0.02em', textTransform: 'uppercase', marginBottom: 2 }}>
            Precision Before / After Viewer
          </p>
          <p style={{ fontSize: 11, color: 'var(--warm-mid)' }}>Drag the line to compare · Use mouse wheel to zoom</p>
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          {[
            { icon: <ZoomIn style={{width:14,height:14}} />, action: () => setZoom(z => Math.min(z*1.5,8)) },
            { icon: <ZoomOut style={{width:14,height:14}} />, action: () => setZoom(z => Math.max(z/1.5,0.5)) },
            { icon: <RotateCcw style={{width:14,height:14}} />, action: () => { setZoom(1); setPos(50); } },
          ].map((b, i) => (
            <button key={i} onClick={b.action} className="btn-secondary" style={{ padding: '6px 8px', borderRadius: 8, lineHeight: 1 }}>{b.icon}</button>
          ))}
          <span style={{ fontSize: 11, color: 'var(--warm-mid)', fontFamily: 'DM Mono, monospace', alignSelf: 'center', marginLeft: 2 }}>{Math.round(zoom*100)}%</span>
        </div>
      </div>

      {/* Labels */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
        <span className="badge badge-amber">● Original {dim?.input?.width}×{dim?.input?.height}</span>
        <span className="badge badge-green">AI Enhanced {dim?.output?.width}×{dim?.output?.height} ●</span>
      </div>

      {/* Slider frame */}
      <div
        ref={ref}
        className="comparison-container rounded-xl overflow-hidden shadow-inner"
        style={{ height: '65vh', minHeight: 500, background: '#111', cursor: 'col-resize', position: 'relative' }}
        onMouseDown={(e) => { e.preventDefault(); setDragging(true); setPos(getX(e)); }}
        onTouchStart={(e) => { setDragging(true); setPos(getX(e)); }}
        onWheel={(e) => {
          e.preventDefault();
          if (e.deltaY < 0) setZoom(z => Math.min(z * 1.1, 8));
          else setZoom(z => Math.max(z / 1.1, 0.5));
        }}
      >
        {/* SR image (back) */}
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <img src={output} alt="AI SR" draggable={false} style={{ width: '100%', height: '100%', objectFit: 'contain', transform: `scale(${zoom})`, imageRendering: zoom > 2 ? 'pixelated' : 'auto', pointerEvents: 'none', transition: dragging ? 'none' : 'transform 0.1s' }} />
        </div>
        {/* Original (clipped left) */}
        <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', width: `${pos}%` }}>
          <div style={{ position: 'absolute', inset: 0, width: `${100/pos*100}%` }}>
            <img src={input} alt="Original" draggable={false} style={{ width: '100%', height: '100%', objectFit: 'contain', transform: `scale(${zoom})`, imageRendering: zoom > 2 ? 'pixelated' : 'auto', pointerEvents: 'none', transition: dragging ? 'none' : 'transform 0.1s' }} />
          </div>
        </div>
        {/* Divider + handle */}
        <div style={{ position: 'absolute', top: 0, bottom: 0, zIndex: 20, pointerEvents: 'none', left: `calc(${pos}% - 1px)` }}>
          <div className="comparison-slider-line" style={{ background: 'white', width: 2, height: '100%', boxShadow: '0 0 10px rgba(0,0,0,0.5)' }} />
          <div className="comparison-handle" style={{ pointerEvents: 'auto', cursor: 'col-resize', background: 'white', width: 28, height: 28, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', boxShadow: '0 2px 6px rgba(0,0,0,0.3)' }}>
            <svg width="14" height="12" viewBox="0 0 14 12" fill="none">
              <path d="M3.5 6L0 3V9L3.5 6Z" fill="#333"/>
              <path d="M10.5 6L14 3V9L10.5 6Z" fill="#333"/>
            </svg>
          </div>
        </div>
        {/* Scale chip */}
        <div style={{ position: 'absolute', bottom: 16, left: '50%', transform: 'translateX(-50%)', zIndex: 10, pointerEvents: 'none' }}>
          <div style={{ padding: '6px 14px', borderRadius: 999, background: 'rgba(255,255,255,0.9)', backdropFilter: 'blur(8px)', border: '1px solid var(--stone)', fontSize: 11, fontWeight: 700, color: 'var(--accent)', letterSpacing: '0.06em', fontFamily: 'DM Mono, monospace', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
            ×{dim?.scale_factor} UPSCALE · EDSR-Lite
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// METRICS PANEL
// ============================================================================
const MetricsPanel = ({ metrics, modelName }) => {
  const fmt = (v, dp = 2) => v != null ? Number(v).toFixed(dp) : '—';

  const cards = [
    { label: 'PSNR', unit: 'dB',   value: fmt(metrics?.psnr),    sub: metrics?.reference_available ? 'vs HR ref' : 'vs baseline', accent: 'var(--blue)' },
    { label: 'SSIM', unit: '',     value: fmt(metrics?.ssim, 4), sub: '1.0 = perfect',              accent: 'var(--ink)' },
    { label: 'SAM',  unit: '°',    value: fmt(metrics?.sam),     sub: 'spectral angle',             accent: 'var(--amber)' },
    { label: '×',    unit: '',     value: `${fmt(metrics?.sharpness_gain)}`, sub: 'sharpness gain', accent: 'var(--accent)' },
    { label: 'Spectral', unit: '%', value: fmt(metrics?.spectral_consistency, 1), sub: 'consistency', accent: 'var(--ink)' },
  ];

  return (
    <div className="card animate-fade-in" style={{ padding: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
        <div style={{ width: 30, height: 30, borderRadius: 8, background: 'var(--accent-pale)', border: '1px solid var(--accent-light)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <BarChart2 style={{ width: 14, height: 14, color: 'var(--accent)' }} />
        </div>
        <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--ink)' }}>Quality Metrics</span>
      </div>

      {modelName && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 7, padding: '6px 10px', borderRadius: 8, background: 'var(--cream-2)', border: '1px solid var(--stone)', marginBottom: 14, fontSize: 11, color: 'var(--warm-dark)', fontFamily: 'DM Mono, monospace' }}>
          <Cpu style={{ width: 11, height: 11, color: 'var(--accent)', flexShrink: 0 }} />
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{modelName}</span>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {cards.map(c => (
          <div key={c.label} className="card-flat" style={{ padding: '12px 14px' }}>
            <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--warm-mid)', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 4 }}>{c.label}</div>
            <div style={{ fontSize: 22, fontWeight: 600, fontFamily: 'DM Mono, monospace', color: c.accent, letterSpacing: '-0.02em', lineHeight: 1 }}>
              {c.value}<span style={{ fontSize: 12, marginLeft: 2 }}>{c.unit}</span>
            </div>
            <div style={{ fontSize: 10, color: 'var(--warm-mid)', marginTop: 3 }}>{c.sub}</div>
          </div>
        ))}
      </div>

      {!metrics?.reference_available && (
        <div style={{ display: 'flex', gap: 8, padding: '10px 12px', marginTop: 10, borderRadius: 8, background: 'var(--amber-light)', border: '1px solid #fde68a' }}>
          <Info style={{ width: 13, height: 13, color: 'var(--amber)', flexShrink: 0, marginTop: 1 }} />
          <p style={{ fontSize: 11, color: 'var(--amber)', margin: 0, lineHeight: 1.55 }}>
            No HR ground truth provided. PSNR, SSIM, and SAM are computed against a standard bicubic baseline to measure the AI's improvement.
          </p>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// GEOSPATIAL PANEL
// ============================================================================
const GeospatialPanel = ({ geoStats, overlayImage }) => (
  <div className="card animate-fade-in" style={{ padding: 20 }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 18 }}>
      <div style={{ width: 30, height: 30, borderRadius: 8, background: 'var(--blue-pale)', border: '1px solid var(--blue-light)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Navigation style={{ width: 14, height: 14, color: 'var(--blue)' }} />
      </div>
      <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--ink)' }}>Multi-Decision Geospatial Analysis</span>
      <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--warm-mid)', fontFamily: 'DM Mono, monospace', letterSpacing: '0.04em' }}>CANNY + VARI + RGB</span>
    </div>

    <div style={{ display: 'flex', gap: 16 }}>
      {/* Stats */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10 }}>
        
        {/* Urban Analysis */}
        <div className="card-inset" style={{ padding: '10px 14px' }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--warm-mid)', textTransform: 'uppercase', marginBottom: 6 }}>Urban / Infrastructure</div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontWeight: 500, color: 'var(--ink-light)', marginBottom: 6 }}>
            <span>Buildup Area</span>
            <span style={{ fontFamily: 'DM Mono, monospace', fontWeight: 600, color: 'var(--ink)' }}>{geoStats?.buildup_percent}%</span>
          </div>
          <div style={{ height: 5, background: 'var(--stone)', borderRadius: 999, overflow: 'hidden', marginBottom: 8 }}>
            <div style={{ height: '100%', width: `${geoStats?.buildup_percent || 0}%`, background: 'var(--red)', borderRadius: 999, transition: 'width 1s' }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 12, color: 'var(--ink-light)' }}>Structures Detected</span>
            <span style={{ fontFamily: 'DM Mono, monospace', fontWeight: 700, fontSize: 14, color: 'var(--ink)' }}>{geoStats?.detected_structures ?? '—'}</span>
          </div>
        </div>

        {/* Agricultural Analysis */}
        <div className="card-inset" style={{ padding: '10px 14px' }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--warm-mid)', textTransform: 'uppercase', marginBottom: 6 }}>Agriculture / Crop Health</div>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontWeight: 500, color: 'var(--ink-light)', marginBottom: 6 }}>
            <span>Healthy Crop</span>
            <span style={{ fontFamily: 'DM Mono, monospace', fontWeight: 600, color: 'var(--ink)' }}>{geoStats?.healthy_crop_percent}%</span>
          </div>
          <div style={{ height: 5, background: 'var(--stone)', borderRadius: 999, overflow: 'hidden', marginBottom: 8 }}>
            <div style={{ height: '100%', width: `${geoStats?.healthy_crop_percent || 0}%`, background: '#22c55e', borderRadius: 999, transition: 'width 1s' }} />
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontWeight: 500, color: 'var(--ink-light)', marginBottom: 6 }}>
            <span>Stressed Crop</span>
            <span style={{ fontFamily: 'DM Mono, monospace', fontWeight: 600, color: 'var(--ink)' }}>{geoStats?.stressed_crop_percent}%</span>
          </div>
          <div style={{ height: 5, background: 'var(--stone)', borderRadius: 999, overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${geoStats?.stressed_crop_percent || 0}%`, background: '#eab308', borderRadius: 999, transition: 'width 1s' }} />
          </div>
        </div>

        {/* Hydrological Analysis */}
        <div className="card-inset" style={{ padding: '10px 14px' }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--warm-mid)', textTransform: 'uppercase', marginBottom: 6 }}>Hydrology / Water</div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontWeight: 500, color: 'var(--ink-light)', marginBottom: 6 }}>
            <span>Water Bodies & Moisture</span>
            <span style={{ fontFamily: 'DM Mono, monospace', fontWeight: 600, color: 'var(--ink)' }}>{geoStats?.water_percent}%</span>
          </div>
          <div style={{ height: 5, background: 'var(--stone)', borderRadius: 999, overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${geoStats?.water_percent || 0}%`, background: '#3b82f6', borderRadius: 999, transition: 'width 1s' }} />
          </div>
        </div>

      </div>
      
      {/* Overlay image */}
      <div style={{ flex: 1, minHeight: 200 }}>
        <div className="card-inset" style={{ height: '100%', minHeight: 200, position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
          {overlayImage
            ? <img src={overlayImage} alt="Geo Overlay" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
            : <span style={{ fontSize: 12, color: 'var(--warm-mid)' }}>No overlay</span>}
          {overlayImage && (
            <div style={{ position: 'absolute', bottom: 8, left: 8, display: 'flex', flexDirection: 'column', gap: 3 }}>
              <span style={{ fontSize: 10, padding: '3px 6px', background: 'rgba(239,68,68,0.85)', color: '#fff', borderRadius: 4, fontFamily: 'DM Mono, monospace' }}>■ STRUCTURES</span>
              <span style={{ fontSize: 10, padding: '3px 6px', background: 'rgba(34,197,94,0.85)', color: '#fff', borderRadius: 4, fontFamily: 'DM Mono, monospace' }}>■ HEALTHY CROP</span>
              <span style={{ fontSize: 10, padding: '3px 6px', background: 'rgba(234,179,8,0.85)', color: '#fff', borderRadius: 4, fontFamily: 'DM Mono, monospace' }}>■ STRESSED CROP</span>
              <span style={{ fontSize: 10, padding: '3px 6px', background: 'rgba(59,130,246,0.85)', color: '#fff', borderRadius: 4, fontFamily: 'DM Mono, monospace' }}>■ WATER/MOISTURE</span>
            </div>
          )}
        </div>
      </div>
    </div>
  </div>
);

// ============================================================================
// CONFIDENCE PANEL
// ============================================================================
const ConfidencePanel = ({ confidenceImage }) => (
  <div className="card animate-fade-in" style={{ padding: 20 }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
      <div style={{ width: 30, height: 30, borderRadius: 8, background: '#f5f0ff', border: '1px solid #ddd6fe', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Eye style={{ width: 14, height: 14, color: '#7c3aed' }} />
      </div>
      <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--ink)' }}>Reconstruction Confidence</span>
      <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--warm-mid)', fontFamily: 'DM Mono, monospace' }}>DIFF-BASED UQ</span>
    </div>
    <div className="card-inset" style={{ aspectRatio: '16/9', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', position: 'relative' }}>
      {confidenceImage
        ? <img src={confidenceImage} alt="Confidence Map" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
        : <span style={{ fontSize: 12, color: 'var(--warm-mid)' }}>No data</span>}
      {confidenceImage && (
        <div style={{ position: 'absolute', bottom: 8, right: 8, display: 'flex', gap: 8, padding: '5px 10px', background: 'rgba(255,255,255,0.85)', backdropFilter: 'blur(8px)', borderRadius: 8, border: '1px solid var(--stone)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, background: '#1d4ed8' }} />
            <span style={{ fontSize: 10, color: 'var(--warm-dark)' }}>High conf.</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, background: '#dc2626' }} />
            <span style={{ fontSize: 10, color: 'var(--warm-dark)' }}>Low conf.</span>
          </div>
        </div>
      )}
    </div>
    <div className="heatmap-legend" style={{ marginTop: 10 }} />
    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: 10, color: 'var(--warm-mid)' }}>
      <span>Blue → Preserved from input (confident)</span>
      <span>Red → AI reconstructed (verify)</span>
    </div>
    <p style={{ fontSize: 11, color: 'var(--warm-dark)', marginTop: 10, lineHeight: 1.6 }}>
      Use this map to identify areas where the AI added detail beyond the original. Red zones should be cross-checked against ground truth or field data before operational use.
    </p>
  </div>
);

// ============================================================================
// EXPORT PANEL
// ============================================================================
const ExportPanel = ({ results }) => (
  <div className="card animate-fade-in" style={{ padding: 20 }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
      <div style={{ width: 30, height: 30, borderRadius: 8, background: 'var(--accent-pale)', border: '1px solid var(--accent-light)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Download style={{ width: 14, height: 14, color: 'var(--accent)' }} />
      </div>
      <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--ink)' }}>Export Results</span>
    </div>

    <div className="card-inset" style={{ padding: '14px 16px', textAlign: 'center', marginBottom: 14 }}>
      <CheckCircle style={{ width: 24, height: 24, color: 'var(--accent)', margin: '0 auto 8px' }} />
      <p style={{ fontSize: 13, fontWeight: 600, color: 'var(--ink)', marginBottom: 3 }}>Processing Complete</p>
      <p style={{ fontSize: 11, color: 'var(--warm-dark)' }}>{results?.processing_time}s · Session {results?.session_id?.substring(0,8)}</p>
      {results?.source_info?.original_filename && (
        <p style={{ fontSize: 10, color: 'var(--warm-mid)', fontFamily: 'DM Mono, monospace', marginTop: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{results.source_info.original_filename}</p>
      )}
    </div>

    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <button className="btn-primary" style={{ width: '100%', justifyContent: 'center' }}
        onClick={() => {
          const a = document.createElement('a');
          a.href = results?.images?.output;
          a.download = `sr_${results?.session_id}.png`;
          a.click();
        }}>
        <Download style={{ width: 14, height: 14 }} /> Download SR Image (PNG)
      </button>
      <button className="btn-secondary" style={{ width: '100%', justifyContent: 'center' }}
        onClick={() => {
          const blob = new Blob([JSON.stringify({
            session_id: results?.session_id, model: results?.model_name,
            metrics: results?.metrics, geospatial: results?.geospatial,
            dimensions: results?.dimensions, processing_time: results?.processing_time,
            source_info: results?.source_info
          }, null, 2)], { type: 'application/json' });
          const a = document.createElement('a');
          a.href = URL.createObjectURL(blob);
          a.download = `metadata_${results?.session_id}.json`;
          a.click();
        }}>
        <FileText style={{ width: 14, height: 14 }} /> Export Metadata (JSON)
      </button>
    </div>
  </div>
);

// ============================================================================
// MAIN APP
// ============================================================================
export default function App() {
  const [state, setState] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  const [filename, setFilename] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [tab, setTab] = useState('comparison');

  const reset = () => { setState('idle'); setResults(null); setError(''); setProgress(0); setFilename(null); };

  const tick = (pBase, maxP, msg) => setInterval(() => {
    setProgress(p => { setStatus(msg(p)); return Math.min(p + 5, maxP); });
  }, 350);

  const handleDemo = async () => {
    setState('processing'); setProgress(5);
    const msgs = (p) => p < 35 ? 'Loading Sentinel-2 demo scene...' : p < 65 ? 'Running EDSR-Lite inference...' : 'Computing metrics & geospatial analysis...';
    const pInt = tick(5, 90, msgs);
    try {
      const res = await runDemo();
      clearInterval(pInt); setProgress(100);
      if (!res.success) throw new Error(res.error || 'Unknown error');
      setResults(res); setTimeout(() => setState('results'), 400);
    } catch (e) { clearInterval(pInt); setError(e.message); setState('error'); }
  };

  const handleFile = async (file) => {
    setFilename(file.name); setState('processing'); setProgress(5); setStatus(`Resizing & uploading ${file.name}...`);
    try {
      const res = await uploadFile(file, (p) => { setProgress(p + 5); setStatus(`Uploading... ${p}%`); });
      setProgress(50); setStatus('Running EDSR-Lite SR inference...');
      const pInt = tick(50, 90, (p) => p < 75 ? 'Running SR inference...' : 'Computing metrics...');
      if (!res.success) throw new Error(res.error || 'Unknown error');
      clearInterval(pInt); setProgress(100);
      setResults(res); setTimeout(() => setState('results'), 400);
    } catch (e) { setError(e.message); setState('error'); }
  };

  return (
    <div style={{ minHeight: '100vh', position: 'relative', zIndex: 1, paddingBottom: 80 }}>
      {/* Globe background */}
      <div style={{ position: 'fixed', inset: 0, zIndex: 0, opacity: 1, pointerEvents: 'none', background: '#000' }}>
        <Globe />
      </div>

      <Header onReset={state !== 'idle' ? reset : null} />

      <main style={{ maxWidth: 1100, margin: '0 auto', padding: '40px 20px 0', position: 'relative', zIndex: 1 }}>

        {state === 'idle' && (
          <div className="animate-fade-in" style={{ paddingTop: 20 }}>
            <UploadPanel onDemo={handleDemo} onFile={handleFile} />
          </div>
        )}

        {state === 'processing' && <ProgressBar progress={progress} status={status} filename={filename} />}

        {state === 'error' && <ErrorCard message={error} onRetry={reset} />}

        {state === 'results' && results && (
          <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {/* Tab bar */}
            <div className="tab-bar">
              {[
                { id: 'comparison', label: 'Visual Comparison' },
                { id: 'analysis',   label: 'Geospatial Analysis' },
              ].map(t => (
                <button key={t.id} className={`tab-btn${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
                  {t.label}
                </button>
              ))}
            </div>

            {tab === 'comparison' && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 16, alignItems: 'start' }}>
                <ComparisonViewer results={results} />
                <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                  <MetricsPanel metrics={results.metrics} modelName={results.model_name} />
                  <ExportPanel results={results} />
                </div>
              </div>
            )}

            {tab === 'analysis' && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <GeospatialPanel geoStats={results.geospatial} overlayImage={results.images?.geo_overlay} />
                <ConfidencePanel confidenceImage={results.images?.confidence} />
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
