import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  Satellite, UploadCloud, Eye, Activity, Navigation,
  Download, FileText, CheckCircle, ZoomIn, ZoomOut, RotateCcw,
  AlertCircle, Cpu, RefreshCw, Info, MapPin, BarChart2,
  Leaf, Building2, Droplets, ChevronRight, Layers, Shield,
  Zap, Image as ImageIcon, ArrowRight,
} from 'lucide-react';
import Globe from '@/components/ui/globe';

// ─────────────────────────────────────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────────────────────────────────────
const API = 'https://ai-srm.onrender.com';

const SCENES = [
  {
    id: 'agriculture',
    label: 'Agricultural',
    sub: 'Crop monitoring & field health',
    icon: <Leaf style={{ width: 18, height: 18 }} />,
    color: '#22c55e',
    pale: '#f0fdf4',
    border: '#bbf7d0',
    tag: 'VARI · Crop Stress · NDWI',
  },
  {
    id: 'urban',
    label: 'Urban',
    sub: 'Infrastructure & planning',
    icon: <Building2 style={{ width: 18, height: 18 }} />,
    color: '#f97316',
    pale: '#fff7ed',
    border: '#fed7aa',
    tag: 'Canny · Density · Roads',
  },
  {
    id: 'coastal',
    label: 'Coastal / Disaster',
    sub: 'Flood & damage assessment',
    icon: <Droplets style={{ width: 18, height: 18 }} />,
    color: '#3b82f6',
    pale: '#eff6ff',
    border: '#bfdbfe',
    tag: 'NDWI · Inundation · Change',
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// API HELPERS
// ─────────────────────────────────────────────────────────────────────────────
async function pingServer(): Promise<boolean> {
  try {
    const r = await fetch(`${API}/api/ping`, { signal: AbortSignal.timeout(8000) });
    return r.ok;
  } catch { return false; }
}

async function runDemo(scene: string): Promise<any> {
  const r = await fetch(`${API}/api/demo/${scene}`, { method: 'POST' });
  if (!r.ok) throw new Error(`Server error ${r.status}`);
  return r.json();
}

const MAX_UPLOAD_PX = 512;

async function resizeToBlob(file: File): Promise<File> {
  if (/\.tiff?$/i.test(file.name)) return file;
  return new Promise((resolve) => {
    const img = new window.Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      const { naturalWidth: w, naturalHeight: h } = img;
      const s = Math.min(1, MAX_UPLOAD_PX / Math.max(w, h));
      const canvas = document.createElement('canvas');
      canvas.width  = Math.round(w * s);
      canvas.height = Math.round(h * s);
      canvas.getContext('2d')!.drawImage(img, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => resolve(new File([blob!], file.name, { type: 'image/png' })),
        'image/png',
      );
    };
    img.onerror = () => { URL.revokeObjectURL(url); resolve(file); };
    img.src = url;
  });
}

async function uploadFile(file: File, onProg: (p: number) => void): Promise<any> {
  const resized = await resizeToBlob(file);
  return new Promise((resolve, reject) => {
    const fd  = new FormData();
    fd.append('file', resized);
    const xhr = new XMLHttpRequest();
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProg(Math.round((e.loaded / e.total) * 30));
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
      else reject(new Error(`Upload failed: ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error('Network error — is the server awake?'));
    xhr.open('POST', `${API}/api/upload`);
    xhr.send(fd);
  });
}

function downloadDataUrl(dataUrl: string, filename: string) {
  const a = document.createElement('a');
  a.href     = dataUrl;
  a.download = filename;
  a.click();
}

// ─────────────────────────────────────────────────────────────────────────────
// WARM-UP BANNER
// ─────────────────────────────────────────────────────────────────────────────
const WarmupBanner = ({ pct }: { pct: number }) => (
  <div style={{
    position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)',
    zIndex: 200, display: 'flex', alignItems: 'center', gap: 12,
    padding: '10px 20px', borderRadius: 99,
    background: 'rgba(30,77,140,0.95)', backdropFilter: 'blur(12px)',
    border: '1px solid rgba(255,255,255,0.15)', boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
    color: '#fff', fontSize: 13, fontWeight: 500,
    animation: 'fadeIn 0.3s ease',
  }}>
    <div style={{
      width: 16, height: 16, border: '2px solid rgba(255,255,255,0.3)',
      borderTopColor: '#fff', borderRadius: '50%',
      animation: 'spin 0.8s linear infinite',
    }} />
    <span>Waking up AI server… {pct}%</span>
    <div style={{
      width: 80, height: 4, background: 'rgba(255,255,255,0.2)',
      borderRadius: 99, overflow: 'hidden',
    }}>
      <div style={{
        height: '100%', width: `${pct}%`,
        background: '#60a5fa', borderRadius: 99,
        transition: 'width 0.3s ease',
      }} />
    </div>
  </div>
);

// ─────────────────────────────────────────────────────────────────────────────
// HEADER
// ─────────────────────────────────────────────────────────────────────────────
const Header = ({ onReset, modelReady }: { onReset: (() => void) | null; modelReady: boolean }) => (
  <header className="header-float sticky top-0 z-50">
    <div className="max-w-6xl mx-auto px-5 h-14 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div style={{
          width: 36, height: 36,
          background: 'var(--accent)',
          borderRadius: 10,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 3px 0 #1a4033, 0 4px 12px rgba(45,106,79,0.25)',
        }}>
          <Satellite style={{ width: 18, height: 18, color: '#fff' }} />
        </div>
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--ink)', letterSpacing: '-0.01em', lineHeight: 1.2 }}>
            Satellite AI · SRM
          </div>
          <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--warm-dark)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            PS-26142 · NTRO · SIH 2024
          </div>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '4px 12px', borderRadius: 999,
          background: modelReady ? 'var(--accent-pale)' : '#fff7ed',
          border: `1px solid ${modelReady ? 'var(--accent-light)' : '#fed7aa'}`,
          fontSize: 11, fontWeight: 600,
          color: modelReady ? 'var(--accent)' : '#c2410c',
          letterSpacing: '0.04em',
        }}>
          <span style={{
            width: 6, height: 6, borderRadius: '50%',
            background: modelReady ? 'var(--accent-2)' : '#f97316',
            display: 'inline-block',
            boxShadow: `0 0 0 2px ${modelReady ? 'var(--accent-light)' : '#fed7aa'}`,
          }} className={modelReady ? 'animate-pulse' : ''} />
          {modelReady ? 'A2N MODEL ONLINE' : 'SERVER WAKING'}
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

// ─────────────────────────────────────────────────────────────────────────────
// HERO LANDING
// ─────────────────────────────────────────────────────────────────────────────
const HeroSection = () => (
  <div style={{ textAlign: 'center', padding: '12px 0 32px' }}>
    {/* Badge */}
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: '4px 14px', borderRadius: 999,
      background: 'var(--blue-pale)', border: '1px solid var(--blue-light)',
      fontSize: 11, fontWeight: 600, color: 'var(--blue)',
      letterSpacing: '0.06em', marginBottom: 18,
    }}>
      <Cpu style={{ width: 11, height: 11 }} />
      SIH 2024 · Problem Statement 26142 · NTRO
    </div>

    <h1 style={{
      fontSize: 40, fontWeight: 800, color: 'var(--ink)',
      letterSpacing: '-0.035em', lineHeight: 1.12, marginBottom: 14,
    }}>
      Deep Learning&nbsp;
      <span style={{ color: 'var(--accent)' }}>Super-Resolution</span>
      <br />Mapping from Satellite Imagery
    </h1>

    <p style={{
      fontSize: 16, color: 'var(--warm-dark)',
      maxWidth: 540, margin: '0 auto 28px',
      lineHeight: 1.7,
    }}>
      Transforms <strong>10 m</strong> Sentinel-2 imagery to <strong>&lt;2.5 m</strong> resolution using A2N
      deep learning — enabling precision crop monitoring, urban analysis, and disaster assessment.
    </p>

    {/* Pipeline steps */}
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      gap: 6, flexWrap: 'wrap', marginBottom: 8,
    }}>
      {[
        { icon: <Layers style={{ width: 13, height: 13 }} />, label: '10 m Sentinel-2' },
        { icon: <Zap style={{ width: 13, height: 13 }} />, label: 'Pre-processing' },
        { icon: <Cpu style={{ width: 13, height: 13 }} />, label: 'A2N ×4 AI Model' },
        { icon: <Shield style={{ width: 13, height: 13 }} />, label: 'Accuracy Validation' },
        { icon: <MapPin style={{ width: 13, height: 13 }} />, label: '2.5 m Output' },
      ].map((step, i, arr) => (
        <React.Fragment key={step.label}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 5,
            padding: '5px 12px', borderRadius: 99,
            background: '#fff', border: '1px solid var(--stone)',
            fontSize: 11, fontWeight: 600, color: 'var(--ink)',
            boxShadow: 'var(--shadow-sm)',
          }}>
            <span style={{ color: 'var(--accent)' }}>{step.icon}</span>
            {step.label}
          </div>
          {i < arr.length - 1 && (
            <ArrowRight style={{ width: 12, height: 12, color: 'var(--warm-mid)', flexShrink: 0 }} />
          )}
        </React.Fragment>
      ))}
    </div>
  </div>
);

// ─────────────────────────────────────────────────────────────────────────────
// SCENE SELECTOR
// ─────────────────────────────────────────────────────────────────────────────
const SceneSelector = ({
  onScene, onFile,
}: { onScene: (id: string) => void; onFile: (f: File) => void }) => {
  const [dragOver, setDragOver] = useState(false);
  const [hoverId, setHoverId]   = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  }, [onFile]);

  return (
    <div style={{ maxWidth: 720, margin: '0 auto' }}>
      {/* Preset scenes */}
      <p style={{ fontSize: 12, fontWeight: 700, color: 'var(--warm-mid)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 12 }}>
        Select a preset Sentinel-2 scene
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 18 }}>
        {SCENES.map((sc) => (
          <button
            key={sc.id}
            onClick={() => onScene(sc.id)}
            onMouseEnter={() => setHoverId(sc.id)}
            onMouseLeave={() => setHoverId(null)}
            style={{
              all: 'unset', cursor: 'pointer',
              display: 'flex', flexDirection: 'column', gap: 8,
              padding: '16px 18px', borderRadius: 14,
              background: hoverId === sc.id ? sc.pale : '#fff',
              border: `1.5px solid ${hoverId === sc.id ? sc.border : 'var(--stone)'}`,
              boxShadow: hoverId === sc.id ? `0 4px 16px ${sc.color}22` : 'var(--shadow-sm)',
              transition: 'all 0.18s ease',
              transform: hoverId === sc.id ? 'translateY(-2px)' : 'none',
            }}
          >
            <div style={{
              width: 36, height: 36, borderRadius: 10,
              background: `${sc.color}18`,
              border: `1px solid ${sc.color}30`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: sc.color,
            }}>
              {sc.icon}
            </div>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--ink)', marginBottom: 2 }}>
                {sc.label}
              </div>
              <div style={{ fontSize: 11, color: 'var(--warm-dark)', marginBottom: 6 }}>{sc.sub}</div>
              <div style={{
                fontSize: 9, fontWeight: 600, letterSpacing: '0.05em',
                fontFamily: 'DM Mono, monospace',
                color: sc.color, textTransform: 'uppercase',
              }}>
                {sc.tag}
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginTop: 4, fontSize: 11, fontWeight: 600, color: sc.color }}>
              Run Analysis <ChevronRight style={{ width: 12, height: 12 }} />
            </div>
          </button>
        ))}
      </div>

      {/* OR divider */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, margin: '16px 0' }}>
        <div className="divider" style={{ flex: 1 }} />
        <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--warm-mid)', letterSpacing: '0.06em' }}>OR</span>
        <div className="divider" style={{ flex: 1 }} />
      </div>

      {/* Upload drop zone */}
      <div
        className={`drop-zone${dragOver ? ' active' : ''}`}
        style={{ padding: '32px 24px', textAlign: 'center' }}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileRef.current?.click()}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".tif,.tiff,.png,.jpg,.jpeg"
          style={{ display: 'none' }}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }}
        />
        <div style={{
          width: 52, height: 52, borderRadius: 14,
          background: dragOver ? 'var(--accent-pale)' : '#fff',
          border: '1.5px solid var(--stone)',
          boxShadow: dragOver
            ? '0 6px 0 var(--accent-light), var(--shadow-lg)'
            : '0 5px 0 var(--stone), var(--shadow-md)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          margin: '0 auto 14px', transition: 'all 0.2s',
        }}>
          <UploadCloud style={{ width: 24, height: 24, color: dragOver ? 'var(--accent)' : 'var(--warm-mid)' }} />
        </div>
        <p style={{ fontSize: 15, fontWeight: 600, color: 'var(--ink)', marginBottom: 5 }}>
          {dragOver ? 'Drop to analyse' : 'Upload your own satellite image'}
        </p>
        <p style={{ fontSize: 12, color: 'var(--warm-dark)', marginBottom: 12 }}>
          GeoTIFF · TIFF · PNG · JPG — any Sentinel-2 or multi-band raster
        </p>
        <div style={{ display: 'flex', gap: 6, justifyContent: 'center', flexWrap: 'wrap' }}>
          {['.tif (GeoTIFF)', '.tiff', '.png', '.jpg'].map((t) => (
            <span key={t} className="badge badge-stone">{t}</span>
          ))}
        </div>
      </div>

      {/* Info note */}
      <div style={{
        display: 'flex', gap: 10, padding: '11px 15px', marginTop: 14,
        borderRadius: 10, background: 'var(--blue-pale)', border: '1px solid var(--blue-light)',
      }}>
        <Info style={{ width: 13, height: 13, color: 'var(--blue)', flexShrink: 0, marginTop: 2 }} />
        <p style={{ fontSize: 11, color: 'var(--blue)', lineHeight: 1.6, margin: 0 }}>
          Multi-band GeoTIFFs: bands 1–3 used as RGB with 2–98 percentile normalization.
          PSNR / SSIM / SAM are measured against the HR reference in demo mode,
          and against a bicubic baseline for custom uploads.
        </p>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// PIPELINE PROGRESS
// ─────────────────────────────────────────────────────────────────────────────
const PipelineProgress = ({
  progress, status, filename,
}: { progress: number; status: string; filename: string | null }) => {
  const steps = [
    { label: 'Pre-processing', icon: <Layers style={{ width: 13, height: 13 }} />, thresh: 20 },
    { label: 'A2N Inference',  icon: <Cpu style={{ width: 13, height: 13 }} />,    thresh: 60 },
    { label: 'Quality Metrics',icon: <BarChart2 style={{ width: 13, height: 13 }} />,thresh: 80 },
    { label: 'Geo Analysis',   icon: <MapPin style={{ width: 13, height: 13 }} />,  thresh: 95 },
  ];

  return (
    <div style={{ maxWidth: 540, margin: '60px auto 0' }}>
      <div className="card animate-fade-in" style={{ padding: 32 }}>
        {/* Spinner + status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 24 }}>
          <div style={{
            width: 46, height: 46, borderRadius: 13,
            background: 'var(--accent-pale)', border: '1px solid var(--accent-light)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
          }}>
            <Cpu style={{ width: 21, height: 21, color: 'var(--accent)' }} className="animate-spin" />
          </div>
          <div>
            <p style={{ fontSize: 15, fontWeight: 600, color: 'var(--ink)', marginBottom: 2 }}>{status}</p>
            {filename && (
              <p style={{ fontSize: 11, color: 'var(--warm-dark)', fontFamily: 'DM Mono, monospace' }}>{filename}</p>
            )}
          </div>
        </div>

        {/* Progress bar */}
        <div className="progress-track" style={{ marginBottom: 16 }}>
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>

        {/* Pipeline steps */}
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 4 }}>
          {steps.map((step) => {
            const done = progress >= step.thresh;
            return (
              <div key={step.label} style={{
                flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5,
              }}>
                <div style={{
                  width: 28, height: 28, borderRadius: 8,
                  background: done ? 'var(--accent-pale)' : 'var(--cream-2)',
                  border: `1px solid ${done ? 'var(--accent-light)' : 'var(--stone)'}`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: done ? 'var(--accent)' : 'var(--warm-mid)',
                  transition: 'all 0.3s',
                }}>
                  {step.icon}
                </div>
                <span style={{ fontSize: 9, fontWeight: 600, textAlign: 'center', color: done ? 'var(--accent)' : 'var(--warm-mid)', letterSpacing: '0.04em' }}>
                  {step.label}
                </span>
              </div>
            );
          })}
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 16, fontSize: 11, color: 'var(--warm-mid)', fontFamily: 'DM Mono, monospace' }}>
          <span>A2N ×4 · Sentinel-2 SRM</span>
          <span>{progress}%</span>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// ERROR CARD
// ─────────────────────────────────────────────────────────────────────────────
const ErrorCard = ({ message, onRetry }: { message: string; onRetry: () => void }) => (
  <div style={{ maxWidth: 480, margin: '70px auto 0', textAlign: 'center' }}>
    <div className="card animate-fade-in" style={{ padding: 40 }}>
      <div style={{
        width: 52, height: 52, borderRadius: 14,
        background: 'var(--red-light)', border: '1px solid #fecaca',
        display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px',
      }}>
        <AlertCircle style={{ width: 24, height: 24, color: 'var(--red)' }} />
      </div>
      <p style={{ fontSize: 16, fontWeight: 600, color: 'var(--ink)', marginBottom: 8 }}>Processing Failed</p>
      <p style={{ fontSize: 13, color: 'var(--warm-dark)', marginBottom: 24, lineHeight: 1.6 }}>{message}</p>
      <button onClick={onRetry} className="btn-primary">Try Again</button>
    </div>
  </div>
);

// ─────────────────────────────────────────────────────────────────────────────
// COMPARISON VIEWER
// ─────────────────────────────────────────────────────────────────────────────
const ComparisonViewer = ({ results }: { results: any }) => {
  const [pos, setPos]         = useState(50);
  const [dragging, setDragging] = useState(false);
  const [zoom, setZoom]       = useState(1);
  const ref = useRef<HTMLDivElement>(null);

  const getX = useCallback((e: MouseEvent | TouchEvent | React.MouseEvent | React.TouchEvent) => {
    if (!ref.current) return 50;
    const rect = ref.current.getBoundingClientRect();
    const cx = ('touches' in e) ? e.touches[0].clientX : (e as MouseEvent).clientX;
    return Math.max(0, Math.min(100, (cx - rect.left) / rect.width * 100));
  }, []);

  const onMove = useCallback((e: MouseEvent | TouchEvent) => {
    if (dragging) { e.preventDefault(); setPos(getX(e)); }
  }, [dragging, getX]);
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
  const dim = results.dimensions;

  return (
    <div className="card animate-fade-in" style={{ padding: 20, gridColumn: '1 / -1' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
        <div>
          <p style={{ fontSize: 13, fontWeight: 700, color: 'var(--ink)', textTransform: 'uppercase', marginBottom: 2 }}>
            Precision Before / After Viewer
          </p>
          <p style={{ fontSize: 11, color: 'var(--warm-mid)' }}>Drag the divider · scroll to zoom · {results.scale_factor}× upscale</p>
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          {[
            { icon: <ZoomIn style={{ width: 14, height: 14 }} />, fn: () => setZoom((z) => Math.min(z * 1.5, 8)) },
            { icon: <ZoomOut style={{ width: 14, height: 14 }} />, fn: () => setZoom((z) => Math.max(z / 1.5, 0.5)) },
            { icon: <RotateCcw style={{ width: 14, height: 14 }} />, fn: () => { setZoom(1); setPos(50); } },
          ].map((b, i) => (
            <button key={i} onClick={b.fn} className="btn-secondary" style={{ padding: '6px 8px', borderRadius: 8 }}>{b.icon}</button>
          ))}
          <span style={{ fontSize: 11, color: 'var(--warm-mid)', fontFamily: 'DM Mono, monospace', alignSelf: 'center', marginLeft: 4 }}>
            {Math.round(zoom * 100)}%
          </span>
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
        <span className="badge badge-amber">● Low-Res Input {dim?.input?.width}×{dim?.input?.height} px</span>
        <span className="badge badge-green">AI Enhanced {dim?.output?.width}×{dim?.output?.height} px ●</span>
      </div>

      <div
        ref={ref}
        style={{
          height: '62vh', minHeight: 460, background: '#0a0a0a',
          borderRadius: 12, overflow: 'hidden', cursor: 'col-resize',
          position: 'relative',
        }}
        onMouseDown={(e) => { e.preventDefault(); setDragging(true); setPos(getX(e)); }}
        onTouchStart={(e) => { setDragging(true); setPos(getX(e)); }}
        onWheel={(e) => {
          e.preventDefault();
          setZoom((z) => e.deltaY < 0 ? Math.min(z * 1.1, 8) : Math.max(z / 1.1, 0.5));
        }}
      >
        {/* SR background */}
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <img src={output} alt="SR" draggable={false}
            style={{ width: '100%', height: '100%', objectFit: 'contain', transform: `scale(${zoom})`, imageRendering: zoom > 2 ? 'pixelated' : 'auto', pointerEvents: 'none', transition: dragging ? 'none' : 'transform 0.1s' }} />
        </div>
        {/* Original clip */}
        <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', width: `${pos}%` }}>
          <div style={{ position: 'absolute', inset: 0, width: `${100 / pos * 100}%` }}>
            <img src={input} alt="Original" draggable={false}
              style={{ width: '100%', height: '100%', objectFit: 'contain', transform: `scale(${zoom})`, imageRendering: zoom > 2 ? 'pixelated' : 'auto', pointerEvents: 'none', transition: dragging ? 'none' : 'transform 0.1s' }} />
          </div>
        </div>
        {/* Divider */}
        <div style={{ position: 'absolute', top: 0, bottom: 0, left: `calc(${pos}% - 1px)`, zIndex: 20, pointerEvents: 'none' }}>
          <div style={{ background: 'white', width: 2, height: '100%', boxShadow: '0 0 12px rgba(0,0,0,0.6)' }} />
          <div style={{
            pointerEvents: 'auto', cursor: 'col-resize',
            background: 'white', width: 30, height: 30, borderRadius: '50%',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            position: 'absolute', top: '50%', left: '50%',
            transform: 'translate(-50%, -50%)',
            boxShadow: '0 2px 10px rgba(0,0,0,0.4)',
          }}>
            <svg width="14" height="12" viewBox="0 0 14 12" fill="none">
              <path d="M3.5 6L0 3V9L3.5 6Z" fill="#333" />
              <path d="M10.5 6L14 3V9L10.5 6Z" fill="#333" />
            </svg>
          </div>
        </div>
        {/* Scale badge */}
        <div style={{ position: 'absolute', bottom: 14, left: '50%', transform: 'translateX(-50%)', zIndex: 10, pointerEvents: 'none' }}>
          <div style={{
            padding: '6px 14px', borderRadius: 99,
            background: 'rgba(255,255,255,0.92)', backdropFilter: 'blur(8px)',
            border: '1px solid var(--stone)', fontSize: 11, fontWeight: 700,
            color: 'var(--accent)', letterSpacing: '0.06em',
            fontFamily: 'DM Mono, monospace', boxShadow: '0 4px 12px rgba(0,0,0,0.12)',
          }}>
            ×{results.scale_factor} · {results.model_name}
          </div>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// METRICS PANEL
// ─────────────────────────────────────────────────────────────────────────────
const MetricsPanel = ({ metrics, modelName }: { metrics: any; modelName: string }) => {
  const fmt = (v: any, dp = 2) => (v != null ? Number(v).toFixed(dp) : '—');

  const cards = [
    { label: 'PSNR', unit: 'dB',  value: fmt(metrics?.psnr),    sub: metrics?.reference_available ? 'vs HR ref' : 'vs bicubic', accent: 'var(--blue)' },
    { label: 'SSIM', unit: '',    value: fmt(metrics?.ssim, 4),  sub: '1.0 = perfect fidelity',   accent: 'var(--ink)'  },
    { label: 'SAM',  unit: '°',   value: fmt(metrics?.sam),      sub: 'spectral angle mapper',     accent: 'var(--amber)' },
    { label: 'Sharp', unit: '×',  value: fmt(metrics?.sharpness_gain), sub: 'sharpness gain',     accent: 'var(--accent)' },
    { label: 'SC',   unit: '%',   value: fmt(metrics?.spectral_consistency, 1), sub: 'spectral consistency', accent: '#7c3aed' },
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
        <div style={{
          display: 'flex', alignItems: 'center', gap: 7,
          padding: '6px 10px', borderRadius: 8,
          background: 'var(--cream-2)', border: '1px solid var(--stone)',
          marginBottom: 14, fontSize: 11, color: 'var(--warm-dark)',
          fontFamily: 'DM Mono, monospace',
        }}>
          <Cpu style={{ width: 11, height: 11, color: 'var(--accent)', flexShrink: 0 }} />
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{modelName}</span>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {cards.map((c) => (
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
            No HR reference provided — PSNR/SSIM/SAM measured against bicubic baseline. Upload a reference for full benchmark.
          </p>
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// GEOSPATIAL PANEL
// ─────────────────────────────────────────────────────────────────────────────
const GeospatialPanel = ({ geoStats, overlayImage, sceneLabel }: { geoStats: any; overlayImage: string; sceneLabel: string }) => {
  const bars = [
    { label: 'Healthy Vegetation', key: 'healthy_crop_percent', color: '#22c55e' },
    { label: 'Stressed Crop',      key: 'stressed_crop_percent', color: '#eab308' },
    { label: 'Water / Moisture',   key: 'water_percent',         color: '#3b82f6' },
    { label: 'Built-up / Urban',   key: 'buildup_percent',       color: '#ef4444' },
  ];

  return (
    <div className="card animate-fade-in" style={{ padding: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
        <div style={{ width: 30, height: 30, borderRadius: 8, background: 'var(--blue-pale)', border: '1px solid var(--blue-light)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Navigation style={{ width: 14, height: 14, color: 'var(--blue)' }} />
        </div>
        <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--ink)' }}>Land-Cover Analysis</span>
        {sceneLabel && (
          <span style={{ marginLeft: 'auto', fontSize: 10, fontWeight: 600, color: 'var(--warm-mid)', fontFamily: 'DM Mono, monospace' }}>
            {sceneLabel.toUpperCase()}
          </span>
        )}
      </div>

      <div style={{ display: 'flex', gap: 16 }}>
        {/* Bars */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10 }}>
          {bars.map((b) => (
            <div key={b.key}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--ink-light)', marginBottom: 5 }}>
                <span>{b.label}</span>
                <span style={{ fontFamily: 'DM Mono, monospace', fontWeight: 600, color: 'var(--ink)' }}>
                  {geoStats?.[b.key] ?? '—'}%
                </span>
              </div>
              <div style={{ height: 6, background: 'var(--stone)', borderRadius: 99, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${geoStats?.[b.key] || 0}%`, background: b.color, borderRadius: 99, transition: 'width 1s ease' }} />
              </div>
            </div>
          ))}
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 14px', borderRadius: 10, background: 'var(--cream-2)', border: '1px solid var(--stone)', marginTop: 4 }}>
            <span style={{ fontSize: 12, color: 'var(--ink-light)' }}>Structures Detected</span>
            <span style={{ fontFamily: 'DM Mono, monospace', fontWeight: 700, fontSize: 15, color: 'var(--ink)' }}>
              {geoStats?.detected_structures ?? '—'}
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 14px', borderRadius: 10, background: 'var(--cream-2)', border: '1px solid var(--stone)' }}>
            <span style={{ fontSize: 12, color: 'var(--ink-light)' }}>Edge Improvement Factor</span>
            <span style={{ fontFamily: 'DM Mono, monospace', fontWeight: 700, fontSize: 15, color: 'var(--accent)' }}>
              {geoStats?.edge_improvement ?? '—'}×
            </span>
          </div>
        </div>

        {/* Overlay image */}
        <div style={{ flex: 1, minHeight: 200 }}>
          <div className="card-inset" style={{ height: '100%', minHeight: 200, position: 'relative', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {overlayImage
              ? <img src={overlayImage} alt="Geo overlay" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              : <span style={{ fontSize: 12, color: 'var(--warm-mid)' }}>No overlay</span>
            }
            {overlayImage && (
              <div style={{ position: 'absolute', bottom: 8, left: 8, display: 'flex', flexDirection: 'column', gap: 3 }}>
                {[
                  { color: 'rgba(255,80,80,0.88)', label: '■ STRUCTURES' },
                  { color: 'rgba(34,197,94,0.88)', label: '■ HEALTHY VEG' },
                  { color: 'rgba(234,179,8,0.88)', label: '■ STRESSED' },
                  { color: 'rgba(59,130,246,0.88)', label: '■ WATER' },
                ].map((l) => (
                  <span key={l.label} style={{ fontSize: 9, padding: '2px 5px', background: l.color, color: '#fff', borderRadius: 4, fontFamily: 'DM Mono, monospace', fontWeight: 600 }}>
                    {l.label}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// CONFIDENCE PANEL
// ─────────────────────────────────────────────────────────────────────────────
const ConfidencePanel = ({ confidenceImage }: { confidenceImage: string }) => (
  <div className="card animate-fade-in" style={{ padding: 20 }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
      <div style={{ width: 30, height: 30, borderRadius: 8, background: '#f5f0ff', border: '1px solid #ddd6fe', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Eye style={{ width: 14, height: 14, color: '#7c3aed' }} />
      </div>
      <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--ink)' }}>Reconstruction Uncertainty</span>
      <span style={{ marginLeft: 'auto', fontSize: 9, color: 'var(--warm-mid)', fontFamily: 'DM Mono, monospace', letterSpacing: '0.04em' }}>SPATIAL UQ MAP</span>
    </div>

    <div className="card-inset" style={{ aspectRatio: '4/3', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', position: 'relative' }}>
      {confidenceImage
        ? <img src={confidenceImage} alt="Confidence map" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
        : <span style={{ fontSize: 12, color: 'var(--warm-mid)' }}>No data</span>
      }
      {confidenceImage && (
        <div style={{
          position: 'absolute', bottom: 8, right: 8,
          display: 'flex', gap: 10, padding: '5px 10px',
          background: 'rgba(255,255,255,0.9)', backdropFilter: 'blur(8px)',
          borderRadius: 8, border: '1px solid var(--stone)',
        }}>
          {[
            { col: '#fff8e1', label: 'High conf.' },
            { col: '#7b2d00', label: 'Low conf.' },
          ].map((l) => (
            <div key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <div style={{ width: 10, height: 10, borderRadius: 2, background: l.col, border: '1px solid rgba(0,0,0,0.1)' }} />
              <span style={{ fontSize: 10, color: 'var(--warm-dark)' }}>{l.label}</span>
            </div>
          ))}
        </div>
      )}
    </div>

    <div className="heatmap-legend" style={{ marginTop: 10 }} />
    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: 10, color: 'var(--warm-mid)' }}>
      <span>Yellow = preserved (confident)</span>
      <span>Dark = AI-reconstructed (verify)</span>
    </div>
    <p style={{ fontSize: 11, color: 'var(--warm-dark)', marginTop: 10, lineHeight: 1.6 }}>
      Bright regions closely match the input — low reconstruction uncertainty.
      Dark regions contain AI-inferred detail that was not observable in the original;
      cross-validate against ground truth before operational use.
    </p>
  </div>
);

// ─────────────────────────────────────────────────────────────────────────────
// EXPORT PANEL
// ─────────────────────────────────────────────────────────────────────────────
const ExportPanel = ({ results }: { results: any }) => {
  const sid = results?.session_id?.substring(0, 8) ?? '—';

  const exportJSON = () => {
    const payload = {
      session_id: results?.session_id,
      model:      results?.model_name,
      scale:      results?.scale_factor,
      scene:      results?.scene_label,
      metrics:    results?.metrics,
      geospatial: results?.geospatial,
      dimensions: results?.dimensions,
      processing_time_s: results?.processing_time,
      source_info: results?.source_info,
      generated: new Date().toISOString(),
      problem_statement: 'PS-26142 NTRO SIH 2024',
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `srm_report_${sid}.json`;
    a.click();
  };

  return (
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
        <p style={{ fontSize: 11, color: 'var(--warm-dark)' }}>
          {results?.processing_time}s · Session {sid}
        </p>
        {results?.source_info?.filename && (
          <p style={{ fontSize: 10, color: 'var(--warm-mid)', fontFamily: 'DM Mono, monospace', marginTop: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {results.source_info.filename}
          </p>
        )}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <button className="btn-primary" style={{ width: '100%', justifyContent: 'center' }}
          onClick={() => downloadDataUrl(results?.images?.output, `sr_output_${sid}.png`)}>
          <Download style={{ width: 14, height: 14 }} /> Download SR Image (PNG)
        </button>
        <button className="btn-secondary" style={{ width: '100%', justifyContent: 'center' }}
          onClick={() => downloadDataUrl(results?.images?.geo_overlay, `geo_overlay_${sid}.png`)}>
          <MapPin style={{ width: 14, height: 14 }} /> Download Geo Overlay (PNG)
        </button>
        <button className="btn-secondary" style={{ width: '100%', justifyContent: 'center' }}
          onClick={exportJSON}>
          <FileText style={{ width: 14, height: 14 }} /> Export Full Report (JSON)
        </button>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// MAIN APP
// ─────────────────────────────────────────────────────────────────────────────
type AppState = 'idle' | 'processing' | 'results' | 'error';

export default function App() {
  const [state, setState]       = useState<AppState>('idle');
  const [progress, setProgress] = useState(0);
  const [status, setStatus]     = useState('');
  const [filename, setFilename] = useState<string | null>(null);
  const [results, setResults]   = useState<any>(null);
  const [error, setError]       = useState('');
  const [tab, setTab]           = useState('comparison');
  const [warmPct, setWarmPct]   = useState<number | null>(null);
  const [modelReady, setModelReady] = useState(false);

  // Warm-up poll on mount
  useEffect(() => {
    let tries = 0;
    const poll = async () => {
      setWarmPct(Math.min(tries * 12, 90));
      const ok = await pingServer();
      if (ok) { setModelReady(true); setWarmPct(null); }
      else if (tries < 20) { tries++; setTimeout(poll, 3000); }
      else setWarmPct(null);
    };
    poll();
  }, []);

  const reset = () => { setState('idle'); setResults(null); setError(''); setProgress(0); setFilename(null); };

  const tick = (max: number, msg: (p: number) => string) =>
    setInterval(() => {
      setProgress((p) => { const np = Math.min(p + 4, max); setStatus(msg(np)); return np; });
    }, 300);

  const handleScene = async (scene: string) => {
    setState('processing'); setProgress(5);
    const msgs = (p: number) =>
      p < 25 ? 'Pre-processing Sentinel-2 scene...'
      : p < 55 ? 'Running A2N ×4 inference...'
      : p < 80 ? 'Computing PSNR / SSIM / SAM...'
      : 'Extracting land-cover features...';
    const pInt = tick(92, msgs);
    try {
      const res = await runDemo(scene);
      clearInterval(pInt); setProgress(100);
      if (!res.success) throw new Error(res.error || 'Unknown error');
      setResults(res);
      setTimeout(() => setState('results'), 400);
    } catch (e: any) { clearInterval(pInt); setError(e.message); setState('error'); }
  };

  const handleFile = async (file: File) => {
    setFilename(file.name); setState('processing'); setProgress(5);
    setStatus(`Uploading ${file.name}…`);
    try {
      const res = await uploadFile(file, (p) => { setProgress(p + 5); setStatus(`Uploading… ${p}%`); });
      setProgress(50); setStatus('Running A2N SR inference…');
      const pInt = tick(92, (p) => p < 75 ? 'Running SR inference…' : 'Computing metrics…');
      if (!res.success) throw new Error(res.error || 'Unknown error');
      clearInterval(pInt); setProgress(100);
      setResults(res);
      setTimeout(() => setState('results'), 400);
    } catch (e: any) { setError(e.message); setState('error'); }
  };

  const TABS = [
    { id: 'comparison', label: 'Visual Comparison' },
    { id: 'analysis',   label: 'Land-Cover Analysis' },
  ];

  return (
    <div style={{ minHeight: '100vh', position: 'relative', zIndex: 1, paddingBottom: 80 }}>
      {/* Globe background */}
      <div style={{ position: 'fixed', inset: 0, zIndex: 0, opacity: 1, pointerEvents: 'none', background: '#000' }}>
        <Globe />
      </div>

      <Header onReset={state !== 'idle' ? reset : null} modelReady={modelReady} />

      <main style={{ maxWidth: 1100, margin: '0 auto', padding: '32px 20px 0', position: 'relative', zIndex: 1 }}>

        {state === 'idle' && (
          <div className="animate-fade-in">
            <HeroSection />
            <SceneSelector onScene={handleScene} onFile={handleFile} />
          </div>
        )}

        {state === 'processing' && (
          <PipelineProgress progress={progress} status={status} filename={filename} />
        )}

        {state === 'error' && <ErrorCard message={error} onRetry={reset} />}

        {state === 'results' && results && (
          <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {/* Tab bar */}
            <div className="tab-bar">
              {TABS.map((t) => (
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
                <GeospatialPanel
                  geoStats={results.geospatial}
                  overlayImage={results.images?.geo_overlay}
                  sceneLabel={results.scene_label || ''}
                />
                <ConfidencePanel confidenceImage={results.images?.confidence} />
              </div>
            )}
          </div>
        )}
      </main>

      {warmPct !== null && <WarmupBanner pct={warmPct} />}
    </div>
  );
}
