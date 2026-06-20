// ModelSelect.tsx — Custom dropdown for model selection with format badges
//
// Replaces native <select> to allow rich rendering of options with
// GGUF/SafeTensors format indicators. Uses click-outside and keyboard
// navigation for accessibility.

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { ChevronDown, Check } from 'lucide-react';

/** Detect model format from the raw model name/path.
 *  ONNX detection: .onnx extension OR known ONNX directory name patterns
 *  (the C++ registry registers ONNX subdirectories by directory name).
 *  TRT detection: TRT-bundle directory names. Two patterns are recognized:
 *    - ``trt-<sourceModel>-<precision>``      (current scheme; e.g. trt-acestep-v15-xl-sft-w8a8)
 *    - ``acestep-v15-2b-<sourceModel>``        (legacy scheme; e.g. acestep-v15-2b-acestep-v15-xl-sft)
 *  The legacy ``2b`` tag was misleading (XL is 4B, not 2B) and collided with
 *  the LM size namespace, so new builds use the ``trt-`` prefix. Legacy
 *  bundles keep loading after the rename. */
export function getModelFormat(name: string): 'gguf' | 'safetensors' | 'onnx' | 'trt' {
  // TRT bundles: current scheme `trt-<sourceModel>-<precision>`.
  // Detected before the acestep-* safetensors fall-through so a built bundle
  // gets the TRT badge, not ST.
  if (/^trt-acestep-v15-/i.test(name)) return 'trt';
  // Legacy TRT bundles: `acestep-v15-2b-<variant>`. Kept for backward compat
  // so existing bundles built before the rename keep loading.
  if (/^acestep-v15-2b-/i.test(name)) return 'trt';
  if (/\.onnx$/i.test(name)) return 'onnx';
  if (/\.gguf$/i.test(name)) return 'gguf';
  // ONNX model directories: names like 'lm-4B', 'dit-xl', etc.
  // that don't have .gguf or .safetensors extensions
  // and don't match safetensors naming patterns (acestep-*, Qwen3-*, vae*, scragvae*)
  const lower = name.toLowerCase();
  if (/^lm-\d/i.test(name)) return 'onnx';
  if (/^dit-/i.test(name) && !lower.includes('acestep')) return 'onnx';
  if (/^vae-/i.test(name) && !lower.endsWith('.safetensors')) return 'onnx';
  if (/^text[_-]enc/i.test(name)) return 'onnx';
  return 'safetensors';
}

interface FormatBadgeProps {
  format: 'gguf' | 'safetensors' | 'onnx' | 'trt';
  compact?: boolean;
}

/** Tiny pill showing GGUF, ST, ONNX, or TRT format */
export const FormatBadge: React.FC<FormatBadgeProps> = ({ format, compact }) => {
  const colorClass = format === 'gguf'
    ? 'bg-sky-500/15 text-sky-400 ring-1 ring-sky-500/20'
    : format === 'onnx'
    ? 'bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/20'
    : format === 'trt'
    ? 'bg-violet-500/15 text-violet-400 ring-1 ring-violet-500/20'
    : 'bg-amber-500/15 text-amber-400 ring-1 ring-amber-500/20';
  const icon = format === 'gguf' ? '◆' : format === 'onnx' ? '⬡' : format === 'trt' ? '▲' : '◈';
  const label = format === 'gguf' ? (compact ? 'GG' : 'GGUF')
    : format === 'onnx' ? (compact ? 'OX' : 'ONNX')
    : format === 'trt' ? (compact ? 'RT' : 'TRT')
    : (compact ? 'ST' : 'ST');
  const title = format === 'gguf' ? 'GGUF quantized format'
    : format === 'onnx' ? 'ONNX TensorRT-accelerated format'
    : format === 'trt' ? 'TensorRT engine bundle'
    : 'SafeTensors native format';
  return (
    <span
      className={`inline-flex items-center gap-0.5 shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold leading-none tracking-wide uppercase ${colorClass}`}
      title={title}
    >
      <span className="text-[9px]">{icon}</span>
      {label}
    </span>
  );
};

interface ModelSelectProps {
  value: string;
  onChange: (v: string) => void;
  options: string[];
  formatLabel?: (name: string) => string;
  placeholder?: string;
  id?: string;
}

export const ModelSelect: React.FC<ModelSelectProps> = ({
  value,
  onChange,
  options,
  formatLabel = (n) => n,
  placeholder = 'Select model…',
  id,
}) => {
  const [open, setOpen] = useState(false);
  const [focusIdx, setFocusIdx] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Close on click outside
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  // Scroll focused item into view
  useEffect(() => {
    if (!open || focusIdx < 0 || !listRef.current) return;
    const el = listRef.current.children[focusIdx] as HTMLElement | undefined;
    el?.scrollIntoView({ block: 'nearest' });
  }, [focusIdx, open]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (!open) {
        if (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          setOpen(true);
          setFocusIdx(Math.max(0, options.indexOf(value)));
        }
        return;
      }
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setFocusIdx((i) => Math.min(i + 1, options.length - 1));
          break;
        case 'ArrowUp':
          e.preventDefault();
          setFocusIdx((i) => Math.max(i - 1, 0));
          break;
        case 'Enter':
        case ' ':
          e.preventDefault();
          if (focusIdx >= 0 && focusIdx < options.length) {
            onChange(options[focusIdx]);
            setOpen(false);
          }
          break;
        case 'Escape':
          e.preventDefault();
          setOpen(false);
          break;
      }
    },
    [open, focusIdx, options, value, onChange]
  );

  const selectedFormat = value ? getModelFormat(value) : null;

  return (
    <div ref={containerRef} className="relative" id={id}>
      {/* Trigger button */}
      <button
        type="button"
        onClick={() => {
          setOpen(!open);
          if (!open) setFocusIdx(Math.max(0, options.indexOf(value)));
        }}
        onKeyDown={handleKeyDown}
        className="w-full flex items-center gap-2 px-3 py-2 rounded-xl
                   bg-zinc-100 dark:bg-zinc-800
                   border border-zinc-300 dark:border-white/10
                   text-sm text-zinc-800 dark:text-zinc-200
                   hover:border-zinc-400 dark:hover:border-white/20
                   focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20
                   outline-none transition-colors cursor-pointer"
      >
        {value ? (
          <>
            {selectedFormat && <FormatBadge format={selectedFormat} compact />}
            <span className="truncate flex-1 text-left">{formatLabel(value)}</span>
          </>
        ) : (
          <span className="truncate flex-1 text-left text-zinc-400">{placeholder}</span>
        )}
        <ChevronDown
          size={14}
          className={`shrink-0 text-zinc-400 transition-transform duration-150 ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {/* Dropdown list */}
      {open && options.length > 0 && (
        <div
          ref={listRef}
          className="absolute z-50 mt-1 w-full max-h-60 overflow-auto rounded-xl
                     bg-white dark:bg-zinc-800
                     border border-zinc-200 dark:border-white/10
                     shadow-lg shadow-black/20
                     py-1"
          role="listbox"
        >
          {options.map((opt, i) => {
            const fmt = getModelFormat(opt);
            const selected = opt === value;
            const focused = i === focusIdx;
            return (
              <button
                key={opt}
                type="button"
                role="option"
                aria-selected={selected}
                onClick={() => {
                  onChange(opt);
                  setOpen(false);
                }}
                onMouseEnter={() => setFocusIdx(i)}
                className={`w-full flex items-center gap-2 px-3 py-2 text-sm text-left transition-colors
                  ${focused ? 'bg-pink-500/10 dark:bg-pink-500/15' : ''}
                  ${selected ? 'text-pink-400' : 'text-zinc-700 dark:text-zinc-200'}
                  hover:bg-pink-500/10 dark:hover:bg-pink-500/15`}
              >
                <FormatBadge format={fmt} />
                <span className="truncate flex-1">{formatLabel(opt)}</span>
                {selected && <Check size={14} className="shrink-0 text-pink-400" />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};
