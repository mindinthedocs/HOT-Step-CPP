// TrtBackendSection.tsx — "TensorRT Backend" section for the Model Manager
//
// Operator surface for building TRT bundles (partner §7):
//  - DiT component checkbox (LM / VAE / Embedded greyed "coming soon")
//  - variant dropdown (q8map-fp16 / w8a16 / fp32)
//  - Build / Rebuild button -> POST /api/trt-bundles/build
//  - build progress bar driven by the build SSE (TrtBuildProgressBar)
//  - error + Retry state (409 single-GPU lock surfaced here too)
//
// Reuses the build-job SSE hook (useTrtBuildStream, sibling of
// useDownloadStream) and the shared design tokens.

import React, { useEffect, useState, useCallback } from 'react';
import { Zap, RefreshCw, Hammer, AlertCircle } from 'lucide-react';
import { trtBundleApi, type TrtBundleEntry } from '../../services/api';
import { useTrtBuildStream } from './useTrtBuildStream';
import { TrtBuildProgressBar } from './TrtBuildProgressBar';

const VARIANTS = ['q8map-fp16', 'w8a16', 'fp32'] as const;

const COMPONENTS = [
  { id: 'dit', label: 'DiT', available: true },
  { id: 'lm', label: 'LM', available: false },
  { id: 'vae', label: 'VAE', available: false },
  { id: 'embedded', label: 'Embedded', available: false },
] as const;

export const TrtBackendSection: React.FC<{ onBundlesChanged?: () => void }> = ({ onBundlesChanged }) => {
  const [variant, setVariant] = useState<string>(VARIANTS[0]);
  const [bundles, setBundles] = useState<TrtBundleEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refreshBundles = useCallback(() => {
    trtBundleApi.list()
      .then((r) => setBundles(r.bundles))
      .catch(() => {});
  }, []);

  useEffect(() => { refreshBundles(); }, [refreshBundles]);

  const { job, isBuilding, startBuild, cancelBuild } = useTrtBuildStream({
    onComplete: () => { refreshBundles(); onBundlesChanged?.(); },
  });

  const handleBuild = useCallback(async () => {
    setError(null);
    try {
      await startBuild(variant);
    } catch (err: any) {
      setError(err?.message || 'Build failed to start');
    }
  }, [variant, startBuild]);

  const existing = bundles.find(b => b.variant === variant || b.name === `acestep-v15-2b-${variant}`);
  const bundleBuilt = existing?.available === true;
  const buildVerb = bundleBuilt ? 'Rebuild' : 'Build';
  const BuildIcon = bundleBuilt ? RefreshCw : Hammer;

  return (
    <div className="rounded-2xl border border-violet-500/20 bg-violet-500/5 p-4" data-testid="trt-backend-section">
      {/* Heading */}
      <h3 className="text-sm font-semibold text-violet-400 uppercase tracking-wider mb-1 flex items-center gap-2">
        <Zap size={14} />
        TensorRT Backend
      </h3>
      <p className="text-xs text-zinc-600 dark:text-zinc-500 mb-4">
        Build a TensorRT engine bundle for accelerated inference. One build runs at a time (single GPU).
      </p>

      {/* Component checkboxes */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        {COMPONENTS.map((c) => (
          <label
            key={c.id}
            className={`flex items-center gap-2 text-sm ${
              c.available ? 'text-zinc-700 dark:text-zinc-200 cursor-default'
                          : 'text-zinc-400 dark:text-zinc-600 cursor-not-allowed'
            }`}
            title={c.available ? undefined : 'Coming soon'}
          >
            <input
              type="checkbox"
              checked={c.available}
              disabled
              readOnly
              data-testid={`trt-component-${c.id}`}
              className="h-4 w-4 rounded border-zinc-400 dark:border-white/20 text-violet-500
                         accent-violet-500 disabled:opacity-60"
            />
            <span>{c.label}</span>
            {!c.available && (
              <span className="text-[9px] uppercase tracking-wide text-zinc-500 dark:text-zinc-600">coming soon</span>
            )}
          </label>
        ))}
      </div>

      {/* Variant dropdown + Build button */}
      <div className="flex items-end gap-3">
        <div className="flex-1 max-w-xs">
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
            DiT Variant
          </label>
          <select
            value={variant}
            onChange={(e) => setVariant(e.target.value)}
            disabled={isBuilding}
            data-testid="trt-variant-select"
            className="w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800
                       border border-zinc-300 dark:border-white/10
                       text-sm text-zinc-800 dark:text-zinc-200
                       hover:border-zinc-400 dark:hover:border-white/20
                       focus:border-violet-500/50 focus:ring-1 focus:ring-violet-500/20
                       outline-none transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {VARIANTS.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>

        <button
          onClick={handleBuild}
          disabled={isBuilding}
          data-testid="trt-build-button"
          className="px-4 py-2 rounded-xl bg-violet-500/15 border border-violet-500/30
                     text-sm font-medium text-violet-300 hover:bg-violet-500/25 hover:text-violet-200
                     transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <BuildIcon size={14} className={isBuilding ? 'animate-spin' : ''} />
          {isBuilding ? 'Building…' : buildVerb}
        </button>
      </div>

      {/* Error / Retry */}
      {error && (
        <div className="mt-3 rounded-xl bg-red-500/10 border border-red-500/20 px-3 py-2 flex items-center gap-2">
          <AlertCircle size={14} className="text-red-400 flex-shrink-0" />
          <span className="text-xs text-red-400 flex-1 truncate" data-testid="trt-build-error">{error}</span>
          <button
            onClick={handleBuild}
            data-testid="trt-build-retry"
            className="text-xs text-red-300 hover:text-red-200 underline underline-offset-2"
          >
            Retry
          </button>
        </div>
      )}

      {/* Build progress */}
      {job && (
        <div className="mt-3" data-testid="trt-build-progress">
          <TrtBuildProgressBar job={job} onCancel={cancelBuild} />
          {job.status === 'failed' && (
            <button
              onClick={handleBuild}
              data-testid="trt-build-retry-failed"
              className="mt-2 text-xs text-violet-300 hover:text-violet-200 underline underline-offset-2"
            >
              Retry build
            </button>
          )}
        </div>
      )}
    </div>
  );
};
