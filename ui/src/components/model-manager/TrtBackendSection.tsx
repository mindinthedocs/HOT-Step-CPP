// TrtBackendSection.tsx — "TensorRT Backend" section for the Model Manager
//
// Operator surface for building TRT bundles (partner §7):
//  - Tabs: DiT | LM | VAE | Embedding
//    - DiT tab: build a DiT bundle (dit + cond_enc + fsq). The text encoder is
//      NOT in the DiT bundle — the GGUF Qwen3 is the default at runtime.
//    - Embedding tab: build a standalone Qwen3-emb TRT bundle (text_enc only).
//      Independent of DiT bundles — own source dir, own bundle folder, own
//      build pipeline. When selected as the text encoder, the runtime routes
//      the text-encoder forward through the TRT engine.
//    - LM / VAE tabs: disabled (coming soon).
//  - variant dropdown (q8map-fp16 / w8a8 / fp32) — DiT tab only
//  - Build / Rebuild button -> POST /api/trt-bundles/build or /build-embedding
//  - build progress bar driven by the build SSE (TrtBuildProgressBar)
//  - error + Retry state (409 single-GPU lock surfaced here too)
//
// Reuses the build-job SSE hook (useTrtBuildStream, sibling of
// useDownloadStream) and the shared design tokens.

import React, { useEffect, useState, useCallback, useMemo } from 'react';
import { Zap, RefreshCw, Hammer, AlertCircle, Trash2 } from 'lucide-react';
import { trtBundleApi, type TrtBundleEntry, type TrtVariant, type SystemDependenciesResponse } from '../../services/api';
import { useTrtBuildStream } from './useTrtBuildStream';
import { TrtBuildProgressBar } from './TrtBuildProgressBar';
import { SystemDependencies } from './SystemDependencies';

const PRECISION_OPTIONS = ['q8map-fp16', 'w8a8'] as const;
const TAB_ITEMS = [
  { id: 'dit', label: 'DiT', disabled: false },
  { id: 'lm', label: 'LM', disabled: true },
  { id: 'vae', label: 'VAE', disabled: true },
  { id: 'embedding', label: 'Embedding', disabled: false },
] as const;
type TabId = (typeof TAB_ITEMS)[number]['id'];

export const TrtBackendSection: React.FC<{ onBundlesChanged?: () => void }> = ({ onBundlesChanged }) => {
  const [activeTab, setActiveTab] = useState<TabId>('dit');
  const [variant, setVariant] = useState<string>('');
  const [precision, setPrecision] = useState<string>('q8map-fp16');
  const [variantOptions, setVariantOptions] = useState<TrtVariant[]>([]);
  const [bundles, setBundles] = useState<TrtBundleEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [deps, setDeps] = useState<SystemDependenciesResponse | null>(null);

  const refreshBundles = useCallback(async () => {
    const r = await trtBundleApi.list();
    setBundles(r.bundles);
  }, []);

  useEffect(() => { refreshBundles().catch(() => {}); }, [refreshBundles]);

  const refreshVariants = useCallback(async () => {
    const r = await trtBundleApi.variants();
    setVariantOptions(r.variants);
  }, []);

  useEffect(() => { refreshVariants().catch(() => {}); }, [refreshVariants]);

  useEffect(() => {
    if (!variantOptions.length) return;
    if (!variantOptions.some((entry) => entry.id === variant)) {
      setVariant(variantOptions[0].id);
    }
  }, [variantOptions, variant]);

  const { job, isBuilding, startBuild, startEmbeddingBuild, cancelBuild } = useTrtBuildStream({
    onComplete: () => {
      refreshBundles().catch(() => {});
      onBundlesChanged?.();
    },
  });

  const handleRefresh = useCallback(async () => {
    setError(null);
    try {
      await Promise.all([onBundlesChanged?.(), refreshVariants(), refreshBundles()]);
    } catch {
      setError('Failed to refresh inventory');
    }
  }, [onBundlesChanged, refreshBundles, refreshVariants]);

  const handleDeleteBundle = useCallback(async (bundleName: string | undefined) => {
    if (!bundleName) return;
    setError(null);
    try {
      await trtBundleApi.deleteBundle(bundleName);
      await Promise.all([
        onBundlesChanged?.(),
        refreshBundles(),
      ]);
    } catch (err: any) {
      setError(err?.message || 'Failed to delete bundle');
    }
  }, [onBundlesChanged, refreshBundles]);

  const handleBuild = useCallback(async () => {
    setError(null);
    try {
      if (activeTab === 'embedding') {
        await startEmbeddingBuild();
      } else {
        if (!variant) {
          setError('Select a DiT variant before building.');
          return;
        }
        await startBuild(variant, precision);
      }
    } catch (err: any) {
      setError(err?.message || 'Build failed to start');
    }
  }, [activeTab, precision, variant, startBuild, startEmbeddingBuild]);

  const handleCopyLogs = useCallback(async () => {
    if (!job) return;
    setError(null);
    try {
      const text = await trtBundleApi.logText(job.jobId);
      await navigator.clipboard.writeText(text);
    } catch (err: any) {
      setError(err?.message || 'Failed to copy logs');
    }
  }, [job]);

  const handleOpenLogFolder = useCallback(async () => {
    if (!job) return;
    setError(null);
    try {
      await trtBundleApi.openLogFolder(job.jobId);
    } catch (err: any) {
      setError(err?.message || 'Failed to open log folder');
    }
  }, [job]);

  useEffect(() => {
    const cap = deps?.gpuCapability;
    if (precision === 'w8a8' && cap != null && cap < 7.5) {
      setPrecision('q8map-fp16');
    }
  }, [deps?.gpuCapability, precision]);

  // DiT tab: inventory row for the selected DiT variant.
  const ditInventoryRow = bundles.find((b) => b.variant === variant);
  const hasSafetensors = ditInventoryRow?.safetensorsComplete ?? false;
  const ditBundleBuilt = precision === 'w8a8'
    ? (ditInventoryRow?.engines.w8a8 ?? false)
    : (ditInventoryRow?.engines['q8map-fp16'] ?? false);
  // Embedding tab: the qwen3-emb bundle row (if it exists).
  const embBundleRow = bundles.find((b) => b.name === 'qwen3-emb' || b.variant === 'qwen3-emb');
  const embBundleBuilt = embBundleRow?.available ?? false;

  const isEmbeddingTab = activeTab === 'embedding';
  const buildVerb = isEmbeddingTab
    ? (embBundleBuilt ? 'Rebuild Engine' : 'Build Engine')
    : (ditBundleBuilt ? 'Rebuild Engine' : (hasSafetensors ? 'Build Engine' : 'Download and build'));
  const BuildIcon = (isEmbeddingTab ? embBundleBuilt : ditBundleBuilt) ? RefreshCw : Hammer;
  const w8a8Disabled = (deps?.gpuCapability ?? 99) < 7.5;
  const buildDisabled = isBuilding || (!isEmbeddingTab && !variant) || (deps?.freeDiskSpaceGb ?? 3) <= 2;

  const variantSelectOptions = useMemo(
    () => variantOptions.map((entry) => ({ value: entry.id, label: entry.displayName || entry.id })),
    [variantOptions],
  );

  const glyph = (ok: boolean) => (ok ? '✓' : '─');

  // Filter bundles for the inventory table based on the active tab.
  const tabBundles = isEmbeddingTab
    ? bundles.filter((b) => b.name === 'qwen3-emb' || b.variant === 'qwen3-emb')
    : bundles.filter((b) => b.name !== 'qwen3-emb' && b.variant !== 'qwen3-emb');

  return (
    <div className="rounded-2xl border border-violet-500/20 bg-violet-500/5 p-4" data-testid="trt-backend-section">
      {/* Heading */}
      <h3 className="text-sm font-semibold text-violet-400 uppercase tracking-wider mb-1 flex items-center gap-2">
        <Zap size={14} />
        TensorRT Backend
      </h3>
      <p className="text-xs text-zinc-600 dark:text-zinc-500 mb-4">
        Build TensorRT engine bundles for accelerated inference. One build runs at a time (single GPU).
      </p>

      <div className="mb-4 flex flex-wrap items-center gap-2">
        {TAB_ITEMS.map((tab) => {
          const isActive = tab.id === activeTab;
          return (
            <button
              key={tab.id}
              type="button"
              disabled={tab.disabled}
              onClick={() => !tab.disabled && setActiveTab(tab.id)}
              className={`rounded-lg border px-3 py-1.5 text-xs font-medium uppercase tracking-wide transition-colors ${
                isActive
                  ? 'border-violet-500/50 bg-violet-500/20 text-violet-200'
                  : tab.disabled
                    ? 'cursor-not-allowed border-zinc-500/30 bg-zinc-500/5 text-zinc-500'
                    : 'border-zinc-400/30 bg-zinc-500/10 text-zinc-300 cursor-pointer hover:border-violet-500/30'
              }`}
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      <SystemDependencies onDependenciesChanged={setDeps} />

      {/* Embedding tab description */}
      {isEmbeddingTab && (
        <p className="text-xs text-zinc-500 mb-3 mt-2">
          Build a standalone TRT Qwen3 text encoder bundle. Independent of DiT bundles —
          lives in <code className="text-zinc-400">models/trt-bundles/qwen3-emb/</code>.
          When selected as the text encoder, the runtime routes the text-encoder forward
          through the TRT engine.
        </p>
      )}

      {/* Variant dropdown + Build button — DiT tab only */}
      {!isEmbeddingTab && (
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
              {!variantSelectOptions.length && <option value="">Loading variants...</option>}
              {variantSelectOptions.map((entry) => (
                <option key={entry.value} value={entry.value}>{entry.label}</option>
              ))}
            </select>
          </div>
          <div className="w-40">
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
              Precision
            </label>
            <select
              value={precision}
              onChange={(e) => setPrecision(e.target.value)}
              disabled={isBuilding}
              data-testid="trt-precision-select"
              className="w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800
                         border border-zinc-300 dark:border-white/10
                         text-sm text-zinc-800 dark:text-zinc-200
                         hover:border-zinc-400 dark:hover:border-white/20
                         focus:border-violet-500/50 focus:ring-1 focus:ring-violet-500/20
                         outline-none transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {PRECISION_OPTIONS.map((option) => (
                <option key={option} value={option} disabled={option === 'w8a8' && w8a8Disabled}>
                  {option}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* Build + Refresh buttons — always visible */}
      <div className="flex items-end gap-3 mt-3">
        <button
          onClick={handleBuild}
          disabled={buildDisabled}
          data-testid="trt-build-button"
          className="px-4 py-2 rounded-xl bg-violet-500/15 border border-violet-500/30
                     text-sm font-medium text-violet-300 hover:bg-violet-500/25 hover:text-violet-200
                     transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <BuildIcon size={14} className={isBuilding ? 'animate-spin' : ''} />
          {isBuilding ? 'Building Engine…' : buildVerb}
        </button>
        <button
          type="button"
          onClick={handleRefresh}
          disabled={isBuilding}
          data-testid="trt-inventory-refresh"
          className="px-3 py-2 rounded-xl bg-zinc-500/10 border border-zinc-400/30
                     text-sm font-medium text-zinc-300 hover:bg-zinc-500/20 hover:text-zinc-200
                     transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <RefreshCw size={14} />
          ↻ Refresh
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
          <TrtBuildProgressBar
            job={job}
            onCancel={cancelBuild}
            onCopyLogs={handleCopyLogs}
            onOpenLogFolder={handleOpenLogFolder}
          />
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

      {/* Inventory table */}
      <div className="mt-4 rounded-xl border border-zinc-400/20 overflow-hidden" data-testid="trt-inventory-table">
        <table className="w-full text-xs">
          <thead className="bg-zinc-500/10 text-zinc-400 uppercase tracking-wider">
            <tr>
              <th className="text-left px-3 py-2 font-medium">
                {isEmbeddingTab ? 'Embedding Bundle' : 'DiT Bundle'}
              </th>
              <th className="text-center px-3 py-2 font-medium">Available</th>
              <th className="text-center px-3 py-2 font-medium">Delete</th>
            </tr>
          </thead>
          <tbody>
            {tabBundles.length === 0 && (
              <tr className="border-t border-zinc-500/10 text-zinc-500">
                <td className="px-3 py-2 italic" colSpan={3}>
                  {isEmbeddingTab
                    ? 'No Qwen3-emb bundle built yet. Click "Build Engine" to create one.'
                    : 'No DiT bundles built yet.'}
                </td>
              </tr>
            )}
            {tabBundles.map((row) => (
              <tr key={row.name} className="border-t border-zinc-500/10 text-zinc-300">
                <td className="px-3 py-2">
                  <div className="flex items-center gap-2">
                    <span>{row.variant ?? row.name}</span>
                  </div>
                </td>
                <td className="px-3 py-2 text-center">{glyph(row.available)}</td>
                <td className="px-3 py-2 text-center">
                  <button
                    type="button"
                    onClick={() => handleDeleteBundle(row.name)}
                    disabled={isBuilding || !row.name}
                    className="inline-flex items-center justify-center rounded-md border border-red-500/30 bg-red-500/10
                               p-1 text-red-300 hover:bg-red-500/20 disabled:opacity-40 disabled:cursor-not-allowed"
                    aria-label={`Delete bundle for ${row.name}`}
                    data-testid={`trt-delete-${row.name}`}
                  >
                    <Trash2 size={14} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
