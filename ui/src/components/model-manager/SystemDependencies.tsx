import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { RefreshCw, ExternalLink } from 'lucide-react';
import { systemApi, type SystemDependenciesResponse, type SystemInstallJob } from '../../services/api';

const CUDA_DOWNLOAD_URL = 'https://developer.nvidia.com/cuda-downloads';

type PillTone = 'green' | 'yellow' | 'red';

function Pill({
  label,
  tone,
}: {
  label: string;
  tone: PillTone;
}) {
  const toneClass = tone === 'green'
    ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300'
    : tone === 'yellow'
      ? 'bg-amber-500/15 border-amber-500/30 text-amber-300'
      : 'bg-red-500/15 border-red-500/30 text-red-300';

  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-medium ${toneClass}`}>
      {label}
    </span>
  );
}

function DependencyRow({
  title,
  detail,
  pill,
  action,
}: {
  title: string;
  detail: string;
  pill: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-xl border border-violet-500/10 bg-zinc-100/60 dark:bg-zinc-900/30 px-3 py-2.5">
      <div>
        <div className="text-xs font-medium text-zinc-700 dark:text-zinc-200">{title}</div>
        <div className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-0.5">{detail}</div>
      </div>
      <div className="flex items-center gap-2">
        {pill}
        {action}
      </div>
    </div>
  );
}

export const SystemDependencies: React.FC<{
  onDependenciesChanged?: (deps: SystemDependenciesResponse | null) => void;
}> = ({ onDependenciesChanged }) => {
  const [deps, setDeps] = useState<SystemDependenciesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [installingTrt, setInstallingTrt] = useState(false);
  const [settingUpVenv, setSettingUpVenv] = useState(false);
  const [installError, setInstallError] = useState<string | null>(null);
  const [lastInstallStatus, setLastInstallStatus] = useState<'completed' | 'failed' | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const next = await systemApi.getDependencies();
      setDeps(next);
      onDependenciesChanged?.(next);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to load system prerequisites';
      setError(message);
      onDependenciesChanged?.(null);
    } finally {
      setLoading(false);
    }
  }, [onDependenciesChanged]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const runInstallTrt = useCallback(async () => {
    setInstallError(null);
    setLastInstallStatus(null);
    setInstallingTrt(true);

    let es: EventSource | null = null;
    try {
      const { jobId } = await systemApi.installTrt();
      await new Promise<void>((resolve, reject) => {
        es = new EventSource(systemApi.installTrtEvents(jobId));
        es.onmessage = (event) => {
          try {
            const job = JSON.parse(event.data) as SystemInstallJob;
            if (job.status === 'completed') {
              setLastInstallStatus('completed');
              es?.close();
              resolve();
            } else if (job.status === 'failed') {
              setLastInstallStatus('failed');
              const parsedError = job.lines.slice().reverse().find((line) => line.stream === 'stderr')?.parsed;
              const message = typeof parsedError?.message === 'string'
                ? parsedError.message
                : (job.error || `Install failed with exit code ${job.exitCode ?? 'unknown'}`);
              setInstallError(message);
              es?.close();
              reject(new Error(message));
            }
          } catch (parseErr) {
            reject(parseErr instanceof Error ? parseErr : new Error('Failed to parse install status'));
          }
        };
        es.onerror = () => {
          // Ignore transient EventSource reconnect attempts while running.
        };
      });
      await refresh();
      setInstallError(null);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to install TensorRT';
      setInstallError(message);
      setLastInstallStatus('failed');
    } finally {
      if (es !== null) es.close();
      setInstallingTrt(false);
    }
  }, [refresh]);

  const runSetupVenv = useCallback(async () => {
    setInstallError(null);
    setSettingUpVenv(true);
    try {
      await systemApi.setupVenv();
      await refresh();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to setup Python venv';
      setInstallError(message);
      setLastInstallStatus('failed');
    } finally {
      setSettingUpVenv(false);
    }
  }, [refresh]);

  const rows = useMemo(() => {
    if (!deps) return [];

    const diskTone: PillTone = deps.freeDiskSpaceGb <= 2 ? 'red' : deps.freeDiskSpaceGb < 50 ? 'yellow' : 'green';
    const diskLabel = deps.freeDiskSpaceGb <= 2
      ? `✗ ${deps.freeDiskSpaceGb} GB Insufficient`
      : deps.freeDiskSpaceGb < 50
        ? `! ${deps.freeDiskSpaceGb} GB Low Space`
        : `✓ ${deps.freeDiskSpaceGb} GB Available`;
    const ramTone: PillTone = deps.totalRamGb < 16 ? 'yellow' : 'green';
    const ramLabel = deps.totalRamGb < 16 ? `! ${deps.totalRamGb} GB` : `✓ ${deps.totalRamGb} GB`;
    const vramTone: PillTone = deps.gpuVramMb === null ? 'yellow' : deps.gpuVramMb < 8192 ? 'red' : 'green';
    const vramLabel = deps.gpuVramMb === null
      ? '! Unknown'
      : deps.gpuVramMb < 8192
        ? `✗ ${Math.round(deps.gpuVramMb / 1024)} GB`
        : `✓ ${Math.round(deps.gpuVramMb / 1024)} GB`;
    const gpuTone: PillTone = deps.gpuCapability !== null && deps.gpuCapability >= 7.5 ? 'green' : 'red';
    const gpuLabel = deps.gpuCapability !== null && deps.gpuCapability >= 7.5
      ? `✓ Compatible (${deps.gpuCapabilityLabel ?? 'sm_??'})`
      : `✗ Incompatible (${deps.gpuCapabilityLabel ?? 'unknown'})`;

    return [
      {
        title: 'CUDA 13.x',
        detail: deps.cuda.version ? `Detected version ${deps.cuda.version}` : 'CUDA toolkit not detected',
        pill: deps.cuda.installed
          ? <Pill tone="green" label="✓ Installed" />
          : <Pill tone="red" label="✗ Missing" />,
        action: !deps.cuda.installed ? (
          <a
            href={CUDA_DOWNLOAD_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 rounded-lg border border-violet-500/30 px-2 py-1 text-[11px] text-violet-300 hover:bg-violet-500/15"
          >
            Download CUDA Toolkit
            <ExternalLink size={12} />
          </a>
        ) : undefined,
      },
      {
        title: 'TensorRT 11.x',
        detail: deps.trt.path ? deps.trt.path : 'Default path: %LOCALAPPDATA%\\NVIDIA\\TensorRT',
        pill: deps.trt.installed
          ? <Pill tone="green" label="✓ Installed" />
          : <Pill tone="red" label="✗ Missing" />,
        action: !deps.trt.installed ? (
          <button
            type="button"
            onClick={runInstallTrt}
            disabled={installingTrt || settingUpVenv}
            className="rounded-lg border border-violet-500/30 px-2 py-1 text-[11px] text-violet-300 hover:bg-violet-500/15 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {installingTrt ? 'Installing TensorRT…' : 'Install TensorRT 11'}
          </button>
        ) : undefined,
      },
      {
        title: 'Python 3.12 venv',
        detail: '.venv runtime for TRT build pipeline',
        pill: deps.venv.installed
          ? <Pill tone="green" label="✓ Installed" />
          : <Pill tone="red" label="✗ Missing" />,
        action: !deps.venv.installed ? (
          <button
            type="button"
            onClick={runSetupVenv}
            disabled={!deps.trt.installed || installingTrt || settingUpVenv}
            className="rounded-lg border border-violet-500/30 px-2 py-1 text-[11px] text-violet-300 hover:bg-violet-500/15 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {settingUpVenv ? 'Setting up venv…' : 'Setup Python Venv'}
          </button>
        ) : undefined,
      },
      {
        title: 'GPU Arch',
        detail: 'Compute capability required for w8a8: sm_75+',
        pill: <Pill tone={gpuTone} label={gpuLabel} />,
      },
      {
        title: 'Available Disk Space',
        detail: 'Builds are blocked at 2 GB or less',
        pill: <Pill tone={diskTone} label={diskLabel} />,
      },
      {
        title: 'System RAM',
        detail: 'Recommended: 16 GB base, 32 GB XL',
        pill: <Pill tone={ramTone} label={ramLabel} />,
      },
      {
        title: 'GPU VRAM',
        detail: 'Required: 8 GB base/sft, 12 GB XL',
        pill: <Pill tone={vramTone} label={vramLabel} />,
      },
    ];
  }, [deps]);

  return (
    <div className="rounded-2xl border border-violet-500/20 bg-violet-500/5 p-4 mb-4" data-testid="system-dependencies">
      <div className="flex items-center justify-between gap-2 mb-3">
        <h4 className="text-xs font-semibold text-violet-300 uppercase tracking-wider">System Prerequisites</h4>
        <button
          type="button"
          onClick={refresh}
          disabled={loading}
          className="inline-flex items-center gap-1 rounded-lg border border-violet-500/30 px-2.5 py-1 text-[11px] text-violet-300 hover:bg-violet-500/15 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
          ↻ Refresh
        </button>
      </div>

      {error && (
        <div className="mb-3 rounded-xl bg-red-500/10 border border-red-500/20 px-3 py-2 text-xs text-red-300">
          {error}
        </div>
      )}
      {installError && (
        <div className="mb-3 rounded-xl bg-red-500/10 border border-red-500/20 px-3 py-2 text-xs text-red-300">
          ❌ Last attempt failed: {installError}
        </div>
      )}
      {lastInstallStatus === 'completed' && (
        <div className="mb-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 px-3 py-2 text-xs text-emerald-300">
          ✓ Install step completed successfully.
        </div>
      )}

      <div className="space-y-2">
        {rows.map((row) => (
          <DependencyRow
            key={row.title}
            title={row.title}
            detail={row.detail}
            pill={row.pill}
            action={row.action}
          />
        ))}
      </div>
    </div>
  );
};
