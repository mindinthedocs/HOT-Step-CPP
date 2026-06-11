// TrtBuildProgressBar.tsx — Build progress bar for TRT-bundle builds
//
// Adapted from DownloadProgressBar (same visual grammar — status pill, gradient
// fill, stats row), driven by the TRT build-job shape (progress 0..1 + step
// name + status) instead of byte counts.

import React from 'react';
import { X } from 'lucide-react';
import type { TrtBuildJob } from '../../services/api';

interface Props {
  job: TrtBuildJob;
  onCancel: () => void;
}

const statusColors: Record<string, string> = {
  running: 'bg-violet-500/20 text-violet-400',
  completed: 'bg-emerald-500/20 text-emerald-400',
  failed: 'bg-red-500/20 text-red-400',
  cancelled: 'bg-zinc-600 text-zinc-600 dark:text-zinc-400',
};

export const TrtBuildProgressBar: React.FC<Props> = ({ job, onCancel }) => {
  const pct = Math.min(100, Math.max(0, job.progress * 100));
  const isActive = job.status === 'running';

  return (
    <div className="rounded-xl border border-zinc-200 dark:border-white/5 bg-zinc-100/50 dark:bg-zinc-800/50 p-3">
      {/* Header row */}
      <div className="flex items-center gap-2 mb-1.5">
        <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${statusColors[job.status] || statusColors.running}`}>
          {job.status === 'running' ? `${pct.toFixed(0)}%` : job.status.toUpperCase()}
        </span>
        <span className="text-sm text-zinc-700 dark:text-zinc-300 font-medium truncate flex-1">
          {job.bundleName}
        </span>
        {isActive && (
          <button onClick={onCancel} title="Cancel build"
            className="p-1 rounded-lg hover:bg-white/10 text-zinc-500 hover:text-red-400 transition-colors flex-shrink-0">
            <X size={12} />
          </button>
        )}
      </div>

      {/* Progress bar */}
      <div className="h-1.5 rounded-full bg-zinc-200 dark:bg-zinc-700/50 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${
            isActive ? 'bg-gradient-to-r from-violet-500 to-violet-400' :
            job.status === 'completed' ? 'bg-emerald-500' :
            job.status === 'failed' ? 'bg-red-500' :
            'bg-zinc-600'
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Stats row */}
      <div className="flex items-center justify-between mt-1 text-[10px] text-zinc-500">
        <span className="truncate">
          {isActive ? `Step: ${job.step}` : job.status === 'completed' ? 'Bundle ready' : job.step}
        </span>
        {job.status === 'failed' && job.error && (
          <span className="text-red-400 truncate ml-2">{job.error}</span>
        )}
      </div>
    </div>
  );
};
