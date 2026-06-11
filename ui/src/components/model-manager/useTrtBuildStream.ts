// useTrtBuildStream.ts — SSE hook for real-time TRT-bundle build progress
//
// Sibling of useDownloadStream, adapted for the TRT build-job SSE API:
// the per-job SSE endpoint (/api/trt-bundles/build/:id/events) streams the
// full BuildJob object on each progress line and closes on a terminal state.
// One active build at a time (single-GPU lock), so this tracks a single job.

import { useState, useEffect, useCallback, useRef } from 'react';
import { trtBundleApi, type TrtBuildJob } from '../../services/api';

interface UseTrtBuildStreamOptions {
  /** Called when the tracked build transitions to 'completed'. */
  onComplete?: () => void;
}

export function useTrtBuildStream(opts?: UseTrtBuildStreamOptions) {
  const [job, setJob] = useState<TrtBuildJob | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);
  const onCompleteRef = useRef(opts?.onComplete);
  onCompleteRef.current = opts?.onComplete;

  // Open an SSE stream whenever a job id becomes active.
  useEffect(() => {
    if (!activeJobId) return;
    const es = new EventSource(trtBundleApi.eventsUrl(activeJobId));
    esRef.current = es;

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as TrtBuildJob;
        setJob(data);
        if (data.status === 'completed') onCompleteRef.current?.();
        if (data.status !== 'running') {
          es.close();
          esRef.current = null;
        }
      } catch { /* ignore parse errors */ }
    };

    es.onerror = () => {
      // The server closes the stream on terminal state; EventSource fires
      // onerror on that close. If the job already reached a terminal state we
      // keep the last snapshot; otherwise EventSource auto-reconnects.
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [activeJobId]);

  /** Start a build and begin streaming its progress. Surfaces 409/errors to the caller. */
  const startBuild = useCallback(async (variant: string, precision?: string) => {
    const { jobId } = await trtBundleApi.build(variant, precision);
    setJob({
      jobId, variant, bundleName: `acestep-v15-2b-${variant}`,
      status: 'running', progress: 0, step: 'starting', lines: [],
    });
    setActiveJobId(jobId);
    return jobId;
  }, []);

  const cancelBuild = useCallback(async () => {
    if (!activeJobId) return;
    await trtBundleApi.cancel(activeJobId);
  }, [activeJobId]);

  const clear = useCallback(() => {
    esRef.current?.close();
    esRef.current = null;
    setActiveJobId(null);
    setJob(null);
  }, []);

  const isBuilding = job?.status === 'running';

  return { job, isBuilding, startBuild, cancelBuild, clear };
}
