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

  useEffect(() => {
    let mounted = true;
    trtBundleApi.activeBuild()
      .then(({ job: active }) => {
        if (!mounted || !active) return;
        setJob(active);
        if (active.status === 'running') {
          setActiveJobId(active.jobId);
        }
      })
      .catch(() => {});
    return () => { mounted = false; };
  }, []);

  /** Start a DiT bundle build and begin streaming its progress. Surfaces 409/errors to the caller. */
  const startBuild = useCallback(async (variant: string, precision?: string) => {
    const { jobId } = await trtBundleApi.build(variant, precision);
    // Bundle directory naming mirrors the server's TrtBundleService.bundleNameFor:
    // ``trt-<sourceModel>-<precision>``. The legacy ``acestep-v15-2b-<variant>``
    // scheme was dropped because the hardcoded ``2b`` was misleading (XL is 4B,
    // not 2B) and collided with the LM size namespace. The legacy pattern is
    // still recognized by ModelSelect.tsx / modelLabels.ts for existing bundles.
    const p = precision ?? 'q8map-fp16';
    setJob({
      jobId, variant, bundleName: `trt-${variant}-${p}`,
      status: 'running', progress: 0, step: 'starting', lines: [],
    });
    setActiveJobId(jobId);
    return jobId;
  }, []);

  /** Start a standalone Qwen3-emb TRT bundle build (independent of DiT bundles). */
  const startEmbeddingBuild = useCallback(async () => {
    const { jobId } = await trtBundleApi.buildEmbedding();
    setJob({
      jobId, variant: 'qwen3-emb', bundleName: 'qwen3-emb',
      status: 'running', progress: 0, step: 'starting', lines: [],
    });
    setActiveJobId(jobId);
    return jobId;
  }, []);

  const cancelBuild = useCallback(async () => {
    if (!activeJobId) return;
    await trtBundleApi.cancel(activeJobId);
    setActiveJobId(null);
  }, [activeJobId]);

  const clear = useCallback(() => {
    esRef.current?.close();
    esRef.current = null;
    setActiveJobId(null);
    setJob(null);
  }, []);

  const isBuilding = job?.status === 'running';

  return { job, isBuilding, startBuild, startEmbeddingBuild, cancelBuild, clear };
}
