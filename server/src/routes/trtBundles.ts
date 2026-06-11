// trtBundles.ts — TRT-bundle build-job + enumeration API
//
// GET    /api/trt-bundles                 — list bundles (on-disk + engine /props trt array)
// GET    /api/trt-bundles/variants        — available DiT build variants
// POST   /api/trt-bundles/build           — start a build { variant, precision } (409 if one running)
// GET    /api/trt-bundles/build/:id       — build job status
// GET    /api/trt-bundles/build/:id/events— SSE progress stream from the CLI
// POST   /api/trt-bundles/build/:id/cancel— cancel a running build
// DELETE /api/trt-bundles/:variant        — delete a built bundle directory

import { Router } from 'express';
import { trtBundleService, BuildLockedError } from '../services/trtBundleService.js';
import { aceClient } from '../services/aceClient.js';
import type { AcePropsWithTrt } from '../types/aceTrtBundle.js';

const router = Router();

// DiT precision recipes — mirrors trt_bundle_manager REGISTRY["dit"].precision_options.
const VARIANTS = ['q8map-fp16', 'w8a16', 'fp32'] as const;

// GET /api/trt-bundles — on-disk bundles merged with the engine's /props trt array
router.get('/', async (_req, res) => {
  const onDisk = trtBundleService.listBundles();
  let engineBundles: unknown[] = [];
  let aceServerDown = false;
  try {
    const props = await aceClient.props() as AcePropsWithTrt;
    engineBundles = props.trt ?? [];
  } catch {
    aceServerDown = true;
  }
  res.json({
    bundles: onDisk,
    engine: engineBundles, // raw /props trt array (the engine's own view)
    building: trtBundleService.isBuilding(),
    aceServerDown,
  });
});

// GET /api/trt-bundles/variants
router.get('/variants', (_req, res) => {
  res.json({ variants: VARIANTS });
});

// POST /api/trt-bundles/build  { variant, precision }
router.post('/build', (req, res) => {
  const { variant } = req.body ?? {};
  if (!variant || !VARIANTS.includes(variant)) {
    res.status(400).json({ error: `variant is required; one of ${VARIANTS.join(', ')}` });
    return;
  }
  try {
    const jobId = trtBundleService.startBuild(variant);
    res.json({ jobId });
  } catch (err: any) {
    if (err instanceof BuildLockedError) {
      res.status(409).json({ error: err.message });
      return;
    }
    res.status(400).json({ error: err.message });
  }
});

// GET /api/trt-bundles/build/:id — status
router.get('/build/:id', (req, res) => {
  const job = trtBundleService.getJob(req.params.id);
  if (!job) {
    res.status(404).json({ error: 'Unknown build job' });
    return;
  }
  res.json(job);
});

// GET /api/trt-bundles/build/:id/events — SSE progress stream
router.get('/build/:id/events', (req, res) => {
  const jobId = req.params.id;
  if (!trtBundleService.getJob(jobId)) {
    res.status(404).json({ error: 'Unknown build job' });
    return;
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  const send = () => {
    const job = trtBundleService.getJob(jobId);
    if (!job) return;
    res.write(`data: ${JSON.stringify(job)}\n\n`);
  };

  // Replay current state immediately (covers progress emitted before connect).
  send();

  const onProgress = () => {
    send();
    const job = trtBundleService.getJob(jobId);
    if (job && job.status !== 'running') {
      // Terminal state reached — flush once more and close the stream.
      trtBundleService.off('progress', onProgress);
      clearInterval(interval);
      res.end();
    }
  };
  trtBundleService.on('progress', onProgress);

  // Heartbeat in case a progress event is missed.
  const interval = setInterval(send, 1000);

  req.on('close', () => {
    trtBundleService.off('progress', onProgress);
    clearInterval(interval);
  });
});

// POST /api/trt-bundles/build/:id/cancel
router.post('/build/:id/cancel', (req, res) => {
  const ok = trtBundleService.cancelBuild(req.params.id);
  res.json({ ok });
});

// DELETE /api/trt-bundles/:variant
router.delete('/:variant', (req, res) => {
  try {
    const ok = trtBundleService.deleteBundle(req.params.variant);
    res.json({ ok });
  } catch (err: any) {
    res.status(400).json({ error: err.message });
  }
});

export default router;
