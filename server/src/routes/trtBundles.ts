// trtBundles.ts — TRT-bundle build-job + enumeration API
//
// GET    /api/trt-bundles                 — list variant inventory + engine /props trt array
// GET    /api/trt-bundles/variants        — available DiT build variants
// POST   /api/trt-bundles/build           — start a build { variant, precision } (409 if one running)
// GET    /api/trt-bundles/build/:id       — build job status
// GET    /api/trt-bundles/build/:id/events— SSE progress stream from the CLI
// POST   /api/trt-bundles/build/:id/cancel— cancel a running build
// DELETE /api/trt-bundles/:bundleName     — delete a built bundle directory

import { Router } from 'express';
import path from 'path';
import { trtBundleService, BuildLockedError } from '../services/trtBundleService.js';
import { aceClient } from '../services/aceClient.js';
import type { AcePropsWithTrt } from '../types/aceTrtBundle.js';
import { checkPrerequisites } from '../services/systemService.js';
import { isAllowedVariant, validateBuildRequest } from '../services/validateBuildRequest.js';
import { modelDownloadService } from '../services/modelDownloadService.js';
import { PROJECT_ROOT } from '../config.js';

const router = Router();

const PRECISIONS = ['q8map-fp16', 'w8a16', 'fp32'] as const;

function getTrtVariants() {
  return modelDownloadService
    .getTrtRegistry()
    .filter((variant) => variant.role === 'dit' || variant.role === 'dit-st');
}

function assertSafeVariantInput(raw: unknown): string {
  if (typeof raw !== 'string') throw new Error('variant is required');
  const variant = raw.trim();
  if (!variant) throw new Error('variant is required');
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(variant)) throw new Error('Invalid variant format');
  if (variant.includes('..') || variant.includes('/') || variant.includes('\\')) throw new Error('Path traversal denied');
  return variant;
}

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
  res.json({ variants: getTrtVariants() });
});

router.get('/active-build', (_req, res) => {
  const job = trtBundleService.getActiveBuild();
  res.json({ job });
});

// POST /api/trt-bundles/build  { variant, precision }
router.post('/build', async (req, res) => {
  const { variant: rawVariant, precision } = req.body ?? {};
  let variant: string;
  try {
    variant = assertSafeVariantInput(rawVariant);
  } catch (err: any) {
    res.status(400).json({ error: err.message });
    return;
  }
  const availableVariants = getTrtVariants();
  const allowedVariantIds = availableVariants.map((entry) => entry.id);
  if (!variant || !isAllowedVariant(variant) || !allowedVariantIds.includes(variant)) {
    res.status(400).json({ error: `variant is required; one of ${allowedVariantIds.join(', ')}` });
    return;
  }
  if (precision && !PRECISIONS.includes(precision)) {
    res.status(400).json({ error: `precision must be one of ${PRECISIONS.join(', ')}` });
    return;
  }

  const prereqs = await checkPrerequisites();
  const validation = validateBuildRequest({ variant, precision, prerequisites: prereqs });
  if (!validation.ok) {
    res.status(validation.status).json({ error: validation.error });
    return;
  }

  const variantMeta = availableVariants.find((entry) => entry.id === variant);
  if (!variantMeta) {
    res.status(400).json({ error: 'Unknown TRT variant' });
    return;
  }

  try {
    await modelDownloadService.ensureTrtVariantSources(variantMeta);
  } catch (err: any) {
    res.status(400).json({ error: err?.message || 'Failed to prepare TRT variant sources' });
    return;
  }

  const normalizedVariantToken = variant.replace(/[^A-Za-z0-9._-]/g, '-');
  const ditDir = path.join(PROJECT_ROOT, '.enc-build', 'trt-src', normalizedVariantToken, 'dit');
  const textEncoderDir = path.join(PROJECT_ROOT, '.enc-build', 'trt-src', normalizedVariantToken, 'qwen3-emb');

  try {
    const jobId = await trtBundleService.startBuild({
      variant,
      precision: (precision as string) ?? 'q8map-fp16',
      ditDir,
      textEncoderDir,
      sourceModel: variantMeta.id,
    });
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
  trtBundleService.cancelBuild(req.params.id)
    .then((ok) => res.json({ ok }))
    .catch((err: any) => res.status(400).json({ error: err?.message || 'Cancel failed' }));
});

router.get('/build/:id/log', (req, res) => {
  const text = trtBundleService.getBuildLog(req.params.id);
  if (text === null) {
    res.status(404).json({ error: 'Build log not found' });
    return;
  }
  res.type('text/plain').send(text);
});

router.post('/build/:id/open-log-folder', (req, res) => {
  const ok = trtBundleService.openBuildLogFolder(req.params.id);
  if (!ok) {
    res.status(404).json({ error: 'Log folder unavailable' });
    return;
  }
  res.json({ ok: true });
});

// DELETE /api/trt-bundles/:bundleName
router.delete('/:bundleName', async (req, res) => {
  try {
    const ok = trtBundleService.deleteBundle(req.params.bundleName);
    res.json({ ok });
  } catch (err: any) {
    res.status(400).json({ error: err.message });
  }
});

export default router;
