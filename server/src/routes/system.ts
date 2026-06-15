import { Router } from 'express';
import { checkPrerequisites } from '../services/systemService.js';
import { systemInstallService } from '../services/systemInstallService.js';

const router = Router();

router.get('/dependencies', async (_req, res) => {
  try {
    const prereqs = await checkPrerequisites();
    res.json({
      cuda: prereqs.cuda,
      trt: prereqs.trt,
      venv: prereqs.venv,
      gpuCapability: prereqs.gpuCapability,
      gpuCapabilityLabel: prereqs.gpuCapabilityLabel,
      freeDiskSpaceGb: prereqs.freeDiskSpaceGb,
      totalRamGb: prereqs.totalRamGb,
      gpuVramMb: prereqs.gpuVramMb,
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Failed to inspect dependencies';
    res.status(500).json({ error: message });
  }
});

router.post('/install-trt', (_req, res) => {
  try {
    const jobId = systemInstallService.startInstallTrt();
    res.json({ jobId });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Failed to start TensorRT installation';
    res.status(500).json({ error: message });
  }
});

router.get('/install-trt/:id/events', (req, res) => {
  const jobId = req.params.id;
  const job = systemInstallService.getInstallJob(jobId);
  if (!job) {
    res.status(404).json({ error: 'Install job not found' });
    return;
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  const sendUpdate = () => {
    const latest = systemInstallService.getInstallJob(jobId);
    if (!latest) {
      res.write(`data: ${JSON.stringify({ error: 'Install job not found' })}\n\n`);
      res.end();
      return;
    }
    res.write(`data: ${JSON.stringify(latest)}\n\n`);
    if (latest.status !== 'running') {
      res.end();
    }
  };

  sendUpdate();
  const onProgress = (updatedJobId: string) => {
    if (updatedJobId === jobId) sendUpdate();
  };
  systemInstallService.on('progress', onProgress);

  const interval = setInterval(sendUpdate, 1000);
  req.on('close', () => {
    systemInstallService.off('progress', onProgress);
    clearInterval(interval);
  });
});

router.post('/setup-venv', async (_req, res) => {
  try {
    const result = await systemInstallService.setupVenv();
    res.json(result);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Failed to setup Python venv';
    const statusCode = typeof (error as any)?.statusCode === 'number' ? (error as any).statusCode : 500;
    res.status(statusCode).json({ error: message });
  }
});

export default router;
