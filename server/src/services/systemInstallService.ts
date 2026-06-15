import { EventEmitter } from 'events';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { spawn, ChildProcess } from 'child_process';
import { randomUUID } from 'crypto';
import { PROJECT_ROOT } from '../config.js';
import { checkPrerequisites } from './systemService.js';

export type InstallJobStatus = 'running' | 'completed' | 'failed';

export interface InstallJobLine {
  stream: 'stdout' | 'stderr';
  raw: string;
  ts: string;
  parsed?: Record<string, unknown>;
}

export interface InstallJob {
  jobId: string;
  kind: 'install-trt';
  status: InstallJobStatus;
  startedAt: string;
  finishedAt?: string;
  exitCode?: number;
  error?: string;
  lines: InstallJobLine[];
}

interface InternalInstallJob extends InstallJob {
  child?: ChildProcess;
}

class SystemInstallService extends EventEmitter {
  private jobs = new Map<string, InternalInstallJob>();

  private get defaultTensorRtPath(): string {
    return path.join(os.homedir(), 'AppData', 'Local', 'NVIDIA', 'TensorRT');
  }

  private publicJob(job: InternalInstallJob): InstallJob {
    return {
      jobId: job.jobId,
      kind: job.kind,
      status: job.status,
      startedAt: job.startedAt,
      finishedAt: job.finishedAt,
      exitCode: job.exitCode,
      error: job.error,
      lines: job.lines,
    };
  }

  getInstallJob(jobId: string): InstallJob | undefined {
    const job = this.jobs.get(jobId);
    return job ? this.publicJob(job) : undefined;
  }

  startInstallTrt(): string {
    const scriptPath = path.join(PROJECT_ROOT, 'tools', 'install_trt.ps1');
    const jobId = randomUUID().slice(0, 8);
    const startedAt = new Date().toISOString();

    const job: InternalInstallJob = {
      jobId,
      kind: 'install-trt',
      status: 'running',
      startedAt,
      lines: [],
    };
    this.jobs.set(jobId, job);

    const child = spawn('powershell', [
      '-NoProfile',
      '-ExecutionPolicy', 'Bypass',
      '-File', scriptPath,
      '-InstallRoot', this.defaultTensorRtPath,
    ], {
      cwd: PROJECT_ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    });
    job.child = child;

    this.wireChild(job, child);
    this.emit('progress', jobId);
    return jobId;
  }

	async setupVenv(): Promise<{ ok: boolean }> {
	  const prereqs = await checkPrerequisites();
	  if (!prereqs.trt.installed) {
		const err = new Error('TensorRT is missing. Install TensorRT before setting up Python venv.');
		(err as any).statusCode = 400;
		throw err;
	  }

	  // The detector may return a path pointing inside bin/ or lib/ because
	  // marker files live there. The PowerShell script expects the TensorRT
	  // *root* so it can find the python/ subdirectory containing the wheel.
	  let resolvedTrtPath = prereqs.trt.path || this.defaultTensorRtPath;
	  const lowerPath = resolvedTrtPath.toLowerCase();
	  if (lowerPath.endsWith('\\bin') || lowerPath.endsWith('/bin')) {
		resolvedTrtPath = path.dirname(resolvedTrtPath);
	  }
	  if (lowerPath.endsWith('\\lib') || lowerPath.endsWith('/lib')) {
		resolvedTrtPath = path.dirname(resolvedTrtPath);
	  }

	  const scriptPath = path.join(PROJECT_ROOT, 'tools', 'setup_venv.ps1');
	  const exitCode = await this.runScript(scriptPath, resolvedTrtPath);
	  if (exitCode === 0) return { ok: true };
	  if (exitCode === 2) {
		const err = new Error('TensorRT is missing. Install TensorRT before setting up Python venv.');
		(err as any).statusCode = 400;
		throw err;
	  }
	  throw new Error(`setup_venv.ps1 failed with exit code ${exitCode}`);
	}

  private runScript(scriptPath: string, installRoot?: string): Promise<number> {
    const args = [
      '-NoProfile',
      '-ExecutionPolicy', 'Bypass',
      '-File', scriptPath,
    ];
    // Pass -InstallRoot only if explicitly provided; when omitted the
    // script runs its own Resolve-TensorRtPath auto-detection (mirrors
    // the Node.js getTensorRtPath() logic).
    if (installRoot) {
      args.push('-InstallRoot', installRoot);
    }

    return new Promise((resolve, reject) => {
      const child = spawn('powershell', args, {
        cwd: PROJECT_ROOT,
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true,
      });

      child.on('error', reject);
      child.on('exit', (code) => resolve(code ?? 1));
    });
  }

  private wireChild(job: InternalInstallJob, child: ChildProcess): void {
    let stdoutBuf = '';
    let stderrBuf = '';

    child.stdout?.on('data', (chunk: Buffer) => {
      stdoutBuf += chunk.toString();
      let nl: number;
      while ((nl = stdoutBuf.indexOf('\n')) >= 0) {
        const line = stdoutBuf.slice(0, nl).trim();
        stdoutBuf = stdoutBuf.slice(nl + 1);
        if (line) this.handleLine(job, 'stdout', line);
      }
    });

    child.stderr?.on('data', (chunk: Buffer) => {
      stderrBuf += chunk.toString();
      let nl: number;
      while ((nl = stderrBuf.indexOf('\n')) >= 0) {
        const line = stderrBuf.slice(0, nl).trim();
        stderrBuf = stderrBuf.slice(nl + 1);
        if (line) this.handleLine(job, 'stderr', line);
      }
    });

    child.on('error', (err) => {
      if (job.status === 'running') {
        job.status = 'failed';
        job.error = `Failed to spawn install script: ${err.message}`;
        job.finishedAt = new Date().toISOString();
      }
      this.emit('progress', job.jobId);
    });

    child.on('exit', (code) => {
      job.exitCode = code ?? undefined;
      job.finishedAt = new Date().toISOString();
      if (code === 0) {
        job.status = 'completed';
      } else {
        job.status = 'failed';
        if (!job.error) {
          job.error = `install_trt.ps1 failed with exit code ${code}`;
        }
      }
      this.emit('progress', job.jobId);
    });
  }

  private handleLine(job: InternalInstallJob, stream: 'stdout' | 'stderr', raw: string): void {
    let parsed: Record<string, unknown> | undefined;
    try {
      const value = JSON.parse(raw);
      if (value && typeof value === 'object') {
        parsed = value as Record<string, unknown>;
      }
    } catch {
      parsed = undefined;
    }

    const line: InstallJobLine = {
      stream,
      raw,
      ts: new Date().toISOString(),
      parsed,
    };
    job.lines.push(line);

    if (stream === 'stderr' && !job.error) {
      job.error = parsed && typeof parsed.message === 'string'
        ? parsed.message
        : raw;
    }

    this.emit('progress', job.jobId);
  }
}

export const systemInstallService = new SystemInstallService();
