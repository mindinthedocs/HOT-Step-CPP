// trtBundleService.ts — Transient TRT-bundle build-job store
//
// Sibling of modelDownloadService: a single-GPU-locked build-job manager that
// spawns the Python TRT-bundle orchestrator (tools/trt/trt_bundle_cli.py bundle)
// and streams its stdout JSON progress protocol via an EventEmitter (SSE).
//
// Only one build runs at a time (single GPU). A second start while one is
// active is rejected by the route with 409.
//
// CLI stdout protocol (one JSON object per line):
//   {"step": N, "name": "...", "progress": 0..1, "status": "skipped|running|done|complete|dry-run"}
// CLI stderr error protocol:
//   {"error": true, "step": "...", "message": "..."}

import { EventEmitter } from 'events';
import fs from 'fs';
import path from 'path';
import { spawn, ChildProcess } from 'child_process';
import { randomUUID } from 'crypto';
import { config, PROJECT_ROOT } from '../config.js';

// ── Types ───────────────────────────────────────────────────

export type BuildStatus = 'running' | 'completed' | 'failed' | 'cancelled';

/** One progress line parsed from the CLI's stdout JSON protocol. */
export interface BuildProgress {
  step: number;
  name: string;
  progress: number; // 0..1
  status: string;   // skipped | running | done | complete | dry-run
}

export interface BuildJob {
  jobId: string;
  variant: string;
  bundleName: string;
  status: BuildStatus;
  progress: number;            // 0..1, latest cumulative
  step: string;                // latest step name
  lines: BuildProgress[];      // full progress history (for replay on SSE connect)
  error?: string;
  exitCode?: number;
}

interface InternalJob extends BuildJob {
  child?: ChildProcess;
}

// ── Service ─────────────────────────────────────────────────

class TrtBundleService extends EventEmitter {
  private jobs = new Map<string, InternalJob>();
  /** Single-GPU lock: the jobId of the currently running build, if any. */
  private activeJobId: string | null = null;

  /** Directory holding built TRT bundles. */
  get bundlesDir(): string {
    return path.join(config.aceServer.models, 'trt-bundles');
  }

  /** Resolve the repo's venv python (the CLI's interpreter). */
  private get pythonExe(): string {
    const winPy = path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe');
    const nixPy = path.join(PROJECT_ROOT, '.venv', 'bin', 'python');
    if (fs.existsSync(winPy)) return winPy;
    if (fs.existsSync(nixPy)) return nixPy;
    return process.platform === 'win32' ? winPy : nixPy;
  }

  private get cliScript(): string {
    return path.join(PROJECT_ROOT, 'tools', 'trt', 'trt_bundle_cli.py');
  }

  /** True when a build is currently running (single-GPU lock held). */
  isBuilding(): boolean {
    return this.activeJobId !== null;
  }

  /** List built bundles on disk with completeness status (manifest + engines present). */
  listBundles(): { name: string; available: boolean; variant?: string }[] {
    const dir = this.bundlesDir;
    if (!fs.existsSync(dir)) return [];
    const out: { name: string; available: boolean; variant?: string }[] = [];
    for (const name of fs.readdirSync(dir)) {
      const bundleDir = path.join(dir, name);
      let isDir = false;
      try { isDir = fs.statSync(bundleDir).isDirectory(); } catch { continue; }
      if (!isDir) continue;

      const manifestPath = path.join(bundleDir, 'manifest.json');
      let available = false;
      let variant: string | undefined;
      if (fs.existsSync(manifestPath)) {
        try {
          const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
          variant = typeof manifest.variant === 'string' ? manifest.variant : undefined;
          // "available" = manifest parses AND every declared engine exists on disk.
          available = this.manifestEnginesPresent(bundleDir, manifest);
        } catch {
          available = false;
        }
      }
      out.push({ name, available, variant });
    }
    return out;
  }

  /** Verify every engine/sidecar a manifest declares exists relative to the bundle root. */
  private manifestEnginesPresent(bundleDir: string, manifest: any): boolean {
    const components = manifest?.components;
    if (!components || typeof components !== 'object') return false;
    for (const comp of Object.values<any>(components)) {
      const rel = comp?.engine ?? comp?.sidecar;
      if (!rel) continue;
      if (!fs.existsSync(path.join(bundleDir, rel))) return false;
    }
    return true;
  }

  /** Bundle directory name for a variant (mirrors the build output-dir convention). */
  private bundleNameFor(variant: string): string {
    return `acestep-v15-2b-${variant}`;
  }

  /** Snapshot of all jobs (newest internal state, no child handle). */
  getJobs(): BuildJob[] {
    return Array.from(this.jobs.values()).map(j => this.publicJob(j));
  }

  getJob(jobId: string): BuildJob | undefined {
    const j = this.jobs.get(jobId);
    return j ? this.publicJob(j) : undefined;
  }

  private publicJob(j: InternalJob): BuildJob {
    return {
      jobId: j.jobId,
      variant: j.variant,
      bundleName: j.bundleName,
      status: j.status,
      progress: j.progress,
      step: j.step,
      lines: j.lines,
      error: j.error,
      exitCode: j.exitCode,
    };
  }

  /** Start a build job. Throws if a build is already running (caller maps to 409). */
  startBuild(variant: string): string {
    if (this.activeJobId !== null) {
      throw new BuildLockedError('A TRT build is already running (single-GPU-locked)');
    }

    const bundleName = this.bundleNameFor(variant);
    const outputDir = path.join('models', 'trt-bundles', bundleName); // CLI runs cwd=PROJECT_ROOT
    const ditDir = path.join(PROJECT_ROOT, '.enc-build', 'dit-fp32');
    const textEncDir = path.join(PROJECT_ROOT, '.enc-build', 'qwen3-emb');
    const gguf = path.join('models', 'acestep-v15-sft-BF16.gguf');

    const argv = [
      this.cliScript, 'bundle',
      '--variant', variant,
      '--output-dir', outputDir,
      '--dit-dir', ditDir,
      '--text-encoder-dir', textEncDir,
      '--gguf', gguf,
    ];

    const jobId = randomUUID().slice(0, 8);
    const job: InternalJob = {
      jobId,
      variant,
      bundleName,
      status: 'running',
      progress: 0,
      step: 'starting',
      lines: [],
    };
    this.jobs.set(jobId, job);
    this.activeJobId = jobId;

    const child = spawn(this.pythonExe, argv, {
      cwd: PROJECT_ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    job.child = child;

    this._wireChild(job, child);
    this.emit('progress');
    return jobId;
  }

  /** Cancel a running build. Returns false if the job is unknown or already settled. */
  cancelBuild(jobId: string): boolean {
    const job = this.jobs.get(jobId);
    if (!job || job.status !== 'running') return false;

    job.status = 'cancelled';
    if (job.child && job.child.pid && !job.child.killed) {
      // Tree-kill on Windows so child python + any TRT subprocess die together.
      try {
        if (process.platform === 'win32') {
          spawn('taskkill', ['/PID', String(job.child.pid), '/T', '/F'], { stdio: 'ignore' });
        } else {
          job.child.kill('SIGTERM');
        }
      } catch { /* process may already be gone */ }
    }
    if (this.activeJobId === jobId) this.activeJobId = null;
    this.emit('progress');
    return true;
  }

  /** Delete a built bundle directory by variant (or bundle name). */
  deleteBundle(variant: string): boolean {
    // Accept either the variant (acestep-v15-2b-<variant>) or a literal bundle name.
    const candidates = [this.bundleNameFor(variant), variant];
    const dir = this.bundlesDir;
    const dirResolved = path.resolve(dir);
    for (const name of candidates) {
      const target = path.join(dir, name);
      const resolved = path.resolve(target);
      // Path-traversal guard: must stay inside the bundles dir.
      if (resolved !== dirResolved && !resolved.startsWith(dirResolved + path.sep)) {
        throw new Error('Path traversal denied');
      }
      if (fs.existsSync(target) && fs.statSync(target).isDirectory()) {
        fs.rmSync(target, { recursive: true, force: true });
        return true;
      }
    }
    return false;
  }

  // ── Internal ──────────────────────────────────────────────

  private _wireChild(job: InternalJob, child: ChildProcess): void {
    let stdoutBuf = '';
    let stderrBuf = '';

    child.stdout?.on('data', (chunk: Buffer) => {
      stdoutBuf += chunk.toString();
      let nl: number;
      while ((nl = stdoutBuf.indexOf('\n')) >= 0) {
        const line = stdoutBuf.slice(0, nl).trim();
        stdoutBuf = stdoutBuf.slice(nl + 1);
        if (line) this._handleStdoutLine(job, line);
      }
    });

    child.stderr?.on('data', (chunk: Buffer) => {
      stderrBuf += chunk.toString();
      let nl: number;
      while ((nl = stderrBuf.indexOf('\n')) >= 0) {
        const line = stderrBuf.slice(0, nl).trim();
        stderrBuf = stderrBuf.slice(nl + 1);
        if (line) this._handleStderrLine(job, line);
      }
    });

    child.on('error', (err) => {
      if (job.status === 'running') {
        job.status = 'failed';
        job.error = `Failed to spawn build: ${err.message}`;
      }
      if (this.activeJobId === job.jobId) this.activeJobId = null;
      this.emit('progress');
    });

    child.on('exit', (code) => {
      job.exitCode = code ?? undefined;
      if (job.status === 'cancelled') {
        // Already settled by cancelBuild.
      } else if (code === 0) {
        job.status = 'completed';
        job.progress = 1;
      } else {
        job.status = 'failed';
        if (!job.error) job.error = `Build exited with code ${code}`;
      }
      if (this.activeJobId === job.jobId) this.activeJobId = null;
      this.emit('progress');
    });
  }

  private _handleStdoutLine(job: InternalJob, line: string): void {
    let parsed: BuildProgress | null = null;
    try {
      const obj = JSON.parse(line);
      if (typeof obj.progress === 'number' && typeof obj.name === 'string') {
        parsed = {
          step: typeof obj.step === 'number' ? obj.step : 0,
          name: obj.name,
          progress: obj.progress,
          status: typeof obj.status === 'string' ? obj.status : '',
        };
      }
    } catch { /* non-JSON stdout (e.g. manifest print) is ignored for progress */ }

    if (parsed) {
      job.lines.push(parsed);
      job.progress = parsed.progress;
      job.step = parsed.name;
      this.emit('progress');
    }
  }

  private _handleStderrLine(job: InternalJob, line: string): void {
    try {
      const obj = JSON.parse(line);
      if (obj && obj.error) {
        // Surface the CLI's structured error (no silent ignore).
        job.error = `[${obj.step ?? '?'}] ${obj.message ?? 'build error'}`;
        this.emit('progress');
        return;
      }
    } catch { /* plain stderr text — capture as last-resort error context */ }
    if (!job.error) job.error = line;
  }
}

/** Thrown by startBuild when the single-GPU lock is held; route maps to 409. */
export class BuildLockedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'BuildLockedError';
  }
}

export const trtBundleService = new TrtBundleService();
