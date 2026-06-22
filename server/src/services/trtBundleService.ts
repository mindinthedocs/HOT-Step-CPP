import { EventEmitter } from 'events';
import fs from 'fs';
import path from 'path';
import { spawn, ChildProcess } from 'child_process';
import { randomUUID } from 'crypto';
import { config, PROJECT_ROOT } from '../config.js';

export type BuildStatus = 'running' | 'completed' | 'failed' | 'cancelled';

export interface BuildProgress {
  step: number;
  name: string;
  progress: number;
  status: string;
}

export interface BuildJob {
  jobId: string;
  variant: string;
  bundleName: string;
  status: BuildStatus;
  progress: number;
  step: string;
  lines: BuildProgress[];
  error?: string;
  exitCode?: number;
  logPath?: string;
  bundleDir?: string;
}

interface StartBuildInput {
  variant: string;
  precision: string;
  ditDir: string;
  sourceModel?: string;
}

interface StartEmbeddingBuildInput {
  textEncoderDir: string;
  sourceModel?: string;
}

interface PersistentBuildLock {
  jobId: string;
  variant: string;
  bundleName: string;
  pid: number;
  startedAt: string;
  logPath: string;
  bundleDir: string;
}

interface InternalJob extends BuildJob {
  child?: ChildProcess;
  childPid?: number;
}

class TrtBundleService extends EventEmitter {
  private jobs = new Map<string, InternalJob>();
  private activeJobId: string | null = null;
  private recoveredPidPoll: NodeJS.Timeout | null = null;

  constructor() {
    super();
    void this.recoverPersistentLock();
  }

  get bundlesDir(): string {
    return path.join(config.aceServer.models, 'trt-bundles');
  }

  get logsDir(): string {
    return path.join(PROJECT_ROOT, 'logs');
  }

  private get lockPath(): string {
    return path.join(PROJECT_ROOT, '.trt-build-lock.json');
  }

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

  isBuilding(): boolean {
    const active = this.activeJobId ? this.jobs.get(this.activeJobId) : undefined;
    return Boolean(active && active.status === 'running');
  }

  getActiveBuild(): BuildJob | null {
    if (!this.activeJobId) return null;
    const job = this.jobs.get(this.activeJobId);
    if (!job) return null;
    return this.publicJob(job);
  }

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
          available = this.manifestEnginesPresent(bundleDir, manifest);
        } catch {
          available = false;
        }
      }
      out.push({ name, available, variant });
    }
    return out;
  }

  getJobs(): BuildJob[] {
    return Array.from(this.jobs.values()).map((j) => this.publicJob(j));
  }

  getJob(jobId: string): BuildJob | undefined {
    const job = this.jobs.get(jobId);
    return job ? this.publicJob(job) : undefined;
  }

  getBuildLog(jobId: string): string | null {
    const job = this.jobs.get(jobId);
    const logPath = job?.logPath;
    if (!logPath || !fs.existsSync(logPath)) return null;
    return fs.readFileSync(logPath, 'utf-8');
  }

  openBuildLogFolder(jobId: string): boolean {
    const job = this.jobs.get(jobId);
    if (!job?.logPath) return false;
    const folder = path.dirname(job.logPath);
    if (!fs.existsSync(folder)) return false;
    try {
      if (process.platform === 'win32') {
        const child = spawn('explorer', [folder], { detached: true, stdio: 'ignore' });
        child.unref();
      } else if (process.platform === 'darwin') {
        const child = spawn('open', [folder], { detached: true, stdio: 'ignore' });
        child.unref();
      } else {
        const child = spawn('xdg-open', [folder], { detached: true, stdio: 'ignore' });
        child.unref();
      }
      return true;
    } catch {
      return false;
    }
  }

  async startBuild(input: StartBuildInput): Promise<string> {
    if (this.activeJobId !== null) {
      throw new BuildLockedError('A TRT build is already running (single-GPU-locked)');
    }

    const variant = this.assertSafeVariant(input.variant);
    if (!fs.existsSync(this.pythonExe)) {
      throw new Error(`Missing required .venv Python interpreter: ${this.pythonExe}`);
    }
    if (!fs.existsSync(this.cliScript)) {
      throw new Error(`Missing TRT bundle CLI script: ${this.cliScript}`);
    }

    const precision = input.precision ?? 'q8map-fp16';
    const bundleName = this.bundleNameFor(variant, precision);
    const outputDir = path.join('models', 'trt-bundles', bundleName);
    const bundleDir = path.join(PROJECT_ROOT, outputDir);

    fs.mkdirSync(this.logsDir, { recursive: true });
    fs.mkdirSync(path.dirname(bundleDir), { recursive: true });

    const jobId = randomUUID().slice(0, 8);
    const logPath = path.join(this.logsDir, `build-${jobId}.log`);
    const job: InternalJob = {
      jobId,
      variant,
      bundleName,
      status: 'running',
      progress: 0,
      step: 'starting',
      lines: [],
      logPath,
      bundleDir,
    };

    const ggufPath = path.join('models', 'acestep-v15-sft-BF16.gguf');
    const argv = [
      this.cliScript,
      'bundle',
      '--variant', input.precision,
      '--output-dir', outputDir,
      '--dit-dir', input.ditDir,
      '--gguf', ggufPath,
    ];
    if (input.sourceModel) {
      argv.push('--source-model', input.sourceModel);
    }

    this.jobs.set(jobId, job);
    this.activeJobId = jobId;

    const child = spawn(this.pythonExe, argv, {
      cwd: PROJECT_ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    job.child = child;
    job.childPid = child.pid ?? undefined;

    this.writeLock({
      jobId,
      variant,
      bundleName,
      pid: child.pid ?? 0,
      startedAt: new Date().toISOString(),
      logPath,
      bundleDir,
    });

    this.wireChild(job, child);
    this.emit('progress');
    return jobId;
  }

  /**
   * Start a standalone Qwen3-emb TRT bundle build. The Qwen3-emb bundle is
   * independent of any DiT bundle: it ships only the TRT Qwen3 text encoder
   * (+ sidecars) and lives in its own folder (trt-bundles/qwen3-emb/). When
   * the user selects it as the text encoder, the runtime routes the text-
   * encoder forward through the TRT engine.
   */
  async startEmbeddingBuild(input: StartEmbeddingBuildInput): Promise<string> {
    if (this.activeJobId !== null) {
      throw new BuildLockedError('A TRT build is already running (single-GPU-locked)');
    }
    if (!fs.existsSync(this.pythonExe)) {
      throw new Error(`Missing required .venv Python interpreter: ${this.pythonExe}`);
    }
    if (!fs.existsSync(this.cliScript)) {
      throw new Error(`Missing TRT bundle CLI script: ${this.cliScript}`);
    }

    const bundleName = 'qwen3-emb';
    const outputDir = path.join('models', 'trt-bundles', bundleName);
    const bundleDir = path.join(PROJECT_ROOT, outputDir);

    fs.mkdirSync(this.logsDir, { recursive: true });
    fs.mkdirSync(path.dirname(bundleDir), { recursive: true });

    const jobId = randomUUID().slice(0, 8);
    const logPath = path.join(this.logsDir, `build-emb-${jobId}.log`);
    const job: InternalJob = {
      jobId,
      variant: bundleName,
      bundleName,
      status: 'running',
      progress: 0,
      step: 'starting',
      lines: [],
      logPath,
      bundleDir,
    };

    const argv = [
      this.cliScript,
      'embedding-bundle',
      '--output-dir', outputDir,
      '--text-encoder-dir', input.textEncoderDir,
    ];
    if (input.sourceModel) {
      argv.push('--source-model', input.sourceModel);
    }

    this.jobs.set(jobId, job);
    this.activeJobId = jobId;

    const child = spawn(this.pythonExe, argv, {
      cwd: PROJECT_ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    job.child = child;
    job.childPid = child.pid ?? undefined;

    this.writeLock({
      jobId,
      variant: bundleName,
      bundleName,
      pid: child.pid ?? 0,
      startedAt: new Date().toISOString(),
      logPath,
      bundleDir,
    });

    this.wireChild(job, child);
    this.emit('progress');
    return jobId;
  }

  async cancelBuild(jobId: string): Promise<boolean> {
    const job = this.jobs.get(jobId);
    if (!job || job.status !== 'running') return false;

    const pid = job.child?.pid ?? job.childPid;
    if (pid) {
      this.killPidTree(pid);
    }

    //this.cleanupPartialBundle(job.bundleDir);
    this.settleJob(job, {
      status: 'cancelled',
      step: 'cancelled',
      error: 'Build cancelled by user.',
    });
    return true;
  }

  deleteBundle(variant: string): boolean {
    const safeName = this.assertSafeVariant(variant);
    const candidates = [safeName];
    const dir = this.bundlesDir;
    const dirResolved = path.resolve(dir);
    for (const name of candidates) {
      const target = path.join(dir, name);
      const resolved = path.resolve(target);
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

  private bundleNameFor(variant: string, precision: string): string {
    // Bundle directory naming: ``trt-<sourceModel>-<precision>``
    //
    // The original scheme was ``acestep-v15-2b-<variant>`` but the hardcoded
    // ``2b`` was misleading — ACE-Step v1.5 XL is a ~4B-parameter DiT, not 2B,
    // and the bare-letter ``b`` collides with the LM size namespace (the LM
    // ships in 4B too). The new scheme:
    //   - drops the ``2b`` size tag entirely (it was wrong and ambiguous);
    //   - prefixes with ``trt-`` so a built bundle is unambiguously a TRT
    //     artifact in the models/trt-bundles/ directory;
    //   - suffixes with the precision recipe (q8map-fp16 | w8a8 | fp32) so
    //     the user can hold multiple precision variants of the same source
    //     model without overwriting each other.
    //
    // The legacy ``acestep-v15-2b-*`` pattern is still recognized by the UI
    // (ModelSelect.tsx / modelLabels.ts) and by listBundles() so existing
    // bundles keep loading after the rename.
    const safeVariant = this.assertSafeVariant(variant);
    const safePrecision = this.assertSafeVariant(precision);
    return `trt-${safeVariant}-${safePrecision}`;
  }

  private publicJob(job: InternalJob): BuildJob {
    return {
      jobId: job.jobId,
      variant: job.variant,
      bundleName: job.bundleName,
      status: job.status,
      progress: job.progress,
      step: job.step,
      lines: job.lines,
      error: job.error,
      exitCode: job.exitCode,
      logPath: job.logPath,
      bundleDir: job.bundleDir,
    };
  }

  private writeBuildLog(job: InternalJob, stream: 'stdout' | 'stderr', line: string): void {
    if (!job.logPath) return;
    try {
      const ts = new Date().toISOString();
      fs.appendFileSync(job.logPath, `${ts} [${stream}] ${line}\n`);
    } catch {
      // best effort logging only
    }
  }

  private wireChild(job: InternalJob, child: ChildProcess): void {
    let stdoutBuf = '';
    let stderrBuf = '';

    child.stdout?.on('data', (chunk: Buffer) => {
      stdoutBuf += chunk.toString();
      let nl = stdoutBuf.indexOf('\n');
      while (nl >= 0) {
        const line = stdoutBuf.slice(0, nl).trim();
        stdoutBuf = stdoutBuf.slice(nl + 1);
        if (line) {
          this.writeBuildLog(job, 'stdout', line);
          this.handleStdoutLine(job, line);
        }
        nl = stdoutBuf.indexOf('\n');
      }
    });

    child.stderr?.on('data', (chunk: Buffer) => {
      stderrBuf += chunk.toString();
      let nl = stderrBuf.indexOf('\n');
      while (nl >= 0) {
        const line = stderrBuf.slice(0, nl).trim();
        stderrBuf = stderrBuf.slice(nl + 1);
        if (line) {
          this.writeBuildLog(job, 'stderr', line);
          this.handleStderrLine(job, line);
        }
        nl = stderrBuf.indexOf('\n');
      }
    });

    child.on('error', (err) => {
      this.settleJob(job, {
        status: 'failed',
        step: 'spawn-failed',
        error: `Failed to spawn build: ${err.message}`,
      });
    });

    child.on('exit', (code, signal) => {
      if (job.status === 'cancelled') {
        this.settleJob(job, {
          status: 'cancelled',
          step: 'cancelled',
          exitCode: code ?? undefined,
        });
        return;
      }
      if (code === 0) {
        this.settleJob(job, {
          status: 'completed',
          step: 'completed',
          progress: 1,
          exitCode: 0,
        });
        return;
      }

      const exitCode = code ?? undefined;
      const wasRunning = job.status === 'running';
      const lastLine = job.lines.length > 0 ? job.lines[job.lines.length - 1] : undefined;
      const lastStatusRunning = lastLine?.status === 'running';

      // Forced-kill / OOM detection across platforms:
      // - POSIX OOM/forced kill surfaces as exit code 137 (128 + SIGKILL) or a kill signal.
      // - Windows taskkill /F can surface as STATUS_CONTROL_C_EXIT (0xC000013A = 3221225786).
      // - Narrow win32 fallback: taskkill /F /T of the Python child commonly bubbles up to the
      //   parent as a plain exit code 1. Only treat code 1 as a forced kill when the job was
      //   actively running mid-step (last progress line status === 'running'), i.e. abrupt
      //   termination during an active build step, so ordinary code-1 build failures are not masked.
      const isForcedKill =
        exitCode === 137 ||
        signal === 'SIGKILL' ||
        signal === 'SIGTERM' ||
        (process.platform === 'win32' && exitCode === 3221225786) ||
        (process.platform === 'win32' && exitCode === 1 && wasRunning && lastStatusRunning);

      if (isForcedKill) {
        // Same partial-bundle cleanup as the cancel path: a forced kill leaves a half-written bundle.
        //this.cleanupPartialBundle(job.bundleDir);
        this.settleJob(job, {
          status: 'failed',
          step: 'failed',
          error: 'Build exited with code 137 (likely OOM or forced kill by OS/GPU driver).',
          exitCode,
        });
        return;
      }

      this.settleJob(job, {
        status: 'failed',
        step: 'failed',
        error: job.error || `Build exited with code ${code}`,
        exitCode,
      });
    });
  }

  private handleStdoutLine(job: InternalJob, line: string): void {
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
    } catch {
      // Non-JSON stdout is still useful in logs but not progress.
    }

    if (!parsed) return;
    job.lines.push(parsed);
    job.progress = parsed.progress;
    job.step = parsed.name;
    this.emit('progress');
  }

  private handleStderrLine(job: InternalJob, line: string): void {
    try {
      const obj = JSON.parse(line);
      if (obj && obj.error) {
        job.error = `[${obj.step ?? '?'}] ${obj.message ?? 'build error'}`;
        this.emit('progress');
        return;
      }
    } catch {
      // Plain stderr; capture best available context.
    }
    if (!job.error) job.error = line;
  }

  private cleanupPartialBundle(bundleDir?: string): void {
    if (!bundleDir) return;
    const bundlesRootResolved = path.resolve(this.bundlesDir);
    const bundleResolved = path.resolve(bundleDir);
    if (bundleResolved !== bundlesRootResolved && !bundleResolved.startsWith(bundlesRootResolved + path.sep)) {
      return;
    }
    try {
      fs.rmSync(bundleResolved, { recursive: true, force: true });
    } catch {
      // best effort only
    }
  }

  private settleJob(
    job: InternalJob,
    update: {
      status: BuildStatus;
      step: string;
      progress?: number;
      error?: string;
      exitCode?: number;
    },
  ): void {
    job.status = update.status;
    job.step = update.step;
    if (typeof update.progress === 'number') job.progress = update.progress;
    if (typeof update.exitCode === 'number') job.exitCode = update.exitCode;
    if (update.error) job.error = update.error;

    if (this.activeJobId === job.jobId) this.activeJobId = null;
    this.clearLock();
    this.clearRecoveredPoll();
    this.emit('progress');
  }

  private killPidTree(pid: number): void {
    try {
      if (process.platform === 'win32') {
        spawn('taskkill', ['/PID', String(pid), '/T', '/F'], { stdio: 'ignore' });
      } else {
        process.kill(pid, 'SIGTERM');
      }
    } catch {
      // process may already be gone
    }
  }

  private writeLock(lock: PersistentBuildLock): void {
    try {
      fs.writeFileSync(this.lockPath, JSON.stringify(lock, null, 2), 'utf-8');
    } catch {
      // best effort only
    }
  }

  private readLock(): PersistentBuildLock | null {
    if (!fs.existsSync(this.lockPath)) return null;
    try {
      const raw = fs.readFileSync(this.lockPath, 'utf-8');
      const parsed = JSON.parse(raw) as Partial<PersistentBuildLock>;
      if (!parsed || typeof parsed !== 'object') return null;
      if (typeof parsed.jobId !== 'string' || typeof parsed.variant !== 'string' || typeof parsed.bundleName !== 'string') {
        return null;
      }
      if (typeof parsed.pid !== 'number' || parsed.pid <= 0) return null;
      if (typeof parsed.logPath !== 'string' || typeof parsed.bundleDir !== 'string') return null;
      return {
        jobId: parsed.jobId,
        variant: parsed.variant,
        bundleName: parsed.bundleName,
        pid: parsed.pid,
        startedAt: typeof parsed.startedAt === 'string' ? parsed.startedAt : '',
        logPath: parsed.logPath,
        bundleDir: parsed.bundleDir,
      };
    } catch {
      return null;
    }
  }

  private clearLock(): void {
    try {
      if (fs.existsSync(this.lockPath)) fs.unlinkSync(this.lockPath);
    } catch {
      // best effort only
    }
  }

  private isPidRunning(pid: number): boolean {
    if (!Number.isInteger(pid) || pid <= 0) return false;
    try {
      process.kill(pid, 0);
      return true;
    } catch {
      return false;
    }
  }

  private recoverPersistentLock(): void {
    const lock = this.readLock();
    if (!lock) {
      this.clearLock();
      return;
    }

    if (!this.isPidRunning(lock.pid)) {
      this.clearLock();
      return;
    }

    const recovered: InternalJob = {
      jobId: lock.jobId,
      variant: lock.variant,
      bundleName: lock.bundleName,
      status: 'running',
      progress: 0,
      step: 'rehydrated',
      lines: [],
      error: 'Build process was rehydrated from persistent lock.',
      logPath: lock.logPath,
      bundleDir: lock.bundleDir,
      childPid: lock.pid,
    };
    this.jobs.set(recovered.jobId, recovered);
    this.activeJobId = recovered.jobId;
    this.emit('progress');

    this.clearRecoveredPoll();
    this.recoveredPidPoll = setInterval(() => {
      if (!this.isPidRunning(lock.pid)) {
        this.settleJob(recovered, {
          status: 'failed',
          step: 'orphaned-build-exited',
          error: 'Recovered build process exited after server startup.',
        });
      }
    }, 2000);
  }

  private clearRecoveredPoll(): void {
    if (this.recoveredPidPoll) {
      clearInterval(this.recoveredPidPoll);
      this.recoveredPidPoll = null;
    }
  }

  private assertSafeVariant(variant: string): string {
    const v = variant.trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(v)) {
      throw new Error('Invalid variant format.');
    }
    if (v.includes('..') || v.includes('/') || v.includes('\\')) {
      throw new Error('Path traversal denied');
    }
    return v;
  }
}

export class BuildLockedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'BuildLockedError';
  }
}

export const trtBundleService = new TrtBundleService();
