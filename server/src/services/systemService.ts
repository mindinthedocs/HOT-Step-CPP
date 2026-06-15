import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { config, PROJECT_ROOT } from '../config.js';

const execFileAsync = promisify(execFile);
const BYTES_PER_GB = 1024 ** 3;

export interface SystemPrerequisites {
  cuda: { installed: boolean; version?: string };
  trt: { installed: boolean; path?: string };
  venv: { installed: boolean };
  gpuCapability: number | null;
  gpuCapabilityLabel: string | null;
  freeDiskSpaceGb: number;
  totalRamGb: number;
  gpuVramMb: number | null;
}

export type DiskThreshold = 'ok' | 'warning' | 'blocked';
export type VariantTier = 'base' | 'xl';

export function isGpuCompatible(capability: number | null): boolean {
  return capability !== null && capability >= 7.5;
}

export function diskThreshold(freeGb: number): DiskThreshold {
  if (freeGb <= 2) return 'blocked';
  if (freeGb < 50) return 'warning';
  return 'ok';
}

export function vramMeetsThreshold(vramMb: number | null, variantTier: VariantTier): boolean {
  if (vramMb === null) return true;
  const minimum = variantTier === 'xl' ? 6_144 : 4_096;
  return vramMb >= minimum;
}

export async function checkPrerequisites(): Promise<SystemPrerequisites> {
  const [cudaVersion, trtPath, venvInstalled, gpuCapability, freeDiskSpaceGb, gpuVramMb] = await Promise.all([
    getCudaVersion(),
    getTensorRtPath(),
    hasVenvPython(),
    getGpuCapability(),
    getFreeDiskSpaceGb(config.aceServer.models),
    getGpuVramMb(),
  ]);

  return {
    cuda: cudaVersion ? { installed: true, version: cudaVersion } : { installed: false },
    trt: trtPath ? { installed: true, path: trtPath } : { installed: false },
    venv: { installed: venvInstalled },
    gpuCapability,
    gpuCapabilityLabel: toGpuCapabilityLabel(gpuCapability),
    freeDiskSpaceGb,
    totalRamGb: roundToOneDecimal(os.totalmem() / BYTES_PER_GB),
    gpuVramMb,
  };
}

async function runCommand(command: string, args: string[]): Promise<string | null> {
  try {
    const { stdout, stderr } = await execFileAsync(command, args, {
      windowsHide: true,
      timeout: 5000,
      maxBuffer: 1024 * 1024,
    });
    const merged = `${stdout ?? ''}\n${stderr ?? ''}`.trim();
    return merged.length > 0 ? merged : null;
  } catch {
    return null;
  }
}

async function getCudaVersion(): Promise<string | null> {
  const [registryVersion, nvccVersion] = await Promise.all([
    getCudaVersionFromRegistry(),
    getCudaVersionFromNvcc(),
  ]);
  return registryVersion ?? nvccVersion;
}

async function getCudaVersionFromRegistry(): Promise<string | null> {
  if (process.platform !== 'win32') return null;

  const regOutput = await runCommand('reg', [
    'query',
    'HKLM\\SOFTWARE\\NVIDIA Corporation\\GPU Computing Toolkit\\CUDA',
    '/s',
  ]);
  if (!regOutput) return null;

  const match = regOutput.match(/v(13(?:\.\d+)?)/i);
  return match ? match[1] : null;
}

async function getCudaVersionFromNvcc(): Promise<string | null> {
  const output = await runCommand('nvcc', ['--version']);
  if (!output) return null;
  const match = output.match(/release\s+(\d+(?:\.\d+)?)/i);
  if (!match) return null;
  return match[1].startsWith('13') ? match[1] : null;
}

async function getTensorRtPath(): Promise<string | null> {
  const userLocalTrtPath = process.platform === 'win32'
    ? path.join(os.homedir(), 'AppData', 'Local', 'NVIDIA', 'TensorRT')
    : path.join(os.homedir(), '.local', 'share', 'nvidia', 'tensorrt');
  const configuredTrtDir = config.aceServer.trtLibs
    ? path.resolve(config.aceServer.trtLibs)
    : '';

  const candidates = [
    process.env.TRT_PATH ? path.resolve(process.env.TRT_PATH) : '',
    userLocalTrtPath,
    configuredTrtDir,
    configuredTrtDir ? path.dirname(configuredTrtDir) : '',
  ].filter(Boolean);

  for (const candidate of candidates) {
    if (await hasTensorRtArtifacts(candidate)) {
      return candidate;
    }
  }

  if (await venvHasTensorRtBindings()) {
    return userLocalTrtPath;
  }
  return null;
}

async function hasTensorRtArtifacts(rootDir: string): Promise<boolean> {
  if (!(await isDirectory(rootDir))) return false;
  const markers = ['trtexec.exe', 'trtexec', 'nvinfer_10.dll', 'nvinfer.dll'];
  for (const marker of markers) {
    if (await fileExists(path.join(rootDir, marker))) return true;
    if (await fileExists(path.join(rootDir, 'bin', marker))) return true;
    if (await fileExists(path.join(rootDir, 'lib', marker))) return true;
  }
  return hasWheelUnder(rootDir, 3);
}

async function hasWheelUnder(dir: string, depth: number): Promise<boolean> {
  if (depth < 0) return false;
  let entries;
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return false;
  }
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isFile() && entry.name.endsWith('.whl') && /tensorrt/i.test(entry.name)) {
      return true;
    }
    if (entry.isDirectory() && depth > 0) {
      if (await hasWheelUnder(full, depth - 1)) return true;
    }
  }
  return false;
}

async function venvHasTensorRtBindings(): Promise<boolean> {
  const venvPython = process.platform === 'win32'
    ? path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe')
    : path.join(PROJECT_ROOT, '.venv', 'bin', 'python');
  if (!(await fileExists(venvPython))) return false;
  const output = await runCommand(venvPython, ['-c', 'import tensorrt; print(tensorrt.__version__)']);
  return Boolean(output && output.trim().length > 0);
}

async function hasVenvPython(): Promise<boolean> {
  const venvPython = process.platform === 'win32'
    ? path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe')
    : path.join(PROJECT_ROOT, '.venv', 'bin', 'python');
  return fileExists(venvPython);
}

async function getGpuCapability(): Promise<number | null> {
  const output = await runCommand('nvidia-smi', ['--query-gpu=compute_cap', '--format=csv,noheader']);
  if (!output) return null;
  const firstLine = output.split(/\r?\n/).find(Boolean);
  if (!firstLine) return null;
  const match = firstLine.trim().match(/(\d+(?:\.\d+)?)/);
  return match ? Number(match[1]) : null;
}

async function getGpuVramMb(): Promise<number | null> {
  const output = await runCommand('nvidia-smi', ['--query-gpu=memory.total', '--format=csv,noheader,nounits']);
  if (!output) return null;
  const firstLine = output.split(/\r?\n/).find(Boolean);
  if (!firstLine) return null;
  const value = Number.parseInt(firstLine.trim(), 10);
  return Number.isFinite(value) ? value : null;
}

async function getFreeDiskSpaceGb(startPath: string): Promise<number> {
  const targetPath = await resolveExistingPath(startPath);
  if (!targetPath) {
    return roundToOneDecimal(os.freemem() / BYTES_PER_GB);
  }

  try {
    const stat = await fs.statfs(targetPath);
    const freeBytes = Number(stat.bavail) * Number(stat.bsize);
    if (!Number.isFinite(freeBytes) || freeBytes <= 0) {
      throw new Error('Invalid statfs result');
    }
    return roundToOneDecimal(freeBytes / BYTES_PER_GB);
  } catch {
    return roundToOneDecimal(os.freemem() / BYTES_PER_GB);
  }
}

async function resolveExistingPath(targetPath: string): Promise<string | null> {
  let currentPath = path.resolve(targetPath);
  for (let i = 0; i < 8; i += 1) {
    if (await fileExists(currentPath)) return currentPath;
    const parent = path.dirname(currentPath);
    if (parent === currentPath) break;
    currentPath = parent;
  }
  return null;
}

function toGpuCapabilityLabel(capability: number | null): string | null {
  if (capability === null) return null;
  const scaled = Math.round(capability * 10);
  return `sm_${String(scaled).padStart(2, '0')}`;
}

function roundToOneDecimal(value: number): number {
  return Math.round(value * 10) / 10;
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function isDirectory(dirPath: string): Promise<boolean> {
  try {
    const stat = await fs.stat(dirPath);
    return stat.isDirectory();
  } catch {
    return false;
  }
}
