import {
  type SystemPrerequisites,
  diskThreshold,
  isGpuCompatible,
  vramMeetsThreshold,
} from './systemService.js';
import { modelDownloadService } from './modelDownloadService.js';

const ALLOWED_VARIANTS = new Set(
  modelDownloadService.getTrtRegistry().map((variant) => variant.id),
);
const ALLOWED_PRECISIONS = new Set(['q8map-fp16', 'w8a8', 'fp32']);

type BuildInput = {
  variant?: string;
  precision?: string;
  prerequisites: SystemPrerequisites;
};

type BuildValidationResult =
  | { ok: true }
  | { ok: false; status: 400 | 403; error: string };

export function validateBuildRequest(input: BuildInput): BuildValidationResult {
  const variant = input.variant?.trim();
  const precision = input.precision?.trim() ?? variant;
  const { prerequisites } = input;

  if (!variant || !ALLOWED_VARIANTS.has(variant)) {
    return {
      ok: false,
      status: 400,
      error: `Invalid variant. Must be one of: ${Array.from(ALLOWED_VARIANTS).join(', ')}`,
    };
  }

  if (!precision || !ALLOWED_PRECISIONS.has(precision)) {
    return {
      ok: false,
      status: 400,
      error: `Invalid precision. Must be one of: ${Array.from(ALLOWED_PRECISIONS).join(', ')}`,
    };
  }

  // w8a8 requires sm_75+ for INT8 tensor cores.
  // The ConvRotInt8Linear TRT plugin uses INT8 MMA instructions that don't
  // exist on pre-Turing GPUs.
  if (precision === 'w8a8' && !isGpuCompatible(prerequisites.gpuCapability)) {
    const capability = prerequisites.gpuCapability ?? 'unknown';
    return {
      ok: false,
      status: 403,
      error: `GPU compute capability ${capability} is below sm_75; w8a8 builds are not supported.`,
    };
  }

  if (diskThreshold(prerequisites.freeDiskSpaceGb) === 'blocked') {
    return {
      ok: false,
      status: 403,
      error: `Insufficient disk space: ${prerequisites.freeDiskSpaceGb} GB available, minimum 2 GB required`,
    };
  }

  const tier = isXlVariant(variant) ? 'xl' : 'base';
  if (!vramMeetsThreshold(prerequisites.gpuVramMb, tier)) {
    const minimum = tier === 'xl' ? 6_144 : 4_096;
    return {
      ok: false,
      status: 403,
      error: `Insufficient VRAM: ${prerequisites.gpuVramMb} MB available, minimum ${minimum} MB required for ${variant}`,
    };
  }

  return { ok: true };
}

export function isAllowedVariant(variant: string): boolean {
  return ALLOWED_VARIANTS.has(variant);
}

function isXlVariant(variant: string): boolean {
  return variant.toLowerCase().includes('xl');
}
