// aceTrtBundle.ts — TRT-bundle shape from the engine's GET /props `trt` array.
//
// The engine enumerates built bundles in a sibling `trt` array on /props.
// AceProps (aceClient.ts) is at its line cap, so this extension lives here and
// is intersected at the read site to thread bundles to the UI without
// dropping the data — selection maps bundle -> synth_model = <bundle name>.

import type { AceProps } from '../services/aceClient.js';

/** One TRT bundle as reported by the engine's GET /props `trt` array. */
export interface AceTrtBundle {
  name: string;
  available: boolean;
  variant?: string;
  source_model?: string;
}

/** AceProps including the optional engine-reported `trt` bundle array. */
export type AcePropsWithTrt = AceProps & { trt?: AceTrtBundle[] };
