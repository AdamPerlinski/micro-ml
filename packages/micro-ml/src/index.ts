export * from './types.js';

import type {
  LinearModel,
} from './types.js';

// WASM module instance (lazily loaded)
let wasmModule: typeof import('../wasm/micro_ml_core.js') | null = null;
let initPromise: Promise<void> | null = null;

/**
 * Initialize the WASM module
 * This is called automatically when using any function, but can be called
 * explicitly for eager loading.
 */
export async function init(): Promise<void> {
  if (wasmModule) return;

  if (!initPromise) {
    initPromise = (async () => {
      const mod = await import('../wasm/micro_ml_core.js');

      // Check if we're in Node.js
      const isNode = typeof globalThis.process !== 'undefined' &&
                     typeof globalThis.process.versions?.node === 'string';

      if (isNode) {
        // Node.js: read WASM file directly
        try {
          const { readFileSync } = await import('fs');
          const { fileURLToPath } = await import('url');
          const { dirname, join } = await import('path');

          const __filename = fileURLToPath(import.meta.url);
          const __dirname = dirname(__filename);
          const wasmPath = join(__dirname, '..', 'wasm', 'micro_ml_core_bg.wasm');

          mod.initSync({ module: readFileSync(wasmPath) });
        } catch {
          await mod.default();
        }
      } else {
        // Browser: use fetch
        await mod.default();
      }

      wasmModule = mod;
    })();
  }

  await initPromise;
}

/**
 * Ensure WASM is loaded before use
 */
async function ensureInit(): Promise<typeof import('../wasm/micro_ml_core.js')> {
  await init();
  return wasmModule!;
}

// Wrapper to convert WASM model to JS-friendly interface
function wrapLinearModel(wasmModel: any): LinearModel {
  return {
    get slope() { return wasmModel.slope; },
    get intercept() { return wasmModel.intercept; },
    get rSquared() { return wasmModel.rSquared; },
    get n() { return wasmModel.n; },
    predict(x: number[]): number[] {
      return Array.from(wasmModel.predict(new Float64Array(x)));
    },
    toString(): string {
      return wasmModel.toString();
    },
  };
}

/**
 * Fit a linear regression model: y = slope * x + intercept
 */
export async function linearRegression(
  x: number[],
  y: number[]
): Promise<LinearModel> {
  const wasm = await ensureInit();
  const wasmModel = wasm.linearRegression(new Float64Array(x), new Float64Array(y));
  return wrapLinearModel(wasmModel);
}

/**
 * Simple linear regression with auto-generated x values (0, 1, 2, ...)
 */
export async function linearRegressionSimple(y: number[]): Promise<LinearModel> {
  const wasm = await ensureInit();
  const wasmModel = wasm.linearRegressionSimple(new Float64Array(y));
  return wrapLinearModel(wasmModel);
}
