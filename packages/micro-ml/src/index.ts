export * from './types.js';

import type {
  LinearModel,
  PolynomialModel,
  ExponentialModel,
  LogarithmicModel,
  PowerModel,
  TrendAnalysis,
  MovingAverageOptions,
  PolynomialOptions,
  SmoothingOptions,
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

function wrapPolynomialModel(wasmModel: any): PolynomialModel {
  return {
    get degree() { return wasmModel.degree; },
    get rSquared() { return wasmModel.rSquared; },
    get n() { return wasmModel.n; },
    getCoefficients(): number[] {
      return Array.from(wasmModel.getCoefficients());
    },
    predict(x: number[]): number[] {
      return Array.from(wasmModel.predict(new Float64Array(x)));
    },
    toString(): string {
      return wasmModel.toString();
    },
  };
}

function wrapExponentialModel(wasmModel: any): ExponentialModel {
  return {
    get a() { return wasmModel.a; },
    get b() { return wasmModel.b; },
    get rSquared() { return wasmModel.rSquared; },
    get n() { return wasmModel.n; },
    predict(x: number[]): number[] {
      return Array.from(wasmModel.predict(new Float64Array(x)));
    },
    toString(): string {
      return wasmModel.toString();
    },
    doublingTime(): number {
      return wasmModel.doublingTime();
    },
  };
}

function wrapLogarithmicModel(wasmModel: any): LogarithmicModel {
  return {
    get a() { return wasmModel.a; },
    get b() { return wasmModel.b; },
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

function wrapPowerModel(wasmModel: any): PowerModel {
  return {
    get a() { return wasmModel.a; },
    get b() { return wasmModel.b; },
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

// ============================================================================
// Linear Regression
// ============================================================================

/**
 * Fit a linear regression model: y = slope * x + intercept
 *
 * @example
 * ```ts
 * const model = await linearRegression([1, 2, 3, 4], [2, 4, 6, 8]);
 * console.log(model.slope); // 2
 * console.log(model.intercept); // 0
 * console.log(model.rSquared); // 1
 * const predictions = model.predict([5, 6]); // [10, 12]
 * ```
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
 * Useful for time series data where x is just the index.
 *
 * @example
 * ```ts
 * const model = await linearRegressionSimple([10, 20, 30, 40]);
 * // Equivalent to linearRegression([0, 1, 2, 3], [10, 20, 30, 40])
 * ```
 */
export async function linearRegressionSimple(y: number[]): Promise<LinearModel> {
  const wasm = await ensureInit();
  const wasmModel = wasm.linearRegressionSimple(new Float64Array(y));
  return wrapLinearModel(wasmModel);
}

// ============================================================================
// Polynomial Regression
// ============================================================================

/**
 * Fit a polynomial regression model: y = c0 + c1*x + c2*x² + ...
 *
 * @example
 * ```ts
 * const model = await polynomialRegression([0, 1, 2, 3], [1, 2, 5, 10], { degree: 2 });
 * console.log(model.getCoefficients()); // [c0, c1, c2]
 * ```
 */
export async function polynomialRegression(
  x: number[],
  y: number[],
  options: PolynomialOptions = {}
): Promise<PolynomialModel> {
  const wasm = await ensureInit();
  const degree = options.degree ?? 2;
  const wasmModel = wasm.polynomialRegression(new Float64Array(x), new Float64Array(y), degree);
  return wrapPolynomialModel(wasmModel);
}

/**
 * Polynomial regression with auto-generated x values (0, 1, 2, ...)
 */
export async function polynomialRegressionSimple(
  y: number[],
  options: PolynomialOptions = {}
): Promise<PolynomialModel> {
  const wasm = await ensureInit();
  const degree = options.degree ?? 2;
  const wasmModel = wasm.polynomialRegressionSimple(new Float64Array(y), degree);
  return wrapPolynomialModel(wasmModel);
}

// ============================================================================
// Exponential & Logarithmic Regression
// ============================================================================

/**
 * Fit an exponential regression model: y = a * e^(b*x)
 * All y values must be positive.
 *
 * @example
 * ```ts
 * const model = await exponentialRegression([0, 1, 2], [1, 2.7, 7.4]);
 * console.log(model.a); // ~1
 * console.log(model.b); // ~1 (e^1 ≈ 2.718)
 * console.log(model.doublingTime()); // Time to double
 * ```
 */
export async function exponentialRegression(
  x: number[],
  y: number[]
): Promise<ExponentialModel> {
  const wasm = await ensureInit();
  const wasmModel = wasm.exponentialRegression(new Float64Array(x), new Float64Array(y));
  return wrapExponentialModel(wasmModel);
}

/**
 * Exponential regression with auto-generated x values (0, 1, 2, ...)
 */
export async function exponentialRegressionSimple(
  y: number[]
): Promise<ExponentialModel> {
  const wasm = await ensureInit();
  const wasmModel = wasm.exponentialRegressionSimple(new Float64Array(y));
  return wrapExponentialModel(wasmModel);
}

/**
 * Fit a logarithmic regression model: y = a + b * ln(x)
 * All x values must be positive.
 *
 * @example
 * ```ts
 * const model = await logarithmicRegression([1, 2, 3, 4], [0, 0.69, 1.1, 1.39]);
 * console.log(model.a); // Intercept
 * console.log(model.b); // Coefficient
 * ```
 */
export async function logarithmicRegression(
  x: number[],
  y: number[]
): Promise<LogarithmicModel> {
  const wasm = await ensureInit();
  const wasmModel = wasm.logarithmicRegression(new Float64Array(x), new Float64Array(y));
  return wrapLogarithmicModel(wasmModel);
}

/**
 * Fit a power regression model: y = a * x^b
 * All x and y values must be positive.
 *
 * @example
 * ```ts
 * const model = await powerRegression([1, 2, 3, 4], [1, 4, 9, 16]);
 * console.log(model.a); // ~1
 * console.log(model.b); // ~2 (quadratic)
 * ```
 */
export async function powerRegression(
  x: number[],
  y: number[]
): Promise<PowerModel> {
  const wasm = await ensureInit();
  const wasmModel = wasm.powerRegression(new Float64Array(x), new Float64Array(y));
  return wrapPowerModel(wasmModel);
}
