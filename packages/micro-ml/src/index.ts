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
  ErrorMetrics,
  ResidualsResult,
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

// ============================================================================
// Moving Averages
// ============================================================================

/**
 * Calculate a moving average
 *
 * @example
 * ```ts
 * const smoothed = await movingAverage(data, { window: 7, type: 'ema' });
 * ```
 */
export async function movingAverage(
  data: number[],
  options: MovingAverageOptions
): Promise<number[]> {
  const wasm = await ensureInit();
  const type = options.type ?? 'sma';
  const typeEnum = type === 'ema' ? wasm.MovingAverageType.EMA
    : type === 'wma' ? wasm.MovingAverageType.WMA
    : wasm.MovingAverageType.SMA;

  return Array.from(
    wasm.movingAverage(new Float64Array(data), options.window, typeEnum)
  );
}

/**
 * Calculate Simple Moving Average
 */
export async function sma(data: number[], window: number): Promise<number[]> {
  const wasm = await ensureInit();
  return Array.from(wasm.sma(new Float64Array(data), window));
}

/**
 * Calculate Exponential Moving Average
 */
export async function ema(data: number[], window: number): Promise<number[]> {
  const wasm = await ensureInit();
  return Array.from(wasm.ema(new Float64Array(data), window));
}

/**
 * Calculate Weighted Moving Average
 */
export async function wma(data: number[], window: number): Promise<number[]> {
  const wasm = await ensureInit();
  return Array.from(wasm.wma(new Float64Array(data), window));
}

// ============================================================================
// Trend Analysis & Forecasting
// ============================================================================

/**
 * Analyze trend and forecast future values
 *
 * @example
 * ```ts
 * const trend = await trendForecast(data, 10);
 * console.log(trend.direction); // 'up'
 * console.log(trend.slope); // 10
 * console.log(trend.getForecast()); // [50, 60, 70]
 * ```
 */
export async function trendForecast(
  data: number[],
  periods: number
): Promise<TrendAnalysis> {
  const wasm = await ensureInit();
  const result = wasm.trendForecast(new Float64Array(data), periods);

  // Map direction enum to string
  const directionMap: Record<number, 'up' | 'down' | 'flat'> = {
    [wasm.TrendDirection.Up]: 'up',
    [wasm.TrendDirection.Down]: 'down',
    [wasm.TrendDirection.Flat]: 'flat',
  };

  return {
    direction: directionMap[result.direction],
    slope: result.slope,
    strength: result.strength,
    getForecast: () => Array.from(result.getForecast()),
  };
}

/**
 * Calculate rate of change as percentage
 */
export async function rateOfChange(
  data: number[],
  periods: number
): Promise<number[]> {
  const wasm = await ensureInit();
  return Array.from(wasm.rateOfChange(new Float64Array(data), periods));
}

/**
 * Calculate momentum (difference from n periods ago)
 */
export async function momentum(
  data: number[],
  periods: number
): Promise<number[]> {
  const wasm = await ensureInit();
  return Array.from(wasm.momentum(new Float64Array(data), periods));
}

/**
 * Apply exponential smoothing to data
 *
 * @example
 * ```ts
 * const smoothed = await exponentialSmoothing([10, 20, 15, 25], { alpha: 0.3 });
 * ```
 */
export async function exponentialSmoothing(
  data: number[],
  options: SmoothingOptions = {}
): Promise<number[]> {
  const wasm = await ensureInit();
  const alpha = options.alpha ?? 0.3;
  return Array.from(wasm.exponentialSmoothing(new Float64Array(data), alpha));
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Find peaks (local maxima) in data
 * Returns indices of peak values.
 */
export async function findPeaks(data: number[]): Promise<number[]> {
  const wasm = await ensureInit();
  return Array.from(wasm.findPeaks(new Float64Array(data)));
}

/**
 * Find troughs (local minima) in data
 * Returns indices of trough values.
 */
export async function findTroughs(data: number[]): Promise<number[]> {
  const wasm = await ensureInit();
  return Array.from(wasm.findTroughs(new Float64Array(data)));
}

// ============================================================================
// Convenience functions for one-liner usage
// ============================================================================

/**
 * Quick predict: fit model and get predictions in one call
 *
 * @example
 * ```ts
 * const predictions = await predict(xData, yData, [100, 200, 300]);
 * ```
 */
export async function predict(
  xTrain: number[],
  yTrain: number[],
  xPredict: number[]
): Promise<number[]> {
  const model = await linearRegression(xTrain, yTrain);
  return model.predict(xPredict);
}

/**
 * Quick trend line: fit model and extrapolate
 *
 * @example
 * ```ts
 * const future = await trendLine(data, 10); // Next 10 points
 * ```
 */
export async function trendLine(
  data: number[],
  futurePoints: number
): Promise<{ model: LinearModel; trend: number[] }> {
  const model = await linearRegressionSimple(data);
  const futureX = Array.from({ length: futurePoints }, (_, i) => data.length + i);
  const trend = model.predict(futureX);
  return { model, trend };
}

// ============================================================================
// Error Metrics
// ============================================================================

/**
 * Root Mean Squared Error between actual and predicted values
 *
 * @example
 * ```ts
 * const error = rmse([1, 2, 3], [1.1, 2.2, 2.8]); // ~0.173
 * ```
 */
export function rmse(actual: number[], predicted: number[]): number {
  if (actual.length !== predicted.length) {
    throw new Error('Arrays must have the same length');
  }
  const n = actual.length;
  const sumSqErr = actual.reduce(
    (sum, val, i) => sum + (val - predicted[i]) ** 2,
    0
  );
  return Math.sqrt(sumSqErr / n);
}

/**
 * Mean Absolute Error between actual and predicted values
 *
 * @example
 * ```ts
 * const error = mae([1, 2, 3], [1.1, 2.2, 2.8]); // ~0.167
 * ```
 */
export function mae(actual: number[], predicted: number[]): number {
  if (actual.length !== predicted.length) {
    throw new Error('Arrays must have the same length');
  }
  const n = actual.length;
  const sumAbsErr = actual.reduce(
    (sum, val, i) => sum + Math.abs(val - predicted[i]),
    0
  );
  return sumAbsErr / n;
}

/**
 * Mean Absolute Percentage Error between actual and predicted values.
 * Returns value as a percentage (e.g. 5.0 means 5%).
 * Skips data points where actual value is zero.
 *
 * @example
 * ```ts
 * const error = mape([100, 200, 300], [110, 190, 310]); // ~5.56
 * ```
 */
export function mape(actual: number[], predicted: number[]): number {
  if (actual.length !== predicted.length) {
    throw new Error('Arrays must have the same length');
  }
  let sum = 0;
  let count = 0;
  for (let i = 0; i < actual.length; i++) {
    if (actual[i] !== 0) {
      sum += Math.abs((actual[i] - predicted[i]) / actual[i]);
      count++;
    }
  }
  if (count === 0) return 0;
  return (sum / count) * 100;
}

/**
 * Compute all error metrics at once
 *
 * @example
 * ```ts
 * const metrics = errorMetrics([1, 2, 3], [1.1, 2.2, 2.8]);
 * console.log(metrics.rmse, metrics.mae, metrics.mape);
 * ```
 */
export function errorMetrics(actual: number[], predicted: number[]): ErrorMetrics {
  return {
    rmse: rmse(actual, predicted),
    mae: mae(actual, predicted),
    mape: mape(actual, predicted),
    n: actual.length,
  };
}

/**
 * Analyze residuals (actual - predicted) for model diagnostics.
 * Useful for checking whether a model's errors are randomly distributed.
 *
 * @example
 * ```ts
 * const result = residuals([1, 2, 3], [1.1, 1.9, 3.2]);
 * console.log(result.mean);         // ~-0.067 (close to 0 = unbiased)
 * console.log(result.standardized); // z-scores of residuals
 * ```
 */
export function residuals(actual: number[], predicted: number[]): ResidualsResult {
  if (actual.length !== predicted.length) {
    throw new Error('Arrays must have the same length');
  }
  const resids = actual.map((val, i) => val - predicted[i]);
  const mean = resids.reduce((a, b) => a + b, 0) / resids.length;
  const variance = resids.reduce((sum, r) => sum + (r - mean) ** 2, 0) / resids.length;
  const stdDev = Math.sqrt(variance);
  const standardized = stdDev === 0
    ? resids.map(() => 0)
    : resids.map(r => (r - mean) / stdDev);

  return { residuals: resids, mean, stdDev, standardized };
}
