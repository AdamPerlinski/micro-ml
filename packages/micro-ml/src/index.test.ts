import { describe, it, expect, beforeAll } from 'vitest';
import {
  init,
  linearRegression,
  linearRegressionSimple,
  polynomialRegression,
  polynomialRegressionSimple,
  exponentialRegression,
  exponentialRegressionSimple,
  logarithmicRegression,
  powerRegression,
  sma,
  ema,
  wma,
  movingAverage,
  trendForecast,
  rateOfChange,
  momentum,
  exponentialSmoothing,
  findPeaks,
  findTroughs,
  predict,
  trendLine,
  rmse,
  mae,
  mape,
  errorMetrics,
  residuals,
} from './index.js';

// Helper for approximate equality
const approx = (a: number, b: number, tolerance = 0.01) => Math.abs(a - b) < tolerance;

beforeAll(async () => {
  await init();
});

// ============================================================================
// Linear Regression
// ============================================================================

describe('linearRegression', () => {
  it('fits a perfect linear relationship', async () => {
    const model = await linearRegression([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]);

    expect(model.slope).toBeCloseTo(2, 5);
    expect(model.intercept).toBeCloseTo(0, 5);
    expect(model.rSquared).toBeCloseTo(1, 5);
    expect(model.n).toBe(5);
  });

  it('predicts new values correctly', async () => {
    const model = await linearRegression([1, 2, 3, 4], [2, 4, 6, 8]);
    const predictions = model.predict([5, 6, 10]);

    expect(predictions[0]).toBeCloseTo(10, 5);
    expect(predictions[1]).toBeCloseTo(12, 5);
    expect(predictions[2]).toBeCloseTo(20, 5);
  });

  it('handles noisy data', async () => {
    const model = await linearRegression([1, 2, 3, 4, 5], [2.1, 3.9, 6.2, 7.8, 10.1]);

    expect(model.slope).toBeGreaterThan(1.5);
    expect(model.slope).toBeLessThan(2.5);
    expect(model.rSquared).toBeGreaterThan(0.95);
  });

  it('returns a valid string representation', async () => {
    const model = await linearRegression([1, 2, 3], [2, 4, 6]);
    const str = model.toString();

    expect(str).toContain('y');
    expect(str).toContain('x');
  });
});

describe('linearRegressionSimple', () => {
  it('auto-generates x values as indices', async () => {
    const model = await linearRegressionSimple([10, 20, 30, 40, 50]);

    expect(model.slope).toBeCloseTo(10, 5);
    expect(model.intercept).toBeCloseTo(10, 5);
    expect(model.rSquared).toBeCloseTo(1, 5);
  });

  it('works with time series data', async () => {
    // Monthly sales growing by ~3000/month
    const sales = [42000, 45000, 48000, 51000, 54000];
    const model = await linearRegressionSimple(sales);

    expect(model.slope).toBeCloseTo(3000, 1);
  });
});

// ============================================================================
// Polynomial Regression
// ============================================================================

describe('polynomialRegression', () => {
  it('fits a quadratic relationship', async () => {
    // y = x^2
    const x = [0, 1, 2, 3, 4];
    const y = [0, 1, 4, 9, 16];
    const model = await polynomialRegression(x, y, { degree: 2 });

    expect(model.degree).toBe(2);
    expect(model.rSquared).toBeCloseTo(1, 3);

    const coeffs = model.getCoefficients();
    expect(coeffs).toHaveLength(3);
  });

  it('predicts quadratic values', async () => {
    const x = [0, 1, 2, 3, 4];
    const y = [0, 1, 4, 9, 16];
    const model = await polynomialRegression(x, y, { degree: 2 });

    const predictions = model.predict([5, 6]);
    expect(predictions[0]).toBeCloseTo(25, 1);
    expect(predictions[1]).toBeCloseTo(36, 1);
  });

  it('defaults to degree 2', async () => {
    const model = await polynomialRegression([0, 1, 2], [0, 1, 4]);
    expect(model.degree).toBe(2);
  });
});

describe('polynomialRegressionSimple', () => {
  it('auto-generates x values', async () => {
    const y = [0, 1, 4, 9, 16];
    const model = await polynomialRegressionSimple(y, { degree: 2 });

    expect(model.rSquared).toBeCloseTo(1, 3);
  });
});

// ============================================================================
// Exponential Regression
// ============================================================================

describe('exponentialRegression', () => {
  it('fits exponential growth', async () => {
    // y = 2 * e^(0.5x)
    const x = [0, 1, 2, 3, 4];
    const y = x.map(xi => 2 * Math.exp(0.5 * xi));
    const model = await exponentialRegression(x, y);

    expect(model.a).toBeCloseTo(2, 1);
    expect(model.b).toBeCloseTo(0.5, 1);
    expect(model.rSquared).toBeCloseTo(1, 3);
  });

  it('calculates doubling time', async () => {
    const x = [0, 1, 2, 3, 4];
    const y = x.map(xi => Math.exp(0.693 * xi)); // ln(2) ≈ 0.693
    const model = await exponentialRegression(x, y);

    expect(model.doublingTime()).toBeCloseTo(1, 1);
  });

  it('predicts future values', async () => {
    const x = [0, 1, 2, 3];
    const y = [1, 2, 4, 8];
    const model = await exponentialRegression(x, y);

    const predictions = model.predict([4]);
    expect(predictions[0]).toBeCloseTo(16, 2);
  });
});

describe('exponentialRegressionSimple', () => {
  it('works with just y values', async () => {
    const y = [1, 2, 4, 8, 16];
    const model = await exponentialRegressionSimple(y);

    expect(model.b).toBeGreaterThan(0); // Growth
    expect(model.rSquared).toBeGreaterThan(0.95);
  });
});

// ============================================================================
// Logarithmic Regression
// ============================================================================

describe('logarithmicRegression', () => {
  it('fits logarithmic relationship', async () => {
    // y = 2 * ln(x)
    const x = [1, 2, 3, 4, 5];
    const y = x.map(xi => 2 * Math.log(xi));
    const model = await logarithmicRegression(x, y);

    expect(model.b).toBeCloseTo(2, 1);
    expect(model.rSquared).toBeCloseTo(1, 3);
  });

  it('predicts correctly', async () => {
    const x = [1, 2, 4, 8];
    const y = x.map(xi => Math.log(xi));
    const model = await logarithmicRegression(x, y);

    const predictions = model.predict([16]);
    expect(predictions[0]).toBeCloseTo(Math.log(16), 1);
  });
});

// ============================================================================
// Power Regression
// ============================================================================

describe('powerRegression', () => {
  it('fits power relationship y = x^2', async () => {
    const x = [1, 2, 3, 4, 5];
    const y = x.map(xi => xi * xi);
    const model = await powerRegression(x, y);

    expect(model.a).toBeCloseTo(1, 1);
    expect(model.b).toBeCloseTo(2, 1);
    expect(model.rSquared).toBeCloseTo(1, 3);
  });

  it('fits sqrt relationship y = sqrt(x)', async () => {
    const x = [1, 4, 9, 16, 25];
    const y = x.map(xi => Math.sqrt(xi));
    const model = await powerRegression(x, y);

    expect(model.b).toBeCloseTo(0.5, 1);
  });
});

// ============================================================================
// Moving Averages
// ============================================================================

describe('sma (Simple Moving Average)', () => {
  it('calculates correct SMA', async () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const result = await sma(data, 3);

    // First two values are partial windows
    expect(result[2]).toBeCloseTo(2, 5); // (1+2+3)/3 = 2
    expect(result[3]).toBeCloseTo(3, 5); // (2+3+4)/3 = 3
    expect(result[9]).toBeCloseTo(9, 5); // (8+9+10)/3 = 9
  });

  it('handles window size of 1', async () => {
    const data = [1, 2, 3, 4, 5];
    const result = await sma(data, 1);

    expect(result).toEqual(data);
  });

  it('returns same length as input', async () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const result = await sma(data, 5);

    expect(result.length).toBe(data.length);
  });
});

describe('ema (Exponential Moving Average)', () => {
  it('gives more weight to recent values', async () => {
    const data = [10, 10, 10, 10, 20]; // Sudden jump
    const smaResult = await sma(data, 3);
    const emaResult = await ema(data, 3);

    // EMA should react faster to the jump
    expect(emaResult[4]).toBeGreaterThan(smaResult[4]);
  });

  it('returns same length as input', async () => {
    const data = [1, 2, 3, 4, 5];
    const result = await ema(data, 3);

    expect(result.length).toBe(data.length);
  });
});

describe('wma (Weighted Moving Average)', () => {
  it('returns same length as input', async () => {
    const data = [1, 2, 3, 4, 5, 6, 7];
    const result = await wma(data, 3);

    expect(result.length).toBe(data.length);
  });
});

describe('movingAverage', () => {
  it('supports all types via options', async () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    const smaResult = await movingAverage(data, { window: 3, type: 'sma' });
    const emaResult = await movingAverage(data, { window: 3, type: 'ema' });
    const wmaResult = await movingAverage(data, { window: 3, type: 'wma' });

    expect(smaResult.length).toBe(data.length);
    expect(emaResult.length).toBe(data.length);
    expect(wmaResult.length).toBe(data.length);
  });
});

// ============================================================================
// Trend Analysis & Forecasting
// ============================================================================

describe('trendForecast', () => {
  it('detects upward trend', async () => {
    const data = [10, 20, 30, 40, 50];
    const result = await trendForecast(data, 3);

    expect(result.direction).toBe('up');
    expect(result.slope).toBeGreaterThan(0);
    expect(result.strength).toBeCloseTo(1, 2);
  });

  it('detects downward trend', async () => {
    const data = [50, 40, 30, 20, 10];
    const result = await trendForecast(data, 3);

    expect(result.direction).toBe('down');
    expect(result.slope).toBeLessThan(0);
  });

  it('detects flat trend', async () => {
    const data = [50, 50, 50, 50, 50];
    const result = await trendForecast(data, 3);

    expect(result.direction).toBe('flat');
    expect(result.slope).toBeCloseTo(0, 5);
  });

  it('forecasts correct number of periods', async () => {
    const data = [10, 20, 30, 40, 50];
    const result = await trendForecast(data, 5);
    const forecast = result.getForecast();

    expect(forecast.length).toBe(5);
  });

  it('forecasts reasonable values', async () => {
    const data = [10, 20, 30, 40, 50];
    const result = await trendForecast(data, 3);
    const forecast = result.getForecast();

    expect(forecast[0]).toBeCloseTo(60, 1);
    expect(forecast[1]).toBeCloseTo(70, 1);
    expect(forecast[2]).toBeCloseTo(80, 1);
  });
});

describe('rateOfChange', () => {
  it('calculates percentage change', async () => {
    const data = [100, 110, 121, 133.1]; // 10% growth each period
    const result = await rateOfChange(data, 1);

    expect(result[1]).toBeCloseTo(10, 1); // 10%
    expect(result[2]).toBeCloseTo(10, 1); // 10%
  });

  it('returns correct length', async () => {
    const data = [100, 110, 120, 130, 140];
    const result = await rateOfChange(data, 2);

    expect(result.length).toBe(data.length);
  });
});

describe('momentum', () => {
  it('calculates difference from n periods ago', async () => {
    const data = [10, 15, 20, 25, 30];
    const result = await momentum(data, 1);

    expect(result[1]).toBeCloseTo(5, 5);
    expect(result[2]).toBeCloseTo(5, 5);
    expect(result[3]).toBeCloseTo(5, 5);
  });

  it('handles larger periods', async () => {
    const data = [10, 20, 30, 40, 50];
    const result = await momentum(data, 2);

    expect(result[2]).toBeCloseTo(20, 5); // 30 - 10
    expect(result[4]).toBeCloseTo(20, 5); // 50 - 30
  });
});

describe('exponentialSmoothing', () => {
  it('smooths noisy data', async () => {
    const noisy = [10, 15, 8, 20, 12, 18, 9, 22];
    const smoothed = await exponentialSmoothing(noisy, { alpha: 0.3 });

    expect(smoothed.length).toBe(noisy.length);

    // Smoothed values should have less variance
    const noisyVariance = variance(noisy);
    const smoothedVariance = variance(smoothed);
    expect(smoothedVariance).toBeLessThan(noisyVariance);
  });

  it('uses default alpha of 0.3', async () => {
    const data = [10, 20, 30];
    const result = await exponentialSmoothing(data);

    expect(result.length).toBe(data.length);
  });
});

// ============================================================================
// Utility Functions
// ============================================================================

describe('findPeaks', () => {
  it('finds local maxima', async () => {
    const data = [1, 3, 2, 5, 1, 4, 2];
    const peaks = await findPeaks(data);

    expect(peaks).toContain(1); // value 3
    expect(peaks).toContain(3); // value 5
    expect(peaks).toContain(5); // value 4
  });

  it('returns empty array for monotonic data', async () => {
    const data = [1, 2, 3, 4, 5];
    const peaks = await findPeaks(data);

    expect(peaks.length).toBe(0);
  });
});

describe('findTroughs', () => {
  it('finds local minima', async () => {
    const data = [5, 2, 4, 1, 3, 0, 2];
    const troughs = await findTroughs(data);

    expect(troughs).toContain(1); // value 2
    expect(troughs).toContain(3); // value 1
    expect(troughs).toContain(5); // value 0
  });

  it('returns empty array for monotonic data', async () => {
    const data = [5, 4, 3, 2, 1];
    const troughs = await findTroughs(data);

    expect(troughs.length).toBe(0);
  });
});

// ============================================================================
// Convenience Functions
// ============================================================================

describe('predict', () => {
  it('fits and predicts in one call', async () => {
    const predictions = await predict([1, 2, 3, 4], [2, 4, 6, 8], [5, 6]);

    expect(predictions[0]).toBeCloseTo(10, 5);
    expect(predictions[1]).toBeCloseTo(12, 5);
  });
});

describe('trendLine', () => {
  it('returns model and future predictions', async () => {
    const data = [10, 20, 30, 40, 50];
    const result = await trendLine(data, 3);

    expect(result.model.slope).toBeCloseTo(10, 5);
    expect(result.trend.length).toBe(3);
    expect(result.trend[0]).toBeCloseTo(60, 1);
    expect(result.trend[1]).toBeCloseTo(70, 1);
    expect(result.trend[2]).toBeCloseTo(80, 1);
  });
});

// ============================================================================
// Error Metrics
// ============================================================================

describe('rmse', () => {
  it('returns 0 for perfect predictions', () => {
    expect(rmse([1, 2, 3], [1, 2, 3])).toBe(0);
  });

  it('calculates RMSE correctly', () => {
    // errors: 1, -1, 1 → squared: 1, 1, 1 → mean: 1 → sqrt: 1
    expect(rmse([1, 2, 3], [2, 1, 4])).toBeCloseTo(1, 5);
  });

  it('throws on mismatched lengths', () => {
    expect(() => rmse([1, 2], [1])).toThrow('same length');
  });
});

describe('mae', () => {
  it('returns 0 for perfect predictions', () => {
    expect(mae([1, 2, 3], [1, 2, 3])).toBe(0);
  });

  it('calculates MAE correctly', () => {
    // errors: |1|, |-1|, |1| → mean: 1
    expect(mae([1, 2, 3], [2, 1, 4])).toBeCloseTo(1, 5);
  });
});

describe('mape', () => {
  it('returns 0 for perfect predictions', () => {
    expect(mape([100, 200, 300], [100, 200, 300])).toBe(0);
  });

  it('calculates percentage error correctly', () => {
    // 10% off on each: (10/100 + 10/200 + 10/300) / 3 * 100
    expect(mape([100, 200, 300], [110, 190, 310])).toBeCloseTo(5.56, 1);
  });

  it('skips zero actual values', () => {
    expect(mape([0, 100], [10, 110])).toBeCloseTo(10, 5);
  });
});

describe('errorMetrics', () => {
  it('returns all metrics at once', () => {
    const metrics = errorMetrics([1, 2, 3], [1, 2, 3]);
    expect(metrics.rmse).toBe(0);
    expect(metrics.mae).toBe(0);
    expect(metrics.mape).toBe(0);
    expect(metrics.n).toBe(3);
  });
});

describe('residuals', () => {
  it('returns zero residuals for perfect fit', () => {
    const result = residuals([1, 2, 3], [1, 2, 3]);
    expect(result.residuals).toEqual([0, 0, 0]);
    expect(result.mean).toBe(0);
    expect(result.stdDev).toBe(0);
  });

  it('calculates residuals correctly', () => {
    const result = residuals([10, 20, 30], [12, 18, 30]);
    expect(result.residuals).toEqual([-2, 2, 0]);
    expect(result.mean).toBeCloseTo(0, 5);
  });

  it('computes standardized residuals', () => {
    const result = residuals([10, 20, 30], [12, 18, 30]);
    // All standardized residuals should have mean ~0 and std ~1
    const stdMean = result.standardized.reduce((a, b) => a + b, 0) / result.standardized.length;
    expect(stdMean).toBeCloseTo(0, 5);
  });

  it('throws on mismatched lengths', () => {
    expect(() => residuals([1, 2], [1])).toThrow('same length');
  });
});

// ============================================================================
// Edge Cases
// ============================================================================

describe('edge cases', () => {
  it('handles two data points', async () => {
    const model = await linearRegression([1, 2], [10, 20]);

    expect(model.slope).toBeCloseTo(10, 5);
    expect(model.n).toBe(2);
  });

  it('handles large datasets', async () => {
    const n = 10000;
    const x = Array.from({ length: n }, (_, i) => i);
    const y = x.map(xi => 2 * xi + 5 + Math.random() * 0.1);

    const model = await linearRegression(x, y);

    expect(model.slope).toBeCloseTo(2, 1);
    expect(model.intercept).toBeCloseTo(5, 0); // 0 decimal places (tolerance ~0.5)
    expect(model.n).toBe(n);
  });

  it('handles negative values', async () => {
    const model = await linearRegression([-5, -2, 0, 2, 5], [-10, -4, 0, 4, 10]);

    expect(model.slope).toBeCloseTo(2, 5);
    expect(model.intercept).toBeCloseTo(0, 5);
  });

  it('handles decimal values', async () => {
    const model = await linearRegression([0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.6, 0.8]);

    expect(model.slope).toBeCloseTo(2, 5);
  });
});

// ============================================================================
// Helpers
// ============================================================================

function variance(data: number[]): number {
  const mean = data.reduce((a, b) => a + b, 0) / data.length;
  return data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
}
