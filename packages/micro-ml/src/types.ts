/**
 * Result of a linear regression fit: y = slope * x + intercept
 */
export interface LinearModel {
  /** Slope (m in y = mx + b) */
  readonly slope: number;
  /** Intercept (b in y = mx + b) */
  readonly intercept: number;
  /** Coefficient of determination (0-1, higher is better fit) */
  readonly rSquared: number;
  /** Number of data points used in fitting */
  readonly n: number;

  /** Predict values for given x coordinates */
  predict(x: number[]): number[];
  /** Get equation as string */
  toString(): string;
}

/**
 * Result of a polynomial regression fit
 */
export interface PolynomialModel {
  /** Polynomial degree */
  readonly degree: number;
  /** Coefficient of determination (0-1) */
  readonly rSquared: number;
  /** Number of data points used in fitting */
  readonly n: number;

  /** Get coefficients [c0, c1, c2, ...] for y = c0 + c1*x + c2*x² + ... */
  getCoefficients(): number[];
  /** Predict values for given x coordinates */
  predict(x: number[]): number[];
  /** Get equation as string */
  toString(): string;
}

/**
 * Result of an exponential regression fit: y = a * e^(b*x)
 */
export interface ExponentialModel {
  /** Amplitude (a in y = a * e^(bx)) */
  readonly a: number;
  /** Growth rate (b in y = a * e^(bx)) */
  readonly b: number;
  /** Coefficient of determination (0-1) */
  readonly rSquared: number;
  /** Number of data points used in fitting */
  readonly n: number;

  /** Predict values for given x coordinates */
  predict(x: number[]): number[];
  /** Get equation as string */
  toString(): string;
  /** Get doubling time (if b > 0) or half-life (if b < 0) */
  doublingTime(): number;
}

/**
 * Result of a logarithmic regression fit: y = a + b * ln(x)
 */
export interface LogarithmicModel {
  /** Intercept (a in y = a + b*ln(x)) */
  readonly a: number;
  /** Coefficient (b in y = a + b*ln(x)) */
  readonly b: number;
  /** Coefficient of determination (0-1) */
  readonly rSquared: number;
  /** Number of data points used in fitting */
  readonly n: number;

  /** Predict values for given x coordinates */
  predict(x: number[]): number[];
  /** Get equation as string */
  toString(): string;
}

/**
 * Result of a power regression fit: y = a * x^b
 */
export interface PowerModel {
  /** Coefficient (a in y = a * x^b) */
  readonly a: number;
  /** Exponent (b in y = a * x^b) */
  readonly b: number;
  /** Coefficient of determination (0-1) */
  readonly rSquared: number;
  /** Number of data points used in fitting */
  readonly n: number;

  /** Predict values for given x coordinates */
  predict(x: number[]): number[];
  /** Get equation as string */
  toString(): string;
}

/**
 * Type of moving average
 */
export type MovingAverageType = 'sma' | 'ema' | 'wma';

/**
 * Options for moving average calculation
 */
export interface MovingAverageOptions {
  /** Window size */
  window: number;
  /** Type of moving average (default: 'sma') */
  type?: MovingAverageType;
}

/**
 * Trend direction
 */
export type TrendDirection = 'up' | 'down' | 'flat';

/**
 * Result of trend analysis
 */
export interface TrendAnalysis {
  /** Trend direction */
  readonly direction: TrendDirection;
  /** Slope (rate of change per period) */
  readonly slope: number;
  /** Trend strength (0-1, based on R²) */
  readonly strength: number;

  /** Get forecasted values */
  getForecast(): number[];
}

/**
 * Options for polynomial regression
 */
export interface PolynomialOptions {
  /** Polynomial degree (default: 2) */
  degree?: number;
}

/**
 * Options for exponential smoothing
 */
export interface SmoothingOptions {
  /** Smoothing factor (0-1, default: 0.3) */
  alpha?: number;
}

/**
 * Statistical error metrics for evaluating model accuracy
 */
export interface ErrorMetrics {
  /** Root Mean Squared Error */
  readonly rmse: number;
  /** Mean Absolute Error */
  readonly mae: number;
  /** Mean Absolute Percentage Error (as percentage, 0-100+) */
  readonly mape: number;
  /** Number of data points compared */
  readonly n: number;
}

/**
 * Result of residuals analysis
 */
export interface ResidualsResult {
  /** Raw residuals (actual - predicted) */
  readonly residuals: number[];
  /** Mean of residuals (should be ~0 for unbiased model) */
  readonly mean: number;
  /** Standard deviation of residuals */
  readonly stdDev: number;
  /** Standardized residuals (residual / stdDev) */
  readonly standardized: number[];
}
