use wasm_bindgen::prelude::*;
use crate::error::MlError;

/// Result of a linear regression fit: y = slope * x + intercept
#[derive(Clone)]
#[wasm_bindgen]
pub struct LinearModel {
    slope: f64,
    intercept: f64,
    r_squared: f64,
    n: usize,
}

#[wasm_bindgen]
impl LinearModel {
    /// Get the slope (m in y = mx + b)
    #[wasm_bindgen(getter)]
    pub fn slope(&self) -> f64 {
        self.slope
    }

    /// Get the intercept (b in y = mx + b)
    #[wasm_bindgen(getter)]
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Get the R-squared (coefficient of determination)
    #[wasm_bindgen(getter, js_name = "rSquared")]
    pub fn r_squared(&self) -> f64 {
        self.r_squared
    }

    /// Get the number of data points used in fitting
    #[wasm_bindgen(getter)]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Predict a single value
    pub fn predict_one(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }

    /// Predict multiple values
    #[wasm_bindgen(js_name = "predict")]
    pub fn predict(&self, x_values: &[f64]) -> Vec<f64> {
        x_values.iter().map(|&x| self.predict_one(x)).collect()
    }

    /// Get the equation as a string
    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        if self.intercept >= 0.0 {
            format!("y = {:.6}x + {:.6}", self.slope, self.intercept)
        } else {
            format!("y = {:.6}x - {:.6}", self.slope, self.intercept.abs())
        }
    }
}

/// Internal implementation that returns MlError (for testing)
/// Optimized single-pass algorithm using running sums
pub fn linear_regression_impl(x: &[f64], y: &[f64]) -> Result<LinearModel, MlError> {
    if x.len() != y.len() {
        return Err(MlError::new("x and y arrays must have the same length"));
    }

    let n = x.len();
    if n < 2 {
        return Err(MlError::new("Need at least 2 data points for linear regression"));
    }

    // Single pass: collect all sums at once
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    let mut sum_xy = 0.0;

    for i in 0..n {
        let xi = x[i];
        let yi = y[i];
        sum_x += xi;
        sum_y += yi;
        sum_xx += xi * xi;
        sum_yy += yi * yi;
        sum_xy += xi * yi;
    }

    let n_f = n as f64;

    // slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    let denominator = n_f * sum_xx - sum_x * sum_x;
    if denominator == 0.0 {
        return Err(MlError::new("Cannot fit regression: all x values are identical"));
    }

    let slope = (n_f * sum_xy - sum_x * sum_y) / denominator;
    let intercept = (sum_y - slope * sum_x) / n_f;

    // R² = (n*Σxy - Σx*Σy)² / [(n*Σx² - (Σx)²) * (n*Σy² - (Σy)²)]
    let ss_tot_factor = n_f * sum_yy - sum_y * sum_y;
    let r_squared = if ss_tot_factor == 0.0 {
        1.0
    } else {
        let numerator = n_f * sum_xy - sum_x * sum_y;
        (numerator * numerator) / (denominator * ss_tot_factor)
    };

    Ok(LinearModel {
        slope,
        intercept,
        r_squared,
        n,
    })
}

/// Fit a linear regression model using ordinary least squares
/// Uses the formula: slope = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
#[wasm_bindgen(js_name = "linearRegression")]
pub fn linear_regression(x: &[f64], y: &[f64]) -> Result<LinearModel, JsError> {
    linear_regression_impl(x, y).map_err(|e| JsError::new(&e.message))
}

/// Simple linear regression with auto-generated x values (0, 1, 2, ...)
/// Optimized: uses closed-form formulas for sequential x values (no allocation)
#[wasm_bindgen(js_name = "linearRegressionSimple")]
pub fn linear_regression_simple(y: &[f64]) -> Result<LinearModel, JsError> {
    let n = y.len();
    if n < 2 {
        return Err(JsError::new("Need at least 2 data points for linear regression"));
    }

    let n_f = n as f64;

    // For x = 0, 1, 2, ..., n-1:
    // Σx = n(n-1)/2
    // Σx² = n(n-1)(2n-1)/6
    let sum_x = n_f * (n_f - 1.0) / 2.0;
    let sum_xx = n_f * (n_f - 1.0) * (2.0 * n_f - 1.0) / 6.0;

    // Single pass for y values
    let mut sum_y = 0.0;
    let mut sum_yy = 0.0;
    let mut sum_xy = 0.0;

    for i in 0..n {
        let xi = i as f64;
        let yi = y[i];
        sum_y += yi;
        sum_yy += yi * yi;
        sum_xy += xi * yi;
    }

    let denominator = n_f * sum_xx - sum_x * sum_x;
    if denominator == 0.0 {
        return Err(JsError::new("Cannot fit regression: insufficient data variance"));
    }

    let slope = (n_f * sum_xy - sum_x * sum_y) / denominator;
    let intercept = (sum_y - slope * sum_x) / n_f;

    let ss_tot_factor = n_f * sum_yy - sum_y * sum_y;
    let r_squared = if ss_tot_factor == 0.0 {
        1.0
    } else {
        let numerator = n_f * sum_xy - sum_x * sum_y;
        (numerator * numerator) / (denominator * ss_tot_factor)
    };

    Ok(LinearModel {
        slope,
        intercept,
        r_squared,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_linear_fit() {
        // y = 2x + 1
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        let model = linear_regression_impl(&x, &y).unwrap();

        assert!((model.slope - 2.0).abs() < 1e-10);
        assert!((model.intercept - 1.0).abs() < 1e-10);
        assert!((model.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_prediction() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        let model = linear_regression_impl(&x, &y).unwrap();

        assert!((model.predict_one(6.0) - 13.0).abs() < 1e-10);
        assert!((model.predict_one(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_simple_regression() {
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let x: Vec<f64> = (0..y.len()).map(|i| i as f64).collect();
        let model = linear_regression_impl(&x, &y).unwrap();

        assert!((model.slope - 2.0).abs() < 1e-10);
        assert!((model.intercept - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_length_mismatch() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];

        assert!(linear_regression_impl(&x, &y).is_err());
    }

    #[test]
    fn test_insufficient_data() {
        let x = vec![1.0];
        let y = vec![1.0];

        assert!(linear_regression_impl(&x, &y).is_err());
    }
}
