use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use crate::error::MlError;

/// Result of an exponential regression fit: y = a * e^(b*x)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct ExponentialModel {
    a: f64,      // Initial value / amplitude
    b: f64,      // Growth rate
    r_squared: f64,
    n: usize,
}

#[wasm_bindgen]
impl ExponentialModel {
    /// Get the amplitude (a in y = a * e^(bx))
    #[wasm_bindgen(getter)]
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Get the growth rate (b in y = a * e^(bx))
    #[wasm_bindgen(getter)]
    pub fn b(&self) -> f64 {
        self.b
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
        self.a * (self.b * x).exp()
    }

    /// Predict multiple values
    #[wasm_bindgen(js_name = "predict")]
    pub fn predict(&self, x_values: &[f64]) -> Vec<f64> {
        x_values.iter().map(|&x| self.predict_one(x)).collect()
    }

    /// Get the equation as a string
    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("y = {:.6} * e^({:.6}x)", self.a, self.b)
    }

    /// Calculate the doubling time (if b > 0) or half-life (if b < 0)
    #[wasm_bindgen(js_name = "doublingTime")]
    pub fn doubling_time(&self) -> f64 {
        if self.b == 0.0 {
            f64::INFINITY
        } else {
            (2.0_f64.ln() / self.b).abs()
        }
    }
}

/// Internal implementation
pub fn exponential_regression_impl(x: &[f64], y: &[f64]) -> Result<ExponentialModel, MlError> {
    if x.len() != y.len() {
        return Err(MlError::new("x and y arrays must have the same length"));
    }

    let n = x.len();
    if n < 2 {
        return Err(MlError::new("Need at least 2 data points for exponential regression"));
    }

    // Check for non-positive y values
    for (i, &yi) in y.iter().enumerate() {
        if yi <= 0.0 {
            return Err(MlError::new(format!(
                "All y values must be positive for exponential regression (y[{}] = {})",
                i, yi
            )));
        }
    }

    // Transform y values to ln(y) and fit linear regression
    let ln_y: Vec<f64> = y.iter().map(|&yi| yi.ln()).collect();

    // Calculate means
    let x_mean: f64 = x.iter().sum::<f64>() / n as f64;
    let ln_y_mean: f64 = ln_y.iter().sum::<f64>() / n as f64;

    // Linear regression on transformed data
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        let x_diff = x[i] - x_mean;
        let y_diff = ln_y[i] - ln_y_mean;
        numerator += x_diff * y_diff;
        denominator += x_diff * x_diff;
    }

    if denominator == 0.0 {
        return Err(MlError::new("Cannot fit regression: all x values are identical"));
    }

    let b = numerator / denominator;
    let ln_a = ln_y_mean - b * x_mean;
    let a = ln_a.exp();

    // Calculate R-squared using original y values
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;

    for i in 0..n {
        let y_pred = a * (b * x[i]).exp();
        ss_res += (y[i] - y_pred).powi(2);
        ss_tot += (y[i] - y_mean).powi(2);
    }

    let r_squared = if ss_tot == 0.0 { 1.0 } else { 1.0 - (ss_res / ss_tot) };

    Ok(ExponentialModel {
        a,
        b,
        r_squared,
        n,
    })
}

/// Fit an exponential regression model: y = a * e^(bx)
/// Uses linearization: ln(y) = ln(a) + bx
#[wasm_bindgen(js_name = "exponentialRegression")]
pub fn exponential_regression(x: &[f64], y: &[f64]) -> Result<ExponentialModel, JsError> {
    exponential_regression_impl(x, y).map_err(|e| JsError::new(&e.message))
}

/// Exponential regression with auto-generated x values (0, 1, 2, ...)
#[wasm_bindgen(js_name = "exponentialRegressionSimple")]
pub fn exponential_regression_simple(y: &[f64]) -> Result<ExponentialModel, JsError> {
    let x: Vec<f64> = (0..y.len()).map(|i| i as f64).collect();
    exponential_regression(&x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_fit() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * (0.5 * xi).exp()).collect();
        let model = exponential_regression_impl(&x, &y).unwrap();
        assert!((model.a - 2.0).abs() < 1e-6);
        assert!((model.b - 0.5).abs() < 1e-6);
        assert!((model.r_squared - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_decay() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 100.0 * (-0.1 * xi).exp()).collect();
        let model = exponential_regression_impl(&x, &y).unwrap();
        assert!((model.a - 100.0).abs() < 1e-4);
        assert!((model.b - (-0.1)).abs() < 1e-6);
    }

    #[test]
    fn test_doubling_time() {
        let model = ExponentialModel { a: 1.0, b: 2.0_f64.ln(), r_squared: 1.0, n: 10 };
        assert!((model.doubling_time() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_non_positive_error() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, -2.0, 3.0];
        assert!(exponential_regression_impl(&x, &y).is_err());
    }
}
