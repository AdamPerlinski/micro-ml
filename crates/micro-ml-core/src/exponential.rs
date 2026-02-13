use wasm_bindgen::prelude::*;
use crate::error::MlError;

/// Result of an exponential regression fit: y = a * e^(b*x)
#[derive(Clone)]
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


/// Result of a logarithmic regression fit: y = a + b * ln(x)
#[derive(Clone)]
#[wasm_bindgen]
pub struct LogarithmicModel {
    a: f64,      // Intercept
    b: f64,      // Coefficient
    r_squared: f64,
    n: usize,
}

#[wasm_bindgen]
impl LogarithmicModel {
    /// Get the intercept (a in y = a + b*ln(x))
    #[wasm_bindgen(getter)]
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Get the coefficient (b in y = a + b*ln(x))
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
        self.a + self.b * x.ln()
    }

    /// Predict multiple values
    #[wasm_bindgen(js_name = "predict")]
    pub fn predict(&self, x_values: &[f64]) -> Vec<f64> {
        x_values.iter().map(|&x| self.predict_one(x)).collect()
    }

    /// Get the equation as a string
    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        if self.b >= 0.0 {
            format!("y = {:.6} + {:.6} * ln(x)", self.a, self.b)
        } else {
            format!("y = {:.6} - {:.6} * ln(x)", self.a, self.b.abs())
        }
    }
}

/// Fit a logarithmic regression model: y = a + b * ln(x)
#[wasm_bindgen(js_name = "logarithmicRegression")]
pub fn logarithmic_regression(x: &[f64], y: &[f64]) -> Result<LogarithmicModel, JsError> {
    if x.len() != y.len() {
        return Err(JsError::new("x and y arrays must have the same length"));
    }

    let n = x.len();
    if n < 2 {
        return Err(JsError::new("Need at least 2 data points for logarithmic regression"));
    }

    // Check for non-positive x values
    for (i, &xi) in x.iter().enumerate() {
        if xi <= 0.0 {
            return Err(JsError::new(&format!(
                "All x values must be positive for logarithmic regression (x[{}] = {})",
                i, xi
            )));
        }
    }

    // Transform x values to ln(x) and fit linear regression
    let ln_x: Vec<f64> = x.iter().map(|&xi| xi.ln()).collect();

    // Calculate means
    let ln_x_mean: f64 = ln_x.iter().sum::<f64>() / n as f64;
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;

    // Linear regression on transformed data
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        let x_diff = ln_x[i] - ln_x_mean;
        let y_diff = y[i] - y_mean;
        numerator += x_diff * y_diff;
        denominator += x_diff * x_diff;
    }

    if denominator == 0.0 {
        return Err(JsError::new("Cannot fit regression: all x values are identical"));
    }

    let b = numerator / denominator;
    let a = y_mean - b * ln_x_mean;

    // Calculate R-squared
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;

    for i in 0..n {
        let y_pred = a + b * ln_x[i];
        ss_res += (y[i] - y_pred).powi(2);
        ss_tot += (y[i] - y_mean).powi(2);
    }

    let r_squared = if ss_tot == 0.0 { 1.0 } else { 1.0 - (ss_res / ss_tot) };

    Ok(LogarithmicModel {
        a,
        b,
        r_squared,
        n,
    })
}


/// Result of a power regression fit: y = a * x^b
#[derive(Clone)]
#[wasm_bindgen]
pub struct PowerModel {
    a: f64,      // Coefficient
    b: f64,      // Exponent
    r_squared: f64,
    n: usize,
}

#[wasm_bindgen]
impl PowerModel {
    /// Get the coefficient (a in y = a * x^b)
    #[wasm_bindgen(getter)]
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Get the exponent (b in y = a * x^b)
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
        self.a * x.powf(self.b)
    }

    /// Predict multiple values
    #[wasm_bindgen(js_name = "predict")]
    pub fn predict(&self, x_values: &[f64]) -> Vec<f64> {
        x_values.iter().map(|&x| self.predict_one(x)).collect()
    }

    /// Get the equation as a string
    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("y = {:.6} * x^{:.6}", self.a, self.b)
    }
}

/// Fit a power regression model: y = a * x^b
/// Uses linearization: ln(y) = ln(a) + b*ln(x)
#[wasm_bindgen(js_name = "powerRegression")]
pub fn power_regression(x: &[f64], y: &[f64]) -> Result<PowerModel, JsError> {
    if x.len() != y.len() {
        return Err(JsError::new("x and y arrays must have the same length"));
    }

    let n = x.len();
    if n < 2 {
        return Err(JsError::new("Need at least 2 data points for power regression"));
    }

    // Check for non-positive values
    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        if xi <= 0.0 {
            return Err(JsError::new(&format!(
                "All x values must be positive for power regression (x[{}] = {})",
                i, xi
            )));
        }
        if yi <= 0.0 {
            return Err(JsError::new(&format!(
                "All y values must be positive for power regression (y[{}] = {})",
                i, yi
            )));
        }
    }

    // Transform to ln-ln space
    let ln_x: Vec<f64> = x.iter().map(|&xi| xi.ln()).collect();
    let ln_y: Vec<f64> = y.iter().map(|&yi| yi.ln()).collect();

    // Calculate means
    let ln_x_mean: f64 = ln_x.iter().sum::<f64>() / n as f64;
    let ln_y_mean: f64 = ln_y.iter().sum::<f64>() / n as f64;

    // Linear regression on transformed data
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        let x_diff = ln_x[i] - ln_x_mean;
        let y_diff = ln_y[i] - ln_y_mean;
        numerator += x_diff * y_diff;
        denominator += x_diff * x_diff;
    }

    if denominator == 0.0 {
        return Err(JsError::new("Cannot fit regression: all x values are identical"));
    }

    let b = numerator / denominator;
    let ln_a = ln_y_mean - b * ln_x_mean;
    let a = ln_a.exp();

    // Calculate R-squared using original y values
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;

    for i in 0..n {
        let y_pred = a * x[i].powf(b);
        ss_res += (y[i] - y_pred).powi(2);
        ss_tot += (y[i] - y_mean).powi(2);
    }

    let r_squared = if ss_tot == 0.0 { 1.0 } else { 1.0 - (ss_res / ss_tot) };

    Ok(PowerModel {
        a,
        b,
        r_squared,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_fit() {
        // y = 2 * e^(0.5x)
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * (0.5 * xi).exp()).collect();

        let model = exponential_regression_impl(&x, &y).unwrap();

        assert!((model.a - 2.0).abs() < 1e-6);
        assert!((model.b - 0.5).abs() < 1e-6);
        assert!((model.r_squared - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_decay() {
        // y = 100 * e^(-0.1x) (decay)
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 100.0 * (-0.1 * xi).exp()).collect();

        let model = exponential_regression_impl(&x, &y).unwrap();

        assert!((model.a - 100.0).abs() < 1e-4);
        assert!((model.b - (-0.1)).abs() < 1e-6);
    }

    #[test]
    fn test_doubling_time() {
        // y = 1 * e^(0.693x) should have doubling time of ~1
        let model = ExponentialModel {
            a: 1.0,
            b: 2.0_f64.ln(), // ln(2) ≈ 0.693
            r_squared: 1.0,
            n: 10,
        };

        assert!((model.doubling_time() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_non_positive_error() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, -2.0, 3.0];

        assert!(exponential_regression_impl(&x, &y).is_err());
    }
}
