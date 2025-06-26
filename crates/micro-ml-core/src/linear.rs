use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use crate::error::MlError;

/// Result of a linear regression fit: y = slope * x + intercept
#[derive(Debug, Clone, Serialize, Deserialize)]
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
