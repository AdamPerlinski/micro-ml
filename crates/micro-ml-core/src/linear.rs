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
