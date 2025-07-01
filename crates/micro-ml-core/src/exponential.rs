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
