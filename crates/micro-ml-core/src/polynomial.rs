use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use crate::error::MlError;

/// Result of a polynomial regression fit: y = c0 + c1*x + c2*x² + ...
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct PolynomialModel {
    coefficients: Vec<f64>,
    degree: usize,
    r_squared: f64,
    n: usize,
}

#[wasm_bindgen]
impl PolynomialModel {
    /// Get the polynomial degree
    #[wasm_bindgen(getter)]
    pub fn degree(&self) -> usize {
        self.degree
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

    /// Get the coefficients as an array [c0, c1, c2, ...]
    #[wasm_bindgen(js_name = "getCoefficients")]
    pub fn get_coefficients(&self) -> Vec<f64> {
        self.coefficients.clone()
    }

    /// Predict a single value using Horner's method for numerical stability
    pub fn predict_one(&self, x: f64) -> f64 {
        let mut result = 0.0;
        for i in (0..=self.degree).rev() {
            result = result * x + self.coefficients[i];
        }
        result
    }

    /// Predict multiple values
    #[wasm_bindgen(js_name = "predict")]
    pub fn predict(&self, x_values: &[f64]) -> Vec<f64> {
        x_values.iter().map(|&x| self.predict_one(x)).collect()
    }

    /// Get the equation as a string
    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        let mut terms: Vec<String> = Vec::new();

        for (i, &coef) in self.coefficients.iter().enumerate() {
            if coef.abs() < 1e-10 {
                continue;
            }

            let term = match i {
                0 => format!("{:.6}", coef),
                1 => format!("{:.6}x", coef),
                _ => format!("{:.6}x^{}", coef, i),
            };
            terms.push(term);
        }

        if terms.is_empty() {
            "y = 0".to_string()
        } else {
            format!("y = {}", terms.join(" + "))
        }
    }
}

/// Solve a system of linear equations using Gaussian elimination with partial pivoting
/// Returns the solution vector x for Ax = b
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();

    // Forward elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if a[k][i].abs() > a[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows
        a.swap(i, max_row);
        b.swap(i, max_row);

        // Check for singular matrix
        if a[i][i].abs() < 1e-12 {
            return None;
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = a[k][i] / a[i][i];
            b[k] -= factor * b[i];
            for j in i..n {
                a[k][j] -= factor * a[i][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }

    Some(x)
}
