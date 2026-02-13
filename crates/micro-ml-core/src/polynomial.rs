use wasm_bindgen::prelude::*;
use crate::error::MlError;

/// Result of a polynomial regression fit: y = c0 + c1*x + c2*x² + ...
#[derive(Clone)]
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

/// Internal implementation
pub fn polynomial_regression_impl(x: &[f64], y: &[f64], degree: usize) -> Result<PolynomialModel, MlError> {
    if x.len() != y.len() {
        return Err(MlError::new("x and y arrays must have the same length"));
    }

    let n = x.len();
    if n <= degree {
        return Err(MlError::new(format!(
            "Need at least {} data points for degree {} polynomial",
            degree + 1,
            degree
        )));
    }

    if degree == 0 {
        return Err(MlError::new("Degree must be at least 1"));
    }

    let m = degree + 1; // Number of coefficients

    // Build X^T X matrix (m x m)
    let mut xtx = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            for k in 0..n {
                xtx[i][j] += x[k].powi((i + j) as i32);
            }
        }
    }

    // Build X^T y vector
    let mut xty = vec![0.0; m];
    for i in 0..m {
        for k in 0..n {
            xty[i] += x[k].powi(i as i32) * y[k];
        }
    }

    // Solve the system
    let coefficients = solve_linear_system(xtx, xty)
        .ok_or_else(|| MlError::new("Failed to solve: matrix is singular or near-singular"))?;

    // Calculate R-squared
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;

    for i in 0..n {
        let mut y_pred = 0.0;
        for (j, &coef) in coefficients.iter().enumerate() {
            y_pred += coef * x[i].powi(j as i32);
        }
        ss_res += (y[i] - y_pred).powi(2);
        ss_tot += (y[i] - y_mean).powi(2);
    }

    let r_squared = if ss_tot == 0.0 { 1.0 } else { 1.0 - (ss_res / ss_tot) };

    Ok(PolynomialModel {
        coefficients,
        degree,
        r_squared,
        n,
    })
}

/// Fit a polynomial regression model using the normal equations
/// Solves: (X^T X) β = X^T y where X is the Vandermonde matrix
#[wasm_bindgen(js_name = "polynomialRegression")]
pub fn polynomial_regression(x: &[f64], y: &[f64], degree: usize) -> Result<PolynomialModel, JsError> {
    polynomial_regression_impl(x, y, degree).map_err(|e| JsError::new(&e.message))
}

/// Polynomial regression with auto-generated x values (0, 1, 2, ...)
#[wasm_bindgen(js_name = "polynomialRegressionSimple")]
pub fn polynomial_regression_simple(y: &[f64], degree: usize) -> Result<PolynomialModel, JsError> {
    let x: Vec<f64> = (0..y.len()).map(|i| i as f64).collect();
    polynomial_regression(&x, y, degree)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadratic_fit() {
        // y = x² + 2x + 1 = (x + 1)²
        let x: Vec<f64> = (-5..=5).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi + 2.0 * xi + 1.0).collect();

        let model = polynomial_regression_impl(&x, &y, 2).unwrap();
        let coefs = model.get_coefficients();

        assert!((coefs[0] - 1.0).abs() < 1e-8);  // constant
        assert!((coefs[1] - 2.0).abs() < 1e-8);  // linear
        assert!((coefs[2] - 1.0).abs() < 1e-8);  // quadratic
        assert!((model.r_squared - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_cubic_fit() {
        // y = x³ - x
        let x: Vec<f64> = (-3..=3).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.powi(3) - xi).collect();

        let model = polynomial_regression_impl(&x, &y, 3).unwrap();
        let coefs = model.get_coefficients();

        assert!((coefs[0]).abs() < 1e-8);         // constant
        assert!((coefs[1] - (-1.0)).abs() < 1e-8); // linear
        assert!((coefs[2]).abs() < 1e-8);         // quadratic
        assert!((coefs[3] - 1.0).abs() < 1e-8);   // cubic
    }

    #[test]
    fn test_prediction() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // (x + 1)²

        let model = polynomial_regression_impl(&x, &y, 2).unwrap();

        assert!((model.predict_one(5.0) - 36.0).abs() < 1e-6);
    }

    #[test]
    fn test_insufficient_data() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 4.0];

        assert!(polynomial_regression_impl(&x, &y, 2).is_err());
    }
}
