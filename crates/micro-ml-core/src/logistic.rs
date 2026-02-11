use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, mat_get};

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

#[wasm_bindgen]
pub struct LogisticModel {
    n_features: usize,
    weights: Vec<f64>,
    bias: f64,
    iterations: usize,
    loss: f64,
}

#[wasm_bindgen]
impl LogisticModel {
    #[wasm_bindgen(getter)]
    pub fn bias(&self) -> f64 { self.bias }

    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> usize { self.iterations }

    #[wasm_bindgen(getter)]
    pub fn loss(&self) -> f64 { self.loss }

    #[wasm_bindgen(js_name = "getWeights")]
    pub fn get_weights(&self) -> Vec<f64> { self.weights.clone() }

    #[wasm_bindgen(js_name = "predictProba")]
    pub fn predict_proba(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let mut z = self.bias;
            for j in 0..self.n_features {
                z += self.weights[j] * data[i * self.n_features + j];
            }
            result.push(sigmoid(z));
        }
        result
    }

    #[wasm_bindgen]
    pub fn predict(&self, data: &[f64]) -> Vec<u32> {
        self.predict_proba(data).iter().map(|&p| if p >= 0.5 { 1 } else { 0 }).collect()
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("LogisticRegression(loss={:.6}, iterations={})", self.loss, self.iterations)
    }
}

pub fn logistic_regression_impl(data: &[f64], n_features: usize, labels: &[f64], lr: f64, max_iter: usize, lambda: f64) -> Result<LogisticModel, MlError> {
    let n = validate_matrix(data, n_features)?;
    if labels.len() != n {
        return Err(MlError::new("labels length must match number of samples"));
    }

    let mut weights = vec![0.0; n_features];
    let mut bias = 0.0;
    let n_f = n as f64;

    for _ in 0..max_iter {
        let mut grad_w = vec![0.0; n_features];
        let mut grad_b = 0.0;

        for i in 0..n {
            let mut z = bias;
            for j in 0..n_features {
                z += weights[j] * mat_get(data, n_features, i, j);
            }
            let pred = sigmoid(z);
            let error = pred - labels[i];
            grad_b += error;
            for j in 0..n_features {
                grad_w[j] += error * mat_get(data, n_features, i, j);
            }
        }

        // Update with L2 regularization
        for j in 0..n_features {
            weights[j] -= lr * (grad_w[j] / n_f + lambda * weights[j]);
        }
        bias -= lr * grad_b / n_f;
    }

    // Compute final loss (binary cross-entropy)
    let mut loss = 0.0;
    for i in 0..n {
        let mut z = bias;
        for j in 0..n_features {
            z += weights[j] * mat_get(data, n_features, i, j);
        }
        let p = sigmoid(z).clamp(1e-15, 1.0 - 1e-15);
        loss -= labels[i] * p.ln() + (1.0 - labels[i]) * (1.0 - p).ln();
    }
    loss /= n_f;

    Ok(LogisticModel { n_features, weights, bias, iterations: max_iter, loss })
}

#[wasm_bindgen(js_name = "logisticRegression")]
pub fn logistic_regression_wasm(data: &[f64], n_features: usize, labels: &[f64], lr: f64, max_iter: usize, lambda: f64) -> Result<LogisticModel, JsError> {
    logistic_regression_impl(data, n_features, labels, lr, max_iter, lambda).map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linearly_separable() {
        let data = vec![
            0.0, 0.0,  0.5, 0.5,  1.0, 0.0,
            5.0, 5.0,  5.5, 5.5,  6.0, 5.0,
        ];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let model = logistic_regression_impl(&data, 2, &labels, 0.1, 1000, 0.0).unwrap();

        let preds = model.predict(&data);
        assert_eq!(preds, vec![0, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn test_probabilities() {
        let data = vec![0.0, 5.0];
        let labels = vec![0.0, 1.0];
        let model = logistic_regression_impl(&data, 1, &labels, 0.5, 1000, 0.0).unwrap();
        let proba = model.predict_proba(&vec![0.0, 5.0]);
        assert!(proba[0] < 0.5);
        assert!(proba[1] > 0.5);
    }
}
