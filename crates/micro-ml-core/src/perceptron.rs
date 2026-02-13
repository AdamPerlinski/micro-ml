use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, mat_get};

#[wasm_bindgen]
pub struct PerceptronModel {
    n_features: usize,
    weights: Vec<f64>,
    bias: f64,
    iterations: usize,
    converged: bool,
}

#[wasm_bindgen]
impl PerceptronModel {
    #[wasm_bindgen(getter)]
    pub fn bias(&self) -> f64 { self.bias }

    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> usize { self.iterations }

    #[wasm_bindgen(getter)]
    pub fn converged(&self) -> bool { self.converged }

    #[wasm_bindgen(js_name = "getWeights")]
    pub fn get_weights(&self) -> Vec<f64> { self.weights.clone() }

    #[wasm_bindgen]
    pub fn predict(&self, data: &[f64]) -> Vec<u32> {
        let n = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let mut sum = self.bias;
            for j in 0..self.n_features {
                sum += self.weights[j] * data[i * self.n_features + j];
            }
            result.push(if sum >= 0.0 { 1 } else { 0 });
        }
        result
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("Perceptron(converged={}, iterations={})", self.converged, self.iterations)
    }
}

pub fn perceptron_impl(data: &[f64], n_features: usize, labels: &[f64], lr: f64, max_iter: usize) -> Result<PerceptronModel, MlError> {
    let n = validate_matrix(data, n_features)?;
    if labels.len() != n {
        return Err(MlError::new("labels length must match number of samples"));
    }

    // Convert 0/1 labels to -1/+1
    let y: Vec<f64> = labels.iter().map(|&l| if l > 0.5 { 1.0 } else { -1.0 }).collect();

    let mut weights = vec![0.0; n_features];
    let mut bias = 0.0;
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;
        let mut errors = 0;

        for i in 0..n {
            let mut sum = bias;
            for j in 0..n_features {
                sum += weights[j] * mat_get(data, n_features, i, j);
            }
            let pred = if sum >= 0.0 { 1.0 } else { -1.0 };

            if pred != y[i] {
                errors += 1;
                for j in 0..n_features {
                    weights[j] += lr * y[i] * mat_get(data, n_features, i, j);
                }
                bias += lr * y[i];
            }
        }

        if errors == 0 {
            converged = true;
            break;
        }
    }

    Ok(PerceptronModel { n_features, weights, bias, iterations, converged })
}

#[wasm_bindgen(js_name = "perceptron")]
pub fn perceptron(data: &[f64], n_features: usize, labels: &[f64], lr: f64, max_iter: usize) -> Result<PerceptronModel, JsError> {
    perceptron_impl(data, n_features, labels, lr, max_iter).map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linearly_separable() {
        let data = vec![
            0.0, 0.0,  0.5, 0.5,  1.0, 0.0,  // class 0
            3.0, 3.0,  3.5, 3.5,  4.0, 3.0,  // class 1
        ];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let model = perceptron_impl(&data, 2, &labels, 0.1, 1000).unwrap();
        assert!(model.converged);

        let preds = model.predict(&data);
        assert_eq!(preds, vec![0, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn test_xor_does_not_converge() {
        let data = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        let model = perceptron_impl(&data, 2, &labels, 0.1, 100).unwrap();
        assert!(!model.converged);
    }
}
