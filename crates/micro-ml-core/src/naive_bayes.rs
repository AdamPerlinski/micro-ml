use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, mat_get};

#[wasm_bindgen]
pub struct NaiveBayesModel {
    n_classes: usize,
    n_features: usize,
    means: Vec<f64>,
    variances: Vec<f64>,
    class_priors: Vec<f64>,
    classes: Vec<u32>,
}

#[wasm_bindgen]
impl NaiveBayesModel {
    #[wasm_bindgen(getter, js_name = "nClasses")]
    pub fn n_classes(&self) -> usize { self.n_classes }

    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize { self.n_features }

    #[wasm_bindgen]
    pub fn predict(&self, data: &[f64]) -> Vec<u32> {
        let n_test = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n_test);

        for i in 0..n_test {
            let mut best_class = 0u32;
            let mut best_log_prob = f64::NEG_INFINITY;

            for c in 0..self.n_classes {
                let mut log_prob = self.class_priors[c].ln();
                for j in 0..self.n_features {
                    let x = data[i * self.n_features + j];
                    let mean = self.means[c * self.n_features + j];
                    let var = self.variances[c * self.n_features + j];
                    // Gaussian log-likelihood
                    log_prob += -0.5 * ((x - mean).powi(2) / var + var.ln() + std::f64::consts::LN_2 + std::f64::consts::PI.ln());
                }
                if log_prob > best_log_prob {
                    best_log_prob = log_prob;
                    best_class = self.classes[c];
                }
            }
            result.push(best_class);
        }
        result
    }

    #[wasm_bindgen(js_name = "predictProba")]
    pub fn predict_proba(&self, data: &[f64]) -> Vec<f64> {
        let n_test = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n_test * self.n_classes);

        for i in 0..n_test {
            let mut log_probs = Vec::with_capacity(self.n_classes);
            for c in 0..self.n_classes {
                let mut log_prob = self.class_priors[c].ln();
                for j in 0..self.n_features {
                    let x = data[i * self.n_features + j];
                    let mean = self.means[c * self.n_features + j];
                    let var = self.variances[c * self.n_features + j];
                    log_prob += -0.5 * ((x - mean).powi(2) / var + var.ln() + std::f64::consts::LN_2 + std::f64::consts::PI.ln());
                }
                log_probs.push(log_prob);
            }
            // Softmax to convert log-probs to probabilities
            let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_lp).exp()).sum();
            for lp in &log_probs {
                result.push(((lp - max_lp).exp()) / sum_exp);
            }
        }
        result
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("NaiveBayes(classes={}, features={})", self.n_classes, self.n_features)
    }
}

pub fn naive_bayes_impl(data: &[f64], n_features: usize, labels: &[f64]) -> Result<NaiveBayesModel, MlError> {
    let n = validate_matrix(data, n_features)?;
    if labels.len() != n {
        return Err(MlError::new("labels length must match number of samples"));
    }

    // Find unique classes
    let mut classes: Vec<u32> = labels.iter().map(|&v| v as u32).collect();
    classes.sort_unstable();
    classes.dedup();
    let n_classes = classes.len();

    let mut means = vec![0.0; n_classes * n_features];
    let mut variances = vec![0.0; n_classes * n_features];
    let mut counts = vec![0usize; n_classes];

    // Compute means
    for i in 0..n {
        let c = classes.iter().position(|&cls| cls == labels[i] as u32).unwrap();
        counts[c] += 1;
        for j in 0..n_features {
            means[c * n_features + j] += mat_get(data, n_features, i, j);
        }
    }
    for c in 0..n_classes {
        for j in 0..n_features {
            means[c * n_features + j] /= counts[c] as f64;
        }
    }

    // Compute variances
    for i in 0..n {
        let c = classes.iter().position(|&cls| cls == labels[i] as u32).unwrap();
        for j in 0..n_features {
            let diff = mat_get(data, n_features, i, j) - means[c * n_features + j];
            variances[c * n_features + j] += diff * diff;
        }
    }
    for c in 0..n_classes {
        for j in 0..n_features {
            variances[c * n_features + j] = variances[c * n_features + j] / counts[c] as f64 + 1e-9;
        }
    }

    let class_priors: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();

    Ok(NaiveBayesModel { n_classes, n_features, means, variances, class_priors, classes })
}

#[wasm_bindgen(js_name = "naiveBayesFit")]
pub fn naive_bayes_fit(data: &[f64], n_features: usize, labels: &[f64]) -> Result<NaiveBayesModel, JsError> {
    naive_bayes_impl(data, n_features, labels).map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_classes() {
        // Class 0 centered at (0,0), class 1 centered at (5,5)
        let data = vec![
            0.1, 0.2,  -0.1, 0.1,  0.2, -0.1,
            5.1, 5.2,  4.9, 5.1,  5.2, 4.9,
        ];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let model = naive_bayes_impl(&data, 2, &labels).unwrap();

        let test = vec![0.0, 0.0, 5.0, 5.0];
        let preds = model.predict(&test);
        assert_eq!(preds, vec![0, 1]);
    }

    #[test]
    fn test_three_classes() {
        let data = vec![
            0.0, 0.0,  0.1, 0.1,
            5.0, 0.0,  5.1, 0.1,
            0.0, 5.0,  0.1, 5.1,
        ];
        let labels = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let model = naive_bayes_impl(&data, 2, &labels).unwrap();
        assert_eq!(model.n_classes, 3);
    }
}
