use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, mat_get};

#[wasm_bindgen]
pub struct PcaResult {
    n_components: usize,
    n_features: usize,
    components: Vec<f64>,
    explained_variance: Vec<f64>,
    explained_variance_ratio: Vec<f64>,
    mean: Vec<f64>,
    transformed: Vec<f64>,
    n_samples: usize,
}

#[wasm_bindgen]
impl PcaResult {
    #[wasm_bindgen(getter, js_name = "nComponents")]
    pub fn n_components(&self) -> usize { self.n_components }

    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize { self.n_features }

    #[wasm_bindgen(js_name = "getComponents")]
    pub fn get_components(&self) -> Vec<f64> { self.components.clone() }

    #[wasm_bindgen(js_name = "getExplainedVariance")]
    pub fn get_explained_variance(&self) -> Vec<f64> { self.explained_variance.clone() }

    #[wasm_bindgen(js_name = "getExplainedVarianceRatio")]
    pub fn get_explained_variance_ratio(&self) -> Vec<f64> { self.explained_variance_ratio.clone() }

    #[wasm_bindgen(js_name = "getMean")]
    pub fn get_mean(&self) -> Vec<f64> { self.mean.clone() }

    #[wasm_bindgen(js_name = "getTransformed")]
    pub fn get_transformed(&self) -> Vec<f64> { self.transformed.clone() }

    /// Project new data onto principal components
    #[wasm_bindgen]
    pub fn transform(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len() / self.n_features;
        let mut result = vec![0.0; n * self.n_components];
        for i in 0..n {
            for c in 0..self.n_components {
                let mut dot = 0.0;
                for j in 0..self.n_features {
                    dot += (data[i * self.n_features + j] - self.mean[j]) * self.components[c * self.n_features + j];
                }
                result[i * self.n_components + c] = dot;
            }
        }
        result
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        let total: f64 = self.explained_variance_ratio.iter().sum();
        format!("PCA(components={}, explained_variance={:.1}%)", self.n_components, total * 100.0)
    }
}

pub fn pca_impl(data: &[f64], n_features: usize, n_components: usize) -> Result<PcaResult, MlError> {
    let n = validate_matrix(data, n_features)?;
    if n_components == 0 || n_components > n_features {
        return Err(MlError::new("n_components must be between 1 and n_features"));
    }

    // Compute mean
    let mut mean = vec![0.0; n_features];
    for i in 0..n {
        for j in 0..n_features {
            mean[j] += mat_get(data, n_features, i, j);
        }
    }
    for j in 0..n_features { mean[j] /= n as f64; }

    // Center data
    let mut centered = vec![0.0; n * n_features];
    for i in 0..n {
        for j in 0..n_features {
            centered[i * n_features + j] = mat_get(data, n_features, i, j) - mean[j];
        }
    }

    // Compute covariance matrix (d x d)
    let d = n_features;
    let mut cov = vec![0.0; d * d];
    for i in 0..n {
        for j in 0..d {
            for k in j..d {
                let val = centered[i * d + j] * centered[i * d + k];
                cov[j * d + k] += val;
                if j != k { cov[k * d + j] += val; }
            }
        }
    }
    let n_f = (n - 1).max(1) as f64;
    for v in cov.iter_mut() { *v /= n_f; }

    // Power iteration with deflation to find top eigenvectors
    let mut components = Vec::with_capacity(n_components * d);
    let mut eigenvalues = Vec::with_capacity(n_components);
    let mut cov_work = cov.clone();

    for _ in 0..n_components {
        let (eigenvalue, eigenvector) = power_iteration(&cov_work, d, 200);
        eigenvalues.push(eigenvalue);
        components.extend_from_slice(&eigenvector);

        // Deflate: cov = cov - eigenvalue * v * v^T
        for j in 0..d {
            for k in 0..d {
                cov_work[j * d + k] -= eigenvalue * eigenvector[j] * eigenvector[k];
            }
        }
    }

    // Use total variance from diagonal of original cov for ratio
    let total_var: f64 = (0..d).map(|j| cov[j * d + j]).sum();
    let explained_variance_ratio: Vec<f64> = eigenvalues.iter()
        .map(|&ev| if total_var > 0.0 { ev / total_var } else { 0.0 })
        .collect();

    // Transform training data
    let mut transformed = vec![0.0; n * n_components];
    for i in 0..n {
        for c in 0..n_components {
            let mut dot = 0.0;
            for j in 0..d {
                dot += centered[i * d + j] * components[c * d + j];
            }
            transformed[i * n_components + c] = dot;
        }
    }

    Ok(PcaResult {
        n_components, n_features, components,
        explained_variance: eigenvalues,
        explained_variance_ratio,
        mean, transformed, n_samples: n,
    })
}

fn power_iteration(matrix: &[f64], d: usize, max_iter: usize) -> (f64, Vec<f64>) {
    // Initialize with [1, 0, 0, ...] or first non-zero direction
    let mut v = vec![0.0; d];
    v[0] = 1.0;

    for _ in 0..max_iter {
        // w = matrix * v
        let mut w = vec![0.0; d];
        for i in 0..d {
            for j in 0..d {
                w[i] += matrix[i * d + j] * v[j];
            }
        }

        // Normalize
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 { break; }
        for x in w.iter_mut() { *x /= norm; }

        // Check convergence
        let diff: f64 = v.iter().zip(w.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        v = w;
        if diff < 1e-10 { break; }
    }

    // Eigenvalue = v^T * matrix * v
    let mut eigenvalue = 0.0;
    for i in 0..d {
        let mut row_sum = 0.0;
        for j in 0..d {
            row_sum += matrix[i * d + j] * v[j];
        }
        eigenvalue += v[i] * row_sum;
    }

    (eigenvalue, v)
}

#[wasm_bindgen(js_name = "pca")]
pub fn pca(data: &[f64], n_features: usize, n_components: usize) -> Result<PcaResult, JsError> {
    pca_impl(data, n_features, n_components).map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_2d() {
        // Data with variance mostly along x-axis
        let data = vec![
            1.0, 0.1,
            2.0, 0.2,
            3.0, 0.15,
            4.0, 0.25,
            5.0, 0.1,
        ];
        let result = pca_impl(&data, 2, 1).unwrap();
        assert_eq!(result.n_components, 1);
        // First component should capture most variance
        assert!(result.explained_variance_ratio[0] > 0.9);
    }

    #[test]
    fn test_explained_variance_sums_to_one() {
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            2.0, 4.0, 1.0,
        ];
        let result = pca_impl(&data, 3, 3).unwrap();
        let total: f64 = result.explained_variance_ratio.iter().sum();
        assert!((total - 1.0).abs() < 0.1); // Should be close to 1
    }

    #[test]
    fn test_transform() {
        let data = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
        let result = pca_impl(&data, 2, 2).unwrap();
        // Transform should preserve data in rotated space
        let new_data = vec![1.0, 0.0];
        let proj = result.transform(&new_data);
        assert_eq!(proj.len(), 2);
    }
}
