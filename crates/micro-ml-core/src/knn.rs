use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, dist_to_point};

#[wasm_bindgen]
pub struct KnnModel {
    n_features: usize,
    training_data: Vec<f64>,
    labels: Vec<u32>,
    k: usize,
    n_samples: usize,
}

#[wasm_bindgen]
impl KnnModel {
    #[wasm_bindgen(getter)]
    pub fn k(&self) -> usize { self.k }

    #[wasm_bindgen(getter, js_name = "nSamples")]
    pub fn n_samples(&self) -> usize { self.n_samples }

    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize { self.n_features }

    #[wasm_bindgen]
    pub fn predict(&self, data: &[f64]) -> Vec<u32> {
        let n_test = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n_test);

        for i in 0..n_test {
            let point = &data[i * self.n_features..(i + 1) * self.n_features];
            // Compute distances to all training points
            let mut dists: Vec<(f64, u32)> = (0..self.n_samples)
                .map(|j| {
                    let d = dist_to_point(&self.training_data, self.n_features, j, point);
                    (d, self.labels[j])
                })
                .collect();

            // Partial sort: find k smallest
            dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Majority vote
            let max_label = *self.labels.iter().max().unwrap_or(&0);
            let mut votes = vec![0u32; max_label as usize + 1];
            for j in 0..self.k.min(dists.len()) {
                votes[dists[j].1 as usize] += 1;
            }
            let predicted = votes.iter().enumerate()
                .max_by_key(|(_, &v)| v)
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            result.push(predicted);
        }
        result
    }

    #[wasm_bindgen(js_name = "predictProba")]
    pub fn predict_proba(&self, data: &[f64]) -> Vec<f64> {
        let n_test = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n_test);

        for i in 0..n_test {
            let point = &data[i * self.n_features..(i + 1) * self.n_features];
            let mut dists: Vec<(f64, u32)> = (0..self.n_samples)
                .map(|j| {
                    let d = dist_to_point(&self.training_data, self.n_features, j, point);
                    (d, self.labels[j])
                })
                .collect();
            dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let k = self.k.min(dists.len());
            let count_1 = dists[..k].iter().filter(|(_, l)| *l == 1).count();
            result.push(count_1 as f64 / k as f64);
        }
        result
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("KNN(k={}, n_samples={})", self.k, self.n_samples)
    }
}

pub fn knn_fit_impl(data: &[f64], n_features: usize, labels: &[f64], k: usize) -> Result<KnnModel, MlError> {
    let n = validate_matrix(data, n_features)?;
    if labels.len() != n {
        return Err(MlError::new("labels length must match number of samples"));
    }
    if k == 0 {
        return Err(MlError::new("k must be > 0"));
    }

    let int_labels: Vec<u32> = labels.iter().map(|&v| v as u32).collect();

    Ok(KnnModel {
        n_features,
        training_data: data.to_vec(),
        labels: int_labels,
        k,
        n_samples: n,
    })
}

#[wasm_bindgen(js_name = "knnFit")]
pub fn knn_fit(data: &[f64], n_features: usize, labels: &[f64], k: usize) -> Result<KnnModel, JsError> {
    knn_fit_impl(data, n_features, labels, k).map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_classification() {
        // Two clear clusters
        let data = vec![
            0.0, 0.0,  0.1, 0.1,  0.2, 0.0,  // class 0
            5.0, 5.0,  5.1, 5.1,  4.9, 5.0,  // class 1
        ];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let model = knn_fit_impl(&data, 2, &labels, 3).unwrap();

        let test = vec![0.05, 0.05, 4.95, 4.95];
        let preds = model.predict(&test);
        assert_eq!(preds, vec![0, 1]);
    }

    #[test]
    fn test_k1_memorizes() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let labels = vec![0.0, 1.0];
        let model = knn_fit_impl(&data, 2, &labels, 1).unwrap();

        let preds = model.predict(&data);
        assert_eq!(preds, vec![0, 1]);
    }

    #[test]
    fn test_predict_proba() {
        let data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let labels = vec![0.0, 1.0, 1.0];
        let model = knn_fit_impl(&data, 2, &labels, 3).unwrap();
        let proba = model.predict_proba(&vec![1.0, 1.0]);
        assert!((proba[0] - 2.0 / 3.0).abs() < 1e-10);
    }
}
