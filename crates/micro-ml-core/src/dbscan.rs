use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, euclidean_dist};

#[wasm_bindgen]
pub struct DbscanResult {
    labels: Vec<i32>,
    n_clusters: usize,
    n_noise: usize,
}

#[wasm_bindgen]
impl DbscanResult {
    #[wasm_bindgen(getter, js_name = "nClusters")]
    pub fn n_clusters(&self) -> usize { self.n_clusters }

    #[wasm_bindgen(getter, js_name = "nNoise")]
    pub fn n_noise(&self) -> usize { self.n_noise }

    #[wasm_bindgen(js_name = "getLabels")]
    pub fn get_labels(&self) -> Vec<i32> { self.labels.clone() }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("DBSCAN(clusters={}, noise={})", self.n_clusters, self.n_noise)
    }
}

pub fn dbscan_impl(data: &[f64], n_features: usize, eps: f64, min_points: usize) -> Result<DbscanResult, MlError> {
    let n = validate_matrix(data, n_features)?;
    if eps <= 0.0 {
        return Err(MlError::new("eps must be > 0"));
    }

    let mut labels = vec![-1i32; n]; // -1 = unvisited/noise
    let mut cluster_id: i32 = 0;

    for i in 0..n {
        if labels[i] != -1 { continue; }

        // Find neighbors
        let neighbors = range_query(data, n_features, n, i, eps);

        if neighbors.len() < min_points {
            // Stays as noise (-1) for now, might be claimed by a cluster later
            continue;
        }

        // Start new cluster
        labels[i] = cluster_id;
        let mut queue = neighbors;
        let mut qi = 0;

        while qi < queue.len() {
            let q = queue[qi];
            qi += 1;

            if labels[q] == -1 {
                labels[q] = cluster_id; // Claim noise point
            }
            if labels[q] != -1 && labels[q] != cluster_id {
                continue; // Already in another cluster (shouldn't happen with -1 init, but safe)
            }
            labels[q] = cluster_id;

            let q_neighbors = range_query(data, n_features, n, q, eps);
            if q_neighbors.len() >= min_points {
                for &nn in &q_neighbors {
                    if labels[nn] == -1 {
                        queue.push(nn);
                    }
                }
            }
        }

        cluster_id += 1;
    }

    let n_clusters = cluster_id as usize;
    let n_noise = labels.iter().filter(|&&l| l == -1).count();

    Ok(DbscanResult { labels, n_clusters, n_noise })
}

fn range_query(data: &[f64], n_features: usize, n: usize, point: usize, eps: f64) -> Vec<usize> {
    let mut neighbors = Vec::new();
    for j in 0..n {
        if euclidean_dist(data, n_features, point, j) <= eps {
            neighbors.push(j);
        }
    }
    neighbors
}

#[wasm_bindgen(js_name = "dbscan")]
pub fn dbscan(data: &[f64], n_features: usize, eps: f64, min_points: usize) -> Result<DbscanResult, JsError> {
    dbscan_impl(data, n_features, eps, min_points).map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_clusters() {
        let data = vec![
            0.0, 0.0,  0.1, 0.1,  0.2, 0.0,
            5.0, 5.0,  5.1, 5.1,  4.9, 5.0,
        ];
        let result = dbscan_impl(&data, 2, 0.5, 2).unwrap();
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.n_noise, 0);
        assert_eq!(result.labels[0], result.labels[1]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_all_noise() {
        let data = vec![0.0, 0.0, 5.0, 5.0, 10.0, 10.0];
        let result = dbscan_impl(&data, 2, 0.1, 2).unwrap();
        assert_eq!(result.n_clusters, 0);
        assert_eq!(result.n_noise, 3);
    }

    #[test]
    fn test_single_cluster_large_eps() {
        let data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let result = dbscan_impl(&data, 2, 100.0, 2).unwrap();
        assert_eq!(result.n_clusters, 1);
    }
}
