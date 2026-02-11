use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, mat_get};

#[derive(Clone)]
struct Node {
    feature: usize,
    threshold: f64,
    left: usize,   // index into nodes vec (0 = none)
    right: usize,
    prediction: f64,
    is_leaf: bool,
}

#[wasm_bindgen]
pub struct DecisionTreeModel {
    nodes: Vec<Node>,
    n_features: usize,
    depth: usize,
    is_classifier: bool,
}

#[wasm_bindgen]
impl DecisionTreeModel {
    #[wasm_bindgen(getter)]
    pub fn depth(&self) -> usize { self.depth }

    #[wasm_bindgen(getter, js_name = "nNodes")]
    pub fn n_nodes(&self) -> usize { self.nodes.len() }

    #[wasm_bindgen]
    pub fn predict(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let mut node_idx = 0;
            loop {
                let node = &self.nodes[node_idx];
                if node.is_leaf {
                    result.push(node.prediction);
                    break;
                }
                let val = data[i * self.n_features + node.feature];
                node_idx = if val <= node.threshold { node.left } else { node.right };
            }
        }
        result
    }

    /// Return flat tree: [feature, threshold, left, right, prediction, is_leaf] per node
    #[wasm_bindgen(js_name = "getTree")]
    pub fn get_tree(&self) -> Vec<f64> {
        let mut flat = Vec::with_capacity(self.nodes.len() * 6);
        for node in &self.nodes {
            flat.push(node.feature as f64);
            flat.push(node.threshold);
            flat.push(node.left as f64);
            flat.push(node.right as f64);
            flat.push(node.prediction);
            flat.push(if node.is_leaf { 1.0 } else { 0.0 });
        }
        flat
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("DecisionTree(depth={}, nodes={})", self.depth, self.nodes.len())
    }
}

/// Count occurrences of each unique label (no HashMap)
fn label_counts(labels: &[f64]) -> Vec<(u32, usize)> {
    let mut counts: Vec<(u32, usize)> = Vec::new();
    for &l in labels {
        let key = l as u32;
        if let Some(entry) = counts.iter_mut().find(|(k, _)| *k == key) {
            entry.1 += 1;
        } else {
            counts.push((key, 1));
        }
    }
    counts
}

fn gini_impurity(labels: &[f64]) -> f64 {
    if labels.is_empty() { return 0.0; }
    let n = labels.len() as f64;
    let counts = label_counts(labels);
    let mut gini = 1.0;
    for &(_, c) in &counts {
        let p = c as f64 / n;
        gini -= p * p;
    }
    gini
}

fn mse_reduction(targets: &[f64]) -> f64 {
    if targets.is_empty() { return 0.0; }
    let mean = targets.iter().sum::<f64>() / targets.len() as f64;
    targets.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / targets.len() as f64
}

fn majority_class(labels: &[f64]) -> f64 {
    let counts = label_counts(labels);
    let (cls, _) = counts.iter().max_by_key(|(_, v)| v).unwrap();
    *cls as f64
}

fn mean_value(targets: &[f64]) -> f64 {
    targets.iter().sum::<f64>() / targets.len() as f64
}

struct TreeBuilder<'a> {
    data: &'a [f64],
    targets: &'a [f64],
    n_features: usize,
    max_depth: usize,
    min_samples_split: usize,
    is_classifier: bool,
    nodes: Vec<Node>,
    max_depth_reached: usize,
}

impl<'a> TreeBuilder<'a> {
    fn build(&mut self, indices: &[usize], depth: usize) -> usize {
        if depth > self.max_depth_reached { self.max_depth_reached = depth; }

        let targets: Vec<f64> = indices.iter().map(|&i| self.targets[i]).collect();

        // Leaf conditions
        if indices.len() < self.min_samples_split || depth >= self.max_depth || self.is_pure(&targets) {
            let pred = if self.is_classifier { majority_class(&targets) } else { mean_value(&targets) };
            let idx = self.nodes.len();
            self.nodes.push(Node { feature: 0, threshold: 0.0, left: 0, right: 0, prediction: pred, is_leaf: true });
            return idx;
        }

        // Find best split
        let (best_feature, best_threshold, best_score) = self.find_best_split(indices, &targets);

        if best_score < 0.0 {
            let pred = if self.is_classifier { majority_class(&targets) } else { mean_value(&targets) };
            let idx = self.nodes.len();
            self.nodes.push(Node { feature: 0, threshold: 0.0, left: 0, right: 0, prediction: pred, is_leaf: true });
            return idx;
        }

        // Split
        let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices.iter()
            .partition(|&&i| mat_get(self.data, self.n_features, i, best_feature) <= best_threshold);

        if left_idx.is_empty() || right_idx.is_empty() {
            let pred = if self.is_classifier { majority_class(&targets) } else { mean_value(&targets) };
            let idx = self.nodes.len();
            self.nodes.push(Node { feature: 0, threshold: 0.0, left: 0, right: 0, prediction: pred, is_leaf: true });
            return idx;
        }

        // Placeholder node
        let node_idx = self.nodes.len();
        self.nodes.push(Node { feature: best_feature, threshold: best_threshold, left: 0, right: 0, prediction: 0.0, is_leaf: false });

        let left = self.build(&left_idx, depth + 1);
        let right = self.build(&right_idx, depth + 1);
        self.nodes[node_idx].left = left;
        self.nodes[node_idx].right = right;

        node_idx
    }

    fn is_pure(&self, targets: &[f64]) -> bool {
        targets.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
    }

    fn find_best_split(&self, indices: &[usize], targets: &[f64]) -> (usize, f64, f64) {
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_score = -1.0;

        let parent_impurity = if self.is_classifier { gini_impurity(targets) } else { mse_reduction(targets) };

        for f in 0..self.n_features {
            // Get sorted unique values for this feature
            let mut vals: Vec<f64> = indices.iter().map(|&i| mat_get(self.data, self.n_features, i, f)).collect();
            vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            vals.dedup();

            for w in vals.windows(2) {
                let threshold = (w[0] + w[1]) / 2.0;

                let (left_pairs, right_pairs): (Vec<(f64, f64)>, Vec<(f64, f64)>) = indices.iter()
                    .map(|&i| (mat_get(self.data, self.n_features, i, f), self.targets[i]))
                    .partition(|&(v, _)| v <= threshold);

                let left_targets: Vec<f64> = left_pairs.iter().map(|&(_, t)| t).collect();
                let right_targets: Vec<f64> = right_pairs.iter().map(|&(_, t)| t).collect();

                if left_targets.is_empty() || right_targets.is_empty() { continue; }

                let n = indices.len() as f64;
                let left_imp = if self.is_classifier { gini_impurity(&left_targets) } else { mse_reduction(&left_targets) };
                let right_imp = if self.is_classifier { gini_impurity(&right_targets) } else { mse_reduction(&right_targets) };

                let weighted = (left_targets.len() as f64 / n) * left_imp + (right_targets.len() as f64 / n) * right_imp;
                let gain = parent_impurity - weighted;

                if gain > best_score {
                    best_score = gain;
                    best_feature = f;
                    best_threshold = threshold;
                }
            }
        }

        (best_feature, best_threshold, best_score)
    }
}

pub fn decision_tree_impl(data: &[f64], n_features: usize, targets: &[f64], max_depth: usize, min_samples_split: usize, is_classifier: bool) -> Result<DecisionTreeModel, MlError> {
    let n = validate_matrix(data, n_features)?;
    if targets.len() != n {
        return Err(MlError::new("targets length must match number of samples"));
    }
    if n < 2 {
        return Err(MlError::new("Need at least 2 samples"));
    }

    let indices: Vec<usize> = (0..n).collect();
    let mut builder = TreeBuilder {
        data, targets, n_features, max_depth, min_samples_split, is_classifier,
        nodes: Vec::new(), max_depth_reached: 0,
    };
    builder.build(&indices, 0);

    Ok(DecisionTreeModel {
        nodes: builder.nodes,
        n_features,
        depth: builder.max_depth_reached,
        is_classifier,
    })
}

#[wasm_bindgen(js_name = "decisionTreeClassify")]
pub fn decision_tree_classify(data: &[f64], n_features: usize, labels: &[f64], max_depth: usize, min_samples_split: usize) -> Result<DecisionTreeModel, JsError> {
    decision_tree_impl(data, n_features, labels, max_depth, min_samples_split, true).map_err(|e| JsError::new(&e.message))
}

#[wasm_bindgen(js_name = "decisionTreeRegress")]
pub fn decision_tree_regress(data: &[f64], n_features: usize, targets: &[f64], max_depth: usize, min_samples_split: usize) -> Result<DecisionTreeModel, JsError> {
    decision_tree_impl(data, n_features, targets, max_depth, min_samples_split, false).map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classification() {
        let data = vec![
            0.0, 0.0,  1.0, 0.0,
            0.0, 1.0,  1.0, 1.0,
        ];
        let labels = vec![0.0, 1.0, 0.0, 1.0]; // class depends on feature 0
        let model = decision_tree_impl(&data, 2, &labels, 10, 2, true).unwrap();
        let preds = model.predict(&data);
        assert_eq!(preds, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_regression() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let model = decision_tree_impl(&data, 1, &targets, 10, 2, false).unwrap();
        let preds = model.predict(&data);
        // Should predict close to actual values
        for (p, t) in preds.iter().zip(targets.iter()) {
            assert!((p - t).abs() < 2.0);
        }
    }

    #[test]
    fn test_max_depth() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let labels = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let model = decision_tree_impl(&data, 1, &labels, 1, 2, true).unwrap();
        assert!(model.depth <= 1);
    }
}
