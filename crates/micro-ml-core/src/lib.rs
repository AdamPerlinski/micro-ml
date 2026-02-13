mod error;
mod matrix;
mod linear;
mod polynomial;
mod exponential;
mod timeseries;
mod kmeans;
mod knn;
mod logistic;
mod dbscan;
mod naive_bayes;
mod decision_tree;
mod pca;
mod perceptron;

use wasm_bindgen::prelude::*;
pub use error::MlError;

#[wasm_bindgen(start)]
pub fn init() {}

// Re-export all public types and functions
pub use linear::*;
pub use polynomial::*;
pub use exponential::*;
pub use timeseries::*;
pub use kmeans::*;
pub use knn::*;
pub use logistic::*;
pub use dbscan::*;
pub use naive_bayes::*;
pub use decision_tree::*;
pub use pca::*;
pub use perceptron::*;
