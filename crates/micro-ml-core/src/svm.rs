use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::validate_matrix;

/// Support Vector Machine for binary classification using linear kernel with hinge loss
#[wasm_bindgen]
pub struct SvmModel {
    n_features: usize,
    weights: Vec<f64>,
    bias: f64,
    n_samples: usize,
}

#[wasm_bindgen]
impl SvmModel {
    /// Train a linear SVM model
    /// 
    /// # Arguments
    /// * `x` - Training data as flat array (samples × features)
    /// * `n_features` - Number of features per sample
    /// * `y` - Binary labels as [0, 1] or [-1, 1]
    /// * `learning_rate` - Learning rate for gradient descent
    /// * `lambda` - L2 regularization parameter (default ~0.01)
    /// * `epochs` - Number of training epochs
    ///
    /// # Returns
    /// An SVM model ready for prediction
    #[wasm_bindgen(js_name = "train")]
    pub fn train(
        x: &[f64],
        n_features: usize,
        y: &[u32],
        learning_rate: f64,
        lambda: f64,
        epochs: u32,
    ) -> Result<SvmModel, JsValue> {
        // Validate input
        validate_matrix(x, n_features).map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let n_samples = x.len() / n_features;
        if y.len() != n_samples {
            return Err(JsValue::from_str("Labels length must match number of samples"));
        }

        // Convert labels: 0 -> -1, 1 -> 1
        let y_signed: Vec<f64> = y.iter().map(|&label| if label == 0 { -1.0 } else { 1.0 }).collect();

        // Initialize weights to zero
        let mut weights = vec![0.0; n_features];
        let mut bias = 0.0;

        // Stochastic gradient descent
        for _epoch in 0..epochs {
            for i in 0..n_samples {
                let x_i = &x[i * n_features..(i + 1) * n_features];
                let y_i = y_signed[i];

                // Compute prediction: w·x + b
                let mut pred = bias;
                for j in 0..n_features {
                    pred += weights[j] * x_i[j];
                }

                // Hinge loss: max(0, 1 - y*pred)
                let loss = 1.0 - y_i * pred;
                
                // Update weights and bias if hinge loss is positive
                if loss > 0.0 {
                    // Gradient for weights: -y*x + lambda*w
                    for j in 0..n_features {
                        weights[j] = weights[j] * (1.0 - learning_rate * lambda) 
                                   - learning_rate * y_i * x_i[j];
                    }
                    // Gradient for bias: -y
                    bias -= learning_rate * y_i;
                } else {
                    // Just apply regularization
                    for j in 0..n_features {
                        weights[j] *= 1.0 - learning_rate * lambda;
                    }
                }
            }
        }

        Ok(SvmModel {
            n_features,
            weights,
            bias,
            n_samples,
        })
    }

    /// Predict class labels for samples
    /// 
    /// # Arguments
    /// * `data` - Test data as flat array (samples × features)
    ///
    /// # Returns
    /// Predicted labels as [0, 1]
    #[wasm_bindgen]
    pub fn predict(&self, data: &[f64]) -> Result<Vec<u32>, JsValue> {
        validate_matrix(data, self.n_features)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let n_test = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n_test);

        for i in 0..n_test {
            let x_i = &data[i * self.n_features..(i + 1) * self.n_features];
            
            // Compute w·x + b
            let mut pred = self.bias;
            for j in 0..self.n_features {
                pred += self.weights[j] * x_i[j];
            }

            // Return 1 if pred >= 0, else 0
            result.push(if pred >= 0.0 { 1 } else { 0 });
        }

        Ok(result)
    }

    /// Get decision function values (distance from decision boundary)
    /// Positive values indicate class 1, negative indicate class 0
    ///
    /// # Arguments
    /// * `data` - Test data as flat array (samples × features)
    ///
    /// # Returns
    /// Decision function values for each sample
    #[wasm_bindgen(js_name = "decisionFunction")]
    pub fn decision_function(&self, data: &[f64]) -> Result<Vec<f64>, JsValue> {
        validate_matrix(data, self.n_features)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let n_test = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n_test);

        for i in 0..n_test {
            let x_i = &data[i * self.n_features..(i + 1) * self.n_features];
            
            let mut pred = self.bias;
            for j in 0..self.n_features {
                pred += self.weights[j] * x_i[j];
            }

            result.push(pred);
        }

        Ok(result)
    }

    /// Get model weights
    #[wasm_bindgen(js_name = "getWeights")]
    pub fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    /// Get model bias
    #[wasm_bindgen(getter)]
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Get number of features
    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Get number of training samples
    #[wasm_bindgen(getter, js_name = "nSamples")]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Get model info as string
    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string(&self) -> String {
        format!(
            "SvmModel(features={}, samples={}, bias={:.6})",
            self.n_features, self.n_samples, self.bias
        )
    }
}
