use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

/// Type of moving average
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MovingAverageType {
    /// Simple Moving Average - equal weight to all periods
    SMA,
    /// Exponential Moving Average - more weight to recent values
    EMA,
    /// Weighted Moving Average - linearly decreasing weights
    WMA,
}

/// Calculate Simple Moving Average
/// Returns NaN for positions where the window isn't full
fn calc_sma(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];

    if window == 0 || window > n {
        return result;
    }

    // Calculate first window sum
    let mut sum: f64 = data[..window].iter().sum();
    result[window - 1] = sum / window as f64;

    // Slide the window
    for i in window..n {
        sum = sum - data[i - window] + data[i];
        result[i] = sum / window as f64;
    }

    result
}

/// Calculate Exponential Moving Average
/// Uses smoothing factor: α = 2 / (window + 1)
fn calc_ema(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];

    if window == 0 || window > n {
        return result;
    }

    let alpha = 2.0 / (window as f64 + 1.0);

    // First EMA value is the SMA of the first window
    let first_sma: f64 = data[..window].iter().sum::<f64>() / window as f64;
    result[window - 1] = first_sma;

    // Calculate subsequent EMA values
    let mut prev_ema = first_sma;
    for i in window..n {
        let ema = alpha * data[i] + (1.0 - alpha) * prev_ema;
        result[i] = ema;
        prev_ema = ema;
    }

    result
}

/// Calculate Weighted Moving Average
/// Uses linearly decreasing weights: w_i = window - i for i in 0..window
fn calc_wma(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];

    if window == 0 || window > n {
        return result;
    }

    // Calculate weight sum: 1 + 2 + ... + window = window * (window + 1) / 2
    let weight_sum = (window * (window + 1)) as f64 / 2.0;

    for i in (window - 1)..n {
        let mut weighted_sum = 0.0;
        let start_idx = i + 1 - window; // Safe: i >= window - 1
        for j in 0..window {
            let weight = (j + 1) as f64;
            weighted_sum += weight * data[start_idx + j];
        }
        result[i] = weighted_sum / weight_sum;
    }

    result
}

/// Calculate a moving average
#[wasm_bindgen(js_name = "movingAverage")]
pub fn moving_average(data: &[f64], window: usize, ma_type: MovingAverageType) -> Vec<f64> {
    match ma_type {
        MovingAverageType::SMA => calc_sma(data, window),
        MovingAverageType::EMA => calc_ema(data, window),
        MovingAverageType::WMA => calc_wma(data, window),
    }
}

/// Calculate SMA (convenience function)
#[wasm_bindgen(js_name = "sma")]
pub fn sma(data: &[f64], window: usize) -> Vec<f64> {
    calc_sma(data, window)
}

/// Calculate EMA (convenience function)
#[wasm_bindgen(js_name = "ema")]
pub fn ema(data: &[f64], window: usize) -> Vec<f64> {
    calc_ema(data, window)
}

/// Calculate WMA (convenience function)
#[wasm_bindgen(js_name = "wma")]
pub fn wma(data: &[f64], window: usize) -> Vec<f64> {
    calc_wma(data, window)
}
