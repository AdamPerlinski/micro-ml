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
