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

/// Trend direction
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrendDirection {
    Up,
    Down,
    Flat,
}

/// Trend analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct TrendAnalysis {
    direction: TrendDirection,
    slope: f64,
    strength: f64,  // 0-1, based on R²
    forecast: Vec<f64>,
}

#[wasm_bindgen]
impl TrendAnalysis {
    /// Get the trend direction
    #[wasm_bindgen(getter)]
    pub fn direction(&self) -> TrendDirection {
        self.direction
    }

    /// Get the slope (rate of change per period)
    #[wasm_bindgen(getter)]
    pub fn slope(&self) -> f64 {
        self.slope
    }

    /// Get the trend strength (0-1, based on R²)
    #[wasm_bindgen(getter)]
    pub fn strength(&self) -> f64 {
        self.strength
    }

    /// Get the forecasted values
    #[wasm_bindgen(js_name = "getForecast")]
    pub fn get_forecast(&self) -> Vec<f64> {
        self.forecast.clone()
    }
}

/// Analyze trend and forecast future values
#[wasm_bindgen(js_name = "trendForecast")]
pub fn trend_forecast(data: &[f64], periods: usize) -> Result<TrendAnalysis, JsError> {
    if data.len() < 2 {
        return Err(JsError::new("Need at least 2 data points for trend analysis"));
    }

    let n = data.len();
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();

    // Linear regression
    let x_mean: f64 = x.iter().sum::<f64>() / n as f64;
    let y_mean: f64 = data.iter().sum::<f64>() / n as f64;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        let x_diff = x[i] - x_mean;
        let y_diff = data[i] - y_mean;
        numerator += x_diff * y_diff;
        denominator += x_diff * x_diff;
    }

    let slope = if denominator == 0.0 { 0.0 } else { numerator / denominator };
    let intercept = y_mean - slope * x_mean;

    // Calculate R² for strength
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;

    for i in 0..n {
        let y_pred = slope * x[i] + intercept;
        ss_res += (data[i] - y_pred).powi(2);
        ss_tot += (data[i] - y_mean).powi(2);
    }

    let r_squared = if ss_tot == 0.0 { 1.0 } else { 1.0 - (ss_res / ss_tot) };

    // Determine direction
    let direction = if slope.abs() < 1e-10 {
        TrendDirection::Flat
    } else if slope > 0.0 {
        TrendDirection::Up
    } else {
        TrendDirection::Down
    };

    // Generate forecast
    let mut forecast = Vec::with_capacity(periods);
    for i in 0..periods {
        let x_val = (n + i) as f64;
        forecast.push(slope * x_val + intercept);
    }

    Ok(TrendAnalysis {
        direction,
        slope,
        strength: r_squared,
        forecast,
    })
}

/// Calculate the rate of change (ROC) as percentage
#[wasm_bindgen(js_name = "rateOfChange")]
pub fn rate_of_change(data: &[f64], periods: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];

    if periods == 0 || periods >= n {
        return result;
    }

    for i in periods..n {
        if data[i - periods] != 0.0 {
            result[i] = ((data[i] - data[i - periods]) / data[i - periods]) * 100.0;
        }
    }

    result
}

/// Calculate momentum (difference from n periods ago)
#[wasm_bindgen]
pub fn momentum(data: &[f64], periods: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];

    if periods == 0 || periods >= n {
        return result;
    }

    for i in periods..n {
        result[i] = data[i] - data[i - periods];
    }

    result
}

/// Smooth data using exponential smoothing (single)
#[wasm_bindgen(js_name = "exponentialSmoothing")]
pub fn exponential_smoothing(data: &[f64], alpha: f64) -> Result<Vec<f64>, JsError> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(JsError::new("Alpha must be between 0 and 1"));
    }

    if data.is_empty() {
        return Ok(vec![]);
    }

    let mut result = Vec::with_capacity(data.len());
    result.push(data[0]); // First value is the same

    for i in 1..data.len() {
        let smoothed = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        result.push(smoothed);
    }

    Ok(result)
}

/// Detect peaks in data (local maxima)
#[wasm_bindgen(js_name = "findPeaks")]
pub fn find_peaks(data: &[f64]) -> Vec<usize> {
    if data.len() < 3 {
        return vec![];
    }

    let mut peaks = Vec::new();

    for i in 1..(data.len() - 1) {
        if data[i] > data[i - 1] && data[i] > data[i + 1] {
            peaks.push(i);
        }
    }

    peaks
}

/// Detect troughs in data (local minima)
#[wasm_bindgen(js_name = "findTroughs")]
pub fn find_troughs(data: &[f64]) -> Vec<usize> {
    if data.len() < 3 {
        return vec![];
    }

    let mut troughs = Vec::new();

    for i in 1..(data.len() - 1) {
        if data[i] < data[i - 1] && data[i] < data[i + 1] {
            troughs.push(i);
        }
    }

    troughs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&data, 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_wma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = wma(&data, 3);

        // WMA for window 3: (1*1 + 2*2 + 3*3) / 6 = 14/6 = 2.333...
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 14.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_trend_forecast_up() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = trend_forecast(&data, 3).unwrap();

        assert_eq!(result.direction(), TrendDirection::Up);
        assert!((result.slope() - 1.0).abs() < 1e-10);
        assert!((result.strength() - 1.0).abs() < 1e-10);

        let forecast = result.get_forecast();
        assert!((forecast[0] - 6.0).abs() < 1e-10);
        assert!((forecast[1] - 7.0).abs() < 1e-10);
        assert!((forecast[2] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_trend_forecast_down() {
        let data = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let result = trend_forecast(&data, 2).unwrap();

        assert_eq!(result.direction(), TrendDirection::Down);
        assert!((result.slope() - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_rate_of_change() {
        let data = vec![100.0, 110.0, 121.0, 133.1];
        let result = rate_of_change(&data, 1);

        assert!(result[0].is_nan());
        assert!((result[1] - 10.0).abs() < 1e-10);
        assert!((result[2] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_momentum() {
        let data = vec![1.0, 3.0, 6.0, 10.0];
        let result = momentum(&data, 2);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 5.0).abs() < 1e-10); // 6 - 1
        assert!((result[3] - 7.0).abs() < 1e-10); // 10 - 3
    }

    #[test]
    fn test_find_peaks() {
        let data = vec![1.0, 3.0, 2.0, 5.0, 1.0];
        let peaks = find_peaks(&data);

        assert_eq!(peaks, vec![1, 3]);
    }

    #[test]
    fn test_find_troughs() {
        let data = vec![3.0, 1.0, 4.0, 2.0, 5.0];
        let troughs = find_troughs(&data);

        assert_eq!(troughs, vec![1, 3]);
    }

    #[test]
    fn test_exponential_smoothing() {
        let data = vec![10.0, 20.0, 15.0, 25.0];
        let result = exponential_smoothing(&data, 0.5).unwrap();

        assert!((result[0] - 10.0).abs() < 1e-10);
        assert!((result[1] - 15.0).abs() < 1e-10); // 0.5*20 + 0.5*10
        assert!((result[2] - 15.0).abs() < 1e-10); // 0.5*15 + 0.5*15
    }
}
