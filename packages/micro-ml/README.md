# micro-ml

**You don't need TensorFlow.js for a trendline.**

Most apps just need simple predictions: forecast next month's sales, add a trendline to a chart, smooth noisy sensor data. You don't need a 500KB neural network library for that.

micro-ml is **~40KB gzipped** of focused functionality: regression, smoothing, and forecasting. Handles 100 million data points in ~1 second.

```
npm install micro-ml
```

---

## What Can You Do With It?

### Predict Future Values
Got historical data? Predict what comes next.

```js
// You have: sales data for 12 months
const sales = [10, 12, 15, 18, 22, 25, 28, 32, 35, 40, 45, 50];

// You want: forecast for next 3 months
const forecast = await trendForecast(sales, 3);
console.log(forecast.getForecast()); // [55, 60, 65]
console.log(forecast.direction);      // "up"
```

### Find Trends in Data
Is your data going up, down, or flat? How strong is the trend?

```js
const model = await linearRegressionSimple(sales);
console.log(model.slope);     // 3.7 (growing by ~3.7 per month)
console.log(model.rSquared);  // 0.98 (98% confidence - strong trend)
```

### Smooth Noisy Data
Sensor readings jumping around? Stock prices too volatile? Smooth them out.

```js
// Raw sensor data (noisy)
const readings = [22.1, 25.3, 21.8, 24.9, 23.2, 26.1, 22.5, ...];

// Smoothed (removes noise, shows real trend)
const smooth = await ema(readings, 5);
```

### Fit Curves to Data
Data doesn't follow a straight line? Fit a curve instead.

```js
// Exponential growth (bacteria, viral spread, compound interest)
const expModel = await exponentialRegression(time, population);
console.log(expModel.doublingTime()); // "Population doubles every 3.2 days"

// Polynomial curve (projectile motion, diminishing returns)
const polyModel = await polynomialRegression(x, y, { degree: 2 });
```

---

## When to Use Which Function?

| Your Data Looks Like | Use This | Example |
|---------------------|----------|---------|
| Straight line trend | `linearRegression` | Stock price over time |
| Curved line | `polynomialRegression` | Ball trajectory, learning curves |
| Exponential growth | `exponentialRegression` | Bacteria growth, viral spread |
| Logarithmic (fast then slow) | `logarithmicRegression` | Learning a skill, diminishing returns |
| Noisy/jumpy data | `ema` or `sma` | Sensor readings, stock prices |
| Need future predictions | `trendForecast` | Sales forecast, weight loss goal |
| Find peaks/valleys | `findPeaks` / `findTroughs` | Detect anomalies, buy/sell signals |

---

## Real-World Use Cases

### 1. Sales Forecasting
**Problem:** "How much will we sell next quarter?"

```js
import { trendForecast, linearRegressionSimple } from 'micro-ml';

const monthlySales = [42000, 45000, 48000, 52000, 55000, 58000];

// Analyze trend
const model = await linearRegressionSimple(monthlySales);
console.log(`Growing by $${model.slope.toFixed(0)}/month`);

// Forecast next 3 months
const forecast = await trendForecast(monthlySales, 3);
console.log('Next 3 months:', forecast.getForecast());
// → [61000, 64000, 67000]
```

### 2. Stock/Crypto Trendlines
**Problem:** "Is this stock trending up or down? Add a trendline to my chart."

```js
import { linearRegression, ema } from 'micro-ml';

const days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const prices = [150, 152, 149, 155, 158, 156, 160, 163, 161, 165];

// Fit trendline
const trend = await linearRegression(days, prices);
const trendlinePoints = trend.predict(days);
// → Draw this as a line on your chart

// Add moving average (smoothed price)
const smoothPrices = await ema(prices, 3);
// → Draw this as another line
```

### 3. Weight Loss Prediction
**Problem:** "When will I reach my goal weight?"

```js
import { linearRegressionSimple } from 'micro-ml';

const weeklyWeights = [200, 198, 196.5, 195, 193, 191.5]; // lbs
const goalWeight = 175;

const model = await linearRegressionSimple(weeklyWeights);
const lossPerWeek = Math.abs(model.slope); // 1.5 lbs/week

const currentWeight = weeklyWeights[weeklyWeights.length - 1];
const weeksToGoal = (currentWeight - goalWeight) / lossPerWeek;

console.log(`Losing ${lossPerWeek.toFixed(1)} lbs/week`);
console.log(`Goal in ${Math.ceil(weeksToGoal)} weeks`);
// → "Losing 1.5 lbs/week, Goal in 11 weeks"
```

### 4. IoT Sensor Smoothing
**Problem:** "Temperature sensor is noisy, I want a stable reading."

```js
import { ema, exponentialSmoothing } from 'micro-ml';

// Raw readings jump around: 22.1, 25.3, 21.8, 24.9, ...
const rawTemperature = getSensorReadings();

// Smoothed readings: 22.5, 23.1, 22.8, 23.2, ...
const smoothed = await ema(rawTemperature, 5);

// Display the last smoothed value
displayTemperature(smoothed[smoothed.length - 1]);
```

### 5. Growth Rate Analysis
**Problem:** "How fast is our user base growing? When will we hit 1 million?"

```js
import { exponentialRegression } from 'micro-ml';

const months = [1, 2, 3, 4, 5, 6];
const users = [1000, 1500, 2200, 3300, 5000, 7500];

const model = await exponentialRegression(months, users);

console.log(`Doubling every ${model.doublingTime().toFixed(1)} months`);
// → "Doubling every 1.4 months"

// When will we hit 1 million?
// Solve: 1000000 = a * e^(b*t)
const monthsToMillion = Math.log(1000000 / model.a) / model.b;
console.log(`1M users in ${monthsToMillion.toFixed(0)} months`);
```

### 6. Detecting Anomalies
**Problem:** "Alert me when sensor readings spike."

```js
import { findPeaks, ema } from 'micro-ml';

const readings = [...sensorData];

// Find all spike indices
const spikes = await findPeaks(readings);

// Alert if recent spike
if (spikes.includes(readings.length - 1)) {
  alert('Anomaly detected!');
}
```

---

## Installation

```bash
npm install micro-ml
```

## Quick Start

```js
import { linearRegression, trendForecast, ema } from 'micro-ml';

// Fit a line to data
const model = await linearRegression([1,2,3,4,5], [2,4,6,8,10]);
console.log(model.slope);        // 2
console.log(model.predict([6])); // [12]

// Forecast future values
const forecast = await trendForecast([10,20,30,40,50], 3);
console.log(forecast.getForecast()); // [60, 70, 80]

// Smooth noisy data
const smooth = await ema([10,15,12,18,14,20], 3);
```

## Browser Usage

```html
<script type="module">
  import { linearRegression } from 'https://esm.sh/micro-ml';

  const model = await linearRegression([1,2,3], [2,4,6]);
  console.log(model.slope); // 2
</script>
```

---

## API Reference

### Regression (Find Patterns)

| Function | What It Does | When to Use |
|----------|--------------|-------------|
| `linearRegression(x, y)` | Fits straight line: y = mx + b | Steady growth/decline |
| `linearRegressionSimple(y)` | Same but x = [0,1,2,...] | Time series data |
| `polynomialRegression(x, y, {degree})` | Fits curve | Curved patterns |
| `exponentialRegression(x, y)` | Fits y = a × e^(bx) | Growth/decay |
| `logarithmicRegression(x, y)` | Fits y = a + b × ln(x) | Diminishing returns |
| `powerRegression(x, y)` | Fits y = a × x^b | Power laws |

### Smoothing (Remove Noise)

| Function | What It Does | When to Use |
|----------|--------------|-------------|
| `sma(data, window)` | Simple Moving Average | General smoothing |
| `ema(data, window)` | Exponential Moving Average | Recent values matter more |
| `wma(data, window)` | Weighted Moving Average | Balance of both |
| `exponentialSmoothing(data, {alpha})` | Single exponential smooth | Quick smoothing |

### Forecasting (Predict Future)

| Function | What It Does | When to Use |
|----------|--------------|-------------|
| `trendForecast(data, periods)` | Analyze trend + predict | Future predictions |
| `predict(xTrain, yTrain, xNew)` | One-liner predict | Quick predictions |
| `trendLine(data, periods)` | Get model + predictions | When you need both |

### Analysis (Understand Data)

| Function | What It Does | When to Use |
|----------|--------------|-------------|
| `findPeaks(data)` | Find local maxima | Detect spikes |
| `findTroughs(data)` | Find local minima | Detect dips |
| `rateOfChange(data, periods)` | % change from n ago | Growth rate |
| `momentum(data, periods)` | Difference from n ago | Trend strength |

---

## Model Properties

All regression models return:

```js
model.rSquared   // 0-1, how well the model fits (1 = perfect)
model.n          // Number of data points used
model.predict(x) // Predict y values for new x values
model.toString() // Human-readable equation
```

Linear models also have:
```js
model.slope      // Rate of change
model.intercept  // Y-intercept
```

Exponential models also have:
```js
model.a          // Initial value
model.b          // Growth rate
model.doublingTime() // Time to double
```

---

## Performance

Benchmarked on real hardware (median of 5 runs):

| Data Size | Linear Regression | Moving Average | Polynomial |
|-----------|------------------|----------------|------------|
| 1,000 | < 1ms | < 1ms | < 1ms |
| 10,000 | < 1ms | < 1ms | < 1ms |
| 100,000 | 1ms | 3-4ms | 5ms |
| 1,000,000 | 6-12ms | 30-35ms | 53ms |
| 10,000,000 | 50-100ms | ~280ms | ~530ms |
| 100,000,000 | ~500ms-1s | ~2.9s | - |

For very large datasets, use Web Workers:

```js
import { createWorker } from 'micro-ml/worker';

const ml = createWorker();
const model = await ml.linearRegression(hugeX, hugeY); // Non-blocking
ml.terminate();
```

---

## Comparison

| Library | Size (gzip) | Speed |
|---------|-------------|-------|
| **micro-ml** | 40KB | Fastest (WASM) |
| TensorFlow.js | 500KB+ | Slow |
| ml.js | 150KB | Medium |
| simple-statistics | 30KB | Pure JS, slower |

---

## License

MIT
