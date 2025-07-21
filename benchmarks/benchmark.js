/**
 * Micro-ML Performance Benchmarks
 *
 * Run with: node --experimental-wasm-modules benchmark.js
 */

import { linearRegression, polynomialRegression, sma, ema } from 'micro-ml';

// Generate test data
function generateData(n) {
  const x = Array.from({ length: n }, (_, i) => i);
  const y = x.map(xi => 2 * xi + 3 + (Math.random() - 0.5) * 10);
  return { x, y };
}

// Simple pure JS linear regression for comparison
function jsLinearRegression(x, y) {
  const n = x.length;
  const xMean = x.reduce((a, b) => a + b, 0) / n;
  const yMean = y.reduce((a, b) => a + b, 0) / n;

  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const xDiff = x[i] - xMean;
    num += xDiff * (y[i] - yMean);
    den += xDiff * xDiff;
  }

  const slope = num / den;
  const intercept = yMean - slope * xMean;

  return { slope, intercept };
}

// Simple pure JS SMA
function jsSma(data, window) {
  const result = new Array(data.length).fill(NaN);
  for (let i = window - 1; i < data.length; i++) {
    let sum = 0;
    for (let j = 0; j < window; j++) {
      sum += data[i - j];
    }
    result[i] = sum / window;
  }
  return result;
}

// Benchmark runner
async function benchmark(name, fn, iterations = 100) {
  // Warmup
  for (let i = 0; i < 5; i++) {
    await fn();
  }

  const times = [];
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await fn();
    times.push(performance.now() - start);
  }

  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  const sorted = [...times].sort((a, b) => a - b);
  const median = sorted[Math.floor(times.length / 2)];
  const p95 = sorted[Math.floor(times.length * 0.95)];
  const min = sorted[0];
  const max = sorted[sorted.length - 1];

  return { name, avg, median, p95, min, max };
}

// Format results
function formatResults(results) {
  console.log('\n┌─────────────────────────────────────────────────────────────────────┐');
  console.log('│ Benchmark Results                                                   │');
  console.log('├──────────────────────────┬────────┬────────┬────────┬────────┬──────┤');
  console.log('│ Test                     │ Avg    │ Median │ P95    │ Min    │ Max  │');
  console.log('├──────────────────────────┼────────┼────────┼────────┼────────┼──────┤');

  for (const r of results) {
    const name = r.name.padEnd(24);
    const avg = r.avg.toFixed(2).padStart(6);
    const median = r.median.toFixed(2).padStart(6);
    const p95 = r.p95.toFixed(2).padStart(6);
    const min = r.min.toFixed(2).padStart(6);
    const max = r.max.toFixed(2).padStart(4);
    console.log(`│ ${name} │ ${avg} │ ${median} │ ${p95} │ ${min} │ ${max} │`);
  }

  console.log('└──────────────────────────┴────────┴────────┴────────┴────────┴──────┘');
  console.log('  All times in milliseconds (ms)');
}

async function main() {
  console.log('🚀 Micro-ML Performance Benchmarks\n');

  const results = [];

  // Test different data sizes
  for (const size of [100, 1000, 10000, 100000]) {
    console.log(`\n📊 Testing with ${size.toLocaleString()} data points...`);
    const { x, y } = generateData(size);

    // Linear regression - WASM
    results.push(
      await benchmark(`WASM Linear (n=${size})`, () => linearRegression(x, y))
    );

    // Linear regression - Pure JS
    results.push(
      await benchmark(`JS Linear (n=${size})`, () => jsLinearRegression(x, y))
    );

    if (size <= 10000) {
      // Polynomial regression (only for smaller datasets)
      results.push(
        await benchmark(`WASM Poly deg=3 (n=${size})`, () =>
          polynomialRegression(x, y, { degree: 3 })
        )
      );
    }

    // SMA - WASM
    results.push(
      await benchmark(`WASM SMA w=20 (n=${size})`, () => sma(y, 20))
    );

    // SMA - Pure JS
    results.push(
      await benchmark(`JS SMA w=20 (n=${size})`, () => jsSma(y, 20))
    );

    // EMA - WASM
    results.push(
      await benchmark(`WASM EMA w=20 (n=${size})`, () => ema(y, 20))
    );
  }

  formatResults(results);

  // Print speedup summary
  console.log('\n📈 Speedup Summary (WASM vs Pure JS):\n');

  const sizes = [100, 1000, 10000, 100000];
  for (const size of sizes) {
    const wasmLinear = results.find(r => r.name === `WASM Linear (n=${size})`);
    const jsLinear = results.find(r => r.name === `JS Linear (n=${size})`);

    if (wasmLinear && jsLinear) {
      const speedup = jsLinear.avg / wasmLinear.avg;
      console.log(
        `  Linear Regression (n=${size.toLocaleString().padStart(7)}): ` +
        `${speedup.toFixed(2)}x ${speedup > 1 ? 'faster' : 'slower'}`
      );
    }

    const wasmSma = results.find(r => r.name === `WASM SMA w=20 (n=${size})`);
    const jsSma = results.find(r => r.name === `JS SMA w=20 (n=${size})`);

    if (wasmSma && jsSma) {
      const speedup = jsSma.avg / wasmSma.avg;
      console.log(
        `  SMA (n=${size.toLocaleString().padStart(7)}):                ` +
        `${speedup.toFixed(2)}x ${speedup > 1 ? 'faster' : 'slower'}`
      );
    }
  }

  console.log('\n✅ Benchmarks complete!');
}

main().catch(console.error);
