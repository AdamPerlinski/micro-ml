/**
 * Comparison benchmarks: micro-ml vs popular JS alternatives
 *
 * Run with: node --experimental-wasm-modules compare.js
 */

import { linearRegression as wasmLinear, sma as wasmSma } from 'micro-ml';
import ss from 'simple-statistics';

// Generate test data
function generateLinearData(n, noise = 0.1) {
  const x = Array.from({ length: n }, (_, i) => i);
  const y = x.map((xi) => 2.5 * xi + 10 + (Math.random() - 0.5) * noise * xi);
  return { x, y };
}

// Convert to the format simple-statistics expects
function toPoints(x, y) {
  return x.map((xi, i) => [xi, y[i]]);
}

// Benchmark runner
async function benchmark(name, fn, iterations = 50) {
  // Warmup
  for (let i = 0; i < 3; i++) {
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

  return { name, avg, median };
}

function printComparison(results) {
  console.log('\n┌─────────────────────────────────────────┐');
  console.log('│ Library Comparison                      │');
  console.log('├─────────────────────────┬───────┬───────┤');
  console.log('│ Library                 │ Avg   │ Med   │');
  console.log('├─────────────────────────┼───────┼───────┤');

  for (const r of results) {
    const name = r.name.padEnd(23);
    const avg = r.avg.toFixed(2).padStart(5);
    const median = r.median.toFixed(2).padStart(5);
    console.log(`│ ${name} │ ${avg} │ ${median} │`);
  }

  console.log('└─────────────────────────┴───────┴───────┘');
  console.log('  Times in milliseconds (ms)');
}

async function main() {
  console.log('📊 Micro-ML vs Alternatives Comparison\n');
  console.log('Comparing against simple-statistics (pure JS)\n');

  const sizes = [1000, 10000, 50000];

  for (const n of sizes) {
    console.log(`\n━━━ Dataset size: ${n.toLocaleString()} points ━━━`);

    const { x, y } = generateLinearData(n);
    const points = toPoints(x, y);

    const results = [];

    // Linear Regression
    console.log('\n📈 Linear Regression:');

    results.push(await benchmark('micro-ml (WASM)', () => wasmLinear(x, y)));
    results.push(
      await benchmark('simple-statistics', () => ss.linearRegression(points))
    );

    printComparison(results);

    // Calculate speedup
    const wasmTime = results[0].avg;
    const ssTime = results[1].avg;
    const speedup = ssTime / wasmTime;

    console.log(
      `\n  ⚡ micro-ml is ${speedup.toFixed(1)}x ${speedup > 1 ? 'faster' : 'slower'} than simple-statistics`
    );

    // R-squared comparison (accuracy check)
    const wasmModel = await wasmLinear(x, y);
    const ssModel = ss.linearRegression(points);

    console.log('\n  📐 Accuracy comparison:');
    console.log(`     micro-ml slope:     ${wasmModel.slope.toFixed(6)}`);
    console.log(`     simple-stats slope: ${ssModel.m.toFixed(6)}`);
    console.log(`     micro-ml R²:        ${wasmModel.rSquared.toFixed(6)}`);
  }

  // Memory usage estimate
  console.log('\n\n📦 Bundle Size Comparison:');
  console.log('┌───────────────────────┬─────────────┐');
  console.log('│ Library               │ Size (gzip) │');
  console.log('├───────────────────────┼─────────────┤');
  console.log('│ micro-ml              │ ~15-20 KB   │');
  console.log('│ simple-statistics     │ ~30 KB      │');
  console.log('│ TensorFlow.js         │ ~500+ KB    │');
  console.log('│ ml.js                 │ ~150 KB     │');
  console.log('└───────────────────────┴─────────────┘');

  console.log('\n✅ Comparison complete!');
}

main().catch(console.error);
