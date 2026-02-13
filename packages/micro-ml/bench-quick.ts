import {
  init,
  linearRegression,
  polynomialRegression,
  exponentialRegression,
  sma, ema, wma,
  trendForecast,
  kmeans,
  knnClassifier,
  logisticRegression,
  dbscan,
  naiveBayes,
  decisionTree,
  pca,
  perceptron,
  seasonalDecompose,
  autocorrelation,
  detectSeasonality,
} from './src/index.js';

function randn() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function blob(cx: number, cy: number, n: number, spread = 0.5): number[][] {
  return Array.from({ length: n }, () => [cx + randn() * spread, cy + randn() * spread]);
}

async function bench(name: string, fn: () => Promise<any>): Promise<number> {
  // Warmup
  await fn();
  // Actual - 3 runs
  const times: number[] = [];
  for (let i = 0; i < 3; i++) {
    const t0 = performance.now();
    await fn();
    times.push(performance.now() - t0);
  }
  const median = times.sort((a, b) => a - b)[1];
  return median;
}

async function main() {
  console.log('Initializing WASM...');
  const t0 = performance.now();
  await init();
  console.log(`WASM loaded in ${(performance.now() - t0).toFixed(1)}ms\n`);

  const results: [string, string, string][] = [];
  function r(name: string, ms: number, detail = '') {
    const t = ms < 0.01 ? '<0.01ms' : ms < 0.1 ? ms.toFixed(3) + 'ms' : ms.toFixed(2) + 'ms';
    results.push([name, t, detail]);
    console.log(`  ✓ ${name}: ${t}`);
  }

  // ═══════════════════════════════════════
  console.log('── REGRESSION ──');
  // ═══════════════════════════════════════

  for (const n of [1_000, 10_000, 100_000]) {
    const x = Array.from({ length: n }, (_, i) => i);
    const y = x.map(xi => 2 * xi + 10 + Math.random());
    const ms = await bench(`linreg-${n}`, () => linearRegression(x, y));
    r(`Linear Regression (${n.toLocaleString()} pts)`, ms);
  }

  {
    const x = Array.from({ length: 1000 }, (_, i) => i * 0.01);
    const y = x.map(xi => 3 * xi * xi - 2 * xi + 1 + randn() * 0.1);
    const ms = await bench('polyreg', () => polynomialRegression(x, y, { degree: 2 }));
    r('Polynomial Regression (1K pts, deg 2)', ms);
  }

  {
    const x = Array.from({ length: 500 }, (_, i) => i * 0.01);
    const y = x.map(xi => 2 * Math.exp(0.5 * xi) + randn() * 0.1);
    const ms = await bench('expreg', () => exponentialRegression(x, y));
    r('Exponential Regression (500 pts)', ms);
  }

  {
    const data = Array.from({ length: 10000 }, (_, i) => i * 1.5 + randn() * 10);
    const ms = await bench('trend', () => trendForecast(data, 10));
    r('Trend Forecast (10K pts)', ms);
  }

  // Moving averages
  {
    const data = Array.from({ length: 100000 }, () => Math.random() * 100);
    const ms1 = await bench('sma', () => sma(data, 20));
    r('SMA (100K pts, w=20)', ms1);
    const ms2 = await bench('ema', () => ema(data, 20));
    r('EMA (100K pts, w=20)', ms2);
    const ms3 = await bench('wma', () => wma(data, 20));
    r('WMA (100K pts, w=20)', ms3);
  }

  // ═══════════════════════════════════════
  console.log('\n── ML ALGORITHMS ──');
  // ═══════════════════════════════════════

  // k-Means
  for (const n of [500, 2000, 10000]) {
    const data: number[][] = [];
    for (let i = 0; i < 5; i++) data.push(...blob((Math.random() - 0.5) * 10, (Math.random() - 0.5) * 10, Math.ceil(n / 5)));
    const ms = await bench(`km-${n}`, () => kmeans(data, { k: 5 }));
    r(`k-Means (${n.toLocaleString()} pts, k=5)`, ms);
  }

  // kNN
  for (const n of [200, 1000, 5000]) {
    const data = [...blob(2, 2, n / 2), ...blob(-2, -2, n / 2)];
    const labels = [...Array(n / 2).fill(0), ...Array(n / 2).fill(1)];
    const ms = await bench(`knn-${n}`, async () => {
      const model = await knnClassifier(data, labels, { k: 5 });
      model.predict(data.slice(0, 20));
    });
    r(`kNN (${n.toLocaleString()} train, predict 20)`, ms);
  }

  // Logistic
  for (const n of [200, 1000, 5000]) {
    const data = [...blob(1, 1, n / 2, 0.8), ...blob(-1, -1, n / 2, 0.8)];
    const labels = [...Array(n / 2).fill(1), ...Array(n / 2).fill(0)];
    const ms = await bench(`log-${n}`, () => logisticRegression(data, labels, { learningRate: 0.1, maxIterations: 200 }));
    r(`Logistic Regression (${n.toLocaleString()} pts, 200 iter)`, ms);
  }

  // DBSCAN
  for (const n of [500, 2000]) {
    const data = [...blob(0, 3, n / 3, 0.5), ...blob(-2, -1, n / 3, 0.5), ...blob(2, -1, n / 3, 0.5)];
    const ms = await bench(`db-${n}`, () => dbscan(data, { eps: 0.8, minPoints: 4 }));
    r(`DBSCAN (${n.toLocaleString()} pts, eps=0.8)`, ms);
  }

  // Naive Bayes
  for (const n of [500, 2000, 10000]) {
    const data = [...blob(2, 2, n / 2), ...blob(-2, -2, n / 2)];
    const labels = [...Array(n / 2).fill(0), ...Array(n / 2).fill(1)];
    const ms = await bench(`nb-${n}`, async () => {
      const model = await naiveBayes(data, labels);
      model.predict(data.slice(0, 50));
    });
    r(`Naive Bayes (${n.toLocaleString()} train, predict 50)`, ms);
  }

  // Decision Tree
  for (const n of [500, 2000, 5000]) {
    const data: number[][] = [];
    const labels: number[] = [];
    for (let i = 0; i < n; i++) {
      const x = (Math.random() - 0.5) * 4, y = (Math.random() - 0.5) * 4;
      data.push([x, y]);
      labels.push(((x > 0) !== (y > 0)) ? 1 : 0);
    }
    const ms = await bench(`dt-${n}`, () => decisionTree(data, labels, { maxDepth: 6, mode: 'classify' }));
    r(`Decision Tree (${n.toLocaleString()} pts, depth=6)`, ms);
  }

  // PCA
  for (const [n, d] of [[500, 10], [2000, 20], [5000, 50]] as [number, number][]) {
    const data: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      const a = randn() * 3;
      for (let j = 0; j < d; j++) row.push(a * (j + 1) * 0.3 + randn() * 0.5);
      data.push(row);
    }
    const ms = await bench(`pca-${n}x${d}`, () => pca(data, { nComponents: 2 }));
    r(`PCA (${n}×${d} → 2)`, ms);
  }

  // Perceptron
  for (const n of [500, 2000, 10000]) {
    const data = [...blob(1.5, 1.5, n / 2, 0.5), ...blob(-1.5, -1.5, n / 2, 0.5)];
    const labels = [...Array(n / 2).fill(1), ...Array(n / 2).fill(0)];
    const ms = await bench(`perc-${n}`, () => perceptron(data, labels, { learningRate: 0.1, maxIterations: 100 }));
    r(`Perceptron (${n.toLocaleString()} pts, 100 iter)`, ms);
  }

  // Seasonality
  {
    const data = Array.from({ length: 1000 }, (_, i) => i * 0.3 + 8 * Math.sin(2 * Math.PI * i / 12) + randn() * 1.5);
    const ms1 = await bench('sd', () => seasonalDecompose(data, 12));
    r('Seasonal Decompose (1K pts)', ms1);
    const ms2 = await bench('ac', () => autocorrelation(data, 50));
    r('Autocorrelation (1K pts, maxLag=50)', ms2);
    const ms3 = await bench('ds', () => detectSeasonality(data));
    r('Detect Seasonality (1K pts)', ms3);
  }

  // ═══════════════════════════════════════
  console.log('\n' + '═'.repeat(72));
  console.log('  BENCHMARK RESULTS (median of 3 runs)');
  console.log('═'.repeat(72));

  const maxName = Math.max(...results.map(r => r[0].length));
  const maxMs = Math.max(...results.map(r => r[1].length));

  for (const [name, ms, detail] of results) {
    console.log(`  ${name.padEnd(maxName)}  ${ms.padStart(maxMs)}${detail ? '  ' + detail : ''}`);
  }

  console.log('═'.repeat(72));
  console.log(`\n  Total: ${results.length} benchmarks completed`);
}

main().catch(console.error);
