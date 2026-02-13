import {
  init,
  linearRegression,
  linearRegressionSimple,
  polynomialRegression,
  exponentialRegression,
  logarithmicRegression,
  powerRegression,
  trendForecast,
  sma, ema, wma,
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

// Helpers
function randn() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function blob(cx: number, cy: number, n: number, spread = 0.5): number[][] {
  return Array.from({ length: n }, () => [cx + randn() * spread, cy + randn() * spread]);
}

async function bench(name: string, fn: () => Promise<any>): Promise<{ name: string; ms: number; result: any }> {
  // Warmup
  await fn();
  // Actual
  const t0 = performance.now();
  const result = await fn();
  const ms = performance.now() - t0;
  return { name, ms, result };
}

async function main() {
  console.log('Initializing WASM...');
  const t0 = performance.now();
  await init();
  console.log(`WASM loaded in ${(performance.now() - t0).toFixed(1)}ms\n`);

  const results: { name: string; ms: string; detail: string }[] = [];

  function report(name: string, ms: number, detail: string) {
    results.push({ name, ms: ms < 0.1 ? '<0.1ms' : ms.toFixed(2) + 'ms', detail });
  }

  // ═══════════════════════════════════════
  // REGRESSION BENCHMARKS
  // ═══════════════════════════════════════
  console.log('--- REGRESSION ---');

  // Linear regression at various sizes
  for (const n of [1_000, 10_000, 100_000, 1_000_000]) {
    const x = Array.from({ length: n }, (_, i) => i);
    const y = x.map(xi => 2 * xi + 10 + Math.random());
    const { ms, result } = await bench(`linearRegression ${n.toLocaleString()}pts`, () => linearRegression(x, y));
    report(`Linear Regression (${n.toLocaleString()} pts)`, ms, `slope=${(result as any).slope.toFixed(4)}, R²=${(result as any).rSquared.toFixed(6)}`);
  }

  // Polynomial regression
  {
    const x = Array.from({ length: 1000 }, (_, i) => i * 0.01);
    const y = x.map(xi => 3 * xi * xi - 2 * xi + 1 + randn() * 0.1);
    const { ms, result } = await bench('polynomialRegression 1K', () => polynomialRegression(x, y, { degree: 2 }));
    report('Polynomial Regression (1K pts, deg 2)', ms, `R²=${(result as any).rSquared.toFixed(6)}`);
  }

  // Exponential regression
  {
    const x = Array.from({ length: 500 }, (_, i) => i * 0.01);
    const y = x.map(xi => 2 * Math.exp(0.5 * xi) + randn() * 0.1);
    const { ms, result } = await bench('exponentialRegression 500', () => exponentialRegression(x, y));
    report('Exponential Regression (500 pts)', ms, `a=${(result as any).a.toFixed(4)}, b=${(result as any).b.toFixed(4)}`);
  }

  // Trend forecast
  {
    const data = Array.from({ length: 10000 }, (_, i) => i * 1.5 + randn() * 10);
    const { ms, result } = await bench('trendForecast 10K', () => trendForecast(data, 10));
    report('Trend Forecast (10K pts, 10 periods)', ms, `direction=${(result as any).direction}, strength=${(result as any).strength.toFixed(4)}`);
  }

  // Moving averages
  {
    const data = Array.from({ length: 100000 }, () => Math.random() * 100);
    const { ms: ms1 } = await bench('SMA 100K', () => sma(data, 20));
    report('SMA (100K pts, window=20)', ms1, '');
    const { ms: ms2 } = await bench('EMA 100K', () => ema(data, 20));
    report('EMA (100K pts, window=20)', ms2, '');
    const { ms: ms3 } = await bench('WMA 100K', () => wma(data, 20));
    report('WMA (100K pts, window=20)', ms3, '');
  }

  // ═══════════════════════════════════════
  // ML ALGORITHM BENCHMARKS
  // ═══════════════════════════════════════
  console.log('\n--- ML ALGORITHMS ---');

  // k-Means
  for (const n of [500, 2000, 10000]) {
    const data: number[][] = [];
    for (let i = 0; i < 5; i++) {
      data.push(...blob((Math.random() - 0.5) * 10, (Math.random() - 0.5) * 10, Math.ceil(n / 5)));
    }
    const { ms, result } = await bench(`kmeans ${n}`, () => kmeans(data, { k: 5 }));
    report(`k-Means (${n.toLocaleString()} pts, k=5)`, ms, `iters=${(result as any).iterations}, inertia=${(result as any).inertia.toFixed(1)}`);
  }

  // kNN
  for (const n of [200, 1000, 5000]) {
    const data = [...blob(2, 2, n / 2), ...blob(-2, -2, n / 2)];
    const labels = [...Array(n / 2).fill(0), ...Array(n / 2).fill(1)];
    const { ms, result } = await bench(`knn fit+predict ${n}`, async () => {
      const model = await knnClassifier(data, labels, { k: 5 });
      const preds = model.predict(data.slice(0, 20));
      return { model, preds };
    });
    report(`kNN (${n.toLocaleString()} train, k=5, predict 20)`, ms, `samples=${(result as any).model.nSamples}`);
  }

  // Logistic Regression
  for (const n of [200, 1000, 5000]) {
    const data = [...blob(1, 1, n / 2, 0.8), ...blob(-1, -1, n / 2, 0.8)];
    const labels = [...Array(n / 2).fill(1), ...Array(n / 2).fill(0)];
    const { ms, result } = await bench(`logistic ${n}`, () => logisticRegression(data, labels, { learningRate: 0.1, maxIterations: 200 }));
    report(`Logistic Regression (${n.toLocaleString()} pts, 200 iters)`, ms, `loss=${(result as any).loss.toFixed(4)}, bias=${(result as any).bias.toFixed(4)}`);
  }

  // DBSCAN
  for (const n of [500, 2000, 5000]) {
    const data = [...blob(0, 3, n / 3, 0.5), ...blob(-2, -1, n / 3, 0.5), ...blob(2, -1, n / 3, 0.5)];
    for (let i = 0; i < n * 0.05; i++) data.push([(Math.random() - 0.5) * 10, (Math.random() - 0.5) * 10]);
    const { ms, result } = await bench(`dbscan ${n}`, () => dbscan(data, { eps: 0.8, minPoints: 4 }));
    report(`DBSCAN (${n.toLocaleString()} pts, eps=0.8)`, ms, `clusters=${(result as any).nClusters}, noise=${(result as any).nNoise}`);
  }

  // Naive Bayes
  for (const n of [500, 2000, 10000]) {
    const data = [...blob(2, 2, n / 2), ...blob(-2, -2, n / 2)];
    const labels = [...Array(n / 2).fill(0), ...Array(n / 2).fill(1)];
    const { ms, result } = await bench(`naiveBayes ${n}`, async () => {
      const model = await naiveBayes(data, labels);
      const preds = model.predict(data.slice(0, 50));
      return { model, preds };
    });
    report(`Naive Bayes (${n.toLocaleString()} train, predict 50)`, ms, `classes=${(result as any).model.nClasses}`);
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
    const { ms, result } = await bench(`decisionTree ${n}`, () => decisionTree(data, labels, { maxDepth: 6, mode: 'classify' }));
    report(`Decision Tree (${n.toLocaleString()} pts, depth=6)`, ms, `depth=${(result as any).depth}, nodes=${(result as any).nNodes}`);
  }

  // PCA
  for (const [n, d] of [[500, 10], [2000, 20], [5000, 50]]) {
    const data: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      const a = randn() * 3;
      for (let j = 0; j < d; j++) row.push(a * (j + 1) * 0.3 + randn() * 0.5);
      data.push(row);
    }
    const { ms, result } = await bench(`pca ${n}x${d}`, () => pca(data, { nComponents: 2 }));
    const ratios = (result as any).getExplainedVarianceRatio();
    report(`PCA (${n}x${d} → 2)`, ms, `var=[${ratios.map((r: number) => (r * 100).toFixed(1) + '%').join(', ')}]`);
  }

  // Perceptron
  for (const n of [500, 2000, 10000]) {
    const data = [...blob(1.5, 1.5, n / 2, 0.5), ...blob(-1.5, -1.5, n / 2, 0.5)];
    const labels = [...Array(n / 2).fill(1), ...Array(n / 2).fill(0)];
    const { ms, result } = await bench(`perceptron ${n}`, () => perceptron(data, labels, { learningRate: 0.1, maxIterations: 100 }));
    report(`Perceptron (${n.toLocaleString()} pts, 100 iters)`, ms, `converged=${(result as any).converged}, iters=${(result as any).iterations}`);
  }

  // Seasonality
  {
    const data = Array.from({ length: 1000 }, (_, i) => i * 0.3 + 8 * Math.sin(2 * Math.PI * i / 12) + randn() * 1.5);
    const { ms: ms1 } = await bench('seasonalDecompose 1K', () => seasonalDecompose(data, 12));
    report('Seasonal Decompose (1K pts, period=12)', ms1, '');
    const { ms: ms2 } = await bench('autocorrelation 1K', () => autocorrelation(data, 50));
    report('Autocorrelation (1K pts, maxLag=50)', ms2, '');
    const { ms: ms3, result } = await bench('detectSeasonality 1K', () => detectSeasonality(data));
    report('Detect Seasonality (1K pts)', ms3, `period=${(result as any).period}, strength=${(result as any).strength.toFixed(3)}`);
  }

  // ═══════════════════════════════════════
  // PRINT RESULTS
  // ═══════════════════════════════════════
  console.log('\n' + '═'.repeat(90));
  console.log('  BENCHMARK RESULTS');
  console.log('═'.repeat(90));

  const maxName = Math.max(...results.map(r => r.name.length));
  const maxMs = Math.max(...results.map(r => r.ms.length));

  for (const r of results) {
    const name = r.name.padEnd(maxName);
    const ms = r.ms.padStart(maxMs);
    const detail = r.detail ? `  ${r.detail}` : '';
    console.log(`  ${name}  ${ms}${detail}`);
  }

  console.log('═'.repeat(90));
}

main().catch(console.error);
