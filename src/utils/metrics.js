/**
 * Evaluation metrics utilities
 *
 * Provides MAE, RMSE, R² calculations for regression model evaluation.
 * Also supports seasonal breakdown when season labels are available.
 */

/**
 * Calculate regression metrics for a set of predictions.
 * @param {number[]} actual    - Ground-truth values
 * @param {number[]} predicted - Model predictions
 * @returns {{ mae: number, rmse: number, r2: number }}
 */
export function calculateMetrics(actual, predicted) {
  const n = actual.length;
  if (n === 0) return { mae: 0, rmse: 0, r2: 0 };

  // Mean Absolute Error
  const mae =
    actual.reduce((sum, val, i) => sum + Math.abs(val - predicted[i]), 0) / n;

  // Root Mean Squared Error
  const mse =
    actual.reduce((sum, val, i) => sum + Math.pow(val - predicted[i], 2), 0) / n;
  const rmse = Math.sqrt(mse);

  // R² (coefficient of determination)
  const mean = actual.reduce((a, b) => a + b, 0) / n;
  const totalSS = actual.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0);
  const residualSS = actual.reduce(
    (sum, val, i) => sum + Math.pow(val - predicted[i], 2),
    0
  );
  const r2 = totalSS === 0 ? 0 : 1 - residualSS / totalSS;

  return {
    mae:  parseFloat(mae.toFixed(4)),
    rmse: parseFloat(rmse.toFixed(4)),
    r2:   parseFloat(r2.toFixed(4)),
  };
}

/**
 * Calculate per-season metrics.
 *
 * @param {number[]} actual    - Ground-truth values (test set)
 * @param {number[]} predicted - Model predictions (test set)
 * @param {string[]} seasons   - Season label for each test sample ("Summer"|"Monsoon"|"Winter")
 * @returns {{ Summer: object, Monsoon: object, Winter: object }}
 */
export function calculateSeasonalMetrics(actual, predicted, seasons) {
  const groups = { Summer: [], Monsoon: [], Winter: [] };

  actual.forEach((val, i) => {
    const season = seasons[i];
    if (groups[season]) {
      groups[season].push({ actual: val, predicted: predicted[i] });
    }
  });

  const result = {};
  Object.entries(groups).forEach(([season, pairs]) => {
    if (pairs.length === 0) {
      result[season] = { mae: null, rmse: null, r2: null, count: 0 };
      return;
    }
    const a = pairs.map((p) => p.actual);
    const p = pairs.map((p) => p.predicted);
    result[season] = { ...calculateMetrics(a, p), count: pairs.length };
  });

  return result;
}
