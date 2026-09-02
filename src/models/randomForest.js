/**
 * Random Forest Regressor — pure JavaScript implementation
 *
 * A genuine ensemble of decision trees using:
 *   - Bootstrap sampling (bagging) per tree
 *   - Random feature subsets at each split (feature randomness)
 *   - Averaging predictions across all trees
 *   - Seeded Pseudo-Random Number Generator (PRNG) for reproducible training
 *
 * Hyperparameters:
 *   nTrees           : number of trees in the ensemble (default 25)
 *   maxDepth         : maximum tree depth (default 7)
 *   minSamples       : minimum samples required to split a node (default 3)
 *   featureSubsetSize: how many features to consider at each split
 *                      (default: floor(sqrt(total features)))
 *   seed             : optional integer seed for reproducibility
 */

// ---------------------------------------------------------------------------
// Seeded PRNG (Mulberry32)
// ---------------------------------------------------------------------------
export function createPrng(seed = 42) {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (t >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// Internal Decision Tree (CART — regression)
// ---------------------------------------------------------------------------

class RegressionTree {
  constructor(maxDepth, minSamples, featureSubsetSize, rng = Math.random) {
    this.maxDepth          = maxDepth;
    this.minSamples        = minSamples;
    this.featureSubsetSize = featureSubsetSize;
    this.rng               = rng;
    this.root              = null;
  }

  // ------- Public API -------

  fit(X, y) {
    this.featureNames = Object.keys(X[0]);
    this.root = this._buildNode(X, y, 0);
  }

  predict(X) {
    return X.map((x) => this._traverse(x, this.root));
  }

  // ------- Tree building -------

  _buildNode(X, y, depth) {
    // Leaf conditions
    if (depth >= this.maxDepth || X.length <= this.minSamples || this._allSame(y)) {
      return { isLeaf: true, value: this._mean(y) };
    }

    const split = this._bestSplit(X, y);

    if (!split) {
      return { isLeaf: true, value: this._mean(y) };
    }

    const { feature, threshold, leftIdx, rightIdx } = split;

    return {
      isLeaf: false,
      feature,
      threshold,
      left:  this._buildNode(leftIdx.map((i) => X[i]), leftIdx.map((i) => y[i]), depth + 1),
      right: this._buildNode(rightIdx.map((i) => X[i]), rightIdx.map((i) => y[i]), depth + 1),
    };
  }

  _bestSplit(X, y) {
    // Random feature subset using seeded rng
    const allFeatures     = Object.keys(X[0]);
    const subsetSize      = Math.min(this.featureSubsetSize, allFeatures.length);
    const shuffled        = [...allFeatures].sort(() => this.rng() - 0.5);
    const candidateFeats  = shuffled.slice(0, subsetSize);

    let bestMse   = Infinity;
    let bestSplit = null;

    for (const feature of candidateFeats) {
      const values  = X.map((x) => x[feature]);
      const sorted  = [...new Set(values)].sort((a, b) => a - b);

      for (let i = 0; i < sorted.length - 1; i++) {
        const threshold = (sorted[i] + sorted[i + 1]) / 2;
        const leftIdx   = [];
        const rightIdx  = [];

        X.forEach((x, idx) => {
          (x[feature] <= threshold ? leftIdx : rightIdx).push(idx);
        });

        if (leftIdx.length === 0 || rightIdx.length === 0) continue;

        const mse = this._weightedMse(
          leftIdx.map((i) => y[i]),
          rightIdx.map((i) => y[i])
        );

        if (mse < bestMse) {
          bestMse   = mse;
          bestSplit = { feature, threshold, leftIdx, rightIdx };
        }
      }
    }

    return bestSplit;
  }

  // ------- Prediction -------

  _traverse(x, node) {
    if (node.isLeaf) return node.value;
    return x[node.feature] <= node.threshold
      ? this._traverse(x, node.left)
      : this._traverse(x, node.right);
  }

  // ------- Helpers -------

  _mean(arr) {
    return arr.length === 0 ? 0 : arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  _weightedMse(leftY, rightY) {
    const totalN   = leftY.length + rightY.length;
    const leftMse  = this._mse(leftY);
    const rightMse = this._mse(rightY);
    return (leftMse * leftY.length + rightMse * rightY.length) / totalN;
  }

  _mse(arr) {
    const m = this._mean(arr);
    return arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length;
  }

  _allSame(arr) {
    return arr.every((v) => v === arr[0]);
  }
}

// ---------------------------------------------------------------------------
// Random Forest Regressor
// ---------------------------------------------------------------------------

export class RandomForest {
  /**
   * @param {object} options
   * @param {number} options.nTrees            - Number of trees (default 25)
   * @param {number} options.maxDepth          - Max tree depth (default 7)
   * @param {number} options.minSamples        - Min samples to split (default 3)
   * @param {number} options.featureSubsetSize - Features per split (default sqrt(nFeatures))
   * @param {number} options.seed              - Seed for PRNG (optional)
   */
  constructor({
    nTrees            = 25,
    maxDepth          = 7,
    minSamples        = 3,
    featureSubsetSize = null,
    seed              = null,
  } = {}) {
    this.nTrees            = nTrees;
    this.maxDepth          = maxDepth;
    this.minSamples        = minSamples;
    this.featureSubsetSize = featureSubsetSize;
    this.seed              = seed;
    this.rng               = seed !== null ? createPrng(seed) : Math.random;
    this.trees             = [];
  }

  /**
   * Train the Random Forest.
   * @param {object[]} X - Feature objects array
   * @param {number[]} y - Target values array
   */
  fit(X, y) {
    this.trees = [];
    const n          = X.length;
    const nFeatures  = Object.keys(X[0]).length;
    const subsetSize = this.featureSubsetSize ?? Math.max(1, Math.floor(Math.sqrt(nFeatures)));

    for (let t = 0; t < this.nTrees; t++) {
      // Bootstrap sample (sampling with replacement using seeded rng)
      const bootstrapIdx = Array.from({ length: n }, () => Math.floor(this.rng() * n));
      const bootX        = bootstrapIdx.map((i) => X[i]);
      const bootY        = bootstrapIdx.map((i) => y[i]);

      const tree = new RegressionTree(this.maxDepth, this.minSamples, subsetSize, this.rng);
      tree.fit(bootX, bootY);
      this.trees.push(tree);
    }

    return this;
  }

  /**
   * Predict target values (average across all trees).
   * @param {object[]} X - Feature objects array
   * @returns {number[]}
   */
  predict(X) {
    if (this.trees.length === 0) throw new Error('RandomForest: call fit() before predict()');

    // Collect predictions from each tree, then average
    const allPreds = this.trees.map((tree) => tree.predict(X));

    return X.map((_, i) => {
      const sum = allPreds.reduce((acc, preds) => acc + preds[i], 0);
      return sum / this.nTrees;
    });
  }

  /**
   * Predict a single sample (convenience wrapper).
   * @param {object} x - Single feature object
   * @returns {number}
   */
  predictOne(x) {
    return this.predict([x])[0];
  }
}

export default RandomForest;
