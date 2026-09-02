/**
 * Data preparation utilities for time-series electricity forecasting.
 *
 * Pipeline:
 *   rawData (ebBillingData)
 *     → enrichFeatures()        add derived columns
 *     → chronologicalSplit()    time-based train/test (no shuffle)
 *     → createRFDataset()       tabular features for Random Forest
 *     → createLSTMSequences()   sliding-window sequences for LSTM
 *     → MinMaxScaler            fit on train, transform both sets
 */

// ---------------------------------------------------------------------------
// Feature engineering
// ---------------------------------------------------------------------------

/**
 * Add derived features to the raw EB billing records.
 * The raw records must be in chronological order.
 *
 * Added features:
 *   prevMonthUnits   – unitsConsumed from the previous month (NaN for first record)
 *   rollingAvg3Month – 3-month rolling average of unitsConsumed
 *   monthSin / monthCos – cyclic encoding of month (avoids ordinal discontinuity)
 *   seasonEncoded    – Summer=2, Monsoon=1, Winter=0
 *
 * @param {object[]} data - Raw EB billing records in chronological order
 * @returns {object[]}    - Enriched records (first record is dropped because
 *                          it has no prevMonthUnits)
 */
export function enrichFeatures(data) {
  const enriched = data.map((d, i) => {
    const prevMonthUnits  = i === 0 ? null : data[i - 1].unitsConsumed;
    const rollingAvg3Month =
      i < 2
        ? null
        : (data[i - 2].unitsConsumed + data[i - 1].unitsConsumed + d.unitsConsumed) / 3;

    // Cyclic month encoding (month 1–12 → sin/cos on 0–2π)
    const angle      = ((d.month - 1) / 12) * 2 * Math.PI;
    const monthSin   = parseFloat(Math.sin(angle).toFixed(6));
    const monthCos   = parseFloat(Math.cos(angle).toFixed(6));

    const seasonMap  = { Summer: 2, Monsoon: 1, Winter: 0 };
    const seasonEncoded = seasonMap[d.season] ?? 0;

    return {
      ...d,
      prevMonthUnits,
      rollingAvg3Month,
      monthSin,
      monthCos,
      seasonEncoded,
    };
  });

  // Drop records that don't have all derived features (first 2 months)
  return enriched.filter((d) => d.prevMonthUnits !== null && d.rollingAvg3Month !== null);
}

// ---------------------------------------------------------------------------
// Train / test split — chronological (no shuffle)
// ---------------------------------------------------------------------------

/**
 * Split enriched records into training and test sets chronologically.
 * Earlier months → training; later months → test.
 *
 * @param {object[]} data      - Enriched records
 * @param {number}   testRatio - Fraction for test set (default 0.2 = last 20%)
 * @returns {{ trainData: object[], testData: object[] }}
 */
export function chronologicalSplit(data, testRatio = 0.2) {
  const splitIdx = Math.floor(data.length * (1 - testRatio));
  return {
    trainData: data.slice(0, splitIdx),
    testData:  data.slice(splitIdx),
  };
}

// ---------------------------------------------------------------------------
// Random Forest dataset — tabular features
// ---------------------------------------------------------------------------

/** Feature columns used by Random Forest */
export const RF_FEATURES = [
  'month',
  'seasonEncoded',
  'prevMonthUnits',
  'rollingAvg3Month',
  'avgTemperature',
  'monthSin',
  'monthCos',
];

/**
 * Extract feature matrix X and label vector y for Random Forest.
 *
 * @param {object[]} data - Enriched records
 * @returns {{ X: object[], y: number[], seasons: string[] }}
 */
export function createRFDataset(data) {
  const X = data.map((d) => {
    const row = {};
    RF_FEATURES.forEach((f) => { row[f] = d[f]; });
    return row;
  });
  const y       = data.map((d) => d.unitsConsumed);
  const seasons = data.map((d) => d.season);
  return { X, y, seasons };
}

// ---------------------------------------------------------------------------
// LSTM sequences — sliding window
// ---------------------------------------------------------------------------

/** Feature columns used in each LSTM time step */
export const LSTM_FEATURES = [
  'unitsConsumed',
  'avgTemperature',
  'monthSin',
  'monthCos',
  'seasonEncoded',
];

/**
 * Build a Min-Max scaler fitted ONLY on training data.
 * Returns scaler object with transform() and inverseTransform() methods.
 *
 * @param {number[][]} trainMatrix - 2-D array [samples × features]
 * @returns {object} scaler
 */
export function fitMinMaxScaler(trainMatrix) {
  const nFeatures = trainMatrix[0].length;
  const min = new Array(nFeatures).fill(Infinity);
  const max = new Array(nFeatures).fill(-Infinity);

  trainMatrix.forEach((row) => {
    row.forEach((val, j) => {
      if (val < min[j]) min[j] = val;
      if (val > max[j]) max[j] = val;
    });
  });

  const range = min.map((mn, j) => (max[j] - mn === 0 ? 1 : max[j] - mn));

  return {
    min,
    max,
    range,
    transform(matrix) {
      return matrix.map((row) =>
        row.map((val, j) => (val - min[j]) / range[j])
      );
    },
    inverseTransform(matrix) {
      return matrix.map((row) =>
        row.map((val, j) => val * range[j] + min[j])
      );
    },
    /** Inverse-transform only the first column (unitsConsumed) */
    inverseTransformTarget(scaledValues) {
      return scaledValues.map((v) => v * range[0] + min[0]);
    },
    /** Scale only the first column (unitsConsumed) for a single value */
    transformTarget(value) {
      return (value - min[0]) / range[0];
    },
  };
}

/**
 * Convert an enriched record array into a row-matrix for the LSTM features.
 *
 * @param {object[]} data - Enriched records
 * @returns {number[][]}  - 2-D matrix [samples × LSTM_FEATURES.length]
 */
export function toFeatureMatrix(data) {
  return data.map((d) => LSTM_FEATURES.map((f) => d[f]));
}

/**
 * Create sliding-window sequences for LSTM training.
 * Each sequence uses `seqLen` consecutive time steps to predict the next step's
 * unitsConsumed (first LSTM feature, index 0).
 *
 * @param {number[][]} scaledMatrix - Already-scaled feature matrix
 * @param {number}     seqLen       - Sequence length (lookback window)
 * @returns {{ sequences: number[][][], targets: number[] }}
 *   sequences shape: [samples, seqLen, nFeatures]
 *   targets   shape: [samples]  (scaled unitsConsumed of next step)
 */
export function createLSTMSequences(scaledMatrix, seqLen = 4) {
  const sequences = [];
  const targets   = [];

  for (let i = 0; i <= scaledMatrix.length - seqLen - 1; i++) {
    sequences.push(scaledMatrix.slice(i, i + seqLen));
    // Target is unitsConsumed (feature index 0) of the step right after the window
    targets.push(scaledMatrix[i + seqLen][0]);
  }

  return { sequences, targets };
}

// ---------------------------------------------------------------------------
// Full preparation pipeline (convenience export)
// ---------------------------------------------------------------------------

/**
 * Run the complete data-preparation pipeline.
 *
 * @param {object[]} rawData  - Raw EB billing records
 * @param {number}   seqLen   - LSTM sequence length
 * @param {number}   testRatio
 * @returns {object} Everything needed for both models
 */
export function prepareAllData(rawData, seqLen = 4, testRatio = 0.2) {
  // 1. Feature engineering (drops first 2 records)
  const enriched = enrichFeatures(rawData);

  // 2. Chronological split
  const { trainData, testData } = chronologicalSplit(enriched, testRatio);

  // 3. Random Forest datasets (no normalization needed for RF)
  const rfTrain = createRFDataset(trainData);
  const rfTest  = createRFDataset(testData);

  // 4. LSTM normalization — fit scaler on training data ONLY
  const trainMatrix = toFeatureMatrix(trainData);
  const testMatrix  = toFeatureMatrix(testData);

  const scaler        = fitMinMaxScaler(trainMatrix);
  const scaledTrain   = scaler.transform(trainMatrix);

  // 5. Sliding-window sequences for LSTM
  // Training sequences from training data
  const lstmTrain = createLSTMSequences(scaledTrain, seqLen);

  // Test sequences: use last `seqLen` records from training as context for the first test month (Jan 2024),
  // then slide across all test months through Dec 2024.
  // This produces exactly testData.length (12) sequences corresponding 1-to-1 with testData[0..11].
  const fullScaled    = scaler.transform([...trainMatrix, ...testMatrix]);
  const splitIdx      = trainMatrix.length;
  const testSequences = [];
  const testTargets   = [];

  for (let i = splitIdx - seqLen; i < fullScaled.length - seqLen; i++) {
    testSequences.push(fullScaled.slice(i, i + seqLen));
    testTargets.push(fullScaled[i + seqLen][0]);
  }

  return {
    enriched,
    trainData,
    testData,
    rfTrain,
    rfTest,
    scaler,
    lstmTrain,
    lstmTest: { sequences: testSequences, targets: testTargets },
  };
}
