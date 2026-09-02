/**
 * LSTM Model for Monthly Electricity Consumption Forecasting
 *
 * Uses TensorFlow.js (browser) to build, train, and predict with an LSTM
 * neural network on sliding-window time-series sequences.
 *
 * Architecture:
 *   Input  : [batchSize, sequenceLength, nFeatures]
 *   Layer 1: LSTM(32, returnSequences=false)
 *   Layer 2: Dropout(0.1)
 *   Layer 3: Dense(1, activation='linear')
 *
 * Hyperparameters (all adjustable):
 *   LSTM_UNITS   = 32
 *   DROPOUT      = 0.1
 *   EPOCHS       = 50
 *   BATCH_SIZE   = 8
 *   LEARNING_RATE= 0.01
 */

import * as tf from '@tensorflow/tfjs';

// ---------------------------------------------------------------------------
// Default hyperparameters
// ---------------------------------------------------------------------------
export const DEFAULT_HYPERPARAMS = {
  lstmUnits:    32,
  dropout:      0.1,
  epochs:       50,
  batchSize:    8,
  learningRate: 0.01,
};

// ---------------------------------------------------------------------------
// Model construction
// ---------------------------------------------------------------------------

/**
 * Build the LSTM model graph.
 * @param {number} seqLen     - Sequence length (lookback window)
 * @param {number} nFeatures  - Number of input features per time step
 * @param {object} params     - Hyperparameters (merged with defaults)
 * @returns {tf.Sequential}
 */
export function buildLSTMModel(seqLen, nFeatures, params = {}) {
  const hp    = { ...DEFAULT_HYPERPARAMS, ...params };
  const model = tf.sequential();

  // LSTM layer
  model.add(
    tf.layers.lstm({
      units:          hp.lstmUnits,
      inputShape:     [seqLen, nFeatures],
      returnSequences: false,
    })
  );

  // Dropout for regularisation
  model.add(tf.layers.dropout({ rate: hp.dropout }));

  // Dense output layer (regression → linear activation)
  model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

  model.compile({
    optimizer: tf.train.adam(hp.learningRate),
    loss:      'meanSquaredError',
  });

  return model;
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

/**
 * Train the LSTM model asynchronously.
 *
 * @param {number[][][]} trainSeqs  - Shape [nSamples, seqLen, nFeatures]
 * @param {number[]}     trainTgts  - Shape [nSamples] — scaled targets
 * @param {number[][][]} valSeqs    - Validation sequences (may be empty)
 * @param {number[]}     valTgts    - Validation targets
 * @param {object}       params     - Hyperparameters
 * @param {Function}     onEpochEnd - Callback (epoch, logs) called each epoch
 * @returns {Promise<{ model: tf.Sequential, history: object }>}
 */
export async function trainLSTM(
  trainSeqs,
  trainTgts,
  valSeqs    = [],
  valTgts    = [],
  params     = {},
  onEpochEnd = null
) {
  const hp       = { ...DEFAULT_HYPERPARAMS, ...params };
  const seqLen   = trainSeqs[0].length;
  const nFeatures = trainSeqs[0][0].length;

  const model = buildLSTMModel(seqLen, nFeatures, hp);

  // Convert to tensors
  const xTrain = tf.tensor3d(trainSeqs);
  const yTrain = tf.tensor2d(trainTgts, [trainTgts.length, 1]);

  const hasVal   = valSeqs.length > 0;
  const xVal     = hasVal ? tf.tensor3d(valSeqs) : null;
  const yVal     = hasVal ? tf.tensor2d(valTgts, [valTgts.length, 1]) : null;

  const fitConfig = {
    epochs:    hp.epochs,
    batchSize: hp.batchSize,
    shuffle:   false,   // NEVER shuffle time-series data
    verbose:   0,
    callbacks: {
      onEpochEnd: onEpochEnd
        ? (epoch, logs) => onEpochEnd(epoch, logs)
        : undefined,
    },
  };

  if (hasVal) {
    fitConfig.validationData = [xVal, yVal];
  }

  const history = await model.fit(xTrain, yTrain, fitConfig);

  // Dispose tensors to free GPU/CPU memory
  xTrain.dispose();
  yTrain.dispose();
  if (xVal) xVal.dispose();
  if (yVal) yVal.dispose();

  return { model, history };
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

/**
 * Run inference with a trained LSTM model.
 *
 * @param {tf.Sequential} model  - Trained model
 * @param {number[][][]}  seqs   - Input sequences [nSamples, seqLen, nFeatures]
 * @returns {Promise<number[]>}  - Scaled predictions (1-D)
 */
export async function predictLSTM(model, seqs) {
  const xTensor = tf.tensor3d(seqs);
  const rawPred = model.predict(xTensor);
  const values  = await rawPred.data();

  xTensor.dispose();
  rawPred.dispose();

  return Array.from(values);
}

/**
 * Predict a single sliding-window sequence.
 *
 * @param {tf.Sequential} model   - Trained model
 * @param {number[][]}    seq     - One sequence [seqLen, nFeatures]
 * @returns {Promise<number>}     - Single scaled prediction
 */
export async function predictOneLSTM(model, seq) {
  const results = await predictLSTM(model, [seq]);
  return results[0];
}
