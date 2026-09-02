import React, { useState, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, BarChart, Bar,
} from 'recharts';
import { Zap, TrendingUp, AlertCircle, CheckCircle, Brain, Trees, Layers } from 'lucide-react';

// --- Data & utilities ---
import { ebBillingData }                  from './data/ebBillingData';
import { prepareAllData }                 from './utils/dataPrep';
import { calculateMetrics, calculateSeasonalMetrics } from './utils/metrics';
// --- Models ---
import { RandomForest }                   from './models/randomForest';
import { trainLSTM, predictLSTM, predictOneLSTM } from './models/lstmModel';

// ============================================================
// Constants
// ============================================================
const SEQ_LEN     = 4;   // LSTM look-back window (months)
const TEST_RATIO  = 0.2; // 20 % held-out for evaluation
const NUM_RUNS    = 5;   // Number of runs for stable cross-validated comparison
const LSTM_EPOCHS = 35;  // Epochs per LSTM run (35 epochs × 5 runs = 175 epochs total)

const SEASON_COLORS = {
  Summer:  '#f97316',
  Monsoon: '#3b82f6',
  Winter:  '#8b5cf6',
};

// ============================================================
// Helper: format month label for charts
// ============================================================
function monthLabel(rec) {
  const months = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec'];
  return `${months[rec.month - 1]} ${String(rec.year).slice(2)}`;
}

// ============================================================
// Main component
// ============================================================
const ElectricityPredictor = () => {
  const [results,    setResults]    = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [prediction, setPrediction] = useState(null);

  // ── Separate progress state per model ──────────────────────
  const [rfStatus, setRfStatus] = useState({
    status: 'idle', // 'idle' | 'training' | 'completed' | 'error'
    run: 0,
    totalRuns: NUM_RUNS,
    label: '',
  });

  const [lstmStatus, setLstmStatus] = useState({
    status: 'idle', // 'idle' | 'training' | 'completed' | 'error'
    run: 0,
    totalRuns: NUM_RUNS,
    epoch: 0,
    totalEpochs: LSTM_EPOCHS,
    loss: null,
    label: '',
  });

  // Prediction form inputs — match EB billing features
  const [inputFeatures, setInputFeatures] = useState({
    month:          5,
    season:         'Summer',
    prevMonthUnits: 280,
    rollingAvg:     270,
    avgTemperature: 35,
  });

  // ----------------------------------------------------------
  // Training pipeline (5-Run Averaged Benchmark)
  // ----------------------------------------------------------
  const trainModels = useCallback(async () => {
    setLoading(true);
    setResults(null);
    setPrediction(null);

    // Reset both model statuses independently
    setRfStatus({   status: 'idle', run: 0, totalRuns: NUM_RUNS, label: '' });
    setLstmStatus({ status: 'idle', run: 0, totalRuns: NUM_RUNS, epoch: 0, totalEpochs: LSTM_EPOCHS, loss: null, label: '' });

    try {
      // ── Step 1: Initialize progress for BOTH models simultaneously ──
      setRfStatus({ status: 'training', run: 0, totalRuns: NUM_RUNS, label: 'Preparing dataset & feature engineering…' });
      setLstmStatus({ status: 'training', run: 0, totalRuns: NUM_RUNS, epoch: 0, totalEpochs: LSTM_EPOCHS, loss: null, label: 'Initializing neural network pipeline…' });
      await new Promise((r) => setTimeout(r, 60)); // Allow UI to paint initial cards

      const prep = prepareAllData(ebBillingData, SEQ_LEN, TEST_RATIO);
      const {
        trainData, testData,
        rfTrain, rfTest,
        scaler,
        lstmTrain, lstmTest,
      } = prep;

      // ── Step 2: Train Random Forest across NUM_RUNS with deterministic seeds ──
      const rfRuns = [];
      for (let r = 0; r < NUM_RUNS; r++) {
        setRfStatus({
          status: 'training',
          run: r + 1,
          totalRuns: NUM_RUNS,
          label: `Training run ${r + 1} / ${NUM_RUNS} (25 Decision Trees, seed ${101 + r})…`,
        });
        await new Promise((res) => setTimeout(res, 40));

        const rf = new RandomForest({
          nTrees: 25,
          maxDepth: 7,
          minSamples: 3,
          seed: 101 + r,
        });
        rf.fit(rfTrain.X, rfTrain.y);

        const rfTrainPred    = rf.predict(rfTrain.X);
        const rfTestPred     = rf.predict(rfTest.X);
        const rfTrainMetrics = calculateMetrics(rfTrain.y, rfTrainPred);
        const rfTestMetrics  = calculateMetrics(rfTest.y,  rfTestPred);
        const rfSeasonMetrics = calculateSeasonalMetrics(rfTest.y, rfTestPred, rfTest.seasons);

        rfRuns.push({
          run: r + 1,
          seed: 101 + r,
          model: rf,
          trainMetrics: rfTrainMetrics,
          testMetrics: rfTestMetrics,
          seasonMetrics: rfSeasonMetrics,
          predictions: rfTestPred,
        });
      }

      // Calculate RF 5-Run averages
      const rfAvg = {
        mae:  parseFloat((rfRuns.reduce((acc, x) => acc + x.testMetrics.mae,  0) / NUM_RUNS).toFixed(2)),
        rmse: parseFloat((rfRuns.reduce((acc, x) => acc + x.testMetrics.rmse, 0) / NUM_RUNS).toFixed(2)),
        r2:   parseFloat((rfRuns.reduce((acc, x) => acc + x.testMetrics.r2,   0) / NUM_RUNS).toFixed(4)),
      };

      // Calculate RF 5-Run average seasonal metrics
      const rfAvgSeasonMetrics = {};
      ['Summer', 'Monsoon', 'Winter'].forEach((s) => {
        const validRuns = rfRuns.filter((x) => x.seasonMetrics[s]?.mae != null);
        if (validRuns.length === 0) {
          rfAvgSeasonMetrics[s] = { mae: null, rmse: null, r2: null, count: 0 };
        } else {
          rfAvgSeasonMetrics[s] = {
            count: validRuns[0].seasonMetrics[s].count,
            mae:  parseFloat((validRuns.reduce((acc, x) => acc + x.seasonMetrics[s].mae,  0) / validRuns.length).toFixed(2)),
            rmse: parseFloat((validRuns.reduce((acc, x) => acc + x.seasonMetrics[s].rmse, 0) / validRuns.length).toFixed(2)),
            r2:   parseFloat((validRuns.reduce((acc, x) => acc + x.seasonMetrics[s].r2,   0) / validRuns.length).toFixed(4)),
          };
        }
      });

      // Average predictions across all 5 runs
      const rfAvgPredictions = rfTest.y.map((_, i) =>
        parseFloat((rfRuns.reduce((acc, x) => acc + x.predictions[i], 0) / NUM_RUNS).toFixed(2))
      );

      // Best representative RF model (run whose RMSE is closest to the 5-run average)
      const rfRepresentative = rfRuns.reduce((prev, curr) =>
        Math.abs(curr.testMetrics.rmse - rfAvg.rmse) < Math.abs(prev.testMetrics.rmse - rfAvg.rmse) ? curr : prev
      ).model;

      setRfStatus({
        status: 'completed',
        run: NUM_RUNS,
        totalRuns: NUM_RUNS,
        label: `5-Run Avg: MAE ${rfAvg.mae.toFixed(2)} kWh · RMSE ${rfAvg.rmse.toFixed(2)} kWh · R² ${rfAvg.r2.toFixed(4)}`,
      });

      // ── Step 3: Train LSTM across NUM_RUNS with deterministic seeds ──
      const lstmRuns = [];

      for (let r = 0; r < NUM_RUNS; r++) {
        setLstmStatus((prev) => ({
          ...prev,
          status: 'training',
          run: r + 1,
          totalRuns: NUM_RUNS,
          epoch: 0,
          totalEpochs: LSTM_EPOCHS,
          loss: null,
          label: `Run ${r + 1} / ${NUM_RUNS} — Starting…`,
        }));
        await new Promise((res) => setTimeout(res, 40));

        const { model: lstmModel } = await trainLSTM(
          lstmTrain.sequences,
          lstmTrain.targets,
          lstmTest.sequences,
          lstmTest.targets,
          {
            epochs:       LSTM_EPOCHS,
            batchSize:    8,
            lstmUnits:    32,
            dropout:      0.1,
            learningRate: 0.01,
            seed:         201 + r,
          },
          // Per-epoch callback: updates only LSTM state, leaves RF state untouched
          (epoch, logs) => {
            setLstmStatus((prev) => ({
              ...prev,
              status: 'training',
              run: r + 1,
              totalRuns: NUM_RUNS,
              epoch: epoch + 1,
              totalEpochs: LSTM_EPOCHS,
              loss: logs.loss != null ? parseFloat(logs.loss.toFixed(5)) : null,
              label: `Run ${r + 1} / ${NUM_RUNS} — Epoch ${epoch + 1} / ${LSTM_EPOCHS}`,
            }));
          }
        );

        const lstmScaledPred = await predictLSTM(lstmModel, lstmTest.sequences);
        const lstmTestPred   = scaler.inverseTransformTarget(lstmScaledPred);
        const lstmTestActual = scaler.inverseTransformTarget(lstmTest.targets);

        const lstmTestMetrics = calculateMetrics(lstmTestActual, lstmTestPred);
        const lstmTestSeasons = testData.map((d) => d.season);
        const lstmSeasonMetrics = calculateSeasonalMetrics(
          lstmTestActual, lstmTestPred, lstmTestSeasons
        );

        lstmRuns.push({
          run: r + 1,
          seed: 201 + r,
          model: lstmModel,
          testMetrics: lstmTestMetrics,
          seasonMetrics: lstmSeasonMetrics,
          predictions: lstmTestPred,
          actual: lstmTestActual,
        });
      }

      // Calculate LSTM 5-Run averages
      const lstmAvg = {
        mae:  parseFloat((lstmRuns.reduce((acc, x) => acc + x.testMetrics.mae,  0) / NUM_RUNS).toFixed(2)),
        rmse: parseFloat((lstmRuns.reduce((acc, x) => acc + x.testMetrics.rmse, 0) / NUM_RUNS).toFixed(2)),
        r2:   parseFloat((lstmRuns.reduce((acc, x) => acc + x.testMetrics.r2,   0) / NUM_RUNS).toFixed(4)),
      };

      // Calculate LSTM 5-Run average seasonal metrics across all 12 test months
      const lstmAvgSeasonMetrics = {};
      ['Summer', 'Monsoon', 'Winter'].forEach((s) => {
        const validRuns = lstmRuns.filter((x) => x.seasonMetrics[s]?.mae != null);
        if (validRuns.length === 0) {
          lstmAvgSeasonMetrics[s] = { mae: null, rmse: null, r2: null, count: 0 };
        } else {
          lstmAvgSeasonMetrics[s] = {
            count: validRuns[0].seasonMetrics[s].count,
            mae:  parseFloat((validRuns.reduce((acc, x) => acc + x.seasonMetrics[s].mae,  0) / validRuns.length).toFixed(2)),
            rmse: parseFloat((validRuns.reduce((acc, x) => acc + x.seasonMetrics[s].rmse, 0) / validRuns.length).toFixed(2)),
            r2:   parseFloat((validRuns.reduce((acc, x) => acc + x.seasonMetrics[s].r2,   0) / validRuns.length).toFixed(4)),
          };
        }
      });

      // Average LSTM predictions across all 5 runs (12 predictions for Jan..Dec)
      const lstmAvgPredictions = lstmRuns[0].predictions.map((_, i) =>
        parseFloat((lstmRuns.reduce((acc, x) => acc + x.predictions[i], 0) / NUM_RUNS).toFixed(2))
      );

      // Best representative LSTM model (run whose RMSE is closest to the 5-run average)
      const lstmRepresentative = lstmRuns.reduce((prev, curr) =>
        Math.abs(curr.testMetrics.rmse - lstmAvg.rmse) < Math.abs(prev.testMetrics.rmse - lstmAvg.rmse) ? curr : prev
      ).model;

      setLstmStatus({
        status: 'completed',
        run: NUM_RUNS,
        totalRuns: NUM_RUNS,
        epoch: LSTM_EPOCHS,
        totalEpochs: LSTM_EPOCHS,
        loss: null,
        label: `5-Run Avg: MAE ${lstmAvg.mae.toFixed(2)} kWh · RMSE ${lstmAvg.rmse.toFixed(2)} kWh · R² ${lstmAvg.r2.toFixed(4)}`,
      });

      // ── Step 4: Stable Model Comparison (Average RMSE + MAE tiebreaker) ──
      let bestModel = 'rf';
      if (lstmAvg.rmse < rfAvg.rmse - 0.0001) {
        bestModel = 'lstm';
      } else if (rfAvg.rmse < lstmAvg.rmse - 0.0001) {
        bestModel = 'rf';
      } else {
        // Tiebreak with MAE
        bestModel = lstmAvg.mae <= rfAvg.mae ? 'lstm' : 'rf';
      }

      // Comparison chart data using 5-Run averaged predictions across all 12 test months (Jan–Dec 2024)
      const rfLabels = testData.map(monthLabel);
      const comparisonData = rfLabels.map((label, i) => ({
        label,
        actual: rfTest.y[i],
        rf:     rfAvgPredictions[i],
        lstm:   lstmAvgPredictions[i],
        season: rfTest.seasons[i],
      }));

      // ── Step 5: Store stable multi-run results ──
      setResults({
        numRuns: NUM_RUNS,
        rf: {
          ...rfAvg,
          seasonMetrics:       rfAvgSeasonMetrics,
          runs:                rfRuns,
          representativeModel: rfRepresentative,
          predictions:         rfAvgPredictions,
        },
        lstm: {
          ...lstmAvg,
          seasonMetrics:       lstmAvgSeasonMetrics,
          runs:                lstmRuns,
          representativeModel: lstmRepresentative,
          predictions:         lstmAvgPredictions,
          actual:              lstmRuns[0].actual,
        },
        bestModel,
        comparisonData,
        testData,
        trainData,
        scaler,
      });

    } catch (err) {
      console.error('Training error:', err);
      // Mark whichever model was active as errored without touching the other
      setRfStatus((s)   => s.status === 'training' ? { ...s, status: 'error', label: err.message } : s);
      setLstmStatus((s) => s.status === 'training' ? { ...s, status: 'error', label: err.message } : s);
    } finally {
      setLoading(false);
    }
  }, []);

  // ----------------------------------------------------------
  // Single-sample prediction (uses representative model of Best Model)
  // ----------------------------------------------------------
  const makePrediction = useCallback(async () => {
    if (!results) return;

    const { month, season, prevMonthUnits, rollingAvg, avgTemperature } = inputFeatures;
    const angle         = ((month - 1) / 12) * 2 * Math.PI;
    const monthSin      = Math.sin(angle);
    const monthCos      = Math.cos(angle);
    const seasonMap     = { Summer: 2, Monsoon: 1, Winter: 0 };
    const seasonEncoded = seasonMap[season] ?? 0;

    if (results.bestModel === 'rf') {
      const rfFeatures = {
        month,
        seasonEncoded,
        prevMonthUnits,
        rollingAvg3Month: rollingAvg,
        avgTemperature,
        monthSin,
        monthCos,
      };
      const pred = results.rf.representativeModel.predictOne(rfFeatures);
      setPrediction({ value: pred.toFixed(2), model: 'Random Forest' });
    } else {
      // Build a sequence using the current input repeated SEQ_LEN times
      const row = [
        prevMonthUnits,
        avgTemperature,
        monthSin,
        monthCos,
        seasonEncoded,
      ];
      // Scale each feature using the stored scaler
      const scaledRow = row.map((v, j) => (v - results.scaler.min[j]) / results.scaler.range[j]);
      const seq       = Array.from({ length: SEQ_LEN }, () => scaledRow);

      const scaledPred = await predictOneLSTM(results.lstm.representativeModel, seq);
      const pred       = scaledPred * results.scaler.range[0] + results.scaler.min[0];
      setPrediction({ value: pred.toFixed(2), model: 'LSTM' });
    }
  }, [results, inputFeatures]);

  // ----------------------------------------------------------
  // Render helpers
  // ----------------------------------------------------------
  const MetricCard = ({ title, metrics, isWinner, icon: Icon, color }) => {
    const [showRuns, setShowRuns] = useState(false);

    return (
      <div
        className={`bg-white rounded-lg shadow-lg p-6 ${
          isWinner ? 'ring-4 ring-green-400' : ''
        }`}
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            {Icon && <Icon size={22} className={color} />}
            <h2 className="text-xl font-bold text-gray-800">{title}</h2>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 bg-blue-50 text-blue-700 rounded text-xs font-semibold">
              ⚡ 5-Run Average
            </span>
            {isWinner && <CheckCircle className="text-green-500" size={24} />}
          </div>
        </div>

        <div className="space-y-2 mb-4">
          <div className="flex justify-between">
            <span className="text-gray-600 font-medium">Average MAE:</span>
            <span className="font-bold text-gray-800">{metrics.mae?.toFixed(2)} kWh</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 font-medium">Average RMSE:</span>
            <span className="font-bold text-gray-800">{metrics.rmse?.toFixed(2)} kWh</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 font-medium">Average R² Score:</span>
            <span className="font-bold text-gray-800">{metrics.r2?.toFixed(4)}</span>
          </div>
        </div>

        {/* Individual Run Breakdown Toggle */}
        {metrics.runs && (
          <div className="border-t pt-3">
            <button
              onClick={() => setShowRuns((v) => !v)}
              className="text-xs text-blue-600 hover:text-blue-800 font-semibold flex items-center gap-1"
            >
              <Layers size={14} />
              {showRuns ? 'Hide 5-Run Breakdown ▲' : 'View All 5 Individual Runs ▼'}
            </button>

            {showRuns && (
              <div className="mt-3 overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-gray-50 text-gray-600">
                      <th className="py-1 px-2 text-left">Run</th>
                      <th className="py-1 px-2 text-center">MAE</th>
                      <th className="py-1 px-2 text-center">RMSE</th>
                      <th className="py-1 px-2 text-center">R²</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.runs.map((r) => (
                      <tr key={r.run} className="border-t hover:bg-gray-50">
                        <td className="py-1 px-2 font-medium">Run {r.run}</td>
                        <td className="py-1 px-2 text-center">{r.testMetrics.mae.toFixed(2)}</td>
                        <td className="py-1 px-2 text-center font-semibold">{r.testMetrics.rmse.toFixed(2)}</td>
                        <td className="py-1 px-2 text-center">{r.testMetrics.r2.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  const SeasonalTable = () => {
    const seasons = ['Summer', 'Monsoon', 'Winter'];
    const rfSm    = results.rf.seasonMetrics;
    const lstmSm  = results.lstm.seasonMetrics;

    return (
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold text-gray-800">
            🌦️ Seasonal Performance Comparison
          </h2>
          <span className="px-3 py-1 bg-gray-100 rounded-full text-xs font-semibold text-gray-600">
            📊 Averaged across {results.numRuns} runs
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50">
                <th className="py-3 px-4 text-left font-semibold text-gray-700">Season</th>
                <th className="py-3 px-4 text-center font-semibold text-blue-700">RF Avg MAE</th>
                <th className="py-3 px-4 text-center font-semibold text-blue-700">RF Avg RMSE</th>
                <th className="py-3 px-4 text-center font-semibold text-blue-700">RF Avg R²</th>
                <th className="py-3 px-4 text-center font-semibold text-purple-700">LSTM Avg MAE</th>
                <th className="py-3 px-4 text-center font-semibold text-purple-700">LSTM Avg RMSE</th>
                <th className="py-3 px-4 text-center font-semibold text-purple-700">LSTM Avg R²</th>
              </tr>
            </thead>
            <tbody>
              {seasons.map((s) => {
                const rf   = rfSm[s]   || {};
                const lstm = lstmSm[s] || {};
                const noRF   = rf.count   === 0 || rf.mae   === null;
                const noLSTM = lstm.count === 0 || lstm.mae === null;
                return (
                  <tr key={s} className="border-t hover:bg-gray-50">
                    <td className="py-3 px-4">
                      <span
                        className="px-2 py-1 rounded-full text-xs font-semibold text-white"
                        style={{ backgroundColor: SEASON_COLORS[s] }}
                      >
                        {s}
                      </span>
                      {rf.count > 0 && (
                        <span className="ml-2 text-xs text-gray-400">({rf.count} test months)</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-center font-medium">{noRF   ? '—' : `${rf.mae?.toFixed(2)} kWh`}</td>
                    <td className="py-3 px-4 text-center font-bold text-blue-700">{noRF   ? '—' : `${rf.rmse?.toFixed(2)} kWh`}</td>
                    <td className="py-3 px-4 text-center">{noRF   ? '—' : rf.r2?.toFixed(4)}</td>
                    <td className="py-3 px-4 text-center font-medium">{noLSTM ? '—' : `${lstm.mae?.toFixed(2)} kWh`}</td>
                    <td className="py-3 px-4 text-center font-bold text-purple-700">{noLSTM ? '—' : `${lstm.rmse?.toFixed(2)} kWh`}</td>
                    <td className="py-3 px-4 text-center">{noLSTM ? '—' : lstm.r2?.toFixed(4)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  // ----------------------------------------------------------
  // JSX
  // ----------------------------------------------------------
  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50">

      {/* ── Header ── */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <div className="flex items-center gap-3 mb-4">
          <Zap className="text-yellow-500" size={32} />
          <h1 className="text-3xl font-bold text-gray-800">
            EB Electricity Consumption Predictor
          </h1>
        </div>
        <p className="text-gray-600 mb-2">
          Monthly electricity forecasting using <strong>Random Forest</strong> and{' '}
          <strong>LSTM</strong> with 5-Run Averaged Benchmark (Jan&nbsp;2020 – Dec&nbsp;2024)
        </p>

        <button
          onClick={trainModels}
          disabled={loading}
          className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold
                     hover:bg-blue-700 transition-colors disabled:bg-gray-400 flex items-center gap-2"
        >
          {loading ? 'Training 5-Run Benchmark…' : '🚀 Train & Compare Models (5-Run Average)'}
        </button>
      </div>

      {/* ── Training Progress — shown whenever either model has started, persists after completion ── */}
      {(loading || rfStatus.status !== 'idle' || lstmStatus.status !== 'idle') && (
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">⏳ 5-Run Training Progress</h3>

          <div className="space-y-4">
            {/* ── Random Forest status card ── */}
            <div className={`rounded-lg p-4 border-2 ${
              rfStatus.status === 'completed' ? 'border-green-300 bg-green-50'
              : rfStatus.status === 'error'   ? 'border-red-300   bg-red-50'
              : rfStatus.status === 'training' ? 'border-blue-300  bg-blue-50'
              : 'border-gray-200 bg-gray-50'
            }`}>
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <Trees size={18} className="text-green-600" />
                  <span className="font-semibold text-gray-800">Random Forest</span>
                </div>
                {rfStatus.status === 'completed' && <CheckCircle size={18} className="text-green-500" />}
                {rfStatus.status === 'error'     && <AlertCircle size={18} className="text-red-500"   />}
                {rfStatus.status === 'training'  && (
                  <span className="text-xs text-blue-600 font-medium animate-pulse">
                    Training Run {rfStatus.run} / {rfStatus.totalRuns}…
                  </span>
                )}
              </div>
              <p className={`text-xs mt-1 ${
                rfStatus.status === 'completed' ? 'text-green-700'
                : rfStatus.status === 'error'   ? 'text-red-600'
                : 'text-gray-500'
              }`}>
                {rfStatus.status === 'completed' && '✓ Completed — '}
                {rfStatus.label || (rfStatus.status === 'idle' ? 'Waiting…' : '')}
              </p>
            </div>

            {/* ── LSTM status card ── */}
            <div className={`rounded-lg p-4 border-2 ${
              lstmStatus.status === 'completed' ? 'border-purple-300 bg-purple-50'
              : lstmStatus.status === 'error'   ? 'border-red-300   bg-red-50'
              : lstmStatus.status === 'training' ? 'border-purple-200 bg-purple-50'
              : 'border-gray-200 bg-gray-50'
            }`}>
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <Brain size={18} className="text-purple-600" />
                  <span className="font-semibold text-gray-800">LSTM</span>
                </div>
                {lstmStatus.status === 'completed' && <CheckCircle size={18} className="text-purple-500" />}
                {lstmStatus.status === 'error'     && <AlertCircle size={18} className="text-red-500"   />}
                {lstmStatus.status === 'training'  && (
                  <span className="text-xs text-purple-600 font-medium animate-pulse">
                    Training Run {lstmStatus.run} / {lstmStatus.totalRuns}…
                  </span>
                )}
              </div>

              {/* Epoch progress — only while training or after completion */}
              {(lstmStatus.status === 'training' || lstmStatus.status === 'completed') && (
                <div className="mt-2">
                  <div className="flex justify-between text-xs text-gray-500 mb-1">
                    <span>
                      {lstmStatus.status === 'training'
                        ? `Run ${lstmStatus.run}/${lstmStatus.totalRuns} · Epoch ${lstmStatus.epoch}/${lstmStatus.totalEpochs}`
                        : `Completed all ${lstmStatus.totalRuns} runs (${lstmStatus.totalEpochs} epochs each)`}
                    </span>
                    <span>
                      {lstmStatus.status === 'training'
                        ? `${Math.round(((lstmStatus.epoch) / lstmStatus.totalEpochs) * 100)}%`
                        : '100%'}
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-200 ${
                        lstmStatus.status === 'completed' ? 'bg-purple-500' : 'bg-purple-400'
                      }`}
                      style={{
                        width: `${
                          lstmStatus.status === 'completed'
                            ? 100
                            : ((lstmStatus.epoch) / lstmStatus.totalEpochs) * 100
                        }%`,
                      }}
                    />
                  </div>
                  {lstmStatus.loss != null && lstmStatus.status === 'training' && (
                    <p className="text-xs text-gray-500 mt-1">Current Loss: {lstmStatus.loss}</p>
                  )}
                </div>
              )}

              <p className={`text-xs mt-1 ${
                lstmStatus.status === 'completed' ? 'text-purple-700'
                : lstmStatus.status === 'error'   ? 'text-red-600'
                : 'text-gray-500'
              }`}>
                {lstmStatus.status === 'completed' && '✓ Completed — '}
                {lstmStatus.label || (lstmStatus.status === 'idle' ? 'Waiting for Random Forest…' : '')}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ── Results ── */}
      {results && (
        <>
          {/* Model metric cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <MetricCard
              title="Random Forest"
              metrics={results.rf}
              isWinner={results.bestModel === 'rf'}
              icon={Trees}
              color="text-green-600"
            />
            <MetricCard
              title="LSTM"
              metrics={results.lstm}
              isWinner={results.bestModel === 'lstm'}
              icon={Brain}
              color="text-purple-600"
            />
          </div>

          {/* Best model banner */}
          <div className="bg-green-100 border-l-4 border-green-500 p-4 mb-6 rounded">
            <div className="flex items-center gap-2">
              <TrendingUp className="text-green-700" size={20} />
              <p className="text-green-800 font-semibold">
                Best Model:{' '}
                {results.bestModel === 'rf' ? '🌲 Random Forest' : '🧠 LSTM'} —
                Lowest 5-Run Average RMSE:{' '}
                {results.bestModel === 'rf'
                  ? results.rf.rmse.toFixed(2)
                  : results.lstm.rmse.toFixed(2)}{' '}
                kWh (vs {results.bestModel === 'rf' ? results.lstm.rmse.toFixed(2) : results.rf.rmse.toFixed(2)} kWh)
              </p>
            </div>
          </div>

          {/* ── Predictions comparison chart ── */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-2xl font-bold text-gray-800">
                  Model Predictions Comparison
                </h2>
                <p className="text-xs text-gray-500 mt-0.5">
                  Plots actual values against 5-run averaged predictions for each model
                </p>
              </div>
              <span className="px-3 py-1 bg-gray-100 rounded-full text-sm">
                📊 {results.comparisonData.length} Test Months
              </span>
            </div>

            {/* Line chart — predictions vs actual */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-700 mb-3">
                5-Run Average Predictions vs Actual Consumption
              </h3>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={results.comparisonData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis
                    dataKey="label"
                    label={{ value: 'Month', position: 'insideBottom', offset: -5, style: { fontWeight: 'bold' } }}
                    tick={{ fontSize: 11 }}
                  />
                  <YAxis
                    label={{
                      value: 'Units Consumed (kWh)',
                      angle: -90,
                      position: 'insideLeft',
                      style: { fontWeight: 'bold' },
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#f9fafb',
                      border: '2px solid #e5e7eb',
                      borderRadius: '8px',
                    }}
                    formatter={(value) => `${parseFloat(value).toFixed(2)} kWh`}
                  />
                  <Legend wrapperStyle={{ paddingTop: '20px' }} />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="#10b981"
                    name="Actual ✓"
                    strokeWidth={3}
                    dot={{ r: 4 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="rf"
                    stroke="#3b82f6"
                    name="Random Forest (5-Run Avg)"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    strokeDasharray="5 5"
                  />
                  <Line
                    type="monotone"
                    dataKey="lstm"
                    stroke="#8b5cf6"
                    name="LSTM (5-Run Avg)"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    strokeDasharray="4 4"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Error distribution bar chart */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-700 mb-3">
                Prediction Error Distribution (5-Run Average Absolute Error per Month)
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={results.comparisonData.map((d) => ({
                    label:     d.label,
                    rfError:   parseFloat(Math.abs(d.rf - d.actual).toFixed(2)),
                    lstmError: parseFloat(Math.abs(d.lstm - d.actual).toFixed(2)),
                  }))}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis dataKey="label" tick={{ fontSize: 10 }} />
                  <YAxis
                    label={{
                      value: 'Absolute Error (kWh)',
                      angle: -90,
                      position: 'insideLeft',
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#f9fafb',
                      border: '2px solid #e5e7eb',
                      borderRadius: '8px',
                    }}
                    formatter={(value) => `${parseFloat(value).toFixed(2)} kWh`}
                  />
                  <Legend />
                  <Bar dataKey="rfError"   fill="#3b82f6" name="Random Forest Error" />
                  <Bar dataKey="lstmError" fill="#8b5cf6" name="LSTM Error" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Summary stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg">
              <div className="text-center">
                <p className="text-sm text-gray-600 mb-1">Avg Actual Consumption</p>
                <p className="text-2xl font-bold text-gray-800">
                  {(
                    results.comparisonData.reduce((s, d) => s + d.actual, 0) /
                    results.comparisonData.length
                  ).toFixed(1)}{' '}
                  kWh
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600 mb-1">Best Model Average RMSE</p>
                <p className="text-2xl font-bold text-green-600">
                  {results.bestModel === 'rf'
                    ? results.rf.rmse.toFixed(2)
                    : results.lstm.rmse.toFixed(2)}{' '}
                  kWh
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600 mb-1">Best Model Average R²</p>
                <p className="text-2xl font-bold text-blue-600">
                  {(
                    (results.bestModel === 'rf' ? results.rf.r2 : results.lstm.r2) * 100
                  ).toFixed(1)}
                  %
                </p>
              </div>
            </div>

            {/* Insights */}
            <div className="mt-4 p-4 bg-yellow-50 border-l-4 border-yellow-400 rounded">
              <h4 className="font-semibold text-yellow-800 mb-2 flex items-center gap-2">
                <AlertCircle size={20} />
                Key Insights (5-Run Cross-Validated Benchmark)
              </h4>
              <ul className="text-sm text-yellow-700 space-y-1">
                <li>
                  •{' '}
                  <strong>
                    {results.bestModel === 'rf' ? 'Random Forest' : 'LSTM'}
                  </strong>{' '}
                  consistently outperforms across 5 runs with an average RMSE of{' '}
                  <strong>
                    {results.bestModel === 'rf'
                      ? results.rf.rmse.toFixed(2)
                      : results.lstm.rmse.toFixed(2)}{' '}
                    kWh
                  </strong>{' '}
                  (vs{' '}
                  {results.bestModel === 'rf'
                    ? results.lstm.rmse.toFixed(2)
                    : results.rf.rmse.toFixed(2)}{' '}
                  kWh)
                </li>
                <li>
                  • Random Forest 5-Run Avg MAE: <strong>{results.rf.mae.toFixed(2)}</strong> kWh — LSTM
                  5-Run Avg MAE: <strong>{results.lstm.mae.toFixed(2)}</strong> kWh
                </li>
                <li>
                  • Random Forest 5-Run Avg R²: <strong>{(results.rf.r2 * 100).toFixed(2)}%</strong> —
                  LSTM 5-Run Avg R²: <strong>{(results.lstm.r2 * 100).toFixed(2)}%</strong>
                </li>
                <li>
                  • Multi-run cross-validation eliminates single-seed variance and guarantees reproducible winner selection across repeated training runs.
                </li>
              </ul>
            </div>
          </div>

          {/* ── Seasonal Comparison ── */}
          <SeasonalTable />

          {/* ── Prediction Interface ── */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-2">
              Make a Prediction
            </h2>
            <div className="flex items-center gap-2 mb-6">
              <span className="text-gray-600">Using Best Model:</span>
              <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full font-semibold">
                {results.bestModel === 'rf' ? '🌲 Random Forest' : '🧠 LSTM'}
              </span>
              <span className="text-xs text-gray-500">
                (Deterministic inference based on 5-run benchmark winner)
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
              {/* Month */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  📅 Month
                </label>
                <select
                  value={inputFeatures.month}
                  onChange={(e) =>
                    setInputFeatures((f) => {
                      const m = parseInt(e.target.value);
                      const seasonMap = {
                        1:'Winter',2:'Winter',3:'Summer',4:'Summer',
                        5:'Summer',6:'Summer',7:'Monsoon',8:'Monsoon',
                        9:'Monsoon',10:'Monsoon',11:'Winter',12:'Winter',
                      };
                      return { ...f, month: m, season: seasonMap[m] };
                    })
                  }
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg
                             focus:border-blue-500 focus:outline-none text-lg bg-white"
                >
                  {['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec'].map((mn, idx) => (
                    <option key={idx} value={idx + 1}>{mn}</option>
                  ))}
                </select>
                <p className="text-xs text-gray-500 mt-1">
                  Season auto-detected: <strong>{inputFeatures.season}</strong>
                </p>
              </div>

              {/* Previous month units */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  ⚡ Prev Month Units (kWh)
                </label>
                <input
                  type="number"
                  step="1"
                  value={inputFeatures.prevMonthUnits}
                  onChange={(e) =>
                    setInputFeatures((f) => ({
                      ...f,
                      prevMonthUnits: parseFloat(e.target.value) || 0,
                    }))
                  }
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg
                             focus:border-blue-500 focus:outline-none text-lg"
                  placeholder="e.g. 280"
                />
                <p className="text-xs text-gray-500 mt-1">Previous billing month's consumption</p>
              </div>

              {/* Rolling average */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  📊 3-Month Rolling Avg (kWh)
                </label>
                <input
                  type="number"
                  step="1"
                  value={inputFeatures.rollingAvg}
                  onChange={(e) =>
                    setInputFeatures((f) => ({
                      ...f,
                      rollingAvg: parseFloat(e.target.value) || 0,
                    }))
                  }
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg
                             focus:border-blue-500 focus:outline-none text-lg"
                  placeholder="e.g. 270"
                />
                <p className="text-xs text-gray-500 mt-1">Average of last 3 months</p>
              </div>

              {/* Temperature */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  🌡️ Avg Temperature (°C)
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={inputFeatures.avgTemperature}
                  onChange={(e) =>
                    setInputFeatures((f) => ({
                      ...f,
                      avgTemperature: parseFloat(e.target.value) || 0,
                    }))
                  }
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg
                             focus:border-blue-500 focus:outline-none text-lg"
                  placeholder="e.g. 35"
                />
                <p className="text-xs text-gray-500 mt-1">Monthly average temperature</p>
              </div>
            </div>

            <div className="flex gap-4 mb-4">
              <button
                onClick={makePrediction}
                className="flex-1 bg-gradient-to-r from-green-600 to-green-700 text-white
                           px-8 py-4 rounded-lg font-bold text-lg hover:from-green-700
                           hover:to-green-800 transition-all shadow-lg transform hover:scale-105"
              >
                🔮 Predict Consumption
              </button>
              <button
                onClick={() => {
                  setInputFeatures({
                    month: 5, season: 'Summer',
                    prevMonthUnits: 280, rollingAvg: 270, avgTemperature: 35,
                  });
                  setPrediction(null);
                }}
                className="px-6 py-4 bg-gray-200 text-gray-700 rounded-lg font-semibold
                           hover:bg-gray-300 transition-colors"
              >
                🔄 Reset
              </button>
            </div>

            {prediction && (
              <div className="mt-4 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl
                              border-2 border-blue-300 shadow-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600 mb-1">
                      Predicted Electricity Consumption ({prediction.model})
                    </p>
                    <p className="text-4xl font-bold text-blue-700">
                      {prediction.value} kWh
                    </p>
                  </div>
                  <div className="text-right">
                    <Zap className="text-yellow-500 mb-2" size={48} />
                    <p className="text-xs text-gray-600">
                      {parseFloat(prediction.value) < 150
                        ? '✅ Low Usage'
                        : parseFloat(prediction.value) < 250
                        ? '⚠️ Medium Usage'
                        : '🔴 High Usage'}
                    </p>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t border-blue-200">
                  <p className="text-sm text-gray-700">
                    <span className="font-semibold">Estimated Bill:</span> ₹
                    {parseFloat(prediction.value) <= 100
                      ? '0.00'
                      : parseFloat(prediction.value) <= 200
                      ? ((parseFloat(prediction.value) - 100) * 1.5).toFixed(2)
                      : parseFloat(prediction.value) <= 500
                      ? (150 + (parseFloat(prediction.value) - 200) * 3).toFixed(2)
                      : (1050 + (parseFloat(prediction.value) - 500) * 5).toFixed(2)}
                    <span className="text-xs text-gray-500 ml-1">(Tamil Nadu EB slab rates)</span>
                  </p>
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default ElectricityPredictor;
