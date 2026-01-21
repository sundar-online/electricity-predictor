import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Zap, TrendingUp, AlertCircle, CheckCircle } from 'lucide-react';

const ElectricityPredictor = () => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [inputFeatures, setInputFeatures] = useState({
    temperature: 25,
    humidity: 60,
    dayOfWeek: 1,
    hour: 12,
    occupancy: 50
  });

  // Generate synthetic training data
  const generateData = (samples = 200) => {
    const data = [];
    for (let i = 0; i < samples; i++) {
      const temp = 15 + Math.random() * 20;
      const humidity = 40 + Math.random() * 40;
      const day = Math.floor(Math.random() * 7);
      const hour = Math.floor(Math.random() * 24);
      const occupancy = Math.random() * 100;
      
      // Simulate electricity consumption with realistic patterns
      let consumption = 50;
      consumption += temp * 2; // Temperature effect
      consumption += humidity * 0.5; // Humidity effect
      consumption += (day >= 5) ? -20 : 10; // Weekend vs weekday
      consumption += (hour >= 9 && hour <= 18) ? 40 : -10; // Business hours
      consumption += occupancy * 0.8; // Occupancy effect
      consumption += (Math.random() - 0.5) * 20; // Random noise
      
      data.push({
        temperature: parseFloat(temp.toFixed(2)),
        humidity: parseFloat(humidity.toFixed(2)),
        dayOfWeek: day,
        hour: hour,
        occupancy: parseFloat(occupancy.toFixed(2)),
        consumption: parseFloat(Math.max(20, consumption).toFixed(2))
      });
    }
    return data;
  };

  // Decision Tree Implementation (simplified CART algorithm)
  class DecisionTree {
    constructor(maxDepth = 5, minSamples = 5) {
      this.maxDepth = maxDepth;
      this.minSamples = minSamples;
      this.tree = null;
    }

    fit(X, y) {
      this.tree = this.buildTree(X, y, 0);
    }

    buildTree(X, y, depth) {
      if (depth >= this.maxDepth || X.length <= this.minSamples) {
        return { value: this.mean(y) };
      }

      const split = this.findBestSplit(X, y);
      if (!split) {
        return { value: this.mean(y) };
      }

      const { feature, threshold, leftIdx, rightIdx } = split;
      
      const leftX = leftIdx.map(i => X[i]);
      const leftY = leftIdx.map(i => y[i]);
      const rightX = rightIdx.map(i => X[i]);
      const rightY = rightIdx.map(i => y[i]);

      return {
        feature,
        threshold,
        left: this.buildTree(leftX, leftY, depth + 1),
        right: this.buildTree(rightX, rightY, depth + 1)
      };
    }

    findBestSplit(X, y) {
      let bestMse = Infinity;
      let bestSplit = null;
      const features = Object.keys(X[0]);

      for (const feature of features) {
        const values = X.map(x => x[feature]);
        const uniqueValues = [...new Set(values)].sort((a, b) => a - b);

        for (let i = 0; i < uniqueValues.length - 1; i++) {
          const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
          const leftIdx = [];
          const rightIdx = [];

          X.forEach((x, idx) => {
            if (x[feature] <= threshold) {
              leftIdx.push(idx);
            } else {
              rightIdx.push(idx);
            }
          });

          if (leftIdx.length === 0 || rightIdx.length === 0) continue;

          const leftY = leftIdx.map(i => y[i]);
          const rightY = rightIdx.map(i => y[i]);
          const mse = this.calculateMse(leftY, rightY);

          if (mse < bestMse) {
            bestMse = mse;
            bestSplit = { feature, threshold, leftIdx, rightIdx };
          }
        }
      }

      return bestSplit;
    }

    calculateMse(leftY, rightY) {
      const leftMean = this.mean(leftY);
      const rightMean = this.mean(rightY);
      
      const leftMse = leftY.reduce((sum, y) => sum + Math.pow(y - leftMean, 2), 0);
      const rightMse = rightY.reduce((sum, y) => sum + Math.pow(y - rightMean, 2), 0);
      
      return (leftMse + rightMse) / (leftY.length + rightY.length);
    }

    predict(X) {
      return X.map(x => this.predictSingle(x, this.tree));
    }

    predictSingle(x, node) {
      if (node.value !== undefined) {
        return node.value;
      }

      if (x[node.feature] <= node.threshold) {
        return this.predictSingle(x, node.left);
      } else {
        return this.predictSingle(x, node.right);
      }
    }

    mean(arr) {
      return arr.reduce((a, b) => a + b, 0) / arr.length;
    }
  }

  // Linear Regression Implementation
  class LinearRegression {
    constructor() {
      this.weights = null;
      this.bias = 0;
    }

    // Normalize features
    normalize(X) {
      const features = Object.keys(X[0]);
      const stats = {};
      
      features.forEach(feature => {
        const values = X.map(x => x[feature]);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const std = Math.sqrt(values.reduce((sum, val) => 
          sum + Math.pow(val - mean, 2), 0) / values.length);
        stats[feature] = { mean, std: std === 0 ? 1 : std };
      });
      
      this.stats = stats;
      
      return X.map(x => {
        const normalized = {};
        features.forEach(feature => {
          normalized[feature] = (x[feature] - stats[feature].mean) / stats[feature].std;
        });
        return normalized;
      });
    }

    fit(X, y, learningRate = 0.001, iterations = 1000) {
      const n = X.length;
      
      // Normalize features
      const X_norm = this.normalize(X);
      const features = Object.keys(X_norm[0]);
      
      // Initialize weights
      this.weights = {};
      features.forEach(f => this.weights[f] = 0);
      this.bias = 0;

      for (let iter = 0; iter < iterations; iter++) {
        const predictions = X_norm.map(x => {
          let pred = this.bias;
          features.forEach(feature => {
            pred += this.weights[feature] * x[feature];
          });
          return pred;
        });
        
        const errors = y.map((actual, i) => predictions[i] - actual);

        // Update weights with gradient descent
        features.forEach(feature => {
          const gradient = errors.reduce((sum, error, i) => 
            sum + error * X_norm[i][feature], 0) / n;
          this.weights[feature] -= learningRate * gradient;
        });

        // Update bias
        const biasGradient = errors.reduce((a, b) => a + b, 0) / n;
        this.bias -= learningRate * biasGradient;
      }
    }

    predict(X) {
      if (!this.stats) return X.map(() => 0);
      
      const features = Object.keys(X[0]);
      
      // Normalize input
      const X_norm = X.map(x => {
        const normalized = {};
        features.forEach(feature => {
          normalized[feature] = (x[feature] - this.stats[feature].mean) / this.stats[feature].std;
        });
        return normalized;
      });
      
      return X_norm.map(x => {
        let prediction = this.bias;
        features.forEach(feature => {
          prediction += this.weights[feature] * x[feature];
        });
        return prediction;
      });
    }
  }

  // Evaluation metrics
  const calculateMetrics = (actual, predicted) => {
    const n = actual.length;
    const mse = actual.reduce((sum, val, i) => 
      sum + Math.pow(val - predicted[i], 2), 0) / n;
    const rmse = Math.sqrt(mse);
    
    const mean = actual.reduce((a, b) => a + b, 0) / n;
    const totalSS = actual.reduce((sum, val) => 
      sum + Math.pow(val - mean, 2), 0);
    const residualSS = actual.reduce((sum, val, i) => 
      sum + Math.pow(val - predicted[i], 2), 0);
    const r2 = 1 - (residualSS / totalSS);
    
    const mae = actual.reduce((sum, val, i) => 
      sum + Math.abs(val - predicted[i]), 0) / n;

    return { mse, rmse, r2, mae };
  };

  const trainModels = () => {
    setLoading(true);
    
    setTimeout(() => {
      // Generate data
      const allData = generateData(200);
      const trainSize = Math.floor(allData.length * 0.8);
      const trainData = allData.slice(0, trainSize);
      const testData = allData.slice(trainSize);

      const trainX = trainData.map(({ consumption, ...features }) => features);
      const trainY = trainData.map(d => d.consumption);
      const testX = testData.map(({ consumption, ...features }) => features);
      const testY = testData.map(d => d.consumption);

      // Train Decision Tree
      const dt = new DecisionTree(5, 5);
      dt.fit(trainX, trainY);
      const dtPred = dt.predict(testX);
      const dtMetrics = calculateMetrics(testY, dtPred);

      // Train Linear Regression
      const lr = new LinearRegression();
      lr.fit(trainX, trainY);
      const lrPred = lr.predict(testX);
      const lrMetrics = calculateMetrics(testY, lrPred);

      // Determine best model
      const bestModel = dtMetrics.rmse < lrMetrics.rmse ? 'dt' : 'lr';
      const bestModelObj = bestModel === 'dt' ? dt : lr;

      // Create comparison data
      const comparisonData = testY.map((actual, i) => ({
        actual,
        dt: dtPred[i],
        lr: lrPred[i],
        index: i
      }));

      setResults({
        dt: { ...dtMetrics, model: dt },
        lr: { ...lrMetrics, model: lr },
        bestModel,
        bestModelObj,
        comparisonData
      });
      setLoading(false);
    }, 500);
  };

  const makePrediction = () => {
    if (!results) return;
    
    const input = [inputFeatures];
    const pred = results.bestModelObj.predict(input);
    setPrediction(pred[0].toFixed(2));
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50">
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <div className="flex items-center gap-3 mb-4">
          <Zap className="text-yellow-500" size={32} />
          <h1 className="text-3xl font-bold text-gray-800">
            Building Electricity Consumption Predictor
          </h1>
        </div>
        <p className="text-gray-600 mb-4">
          Compare Decision Tree vs Linear Regression models and predict electricity consumption
        </p>
        
        <button
          onClick={trainModels}
          disabled={loading}
          className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors disabled:bg-gray-400"
        >
          {loading ? 'Training Models...' : 'Train & Compare Models'}
        </button>
      </div>

      {results && (
        <>
          {/* Model Comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div className={`bg-white rounded-lg shadow-lg p-6 ${results.bestModel === 'dt' ? 'ring-4 ring-green-400' : ''}`}>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-gray-800">Decision Tree</h2>
                {results.bestModel === 'dt' && <CheckCircle className="text-green-500" size={24} />}
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">RMSE:</span>
                  <span className="font-semibold">{results.dt.rmse.toFixed(2)} kWh</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">MAE:</span>
                  <span className="font-semibold">{results.dt.mae.toFixed(2)} kWh</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">R² Score:</span>
                  <span className="font-semibold">{results.dt.r2.toFixed(4)}</span>
                </div>
              </div>
            </div>

            <div className={`bg-white rounded-lg shadow-lg p-6 ${results.bestModel === 'lr' ? 'ring-4 ring-green-400' : ''}`}>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-gray-800">Linear Regression</h2>
                {results.bestModel === 'lr' && <CheckCircle className="text-green-500" size={24} />}
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">RMSE:</span>
                  <span className="font-semibold">{results.lr.rmse.toFixed(2)} kWh</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">MAE:</span>
                  <span className="font-semibold">{results.lr.mae.toFixed(2)} kWh</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">R² Score:</span>
                  <span className="font-semibold">{results.lr.r2.toFixed(4)}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-green-100 border-l-4 border-green-500 p-4 mb-6 rounded">
            <div className="flex items-center gap-2">
              <TrendingUp className="text-green-700" size={20} />
              <p className="text-green-800 font-semibold">
                Best Model: {results.bestModel === 'dt' ? 'Decision Tree' : 'Linear Regression'}
              </p>
            </div>
          </div>

          {/* Predictions Comparison Chart */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-800">Model Predictions Comparison</h2>
              <div className="flex gap-2">
                <span className="px-3 py-1 bg-gray-100 rounded-full text-sm">
                  📊 {results.comparisonData.length} Test Samples
                </span>
              </div>
            </div>

            {/* Line Chart - Predictions vs Actual */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-700 mb-3">Predictions vs Actual Values</h3>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={results.comparisonData.slice(0, 30)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis 
                    dataKey="index" 
                    label={{ value: 'Test Sample Number', position: 'insideBottom', offset: -5, style: { fontWeight: 'bold' } }} 
                  />
                  <YAxis 
                    label={{ value: 'Electricity Consumption (kWh)', angle: -90, position: 'insideLeft', style: { fontWeight: 'bold' } }} 
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#f9fafb', border: '2px solid #e5e7eb', borderRadius: '8px' }}
                    formatter={(value) => `${parseFloat(value).toFixed(2)} kWh`}
                  />
                  <Legend wrapperStyle={{ paddingTop: '20px' }} />
                  <Line type="monotone" dataKey="actual" stroke="#10b981" name="Actual ✓" strokeWidth={3} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="dt" stroke="#3b82f6" name="Decision Tree" strokeWidth={2} dot={{ r: 3 }} strokeDasharray="5 5" />
                  <Line type="monotone" dataKey="lr" stroke="#ef4444" name="Linear Regression" strokeWidth={2} dot={{ r: 3 }} strokeDasharray="5 5" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Error Distribution Bar Chart */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-700 mb-3">Prediction Error Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={results.comparisonData.slice(0, 20).map((d, i) => ({
                  index: i,
                  dtError: Math.abs(d.dt - d.actual),
                  lrError: Math.abs(d.lr - d.actual)
                }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis dataKey="index" label={{ value: 'Sample', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'Absolute Error (kWh)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#f9fafb', border: '2px solid #e5e7eb', borderRadius: '8px' }}
                    formatter={(value) => `${parseFloat(value).toFixed(2)} kWh`}
                  />
                  <Legend />
                  <Bar dataKey="dtError" fill="#3b82f6" name="Decision Tree Error" />
                  <Bar dataKey="lrError" fill="#ef4444" name="Linear Regression Error" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Statistics Summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg">
              <div className="text-center">
                <p className="text-sm text-gray-600 mb-1">Average Actual Consumption</p>
                <p className="text-2xl font-bold text-gray-800">
                  {(results.comparisonData.reduce((sum, d) => sum + d.actual, 0) / results.comparisonData.length).toFixed(2)} kWh
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600 mb-1">Best Model Avg Error</p>
                <p className="text-2xl font-bold text-green-600">
                  {results.bestModel === 'dt' 
                    ? (results.comparisonData.reduce((sum, d) => sum + Math.abs(d.dt - d.actual), 0) / results.comparisonData.length).toFixed(2)
                    : (results.comparisonData.reduce((sum, d) => sum + Math.abs(d.lr - d.actual), 0) / results.comparisonData.length).toFixed(2)
                  } kWh
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600 mb-1">Prediction Accuracy</p>
                <p className="text-2xl font-bold text-blue-600">
                  {(results.bestModel === 'dt' ? results.dt.r2 * 100 : results.lr.r2 * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            {/* Insights */}
            <div className="mt-4 p-4 bg-yellow-50 border-l-4 border-yellow-400 rounded">
              <h4 className="font-semibold text-yellow-800 mb-2 flex items-center gap-2">
                <AlertCircle size={20} />
                Key Insights
              </h4>
              <ul className="text-sm text-yellow-700 space-y-1">
                <li>• <strong>{results.bestModel === 'dt' ? 'Decision Tree' : 'Linear Regression'}</strong> performs better with {results.bestModel === 'dt' ? results.dt.rmse.toFixed(2) : results.lr.rmse.toFixed(2)} kWh RMSE</li>
                <li>• Average prediction error: {(results.bestModel === 'dt' ? results.dt.mae : results.lr.mae).toFixed(2)} kWh</li>
                <li>• The model explains {((results.bestModel === 'dt' ? results.dt.r2 : results.lr.r2) * 100).toFixed(1)}% of variance in electricity consumption</li>
                <li>• {results.comparisonData.filter(d => {
                  const bestPred = results.bestModel === 'dt' ? d.dt : d.lr;
                  return Math.abs(bestPred - d.actual) < 10;
                }).length} out of {results.comparisonData.length} predictions are within ±10 kWh of actual values</li>
              </ul>
            </div>
          </div>

          {/* Prediction Interface */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-2">
              Make Prediction
            </h2>
            <div className="flex items-center gap-2 mb-6">
              <span className="text-gray-600">Using Best Model:</span>
              <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full font-semibold">
                {results.bestModel === 'dt' ? 'Decision Tree' : 'Linear Regression'}
              </span>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  🌡️ Temperature (°C)
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={inputFeatures.temperature}
                  onChange={(e) => setInputFeatures({...inputFeatures, temperature: parseFloat(e.target.value) || 0})}
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none text-lg"
                  placeholder="e.g., 25"
                />
                <p className="text-xs text-gray-500 mt-1">Range: 15-35°C</p>
              </div>
              
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  💧 Humidity (%)
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={inputFeatures.humidity}
                  onChange={(e) => setInputFeatures({...inputFeatures, humidity: parseFloat(e.target.value) || 0})}
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none text-lg"
                  placeholder="e.g., 60"
                />
                <p className="text-xs text-gray-500 mt-1">Range: 40-80%</p>
              </div>
              
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  📅 Day of Week
                </label>
                <select
                  value={inputFeatures.dayOfWeek}
                  onChange={(e) => setInputFeatures({...inputFeatures, dayOfWeek: parseInt(e.target.value)})}
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none text-lg bg-white"
                >
                  <option value="0">Monday</option>
                  <option value="1">Tuesday</option>
                  <option value="2">Wednesday</option>
                  <option value="3">Thursday</option>
                  <option value="4">Friday</option>
                  <option value="5">Saturday</option>
                  <option value="6">Sunday</option>
                </select>
                <p className="text-xs text-gray-500 mt-1">Weekday vs Weekend impact</p>
              </div>
              
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  🕐 Hour of Day
                </label>
                <input
                  type="range"
                  min="0"
                  max="23"
                  value={inputFeatures.hour}
                  onChange={(e) => setInputFeatures({...inputFeatures, hour: parseInt(e.target.value)})}
                  className="w-full"
                />
                <div className="flex justify-between items-center mt-2">
                  <span className="text-lg font-bold text-blue-600">{inputFeatures.hour}:00</span>
                  <span className="text-xs text-gray-500">
                    {inputFeatures.hour >= 9 && inputFeatures.hour <= 18 ? '🏢 Business Hours' : '🌙 Off Hours'}
                  </span>
                </div>
                <p className="text-xs text-gray-500 mt-1">Range: 0-23 (24hr format)</p>
              </div>
              
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  👥 Occupancy (%)
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={inputFeatures.occupancy}
                  onChange={(e) => setInputFeatures({...inputFeatures, occupancy: parseFloat(e.target.value)})}
                  className="w-full"
                />
                <div className="flex justify-between items-center mt-2">
                  <span className="text-lg font-bold text-blue-600">{inputFeatures.occupancy.toFixed(0)}%</span>
                  <span className="text-xs text-gray-500">
                    {inputFeatures.occupancy < 30 ? '🟢 Low' : inputFeatures.occupancy < 70 ? '🟡 Medium' : '🔴 High'}
                  </span>
                </div>
                <p className="text-xs text-gray-500 mt-1">Building occupancy level</p>
              </div>
            </div>
            
            <div className="flex gap-4 mb-4">
              <button
                onClick={makePrediction}
                className="flex-1 bg-gradient-to-r from-green-600 to-green-700 text-white px-8 py-4 rounded-lg font-bold text-lg hover:from-green-700 hover:to-green-800 transition-all shadow-lg transform hover:scale-105"
              >
                🔮 Predict Consumption
              </button>
              <button
                onClick={() => {
                  setInputFeatures({
                    temperature: 25,
                    humidity: 60,
                    dayOfWeek: 1,
                    hour: 12,
                    occupancy: 50
                  });
                  setPrediction(null);
                }}
                className="px-6 py-4 bg-gray-200 text-gray-700 rounded-lg font-semibold hover:bg-gray-300 transition-colors"
              >
                🔄 Reset
              </button>
            </div>
            
            {prediction && (
              <div className="mt-4 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border-2 border-blue-300 shadow-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600 mb-1">Predicted Electricity Consumption</p>
                    <p className="text-4xl font-bold text-blue-700">{prediction} kWh</p>
                  </div>
                  <div className="text-right">
                    <Zap className="text-yellow-500 mb-2" size={48} />
                    <p className="text-xs text-gray-600">
                      {parseFloat(prediction) < 100 ? '✅ Low Usage' : parseFloat(prediction) < 150 ? '⚠️ Medium Usage' : '🔴 High Usage'}
                    </p>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t border-blue-200">
                  <p className="text-sm text-gray-700">
                    <span className="font-semibold">Estimated Cost:</span> ${(parseFloat(prediction) * 0.12).toFixed(2)} 
                    <span className="text-xs text-gray-500 ml-1">(@ $0.12/kWh)</span>
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
