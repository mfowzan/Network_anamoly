import React, { useState } from "react";
import "./App.css";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import { Shield, AlertTriangle, Activity, TrendingUp, Zap, Server } from "lucide-react";

function App() {
  // ✅ CRITICAL: Feature order MUST match backend (utils.py, train_all.py, model.py)
  const [formData, setFormData] = useState({
    duration: "",
    src_bytes: "",
    dst_bytes: "",
    count: "",
    srv_count: "",
    wrong_fragment: "",  // ⚠️ Moved to match backend order
    serror_rate: "",
    srv_serror_rate: "",
    rerror_rate: "",
    srv_rerror_rate: "",
    same_srv_rate: "",
    diff_srv_rate: "",
    dst_host_count: "",
    dst_host_srv_count: "",
    dst_host_same_srv_rate: "",
    dst_host_diff_srv_rate: "",
  });

  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handlePreset = (type) => {
    if (type === "normal") {
      setFormData({
        duration: "0", 
        src_bytes: "181", 
        dst_bytes: "5450", 
        count: "8", 
        srv_count: "8",
        wrong_fragment: "0", 
        serror_rate: "0.0", 
        srv_serror_rate: "0.0", 
        rerror_rate: "0.0",
        srv_rerror_rate: "0.0", 
        same_srv_rate: "1.0", 
        diff_srv_rate: "0.0", 
        dst_host_count: "9",
        dst_host_srv_count: "9", 
        dst_host_same_srv_rate: "1.0", 
        dst_host_diff_srv_rate: "0.0",
      });
    } else if (type === "suspicious") {
      setFormData({
        duration: "0", 
        src_bytes: "105", 
        dst_bytes: "146", 
        count: "50", 
        srv_count: "10",
        wrong_fragment: "0", 
        serror_rate: "0.5", 
        srv_serror_rate: "0.4", 
        rerror_rate: "0.1",
        srv_rerror_rate: "0.1", 
        same_srv_rate: "0.2", 
        diff_srv_rate: "0.4", 
        dst_host_count: "100",
        dst_host_srv_count: "20", 
        dst_host_same_srv_rate: "0.3", 
        dst_host_diff_srv_rate: "0.3",
      });
    } else if (type === "anomalous") {
      setFormData({
        duration: "0", 
        src_bytes: "0", 
        dst_bytes: "0", 
        count: "123", 
        srv_count: "6",
        wrong_fragment: "0", 
        serror_rate: "1.0", 
        srv_serror_rate: "1.0", 
        rerror_rate: "0.0",
        srv_rerror_rate: "0.0", 
        same_srv_rate: "0.05", 
        diff_srv_rate: "0.06", 
        dst_host_count: "255",
        dst_host_srv_count: "6", 
        dst_host_same_srv_rate: "0.04", 
        dst_host_diff_srv_rate: "0.06",
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);

    try {
      // Convert all form values to numbers
      const payload = {};
      for (const key in formData) {
        const value = parseFloat(formData[key]);
        if (isNaN(value)) {
          throw new Error(`Invalid value for ${key}: "${formData[key]}". Please enter a valid number.`);
        }
        payload[key] = value;
      }

      const response = await fetch("http://127.0.0.1:8000/predict_single", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error("Error:", err);
      setError(err.message || "Could not fetch prediction. Please check if the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const renderComparisonChart = (results) => {
    const chartData = Object.keys(results).map((key) => ({
      model: key.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase()),
      score: (results[key].score || 0) * 100,  // ✅ Convert to percentage
    }));

    return (
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 60 }}>
          <defs>
            <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#6366f1" stopOpacity={0.8}/>
              <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.6}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
          <XAxis dataKey="model" angle={-45} textAnchor="end" height={80} tick={{ fill: '#9ca3af', fontSize: 12 }} />
          <YAxis tick={{ fill: '#9ca3af' }} domain={[0, 100]} label={{ value: 'Confidence (%)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} />
          <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px', color: '#f3f4f6' }} />
          <Legend wrapperStyle={{ paddingTop: '20px' }} />
          <Bar dataKey="score" fill="url(#barGradient)" radius={[8, 8, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    );
  };

  const renderRadarChart = (results) => {
    const radarData = Object.keys(results).map((key) => ({
      model: key.replace(/_/g, " ").slice(0, 15),
      score: (results[key].score || 0) * 100,
    }));

    return (
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={radarData}>
          <PolarGrid stroke="#374151" />
          <PolarAngleAxis dataKey="model" tick={{ fill: '#9ca3af', fontSize: 11 }} />
          <PolarRadiusAxis tick={{ fill: '#9ca3af' }} domain={[0, 100]} />
          <Radar name="Confidence (%)" dataKey="score" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} />
        </RadarChart>
      </ResponsiveContainer>
    );
  };

  // ✅ Field groups with correct order matching backend
  const fieldGroups = {
    "Traffic Metrics": ["duration", "src_bytes", "dst_bytes"],
    "Connection Stats": ["count", "srv_count", "wrong_fragment"],
    "Error Rates": ["serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate"],
    "Service Patterns": ["same_srv_rate", "diff_srv_rate"],
    "Host Statistics": ["dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate"],
  };

  return (
    <div className="dashboard-container">
      <div className="background-effects">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
        <div className="blob blob-3"></div>
      </div>

      <div className="content-wrapper">
        <header className="dashboard-header">
          <div className="header-content">
            <Shield className="header-icon" size={48} />
            <h1 className="main-title">Network Anomaly Detection</h1>
          </div>
          <p className="subtitle">AI-Powered Security Analysis Dashboard</p>
        </header>

        <div className="preset-buttons">
          <button onClick={() => handlePreset("normal")} className="preset-btn preset-normal">
            <Shield size={20} />
            Normal Traffic
          </button>
          <button onClick={() => handlePreset("suspicious")} className="preset-btn preset-suspicious">
            <AlertTriangle size={20} />
            Suspicious Traffic
          </button>
          <button onClick={() => handlePreset("anomalous")} className="preset-btn preset-anomalous">
            <Zap size={20} />
            Anomalous Traffic
          </button>
        </div>

        <div className="main-grid">
          <div className="input-section">
            <div className="card">
              <div className="card-header">
                <Activity size={24} className="section-icon" />
                <h2>Traffic Parameters</h2>
              </div>
              
              <div className="form-container">
                {Object.entries(fieldGroups).map(([groupName, fields]) => (
                  <div key={groupName} className="field-group">
                    <h3 className="group-title">{groupName}</h3>
                    {fields.map((field) => (
                      <div key={field} className="input-field">
                        <label>{field.replaceAll("_", " ")}</label>
                        <input
                          type="number"
                          step="any"
                          name={field}
                          value={formData[field]}
                          onChange={handleChange}
                          required
                          placeholder="Enter value"
                        />
                      </div>
                    ))}
                  </div>
                ))}

                <button onClick={handleSubmit} disabled={loading} className="submit-btn">
                  {loading ? (
                    <>
                      <div className="spinner"></div>
                      Analyzing Traffic...
                    </>
                  ) : (
                    <>
                      <Server size={20} />
                      Run Analysis
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          <div className="results-section">
            {loading && (
              <div className="card loading-card">
                <div className="loading-content">
                  <div className="loading-spinner-wrapper">
                    <div className="loading-ring"></div>
                    <Shield size={48} className="loading-icon" />
                  </div>
                  <p className="loading-title">Processing Models</p>
                  <p className="loading-subtitle">Analyzing network traffic patterns...</p>
                </div>
              </div>
            )}

            {result && !loading && (
              <>
                <div className="summary-grid">
                  <div className="summary-card summary-card-purple">
                    <div className="summary-content">
                      <div>
                        <p className="summary-label">Models Flagged</p>
                        <p className="summary-value">{result.summary.anomaly_models}</p>
                      </div>
                      <AlertTriangle size={48} className="summary-icon" />
                    </div>
                  </div>
                  
                  <div className="summary-card summary-card-blue">
                    <div className="summary-content">
                      <div>
                        <p className="summary-label">Confidence Level</p>
                        <p className="summary-value">
                          {result.summary.insight.includes("High confidence") ? "High" : 
                           result.summary.insight.includes("Moderate") ? "Medium" : "Low"}
                        </p>
                      </div>
                      <TrendingUp size={48} className="summary-icon" />
                    </div>
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">
                    <Activity size={24} className="section-icon" />
                    <h3>Analysis Insights</h3>
                  </div>
                  <p className="insight-text">{result.summary.insight}</p>
                </div>

                <div className="charts-grid">
                  <div className="card">
                    <h3 className="chart-title">Model Comparison</h3>
                    {renderComparisonChart(result.results)}
                  </div>
                  
                  <div className="card">
                    <h3 className="chart-title">Confidence Radar</h3>
                    {renderRadarChart(result.results)}
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">
                    <Server size={24} className="section-icon" />
                    <h3>Detailed Model Outputs</h3>
                  </div>
                  <div className="models-grid">
                    {Object.keys(result.results).map((modelName) => {
                      const res = result.results[modelName];
                      // ✅ Fixed: Check for string "anomaly" instead of numeric 0
                      const isAnomaly = res.prediction === "anomaly";
                      const isNormal = res.prediction === "normal";
                      
                      return (
                        <div key={modelName} className={`model-card ${isAnomaly ? 'model-anomaly' : 'model-normal'}`}>
                          <h4 className="model-name">{modelName.replaceAll("_", " ")}</h4>
                          <div className="model-details">
                            <div className="model-row">
                              <span className="model-label">Status:</span>
                              <span className="model-status">
                                {isAnomaly ? <AlertTriangle size={16} /> : <Shield size={16} />}
                                {isAnomaly ? "Anomaly" : isNormal ? "Normal" : "Unknown"}
                              </span>
                            </div>
                            <div className="model-row">
                              <span className="model-label">Confidence:</span>
                              <span className="model-confidence">
                                {res.score !== undefined && res.score !== null ? (res.score * 100).toFixed(1) + "%" : "N/A"}
                              </span>
                            </div>
                            <div className="progress-bar">
                              <div className="progress-fill" style={{ width: `${(res.score || 0) * 100}%` }}></div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </>
            )}

            {error && !loading && (
              <div className="card error-card">
                <AlertTriangle size={24} />
                <p>{error}</p>
              </div>
            )}

            {!result && !loading && !error && (
              <div className="card empty-card">
                <Shield size={64} className="empty-icon" />
                <p>Enter traffic parameters and run analysis to view results</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;