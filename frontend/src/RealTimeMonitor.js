import React, { useState, useEffect } from "react";
import "./App.css";
import { 
  Shield, 
  Activity, 
  AlertTriangle, 
  Play, 
  Square, 
  Signal, 
  Zap, 
  Eye,
  EyeOff,
  BarChart3,
  TrendingUp,
  Clock,
  Server
} from "lucide-react";

export default function RealTimeMonitor() {
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({
    total: 0,
    anomalies: 0,
    suspicious: 0,
    normal: 0
  });
  const [showOnlyAnomalies, setShowOnlyAnomalies] = useState(false);

  // Calculate stats whenever results change
  useEffect(() => {
    const anomalies = results.filter(r => r.ensemble?.prediction === "anomaly").length;
    const suspicious = results.filter(r => r.ensemble?.prediction === "suspicious").length;
    const normal = results.filter(r => r.ensemble?.prediction === "normal").length;
    
    setStats({
      total: results.length,
      anomalies,
      suspicious,
      normal
    });
  }, [results]);

  // Fetch results every 2 seconds while running
  useEffect(() => {
    let interval = null;

    if (running) {
      interval = setInterval(() => {
        fetch("http://127.0.0.1:8000/realtime/latest")
          .then((res) => res.json())
          .then((data) => {
            if (data.results) {
              setResults(prev => [...prev, ...data.results].slice(-50)); // Keep last 50 results
            }
          })
          .catch((err) => setError(err.message));
      }, 1500);
    }

    return () => clearInterval(interval);
  }, [running]);

  // Start monitoring
  const startMonitoring = async () => {
    setError(null);
    try {
      await fetch("http://127.0.0.1:8000/realtime/start", { method: "POST" });
      setRunning(true);
      setResults([]);
    } catch (err) {
      setError("Cannot start: " + err.message);
    }
  };

  // Stop monitoring
  const stopMonitoring = async () => {
    setError(null);
    try {
      await fetch("http://127.0.0.1:8000/realtime/stop", { method: "POST" });
      setRunning(false);
    } catch (err) {
      setError("Cannot stop: " + err.message);
    }
  };

  const getPredictionColor = (prediction) => {
    switch (prediction) {
      case "anomaly": return "#ef4444";
      case "suspicious": return "#f59e0b";
      case "normal": return "#10b981";
      default: return "#6b7280";
    }
  };

  const getPredictionIcon = (prediction) => {
    switch (prediction) {
      case "anomaly": return <Zap size={16} />;
      case "suspicious": return <AlertTriangle size={16} />;
      case "normal": return <Shield size={16} />;
      default: return <Activity size={16} />;
    }
  };

  const filteredResults = showOnlyAnomalies 
    ? results.filter(r => r.ensemble?.prediction === "anomaly" || r.ensemble?.prediction === "suspicious")
    : results;

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
            <Signal className="header-icon" size={48} />
            <h1 className="main-title">Real-Time Network Monitor</h1>
          </div>
          <p className="subtitle">Live AI-powered anomaly detection from network interface</p>
        </header>

        <nav className="top-nav">
          <a href="/" className="nav-link">Single Flow Analysis</a>
          <a href="/pcap" className="nav-link">PCAP Upload</a>
          <a href="/realtime" className="nav-link nav-active">Real-Time Monitor</a>
          <a href="/chatbot" className="nav-btn">
            <Zap size={16} />
            Cyber AI Chatbot
          </a>
          <a href="/explain" className="nav-link">Explain</a>
        </nav>

        {/* Stats Dashboard */}
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-content">
              <div>
                <p className="stat-label">Total Flows</p>
                <p className="stat-value">{stats.total}</p>
              </div>
              <BarChart3 size={48} className="stat-icon stat-icon-purple" />
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-content">
              <div>
                <p className="stat-label">Anomalies</p>
                <p className="stat-value">{stats.anomalies}</p>
              </div>
              <AlertTriangle size={48} className="stat-icon stat-icon-red" />
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-content">
              <div>
                <p className="stat-label">Suspicious</p>
                <p className="stat-value">{stats.suspicious}</p>
              </div>
              <AlertTriangle size={48} className="stat-icon stat-icon-yellow" />
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-content">
              <div>
                <p className="stat-label">Normal</p>
                <p className="stat-value">{stats.normal}</p>
              </div>
              <Shield size={48} className="stat-icon stat-icon-green" />
            </div>
          </div>
        </div>

        {/* Control Card */}
        <div className="card control-card">
          <div className="card-header">
            <Activity size={24} className="section-icon" />
            <h2>Monitor Controls</h2>
          </div>

          <div className="control-grid">
            <div className="control-info">
              <div className="status-indicator">
                <div className={`status-dot ${running ? 'status-dot-active' : 'status-dot-inactive'}`}></div>
                <span className="status-text">
                  {running ? 'Active Monitoring' : 'Monitoring Stopped'}
                </span>
              </div>
              <p className="status-subtext">
                {running 
                  ? 'Live data is being captured and analyzed' 
                  : 'Start monitoring to begin live analysis'}
              </p>
            </div>

            <div className="control-actions">
              {!running ? (
                <button className="control-btn control-btn-start" onClick={startMonitoring}>
                  <Play size={20} />
                  Start Live Monitoring
                </button>
              ) : (
                <button className="control-btn control-btn-stop" onClick={stopMonitoring}>
                  <Square size={20} />
                  Stop Monitoring
                </button>
              )}
            </div>
          </div>

          {error && (
            <div className="error-message">
              <AlertTriangle size={20} />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Live Feed Card */}
        <div className="card">
          <div className="card-header">
            <TrendingUp size={24} className="section-icon" />
            <h2>Live Detection Feed</h2>
            <div className="feed-controls">
              <button 
                className={`filter-btn ${showOnlyAnomalies ? 'filter-btn-active' : ''}`}
                onClick={() => setShowOnlyAnomalies(!showOnlyAnomalies)}
              >
                {showOnlyAnomalies ? <EyeOff size={16} /> : <Eye size={16} />}
                {showOnlyAnomalies ? 'Show All' : 'Show Threats Only'}
              </button>
              <div className="feed-stats">
                <Clock size={16} />
                <span>Showing {filteredResults.length} of {results.length} flows</span>
              </div>
            </div>
          </div>

          {filteredResults.length === 0 ? (
            <div className="empty-feed">
              <Server size={64} className="empty-icon" />
              <p className="empty-text">
                {running 
                  ? 'Waiting for network data...' 
                  : 'Start monitoring to see live results'}
              </p>
            </div>
          ) : (
            <div className="live-feed-container">
              <div className="live-feed-header">
                <div className="feed-column">Flow ID</div>
                <div className="feed-column">Timestamp</div>
                <div className="feed-column">Ensemble Prediction</div>
                <div className="feed-column">Model Consensus</div>
                <div className="feed-column">Confidence</div>
              </div>

              <div className="live-feed-items">
                {[...filteredResults].reverse().map((r, idx) => {
                  const ensemble = r.ensemble || { prediction: "unknown", score: 0 };
                  const predictions = Object.values(r).filter(v => typeof v === 'object' && v.prediction);
                  const consensus = predictions.reduce((acc, curr) => {
                    acc[curr.prediction] = (acc[curr.prediction] || 0) + 1;
                    return acc;
                  }, {});
                  const topPrediction = Object.keys(consensus).reduce((a, b) => consensus[a] > consensus[b] ? a : b);

                  return (
                    <div 
                      key={idx} 
                      className="live-feed-item"
                      style={{
                        borderLeft: `4px solid ${getPredictionColor(ensemble.prediction)}`,
                        background: `linear-gradient(90deg, rgba(${getPredictionColor(ensemble.prediction)}, 0.05) 0%, rgba(15, 23, 42, 0.7) 100%)`
                      }}
                    >
                      <div className="feed-column">
                        <span className="flow-id">#{(filteredResults.length - idx).toString().padStart(3, '0')}</span>
                      </div>
                      <div className="feed-column">
                        <span className="timestamp">Just now</span>
                      </div>
                      <div className="feed-column">
                        <div className="prediction-badge" style={{ 
                          backgroundColor: getPredictionColor(ensemble.prediction) + '20',
                          borderColor: getPredictionColor(ensemble.prediction)
                        }}>
                          {getPredictionIcon(ensemble.prediction)}
                          <span>{ensemble.prediction.toUpperCase()}</span>
                        </div>
                      </div>
                      <div className="feed-column">
                        <div className="consensus">
                          {topPrediction}
                          <span className="consensus-count">({consensus[topPrediction]}/{predictions.length})</span>
                        </div>
                      </div>
                      <div className="feed-column">
                        <div className="confidence-meter">
                          <div 
                            className="confidence-fill" 
                            style={{ 
                              width: `${(ensemble.score || 0) * 100}%`,
                              background: `linear-gradient(90deg, ${getPredictionColor(ensemble.prediction)}, ${getPredictionColor(ensemble.prediction)}80)`
                            }}
                          ></div>
                          <span className="confidence-value">{(ensemble.score * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Model Breakdown Card */}
        {filteredResults.length > 0 && (
          <div className="card">
            <div className="card-header">
              <Server size={24} className="section-icon" />
              <h2>Model Performance Breakdown</h2>
            </div>
            <div className="models-breakdown">
              {Object.keys(filteredResults[0] || {}).filter(k => k !== 'ensemble').map(modelName => {
                const modelResults = filteredResults.map(r => r[modelName]);
                const anomalyCount = modelResults.filter(r => r?.prediction === "anomaly").length;
                const scoreAvg = modelResults.reduce((acc, r) => acc + (r?.score || 0), 0) / modelResults.length;
                
                return (
                  <div key={modelName} className="model-breakdown-card">
                    <h4 className="model-breakdown-name">{modelName.replace(/_/g, ' ')}</h4>
                    <div className="model-breakdown-stats">
                      <div className="breakdown-stat">
                        <span className="breakdown-label">Anomalies Detected</span>
                        <span className="breakdown-value">{anomalyCount}</span>
                      </div>
                      <div className="breakdown-stat">
                        <span className="breakdown-label">Avg Confidence</span>
                        <span className="breakdown-value">{(scoreAvg * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                    <div className="breakdown-progress">
                      <div 
                        className="breakdown-progress-fill"
                        style={{ width: `${(anomalyCount / filteredResults.length) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}