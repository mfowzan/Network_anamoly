import React, { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  PieChart,
  Pie,
  Cell
} from "recharts";
import { 
  Shield, 
  Loader2, 
  AlertTriangle, 
  Brain, 
  TrendingUp,
  BarChart3,
  Zap,
  Info,
  ChevronDown,
  Cpu,
  Server,
  Target,
  HelpCircle,
  BookOpen
} from "lucide-react";

export default function ExplainPage() {
  const [loading, setLoading] = useState(true);
  const [explain, setExplain] = useState(null);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState("ensemble");
  const [activeTab, setActiveTab] = useState("features");

  useEffect(() => {
    fetchExplanation();
  }, []);

  async function fetchExplanation() {
    try {
      const res = await fetch("http://127.0.0.1:8000/explain_last");
      const data = await res.json();

      if (!res.ok) throw new Error(data.error || "Server error");

      setExplain(data.explanation);
      setSelectedModel("ensemble");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const getPredictionColor = (prediction) => {
    switch (prediction?.toLowerCase()) {
      case "anomaly": return "#ef4444";
      case "suspicious": return "#f59e0b";
      case "normal": return "#10b981";
      default: return "#6b7280";
    }
  };

  const getModelType = (modelName) => {
    if (modelName.includes('autoencoder') || modelName.includes('ae')) return "Reconstruction";
    if (modelName.includes('forest') || modelName.includes('rf')) return "Feature Importance";
    if (modelName.includes('isolation') || modelName.includes('if')) return "Isolation Score";
    if (modelName.includes('svm')) return "Boundary Score";
    if (modelName.includes('ensemble')) return "Aggregate";
    return "Statistical";
  };

  const getModelIcon = (modelName) => {
    switch (getModelType(modelName)) {
      case "Reconstruction": return <Brain size={16} />;
      case "Feature Importance": return <BarChart3 size={16} />;
      case "Isolation Score": return <Target size={16} />;
      case "Boundary Score": return <Zap size={16} />;
      case "Aggregate": return <Server size={16} />;
      default: return <Cpu size={16} />;
    }
  };

  const featureData = (obj) =>
    Object.entries(obj)
      .map(([k, v]) => ({
        feature: k,
        value: Math.abs(Number(v)),
        raw: Number(v),
        impact: Number(v) > 0 ? "positive" : "negative"
      }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 8);

  const deviationData = (obj) =>
    Object.entries(obj)
      .map(([k, v]) => ({
        feature: k,
        deviation: Math.abs(Number(v)),
        direction: Number(v) > 0 ? "above" : "below"
      }))
      .sort((a, b) => b.deviation - a.deviation)
      .slice(0, 8);

  const getInterpretation = (model) => {
    if (model?.feature_importance) {
      const topFeature = Object.entries(model.feature_importance)
        .sort(([,a], [,b]) => Math.abs(b) - Math.abs(a))[0];
      return {
        title: "Feature-Driven Decision",
        description: `The model's decision was primarily influenced by "${topFeature[0]}" with an impact of ${Math.abs(topFeature[1]).toFixed(4)}`,
        insight: "Higher feature importance values indicate stronger influence on the final prediction."
      };
    }
    
    if (model?.deviation) {
      const maxDev = Object.entries(model.deviation)
        .sort(([,a], [,b]) => Math.abs(b) - Math.abs(a))[0];
      return {
        title: "Anomaly Pattern Detected",
        description: `Significant deviation found in "${maxDev[0]}" (${Math.abs(maxDev[1]).toFixed(4)} standard deviations)`,
        insight: "Large deviations from normal patterns trigger anomaly alerts."
      };
    }
    
    if (model?.score !== undefined) {
      return {
        title: "Statistical Outlier Detection",
        description: `Outlier score of ${model.score.toFixed(4)} indicates ${model.score > 0.5 ? "strong" : "moderate"} anomaly characteristics`,
        insight: "Higher scores represent greater deviation from normal data patterns."
      };
    }
    
    return {
      title: "Model Decision Analysis",
      description: "Analyzing prediction patterns and contributing factors",
      insight: "Review feature contributions and deviation patterns"
    };
  };

  if (loading) {
    return (
      <div className="dashboard-container">
        <div className="background-effects">
          <div className="blob blob-1"></div>
          <div className="blob blob-2"></div>
          <div className="blob blob-3"></div>
        </div>
        <div className="content-wrapper">
          <div className="loading-card">
            <div className="loading-content">
              <div className="loading-spinner-wrapper">
                <div className="loading-ring"></div>
                <Brain size={48} className="loading-icon" />
              </div>
              <p className="loading-title">Generating Explanations</p>
              <p className="loading-subtitle">Analyzing model decision patterns...</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-container">
        <div className="background-effects">
          <div className="blob blob-1"></div>
          <div className="blob blob-2"></div>
          <div className="blob blob-3"></div>
        </div>
        <div className="content-wrapper">
          <div className="card error-card">
            <AlertTriangle size={32} />
            <div>
              <h3>Explanation Error</h3>
              <p>{error}</p>
              <button 
                onClick={fetchExplanation} 
                className="action-btn"
                style={{ marginTop: '12px' }}
              >
                <Loader2 size={16} />
                Retry
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!explain) {
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
              <Brain className="header-icon" size={48} />
              <h1 className="main-title">Explainable AI Insights</h1>
            </div>
            <p className="subtitle">Understanding model decisions and feature importance</p>
          </header>
          
          <nav className="top-nav">
            <a href="/" className="nav-link">Single Flow Analysis</a>
            <a href="/pcap" className="nav-link">PCAP Upload</a>
            <a href="/realtime" className="nav-link">Real-Time Monitor</a>
            <a href="/chatbot" className="nav-btn">
              <Zap size={16} />
              Cyber AI Chatbot
            </a>
            <a href="/explain" className="nav-link nav-active">Explain</a>
          </nav>
          
          <div className="card empty-card">
            <Brain size={64} className="empty-icon" />
            <h3>No Analysis Data Found</h3>
            <p>Please run an analysis first to generate AI explanations</p>
            <a href="/" className="action-btn" style={{ marginTop: '16px' }}>
              <Shield size={16} />
              Go to Single Flow Analysis
            </a>
          </div>
        </div>
      </div>
    );
  }

  const model = explain[selectedModel];
  const interpretation = getInterpretation(model);

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
            <Brain className="header-icon" size={48} />
            <h1 className="main-title">Explainable AI Insights</h1>
          </div>
          <p className="subtitle">Understanding model decisions and feature importance</p>
        </header>

        <nav className="top-nav">
          <a href="/" className="nav-link">Single Flow Analysis</a>
          <a href="/pcap" className="nav-link">PCAP Upload</a>
          <a href="/realtime" className="nav-link">Real-Time Monitor</a>
          <a href="/chatbot" className="nav-btn">
            <Zap size={16} />
            Cyber AI Chatbot
          </a>
          <a href="/explain" className="nav-link nav-active">Explain</a>
        </nav>

        {/* Prediction Overview */}
        <div className="summary-grid">
          <div className="summary-card" style={{
            background: `linear-gradient(135deg, ${getPredictionColor(explain.ensemble?.prediction)}20, rgba(15, 23, 42, 0.7))`,
            borderColor: getPredictionColor(explain.ensemble?.prediction)
          }}>
            <div className="summary-content">
              <div>
                <p className="summary-label">Final Prediction</p>
                <p className="summary-value" style={{ color: getPredictionColor(explain.ensemble?.prediction) }}>
                  {explain.ensemble?.prediction?.toUpperCase() || "UNKNOWN"}
                </p>
              </div>
              <Shield size={48} className="summary-icon" />
            </div>
          </div>

          <div className="summary-card summary-card-purple">
            <div className="summary-content">
              <div>
                <p className="summary-label">Confidence Score</p>
                <p className="summary-value">
                  {Math.round((explain.ensemble?.score || 0) * 100)}%
                </p>
              </div>
              <TrendingUp size={48} className="summary-icon" />
            </div>
          </div>

          <div className="summary-card summary-card-blue">
            <div className="summary-content">
              <div>
                <p className="summary-label">Models Analyzed</p>
                <p className="summary-value">{Object.keys(explain).length}</p>
              </div>
              <Server size={48} className="summary-icon" />
            </div>
          </div>
        </div>

        {/* Model Selection & Interpretation */}
        <div className="card">
          <div className="card-header">
            <Cpu size={24} className="section-icon" />
            <h2>Model Interpretation</h2>
          </div>

          <div className="model-selection">
            <div className="model-selector">
              <label>Select Model for Explanation</label>
              <div className="custom-select">
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="model-select"
                >
                  {Object.keys(explain).map((key) => (
                    <option key={key} value={key}>
                      {key.toUpperCase()} ({getModelType(key)})
                    </option>
                  ))}
                </select>
                <ChevronDown size={20} className="select-icon" />
              </div>
            </div>

            <div className="model-info">
              <div className="model-type">
                {getModelIcon(selectedModel)}
                <span>{getModelType(selectedModel)} Model</span>
              </div>
            </div>
          </div>

          <div className="interpretation-box">
            <div className="interpretation-header">
              <Info size={20} />
              <h3>{interpretation.title}</h3>
            </div>
            <p className="interpretation-text">{interpretation.description}</p>
            <div className="interpretation-insight">
              <HelpCircle size={16} />
              <span>{interpretation.insight}</span>
            </div>
          </div>
        </div>

        {/* Analysis Tabs */}
        <div className="card">
          <div className="tabs-header">
            <div className="tabs">
              <button 
                className={`tab-btn ${activeTab === "features" ? "active" : ""}`}
                onClick={() => setActiveTab("features")}
              >
                <BarChart3 size={18} />
                Feature Analysis
              </button>
              <button 
                className={`tab-btn ${activeTab === "visualization" ? "active" : ""}`}
                onClick={() => setActiveTab("visualization")}
              >
                <TrendingUp size={18} />
                Visualization
              </button>
              <button 
                className={`tab-btn ${activeTab === "details" ? "active" : ""}`}
                onClick={() => setActiveTab("details")}
              >
                <BookOpen size={18} />
                Technical Details
              </button>
            </div>
          </div>

          <div className="tab-content">
            {/* Feature Analysis Tab */}
            {activeTab === "features" && (
              <div className="features-grid">
                {model?.feature_importance && (
                  <>
                    <div className="feature-list">
                      <h4>Top Contributing Features</h4>
                      {model.top_features.slice(0, 5).map(([name, value], i) => (
                        <div key={i} className="feature-item">
                          <div className="feature-header">
                            <span className="feature-name">{name}</span>
                            <span className={`feature-impact ${value > 0 ? "positive" : "negative"}`}>
                              {value > 0 ? "+" : ""}{value.toFixed(4)}
                            </span>
                          </div>
                          <div className="feature-bar">
                            <div 
                              className="feature-bar-fill"
                              style={{ 
                                width: `${Math.min(Math.abs(value) * 100, 100)}%`,
                                background: value > 0 
                                  ? "linear-gradient(90deg, #10b981, #14b8a6)" 
                                  : "linear-gradient(90deg, #ef4444, #f97316)"
                              }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className="feature-chart">
                      <h4>Feature Impact Distribution</h4>
                      <div style={{ height: 280 }}>
                        <ResponsiveContainer>
                          <BarChart data={featureData(model.feature_importance)}>
                            <defs>
                              <linearGradient id="featureGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#a855f7" stopOpacity={1} />
                                <stop offset="100%" stopColor="#ec4899" stopOpacity={0.8} />
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                            <XAxis 
                              dataKey="feature" 
                              angle={-45} 
                              textAnchor="end" 
                              height={80} 
                              tick={{ fill: "#94a3b8", fontSize: 11 }} 
                            />
                            <YAxis tick={{ fill: "#94a3b8", fontSize: 12 }} />
                            <Tooltip 
                              contentStyle={{ 
                                background: 'rgba(15, 23, 42, 0.95)', 
                                border: '1px solid #475569',
                                borderRadius: '8px',
                                color: '#e2e8f0'
                              }} 
                              formatter={(value) => [value.toFixed(4), "Impact"]}
                            />
                            <Bar 
                              dataKey="value" 
                              fill="url(#featureGradient)" 
                              radius={[8, 8, 0, 0]} 
                              name="Impact"
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </>
                )}

                {model?.deviation && (
                  <>
                    <div className="feature-list">
                      <h4>Highest Deviations</h4>
                      {deviationData(model.deviation).map((item, i) => (
                        <div key={i} className="feature-item">
                          <div className="feature-header">
                            <span className="feature-name">{item.feature}</span>
                            <span className={`deviation-value ${item.direction}`}>
                              {item.direction === "above" ? "↑" : "↓"} {item.deviation.toFixed(4)}
                            </span>
                          </div>
                          <div className="deviation-bar">
                            <div 
                              className="deviation-fill"
                              style={{ 
                                width: `${Math.min(item.deviation * 50, 100)}%`,
                                background: item.direction === "above" 
                                  ? "linear-gradient(90deg, #f59e0b, #f97316)" 
                                  : "linear-gradient(90deg, #6366f1, #8b5cf6)"
                              }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className="feature-chart">
                      <h4>Deviation Pattern Analysis</h4>
                      <div style={{ height: 280 }}>
                        <ResponsiveContainer>
                          <RadarChart data={deviationData(model.deviation)}>
                            <PolarGrid stroke="#334155" />
                            <PolarAngleAxis 
                              dataKey="feature" 
                              tick={{ fill: "#94a3b8", fontSize: 11 }} 
                            />
                            <PolarRadiusAxis 
                              tick={{ fill: "#94a3b8" }} 
                              domain={[0, dataMax => Math.ceil(dataMax * 1.2)]}
                            />
                            <Radar 
                              name="Deviation" 
                              dataKey="deviation" 
                              stroke="#f59e0b" 
                              fill="#f59e0b" 
                              fillOpacity={0.5} 
                            />
                            <Tooltip 
                              contentStyle={{ 
                                background: 'rgba(15, 23, 42, 0.95)', 
                                border: '1px solid #475569',
                                borderRadius: '8px',
                                color: '#e2e8f0'
                              }} 
                            />
                          </RadarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </>
                )}

                {model?.score !== undefined && (
                  <div className="score-analysis">
                    <h4>Anomaly Score Analysis</h4>
                    <div className="score-display">
                      <div className="score-value">
                        <span className="score-number">{model.score.toFixed(4)}</span>
                        <span className="score-label">Outlier Score</span>
                      </div>
                      <div className="score-meter">
                        <div 
                          className="score-fill"
                          style={{ 
                            width: `${model.score * 100}%`,
                            background: "linear-gradient(90deg, #ef4444, #f97316)"
                          }}
                        ></div>
                        <div className="score-markers">
                          <span className="marker">Normal</span>
                          <span className="marker">Suspicious</span>
                          <span className="marker">Anomaly</span>
                        </div>
                      </div>
                    </div>
                    <p className="score-interpretation">{model.meaning}</p>
                  </div>
                )}
              </div>
            )}

            {/* Visualization Tab */}
            {activeTab === "visualization" && (
              <div className="visualization-grid">
                {model?.feature_importance && (
                  <div className="viz-card">
                    <h4>Impact Direction</h4>
                    <div style={{ height: 300 }}>
                      <ResponsiveContainer>
                        <PieChart>
                          <Pie
                            data={featureData(model.feature_importance).map(d => ({
                              ...d,
                              name: d.feature
                            }))}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={(entry) => entry.name}
                            outerRadius={100}
                            fill="#8884d8"
                            dataKey="value"
                          >
                            {featureData(model.feature_importance).map((entry, index) => (
                              <Cell 
                                key={`cell-${index}`} 
                                fill={entry.impact === "positive" ? "#10b981" : "#ef4444"} 
                              />
                            ))}
                          </Pie>
                          <Tooltip 
                            contentStyle={{ 
                              background: 'rgba(15, 23, 42, 0.95)', 
                              border: '1px solid #475569',
                              borderRadius: '8px',
                              color: '#e2e8f0'
                            }} 
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                <div className="viz-card">
                  <h4>Model Contribution</h4>
                  <div className="contribution-breakdown">
                    {Object.keys(explain).map(key => {
                      const m = explain[key];
                      const contribution = m.score || Object.values(m.feature_importance || {})[0] || 0;
                      return (
                        <div key={key} className="contribution-item">
                          <div className="contribution-info">
                            {getModelIcon(key)}
                            <span>{key}</span>
                          </div>
                          <div className="contribution-value">
                            {Math.abs(contribution).toFixed(4)}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {/* Details Tab */}
            {activeTab === "details" && (
              <div className="details-content">
                <div className="details-grid">
                  <div className="detail-card">
                    <h4>Model Type</h4>
                    <p>{getModelType(selectedModel)}</p>
                  </div>
                  <div className="detail-card">
                    <h4>Analysis Method</h4>
                    <p>
                      {model?.feature_importance ? "Feature Importance" : 
                       model?.deviation ? "Pattern Deviation" : 
                       model?.score !== undefined ? "Statistical Scoring" : "Composite Analysis"}
                    </p>
                  </div>
                  <div className="detail-card">
                    <h4>Features Analyzed</h4>
                    <p>{Object.keys(model?.feature_importance || model?.deviation || {}).length}</p>
                  </div>
                  <div className="detail-card">
                    <h4>Decision Threshold</h4>
                    <p>{model?.threshold || "0.5"}</p>
                  </div>
                </div>
                
                <div className="technical-notes">
                  <h4>Technical Notes</h4>
                  <ul>
                    <li>Positive feature values increase anomaly likelihood</li>
                    <li>Negative values indicate normal patterns</li>
                    <li>Higher deviations suggest unusual behavior</li>
                    <li>Scores above 0.5 typically indicate anomalies</li>
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Additional Insights */}
        <div className="card">
          <div className="card-header">
            <Brain size={24} className="section-icon" />
            <h2>Additional Insights</h2>
          </div>
          <div className="insights-grid">
            <div className="insight-card">
              <h4>⚡ Key Finding</h4>
              <p>{interpretation.description}</p>
            </div>
            <div className="insight-card">
              <h4>🔍 Recommendation</h4>
              <p>
                {explain.ensemble?.prediction === "anomaly" 
                  ? "Consider immediate investigation and network traffic monitoring." 
                  : explain.ensemble?.prediction === "suspicious"
                  ? "Monitor closely and review related network activities."
                  : "No immediate action required. Continue regular monitoring."}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}