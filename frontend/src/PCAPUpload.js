import React, { useState } from "react";
import { 
  Upload, 
  Shield, 
  FileUp, 
  BarChart3, 
  AlertTriangle, 
  CheckCircle, 
  XCircle,
  Download,
  Zap,
  FolderUp,
  Activity,
  Server
} from "lucide-react";

function PCAPUpload() {
  const [pcapFile, setPcapFile] = useState(null);
  const [pcapResult, setPcapResult] = useState(null);
  const [pcapLoading, setPcapLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [showDetails, setShowDetails] = useState({});

  const handlePcapUpload = async () => {
    if (!pcapFile) {
      setError("Please select a PCAP file.");
      return;
    }

    setError(null);
    setPcapLoading(true);
    setPcapResult(null);

    try {
      const formData = new FormData();
      formData.append("file", pcapFile);

      const res = await fetch("http://127.0.0.1:8000/upload_pcap", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setPcapResult(data);
    } catch (err) {
      setError("Failed to analyze PCAP. Please check the file format and try again.");
    }

    setPcapLoading(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && (file.name.endsWith('.pcap') || file.name.endsWith('.pcapng'))) {
      setPcapFile(file);
      setError(null);
    } else {
      setError("Please upload only PCAP or PCAPNG files.");
    }
  };

  const toggleFlowDetails = (flowId) => {
    setShowDetails(prev => ({
      ...prev,
      [flowId]: !prev[flowId]
    }));
  };

  const getFlowStatus = (predictions) => {
    const anomalyCount = Object.values(predictions).filter(p => p.prediction === "anomaly").length;
    const totalModels = Object.keys(predictions).length;
    
    if (anomalyCount === 0) return { status: "normal", confidence: "high" };
    if (anomalyCount <= totalModels / 3) return { status: "suspicious", confidence: "medium" };
    return { status: "anomaly", confidence: "high" };
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "normal": return "#10b981";
      case "suspicious": return "#f59e0b";
      case "anomaly": return "#ef4444";
      default: return "#6b7280";
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "normal": return <CheckCircle size={16} />;
      case "suspicious": return <AlertTriangle size={16} />;
      case "anomaly": return <XCircle size={16} />;
      default: return <Activity size={16} />;
    }
  };

  const downloadReport = () => {
    if (!pcapResult) return;
    
    const report = {
      filename: pcapFile.name,
      uploadedAt: new Date().toISOString(),
      analysis: pcapResult
    };
    
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pcap-analysis-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const calculateStats = () => {
    if (!pcapResult) return { normal: 0, suspicious: 0, anomalies: 0 };
    
    let normal = 0, suspicious = 0, anomalies = 0;
    
    pcapResult.analysis.forEach(flow => {
      const status = getFlowStatus(flow.predictions).status;
      if (status === "normal") normal++;
      else if (status === "suspicious") suspicious++;
      else anomalies++;
    });
    
    return { normal, suspicious, anomalies };
  };

  const stats = calculateStats();

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
            <Upload className="header-icon" size={48} />
            <h1 className="main-title">PCAP Network Analysis</h1>
          </div>
          <p className="subtitle">Upload packet captures for comprehensive multi-flow anomaly detection</p>
        </header>

        <nav className="top-nav">
          <a href="/" className="nav-link">Single Flow Analysis</a>
          <a href="/pcap" className="nav-link nav-active">PCAP Upload</a>
          <a href="/realtime" className="nav-link">Real-Time Monitor</a>
          <a href="/chatbot" className="nav-btn">
            <Zap size={16} />
            Cyber AI Chatbot
          </a>
          <a href="/explain" className="nav-link">Explain</a>
        </nav>

        {/* Upload Card */}
        <div className="card">
          <div className="card-header">
            <FolderUp size={24} className="section-icon" />
            <h2>Upload PCAP File</h2>
          </div>

          <div className="upload-container">
            <div 
              className={`upload-area ${dragOver ? 'upload-area-dragover' : ''} ${pcapFile ? 'upload-area-hasfile' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => document.getElementById('pcap-file-input')?.click()}
            >
              <input
                id="pcap-file-input"
                type="file"
                accept=".pcap,.pcapng"
                onChange={(e) => {
                  setPcapFile(e.target.files[0]);
                  setError(null);
                }}
                className="pcap-input-hidden"
              />
              
              <div className="upload-content">
                <FileUp size={48} className="upload-icon" />
                <div className="upload-text">
                  <h3>{pcapFile ? pcapFile.name : 'Drag & drop PCAP file'}</h3>
                  <p>{pcapFile ? `${(pcapFile.size / 1024 / 1024).toFixed(2)} MB` : 'or click to browse (PCAP/PCAPNG)'}</p>
                </div>
                {pcapFile && (
                  <div className="file-info">
                    <CheckCircle size={20} className="file-check" />
                  </div>
                )}
              </div>
            </div>

            <div className="upload-actions">
              <button 
                onClick={handlePcapUpload} 
                disabled={pcapLoading || !pcapFile} 
                className="upload-btn"
              >
                {pcapLoading ? (
                  <>
                    <div className="spinner"></div>
                    Analyzing PCAP...
                  </>
                ) : (
                  <>
                    <Upload size={20} />
                    Run PCAP Analysis
                  </>
                )}
              </button>
              
              {pcapFile && !pcapLoading && (
                <button 
                  onClick={() => setPcapFile(null)} 
                  className="upload-clear-btn"
                >
                  <XCircle size={20} />
                  Clear File
                </button>
              )}
            </div>

            {error && (
              <div className="upload-error">
                <AlertTriangle size={20} />
                <span>{error}</span>
              </div>
            )}
          </div>
        </div>

        {/* Loading State */}
        {pcapLoading && (
          <div className="card loading-card">
            <div className="loading-content">
              <div className="loading-spinner-wrapper">
                <div className="loading-ring"></div>
                <Shield size={48} className="loading-icon" />
              </div>
              <p className="loading-title">Analyzing PCAP File</p>
              <p className="loading-subtitle">
                Processing {pcapFile?.name || 'file'}...
              </p>
            </div>
          </div>
        )}

        {/* Results Section */}
        {pcapResult && !pcapLoading && (
          <>
            {/* Stats Summary */}
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-content">
                  <div>
                    <p className="stat-label">Total Flows</p>
                    <p className="stat-value">{pcapResult.flow_count}</p>
                  </div>
                  <BarChart3 size={48} className="stat-icon stat-icon-purple" />
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-content">
                  <div>
                    <p className="stat-label">Normal Flows</p>
                    <p className="stat-value">{stats.normal}</p>
                  </div>
                  <CheckCircle size={48} className="stat-icon stat-icon-green" />
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-content">
                  <div>
                    <p className="stat-label">Suspicious Flows</p>
                    <p className="stat-value">{stats.suspicious}</p>
                  </div>
                  <AlertTriangle size={48} className="stat-icon stat-icon-yellow" />
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-content">
                  <div>
                    <p className="stat-label">Anomalous Flows</p>
                    <p className="stat-value">{stats.anomalies}</p>
                  </div>
                  <XCircle size={48} className="stat-icon stat-icon-red" />
                </div>
              </div>
            </div>

            {/* Analysis Results */}
            <div className="card">
              <div className="card-header">
                <Activity size={24} className="section-icon" />
                <h2>Flow Analysis Results</h2>
                <div className="header-actions">
                  <button onClick={downloadReport} className="action-btn">
                    <Download size={16} />
                    Download Report
                  </button>
                </div>
              </div>

              <div className="analysis-results">
                {pcapResult.analysis.map((flow) => {
                  const status = getFlowStatus(flow.predictions);
                  const models = Object.entries(flow.predictions);
                  
                  return (
                    <div key={flow.flow_id} className="flow-analysis-card">
                      <div 
                        className="flow-card-header"
                        onClick={() => toggleFlowDetails(flow.flow_id)}
                      >
                        <div className="flow-info">
                          <div className="flow-id">
                            <Server size={16} />
                            Flow #{flow.flow_id}
                          </div>
                          <div 
                            className="flow-status"
                            style={{ 
                              backgroundColor: `${getStatusColor(status.status)}20`,
                              borderColor: getStatusColor(status.status)
                            }}
                          >
                            {getStatusIcon(status.status)}
                            <span>{status.status.toUpperCase()}</span>
                          </div>
                        </div>
                        
                        <div className="flow-stats">
                          <div className="flow-stat">
                            <span className="stat-label">Models:</span>
                            <span className="stat-value">{models.length}</span>
                          </div>
                          <div className="flow-stat">
                            <span className="stat-label">Anomalies:</span>
                            <span className="stat-value">
                              {models.filter(([_, p]) => p.prediction === "anomaly").length}
                            </span>
                          </div>
                          <div className="flow-stat">
                            <span className="stat-label">Confidence:</span>
                            <span className="stat-value">{status.confidence}</span>
                          </div>
                        </div>
                        
                        <div className={`flow-toggle ${showDetails[flow.flow_id] ? 'expanded' : ''}`}>
                          ▼
                        </div>
                      </div>

                      {showDetails[flow.flow_id] && (
                        <div className="flow-details">
                          <div className="models-breakdown">
                            {models.map(([modelName, prediction]) => (
                              <div key={modelName} className="model-prediction-card">
                                <h4 className="model-name">{modelName.replace(/_/g, ' ')}</h4>
                                <div className="prediction-details">
                                  <div className="prediction-row">
                                    <span className="prediction-label">Status:</span>
                                    <span 
                                      className="prediction-value"
                                      style={{ color: getStatusColor(prediction.prediction) }}
                                    >
                                      {getStatusIcon(prediction.prediction)}
                                      {prediction.prediction.toUpperCase()}
                                    </span>
                                  </div>
                                  <div className="prediction-row">
                                    <span className="prediction-label">Confidence:</span>
                                    <span className="prediction-confidence">
                                      {(prediction.score * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                  <div className="prediction-progress">
                                    <div 
                                      className="prediction-progress-fill"
                                      style={{ 
                                        width: `${prediction.score * 100}%`,
                                        background: getStatusColor(prediction.prediction)
                                      }}
                                    ></div>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Summary Card */}
            <div className="card">
              <div className="card-header">
                <Shield size={24} className="section-icon" />
                <h2>Analysis Summary</h2>
              </div>
              
              <div className="summary-content">
                <div className="summary-text">
                  <h3>Overall Assessment</h3>
                  <p>
                    {stats.anomalies > 0 ? (
                      <span className="warning-text">
                        ⚠️ {stats.anomalies} anomalous flow{stats.anomalies !== 1 ? 's' : ''} detected. 
                        This requires immediate attention and investigation.
                      </span>
                    ) : stats.suspicious > 0 ? (
                      <span className="caution-text">
                        ⚠️ {stats.suspicious} suspicious flow{stats.suspicious !== 1 ? 's' : ''} detected. 
                        Consider reviewing these flows for potential security concerns.
                      </span>
                    ) : (
                      <span className="safe-text">
                        ✅ All {pcapResult.flow_count} flows appear normal. No anomalies detected in this capture.
                      </span>
                    )}
                  </p>
                </div>
                
                <div className="summary-actions">
                  <button 
                    className="explain-btn"
                    onClick={async () => {
                      try {
                        const res = await fetch("http://127.0.0.1:8000/explain_last");
                        const data = await res.json();
                        localStorage.setItem("xai_data", JSON.stringify(data));
                        window.location.href = "/explain";
                      } catch (err) {
                        alert("Unable to generate explanation. Please try again.");
                      }
                    }}
                  >
                    🔍 Explain Findings
                  </button>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default PCAPUpload;