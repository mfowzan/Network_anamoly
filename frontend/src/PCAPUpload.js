import React, { useState } from "react";
import { Upload, Shield } from "lucide-react";

function PCAPUpload() {
  const [pcapFile, setPcapFile] = useState(null);
  const [pcapResult, setPcapResult] = useState(null);
  const [pcapLoading, setPcapLoading] = useState(false);
  const [error, setError] = useState(null);

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
      setError("Failed to analyze PCAP.");
    }

    setPcapLoading(false);
  };

  return (
    <div className="content-wrapper">
      <header className="dashboard-header">
        <div className="header-content">
          <Upload size={48} className="header-icon" />
          <h1 className="main-title">PCAP Flow Analysis</h1>
        </div>
        <p className="subtitle">Upload a PCAP file for multi-flow anomaly detection</p>
      </header>

      <div className="card">
        <div className="card-header">
          <Upload size={24} className="section-icon" />
          <h2>Upload PCAP File</h2>
        </div>

        <input
          type="file"
          accept=".pcap,.pcapng"
          onChange={(e) => setPcapFile(e.target.files[0])}
          className="pcap-input"
        />

        <button onClick={handlePcapUpload} className="submit-btn" disabled={pcapLoading}>
          {pcapLoading ? "Analyzing PCAP..." : "Run PCAP Analysis"}
        </button>

        {error && <div className="card error-card">{error}</div>}

        {pcapResult && (
          <div className="card">
            <h3>Detected Flows: {pcapResult.flow_count}</h3>

            <div className="models-grid">
              {pcapResult.analysis.map((flow) => (
                <div key={flow.flow_id} className="model-card model-anomaly">
                  <h4>Flow #{flow.flow_id}</h4>
                  <pre>{JSON.stringify(flow.predictions, null, 2)}</pre>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default PCAPUpload;
