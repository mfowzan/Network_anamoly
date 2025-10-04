import React, { useState } from "react";
import ClipLoader from "react-spinners/ClipLoader";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

function App() {
  const [formData, setFormData] = useState({
    duration: "",
    src_bytes: "",
    dst_bytes: "",
    count: "",
    srv_count: "",
    wrong_fragment: "",
  });
  const [selectedModel, setSelectedModel] = useState("autoencoder"); // default model
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  // baseline "normal" values
  const baseline = {
    duration: 20,
    src_bytes: 5000,
    dst_bytes: 5000,
    count: 10,
    srv_count: 10,
    wrong_fragment: 0,
  };

  // max values for normalization
  const maxValues = {
    duration: 100,
    src_bytes: 100000,
    dst_bytes: 100000,
    count: 200,
    srv_count: 200,
    wrong_fragment: 10,
  };

  const getInsights = (data, prediction) => {
    if (prediction === 1) {
      return "✅ Normal – Traffic looks balanced, no unusual patterns detected.";
    }

    let reasons = [];
    if (data.count > 50) reasons.push("High number of connections (possible DoS).");
    if (data.wrong_fragment > 0) reasons.push("Wrong fragments detected (malicious packet crafting).");
    if (data.src_bytes > 100000) reasons.push("Very high source bytes (possible data exfiltration).");
    if (data.dst_bytes === 0) reasons.push("Destination not responding (possible failed connections).");
    if (data.srv_count > 100) reasons.push("Too many requests to same service (possible probing).");

    return reasons.length === 0
      ? "⚠️ Suspicious – Unusual activity detected, but no clear dominant feature."
      : "⚠️ Suspicious Activity Detected – " + reasons.join(" ");
  };

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);

    try {
      const response = await fetch(`http://127.0.0.1:8000/predict?model=${selectedModel}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify([formData]),
      });

      if (!response.ok) throw new Error("Failed to fetch prediction");

      const data = await response.json();
      const prediction = data.results[0].prediction;
      const score = data.results[0].score;
      const top_features = data.results[0].top_features || [];
      const explanation_method = data.results[0].explanation_method || "unknown";

      setResult({
        prediction,
        score,
        insights: getInsights(formData, prediction),
        top_features,
        explanation_method,
        model: data.model,
      });
    } catch (err) {
      console.error("Error:", err);
      setError("Could not fetch prediction. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // Prepare normalized chart data
  const chartData = Object.keys(formData).map((key) => {
    const userVal = Number(formData[key]) || 0;
    const normalVal = baseline[key];
    const maxVal = maxValues[key] || 1;

    return {
      metric: key.replace("_", " "),
      user: (userVal / maxVal) * 100,
      normal: (normalVal / maxVal) * 100,
    };
  });

  // Feature importance chart
  const featureImportanceData =
    result?.top_features?.map((item) => {
      const feature = Array.isArray(item) ? item[0] : item.feature;
      const value = Array.isArray(item) ? Number(item[1]) : Number(item.value);
      return { feature, impact: Number(value) };
    }) || [];

  return (
    <div style={{ fontFamily: "Arial, sans-serif", padding: "20px", maxWidth: "900px", margin: "auto" }}>
      <h2>🔍 Network Anomaly Detection</h2>
      <form onSubmit={handleSubmit} style={{ marginBottom: "20px" }}>
        {/* Model Selector */}
        <div style={{ marginBottom: "15px" }}>
          <label style={{ marginRight: "10px" }}>Select Model:</label>
          <select value={selectedModel} onChange={handleModelChange} style={{ padding: "5px" }}>
            <option value="autoencoder">Autoencoder</option>
            <option value="isolation_forest">Isolation Forest</option>
          </select>
        </div>

        {/* Input Fields */}
        {Object.keys(formData).map((field) => (
          <div key={field} style={{ marginBottom: "10px" }}>
            <label style={{ marginRight: "10px", textTransform: "capitalize" }}>
              {field.replace("_", " ")}:
            </label>
            <input
              type="number"
              name={field}
              value={formData[field]}
              onChange={handleChange}
              required
              style={{ padding: "5px", width: "200px" }}
            />
          </div>
        ))}
        <button
          type="submit"
          disabled={loading}
          style={{
            padding: "10px 20px",
            backgroundColor: loading ? "#6c757d" : "#007BFF",
            color: "#fff",
            border: "none",
            cursor: loading ? "not-allowed" : "pointer",
            borderRadius: "5px",
          }}
        >
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {/* Spinner */}
      {loading && (
        <div style={{ textAlign: "center", margin: "20px 0" }}>
          <ClipLoader size={50} color="#007BFF" />
          <p style={{ marginTop: "10px", fontWeight: "bold" }}>Analyzing network traffic...</p>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <div>
          <div
            style={{
              padding: "15px",
              borderRadius: "8px",
              backgroundColor: result.prediction === 1 ? "#d4edda" : "#f8d7da",
              color: result.prediction === 1 ? "#155724" : "#721c24",
              marginBottom: "20px",
            }}
          >
            <h3>
              {result.prediction === 1 ? "✅ Normal Network Activity" : "⚠️ Suspicious Activity Detected"}
            </h3>
            <p>
              Confidence Score: <strong>{(result.score * 100).toFixed(2)}%</strong>
            </p>
            <p>
              Model Used: <strong>{result.model}</strong>
            </p>
            <p style={{ marginTop: "10px", fontStyle: "italic" }}>{result.insights}</p>
          </div>

          {/* Charts */}
          {result.top_features && result.top_features.length > 0 && (
            <div style={{ marginTop: "15px" }}>
              <h4>🔍 Key Factors Influencing This Prediction (method: {result.explanation_method}):</h4>
              <ul>
                {result.top_features.map(([feature, value], idx) => (
                  <li key={idx}>
                    <strong>{feature}</strong> (impact: {Number(value).toFixed(3)})
                  </li>
                ))}
              </ul>
              <div style={{ marginTop: "12px", minHeight: 220 }}>
                <h5 style={{ marginBottom: 8 }}>📊 Feature Importance (top contributors)</h5>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={featureImportanceData} layout="vertical" margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="feature" type="category" width={120} />
                    <Tooltip formatter={(value) => Number(value).toFixed(3)} />
                    <Bar dataKey="impact" fill="#ff7f0e" name="Impact" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          <h4>📊 Input vs Normal Profile (Normalized 0–100)</h4>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="metric" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Legend />
              <Bar dataKey="user" fill="#007BFF" name="User Input (normalized)" />
              <Bar dataKey="normal" fill="#28a745" name="Normal Traffic (normalized)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Errors */}
      {error && !loading && (
        <div style={{ padding: "10px", borderRadius: "8px", backgroundColor: "#f8d7da", color: "#721c24" }}>
          {error}
        </div>
      )}
    </div>
  );
}

export default App;
