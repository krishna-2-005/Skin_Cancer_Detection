import { useEffect, useMemo, useState } from "react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const FIXED_TTA_RUNS = 8;
const FIXED_CONFIDENCE_TEMPERATURE = 0.6;

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showFullBreakdown, setShowFullBreakdown] = useState(false);
  const [apiHealth, setApiHealth] = useState(null);
  const [healthError, setHealthError] = useState("");

  useEffect(() => {
    let mounted = true;

    async function loadHealth() {
      try {
        const response = await axios.get(`${API_BASE}/health`);
        if (mounted) {
          setApiHealth(response.data);
          setHealthError("");
        }
      } catch (err) {
        if (mounted) {
          setHealthError(err.message || "API unavailable");
        }
      }
    }

    loadHealth();
    const interval = setInterval(loadHealth, 12000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  async function refreshModelCheckpoint() {
    try {
      await axios.post(`${API_BASE}/reload-model`);
      const health = await axios.get(`${API_BASE}/health`);
      setApiHealth(health.data);
      setHealthError("");
    } catch (err) {
      setHealthError(err.message || "Failed to reload model");
    }
  }

  const confidencePct = useMemo(() => {
    if (!result) return 0;
    return Math.round(result.confidence * 1000) / 10;
  }, [result]);

  const sortedProbabilities = useMemo(() => {
    if (!result?.probabilities) return [];
    return Object.entries(result.probabilities)
      .map(([label, value]) => ({
        label,
        disease: value.disease,
        probability: value.probability,
      }))
      .sort((a, b) => b.probability - a.probability);
  }, [result]);

  const confidenceBand = useMemo(() => {
    if (!result) return "";
    if (result.confidence >= 0.85) return "High confidence";
    if (result.confidence >= 0.6) return "Moderate confidence";
    return "Low confidence";
  }, [result]);

  function onFileChange(event) {
    const nextFile = event.target.files?.[0] || null;
    setResult(null);
    setError("");
    setFile(nextFile);

    if (!nextFile) {
      setPreview("");
      return;
    }

    const fileReader = new FileReader();
    fileReader.onload = () => setPreview(fileReader.result?.toString() || "");
    fileReader.readAsDataURL(nextFile);
  }

  async function onSubmit(event) {
    event.preventDefault();
    if (!file) {
      setError("Upload a lesion image before running prediction.");
      return;
    }

    setError("");
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post(
        `${API_BASE}/predict?explain=true&tta_runs=${FIXED_TTA_RUNS}&confidence_temperature=${FIXED_CONFIDENCE_TEMPERATURE}&include_probabilities=${showFullBreakdown}`,
        formData,
        {
        headers: { "Content-Type": "multipart/form-data" },
        },
      );
      setResult(response.data);
    } catch (err) {
      const message = err.response?.data?.detail || err.message || "Prediction failed";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="page">
      <section className="hero">
        <p className="eyebrow">DermAegis AI</p>
        <h1>SKINVISION AI</h1>
        <p className="hero-subtitle">Skin Lesion Intelligence Workspace</p>
        <p>
          Upload a lesion image, run your trained model, inspect probabilities,
          and verify attention focus with Grad-CAM.
        </p>
      </section>

      <section className="status-strip">
        <div>
          <p className="status-kicker">API</p>
          <p className={`status-value ${apiHealth?.status === "ok" ? "ok" : "bad"}`}>
            {apiHealth?.status === "ok" ? "Connected" : "Disconnected"}
          </p>
        </div>
        <div>
          <p className="status-kicker">Model</p>
          <p className={`status-value ${apiHealth?.model_loaded ? "ok" : "bad"}`}>
            {apiHealth?.model_loaded ? "Loaded" : "Not loaded yet"}
          </p>
        </div>
        <div>
          <p className="status-kicker">Model Accuracy</p>
          <p className="status-value">
            {typeof apiHealth?.model_metrics?.accuracy === "number"
              ? `${(apiHealth.model_metrics.accuracy * 100).toFixed(1)}% test`
              : "Unavailable"}
          </p>
        </div>
        <div className="status-path-wrap">
          <p className="status-kicker">Model Path</p>
          <p className="status-path">{apiHealth?.model_path || "Waiting for model file"}</p>
        </div>
      </section>
      <div className="toolbar">
        <button type="button" onClick={refreshModelCheckpoint}>
          Load Latest Checkpoint
        </button>
      </div>
      {healthError && <p className="error">API check failed: {healthError}</p>}

      <section className="panel">
        <form onSubmit={onSubmit} className="upload-form">
          <label htmlFor="image" className="upload-box">
            <span>Choose image (JPG, PNG, WEBP)</span>
            <input id="image" type="file" accept="image/*" onChange={onFileChange} />
          </label>
          <button disabled={isLoading} type="submit">
            {isLoading ? "Analyzing..." : "Run Prediction"}
          </button>
        </form>
        <div className="tta-controls">
          <label htmlFor="breakdown">
            <input
              id="breakdown"
              type="checkbox"
              checked={showFullBreakdown}
              onChange={(e) => setShowFullBreakdown(e.target.checked)}
            />{" "}
            Show full probability breakdown
          </label>
        </div>

        {error && <p className="error">{error}</p>}

        <div className="grid">
          <article className="card">
            <h2>Input</h2>
            {preview ? <img src={preview} alt="Uploaded lesion" className="preview" /> : <p>No image selected.</p>}
          </article>

          <article className="card">
            <h2>Prediction</h2>
            {result ? (
              <>
                <p className="prediction-label">{result.predicted_disease}</p>
                <p className="prediction-meta">Label: {result.predicted_label}</p>
                <p className="prediction-meta">Single-image confidence: {confidencePct}%</p>
                <p className="prediction-band">{confidenceBand}</p>
                {showFullBreakdown && sortedProbabilities.length > 0 && (
                  <ul className="prob-list">
                    {sortedProbabilities.map((item) => (
                      <li key={item.label}>
                        <div className="prob-head">
                          <span>{item.disease}</span>
                          <span>{(item.probability * 100).toFixed(1)}%</span>
                        </div>
                        <div className="bar-track">
                          <div className="bar-fill" style={{ width: `${item.probability * 100}%` }} />
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </>
            ) : (
              <p>Run inference to view results.</p>
            )}
          </article>

          <article className="card card-wide">
            <h2>Grad-CAM Explainability</h2>
            {result?.gradcam_base64 ? (
              <img
                className="preview"
                src={`data:image/png;base64,${result.gradcam_base64}`}
                alt="Grad CAM"
              />
            ) : (
              <p>Heatmap will appear after prediction.</p>
            )}
          </article>
        </div>

        <div className="notice">
          <strong>Clinical Safety Notice:</strong> this tool supports research/learning only.
          Final diagnosis must be performed by a qualified dermatologist.
        </div>
      </section>
    </main>
  );
}

export default App;
