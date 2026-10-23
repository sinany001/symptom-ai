import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [mode, setMode] = useState("login");
  const [token, setToken] = useState(
    () => localStorage.getItem("symptomai_token") || ""
  );
  const [user, setUser] = useState(null);

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [symptoms, setSymptoms] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function apiRequest(path, options = {}, authToken = token) {
    const headers = {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    };

    if (authToken) {
      headers.Authorization = `Bearer ${authToken}`;
    }

    const response = await fetch(`${API_URL}${path}`, {
      ...options,
      headers,
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(
        data.detail ||
          data.message ||
          `İstek başarısız oldu (${response.status}).`
      );
    }

    return data;
  }

  // Sayfa açıldığında kayıtlı oturumu kontrol et.
  useEffect(() => {
    if (!token) return;

    let cancelled = false;

    async function checkSession() {
      try {
        const currentUser = await apiRequest("/me", {}, token);

        if (!cancelled) {
          setUser(currentUser);
        }
      } catch {
        if (!cancelled) {
          localStorage.removeItem("symptomai_token");
          setToken("");
          setUser(null);
        }
      }
    }

    checkSession();

    return () => {
      cancelled = true;
    };
  }, [token]);

  // Giriş yapılınca kullanıcının analiz geçmişini getir.
  useEffect(() => {
    if (!token || !user) return;

    loadHistory();
  }, [token, user]);

  async function loadHistory() {
    try {
      const data = await apiRequest("/analyses");
      setHistory(data.analyses || []);
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleAuth(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setMessage("");

    try {
      if (mode === "register") {
        await apiRequest(
          "/register",
          {
            method: "POST",
            body: JSON.stringify({ name, email, password }),
          },
          ""
        );
      }

      // Kayıttan sonra da giriş yaparak token al.
      const data = await apiRequest(
        "/login",
        {
          method: "POST",
          body: JSON.stringify({ email, password }),
        },
        ""
      );

      localStorage.setItem("symptomai_token", data.access_token);
      setToken(data.access_token);
      setUser(data.user);
      setPassword("");
      setMessage("Başarıyla giriş yaptın.");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleAnalyze(event) {
    event.preventDefault();

    if (symptoms.trim().length < 3) {
      setError("Lütfen belirtilerini en az 3 karakterle açıkla.");
      return;
    }

    setLoading(true);
    setError("");
    setMessage("");
    setResult(null);

    try {
      const data = await apiRequest("/predict", {
        method: "POST",
        body: JSON.stringify({ symptoms: symptoms.trim() }),
      });

      setResult(data);
      setMessage("Analiz tamamlandı.");
      await loadHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleLogout() {
    localStorage.removeItem("symptomai_token");
    setToken("");
    setUser(null);
    setResult(null);
    setHistory([]);
    setSymptoms("");
    setMessage("");
    setError("");
    setMode("login");
  }

  const riskClass = result?.risk_level?.toLowerCase() || "";

  // Giriş ve kayıt ekranı
  if (!token || !user) {
    return (
      <main className="auth-page">
        <section className="auth-card">
          <div className="brand">
            <div className="brand-icon">✚</div>
            <h1>SymptomAI</h1>
            <p>AI-Based Preliminary Health Assessment</p>
          </div>

          <div className="auth-heading">
            <h2>{mode === "login" ? "Welcome back" : "Create account"}</h2>
            <p>
              {mode === "login"
                ? "Log in to continue to SymptomAI."
                : "Create an account to save your analyses."}
            </p>
          </div>

          <form onSubmit={handleAuth} className="form">
            {mode === "register" && (
              <label>
                Full name
                <input
                  type="text"
                  placeholder="Your name"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  maxLength={100}
                  required
                />
              </label>
            )}

            <label>
              Email address
              <input
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                required
              />
            </label>

            <label>
              Password
              <input
                type="password"
                placeholder="At least 8 characters"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                minLength={mode === "register" ? 8 : 1}
                maxLength={72}
                required
              />
            </label>

            {error && <div className="alert error">{error}</div>}
            {message && <div className="alert success">{message}</div>}

            <button className="primary-button" disabled={loading}>
              {loading
                ? "Please wait..."
                : mode === "login"
                  ? "Log in"
                  : "Create account"}
            </button>
          </form>

          <p className="switch-mode">
            {mode === "login"
              ? "Don't have an account?"
              : "Already have an account?"}{" "}
            <button
              type="button"
              className="text-button"
              onClick={() => {
                setMode(mode === "login" ? "register" : "login");
                setError("");
                setMessage("");
              }}
            >
              {mode === "login" ? "Sign up" : "Log in"}
            </button>
          </p>

          <p className="auth-footer">
            Your symptoms are personal information. Use this prototype for
            testing only.
          </p>
        </section>
      </main>
    );
  }

  // Ana uygulama
  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand compact" href="#home">
          <span className="brand-icon">✚</span>
          <span>SymptomAI</span>
        </a>

        <div className="user-menu">
          <span className="welcome">Hi, {user.name}</span>
          <button className="logout-button" onClick={handleLogout}>
            Log out
          </button>
        </div>
      </header>

      <main className="main-content" id="home">
        <section className="hero">
          <span className="eyebrow">AI-POWERED HEALTH SUPPORT</span>
          <h1>Understand your symptoms.</h1>
          <p>
            Describe how you're feeling to receive a preliminary,
            dataset-based assessment.
          </p>
        </section>

        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Describe your symptoms</h2>
              <p>Include how you feel and which symptoms you have noticed.</p>
            </div>
            <span className="panel-icon">✚</span>
          </div>

          <form onSubmit={handleAnalyze} className="form">
            <label>
              Symptoms
              <textarea
                value={symptoms}
                onChange={(event) => setSymptoms(event.target.value)}
                placeholder="Example: I have a headache, fever and cough..."
                rows={5}
                maxLength={5000}
                required
              />
            </label>

            <div className="form-bottom">
              <span className="character-count">
                {symptoms.length}/5000
              </span>
              <button className="primary-button analyze-button" disabled={loading}>
                {loading ? "Analyzing..." : "Analyze symptoms →"}
              </button>
            </div>
          </form>

          {error && <div className="alert error">{error}</div>}
          {message && <div className="alert success">{message}</div>}
        </section>

        {result && (
          <section className="panel result-panel">
            <div className="result-title">
              <div>
                <span className="eyebrow">YOUR RESULT</span>
                <h2>Preliminary assessment</h2>
              </div>
              <span className={`risk-badge ${riskClass}`}>
                {result.risk_level} RISK
              </span>
            </div>

            <div className="classification-box">
              <span className="result-label">Model classification</span>
              <h3>{result.classification}</h3>
              <p>
                This is the model's classification based on the dataset, not a
                confirmed medical condition.
              </p>
            </div>

            <div className="result-section">
              <h3>Detected symptoms</h3>
              {result.detected_symptoms?.length > 0 ? (
                <div className="symptom-tags">
                  {result.detected_symptoms.map((symptom) => (
                    <span className="symptom-tag" key={symptom}>
                      {symptom.replaceAll("_", " ")}
                    </span>
                  ))}
                </div>
              ) : (
                <p>No known symptoms were detected by the current dictionary.</p>
              )}
            </div>

            <div className="risk-message">
              <strong>Risk guidance</strong>
              <p>{result.risk_message}</p>
            </div>

            <p className="disclaimer">
              Important: SymptomAI is a student project and is not a medical
              diagnostic tool. Its risk rules and predictions are not
              clinically validated. Seek professional medical care when
              needed. For severe or rapidly worsening symptoms, seek urgent
              medical help.
            </p>
          </section>
        )}

        <section className="panel history-panel">
          <div className="panel-heading">
            <div>
              <h2>Recent analyses</h2>
              <p>Your saved analysis history.</p>
            </div>
            <button
              className="secondary-button"
              onClick={loadHistory}
              disabled={loading}
            >
              Refresh
            </button>
          </div>

          {history.length === 0 ? (
            <div className="empty-state">
              <span>▤</span>
              <p>No analyses yet. Submit your symptoms to get started.</p>
            </div>
          ) : (
            <div className="history-list">
              {history.map((item) => (
                <article className="history-item" key={item.id}>
                  <div className="history-item-main">
                    <h3>{item.classification}</h3>
                    <p>{item.symptoms}</p>
                    <time>
                      {item.created_at
                        ? new Date(item.created_at).toLocaleString()
                        : ""}
                    </time>
                  </div>
                  <span
                    className={`risk-badge ${(
                      item.risk_level || ""
                    ).toLowerCase()}`}
                  >
                    {item.risk_level || "UNKNOWN"}
                  </span>
                </article>
              ))}
            </div>
          )}
        </section>

        <footer className="page-footer">
          <p>SymptomAI © 2026</p>
          <p>Educational prototype · Not a medical diagnosis</p>
        </footer>
      </main>
    </div>
  );
}

export default App;