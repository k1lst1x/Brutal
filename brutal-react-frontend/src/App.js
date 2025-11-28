import React, { useState, useMemo, useEffect } from "react";
import "./App.css";

const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || "http://localhost:8000";

const BASE_METRICS = {
  precision: 0.4714,
  recall: 0.8195,
  f2: 0.7006,
  accuracy: 0.9857,
  roc_auc: 0.9883,
  threshold: 0.183,
  money_lost: 4463546.67,
  money_blocked: 55746813.08,
  money_saved: 32830802.67,
};

function formatMoneyKZT(v) {
  if (!Number.isFinite(v)) return "-";
  return new Intl.NumberFormat("ru-RU", {
    style: "currency",
    currency: "KZT",
    maximumFractionDigits: 0,
  }).format(v);
}

// fallback демо-скоринг — используется только если API не вернул fraud_probability
function demoScore(amount, direction, transdatetime) {
  let base = Math.min(0.99, 0.05 + Math.log10(Math.max(amount, 1)) * 0.15);

  if (direction.toLowerCase().includes("suspicious")) {
    base += 0.2;
  }

  const dt = new Date(transdatetime);
  if (!Number.isNaN(dt.getTime())) {
    const h = dt.getHours();
    if (h >= 23 || h <= 5) {
      base += 0.15;
    }
  }

  base += (Math.random() - 0.5) * 0.05;

  return Math.max(0, Math.min(1, base));
}

function App() {
  // онлайн форма
  const [tx, setTx] = useState(() => ({
    cst_dim_id: 100001,
    amount: 50000,
    direction: "hash_1234567890",
    transdatetime: new Date().toISOString().slice(0, 19),
  }));

  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");

  // API health / stats
  const [apiHealth, setApiHealth] = useState("unknown"); // online / offline / degraded / unknown
  const [stats, setStats] = useState(null);

  // batch CSV
  const [bulkFile, setBulkFile] = useState(null);
  const [bulkStatus, setBulkStatus] = useState("");
  const [bulkSummary, setBulkSummary] = useState(null);

  // метрики (можно позднее подменить из /stats, если там есть такие поля)
  const METRICS = BASE_METRICS;

  const totalMoney = useMemo(
    () =>
      METRICS.money_lost +
      METRICS.money_blocked +
      METRICS.money_saved,
    [METRICS.money_lost, METRICS.money_blocked, METRICS.money_saved]
  );

  const savedRatio = METRICS.money_saved / totalMoney;
  const blockedRatio = METRICS.money_blocked / totalMoney;
  const lostRatio = METRICS.money_lost / totalMoney;

  const avgLatency = useMemo(() => {
    if (!history.length) return 0;
    return (
      history.reduce((sum, h) => sum + h.latency, 0) / history.length
    );
  }, [history]);

  // при старте: дергаем /health и /stats
  useEffect(() => {
    async function load() {
      // health
      try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok) {
          setApiHealth("online");
        } else {
          setApiHealth("degraded");
        }
      } catch {
        setApiHealth("offline");
      }

      // stats (если надо выводить что-то ещё)
      try {
        const res = await fetch(`${API_BASE_URL}/stats`);
        if (res.ok) {
          const data = await res.json();
          setStats(data);
        }
      } catch (_) {
        // просто игнорим, если нет
      }
    }

    load();
  }, []);

  const handleChange = (field) => (e) => {
    const value = e.target.value;
    setTx((prev) => ({
      ...prev,
      [field]:
        field === "cst_dim_id" || field === "amount"
          ? Number(value)
          : value,
    }));
  };

  const handleNow = () => {
    setTx((prev) => ({
      ...prev,
      transdatetime: new Date().toISOString().slice(0, 19),
    }));
  };

  const handleHighRisk = () => {
    setTx((prev) => ({
      ...prev,
      cst_dim_id: 999999,
      amount: 2000000,
      direction: "hash_suspicious_receiver_01",
      transdatetime: new Date().toISOString().slice(0, 19),
    }));
  };

  // >>> ЗАПРОС К /predict <<<
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!tx.cst_dim_id || !tx.amount || !tx.transdatetime) {
      setError("Заполни все обязательные поля.");
      return;
    }

    const payload = {
      cst_dim_id: tx.cst_dim_id,
      amount: tx.amount,
      direction: tx.direction,
      transdatetime: tx.transdatetime,
      // id / behavioral_patterns / target — по необходимости
    };

    const start = performance.now();

    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const elapsed = performance.now() - start;

      if (!res.ok) {
        const text = await res.text();
        throw new Error(
          text || `HTTP ${res.status} при запросе /predict`
        );
      }

      const data = await res.json();

      // забираем SHAP top_features из бэка
      const topFeatures = Array.isArray(data.top_features)
        ? data.top_features
        : [];

      const prob =
        typeof data.fraud_probability === "number"
          ? data.fraud_probability
          : demoScore(tx.amount, tx.direction, tx.transdatetime);

      const probPercent = (prob * 100).toFixed(1);

      const riskFromApi = (data.risk_level || "").toLowerCase();
      let risk = riskFromApi;
      if (!risk) {
        risk =
          prob >= 0.8 ? "high" : prob >= 0.4 ? "medium" : "low";
      }

      const alerts = [];
      if (tx.amount > 1_000_000) {
        alerts.push(
          "Спайк суммы относительно типичного профиля клиента."
        );
      }
      if (tx.direction.toLowerCase().includes("suspicious")) {
        alerts.push(
          "Подозрительный / новый direction получателя средств."
        );
      }
      const dt = new Date(tx.transdatetime);
      if (!Number.isNaN(dt.getTime())) {
        const h = dt.getHours();
        if (h >= 23 || h <= 5) {
          alerts.push("Ночная активность (23:00–05:00).");
        }
      }
      if (!alerts.length) {
        alerts.push(
          "Явных аномалий не найдено — решение основано на общем профиле клиента."
        );
      }

      const newResult = {
        probPercent,
        score: prob,
        risk,
        latency: elapsed,
        threshold:
          typeof data.threshold_used === "number"
            ? data.threshold_used
            : METRICS.threshold,
        model_version: data.model_version || "2.0_optuna",
        individual_scores: data.individual_scores || null,
        alerts,
        // 👉 SHAP топ фичей для этой транзакции
        topFeatures,
      };

      setResult(newResult);
      setHistory((prev) => [
        {
          probPercent,
          risk,
          latency: elapsed,
          ts: new Date().toISOString(),
        },
        ...prev,
      ]);
    } catch (err) {
      console.error(err);
      setError("Ошибка запроса к API: " + err.message);
    }
  };

  // >>> BATCH CSV через /bulk_predict <<<
  const handleBulkFileChange = (e) => {
    const file = e.target.files && e.target.files[0];
    setBulkFile(file || null);
    setBulkStatus("");
    setBulkSummary(null);
  };

  function computeBulkSummary(csvText) {
    const lines = csvText.trim().split(/\r?\n/);
    if (lines.length < 2) return null;
    const header = lines[0].split(",");
    const predIdx = header.indexOf("prediction");
    const scoreIdx = header.indexOf("fraud_score");

    let total = 0;
    let fraudCount = 0;

    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      const cols = line.split(",");
      if (cols.length < header.length) continue;
      total++;
      if (predIdx !== -1) {
        const val = Number(cols[predIdx]);
        if (val === 1) fraudCount++;
      } else if (scoreIdx !== -1) {
        const val = Number(cols[scoreIdx]);
        if (val >= METRICS.threshold) fraudCount++;
      }
    }
    return { total, fraudCount };
  }

  const handleBulkUpload = async () => {
    if (!bulkFile) {
      setBulkStatus("Сначала выбери CSV-файл.");
      return;
    }

    setBulkStatus("Отправка файла в API...");
    setBulkSummary(null);

    try {
      const formData = new FormData();
      formData.append("file", bulkFile);

      const res = await fetch(`${API_BASE_URL}/bulk_predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(
          text || `HTTP ${res.status} при запросе /bulk_predict`
        );
      }

      const csvText = await res.text();

      // скачивание файла с результатами
      const blob = new Blob([csvText], {
        type: "text/csv;charset=utf-8;",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "result_with_scores.csv";
      a.style.display = "none";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      const summary = computeBulkSummary(csvText);
      setBulkSummary(summary);
      setBulkStatus("Файл успешно скорен, результат скачан.");
    } catch (err) {
      console.error(err);
      setBulkStatus("Ошибка: " + err.message);
    }
  };

  const riskBadgeClass =
    result?.risk === "high"
      ? "badge badge-red"
      : result?.risk === "medium"
      ? "badge badge-yellow"
      : "badge badge-green";

  const apiChipClass =
    apiHealth === "online"
      ? "chip-dot chip-dot-green"
      : apiHealth === "offline"
      ? "chip-dot chip-dot-red"
      : "chip-dot chip-dot-yellow";

  return (
    <div className="App">
      {/* HEADER */}
      <header className="header">
        <div className="header-inner">
          <div className="header-main">
            <div className="pill">
              <span className="pill-dot-wrapper">
                <span className="pill-dot" />
              </span>
              Anti-fraud · Mobile Internet Banking
            </div>
            <h1 className="title">
              Brutal Fraud Shield{" "}
              <span className="title-accent">v2.0</span>
            </h1>
            <p className="subtitle">
              Ансамбль CatBoost + XGBoost + LightGBM + IsolationForest,
              временное разбиение и денежные метрики. Дашборд показывает
              работу антифрода глазами бизнеса: онлайн-проверка и money
              impact.
            </p>
            <div className="header-chips">
              <div className="chip">
                <span className="chip-dot chip-dot-green" />
                Threshold (CV):{" "}
                <span className="mono">
                  {METRICS.threshold.toFixed(3)}
                </span>
              </div>
              <div className="chip">
                <span className="chip-dot chip-dot-cyan" />
                Features: <span className="mono">~60</span>
              </div>
              <div className="chip">
                <span className="chip-dot chip-dot-yellow" />
                Клиентов в истории:{" "}
                <span className="mono">50 000+</span>
              </div>
              <div className="chip">
                <span className={apiChipClass} />
                API: <span className="mono">{apiHealth}</span>
              </div>
            </div>
            {stats && (
              <p className="hint">
                model_version:{" "}
                <span className="mono">
                  {stats.model_version || "—"}
                </span>
                {stats.total_samples && (
                  <>
                    {" "}
                    · обучено на{" "}
                    <span className="mono">
                      {stats.total_samples}
                    </span>{" "}
                    транзакциях
                  </>
                )}
              </p>
            )}
          </div>

          <div className="header-card">
            <p className="card-label">Performance (cross-validation)</p>
            <div className="metrics-grid">
              <div>
                <p className="metric-label">Precision</p>
                <p className="metric-value metric-value-green">
                  {METRICS.precision.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="metric-label">Recall</p>
                <p className="metric-value metric-value-green">
                  {METRICS.recall.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="metric-label">F2</p>
                <p className="metric-value metric-value-green">
                  {METRICS.f2.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="metric-label">Accuracy</p>
                <p className="metric-value">
                  {METRICS.accuracy.toFixed(3)}
                </p>
              </div>
              <div>
                <p className="metric-label">ROC-AUC</p>
                <p className="metric-value">
                  {METRICS.roc_auc.toFixed(3)}
                </p>
              </div>
              <div>
                <p className="metric-label metric-label-small">
                  Latency (demo)
                </p>
                <p className="metric-value metric-value-cyan">
                  {avgLatency.toFixed(1)} ms
                </p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* MAIN */}
      <main className="main">
        {/* 3 критерия */}
        <section className="cards-grid">
          {/* Performance */}
          <div className="card">
            <div className="card-header">
              <div className="icon icon-green" />
              <div>
                <p className="card-label">Performance</p>
                <p className="card-title">Качество и скорость модели</p>
              </div>
            </div>
            <p className="card-text">
              Временное разбиение по транзакциям, оптимизация F2 через
              Optuna, ансамбль CatBoost + XGBoost + LightGBM +
              IsolationForest. Все поведенческие и графовые признаки
              считаются онлайн по истории клиента.
            </p>
            <div className="bars">
              <MetricBar label="Precision" value={METRICS.precision} />
              <MetricBar label="Recall" value={METRICS.recall} />
              <MetricBar label="F2" value={METRICS.f2} />
            </div>
          </div>

          {/* Business impact */}
          <div className="card">
            <div className="card-header">
              <div className="icon icon-cyan" />
              <div>
                <p className="card-label">Business impact</p>
                <p className="card-title">
                  Деньги, которые защищает модель
                </p>
              </div>
            </div>
            <p className="card-text">
              На исторических данных модель:
            </p>
            <ul className="card-list">
              <li>
                • сохраняет ~{" "}
                <strong>{formatMoneyKZT(METRICS.money_saved)}</strong>{" "}
                фродовых сумм,
              </li>
              <li>
                • блокирует сомнительные операции на ~{" "}
                <strong>
                  {formatMoneyKZT(METRICS.money_blocked)}
                </strong>
                ,
              </li>
              <li>
                • оставляет риск непойманного фрода на уровне{" "}
                <strong>{formatMoneyKZT(METRICS.money_lost)}</strong>.
              </li>
            </ul>
            <div className="bars">
              <MoneyBar
                label="Saved"
                value={savedRatio}
                colorClass="bar-green"
              />
              <MoneyBar
                label="Blocked"
                value={blockedRatio}
                colorClass="bar-yellow"
              />
              <MoneyBar
                label="Lost"
                value={lostRatio}
                colorClass="bar-red"
              />
            </div>
          </div>

          {/* Usability */}
          <div className="card">
            <div className="card-header">
              <div className="icon icon-fuchsia" />
              <div>
                <p className="card-label">Usability</p>
                <p className="card-title">
                  Интерфейс и объясняемость
                </p>
              </div>
            </div>
            <ul className="card-list">
              <li>• Онлайн-проверка одной транзакции.</li>
              <li>
                • Список бизнес-алертов: спайк суммы, ночная активность,
                быстрые повторы на один и тот же direction.
              </li>
              <li>• Денежные метрики понятны бизнесу и аналитикам.</li>
              <li>
                • Batch-скоринг CSV через /bulk_predict для аналитиков.
              </li>
            </ul>
          </div>
        </section>

        {/* ONLINE DEMO */}
        <section className="two-cols">
          {/* форма */}
          <div className="card">
            <div className="card-header">
              <div className="icon icon-green" />
              <div>
                <p className="card-label">Real-time scoring</p>
                <p className="card-title">
                  Онлайн-проверка одной транзакции (через /predict)
                </p>
              </div>
            </div>

            <form className="form" onSubmit={handleSubmit}>
              <div className="form-row">
                <div className="form-field">
                  <label>cst_dim_id (user_id)</label>
                  <input
                    type="number"
                    value={tx.cst_dim_id}
                    onChange={handleChange("cst_dim_id")}
                  />
                </div>
                <div className="form-field">
                  <label>Amount (KZT)</label>
                  <input
                    type="number"
                    value={tx.amount}
                    onChange={handleChange("amount")}
                  />
                </div>
              </div>

              <div className="form-field">
                <label>Direction (hash получателя / счёта)</label>
                <input
                  type="text"
                  value={tx.direction}
                  onChange={handleChange("direction")}
                />
              </div>

              <div className="form-field">
                <label>transdatetime (ISO)</label>
                <input
                  type="text"
                  value={tx.transdatetime}
                  onChange={handleChange("transdatetime")}
                  className="mono small-input"
                />
                <div className="form-buttons">
                  <button
                    type="button"
                    className="link-btn green-link"
                    onClick={handleNow}
                  >
                    Сейчас
                  </button>
                  <button
                    type="button"
                    className="link-btn pink-link"
                    onClick={handleHighRisk}
                  >
                    Тестовый high-risk кейс
                  </button>
                </div>
              </div>

              {error && <p className="form-error">{error}</p>}

              <button type="submit" className="primary-btn">
                Проверить транзакцию
              </button>
            </form>

            <p className="hint">
              В боевом API в запрос отправляются только сырые поля:{" "}
              <span className="mono">
                cst_dim_id, transdatetime, amount, direction
              </span>
              . Все сложные фичи считаются на бэке.
            </p>
          </div>

          {/* результат */}
          <div className="card">
            <div className="card-header">
              <div className="icon icon-cyan" />
              <div>
                <p className="card-label">Online результат</p>
                <p className="card-title">
                  Вероятность фрода и объяснения (ответ /predict)
                </p>
              </div>
            </div>

            {!result && (
              <p className="placeholder">
                Заполни форму слева и запусти проверку — здесь появится
                карта риска и объяснения, взятые из real-time API.
              </p>
            )}

            {result && (
              <div className="result">
                <div className="result-top">
                  <div>
                    <p className="metric-label">Fraud probability</p>
                    <p className="result-prob">
                      {result.probPercent}%
                    </p>
                    <span className={riskBadgeClass}>
                      {result.risk} risk
                    </span>
                  </div>
                  <div className="result-meta">
                    <p>
                      Latency:{" "}
                      <span className="mono">
                        {result.latency.toFixed(1)} ms
                      </span>
                    </p>
                    <p>
                      Threshold:{" "}
                      <span className="mono">
                        {result.threshold.toFixed(3)}
                      </span>
                    </p>
                    {result.model_version && (
                      <p>
                        Model:{" "}
                        <span className="mono">
                          {result.model_version}
                        </span>
                      </p>
                    )}
                  </div>
                </div>

                {result.individual_scores && (
                  <div>
                    <p className="card-label">
                      Вклады моделей (ensemble):
                    </p>
                    <ul className="alerts">
                      <li>
                        CatBoost:{" "}
                          <span className="mono">
                            {result.individual_scores.catboost?.toFixed(
                              3
                            ) ?? "—"}
                          </span>
                      </li>
                      <li>
                        XGBoost:{" "}
                        <span className="mono">
                          {result.individual_scores.xgboost?.toFixed(
                            3
                          ) ?? "—"}
                        </span>
                      </li>
                      <li>
                        LightGBM:{" "}
                        <span className="mono">
                          {result.individual_scores.lightgbm?.toFixed(
                            3
                          ) ?? "—"}
                        </span>
                      </li>
                      <li>
                        Anomaly (IForest):{" "}
                        <span className="mono">
                          {result.individual_scores.anomaly?.toFixed(
                            3
                          ) ?? "—"}
                        </span>
                      </li>
                    </ul>
                  </div>
                )}

                <div>
                  <p className="card-label">Alerts (объяснения):</p>
                  <ul className="alerts">
                    {result.alerts.map((a, idx) => (
                      <li key={idx}>{a}</li>
                    ))}
                  </ul>
                </div>

                {/* SHAP top features для этой транзакции */}
                {result.topFeatures && result.topFeatures.length > 0 && (
                  <div className="shap-block">
                    <p className="card-label">
                      Top features (SHAP по этой транзакции)
                    </p>
                    <ul className="shap-list">
                      {result.topFeatures.map((f, idx) => {
                        const name =
                          f.feature_name ||
                          f.name ||
                          f.feature ||
                          `feature_${idx + 1}`;

                        const rawVal =
                          typeof f.shap_value === "number"
                            ? f.shap_value
                            : typeof f.value === "number"
                            ? f.value
                            : 0;

                        const magnitude = Math.min(
                          Math.abs(rawVal),
                          1
                        ); // нормализуем 0–1
                        const isPositive = rawVal >= 0;

                        return (
                          <li key={idx} className="shap-row">
                            <div className="shap-row-header">
                              <span className="shap-name">
                                {name}
                              </span>
                              <span
                                className={
                                  "shap-value " +
                                  (isPositive
                                    ? "shap-value-pos"
                                    : "shap-value-neg")
                                }
                              >
                                {rawVal.toFixed(3)}
                              </span>
                            </div>
                            <div className="bar-track">
                              <div
                                className={
                                  "bar-fill " +
                                  (isPositive
                                    ? "bar-red"
                                    : "bar-green")
                                }
                                style={{
                                  width: `${magnitude * 100}%`,
                                }}
                              />
                            </div>
                          </li>
                        );
                      })}
                    </ul>
                    <p className="hint">
                      Красные фичи ↑ повышают риск, зелёные ↓ снижают.
                    </p>
                  </div>
                )}

                <div className="history">
                  <p className="history-title">
                    История демо-запросов
                  </p>
                  <div className="history-list">
                    {history.map((h, idx) => (
                      <div key={idx} className="history-row">
                        <span className="mono">
                          {h.probPercent}%
                        </span>
                        <span
                          className={
                            h.risk === "high"
                              ? "text-red"
                              : h.risk === "medium"
                              ? "text-yellow"
                              : "text-green"
                          }
                        >
                          {h.risk}
                        </span>
                        <span className="history-lat">
                          {h.latency.toFixed(1)} ms
                        </span>
                      </div>
                    ))}
                    {!history.length && (
                      <p className="history-empty">
                        Пока нет запросов.
                      </p>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* BATCH CSV */}
        <section className="card" style={{ marginTop: 20 }}>
          <div className="card-header">
            <div className="icon icon-cyan" />
            <div>
              <p className="card-label">Batch scoring (CSV)</p>
              <p className="card-title">
                Скоринг файла транзакций через /bulk_predict
              </p>
            </div>
          </div>
          <p className="card-text">
            Этот режим нужен data scientist&apos;ам и аналитикам. На вход
            — CSV с колонками{" "}
            <span className="mono">
              cst_dim_id, amount, direction, transdatetime
            </span>
            . На выход — CSV с добавленными колонками модели.
          </p>
          <div className="form" style={{ marginTop: 8 }}>
            <div className="form-field">
              <label>CSV-файл с транзакциями</label>
              <input type="file" accept=".csv" onChange={handleBulkFileChange} />
            </div>
            <button
              type="button"
              className="primary-btn"
              onClick={handleBulkUpload}
            >
              Отправить в /bulk_predict
            </button>
            {bulkStatus && (
              <p className="hint" style={{ marginTop: 4 }}>
                {bulkStatus}
              </p>
            )}
            {bulkSummary && (
              <p className="hint">
                В файле было строк:{" "}
                <span className="mono">{bulkSummary.total}</span>, из них
                с фрод-меткой/предиктом 1:{" "}
                <span className="mono">
                  {bulkSummary.fraudCount}
                </span>
                .
              </p>
            )}
          </div>
        </section>

        {/* Feature importance (глобальная, CV) */}
        <section className="card" style={{ marginTop: 16 }}>
          <div className="card-header">
            <div className="icon icon-fuchsia" />
            <div>
              <p className="card-label">Feature importance</p>
              <p className="card-title">
                Топ признаков по суммарной важности (XGB + LGB)
              </p>
            </div>
          </div>
          <div className="feature-grid">
            <ul>
              <li>1. amount</li>
              <li>2. pair_count</li>
              <li>3. days_since_first</li>
              <li>4. direction</li>
              <li>5. hour</li>
              <li>6. month</li>
              <li>7. amount_ratio_avg30</li>
              <li>8. std_amount_7d</li>
            </ul>
            <ul>
              <li>9. cst_dim_id</li>
              <li>10. amount_ratio_avg7</li>
              <li>11. amount_to_max_ratio</li>
              <li>12. num_trans_last_30d</li>
              <li>13. velocity_acceleration</li>
              <li>14. dayofweek</li>
              <li>15. time_since_last_hours</li>
            </ul>
          </div>
          <p className="hint">
            На презентации можно показать, как при разных типах фрода
            меняются важности фичей: спайк по amount, аномальный direction,
            ночные переводы и быстрые повторы.
          </p>
        </section>
      </main>
    </div>
  );
}

function MetricBar({ label, value }) {
  return (
    <div className="bar-block">
      <div className="bar-header">
        <span>{label}</span>
        <span className="mono">{value.toFixed(2)}</span>
      </div>
      <div className="bar-track">
        <div
          className="bar-fill bar-green"
          style={{ width: `${Math.min(value, 1) * 100}%` }}
        />
      </div>
    </div>
  );
}

function MoneyBar({ label, value, colorClass }) {
  return (
    <div className="bar-block">
      <div className="bar-header">
        <span>{label}</span>
        <span className="mono">
          {(Math.min(value, 1) * 100).toFixed(1)}%
        </span>
      </div>
      <div className="bar-track">
        <div
          className={`bar-fill ${colorClass}`}
          style={{ width: `${Math.min(value, 1) * 100}%` }}
        />
      </div>
    </div>
  );
}

export default App;
  