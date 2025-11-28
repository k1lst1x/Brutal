import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time
import io
import math

# ---------- Config ----------
st.set_page_config(page_title="Fortebank AI — Fraud Dashboard", layout="wide", initial_sidebar_state="expanded")

DEFAULT_API = "http://localhost:8000"

# ---------- Helpers ----------

def call_predict(api_url: str, payload: dict):
    try:
        r = requests.post(f"{api_url.rstrip('/')}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def call_stats(api_url: str):
    try:
        r = requests.get(f"{api_url.rstrip('/')}/stats", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def call_predict_batch(api_url: str, payloads: list, timeout=120):
    try:
        r = requests.post(f"{api_url.rstrip('/')}/predict/batch", json=payloads, timeout=timeout)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, str(e)

def call_bulk_predict(api_url: str, file_bytes: bytes, filename: str = "transactions.csv", timeout=1000):
    """
    Try to POST file to /bulk_predict as multipart. Returns tuple (ok_bool, content_bytes_or_error_msg).
    """
    try:
        files = {"file": (filename, file_bytes, "text/csv")}
        r = requests.post(f"{api_url.rstrip('/')}/bulk_predict", files=files, timeout=timeout)
        r.raise_for_status()
        return True, r.content  # bytes (CSV)
    except Exception as e:
        return False, str(e)

def safe_int(v, default=0):
    if v is None:
        return default
    if isinstance(v, (int,)):
        return v
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    try:
        return int(float(v))
    except Exception:
        return default

def safe_float(v, default=0.0):
    if v is None:
        return default
    if isinstance(v, (float, int)):
        return float(v)
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return default

def safe_str(v, default=""):
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    if v is None:
        return default
    return str(v)

def parse_csv_with_fallback(uploaded_file):
    """
    Try multiple encodings and separators to load CSV robustly.
    Returns DataFrame or raises Exception.
    """
    uploaded_file.seek(0)
    content = uploaded_file.read()
    # try encodings and separators
    encodings = ["utf-8", "cp1251", "latin1"]
    separators = [",", ";", "\t"]
    last_exc = None
    for enc in encodings:
        for sep in separators:
            try:
                # pandas can read from bytes via BytesIO with correct encoding
                buf = io.BytesIO(content)
                text = buf.read().decode(enc)
                # use pandas read_csv from string with given sep
                df = pd.read_csv(io.StringIO(text), sep=sep)
                return df
            except Exception as e:
                last_exc = e
                continue
    # final fallback: try with engine='python' and no sep
    try:
        buf = io.BytesIO(content)
        text = buf.read().decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(text), engine="python")
        return df
    except Exception as e:
        raise last_exc or e

def gauge_figure(value: float, title: str = "Fraud probability"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(value * 100, 2),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'thickness': 0.35},
               'steps': [
                   {'range': [0, 40], 'color': "#16a34a"},
                   {'range': [40, 60], 'color': "#f59e0b"},
                   {'range': [60, 80], 'color': "#f97316"},
                   {'range': [80, 100], 'color': "#dc2626"}
               ]}
    ))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=280)
    return fig

def scores_bar(scores: dict):
    names = list(scores.keys())
    vals = [scores[k] * 100 for k in names]
    fig = go.Figure(go.Bar(x=names, y=vals, text=[f"{v:.1f}%" for v in vals], textposition='auto'))
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=260)
    return fig

def hist_figure(probs: pd.Series, title="Fraud probability distribution"):
    probs_clean = probs.dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=probs_clean * 100, nbinsx=40))
    percentiles = probs_clean.quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
    # add percentile lines
    for q, v in percentiles.items():
        fig.add_vline(x=v * 100, line_dash="dash", annotation_text=f"{int(q*100)}%={v:.2f}", annotation_position="top right")
    fig.update_layout(title=title, xaxis_title="Fraud probability (%)", yaxis_title="Count", margin=dict(l=10, r=10, t=35, b=10), height=350)
    return fig, percentiles

def risk_pie_figure(risks: pd.Series, title="Risk level share"):
    counts = risks.fillna("UNKNOWN").value_counts().to_dict()
    labels = list(counts.keys())
    values = list(counts.values())
    fig = go.Figure(go.Pie(labels=labels, values=values, textinfo='label+percent'))
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=35, b=10), height=350)
    return fig

def timeseries_figure(recent_df: pd.DataFrame, window_minutes: int = 60, freq_seconds: int = 30, title="Checks over time"):
    if recent_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title, margin=dict(l=10, r=10, t=35, b=10), height=300)
        return fig
    df = recent_df.copy()
    # ensure datetime
    df['parse_time'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('parse_time').sort_index()
    # resample per freq_seconds
    res = df['fraud_probability'].resample(f'{freq_seconds}S').agg(['count', 'mean']).fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=res.index, y=res['count'], name='count', yaxis='y1', opacity=0.6))
    fig.add_trace(go.Scatter(x=res.index, y=res['mean'] * 100, name='avg_prob(%)', yaxis='y2', mode='lines+markers'))
    fig.update_layout(
        title=title,
        xaxis=dict(type='date'),
        yaxis=dict(title='Count', side='left'),
        yaxis2=dict(title='Avg prob (%)', overlaying='y', side='right'),
        legend=dict(orientation='h'),
        margin=dict(l=10, r=10, t=35, b=10),
        height=350
    )
    return fig

def top_customers_table(results_df: pd.DataFrame, top_n: int = 10):
    if results_df.empty:
        return pd.DataFrame()
    df = results_df.copy()
    # prefer cst_dim_id field (or cst)
    if 'cst_dim_id' in df.columns:
        id_col = 'cst_dim_id'
    elif 'cst' in df.columns:
        id_col = 'cst'
    else:
        # try other columns
        id_col = next((c for c in df.columns if 'cust' in c.lower() or 'cst' in c.lower()), None)
    if id_col is None:
        df['_client'] = df.index.astype(str)
        id_col = '_client'
    # compute avg prob and count
    summary = df.groupby(id_col).agg(
        avg_prob=('fraud_probability', 'mean'),
        checks=('fraud_probability', 'count')
    ).reset_index().sort_values('avg_prob', ascending=False).head(top_n)
    summary['avg_prob_pct'] = (summary['avg_prob'] * 100).round(2)
    return summary[[id_col, 'checks', 'avg_prob_pct']].rename(columns={id_col: 'client_id', 'checks': 'checks', 'avg_prob_pct': 'avg_prob (%)'})

def chunked_bulk_via_batch(api_url: str, df: pd.DataFrame, chunk_size: int = 1000, progress_cb=None):
    """
    Fallback: split df into chunks and call /predict/batch for each chunk.
    Returns aggregated list of results.
    progress_cb: function(current_chunk, total_chunks)
    """
    total = len(df)
    if total == 0:
        return []
    chunks = math.ceil(total / chunk_size)
    aggregated = []
    for i in range(chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        sub = df.iloc[start:end]
        payloads = []
        for _, r in sub.iterrows():
            payloads.append({
                "cst_dim_id": safe_int(r.get('cst_dim_id') or r.get('customer_id') or r.get('cst')),
                "transdatetime": safe_str(r.get('transdatetime') or r.get('timestamp')) or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "amount": safe_float(r.get('amount') or r.get('amt')),
                "direction": safe_str(r.get('direction')) or 'card_transfer'
            })
        ok, res = call_predict_batch(api_url, payloads, timeout=180)
        if not ok:
            raise RuntimeError(f"Batch chunk {i+1}/{chunks} failed: {res}")
        aggregated.extend(res)
        if progress_cb:
            progress_cb(i+1, chunks)
    return aggregated

# ---------- Sidebar ----------
with st.sidebar:
    st.title("Fortebank AI — Brutal")
    api_url = st.text_input("FastAPI URL", value=DEFAULT_API)
    st.markdown("---")
    st.markdown("**Quick examples**")
    example = st.selectbox("Choose example transaction", [
        "Small transfer (LOW)",
        "Night big transfer (HIGH)",
        "Huge spike (CRITICAL)",
        "Custom"
    ])
    if example == "Small transfer (LOW)":
        sample = {"cst_dim_id": 1001, "transdatetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "amount": 1200.0, "direction": "card_payment"}
    elif example == "Night big transfer (HIGH)":
        sample = {"cst_dim_id": 42, "transdatetime": (datetime.now().replace(hour=2)).strftime("%Y-%m-%d %H:%M:%S"), "amount": 25000.0, "direction": "card_transfer"}
    elif example == "Huge spike (CRITICAL)":
        sample = {"cst_dim_id": 7, "transdatetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "amount": 999999.0, "direction": "card_transfer"}
    else:
        sample = None

    st.markdown("---")
    st.markdown("**Batch / Simulation**")
    uploaded_file = st.file_uploader("Upload CSV with transactions", type=["csv"])
    simulate = st.checkbox("Run fake stream simulator", value=False)
    if simulate:
        sim_rate = st.slider("Transactions per second", 0.2, 5.0, 1.0)

    st.markdown("---")
    st.caption("Built for Fortebank AI hackathon — Brutal ensemble + IsolationForest")

# ---------- Main layout ----------
col1, col2 = st.columns((1.2, 2))

with col1:
    st.header("Realtime transaction check")

    with st.form(key='predict_form'):
        st.subheader("Input transaction")
        cst_dim_id = st.number_input("Customer ID (cst_dim_id)", min_value=1, value=1234)
        transdatetime = st.text_input("transdatetime", value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        amount = st.number_input("amount", min_value=0.0, value=50000.0, step=1.0)
        direction = st.selectbox("direction", options=["card_transfer", "card_payment", "cash_withdrawal", "p2p"])
        submit = st.form_submit_button("Send to model")

    if submit or (sample and example != "Custom" and st.button("Use example")):
        if not sample:
            payload = {"cst_dim_id": int(cst_dim_id), "transdatetime": transdatetime, "amount": float(amount), "direction": direction}
        else:
            payload = sample

        with st.spinner("Sending to API..."):
            res = call_predict(api_url, payload)

        if res.get('error'):
            st.error(f"API error: {res['error']}")
        else:
            # Summary card
            st.markdown("### Result")
            risk = res.get('risk_level', 'UNKNOWN')
            prob = res.get('fraud_probability', 0.0)
            processing_time = res.get('processing_time_ms', 0.0)
            model_version = res.get('model_version', 'n/a')

            # write single check to recent_checks (so metrics update)
            if 'recent_checks' not in st.session_state:
                st.session_state.recent_checks = []
            st.session_state.recent_checks.insert(0, {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'cst': payload.get('cst_dim_id'),
                'fraud_probability': float(res.get('fraud_probability', 0.0)) if not res.get('error') else None,
                'risk_level': res.get('risk_level', 'ERR'),
                'processing_time_ms': float(res.get('processing_time_ms', 0.0)) if not res.get('error') else None,
                'model_version': res.get('model_version', 'n/a')
            })

            rcol1, rcol2 = st.columns([1, 1])
            with rcol1:
                st.plotly_chart(gauge_figure(prob), use_container_width=True)
            with rcol2:
                st.metric("Risk level", f"{risk}")
                st.metric("Processing time (ms)", f"{processing_time:.2f}")
                st.write("Model version:", model_version)

            # Individual scores
            if 'individual_scores' in res:
                scores = res['individual_scores']
                st.subheader("Individual model scores")
                st.plotly_chart(scores_bar(scores), use_container_width=True)

            # Alerts
            if res.get('alerts'):
                st.subheader("Alerts")
                for a in res['alerts']:
                    if a.startswith('🚨') or 'CRITICAL' in a or risk == 'CRITICAL':
                        st.error(a)
                    else:
                        st.warning(a)

            # Raw response
            with st.expander("Raw response JSON"):
                st.json(res)

with col2:
    st.header("Dashboard & Stats")

    stats = call_stats(api_url)
    if stats.get('error'):
        st.error(f"Could not fetch stats: {stats['error']}")
    else:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Customers in history", stats.get('total_customers_in_history', 0))
        s2.metric("Model version", stats.get('model_version', 'n/a'))
        s3.metric("Threshold", f"{stats.get('threshold', 0):.4f}")
        s4.metric("Num features", stats.get('num_features', 0))

    st.markdown("---")
    st.subheader("Recent checks")
    if 'recent_checks' not in st.session_state:
        st.session_state.recent_checks = []

    # If CSV uploaded, show batch preview and allow sending
    df = None
    if uploaded_file is not None:
        try:
            df = parse_csv_with_fallback(uploaded_file)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            df = None

    if df is not None:
        st.markdown(f"**Preview — {len(df)} rows**")
        st.dataframe(df.head(20))

        st.markdown("### Batch send options")
        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("Send batch via /bulk_predict"):
                with st.spinner("Uploading CSV to /bulk_predict..."):
                    uploaded_file.seek(0)
                    file_bytes = uploaded_file.read()
                    ok, res = call_bulk_predict(api_url, file_bytes, filename=getattr(uploaded_file, 'name', 'transactions.csv'))
                    if ok:
                        st.success("bulk_predict returned result — ready to download")
                        st.session_state._bulk_result = res
                        st.session_state._bulk_name = f"bulk_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    else:
                        st.error(f"bulk_predict failed: {res}")
                        st.info("Falling back to chunked /predict/batch approach...")
                        try:
                            progress = st.progress(0)
                            status_text = st.empty()
                            def progress_cb(curr, total):
                                progress.progress(int(curr / total * 100))
                                status_text.text(f"Chunk {curr}/{total}")
                            aggregated = chunked_bulk_via_batch(api_url, df, chunk_size=1000, progress_cb=progress_cb)
                            out_df = pd.DataFrame(aggregated)
                            buf = io.BytesIO()
                            out_df.to_csv(buf, index=False)
                            st.session_state._bulk_result = buf.getvalue()
                            st.session_state._bulk_name = f"bulk_chunked_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            st.success("Chunked batch finished — ready to download")
                            # push sample aggregated into recent_checks
                            for r in aggregated[::-1]:
                                st.session_state.recent_checks.insert(0, {
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'cst': r.get('cst_dim_id') or r.get('cst') or None,
                                    'fraud_probability': r.get('fraud_probability', 0.0),
                                    'risk_level': r.get('risk_level', 'n/a'),
                                    'processing_time_ms': r.get('processing_time_ms', None),
                                    'model_version': r.get('model_version', 'n/a')
                                })
                        except Exception as e:
                            st.error(f"Chunked fallback failed: {e}")

        with col_b:
            if st.button("Send batch via /predict/batch (single request)"):
                rows = df.to_dict(orient='records')
                payloads = []
                for r in rows:
                    payloads.append({
                        "cst_dim_id": safe_int(r.get('cst_dim_id') or r.get('customer_id') or r.get('cst')),
                        "transdatetime": safe_str(r.get('transdatetime') or r.get('timestamp')) or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "amount": safe_float(r.get('amount') or r.get('amt')),
                        "direction": safe_str(r.get('direction')) or 'card_transfer'
                    })
                with st.spinner("Sending batch... this may take a while"):
                    ok, res = call_predict_batch(api_url, payloads, timeout=1000)
                    if ok:
                        st.success(f"Batch finished: {len(res)} items")
                        out_df = pd.DataFrame(res)
                        buf = io.BytesIO()
                        out_df.to_csv(buf, index=False)
                        st.session_state._bulk_result = buf.getvalue()
                        st.session_state._bulk_name = f"batch_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        # push to recent checks
                        for r in res:
                            st.session_state.recent_checks.insert(0, {
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'cst': r.get('cst_dim_id') or r.get('cst') or None,
                                'fraud_probability': r.get('fraud_probability', 0.0),
                                'risk_level': r.get('risk_level', 'n/a'),
                                'processing_time_ms': r.get('processing_time_ms', None),
                                'model_version': r.get('model_version', 'n/a')
                            })
                    else:
                        st.error(f"Batch error: {res}")

    st.markdown("---")
    st.subheader("Recent checks (live)")
    if st.session_state.recent_checks:
        df_recent = pd.DataFrame(st.session_state.recent_checks)
        st.table(df_recent.head(10))
    else:
        st.info("No checks yet — send a transaction using the form on the left")

    # Download button for last bulk result
    if st.session_state.get('_bulk_result'):
        st.markdown("---")
        st.success("Download last batch result below")
        st.download_button(
            label="Download batch result CSV",
            data=st.session_state._bulk_result,
            file_name=st.session_state._bulk_name,
            mime="text/csv"
        )

    # ----- New visualizations & metrics (based on recent_checks and last bulk result) -----
    st.markdown("---")
    st.subheader("Diagnostics & Visualizations")

    # prepare a combined results dataframe (recent checks + last bulk result if any)
    combined_df = pd.DataFrame(st.session_state.get('recent_checks', []))
    # if bulk result exists in memory, try to append it for visualizations
    if st.session_state.get('_bulk_result'):
        try:
            bulk_buf = io.BytesIO(st.session_state._bulk_result)
            bulk_df = pd.read_csv(bulk_buf)
            # normalize column names and ensure fraud_probability exists
            if 'fraud_probability' in bulk_df.columns:
                bulk_df_small = bulk_df[['fraud_probability']].copy()
            else:
                bulk_df_small = pd.DataFrame({'fraud_probability': [None] * len(bulk_df)})
            # try to get risk_level and cst_dim_id
            if 'risk_level' in bulk_df.columns:
                bulk_df_small['risk_level'] = bulk_df['risk_level']
            if 'cst_dim_id' in bulk_df.columns:
                bulk_df_small['cst_dim_id'] = bulk_df['cst_dim_id']
            # add timestamp if missing
            if 'timestamp' not in bulk_df_small.columns:
                bulk_df_small['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # append
            # convert to same schema as recent_checks
            for _, row in bulk_df_small.iterrows():
                combined_df = pd.concat([combined_df, pd.DataFrame([{
                    'timestamp': row.get('timestamp'),
                    'cst': row.get('cst_dim_id'),
                    'fraud_probability': safe_float(row.get('fraud_probability')),
                    'risk_level': row.get('risk_level', None),
                    'processing_time_ms': None,
                    'model_version': None
                }])], ignore_index=True)
        except Exception:
            # ignore bulk parsing errors for diagnostics
            pass

    # Metrics computed from combined_df
    if not combined_df.empty:
        # ensure numeric
        combined_df['fraud_probability'] = pd.to_numeric(combined_df['fraud_probability'], errors='coerce')
        combined_df['processing_time_ms'] = pd.to_numeric(combined_df.get('processing_time_ms', pd.Series([])), errors='coerce')

        # basic metrics
        avg_prob = combined_df['fraud_probability'].mean(skipna=True)
        pct_above_threshold = None
        try:
            threshold = float(stats.get('threshold', 0.0)) if not stats.get('error') else 0.0
            pct_above_threshold = (combined_df['fraud_probability'] >= threshold).mean() * 100
        except Exception:
            pct_above_threshold = None
        avg_processing = combined_df['processing_time_ms'].mean(skipna=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg fraud prob (all)", f"{(avg_prob or 0.0):.3f}")
        m2.metric("Pct >= threshold", f"{(pct_above_threshold or 0.0):.2f}%")
        m3.metric("Avg processing ms", f"{(avg_processing or 0.0):.2f}")
        m4.metric("Total checks", len(combined_df))

        # Histogram & percentiles
        hist_col, pie_col = st.columns(2)
        with hist_col:
            fig_hist, percentiles = hist_figure(combined_df['fraud_probability'].fillna(0))
            st.plotly_chart(fig_hist, use_container_width=True)
            st.write("Percentiles (25/50/75/90):", ", ".join([f"{k*100:.0f}%={v:.3f}" for k, v in percentiles.items()]))
        with pie_col:
            fig_pie = risk_pie_figure(combined_df.get('risk_level', pd.Series([])))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Time-series of recent checks (if we have timestamps)
        try:
            ts_fig = timeseries_figure(combined_df, window_minutes=60, freq_seconds=30, title="Checks count & avg prob over time")
            st.plotly_chart(ts_fig, use_container_width=True)
        except Exception:
            pass

        # Top customers
        top_n = st.number_input("Top N customers by avg fraud prob", min_value=3, max_value=100, value=10, step=1)
        try:
            # prepare results df from combined_df: ensure columns cst and fraud_probability present
            results_df = combined_df.rename(columns={'cst': 'cst_dim_id'})
            top_table = top_customers_table(results_df[['cst_dim_id', 'fraud_probability']].dropna(), top_n)
            if not top_table.empty:
                st.subheader("Top customers by avg fraud probability")
                st.dataframe(top_table)
        except Exception:
            pass
    else:
        st.info("Not enough data for diagnostics — send transactions or upload a batch")

# ---------- Simulation / Streamer ----------
if simulate:
    st.markdown("---")
    st.subheader("Simulator")

    if 'simulator_running' not in st.session_state:
        st.session_state.simulator_running = False

    start_col, stop_col = st.columns(2)
    if start_col.button("Start simulator"):
        st.session_state.simulator_running = True
    if stop_col.button("Stop simulator"):
        st.session_state.simulator_running = False

    placeholder = st.empty()
    try:
        while st.session_state.simulator_running:
            fake = {
                "cst_dim_id": safe_int(1000 + time.time() % 1000),
                "transdatetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "amount": safe_float(abs(int(time.time() * 100) % 200000)),
                "direction": "card_transfer"
            }
            res = call_predict(api_url, fake)
            # record
            if 'recent_checks' not in st.session_state:
                st.session_state.recent_checks = []
            st.session_state.recent_checks.insert(0, {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'cst': fake['cst_dim_id'],
                'fraud_probability': float(res.get('fraud_probability', 0.0)) if not res.get('error') else None,
                'risk_level': res.get('risk_level', 'ERR'),
                'processing_time_ms': float(res.get('processing_time_ms', 0.0)) if not res.get('error') else None,
                'model_version': res.get('model_version', 'n/a')
            })

            with placeholder.container():
                c1, c2, c3 = st.columns(3)
                if not res.get('error'):
                    c1.metric('Last risk', res.get('risk_level', 'n/a'))
                    c2.metric('Last prob', f"{res.get('fraud_probability', 0.0):.3f}")
                    c3.metric('Alerts', len(res.get('alerts', [])))
                else:
                    c1.error('Simulator error')
            time.sleep(max(0.01, 1.0 / float(sim_rate)))
    except Exception as e:
        st.error(f"Simulator stopped with error: {e}")

# ---------- Footer ----------
st.markdown("---")
st.caption("Made with ❤️ for Fortebank AI Hackathon — Brutal ensemble + IsolationForest")
