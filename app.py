"""MFI SmartCollect AI - collections prioritisation prototype (read-only, CSV-first)."""
from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="MFI SmartCollect AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
ROOT = Path(__file__).parent
DEFAULT_FILE = ROOT / "data" / "demo_invoices.csv"
REQUIRED = [
    "invoice_id", "customer_id", "invoice_date", "due_date",
    "invoice_amount", "amount_paid", "outstanding_amount", "snapshot_date",
]
DATE_COLS = ["invoice_date", "due_date", "snapshot_date", "fully_paid_date"]
NUMERIC = [
    "invoice_amount", "amount_paid", "outstanding_amount", "payment_terms_days",
    "dispute_flag", "historical_late_rate", "credit_note_amount",
]
CATEGORICAL = ["currency", "business_unit"]
TARGET = "unpaid_30_days_after_due"
LABEL_WINDOW_DAYS = 30
MAX_UPLOAD_MB = 25

TEAL = "#087E8B"
NAVY = "#152536"
AMBER = "#F2A541"
RED = "#C8453B"
GREY = "#8A99A8"
RISK_COLORS = {"High": RED, "Medium": AMBER, "Low": TEAL, "Unscored": GREY}
AGING_ORDER = ["1–30 days", "31–60 days", "61–90 days", "90+ days"]

CSS = f"""
<style>
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
.hero {{
    background: linear-gradient(120deg, {NAVY} 0%, #1D4E5F 60%, {TEAL} 100%);
    border-radius: 14px; padding: 1.4rem 1.7rem; color: #fff; margin-bottom: 1rem;
}}
.hero h1 {{ color: #fff; font-size: 1.9rem; margin: 0 0 .25rem 0; padding: 0; }}
.hero p {{ color: #D5E6EA; margin: 0; font-size: .95rem; }}
.badge {{
    display: inline-block; background: rgba(255,255,255,.16); color: #fff;
    border-radius: 999px; padding: .1rem .65rem; font-size: .75rem; margin-right: .4rem;
}}
.kpi {{
    background: #fff; border: 1px solid #E3E9EE; border-left: 4px solid {TEAL};
    border-radius: 10px; padding: .85rem 1rem; height: 100%;
}}
.kpi.alert {{ border-left-color: {RED}; }}
.kpi .label {{ font-size: .78rem; color: #5B6B7B; text-transform: uppercase; letter-spacing: .04em; }}
.kpi .value {{ font-size: 1.65rem; font-weight: 700; color: {NAVY}; line-height: 1.25; }}
.kpi .sub {{ font-size: .78rem; color: #7A8896; }}
.footer {{ color: #7A8896; font-size: .8rem; margin-top: 2rem; border-top: 1px solid #E3E9EE; padding-top: .8rem; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data loading and preparation
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def prepare_data(raw: pd.DataFrame, shift_to_today: bool = False) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if shift_to_today and df["snapshot_date"].notna().any():
        delta = pd.Timestamp(date.today()) - df["snapshot_date"].max().normalize()
        for c in DATE_COLS:
            if c in df.columns:
                df[c] = df[c] + delta

    for c in NUMERIC:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL:
        if c not in df.columns:
            df[c] = "Unknown"
        df[c] = df[c].astype("string").fillna("Unknown").astype(str).replace({"nan": "Unknown", "": "Unknown"})
    for c in ["invoice_id", "customer_id"]:
        df[c] = df[c].astype(str).str.strip()

    computed = df["invoice_amount"].fillna(0) - df["amount_paid"].fillna(0) - df["credit_note_amount"].fillna(0)
    df["outstanding_amount"] = df["outstanding_amount"].fillna(computed)

    df["days_past_due"] = (df["snapshot_date"] - df["due_date"]).dt.days.clip(lower=0)
    df["is_open"] = df["outstanding_amount"].fillna(0) > 0.01
    df["is_overdue"] = df["is_open"] & (df["due_date"] < df["snapshot_date"])
    df["aging_bucket"] = pd.cut(
        df["days_past_due"].fillna(0),
        bins=[-1, 0, 30, 60, 90, np.inf],
        labels=["Not overdue", *AGING_ORDER],
    ).astype(str)
    return df


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training baseline model…")
def build_model(df: pd.DataFrame):
    """Train a baseline logistic regression on invoices whose 30-day window has elapsed."""
    if TARGET not in df.columns:
        return None, None, "No target column found. Historical labels are required to train a model."

    train_df = df.copy()
    train_df[TARGET] = pd.to_numeric(train_df[TARGET], errors="coerce")
    train_df = train_df[train_df[TARGET].isin([0, 1])]
    # Only use labels whose observation window has fully elapsed at the snapshot.
    window_end = train_df["due_date"] + pd.Timedelta(days=LABEL_WINDOW_DAYS)
    train_df = train_df[window_end <= train_df["snapshot_date"]]

    if len(train_df) < 100 or train_df[TARGET].nunique() < 2:
        return None, None, (
            "At least 100 labelled invoices (with a completed 30-day window) and both "
            "target classes are needed to train the demo model."
        )

    train_df = train_df.sort_values("due_date")
    cut = int(len(train_df) * 0.8)
    train, test = train_df.iloc[:cut], train_df.iloc[cut:]
    split = "Chronological 80/20 split by due date"
    if train[TARGET].nunique() < 2 or test[TARGET].nunique() < 2:
        train, test = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df[TARGET])
        split = "Random stratified 80/20 split (chronological split had a single class)"

    features_num = [
        c for c in ["invoice_amount", "payment_terms_days", "dispute_flag", "historical_late_rate"]
        if train_df[c].notna().any()
    ]
    features_cat = [c for c in CATEGORICAL if train_df[c].nunique() > 1]
    features = features_num + features_cat
    if not features:
        return None, None, "No usable predictive features were found in the dataset."

    # Final amount_paid / outstanding_amount / fully_paid_date are deliberately excluded (leakage).
    transformers = []
    if features_num:
        transformers.append((
            "num",
            Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            features_num,
        ))
    if features_cat:
        transformers.append((
            "cat",
            Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
            features_cat,
        ))
    model = Pipeline([
        ("prep", ColumnTransformer(transformers)),
        ("clf", LogisticRegression(max_iter=1500, class_weight="balanced")),
    ])
    model.fit(train[features], train[TARGET].astype(int))

    y_test = test[TARGET].astype(int)
    probs = model.predict_proba(test[features])[:, 1]
    preds = (probs >= 0.5).astype(int)

    baseline_auc = None
    if "historical_late_rate" in features_num:
        base = test["historical_late_rate"].fillna(test["historical_late_rate"].median())
        baseline_auc = roc_auc_score(y_test, base)

    metrics = {
        "n_train": len(train), "n_test": len(test), "split": split,
        "base_rate": float(y_test.mean()),
        "roc_auc": roc_auc_score(y_test, probs),
        "avg_precision": average_precision_score(y_test, probs),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "brier": brier_score_loss(y_test, probs),
        "baseline_auc": baseline_auc,
        "features": features,
    }
    return model, metrics, None


def explain_row(model, row: pd.Series, features: list[str], top_n: int = 3) -> list[tuple[str, float]]:
    """Top logistic-regression drivers (coefficient x scaled value) for one invoice."""
    try:
        prep, clf = model.named_steps["prep"], model.named_steps["clf"]
        x = prep.transform(row[features].to_frame().T)
        x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        contrib = x[0] * clf.coef_[0]
        names = prep.get_feature_names_out()
        pretty = [n.split("__", 1)[-1].replace("_", " ") for n in names]
        order = np.argsort(-np.abs(contrib))[:top_n]
        return [(pretty[i], float(contrib[i])) for i in order if abs(contrib[i]) > 1e-6]
    except Exception:
        return []


def risk_band(p: float, high: float, medium: float) -> str:
    if pd.isna(p):
        return "Unscored"
    if p >= high:
        return "High"
    if p >= medium:
        return "Medium"
    return "Low"


def recommended_action(row: pd.Series) -> str:
    if row.get("dispute_flag", 0) == 1:
        return "Resolve/confirm dispute before collection escalation"
    if row.get("days_past_due", 0) > 60:
        return "Escalate to collections owner; confirm payment commitment"
    if row.get("days_past_due", 0) > 0:
        return "Send reminder and confirm expected payment date"
    return "Send pre-due reminder and monitor"


def kpi(label: str, value: str, sub: str = "", alert: bool = False) -> str:
    cls = "kpi alert" if alert else "kpi"
    return f'<div class="{cls}"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


def style_fig(fig, height: int = 340):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=NAVY), legend=dict(orientation="h", y=-0.15),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E8EDF1")
    return fig


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload normalized invoice CSV", type=["csv"],
                                help=f"Max {MAX_UPLOAD_MB} MB. Schema: see the About tab.")
    using_demo = uploaded is None
    if using_demo:
        raw_bytes = DEFAULT_FILE.read_bytes()
        st.caption("Using the bundled **synthetic** dataset.")
        shift_dates = st.toggle("Shift demo dates to today", value=True,
                                help="Moves every date forward so aging and overdue figures look current.")
    else:
        raw_bytes = uploaded.getvalue()
        shift_dates = False
        st.caption(f"Using `{uploaded.name}`")
        if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File is larger than {MAX_UPLOAD_MB} MB.")
            st.stop()

    st.header("Risk bands")
    high_thr = st.slider("High risk at or above", 0.50, 0.95, 0.70, 0.05, format="%.2f")
    med_thr = st.slider("Medium risk at or above", 0.10, high_thr, min(0.40, high_thr), 0.05, format="%.2f")
    st.caption("Demo thresholds only; not approved business rules.")

try:
    df = prepare_data(load_csv(raw_bytes), shift_to_today=shift_dates)
except Exception as exc:  # noqa: BLE001 - show any load problem to the user
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

snapshot_label = df["snapshot_date"].max().date().isoformat() if df["snapshot_date"].notna().any() else "Unknown"
currencies = sorted(df["currency"].unique())
single_ccy = len(currencies) == 1
ccy = currencies[0] if single_ccy and currencies[0] != "Unknown" else ""

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
      <h1>MFI SmartCollect AI</h1>
      <p>Collections prioritisation and payment-risk prediction for accounts receivable.</p>
      <p style="margin-top:.6rem">
        <span class="badge">Prototype</span><span class="badge">Read-only</span>
        <span class="badge">{'Synthetic demo data' if using_demo else 'Uploaded data'}</span>
        <span class="badge">Snapshot {snapshot_label}</span><span class="badge">{len(df):,} invoices</span>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.info("Scores are estimates, not guarantees. Verify every alert against ERP records before acting. "
        "Do not use synthetic data for real customer decisions.", icon="ℹ️")

if df["invoice_id"].duplicated().any():
    st.warning(f"{df['invoice_id'].duplicated().sum()} duplicate invoice IDs found. "
               "Confirm one row per invoice before relying on results.")
if not single_ccy:
    st.warning("Multiple currencies detected (" + ", ".join(currencies) + "). Amounts are summed "
               "without conversion; filter or convert before using totals.")

open_df = df[df["is_open"]].copy()
overdue_df = open_df[open_df["is_overdue"]].copy()
open_value = open_df["outstanding_amount"].clip(lower=0).sum()
overdue_value = overdue_df["outstanding_amount"].clip(lower=0).sum()

# Model + scoring
model, metrics, model_error = build_model(df)
scored = open_df.copy()
if model is not None:
    scored["predicted_late_risk"] = model.predict_proba(scored[metrics["features"]])[:, 1]
else:
    scored["predicted_late_risk"] = np.nan
scored["risk_band"] = scored["predicted_late_risk"].map(lambda p: risk_band(p, high_thr, med_thr))
if len(scored):
    scored["recommended_action"] = scored.apply(recommended_action, axis=1)
    max_out = max(scored["outstanding_amount"].clip(lower=0).max(), 1)
    scored["priority_score"] = (
        scored["predicted_late_risk"].fillna(0.0) * 0.55
        + np.minimum(scored["days_past_due"].fillna(0) / 90, 1) * 0.30
        + scored["outstanding_amount"].clip(lower=0) / max_out * 0.15
    )
    scored = scored.sort_values(["priority_score", "outstanding_amount"], ascending=False)
high_risk_n = int((scored["risk_band"] == "High").sum()) if len(scored) else 0

tab_over, tab_work, tab_model, tab_quality, tab_about = st.tabs(
    ["Overview", "Collections worklist", "Model", "Data quality", "About"]
)

# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Open invoices", f"{len(open_df):,}", "Positive outstanding balance"), unsafe_allow_html=True)
    c2.markdown(kpi("Open balance", f"{open_value:,.0f}", ccy or "Mixed / unknown currency"), unsafe_allow_html=True)
    c3.markdown(kpi("Overdue invoices", f"{len(overdue_df):,}",
                    f"{len(overdue_df) / max(len(open_df), 1):.0%} of open", alert=True), unsafe_allow_html=True)
    c4.markdown(kpi("Overdue balance", f"{overdue_value:,.0f}",
                    f"{overdue_value / max(open_value, 1):.0%} of open balance", alert=True), unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Overdue balance by aging")
        if len(overdue_df):
            aging = (overdue_df.groupby("aging_bucket")["outstanding_amount"].sum()
                     .reindex(AGING_ORDER, fill_value=0).reset_index())
            fig = px.bar(aging, x="aging_bucket", y="outstanding_amount",
                         labels={"aging_bucket": "Days overdue", "outstanding_amount": "Outstanding"},
                         color_discrete_sequence=[TEAL])
            st.plotly_chart(style_fig(fig), width="stretch")
        else:
            st.write("No overdue open invoices.")
    with right:
        st.subheader("Open balance by business unit")
        if len(open_df):
            unit = open_df.groupby("business_unit")["outstanding_amount"].sum().reset_index()
            fig = px.pie(unit, names="business_unit", values="outstanding_amount", hole=0.5,
                         color_discrete_sequence=[TEAL, NAVY, AMBER, GREY, RED])
            st.plotly_chart(style_fig(fig), width="stretch")
        else:
            st.write("No open invoices.")

    st.subheader("Top 10 customers by overdue balance")
    if len(overdue_df):
        top = (overdue_df.groupby("customer_id")
               .agg(overdue_balance=("outstanding_amount", "sum"), invoices=("invoice_id", "count"),
                    max_days_past_due=("days_past_due", "max"))
               .sort_values("overdue_balance", ascending=False).head(10).reset_index())
        top["overdue_balance"] = top["overdue_balance"].round(0)
        st.dataframe(
            top, width="stretch", hide_index=True,
            column_config={
                "customer_id": "Customer",
                "overdue_balance": st.column_config.NumberColumn("Overdue balance", format="localized"),
                "invoices": "Overdue invoices",
                "max_days_past_due": st.column_config.NumberColumn("Max days past due", format="%d"),
            },
        )
    else:
        st.write("No overdue customers.")

# ----------------------------------------------------------------------------
# Worklist
# ----------------------------------------------------------------------------
with tab_work:
    if not len(scored):
        st.info("No open invoices found in the selected dataset.")
    else:
        if model is None:
            st.warning((model_error or "Model unavailable.") + " Worklist is ranked by days past due and balance only.")
        w1, w2, w3, w4 = st.columns(4)
        w1.markdown(kpi("Open invoices scored", f"{len(scored):,}"), unsafe_allow_html=True)
        w2.markdown(kpi("High risk", f"{high_risk_n:,}", f"Risk ≥ {high_thr:.0%}", alert=True), unsafe_allow_html=True)
        w3.markdown(kpi("High-risk balance",
                        f"{scored.loc[scored['risk_band'] == 'High', 'outstanding_amount'].sum():,.0f}", ccy),
                    unsafe_allow_html=True)
        w4.markdown(kpi("Disputed open invoices", f"{int((scored['dispute_flag'] == 1).sum()):,}"),
                    unsafe_allow_html=True)
        st.write("")

        f1, f2, f3 = st.columns(3)
        bands = ["High", "Medium", "Low", "Unscored"]
        units = sorted(scored["business_unit"].unique().tolist())
        sel_risk = f1.multiselect("Risk band", bands, default=bands)
        sel_unit = f2.multiselect("Business unit", units, default=units)
        min_overdue = f3.number_input("Minimum days overdue", min_value=0, max_value=3650, value=0, step=5)

        view = scored[
            scored["risk_band"].isin(sel_risk)
            & scored["business_unit"].isin(sel_unit)
            & (scored["days_past_due"].fillna(0) >= min_overdue)
        ].copy()
        st.caption(f"Showing {len(view):,} of {len(scored):,} open invoices, ranked by priority score "
                   "(55% predicted risk, 30% days overdue, 15% balance).")

        cols = ["invoice_id", "customer_id", "business_unit", "currency", "outstanding_amount",
                "days_past_due", "predicted_late_risk", "risk_band", "recommended_action"]
        shown = view[cols].copy()
        shown["outstanding_amount"] = shown["outstanding_amount"].round(0)
        st.dataframe(
            shown, width="stretch", hide_index=True, height=420,
            column_config={
                "invoice_id": "Invoice", "customer_id": "Customer", "business_unit": "Unit",
                "currency": "Ccy",
                "outstanding_amount": st.column_config.NumberColumn("Outstanding", format="localized"),
                "days_past_due": st.column_config.NumberColumn("Days past due", format="%d"),
                "predicted_late_risk": st.column_config.ProgressColumn(
                    "Predicted risk", min_value=0.0, max_value=1.0, format="percent"),
                "risk_band": "Band", "recommended_action": "Suggested action",
            },
        )
        export_df = view[cols].copy()
        st.download_button("Download worklist (CSV)", export_df.to_csv(index=False).encode("utf-8"),
                           file_name="mfi_smartcollect_worklist.csv", mime="text/csv")

        st.subheader("Inspect one invoice")
        if len(view):
            sel_id = st.selectbox("Invoice", view["invoice_id"].tolist())
            r = view[view["invoice_id"] == sel_id].iloc[0]
            i1, i2, i3, i4 = st.columns(4)
            i1.metric("Outstanding", f"{r['outstanding_amount']:,.0f} {r['currency']}")
            i2.metric("Days past due", f"{int(r['days_past_due'])}" if pd.notna(r["days_past_due"]) else "N/A")
            i3.metric("Predicted risk", f"{r['predicted_late_risk']:.0%}" if pd.notna(r["predicted_late_risk"]) else "N/A")
            i4.metric("Risk band", r["risk_band"])
            st.write(f"**Suggested action:** {r['recommended_action']}")
            if model is not None:
                drivers = explain_row(model, r, metrics["features"])
                if drivers:
                    st.write("**Main drivers of this score** (relative to an average invoice):")
                    for name, val in drivers:
                        st.write(f"- {name}: {'raises' if val > 0 else 'lowers'} risk")
            st.caption("A collections owner should verify the invoice, payment allocations, credits and disputes before acting.")
        else:
            st.write("No invoices match the current filters.")

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
with tab_model:
    st.subheader("Payment prediction")
    st.caption("Target: invoice is still unpaid 30 days after its due date. Baseline logistic regression "
               "using only features known before the outcome.")
    if model is None:
        st.warning(model_error or "Model unavailable.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test ROC-AUC", f"{metrics['roc_auc']:.3f}")
        m2.metric("Average precision", f"{metrics['avg_precision']:.3f}",
                  help=f"Random guessing would score about {metrics['base_rate']:.2f} (the late-payment rate).")
        m3.metric("Precision @ 0.5", f"{metrics['precision']:.3f}")
        m4.metric("Recall @ 0.5", f"{metrics['recall']:.3f}")

        if metrics["baseline_auc"] is not None:
            gain = metrics["roc_auc"] - metrics["baseline_auc"]
            st.write(f"**Baseline comparison:** ranking by `historical_late_rate` alone gives ROC-AUC "
                     f"{metrics['baseline_auc']:.3f}; the model is {gain:+.3f} versus that baseline.")
        st.write(f"Training rows: {metrics['n_train']:,} · Test rows: {metrics['n_test']:,} · "
                 f"{metrics['split']} · Brier score {metrics['brier']:.3f} (lower is better).")
        st.write("Features used: " + ", ".join(f"`{f}`" for f in metrics["features"]))

        if len(scored):
            fig = px.histogram(scored, x="predicted_late_risk", color="risk_band", nbins=20,
                               color_discrete_map=RISK_COLORS,
                               category_orders={"risk_band": ["High", "Medium", "Low", "Unscored"]},
                               labels={"predicted_late_risk": "Predicted late-payment risk"})
            st.plotly_chart(style_fig(fig, 300), width="stretch")
        with st.expander("Caveats"):
            st.markdown(
                "- Metrics are only meaningful if labels and features are correctly time-aligned.\n"
                "- Only invoices whose 30-day window has elapsed are used for training and testing.\n"
                "- In the synthetic data, `historical_late_rate` is a noisy proxy for the hidden driver of the "
                "label, so scores look better than they would on real data.\n"
                "- Validate on a later, untouched period before any pilot."
            )

# ----------------------------------------------------------------------------
# Data quality
# ----------------------------------------------------------------------------
with tab_quality:
    st.subheader("Data quality checks")
    checks = [
        ("Unique invoice IDs", not df["invoice_id"].duplicated().any()),
        ("Non-negative invoice amounts", bool((df["invoice_amount"].dropna() >= 0).all())),
        ("Non-negative outstanding amounts", bool((df["outstanding_amount"].dropna() >= -0.01).all())),
        ("Invoice and due dates present", bool(df["due_date"].notna().all() and df["invoice_date"].notna().all())),
        ("Snapshot dates present", bool(df["snapshot_date"].notna().all())),
        ("Due date on or after invoice date", bool((df["due_date"] >= df["invoice_date"]).fillna(True).all())),
        ("Single currency", single_ccy),
    ]
    for label, passed in checks:
        st.write(("✅ " if passed else "⚠️ ") + label)
    st.caption("Basic guards only; not a substitute for reconciling totals and allocations to Sage Evolution reports.")
    with st.expander("Preview loaded data"):
        st.dataframe(df.drop(columns=["aging_bucket"]).head(200), width="stretch", hide_index=True)

# ----------------------------------------------------------------------------
# About
# ----------------------------------------------------------------------------
with tab_about:
    st.subheader("About this prototype")
    st.markdown(
        """
SmartCollect ranks open invoices by a blend of predicted late-payment risk, days overdue and balance, and
suggests a next action for each. It is **read-only** and does not connect to Sage Evolution or SQL Server.

**Required CSV columns:** `invoice_id`, `customer_id`, `invoice_date`, `due_date`, `snapshot_date`,
`invoice_amount`, `amount_paid`, `outstanding_amount`.

**Optional columns:** `fully_paid_date`, `payment_terms_days`, `dispute_flag`, `credit_note_amount`,
`currency`, `business_unit`, `historical_late_rate`, `unpaid_30_days_after_due`.

The training label `unpaid_30_days_after_due` must be reconstructed from payment-allocation history
(1 if a balance remained 30 days after the due date, 0 if settled) and only for invoices whose 30-day
window has elapsed. Every feature must have been known at prediction time.

**Before any live pilot:** agree definitions and a read-only source with Finance and IT, map real Sage fields,
reconcile to Sage reports, validate on a later holdout, and keep a human in the loop for every collection action.
        """
    )

st.markdown(
    '<div class="footer">MFI SmartCollect AI · prototype for demonstration · not a production collections '
    "decision system.</div>",
    unsafe_allow_html=True,
)
