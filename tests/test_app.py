"""Smoke tests: the app renders on the bundled data without exceptions."""
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_app_runs_without_exceptions():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 5


def test_demo_data_contract():
    df = pd.read_csv(ROOT / "data" / "demo_invoices.csv")
    required = {"invoice_id", "customer_id", "invoice_date", "due_date", "invoice_amount",
                "amount_paid", "outstanding_amount", "snapshot_date"}
    assert required <= set(df.columns)
    assert df["invoice_id"].is_unique
    assert (df["outstanding_amount"] >= 0).all()
