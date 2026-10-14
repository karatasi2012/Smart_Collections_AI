"""Generate the synthetic demo dataset for MFI SmartCollect AI.

The output is fully synthetic and is NOT MFI Group data.

The snapshot date is fixed (not "today") so the file is reproducible. The app can
shift all dates forward at load time so the demo always looks current - see the
"Shift demo dates to today" option in the sidebar.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT_DATE = pd.Timestamp("2026-10-01")


def make_data(n: int = 1800, seed: int = 42, snapshot: pd.Timestamp = SNAPSHOT_DATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customer_n = max(100, n // 8)
    cust_ids = np.array([f"C{idx:04d}" for idx in range(1, customer_n + 1)])
    late_propensity = dict(zip(cust_ids, rng.beta(2.2, 4.5, customer_n)))

    start = snapshot - pd.Timedelta(days=730)
    rows = []
    for i in range(n):
        customer = rng.choice(cust_ids)
        # Invoices run right up to ~2 weeks before the snapshot so some are not yet due.
        invoice_date = start + pd.Timedelta(days=int(rng.integers(0, 715)))
        terms = int(rng.choice([14, 30, 45, 60], p=[0.12, 0.58, 0.20, 0.10]))
        due_date = invoice_date + pd.Timedelta(days=terms)
        amount = float(np.round(np.exp(rng.normal(np.log(42000), 0.85)) / 100) * 100)
        prop = late_propensity[customer]

        # ~10% of invoices are settled in instalments and stay partly open.
        if rng.random() < 0.10:
            payment_delay = int(max(0, rng.normal(45 + prop * 55, 25)))
            paid_fraction = float(rng.choice([0.4, 0.65, 0.8]))
        else:
            payment_delay = int(max(0, rng.normal(5 + prop * 48, 17)))
            paid_fraction = 1.0
        paid_date = due_date + pd.Timedelta(days=payment_delay)

        # Label: unpaid 30 days after due. Only defined once the 30-day window has elapsed.
        window_end = due_date + pd.Timedelta(days=30)
        if window_end <= snapshot:
            target = int(paid_date > window_end or paid_fraction < 1.0)
        else:
            target = np.nan

        if paid_date <= snapshot:
            amount_paid = amount * paid_fraction
            outstanding = amount - amount_paid
            fully_paid_date = paid_date if paid_fraction >= 1.0 else pd.NaT
        else:
            amount_paid, outstanding, fully_paid_date = 0.0, amount, pd.NaT

        dispute = int(rng.random() < (0.04 + 0.10 * prop))
        rows.append({
            "invoice_id": f"INV{i + 1:06d}",
            "customer_id": customer,
            "invoice_date": invoice_date.date().isoformat(),
            "due_date": due_date.date().isoformat(),
            "invoice_amount": round(amount, 2),
            "amount_paid": round(amount_paid, 2),
            "outstanding_amount": round(outstanding, 2),
            "snapshot_date": snapshot.date().isoformat(),
            "fully_paid_date": "" if pd.isna(fully_paid_date) else fully_paid_date.date().isoformat(),
            "payment_terms_days": terms,
            "dispute_flag": dispute,
            "credit_note_amount": 0.0,
            "currency": "KES",
            "business_unit": rng.choice(["Kenya", "Uganda", "Tanzania", "Rwanda"]),
            "unpaid_30_days_after_due": target,
            # Demo-only: a noisy proxy for customer payment behaviour known before the invoice.
            "historical_late_rate": round(float(np.clip(prop + rng.normal(0, 0.15), 0, 1)), 3),
        })

    df = pd.DataFrame(rows)
    df["unpaid_30_days_after_due"] = df["unpaid_30_days_after_due"].astype("Int64")
    return df


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "demo_invoices.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = make_data()
    data.to_csv(out, index=False)
    print(f"Wrote {out} ({len(data):,} rows)")
