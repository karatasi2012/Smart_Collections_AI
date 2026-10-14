# Deployment checklist

## Before publishing
- [ ] Only synthetic demo data is in the repository.
- [ ] No passwords, connection strings, exports or secrets in any committed file.
- [ ] `app.py`, `requirements.txt`, `.streamlit/config.toml` and `data/demo_invoices.csv` are committed at the expected paths.
- [ ] `sql/` contains `data_dictionary.md` and `normalized_views_template.sql`.
- [ ] Tests pass (`python -m pytest`) or the GitHub Actions run is green.
- [ ] Repository is private if organisational policy requires it.

## Streamlit Community Cloud
- [ ] Sign in at https://share.streamlit.io/ with GitHub.
- [ ] Create app, choose repository and branch, main file path `app.py`.
- [ ] Deploy and wait for dependency installation.
- [ ] Check all five tabs, the filters, risk-band sliders, invoice inspector and CSV download.
- [ ] Upload a copy of `data/demo_invoices.csv` through the sidebar and confirm it loads.
- [ ] Toggle "Shift demo dates to today" and confirm the ageing chart updates.

## Important
A working demo does not mean the model is validated or approved for real customer or financial data. Do not add production data or ERP credentials to a public repository. Live integration needs IT/security approval and a validated read-only reporting layer.
