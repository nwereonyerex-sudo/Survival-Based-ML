"""Stage 9 (added beyond the original §11 8-stage layout, per the user's deployment request —
see CLAUDE.md §15.5/§17) — builds the static results dashboard for GitHub Pages.

Reads the actual Stage 3/7/8 output tables (never hand-transcribed) and renders a single
static HTML page to `docs/index.html`, with the Stage 4/7/8 figures copied to `docs/figures/`.
GitHub Pages serves static files only, so this is plain HTML/CSS — no server, no framework,
no external requests (self-contained, works offline once downloaded).

Design follows the project's dataviz skill: system-sans typography, tabular-nums for the
results table, light/dark mode via `prefers-color-scheme` (not a click toggle — there's no
viewer chrome on a static page to host one), and significance is marked by bold text + a
footnote symbol rather than colour alone (a 95% CI excluding zero is the only thing flagged
this way — deliberately not framed as "good/bad" with a status colour, since a non-significant
difference is itself one of this project's genuine findings, not a failure state).
"""

import shutil
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_FIGURES = DOCS_DIR / "figures"

REPO_URL = "https://github.com/nwereonyerex-sudo/Survival-Based-ML"

MODEL_ORDER = ["coxph", "qrisk3_style", "rsf", "gbsa", "deepsurv", "deephit"]
MODEL_DISPLAY = {
    "coxph": "CoxPH",
    "qrisk3_style": "QRISK3-style",
    "rsf": "Random Survival Forest",
    "gbsa": "Gradient-Boosting Survival",
    "deepsurv": "DeepSurv",
    "deephit": "DeepHit",
}
SEX_ORDER = ["male", "female"]
SEX_DISPLAY = {"male": "Male cohort", "female": "Female cohort"}
FIGURES = [
    ("km_curve", "Kaplan–Meier survival (descriptive)"),
    ("calibration", "Calibration — decile-grouped, observed vs predicted 10-year risk"),
    ("shap_summary", "SHAP global feature importance, compared to CoxPH coefficients"),
    ("shap_dependence", "SHAP dependence — age and systolic blood pressure"),
]


def fmt(x: float, decimals: int = 3) -> str:
    return f"{x:.{decimals}f}"


def ci_excludes_zero(low: float, high: float) -> bool:
    return not (low <= 0 <= high)


def build_results_rows(summary: pd.DataFrame) -> str:
    rows = []
    for sex in SEX_ORDER:
        rows.append(f'<tr class="sex-header"><td colspan="4">{SEX_DISPLAY[sex]}</td></tr>')
        sub = summary[summary["sex"] == sex].set_index("model")
        for model in MODEL_ORDER:
            r = sub.loc[model]
            cindex_cell = f"{fmt(r['cindex'])} ({fmt(r['cindex_ci_low'])}–{fmt(r['cindex_ci_high'])})"
            uno_cell = f"{fmt(r['uno_cindex'])} ({fmt(r['uno_cindex_ci_low'])}–{fmt(r['uno_cindex_ci_high'])})"
            brier_suffix = "" if r["brier_score_type"] == "integrated" else "¹"
            brier_cell = f"{fmt(r['brier'], 4)} ({fmt(r['brier_ci_low'], 4)}–{fmt(r['brier_ci_high'], 4)}){brier_suffix}"

            if model in ("coxph", "qrisk3_style"):
                delta_cell = "— (baseline)"
            else:
                sig = ci_excludes_zero(r["delta_cindex_vs_coxph_ci_low"],
                                        r["delta_cindex_vs_coxph_ci_high"])
                sign = "+" if r["delta_cindex_vs_coxph"] >= 0 else ""
                marker = " †" if sig else ""
                cls = " class=\"significant\"" if sig else ""
                delta_cell = (f"<span{cls}>{sign}{fmt(r['delta_cindex_vs_coxph'])} vs CoxPH"
                               f"{marker}</span>")

            rows.append(
                "<tr>"
                f"<td>{MODEL_DISPLAY[model]}</td>"
                f"<td class=\"num\">{cindex_cell}</td>"
                f"<td class=\"num\">{uno_cell}</td>"
                f"<td class=\"num\">{brier_cell}</td>"
                "</tr>"
            )
            # second row: delta vs baselines, spans under the metrics for readability
            rows.append(
                f'<tr class="delta-row"><td></td><td colspan="3" class="num delta">'
                f'{delta_cell}</td></tr>'
            )
    return "\n".join(rows)


def build_feature_set_html(sex: str) -> str:
    df = pd.read_csv(RESULTS_TABLES / f"stage3_feature_selection_{sex}.csv")
    kept = sorted(df.loc[df["final_selected_feature_set"], "predictor"].tolist())
    items = "".join(f"<li>{p.replace('_', ' ')}</li>" for p in kept)
    return f"<ul class=\"feature-list\">{items}</ul>"


def compute_explainability_stats() -> dict:
    stats = {}
    for sex in SEX_ORDER:
        stability = pd.read_csv(RESULTS_TABLES / f"stage8_stability_{sex}.csv")
        agreement = pd.read_csv(RESULTS_TABLES / f"stage8_shap_lime_agreement_{sex}.csv")
        for model in ["rsf", "gbsa", "deepsurv", "deephit"]:
            n_total = (stability["model"] == model).sum()
            n_unstable = ((stability["model"] == model) & (~stability["robust"])).sum()
            mean_agree = agreement[f"{model}_agreement_count"].mean()
            stats[(sex, model)] = {
                "n_unstable": int(n_unstable), "n_total": int(n_total),
                "mean_agree": mean_agree,
            }
    return stats


def build_stability_summary_html(stats: dict) -> str:
    rows = []
    for sex in SEX_ORDER:
        cells = []
        for model in ["rsf", "gbsa", "deepsurv", "deephit"]:
            s = stats[(sex, model)]
            cells.append(
                f"<td>{MODEL_DISPLAY[model]}: <strong>{s['n_unstable']}/{s['n_total']}</strong> "
                f"unstable, top-3 overlap <strong>{s['mean_agree']:.2f}/3</strong></td>"
            )
        rows.append(f"<tr><td>{SEX_DISPLAY[sex]}</td>{''.join(cells)}</tr>")
    return "\n".join(rows)


def copy_figures() -> None:
    DOCS_FIGURES.mkdir(parents=True, exist_ok=True)
    for prefix, _ in FIGURES:
        for sex in SEX_ORDER:
            src = RESULTS_FIGURES / f"{prefix}_{sex}.png"
            if not src.exists():
                raise FileNotFoundError(f"{src} not found — run the full pipeline first.")
            shutil.copy2(src, DOCS_FIGURES / src.name)


def figure_section_html() -> str:
    sections = []
    for prefix, title in FIGURES:
        cards = "".join(
            f'<figure><img src="figures/{prefix}_{sex}.png" alt="{title} — {sex}">'
            f'<figcaption>{SEX_DISPLAY[sex]}</figcaption></figure>'
            for sex in SEX_ORDER
        )
        sections.append(f'<h3>{title}</h3><div class="figure-row">{cards}</div>')
    return "\n".join(sections)


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Survival-Based ML for 10-Year CVD Risk</title>
<style>
:root {{
  color-scheme: light;
  --page: #f9f9f7;
  --surface: #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --gridline: #e1e0d9;
  --border: rgba(11,11,11,0.10);
  --accent: #2a78d6;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    color-scheme: dark;
    --page: #0d0d0d;
    --surface: #1a1a19;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --gridline: #2c2c2a;
    --border: rgba(255,255,255,0.10);
    --accent: #3987e5;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--page);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.55;
}}
.wrap {{ max-width: 920px; margin: 0 auto; padding: 2.5rem 1.25rem 5rem; }}
header.hero {{ margin-bottom: 2.5rem; }}
h1 {{ font-size: 1.9rem; margin: 0 0 0.4rem; }}
.subtitle {{ color: var(--text-secondary); font-size: 1.05rem; margin: 0 0 1rem; }}
.repo-link {{ font-size: 0.9rem; }}
.repo-link a {{ color: var(--accent); text-decoration: none; }}
.repo-link a:hover {{ text-decoration: underline; }}
section {{ margin: 3rem 0; }}
h2 {{
  font-size: 1.3rem; border-bottom: 1px solid var(--gridline);
  padding-bottom: 0.5rem; margin-bottom: 1rem;
}}
h3 {{ font-size: 1.05rem; color: var(--text-secondary); margin: 1.75rem 0 0.75rem; }}
p {{ color: var(--text-secondary); }}
.ethics {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 1rem 1.25rem; font-size: 0.92rem;
}}
table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
th {{
  text-align: left; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.03em;
  color: var(--text-muted); border-bottom: 1px solid var(--gridline); padding: 0.5rem 0.6rem;
}}
td {{ padding: 0.45rem 0.6rem; border-bottom: 1px solid var(--gridline); vertical-align: top; }}
td.num {{ font-variant-numeric: tabular-nums; }}
tr.sex-header td {{
  padding-top: 1.5rem; font-weight: 600; border-bottom: 1px solid var(--text-muted);
  color: var(--text-primary);
}}
tr.delta-row td {{ border-bottom: 1px solid var(--gridline); padding-top: 0; color: var(--text-muted); font-size: 0.85rem; }}
td.delta {{ text-align: left; }}
span.significant {{ font-weight: 600; color: var(--text-primary); }}
.legend {{ font-size: 0.82rem; color: var(--text-muted); margin-top: 0.6rem; }}
.feature-list {{ columns: 2; column-gap: 2rem; font-size: 0.9rem; color: var(--text-secondary); }}
.figure-row {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
figure {{
  margin: 0; flex: 1 1 380px; background: #ffffff; border: 1px solid var(--border);
  border-radius: 8px; padding: 0.6rem; text-align: center;
}}
figure img {{ max-width: 100%; height: auto; border-radius: 4px; }}
figcaption {{ font-size: 0.8rem; color: #52514e; margin-top: 0.4rem; }}
.stability-table td {{ font-size: 0.85rem; }}
.findings li {{ margin-bottom: 0.9rem; color: var(--text-secondary); }}
footer {{
  margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid var(--gridline);
  font-size: 0.85rem; color: var(--text-muted);
}}
footer a {{ color: var(--accent); }}
code {{ font-size: 0.85em; background: var(--surface); padding: 0.1em 0.35em; border-radius: 4px; }}
</style>
</head>
<body>
<div class="wrap">

<header class="hero">
  <h1>Survival-Based ML for 10-Year Heart Attack/Stroke Risk</h1>
  <p class="subtitle">Benchmarking six sex-stratified survival models — CoxPH, a
  QRISK3-style clinical score, Random Survival Forest, Gradient-Boosting Survival Analysis,
  DeepSurv, and DeepHit — on discrimination, calibration, and explainability.</p>
  <p class="repo-link"><a href="{repo_url}">{repo_url}</a></p>
</header>

<section id="ethics">
  <div class="ethics">
    <strong>Data and ethics.</strong> Trained and evaluated exclusively on
    Burns, Richardson and Driessens (2024)'s CC0-licensed, 100%-synthetic primary-care
    dataset (doi: 10.3310/nihropenres.13651.1). No real, identifiable, or non-synthetic
    patient data was used at any stage.
  </div>
</section>

<section id="results">
  <h2>Results — six models × two sexes</h2>
  <p>Harrell's C-index and Uno's IPCW C-statistic (discrimination), and the Brier score
  (calibration), all with bootstrap 95% confidence intervals from 1,000 resamples of the held-out
  test set. The row beneath each model shows its effect size vs. the sex-specific CoxPH
  baseline.</p>
  <table>
    <thead>
      <tr><th>Model</th><th>C-index (95% CI)</th><th>Uno's C (95% CI)</th><th>Brier score (95% CI)</th></tr>
    </thead>
    <tbody>
{results_rows}
    </tbody>
  </table>
  <p class="legend">† 95% CI on the delta excludes zero (statistically distinguishable
  from the CoxPH baseline). ¹ QRISK3-style is a fixed single-horizon clinical formula, not
  a fitted survival model — its Brier score is evaluated at a single ~10-year timepoint,
  not integrated over the full follow-up like the other five models.</p>
</section>

<section id="features">
  <h2>Selected predictors per sex</h2>
  <p>Feature selection (multicollinearity screening + validation-tuned LASSO) ran
  independently per sex; the surviving set is used identically across CoxPH, RSF, GBSA,
  DeepSurv, and DeepHit for that sex. QRISK3-style uses its own fixed clinical predictor set
  regardless.</p>
  <h3>{male_label}</h3>
  {male_features}
  <h3>{female_label}</h3>
  {female_features}
</section>

<section id="figures">
  <h2>Descriptive survival, calibration, and explainability</h2>
{figure_sections}
</section>

<section id="stability">
  <h2>SHAP stability and SHAP/LIME agreement</h2>
  <p>Unstable = the predictor's SHAP-importance rank shifted by more than 2 positions across
  two independent bootstrap resamples of the training fold. Top-3 overlap = mean count of
  shared features between each patient's top-3 SHAP and top-3 LIME features (out of 3).</p>
  <table class="stability-table">
    <tbody>
{stability_rows}
    </tbody>
  </table>
</section>

<section id="findings">
  <h2>Notable findings</h2>
  <ul class="findings">
    <li><strong>QRISK3-style leads discrimination but is clearly the worst-calibrated model,
    in both sexes.</strong> Its C-index point estimate is highest in both sexes, and its
    Brier score's 95% CI does not overlap any of the other five models' in either sex — a
    genuine discrimination/calibration split, not sampling noise. The discrimination
    advantage itself is only statistically significant for males, though — for females the
    same comparison's confidence interval touches zero.</li>
    <li><strong>CoxPH is a hard baseline to beat: none of the four fitted ML/DL models
    significantly improves on it, and several significantly underperform it.</strong> RSF,
    GBSA, DeepSurv, and DeepHit never show a statistically significant <em>improvement</em>
    over CoxPH on any metric, in either sex — but for males, GBSA and DeepHit calibrate
    significantly <em>worse</em>; for females, RSF, GBSA, and DeepHit all discriminate
    significantly worse. DeepSurv is the only one of the four that never differs
    significantly from CoxPH on any metric, in either sex.</li>
    <li><strong>Tree ensembles concentrate importance on age; CoxPH and the deep models
    spread it across comorbidities.</strong> RSF and GBSA's SHAP importance is dominated by
    age alone; CoxPH's coefficients and DeepSurv/DeepHit's SHAP values distribute more evenly
    across diabetes, atrial fibrillation, family history, and treated hypertension —
    consistent in both sexes.</li>
    <li><strong>DeepSurv and DeepHit's explanations are considerably less stable than
    RSF/GBSA's, despite similar predictive performance.</strong> The majority of predictors
    for the two deep models shift rank materially across resampled training folds, and their
    SHAP/LIME agreement is markedly lower — a real explainability-vs-performance tradeoff,
    not visible from C-index alone.</li>
  </ul>
</section>

<footer>
  Full methodology, every resolved spec deviation, and the complete stage-by-stage
  implementation journal: see <a href="{repo_url}/blob/main/CLAUDE.md">CLAUDE.md</a> and
  <a href="{repo_url}/blob/main/journal/agent_journal.md">journal/agent_journal.md</a> in the
  repository. Built from the pipeline's own output tables — not hand-transcribed.
</footer>

</div>
</body>
</html>
"""


def main() -> None:
    summary = pd.read_csv(RESULTS_TABLES / "stage7_evaluation_summary.csv")
    stats = compute_explainability_stats()

    copy_figures()

    html = PAGE_TEMPLATE.format(
        repo_url=REPO_URL,
        results_rows=build_results_rows(summary),
        male_label=SEX_DISPLAY["male"],
        female_label=SEX_DISPLAY["female"],
        male_features=build_feature_set_html("male"),
        female_features=build_feature_set_html("female"),
        figure_sections=figure_section_html(),
        stability_rows=build_stability_summary_html(stats),
    )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DIR / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] Dashboard written to {out_path}")
    print(f"[OK] Figures copied to {DOCS_FIGURES}")


if __name__ == "__main__":
    main()
