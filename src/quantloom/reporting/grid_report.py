"""Renders a multi-combination grid search as a sortable HTML table instead of a wall of stacked
console reports. One row per combination (hyperparameters / train Sharpe / test Sharpe), sorted by
train Sharpe descending, with the train/test Sharpe correlation summarized underneath -- a low or
negative value is a quick overfitting signal (a parameter that helps train Sharpe without helping
test Sharpe). Clicking a row reveals that combination's full comprehensive report (the same text
`format_report` would otherwise print) plus its equity curve, without ever having to print or plot
all of them at once. Plotly.js itself is inlined a single time (rather than once per row, the way
`Figure.show()`/`Figure.to_html()` would) and each row's chart is re-rendered into the same `<div>`
on click, so an N-combination grid ships one copy of the ~4.5MB library, not N.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass

from plotly.offline import get_plotlyjs

from quantloom.backtest.metrics import BacktestReport, ModelPeriodReport
from quantloom.config.grid import GridPoint
from quantloom.reporting.graphing import plot_equity_curve
from quantloom.reporting.report import format_report


@dataclass
class GridRow:
    label: str
    train_sharpe: float | None
    test_sharpe: float | None
    report_text: str
    equity_json: str | None


def _format_overrides(overrides: dict[str, object]) -> str:
    if not overrides:
        return "(baseline)"
    return ", ".join(f"{path} = {value}" for path, value in overrides.items())


def _sharpe(period: ModelPeriodReport | None) -> float | None:
    if period is None or period.stats.sharpe_ratio is None or math.isnan(period.stats.sharpe_ratio):
        return None
    return period.stats.sharpe_ratio


def build_grid_rows(
    results: list[tuple[GridPoint, BacktestReport]], *, include_equity_graphs: bool = True
) -> list[GridRow]:
    """One `GridRow` per grid combination, sorted by train Sharpe descending (missing/NaN train
    Sharpe sorts last, since there's nothing to rank it against)."""
    rows = [
        GridRow(
            label=_format_overrides(point.overrides),
            train_sharpe=_sharpe(report.train),
            test_sharpe=_sharpe(report.test),
            report_text=format_report(report),
            equity_json=plot_equity_curve(report).to_json() if include_equity_graphs else None,
        )
        for point, report in results
    ]
    rows.sort(
        key=lambda row: row.train_sharpe if row.train_sharpe is not None else -math.inf,
        reverse=True,
    )
    return rows


def _pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def train_test_sharpe_correlation(rows: list[GridRow]) -> float | None:
    """Pearson correlation between train and test Sharpe across every row that has both --
    `None` (not NaN) when fewer than two rows qualify, since a correlation isn't meaningfully
    defined there."""
    pairs = [
        (r.train_sharpe, r.test_sharpe)
        for r in rows
        if r.train_sharpe is not None and r.test_sharpe is not None
    ]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs, strict=True)
    return _pearson_correlation(list(xs), list(ys))


def _fmt_sharpe(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "N/A"


_PAGE_TEMPLATE = """<title>Grid Search Report</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #ffffff;
    --fg: #1a1a1a;
    --muted: #666666;
    --border: #dddddd;
    --row-hover: #f2f4f7;
    --row-selected: #e3ecff;
    --accent: #3060c8;
    --panel-bg: #f8f9fb;
    --mono-bg: #ffffff;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #15171c;
      --fg: #e8e8e8;
      --muted: #9a9fa8;
      --border: #33363d;
      --row-hover: #202329;
      --row-selected: #2a3a5c;
      --accent: #7ea3ff;
      --panel-bg: #1b1e24;
      --mono-bg: #101216;
    }}
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    height: 100%;
    overflow: hidden;
  }}
  body {{
    margin: 0;
    display: flex;
    flex-direction: column;
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--fg);
  }}
  header {{
    flex: 0 0 auto;
    padding: 14px 24px 10px;
    border-bottom: 1px solid var(--border);
  }}
  h1 {{
    font-size: 1.2rem;
    margin: 0 0 4px;
  }}
  .subtitle {{
    color: var(--muted);
    font-size: 0.9rem;
  }}
  .layout {{
    flex: 1 1 auto;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(320px, 1fr) 2fr;
    gap: 0;
  }}
  @media (max-width: 800px) {{
    html, body {{ height: auto; overflow: auto; }}
    .layout {{ grid-template-columns: 1fr; }}
    .table-pane, .detail-pane {{ max-height: 60vh; }}
  }}
  .table-pane {{
    overflow-y: auto;
    min-height: 0;
    border-right: 1px solid var(--border);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }}
  thead th {{
    position: sticky;
    top: 0;
    background: var(--panel-bg);
    text-align: left;
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }}
  thead th.col-num {{ text-align: right; }}
  tbody td {{
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }}
  td.col-num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  td.col-label {{ word-break: break-word; }}
  tbody tr {{ cursor: pointer; }}
  tbody tr:hover {{ background: var(--row-hover); }}
  tbody tr.selected {{ background: var(--row-selected); }}
  .rank {{ color: var(--muted); font-size: 0.8rem; margin-right: 6px; }}
  .detail-pane {{
    overflow-y: auto;
    min-height: 0;
    padding: 16px 24px;
    background: var(--panel-bg);
  }}
  .detail-label {{
    font-weight: 600;
    margin-bottom: 10px;
    font-size: 0.95rem;
  }}
  .placeholder {{
    color: var(--muted);
    font-size: 0.9rem;
  }}
  #detail-equity {{
    height: 360px;
    margin-bottom: 16px;
  }}
  pre {{
    background: var(--mono-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 16px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.82rem;
    line-height: 1.4;
    white-space: pre-wrap;
    word-break: break-word;
  }}
  footer {{
    flex: 0 0 auto;
    padding: 10px 24px;
    border-top: 1px solid var(--border);
    font-size: 0.85rem;
    color: var(--muted);
  }}
  footer strong {{ color: var(--fg); }}
</style>
<header>
  <h1>Grid Search Report</h1>
  <div class="subtitle">{combination_count} combinations, sorted by train Sharpe ratio
  (descending)</div>
</header>
<div class="layout">
  <div class="table-pane">
    <table>
      <thead>
        <tr>
          <th>hyperparameters</th>
          <th class="col-num">train sharpe</th>
          <th class="col-num">test sharpe</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
  <div class="detail-pane">
    <div id="detail-label" class="detail-label"></div>
    <p id="detail-placeholder" class="placeholder">
      Click a row to see its equity curve and comprehensive report.
    </p>
    <div id="detail-equity" style="display: none;"></div>
    <div id="detail-body"></div>
  </div>
</div>
<footer>
  Correlation between train and test Sharpe: <strong>{correlation_text}</strong>
</footer>
{plotly_script}
<script>
  const labels = {labels_json};
  const reports = {reports_json};
  const equityFigures = {equity_json};
  const rows = document.querySelectorAll("tbody tr");
  const detailLabel = document.getElementById("detail-label");
  const detailPlaceholder = document.getElementById("detail-placeholder");
  const detailEquity = document.getElementById("detail-equity");
  const detailBody = document.getElementById("detail-body");

  rows.forEach((row) => {{
    row.addEventListener("click", () => {{
      const index = Number(row.dataset.index);
      rows.forEach((r) => r.classList.remove("selected"));
      row.classList.add("selected");

      detailPlaceholder.style.display = "none";
      detailLabel.textContent = labels[index];

      detailBody.innerHTML = "";
      const pre = document.createElement("pre");
      // innerHTML, not textContent: format_report() embeds <span style="color:..."> markup for
      // the two-column table's better/worse highlighting. Safe here -- report_text is entirely
      // program-generated (numbers/labels from BacktestReport), with any dynamic string content
      // (e.g. the benchmark label) already html-escaped by format_report() itself.
      pre.innerHTML = reports[index];
      detailBody.appendChild(pre);

      const figureJson = equityFigures[index];
      if (figureJson && window.Plotly) {{
        const figure = JSON.parse(figureJson);
        detailEquity.style.display = "";
        Plotly.react(detailEquity, figure.data, figure.layout, {{responsive: true}});
      }} else {{
        detailEquity.style.display = "none";
      }}
    }});
  }});
</script>
"""


def _json_for_script_tag(value: object) -> str:
    """`json.dumps`, with `</` escaped so a value containing a literal `</script>` (e.g. a swept
    hyperparameter value) can't prematurely close the surrounding `<script>` block."""
    return json.dumps(value).replace("</", "<\\/")


def build_grid_report_html(
    results: list[tuple[GridPoint, BacktestReport]], *, include_equity_graphs: bool = True
) -> str:
    rows = build_grid_rows(results, include_equity_graphs=include_equity_graphs)
    correlation = train_test_sharpe_correlation(rows)

    rows_html = "\n        ".join(
        f'<tr class="grid-row" data-index="{i}">'
        f'<td class="col-label"><span class="rank">#{i + 1}</span>{html.escape(row.label)}</td>'
        f'<td class="col-num">{_fmt_sharpe(row.train_sharpe)}</td>'
        f'<td class="col-num">{_fmt_sharpe(row.test_sharpe)}</td>'
        "</tr>"
        for i, row in enumerate(rows)
    )

    # Plotly.js is inlined once here rather than per-row (each row only carries its small
    # data/layout JSON), so an N-combination grid ships one ~4.5MB library, not N.
    plotly_script = (
        f"<script>{get_plotlyjs().replace('</', '<\\/')}</script>" if include_equity_graphs else ""
    )

    return _PAGE_TEMPLATE.format(
        combination_count=len(rows),
        rows_html=rows_html,
        correlation_text=_fmt_sharpe(correlation),
        plotly_script=plotly_script,
        labels_json=_json_for_script_tag([row.label for row in rows]),
        reports_json=_json_for_script_tag([row.report_text for row in rows]),
        equity_json=_json_for_script_tag([row.equity_json for row in rows]),
    )
