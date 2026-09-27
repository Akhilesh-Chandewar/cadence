"""ReportAgent (spec §7.5): white-box, plain-language reporting.

Takes the structured per-series report the graph produced and renders it as
Markdown for humans (what the data looked like, what preprocessing was applied
and why, which models were tried, which won the backtest and by how much, the
final forecast with bands) — the machine-consumable JSON *is* the report dict
itself. HTML is a minimal converter over the exact Markdown subset this agent
emits (headers, lists, bold, tables) — no extra dependency, deterministic.
"""

from __future__ import annotations

import html as _html
import re


class ReportAgent:
    """Renders the graph's structured report dict (§7.5)."""

    def render_markdown(
        self, report: dict, source_meta: dict | None = None, errors: list[dict] | None = None
    ) -> str:
        lines: list[str] = ["# Cadence forecast report", ""]

        if source_meta:
            lines += [
                f"**Run**: {source_meta.get('row_count', '?')} rows, "
                f"{source_meta.get('series_count', '?')} series "
                f"(source: {source_meta.get('source_type', '?')}, "
                f"ingested {str(source_meta.get('ingested_at', ''))[:19]})",
                "",
            ]

        for uid, rep in report.items():
            lines += self._series_section(uid, rep)

        if errors:
            lines += ["## Errors", ""]
            for e in errors:
                uid = e.get("unique_id") or "-"
                lines.append(f"- `[{e.get('stage')}]` {uid}: {e.get('error')}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _series_section(self, uid: str, rep: dict) -> list[str]:
        lines = [f"## Series `{uid}`", ""]

        # data + preprocessing (§7.5: what it looked like, what was done and why)
        pp = rep.get("preprocessing", {})
        reason = pp.get("reason") or "deterministic defaults"
        lines += [
            f"- **Data**: {rep.get('length')} points, frequency `{rep.get('freq')}`",
            f"- **Preprocessing**: missing={pp.get('missing')}, "
            f"outliers={pp.get('outliers')}, transform={pp.get('transform')} "
            f"— *{reason}*",
        ]

        # models tried + backtest outcome (§7.5: which won and by how much)
        scores = sorted(rep.get("scores", []), key=lambda s: s.get("mase_mean", 9e9))
        if scores:
            lines.append("- **Backtest (§9 rolling windows, best first)**:")
            lines += ["", "| model | MASE | sMAPE | WQL |", "|---|---|---|---|"]
            for s in scores:
                wql = s.get("wql")
                lines.append(
                    f"| {s['model']} | {s['mase_mean']:.3f} | {s['smape_mean']:.3f} | "
                    f"{'—' if wql is None else f'{wql:.4f}'} |"
                )
            lines.append("")

        decision = rep.get("decision", "best")
        selected = rep.get("selected", [])
        weights = rep.get("weights") or {}
        if decision == "ensemble" and weights:
            weight_str = ", ".join(
                f"{n} {w:.2f}" for n, w in sorted(weights.items(), key=lambda kv: -kv[1])
            )
            lines.append(
                f"- **Decision**: no single model was a clear, consistent winner, so "
                f"an inverse-error weighted ensemble was used ({weight_str})"
            )
        else:
            lines.append(f"- **Decision**: {selected[0] if selected else '?'} won the backtest")

        # forecast with bands (§7.5: final forecast + confidence intervals)
        forecast = rep.get("forecast", [])
        intervals = rep.get("intervals") or []
        if forecast:
            lo_hi = {(r["ds"], r.get("yhat-lo-95"), r.get("yhat-hi-95")) for r in intervals}
            lines += [
                "",
                f"**Forecast ({len(forecast)} steps, 95% bands)**:",
                "",
                "| ds | yhat | lo-95 | hi-95 |",
                "|---|---|---|---|",
            ]
            for r in forecast:
                lo, hi = None, None
                for ds, lo_v, hi_v in lo_hi:
                    if ds == r["ds"]:
                        lo, hi = lo_v, hi_v
                        break
                lo_s = "—" if lo is None else f"{lo:.2f}"
                hi_s = "—" if hi is None else f"{hi:.2f}"
                lines.append(f"| {str(r['ds'])[:10]} | {r['yhat']:.2f} | {lo_s} | {hi_s} |")
        lines.append("")
        return lines

    # ------------------------------------------------------------- HTML mode
    _CELL = re.compile(r"(?<!\|)\|(?!\|)")

    def render_html(self, markdown: str) -> str:
        """Convert the subset this agent emits: #/##/###, - lists, **bold**, `code`,
        tables, blank-line paragraphs. Escapes everything else."""
        out: list[str] = [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            "<title>Cadence report</title>",
            "<style>body{font-family:system-ui;max-width:56em;margin:2rem auto}"
            "table{border-collapse:collapse}td,th{border:1px solid #ccc;"
            "padding:2px 8px}</style></head><body>",
        ]
        in_list = in_table = False
        for raw_line in markdown.splitlines():
            line = raw_line.strip()

            def _close():
                nonlocal in_list, in_table
                if in_list:
                    out.append("</ul>")
                    in_list = False
                if in_table:
                    out.append("</table>")
                    in_table = False

            if not line:
                _close()
                continue
            if line.startswith("|"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    continue  # table separator row
                if not in_table:
                    _close()
                    out.append("<table>")
                    in_table = True
                tag = "th" if not any(t == "<tr>" for t in out[-3:]) else "td"
                out.append(
                    "<tr>" + "".join(f"<{tag}>{self._inline(c)}</{tag}>" for c in cells) + "</tr>"
                )
                continue
            if line.startswith("- "):
                if not in_list:
                    _close()
                    out.append("<ul>")
                    in_list = True
                out.append(f"<li>{self._inline(line[2:])}</li>")
                continue
            _close()
            m = re.match(r"(#{1,4})\s+(.*)", line)
            if m:
                level = len(m.group(1))
                out.append(f"<h{level}>{self._inline(m.group(2))}</h{level}>")
            else:
                out.append(f"<p>{self._inline(line)}</p>")
        if in_list:
            out.append("</ul>")
        if in_table:
            out.append("</table>")
        out.append("</body></html>")
        return "\n".join(out)

    @staticmethod
    def _inline(text: str) -> str:
        text = _html.escape(text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
        return re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
