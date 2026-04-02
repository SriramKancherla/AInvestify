from __future__ import annotations

import html
import logging
from datetime import datetime, timezone
from typing import Any

from fpdf import FPDF

_LOG = logging.getLogger("ainvestify.report")


def _esc(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value).strip(), quote=True)


def _safe_pdf_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    return (
        s.replace("\r\n", "\n")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2022", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .encode("latin-1", "replace")
        .decode("latin-1")
    )


def _tone_from_recommendation(rec: str | None) -> str:
    """Badge tone for PDF callout; prefers explicit sentiment clause from model-view strings."""
    s = (rec or "").lower()
    if "bearish sentiment" in s:
        return "tone-bear"
    if "bullish sentiment" in s:
        return "tone-bull"
    if "neutral sentiment" in s:
        return "tone-neutral"
    if any(x in s for x in ("bear", "negative", "sell", "mildly bear")):
        return "tone-bear"
    if any(x in s for x in ("bull", "positive", "buy", "mildly bull")):
        return "tone-bull"
    return "tone-neutral"


def _sentiment_badge_class(sent: str | None) -> str:
    s = (sent or "").lower()
    if "bull" in s:
        return "badge-bull"
    if "bear" in s:
        return "badge-bear"
    return "badge-neutral"


def _stock_block(payload: dict, key: str) -> dict[str, Any] | None:
    s = payload.get(key)
    if isinstance(s, dict) and (s.get("ticker") or s.get("company_name")):
        return s
    return None


def _fundamentals_rows(stock: dict[str, Any]) -> list[tuple[str, str]]:
    fund = stock.get("fundamentals")
    if not isinstance(fund, dict) or not fund:
        return []
    return [(str(k), str(v)) for k, v in fund.items()]


def _kpi_triple(stock: dict[str, Any]) -> tuple[str, str, str]:
    fund = stock.get("fundamentals") if isinstance(stock.get("fundamentals"), dict) else {}
    final = str(fund.get("Final Score", "—")) if fund else "—"
    fscore = str(fund.get("Fundamentals Score", "—")) if fund else "—"
    sscore = str(fund.get("Sentiment Score", "—")) if fund else "—"
    return final, fscore, sscore


def build_report_html(payload: dict[str, Any]) -> str:
    """Print-oriented HTML document (WeasyPrint). All dynamic text is escaped."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    primary = _stock_block(payload, "selected")
    compare = _stock_block(payload, "compare")

    def stock_section(title: str, stock: dict[str, Any]) -> str:
        ticker = _esc(stock.get("ticker") or "—")
        company = _esc(stock.get("company_name") or "—")
        rec = stock.get("recommendation")
        rec_s = _esc(rec)
        tone = _tone_from_recommendation(str(rec) if rec else None)
        sent_label = _esc(stock.get("sentiment_label") or "")
        sent_badge = _sentiment_badge_class(str(stock.get("sentiment_label")))
        final_v, fund_v, sent_v = _kpi_triple(stock)

        price = stock.get("current_price")
        chg = stock.get("change_percent")
        mcap = _esc(stock.get("market_cap"))
        vol = _esc(stock.get("volume"))
        hi = stock.get("high_52w")
        lo = stock.get("low_52w")

        rec_block = ""
        if rec:
            rec_block = f'<div class="callout {tone}"><p class="callout-label">Model view</p><p class="callout-text">{rec_s}</p></div>'

        fund_rows = _fundamentals_rows(stock)
        fund_html = ""
        if fund_rows:
            cells = "".join(
                f'<tr><td class="k">{_esc(k)}</td><td class="v">{_esc(v)}</td></tr>' for k, v in fund_rows
            )
            fund_html = f'<h3>Model &amp; valuation metrics</h3><table class="metrics">{cells}</table>'

        news = stock.get("news")
        news_html = ""
        if isinstance(news, list) and news:
            parts = []
            for item in news[:12]:
                if not isinstance(item, dict):
                    continue
                headline = _esc(item.get("title") or "—")
                st = _esc(item.get("sentiment") or "—")
                src = _esc(item.get("source") or "")
                link = item.get("link") or ""
                link_s = str(link).strip()
                nb = _sentiment_badge_class(str(item.get("sentiment")))
                link_disp = link_s[:80] + ("…" if len(link_s) > 80 else "")
                link_html = f'<a href="{_esc(link_s)}">{_esc(link_disp)}</a>' if link_s else ""
                parts.append(
                    f'<div class="news-item {nb}"><p class="nh">{headline}</p>'
                    f'<p class="nm">Sentiment: {st}' + (f" · {src}" if src else "") + "</p>"
                    + (f'<p class="nl">{link_html}</p>' if link_html else "")
                    + "</div>"
                )
            news_html = "<h3>Recent headlines (sample)</h3>" + '<div class="news-list">' + "".join(parts) + "</div>"

        return f"""
        <section class="stock-block">
          <h2>{_esc(title)}</h2>
          <p class="subhead">{company} <span class="tick">{ticker}</span></p>
          <div class="kpi-row">
            <div class="kpi"><span class="kpi-label">Final score</span><span class="kpi-val">{_esc(final_v)}</span></div>
            <div class="kpi"><span class="kpi-label">Fundamentals</span><span class="kpi-val">{_esc(fund_v)}</span></div>
            <div class="kpi"><span class="kpi-label">Sentiment</span><span class="kpi-val">{_esc(sent_v)}</span>
              <span class="badge {sent_badge}">{sent_label or "—"}</span></div>
          </div>
          {rec_block}
          <h3>Price &amp; activity</h3>
          <div class="grid-2">
            <ul class="facts">
              <li><span>Current price</span><strong>{_esc(price if price is not None else "—")}</strong></li>
              <li><span>Change (window)</span><strong>{_esc(chg if chg is not None else "—")} %</strong></li>
              <li><span>Market cap</span><strong>{mcap or "—"}</strong></li>
              <li><span>Volume (last bar)</span><strong>{vol or "—"}</strong></li>
              <li><span>52-week high</span><strong>{_esc(hi if hi is not None else "—")}</strong></li>
              <li><span>52-week low</span><strong>{_esc(lo if lo is not None else "—")}</strong></li>
            </ul>
          </div>
          {fund_html}
          {news_html}
        </section>
        """

    stocks_html = ""
    if primary:
        stocks_html += stock_section("Primary holding", primary)
    else:
        stocks_html += '<section class="stock-block empty"><p class="muted">No primary stock was included in this export.</p></section>'

    if compare:
        stocks_html += stock_section("Comparison holding", compare)

    port_html = ""
    portfolio = payload.get("portfolio")
    if isinstance(portfolio, dict) and portfolio:
        summary = portfolio.get("summary") if isinstance(portfolio.get("summary"), dict) else portfolio
        tmv = summary.get("total_market_value")
        tcb = summary.get("total_cost_basis")
        pnl = summary.get("total_pnl")
        pnl_pct = summary.get("total_pnl_pct")
        port_html = f"""
        <section class="page-break portfolio">
          <h2>Portfolio summary</h2>
          <p class="note">Session server state — not tied to user accounts.</p>
          <table class="metrics">
            <tr><td class="k">Total market value</td><td class="v">{_esc(tmv if tmv is not None else "—")}</td></tr>
            <tr><td class="k">Total cost basis</td><td class="v">{_esc(tcb if tcb is not None else "—")}</td></tr>
            <tr><td class="k">Total P/L</td><td class="v">{_esc(pnl if pnl is not None else "—")}</td></tr>
            <tr><td class="k">Total P/L %</td><td class="v">{_esc(pnl_pct if pnl_pct is not None else "—")}</td></tr>
          </table>
        </section>
        """

    events_html = ""
    events = payload.get("events")
    if isinstance(events, list) and events:
        ev_parts = []
        for ev in events[:20]:
            if not isinstance(ev, dict):
                continue
            title = _esc(ev.get("event_type") or "Event")
            when = _esc(ev.get("event_date") or "—")
            risk = _esc(ev.get("risk_level") or "")
            details = _esc(ev.get("details") or "")
            src = _esc(ev.get("source") or "")
            ev_parts.append(
                f'<div class="event-card"><p class="et">{title}</p>'
                f'<p class="ed">{when} · Importance: {risk}</p><p class="ed">{details}</p>'
                + (f'<p class="es">Source: {src}</p>' if src else "")
                + "</div>"
            )
        events_html = f"""
        <section class="page-break events">
          <h2>Event Calendar</h2>
          <p class="note">Best-effort upcoming events for this ticker from public data sources.</p>
          <div class="event-list">{"".join(ev_parts)}</div>
        </section>
        """

    closing = """
        <section class="closing page-break">
          <h2>Important disclaimer</h2>
          <p class="disclaimer">
            AInvestify uses machine learning and public data sources; outputs can be wrong, incomplete, or delayed.
            Past performance does not guarantee future results. This report summarizes dashboard-style information
            for <strong>education and research only</strong> — not investment, tax, or legal advice. Always verify
            material facts and consult a licensed professional before making financial decisions.
          </p>
          <div class="creators">
            <p class="creators-title">About this report</p>
            <p>Generated by <strong>AInvestify</strong> — an equity insight and research tool.</p>
            <p class="muted small">
              Creators:
              <a href="https://www.linkedin.com/in/sriram-kancherla-80a7b028a/">Sriram Kancherla</a>
              ·
              <a href="https://www.linkedin.com/in/vishwa-yadavalli-65503628b/">Viswanath Parshuram Yadavalli</a>
            </p>
          </div>
        </section>
    """

    css = """
    @page { size: A4; margin: 14mm 16mm 18mm 16mm;
      @bottom-center { content: "AInvestify · Confidential research summary"; font-size: 8pt; color: #64748b; }
    }
    * { box-sizing: border-box; }
    body {
      font-family: DejaVu Sans, Helvetica, Arial, sans-serif;
      font-size: 10pt;
      line-height: 1.45;
      color: #0f172a;
      background: #fff;
      margin: 0;
    }
    .cover {
      background: #0f172a;
      border-bottom: 4px solid #0d9488;
      color: #f8fafc;
      padding: 28px 24px 32px;
      margin: -14mm -16mm 20px -16mm;
    }
    .cover h1 { margin: 0; font-size: 26pt; font-weight: 700; letter-spacing: -0.02em; }
    .cover .tag { margin: 8px 0 0; font-size: 11pt; opacity: 0.92; }
    .cover .when { margin: 14px 0 0; font-size: 9pt; opacity: 0.75; }
    .intro {
      background: #f1f5f9;
      border-left: 4px solid #0d9488;
      padding: 12px 14px;
      margin-bottom: 22px;
      font-size: 9.5pt;
      color: #334155;
    }
    h2 { font-size: 14pt; color: #0f172a; margin: 0 0 6px; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }
    h3 { font-size: 11pt; color: #475569; margin: 18px 0 8px; text-transform: uppercase; letter-spacing: 0.06em; }
    .subhead { margin: 0 0 14px; font-size: 11pt; color: #475569; }
    .tick { display: inline-block; background: #e2e8f0; color: #0f172a; padding: 2px 10px; border-radius: 6px;
      font-weight: 700; font-size: 10pt; margin-left: 6px; }
    .kpi-row { display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
    .kpi {
      flex: 1 1 120px;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      padding: 12px 14px;
      background: #fafafa;
    }
    .kpi-label { display: block; font-size: 8pt; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; }
    .kpi-val { display: block; font-size: 16pt; font-weight: 700; color: #0f172a; margin-top: 4px; }
    .badge { display: inline-block; margin-top: 6px; font-size: 8pt; padding: 2px 8px; border-radius: 999px; font-weight: 600; }
    .badge-bull { background: #dcfce7; color: #166534; }
    .badge-bear { background: #fee2e2; color: #991b1b; }
    .badge-neutral { background: #e2e8f0; color: #475569; }
    .callout { border-radius: 10px; padding: 12px 14px; margin: 12px 0; border: 1px solid #e2e8f0; }
    .callout-label { margin: 0; font-size: 8pt; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; }
    .callout-text { margin: 6px 0 0; font-size: 10pt; }
    .tone-bull { background: #ecfdf5; border-color: #6ee7b7; }
    .tone-bear { background: #fef2f2; border-color: #fca5a5; }
    .tone-neutral { background: #f8fafc; border-color: #cbd5e1; }
    .facts { list-style: none; padding: 0; margin: 0; columns: 2; column-gap: 24px; }
    .facts li { break-inside: avoid; margin-bottom: 8px; font-size: 9.5pt; }
    .facts span { color: #64748b; display: block; font-size: 8pt; }
    .facts strong { font-size: 11pt; color: #0f172a; }
    table.metrics { width: 100%; border-collapse: collapse; font-size: 9pt; margin-top: 6px; }
    table.metrics tr { border-bottom: 1px solid #e2e8f0; }
    table.metrics td { padding: 8px 6px; vertical-align: top; }
    table.metrics td.k { width: 42%; color: #64748b; }
    table.metrics td.v { font-weight: 600; color: #0f172a; }
    .news-list { display: flex; flex-direction: column; gap: 10px; }
    .news-item { border-left: 4px solid #94a3b8; padding: 8px 0 8px 12px; break-inside: avoid; }
    .news-item.badge-bull { border-left-color: #22c55e; }
    .news-item.badge-bear { border-left-color: #ef4444; }
    .nh { margin: 0; font-weight: 600; font-size: 9.5pt; }
    .nm, .nl { margin: 4px 0 0; font-size: 8pt; color: #64748b; }
    .nl a { color: #0d9488; word-break: break-all; }
    .stock-block { margin-bottom: 28px; }
    .stock-block.empty { padding: 20px; text-align: center; }
    .page-break { break-before: page; padding-top: 8px; }
    .note { font-size: 8.5pt; font-style: italic; color: #64748b; margin: 0 0 12px; }
    .event-list { display: flex; flex-direction: column; gap: 12px; }
    .event-card { border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 12px; background: #fafafa; break-inside: avoid; }
    .et { margin: 0; font-weight: 700; font-size: 10pt; }
    .ed, .es { margin: 4px 0 0; font-size: 8.5pt; color: #475569; }
    .es { color: #94a3b8; font-size: 8pt; }
    .closing h2 { border-color: #cbd5e1; }
    .disclaimer { font-size: 9pt; color: #334155; line-height: 1.55; }
    .creators { margin-top: 22px; padding-top: 16px; border-top: 1px solid #e2e8f0; }
    .creators-title { font-size: 10pt; font-weight: 700; margin: 0 0 8px; color: #0f172a; }
    .creators p { margin: 6px 0; font-size: 9.5pt; color: #475569; }
    .muted { color: #64748b; }
    .small { font-size: 8.5pt; }
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>AInvestify Report</title>
  <style>{css}</style>
</head>
<body>
  <header class="cover">
    <h1>AInvestify</h1>
    <p class="tag">Equity insight report</p>
    <p class="when">Generated {_esc(generated)}</p>
  </header>
  <p class="intro">
    This report mirrors your dashboard: model-based scores, market metrics, news sentiment samples,
    and optional compare data. It is for <strong>education and research only</strong> — not investment,
    tax, or legal advice.
  </p>
  {stocks_html}
  {port_html}
  {events_html}
  {closing}
</body>
</html>
"""


def _write_stock_sections(pdf: FPDF, title: str, stock: dict[str, Any]) -> None:
    pdf.ln(2)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 8, _safe_pdf_text(title))
    pdf.set_font("Helvetica", "", 10)

    ticker = stock.get("ticker") or "—"
    company = stock.get("company_name") or "—"
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, _safe_pdf_text(f"Company: {company}\nTicker: {ticker}"))

    rec = stock.get("recommendation")
    if rec:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, "Model view")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5, _safe_pdf_text(str(rec)))

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6, "Price & activity snapshot")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(pdf.l_margin)
    lines = [
        f"Current price: {stock.get('current_price') if stock.get('current_price') is not None else '—'}",
        f"Change (selected window): {stock.get('change_percent') if stock.get('change_percent') is not None else '—'} %",
        f"Market cap: {stock.get('market_cap') or '—'}",
        f"Volume (last bar): {stock.get('volume') or '—'}",
        f"52-week high: {stock.get('high_52w') if stock.get('high_52w') is not None else '—'}",
        f"52-week low: {stock.get('low_52w') if stock.get('low_52w') is not None else '—'}",
        f"Sentiment (UI scale): {stock.get('sentiment_score') if stock.get('sentiment_score') is not None else '—'} — {stock.get('sentiment_label') or ''}",
    ]
    pdf.multi_cell(0, 5, _safe_pdf_text("\n".join(lines)))

    fund = stock.get("fundamentals")
    if isinstance(fund, dict) and fund:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, "Model & valuation metrics")
        pdf.set_font("Helvetica", "", 9)
        for k, v in fund.items():
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 4, _safe_pdf_text(f"- {k}: {v}"))

    news = stock.get("news")
    if isinstance(news, list) and news:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, "Recent headlines (sample)")
        pdf.set_font("Helvetica", "", 9)
        for item in news[:12]:
            if not isinstance(item, dict):
                continue
            headline = item.get("title") or "—"
            sent = item.get("sentiment") or "—"
            src = item.get("source") or ""
            link = str(item.get("link") or "").strip()

            # Headline line(s)
            block = f"• [{sent}] {headline}"
            if src:
                block += f" (source: {src})"
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 4, _safe_pdf_text(block))

            # Optional clickable URL (short display, full hyperlink)
            if link:
                disp = link if len(link) <= 96 else (link[:96] + "…")
                pdf.set_x(pdf.l_margin + 4)
                pdf.set_text_color(13, 148, 136)
                pdf.set_font("Helvetica", "U", 8)
                pdf.cell(0, 4, _safe_pdf_text(disp), ln=True, link=link)
                pdf.set_text_color(15, 23, 42)
                pdf.set_font("Helvetica", "", 9)

            pdf.ln(1)


def _write_portfolio(pdf: FPDF, summary: dict[str, Any] | None) -> None:
    if not isinstance(summary, dict):
        return
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Portfolio summary (session server state)", ln=True)
    pdf.set_font("Helvetica", "", 10)
    tmv = summary.get("total_market_value")
    tcb = summary.get("total_cost_basis")
    pnl = summary.get("total_pnl")
    pnl_pct = summary.get("total_pnl_pct")
    lines = [
        f"Total market value: {tmv if tmv is not None else '—'}",
        f"Total cost basis: {tcb if tcb is not None else '—'}",
        f"Total P/L: {pnl if pnl is not None else '—'}",
        f"Total P/L %: {pnl_pct if pnl_pct is not None else '—'}",
    ]
    pdf.multi_cell(0, 5, _safe_pdf_text("\n".join(lines)))
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0,
        4,
        _safe_pdf_text(
            "Note: Holdings are not yet tied to user accounts. "
            "Figures reflect this browser session / server memory only."
        ),
    )


def _write_events(pdf: FPDF, events: Any) -> None:
    if not isinstance(events, list) or not events:
        return
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Event Calendar", ln=True)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0,
        4,
        _safe_pdf_text(
            "These entries are best-effort upcoming events for this ticker from public data sources."
        ),
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    for ev in events[:20]:
        if not isinstance(ev, dict):
            continue
        title = ev.get("event_type") or "Event"
        when = ev.get("event_date") or "—"
        risk = ev.get("risk_level") or ""
        details = ev.get("details") or ""
        src = ev.get("source") or ""
        block = f"- {title}\n  Date: {when}   Importance: {risk}\n  {details}"
        if src:
            block += f"\n  Source: {src}"
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 4, _safe_pdf_text(block))
        pdf.ln(2)


def _build_report_pdf_fpdf(payload: dict[str, Any]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(18, 18, 18)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 220, 36, "F")
    pdf.set_y(10)
    pdf.set_x(18)
    pdf.set_text_color(248, 250, 252)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 8, "AInvestify", ln=True)
    pdf.set_x(18)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 5, "Equity insight report", ln=True)
    pdf.set_x(18)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 4, f"Generated: {generated}", ln=True)
    pdf.set_text_color(15, 23, 42)
    pdf.set_y(44)
    pdf.set_x(pdf.l_margin)

    pdf.set_font("Helvetica", "", 10)
    # Keep some breathing room under the header band.
    pdf.ln(2)
    pdf.multi_cell(
        0,
        5,
        _safe_pdf_text(
            "This report summarizes the same information shown on your dashboard: "
            "model-based scores, market-style metrics, news sentiment samples, and optional compare data. "
            "It is for education and research only — not investment, tax, or legal advice."
        ),
    )

    primary = _stock_block(payload, "selected")
    if primary:
        _write_stock_sections(pdf, "Primary holding", primary)
    else:
        pdf.ln(4)
        pdf.set_font("Helvetica", "I", 10)
        pdf.multi_cell(0, 5, _safe_pdf_text("No primary stock was included in this export."))

    compare = _stock_block(payload, "compare")
    if compare:
        if pdf.get_y() > 250:
            pdf.add_page()
        _write_stock_sections(pdf, "Comparison holding", compare)

    portfolio = payload.get("portfolio")
    if isinstance(portfolio, dict) and portfolio:
        pdf.add_page()
        summ = portfolio.get("summary") if isinstance(portfolio.get("summary"), dict) else portfolio
        _write_portfolio(pdf, summ if isinstance(summ, dict) else None)

    events = payload.get("events")
    if isinstance(events, list) and events:
        pdf.add_page()
        _write_events(pdf, events)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Important disclaimer", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        0,
        4,
        _safe_pdf_text(
            "AInvestify uses machine learning and public data sources; outputs can be wrong, "
            "incomplete, or delayed. Past performance does not guarantee future results. "
            "Always verify material facts and consult a licensed professional before making financial decisions."
        ),
    )
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "About this report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 4, _safe_pdf_text("Generated by AInvestify."))
    pdf.ln(1)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 4, "Creators:", ln=True)
    pdf.set_font("Helvetica", "U", 9)
    pdf.set_text_color(13, 148, 136)
    pdf.set_x(pdf.l_margin)
    pdf.cell(
        0,
        4,
        _safe_pdf_text("Sriram Kancherla"),
        ln=True,
        link="https://www.linkedin.com/in/sriram-kancherla-80a7b028a/",
    )
    pdf.set_x(pdf.l_margin)
    pdf.cell(
        0,
        4,
        _safe_pdf_text("Viswanath Parshuram Yadavalli"),
        ln=True,
        link="https://www.linkedin.com/in/vishwa-yadavalli-65503628b/",
    )
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "", 9)

    return bytes(pdf.output(dest="S"))


def build_report_pdf_bytes(payload: dict[str, Any]) -> bytes:
    """Build a styled PDF via HTML+WeasyPrint when available; otherwise fpdf2."""
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as e:
        _LOG.warning("weasyprint unavailable (%s); using fpdf2 fallback for reports", e)
        return _build_report_pdf_fpdf(payload)

    html_doc = build_report_html(payload)
    try:
        return HTML(string=html_doc, base_url=".").write_pdf()
    except Exception as e:
        _LOG.warning("weasyprint render failed (%s); using fpdf2 fallback", e)
        return _build_report_pdf_fpdf(payload)


def report_attachment_filename(payload: dict[str, Any]) -> str:
    """Download filename: AInvestify-<TICKER>-report.pdf (ASCII-safe)."""
    sel = payload.get("selected")
    t = ""
    if isinstance(sel, dict) and sel.get("ticker"):
        t = str(sel.get("ticker")).strip().upper()
    safe = "".join(c for c in t if c.isalnum() or c in "._-") or "Report"
    return f"AInvestify-{safe}-report.pdf"
