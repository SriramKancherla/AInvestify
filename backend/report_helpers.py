"""Shared helpers for institutional PDF report rendering."""

from __future__ import annotations

import math
import statistics
from typing import Any

METRIC_SKIP = frozenset({"Final Score", "Fundamentals Score", "Sentiment Score"})

CONFIDENCE_FOOTNOTE = (
    "Model confidence reflects agreement across fundamentals and sentiment sub-models and input quality."
)

_TAG = "div"


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
        if math.isfinite(v):
            return v
    except (TypeError, ValueError):
        pass
    return None


def ai_score_01(stock: dict[str, Any]) -> float | None:
    v = as_float(stock.get("ai_score"))
    if v is not None:
        return max(0.0, min(1.0, v))
    fund = stock.get("fundamentals")
    if isinstance(fund, dict):
        v = as_float(fund.get("Final Score"))
        if v is not None:
            return max(0.0, min(1.0, v))
    return None


def ai_score_100(stock: dict[str, Any]) -> str:
    v = ai_score_01(stock)
    return str(round(v * 100)) if v is not None else "—"


def model_confidence_pct(stock: dict[str, Any]) -> str:
    v = as_float(stock.get("model_confidence"))
    return f"{round(v * 100)}%" if v is not None else "—"


def fundamentals_100(stock: dict[str, Any]) -> str:
    v = as_float(stock.get("fundamentals_score"))
    if v is None and isinstance(stock.get("fundamentals"), dict):
        v = as_float(stock["fundamentals"].get("Fundamentals Score"))
    return str(round(v * 100)) if v is not None else "—"


def sentiment_100(stock: dict[str, Any]) -> str:
    if isinstance(stock.get("fundamentals"), dict):
        v = as_float(stock["fundamentals"].get("Sentiment Score"))
        if v is not None:
            return str(round(v * 100))
    pct = as_float(stock.get("sentiment_score"))
    if pct is not None:
        return str(round(pct)) if pct > 1 else str(round(pct * 100))
    return "—"


def outlook(stock: dict[str, Any]) -> str:
    label = stock.get("outlook")
    return str(label).strip() if label else "—"


def model_agreement(stock: dict[str, Any]) -> tuple[str, float | None]:
    scores = stock.get("model_scores")
    if not isinstance(scores, dict) or len(scores) < 2:
        return "—", None
    vals = [v for v in (as_float(x) for x in scores.values()) if v is not None]
    if len(vals) < 2:
        return "—", None
    spread = float(statistics.pstdev(vals))
    if spread < 0.05:
        return "High agreement", spread
    if spread < 0.12:
        return "Moderate agreement", spread
    return "Low agreement", spread


def tone_from_outlook(outlook_label: str | None) -> str:
    s = (outlook_label or "").lower()
    if "bear" in s:
        return "tone-bear"
    if "bull" in s:
        return "tone-bull"
    return "tone-neutral"


def change_class(chg: Any) -> str:
    v = as_float(chg)
    if v is None:
        return "chg-neutral"
    if v > 0:
        return "chg-pos"
    if v < 0:
        return "chg-neg"
    return "chg-neutral"


def format_price(price: Any) -> str:
    v = as_float(price)
    return f"${v:,.2f}" if v is not None else "—"


def format_change(chg: Any) -> str:
    v = as_float(chg)
    if v is None:
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.2f}%"


def exec_paragraph(stock: dict[str, Any]) -> str:
    rec = str(stock.get("recommendation") or "").strip()
    if rec and "unavailable" not in rec.lower():
        return rec if rec.endswith(".") else f"{rec[0].upper()}{rec[1:]}."

    ol = outlook(stock).lower()
    fund = as_float(stock.get("fundamentals_score"))
    sent = str(stock.get("sentiment_label") or "").lower()

    if fund is not None:
        fund_phrase = (
            "fundamentals remain strong"
            if fund >= 0.6
            else "fundamentals are mixed"
            if fund >= 0.4
            else "fundamentals remain weak"
        )
    else:
        fund_phrase = "fundamentals are unavailable"

    if "bull" in sent:
        sent_phrase = "sentiment is supportive"
    elif "bear" in sent:
        sent_phrase = "sentiment is cautious"
    else:
        sent_phrase = "sentiment is unclear"

    if "bull" in ol:
        tail = "suggest moderate upside potential over the medium term."
    elif "bear" in ol:
        tail = "suggest caution over the near term."
    elif "mixed" in ol or "uncertain" in ol:
        tail = "suggest a balanced, wait-and-see posture."
    else:
        tail = "warrant continued monitoring."

    return f"While {fund_phrase}, {sent_phrase}, which {tail}"


def key_signals(stock: dict[str, Any]) -> list[str]:
    signals: list[str] = []
    fund = as_float(stock.get("fundamentals_score"))
    if fund is not None:
        signals.append(f"Fundamentals score {fund:.2f} ({fundamentals_100(stock)}/100)")
    sent_lbl = stock.get("sentiment_label")
    sent_n = stock.get("news_item_count")
    if sent_lbl:
        extra = f", n={sent_n} articles" if sent_n else ""
        signals.append(f"Sentiment {sentiment_100(stock)}/100 ({sent_lbl}{extra})")
    agreement, spread = model_agreement(stock)
    if agreement != "—" and spread is not None:
        prefix = "−" if spread >= 0.12 else "+"
        signals.append(f"{prefix} Fundamentals models: σ={spread:.2f} ({agreement.lower()})")
    ol = outlook(stock)
    if ol != "—":
        signals.append(f"Outlook: {ol}")
    return signals[:4]


def sparkline_svg(points: list[Any], width: int = 220, height: int = 44) -> str:
    vals = [v for v in (as_float(p) for p in points) if v is not None]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = hi - lo or 1.0
    coords: list[str] = []
    for i, v in enumerate(vals):
        x = (i / (len(vals) - 1)) * width
        y = height - ((v - lo) / span) * (height - 6) - 3
        coords.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(coords)
    rising = vals[-1] >= vals[0]
    stroke = "#0F9D58" if rising else "#D93025"
    fill = "#0F9D5820" if rising else "#D9302520"
    area = f"M0,{height} L" + " L".join(coords) + f" L{width},{height} Z"
    return (
        f'<svg class="spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        f'<path d="{area}" fill="{fill}"/>'
        f'<polyline fill="none" stroke="{stroke}" stroke-width="1.8" points="{poly}"/>'
        f"</svg>"
    )


def _hcard(label: str, value: str, val_class: str = "hero-val") -> str:
    t = _TAG
    return (
        f"<{t} class='hero-card'>"
        f"<span class='hero-label'>{label}</span>"
        f"<span class='{val_class}'>{value}</span>"
        f"</{t}>"
    )


def hero_cards_html(esc: Any, stock: dict[str, Any]) -> str:
    t = _TAG
    cards = (
        _hcard("Outlook", esc(outlook(stock)), "hero-val hero-val-text")
        + _hcard("AI Score", f"{esc(ai_score_100(stock))}<span class='hero-suffix'>/100</span>")
        + _hcard("Model confidence", esc(model_confidence_pct(stock)), "hero-val hero-val-pct")
    )
    return f"<{t} class='hero-cards'>{cards}</{t}>"


def exec_summary_html(esc: Any, stock: dict[str, Any], sentiment_dot_fn: Any) -> str:
    t = _TAG
    agreement, _ = model_agreement(stock)
    sent_lbl = esc(stock.get("sentiment_label") or "—")
    sent_dot = sentiment_dot_fn(str(stock.get("sentiment_label")))
    tone = tone_from_outlook(outlook(stock))
    sig_html = "".join(f"<li>{esc(s)}</li>" for s in key_signals(stock))
    grid = (
        _hcard("Outlook", esc(outlook(stock)), "exec-val")
        + _hcard("AI Score", f"{esc(ai_score_100(stock))}<span class='hero-suffix'>/100</span>", "exec-val mono")
        + _hcard("Model confidence", esc(model_confidence_pct(stock)), "exec-val mono pct")
        + _hcard("Fundamentals", f"{esc(fundamentals_100(stock))}<span class='hero-suffix'>/100</span>", "exec-val mono")
        + _hcard(
            "Sentiment",
            f"<span class='dot {sent_dot}'></span> {sent_lbl}",
            "exec-val",
        )
        + _hcard("Model agreement", esc(agreement), "exec-val")
    )
    return (
        f"<section class='exec-summary'>"
        f"<h2>Executive summary</h2>"
        f"<{t} class='exec-grid'>{grid}</{t}>"
        f"<{t} class='callout {tone}'><p class='callout-text'>{esc(exec_paragraph(stock))}</p></{t}>"
        f"<h3>Key signals</h3>"
        f"<ul class='signals'>{sig_html}</ul>"
        f"<p class='footnote'>* {CONFIDENCE_FOOTNOTE}</p>"
        f"</section>"
    )


def _company_logo_img(ticker: str | None) -> str:
    from backend.logo_assets import logo_data_uri

    uri = logo_data_uri(ticker)
    if not uri:
        return ""
    return f"<img class='company-logo' src='{uri}' alt='' />"


def premium_cover_html(esc: Any, stock: dict[str, Any], generated: str, notice: str) -> str:
    t = _TAG
    chg_cls = change_class(stock.get("change_percent"))
    spark = stock.get("sparkline")
    spark_html = sparkline_svg(spark if isinstance(spark, list) else [])
    rec = stock.get("recommendation")
    rec_block = ""
    if rec:
        rec_block = (
            f"<p class='model-view-line'><span class='mv-label'>Model view</span> "
            f"<span class='mv-text'>{esc(rec)}</span></p>"
        )
    spark_block = f"<{t} class='spark-wrap'>{spark_html}</{t}>" if spark_html else ""
    logo_html = _company_logo_img(stock.get("ticker"))
    return (
        f"<section class='stock-hero'>"
        f"<{t} class='brand-row'>"
        f"<span class='mark'>AI</span>"
        f"<span class='brand-name'>AInvestify</span>"
        f"<span class='brand-when'>Generated {esc(generated)}</span>"
        f"</{t}>"
        f"<{t} class='notice'>{esc(notice)}</{t}>"
        f"<{t} class='hero-title'>"
        f"{logo_html}"
        f"<h1 class='hero-company'>{esc(stock.get('company_name') or '—')}</h1>"
        f"<span class='tick'>{esc(stock.get('ticker') or '—')}</span>"
        f"</{t}>"
        f"<{t} class='hero-price-row'>"
        f"<span class='hero-price'>{esc(format_price(stock.get('current_price')))}</span>"
        f"<span class='hero-chg {chg_cls}'>{esc(format_change(stock.get('change_percent')))}</span>"
        f"</{t}>"
        f"{spark_block}"
        f"{hero_cards_html(esc, stock)}"
        f"{rec_block}"
        f"<p class='footnote'>* {CONFIDENCE_FOOTNOTE}</p>"
        f"</section>"
    )
