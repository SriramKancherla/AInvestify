from __future__ import annotations

import html
import logging
import os
import smtplib
from email.message import EmailMessage
from typing import Any

_LOG = logging.getLogger("ainvestify.notify")


def _env_int(key: str, default: int) -> int:
    raw = (os.getenv(key) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def smtp_send_html(sender: str, password: str, to_addr: str, subject: str, html_body: str, text_fallback: str) -> None:
    host = (os.getenv("EMAIL_SMTP_HOST") or "smtp.gmail.com").strip()
    port = _env_int("EMAIL_SMTP_PORT", 587)
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(text_fallback)
    msg.add_alternative(html_body, subtype="html")
    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=20) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=20) as smtp:
            if port == 587:
                smtp.starttls()
            smtp.login(sender, password)
            smtp.send_message(msg)


def send_price_alert_email(to_email: str, ticker: str, price: float, rule_type: str, threshold: float) -> bool:
    sender = (os.getenv("EMAIL_USER") or "").strip()
    password = (os.getenv("EMAIL_PASS") or "").strip().replace(" ", "")
    if not sender or not password or not to_email:
        _LOG.warning("Skipping alert email: missing EMAIL_USER/EMAIL_PASS or recipient.")
        return False
    subj = f"AInvestify - Stock Insights — {ticker.upper()} @ ${price:.2f}"
    safe_t = html.escape(ticker.upper())
    body_html = f"""
<div style="font-family:Arial,Helvetica,sans-serif;color:#0f172a;">
  <h2>AInvestify - Stock Insights</h2>
  <p style="margin:0 0 12px 0;font-size:13px;color:#64748b;">Price alert</p>
  <p><strong>{safe_t}</strong> is at <strong>${price:.2f}</strong>.</p>
  <p>Rule: {html.escape(rule_type)} threshold {html.escape(str(threshold))}.</p>
  <p style="font-size:12px;color:#64748b;">Automated message from AInvestify. Not investment advice.</p>
</div>
"""
    text = f"Price alert: {ticker.upper()} at ${price:.2f} ({rule_type} {threshold})."
    try:
        smtp_send_html(sender, password, to_email, subj, body_html, text)
        return True
    except Exception as e:
        _LOG.warning("Alert email failed: %s", e)
        return False
