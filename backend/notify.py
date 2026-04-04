from __future__ import annotations

import html
import logging
import os
import smtplib
import socket
import ssl
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


def _env_flag(key: str, default: bool = False) -> bool:
    raw = (os.getenv(key) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _smtp_force_ipv4() -> bool:
    """Many hosts (e.g. some cloud runtimes) fail IPv6 SMTP with errno 101; IPv4 works."""
    return _env_flag("EMAIL_SMTP_FORCE_IPV4", True)


def _socket_connect_ipv4(host: str, port: int, timeout: float) -> socket.socket:
    infos = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
    last_err: OSError | None = None
    for _fam, socktype, proto, _canon, sa in infos:
        sock: socket.socket | None = None
        try:
            sock = socket.socket(socket.AF_INET, socktype, proto)
            sock.settimeout(timeout)
            sock.connect(sa)
            return sock
        except OSError as e:
            last_err = e
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
    if last_err:
        raise OSError(f"No IPv4 route to {host}:{port}") from last_err
    raise OSError(f"No IPv4 route to {host}:{port}")


class _SMTP_IPV4(smtplib.SMTP):
    def _get_socket(self, host: str, port: int, timeout: Any) -> socket.socket:
        t = 20.0 if timeout is None else float(timeout)
        return _socket_connect_ipv4(host, port, t)


class _SMTP_SSL_IPV4(smtplib.SMTP_SSL):
    def _get_socket(self, host: str, port: int, timeout: Any) -> socket.socket:
        t = 20.0 if timeout is None else float(timeout)
        raw = _socket_connect_ipv4(host, port, t)
        ctx = self.context if getattr(self, "context", None) else ssl.create_default_context()
        return ctx.wrap_socket(raw, server_hostname=host)


def smtp_deliver_message(sender: str, password: str, msg: EmailMessage, *, timeout: float = 20) -> None:
    """Send a prepared message via Gmail-style SMTP (465 SSL or 587 STARTTLS)."""
    host = (os.getenv("EMAIL_SMTP_HOST") or "smtp.gmail.com").strip()
    port = _env_int("EMAIL_SMTP_PORT", 587)
    if _smtp_force_ipv4():
        if port == 465:
            with _SMTP_SSL_IPV4(host, port, timeout=timeout) as smtp:
                smtp.login(sender, password)
                smtp.send_message(msg)
        else:
            with _SMTP_IPV4(host, port, timeout=timeout) as smtp:
                if port == 587:
                    smtp.starttls()
                smtp.login(sender, password)
                smtp.send_message(msg)
    else:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=timeout) as smtp:
                smtp.login(sender, password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=timeout) as smtp:
                if port == 587:
                    smtp.starttls()
                smtp.login(sender, password)
                smtp.send_message(msg)


def smtp_send_html(sender: str, password: str, to_addr: str, subject: str, html_body: str, text_fallback: str) -> None:
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(text_fallback)
    msg.add_alternative(html_body, subtype="html")
    smtp_deliver_message(sender, password, msg, timeout=20)


def send_price_alert_email(to_email: str, ticker: str, price: float, rule_type: str, threshold: float) -> tuple[bool, str | None]:
    sender = (os.getenv("EMAIL_USER") or "").strip()
    password = (os.getenv("EMAIL_PASS") or "").strip().replace(" ", "")
    if not sender or not password or not to_email:
        _LOG.warning("Skipping alert email: missing EMAIL_USER/EMAIL_PASS or recipient.")
        return False, "missing_smtp_credentials_or_recipient"
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
        return True, None
    except Exception as e:
        _LOG.warning("Alert email failed: %s", e)
        detail = f"{type(e).__name__}: {e}"
        return False, detail[:280]
