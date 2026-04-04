from __future__ import annotations

import base64
import html
import logging
import os
import smtplib
import socket
import ssl
from email.message import EmailMessage
from email.utils import getaddresses
from typing import Any

import requests

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


def _resend_api_key() -> str:
    return (os.getenv("RESEND_API_KEY") or "").strip()


def outbound_email_configured() -> bool:
    """True if we can send mail via Resend HTTPS or classic SMTP credentials."""
    if _resend_api_key():
        return True
    u = (os.getenv("EMAIL_USER") or "").strip()
    p = (os.getenv("EMAIL_PASS") or "").strip()
    return bool(u and p)


def _resend_from_address(fallback_sender: str) -> str:
    return (os.getenv("RESEND_FROM_EMAIL") or fallback_sender or "onboarding@resend.dev").strip()


def _smtp_force_ipv4() -> bool:
    """Opt-in: some networks break IPv6 SMTP; Render free tier blocks SMTP entirely (use Resend)."""
    return _env_flag("EMAIL_SMTP_FORCE_IPV4", False)


def send_via_resend(
    *,
    to_addrs: list[str],
    subject: str,
    text: str,
    html_body: str | None = None,
    from_addr: str | None = None,
    attachments: list[tuple[str, bytes]] | None = None,
) -> None:
    """Send via Resend HTTPS API (works on hosts that block outbound SMTP, e.g. Render free web services)."""
    api_key = _resend_api_key()
    if not api_key:
        raise RuntimeError("RESEND_API_KEY is not set.")
    if not to_addrs:
        raise ValueError("No recipients.")
    from_email = _resend_from_address(from_addr or "")
    body: dict[str, Any] = {
        "from": from_email,
        "to": to_addrs[:50],
        "subject": subject,
    }
    if html_body:
        body["html"] = html_body
    if text:
        body["text"] = text
    elif html_body:
        body["text"] = "HTML email — use an HTML-capable client."
    if attachments:
        body["attachments"] = [
            {"filename": fn, "content": base64.b64encode(raw).decode("ascii")} for fn, raw in attachments
        ]
    r = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=30,
    )
    if r.status_code >= 400:
        _LOG.warning("Resend API error: %s %s", r.status_code, r.text[:500])
        raise RuntimeError(f"Resend HTTP {r.status_code}: {r.text[:220]}")


def _extract_from_email_message(msg: EmailMessage) -> tuple[list[str], str, str, str, list[tuple[str, bytes]]]:
    to_header = msg.get("To", "") or ""
    recipients = [a for _, a in getaddresses([to_header]) if a]
    subject = msg.get("Subject", "") or ""
    text_body = ""
    html_body = ""
    attachments: list[tuple[str, bytes]] = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            cdisp = str(part.get("Content-Disposition") or "")
            if "attachment" in cdisp.lower():
                fn = part.get_filename() or "attachment.bin"
                raw = part.get_payload(decode=True)
                if isinstance(raw, bytes):
                    attachments.append((fn, raw))
            else:
                ctype = part.get_content_type()
                if ctype == "text/plain" and not text_body:
                    payload = part.get_payload(decode=True)
                    text_body = (
                        payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else (part.get_content() or "")
                    )
                elif ctype == "text/html" and not html_body:
                    payload = part.get_payload(decode=True)
                    html_body = (
                        payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else (part.get_content() or "")
                    )
    else:
        ctype = msg.get_content_type()
        if ctype == "text/plain":
            text_body = msg.get_content() or ""
        elif ctype == "text/html":
            html_body = msg.get_content() or ""
    if not text_body and not html_body:
        text_body = "(no body)"
    return recipients, subject, text_body, html_body, attachments


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
    """Send via Resend (if RESEND_API_KEY) or Gmail-style SMTP (465 SSL or 587 STARTTLS)."""
    if _resend_api_key():
        recipients, subject, text_body, html_body, attachments = _extract_from_email_message(msg)
        if not recipients:
            raise ValueError("Email message has no To: recipients.")
        send_via_resend(
            to_addrs=recipients,
            subject=subject,
            text=text_body,
            html_body=html_body or None,
            from_addr=sender or None,
            attachments=attachments or None,
        )
        return

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
    if _resend_api_key():
        send_via_resend(
            to_addrs=[to_addr],
            subject=subject,
            text=text_fallback,
            html_body=html_body,
            from_addr=sender or None,
            attachments=None,
        )
        return
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
    use_resend = bool(_resend_api_key())
    if not to_email:
        return False, "missing_recipient"
    if not use_resend and (not sender or not password):
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
        if not use_resend and "No IPv4 route" in detail:
            detail += " (Render free web services block SMTP; set RESEND_API_KEY and RESEND_FROM_EMAIL.)"
        return False, detail[:400]
