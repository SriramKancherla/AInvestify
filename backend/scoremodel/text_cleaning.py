"""
Single source of truth for sentiment text cleaning.

The TF-IDF vectorizer in `training/sentiment_logreg.py` builds its vocabulary from
`clean_text_for_model`. Every place that later feeds text into that vectorizer at
serving time MUST use this same function, or the tokens won't match the vocabulary
and the model silently degrades (train/serve skew). The call sites are:

  - backend/scoremodel/training/sentiment_logreg.py  (training)
  - backend/scoremodel/engine.py                      (serving: dataset-tweet path)
  - backend/scoremodel/webscrapernews.py               (serving: live-news path)

This module intentionally depends on nothing but the standard library so it is cheap
and safe to import from both the app runtime and the training scripts.

Carried over unchanged from the previous Model_gens/text_cleaning.py (now archived
under old/scoremodel_old/) so the new scoremodel's sentiment artifacts - trained with
an equivalent cleaning routine (see training/sentiment_logreg.py's clean_tweet) -
tokenize the same way at serving time.
"""

import re

_URL_RE = re.compile(r"http\S+|www\S+")
_MENTION_RE = re.compile(r"@\w+")
_HTML_RE = re.compile(r"<[^>]+>")
_HASH_RE = re.compile(r"#")
_AMP_RE = re.compile(r"&amp;")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_WS_RE = re.compile(r"\s+")


def clean_text_for_model(text: str) -> str:
    """Normalize text to the form the sentiment TF-IDF vectorizer was trained on.

    Steps mirror the original tweet cleaner, plus an HTML-tag strip that is a no-op
    for tweets but needed for RSS news summaries. Non-alphabetic characters are
    removed (not replaced with spaces) to match the trained vocabulary; whitespace
    between words is preserved and collapsed to single spaces.
    """
    text = str(text).lower()
    text = _URL_RE.sub("", text)        # strip URLs
    text = _MENTION_RE.sub("", text)    # strip @mentions
    text = _HTML_RE.sub(" ", text)      # strip HTML tags (news RSS); no-op for tweets
    text = _HASH_RE.sub("", text)       # drop '#' but keep the hashtag word
    text = _AMP_RE.sub("and", text)     # &amp; -> and
    text = _NON_ALPHA_RE.sub("", text)  # keep only [a-z] and whitespace
    text = _WS_RE.sub(" ", text).strip()
    return text


def fix_mojibake(text: str) -> str:
    """Repair double-encoded UTF-8 text (mojibake), e.g. ``â€œ`` -> ``“``.

    The source data is UTF-8 bytes that were mis-decoded as cp1252 and re-saved, so we
    reverse it: re-encode as cp1252 to recover the original bytes, then decode as UTF-8.
    Strings that aren't representable that way (already-correct unicode, emoji, accents)
    raise during the round-trip and are returned unchanged. This is a conservative,
    dependency-free stopgap; `ftfy.fix_text` is the robust drop-in if a dependency is
    acceptable.
    """
    s = str(text)
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s
