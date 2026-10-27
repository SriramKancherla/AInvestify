#!/bin/zsh
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python3.11 backend/run_website.py
