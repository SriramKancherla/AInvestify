"""Start the complete local AInvestify demonstration with one command."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    env = dict(os.environ, START_REVIEWER_LABS="true")
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=ROOT,
        env=env,
    )
    frontend = subprocess.Popen(["npm", "run", "dev", "--", "--host", "127.0.0.1"], cwd=ROOT / "frontend", env=env)
    processes = [frontend, api]
    print("AInvestify: http://127.0.0.1:8080", flush=True)
    print("Reviewer tools: backend walkthrough :8766 · inference pipeline :8767", flush=True)
    try:
        while api.poll() is None and frontend.poll() is None:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    main()
