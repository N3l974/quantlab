from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="ohlcv-optimizer-v1")
    p.add_argument("command", nargs="?", default="help", choices=["help"], help="Command")
    _ = p.parse_args()

    print("Use: streamlit run streamlit_app.py (from the project folder)")
