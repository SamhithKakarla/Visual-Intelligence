"""Backward-compatible entry point for the end-to-end pipeline."""

from visual_intelligence.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
