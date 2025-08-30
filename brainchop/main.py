"""
BrainChop CLI entry point.

This module serves as the main entry point for the brainchop command-line tool.
All CLI logic is implemented in cli.py.
"""

from brainchop.cli import run_cli


def main():
    """Entry point for the brainchop console script."""
    run_cli()


if __name__ == "__main__":
    main()