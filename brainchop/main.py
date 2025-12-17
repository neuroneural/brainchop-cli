"""
Redirect to cli.py for backwards compatibility.

The main CLI implementation is now in brainchop.cli.
"""

from brainchop.cli import main

if __name__ == "__main__":
    main()
