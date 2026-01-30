#!/usr/bin/env python3
"""
Quest Fast Capture - Legacy compatibility wrapper.

This file maintains backwards compatibility with the original fast_capture.py.
It now uses the modular architecture under the hood.

For new development, use main.py instead:
    python main.py --preset QUALITY

This wrapper provides the same behavior as the original script.
"""

# Simply run the new modular main
if __name__ == "__main__":
    from main import main
    main()
