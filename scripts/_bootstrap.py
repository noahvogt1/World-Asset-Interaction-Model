import os, sys

# Ensure repository root is on sys.path so `import worldflow` works without installation.
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
