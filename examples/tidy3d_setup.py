"""Tidy3D connection check helper.

IMPORTANT: do NOT write your API key here. Enter the key once from the terminal:
    tidy3d configure            # (CLI) prompts for the key
  or:
    python -c "import tidy3d.web as web; web.configure('YOUR_API_KEY')"
The key is stored in ~/.tidy3d/config (never share or commit that file).

Then this script verifies the connection and your credits:
    python examples/tidy3d_setup.py
"""
import sys
try:
    import tidy3d.web as web
except ImportError:
    sys.exit("Tidy3D not found: pip install tidy3d")

print("Testing the connection...")
try:
    web.test()   # silent/successful if the key is valid
    print("✓ Connection OK.")
    try:
        acc = web.account()
        print("Account / credits:", acc)
    except Exception as e:
        print("(could not fetch account info:", e, ")")
except Exception as e:
    print("✗ Connection failed:", e)
    print("Configure the key first:  tidy3d configure   (or web.configure('...'))")
