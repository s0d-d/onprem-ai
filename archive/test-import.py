# test_import.py
import sys
print(sys.path)
try:
    import livekit.agents
    print("Successfully imported livekit.agents")
except ImportError as e:
    print(f"Import error: {e}")
    import importlib
    print(importlib.util.find_spec("livekit"))
