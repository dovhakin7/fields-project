"""Path setup shared by the milestone scripts.  PROVIDED.

Run a script normally to exercise YOUR code:

    python scripts/m1_navigation_field.py

Instructors can run the same script against the reference implementation:

    EMLAB_IMPL=solution python scripts/m1_navigation_field.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)

if os.environ.get("EMLAB_IMPL", "student").lower() == "solution":
    try:
        import emlab_ref  # noqa: F401
    except ModuleNotFoundError:
        raise SystemExit(
            "EMLAB_IMPL=solution needs the reference package 'emlab_ref', which is "
            "not part of the student kit. Unset EMLAB_IMPL and run your own code."
        )
    import emlab
    import emlab_ref.conduction
    import emlab_ref.magnetostatics
    import emlab_ref.steering
    import emlab_ref.thermal

    for name in ("magnetostatics", "steering", "conduction", "thermal"):
        mod = getattr(emlab_ref, name)
        sys.modules[f"emlab.{name}"] = mod
        setattr(emlab, name, mod)


# A milestone script calls functions you have not written yet, so the normal
# Python traceback here would be forty lines of machinery around one real line.
# This hook replaces it with the file, the function, and what to do next.
def _friendly_errors(exc_type, exc, tb):
    import traceback

    if exc_type is NotImplementedError:
        label = str(exc)                      # e.g. "Milestone 1: coil_pair_field"
        func = label.split(":")[-1].strip()
        frames = traceback.extract_tb(tb)
        where = frames[-1]
        print("")
        print("=" * 74)
        print(f"  STOP: {func}() has no code in it yet.")
        print("=" * 74)
        print(f"  file : {where.filename}")
        print(f"  line : {where.lineno}")
        print("")
        print("  What to do:")
        print(f"    1. Open that file and find   def {func}(...)")
        print("    2. Read the RECIPE in its docstring. It names every numpy call.")
        print("    3. Delete the  raise NotImplementedError(...)  line.")
        print("    4. Write the body, then run   python check.py")
        print("")
        print("  Stuck on the Python rather than the physics?")
        print("    MATLAB_TO_PYTHON.pdf has the equivalent of every call you need.")
        print("=" * 74)
        print("")
        return
    traceback.print_exception(exc_type, exc, tb)


sys.excepthook = _friendly_errors
