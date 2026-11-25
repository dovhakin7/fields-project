"""Test configuration.

By default the tests exercise the student package `emlab`.  Instructors can run
the same suite against the reference implementation with

    EMLAB_IMPL=solution pytest -q
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

    sys.modules["emlab.magnetostatics"] = emlab_ref.magnetostatics
    sys.modules["emlab.steering"] = emlab_ref.steering
    sys.modules["emlab.conduction"] = emlab_ref.conduction
    sys.modules["emlab.thermal"] = emlab_ref.thermal
    emlab.magnetostatics = emlab_ref.magnetostatics
    emlab.steering = emlab_ref.steering
    emlab.conduction = emlab_ref.conduction
    emlab.thermal = emlab_ref.thermal
