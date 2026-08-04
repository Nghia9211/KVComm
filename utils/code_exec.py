"""
code_exec.py — Code execution utilities for KVComm

Provides run_with_timeout() for execute-based evaluation of code generation tasks
(e.g. MBPP+, HumanEval+).

Ported from LatentMAS/utils.py with no modification to the logic.
Placed in KVComm/utils/ so it can be imported via:
    from utils import run_with_timeout
"""

import traceback
from multiprocessing import Process, Manager
from typing import Tuple, Optional


def run_with_timeout(code: str, timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """
    Execute `code` in a separate process with a hard timeout.

    Args:
        code:    Python code string to execute (includes test harness).
        timeout: Maximum seconds to wait before killing the process.

    Returns:
        (ok, error_msg):
            ok        – True if execution completed without exception.
            error_msg – Traceback string on failure, None on success,
                        or TimeoutError message if killed.
    """
    def _worker(ns: dict, code: str) -> None:
        try:
            local_ns: dict = {}
            exec(code, local_ns)
            ns["ok"] = True
            ns["error"] = None
        except Exception:
            ns["ok"] = False
            ns["error"] = traceback.format_exc()

    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=_worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            return False, f"TimeoutError: execution exceeded {timeout} seconds"
        return ns.get("ok", False), ns.get("error", None)
