"""voicelab -- local voice synthesis + conversion.

IMPORTANT: the os.environ block below must run before torch, faiss, sklearn or
numba are imported anywhere in the process. That is why it lives here, at the
top of the package __init__, rather than in config.py or the server.

Why: torch, faiss and sklearn each vendor their own libomp.dylib. macOS will
happily load all three into one process, at which point the OpenMP runtimes
share thread state and collide -- a SIGSEGV inside __kmp_suspend_initialize_
thread under __kmp_fork_barrier, which kills the interpreter outright (no
Python traceback, the web UI just sees the socket drop as "TypeError: Load
failed"). RVC conversion touches all three libraries at once, so it hits this
reliably.

KMP_DUPLICATE_LIB_OK lets the duplicate runtimes coexist; OMP_NUM_THREADS=1
keeps them off the fork-barrier path where the crash occurs. The CPU cost is
minor here because inference runs on MPS.
"""
import os as _os

for _k, _v in (
    ("KMP_DUPLICATE_LIB_OK", "TRUE"),
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
):
    _os.environ.setdefault(_k, _v)   # setdefault: never override the operator

__version__ = "0.1.0"
