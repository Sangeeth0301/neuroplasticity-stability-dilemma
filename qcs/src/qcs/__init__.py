"""Dual-Manifold ContextSync — quantum-classical continual learning for chat memory."""

from __future__ import annotations

import sys

__version__ = "0.1.0"

# Windows consoles default to cp1252, which cannot encode the maths symbols this
# project prints everywhere (‖ψ⟩, θ, ⊗, →). Reconfiguring once here keeps every script
# and notebook readable instead of scattering .encode() calls through the codebase.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # non-reconfigurable stream (pytest capture, pipes)
        pass
