"""Global hotkeys via pynput (optional dependency)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from src.ui.state import DemoControlState


def has_pynput() -> bool:
    """Return True when the optional hotkey backend is installed."""

    try:
        import pynput.keyboard  # noqa: F401
    except ImportError:
        return False
    return True


def _to_pynput_combo(spec: str) -> str:
    """Map ``ctrl+shift+m`` to ``<ctrl>+<shift>+m``."""

    parts = [p.strip().lower() for p in spec.replace(" ", "").split("+")]
    out: list[str] = []
    for p in parts:
        if p == "ctrl":
            out.append("<ctrl>")
        elif p == "shift":
            out.append("<shift>")
        elif p == "alt":
            out.append("<alt>")
        elif p == "win" or p == "cmd":
            out.append("<cmd>")
        elif len(p) == 1:
            out.append(p)
        else:
            out.append(f"<{p}>")
    return "+".join(out)


def start_hotkey_listener(
    mapping: dict[str, str],
    state: DemoControlState,
    *,
    on_describe: Callable[[], None] | None = None,
) -> tuple[Any, threading.Thread] | None:
    """Start non-blocking global hotkeys; returns (listener, join_thread) or None."""

    try:
        from pynput.keyboard import GlobalHotKeys
    except ImportError:
        return None

    bindings: dict[str, Callable[[], None]] = {}

    def _mute() -> None:
        state.muted = not state.muted

    def _describe() -> None:
        state.force_describe = True
        if on_describe is not None:
            on_describe()

    def _read_text() -> None:
        state.read_text_request = True

    def _toggle() -> None:
        state.rag_enabled = not state.rag_enabled

    if mute := mapping.get("mute"):
        bindings[_to_pynput_combo(str(mute))] = _mute
    if dn := mapping.get("describe_now"):
        bindings[_to_pynput_combo(str(dn))] = _describe
    if rt := mapping.get("read_text"):
        bindings[_to_pynput_combo(str(rt))] = _read_text
    if tg := mapping.get("toggle"):
        bindings[_to_pynput_combo(str(tg))] = _toggle

    if not bindings:
        return None

    hk = GlobalHotKeys(bindings)

    def _join() -> None:
        hk.join()

    hk.start()
    joiner = threading.Thread(target=_join, name="HotkeyJoin", daemon=True)
    joiner.start()
    return (hk, joiner)


def stop_hotkey_listener(hk: Any) -> None:
    """Stop listener created by ``start_hotkey_listener``."""

    try:
        hk.stop()
    except Exception:
        pass
