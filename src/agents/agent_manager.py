"""Parallel agent execution with per-agent timeouts (Tier 1)."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any

from src.capture.types import FramePacket


@dataclass(frozen=True, slots=True)
class AgentResult:
    """One agent output with timing and status."""

    agent: str
    status: str
    latency_ms: float
    payload: Any
    error: str | None = None


@dataclass
class AgentManagerStats:
    success: int = 0
    timeout: int = 0
    error: int = 0


@dataclass
class AgentManager:
    """Runs selected experts in parallel with timeouts."""

    object_agent: Any
    action_agent: Any | None
    ocr_agent: Any
    face_pose_agent: Any
    timeout_object_s: float = 2.0
    timeout_action_s: float = 2.0
    timeout_ocr_s: float = 5.0
    timeout_face_pose_s: float = 3.0
    max_workers: int = 4
    stats: AgentManagerStats = field(default_factory=AgentManagerStats)

    def __post_init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=self.max_workers)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=False)

    def run_parallel(
        self,
        *,
        frame_rgb: Any,
        active: frozenset[str],
        action_packets: list[FramePacket] | None = None,
    ) -> dict[str, AgentResult]:
        """Execute active agents; return mapping agent_name -> AgentResult."""

        import time

        results: dict[str, AgentResult] = {}
        futures: dict[str, Any] = {}
        future_to_name: dict[Any, str] = {}

        def wrap(name: str, fn: Callable[[], Any], timeout_s: float) -> None:
            fut = self._pool.submit(fn)
            futures[name] = (fut, timeout_s)
            future_to_name[fut] = name

        if "object" in active:
            wrap(
                "object",
                lambda: self.object_agent.detect_instances(frame_rgb),
                self.timeout_object_s,
            )
        action_agent = self.action_agent
        if "action" in active and action_agent is not None and action_packets:
            wrap(
                "action",
                lambda: action_agent.predict(action_packets),
                self.timeout_action_s,
            )
        ocr_agent = self.ocr_agent
        if "ocr" in active and ocr_agent is not None:
            wrap(
                "ocr",
                lambda: ocr_agent.read_text(frame_rgb),
                self.timeout_ocr_s,
            )
        face_pose_agent = self.face_pose_agent
        if "face_pose" in active and face_pose_agent is not None:
            wrap(
                "face_pose",
                lambda: face_pose_agent.analyze(frame_rgb),
                self.timeout_face_pose_s,
            )

        started_at = time.perf_counter()
        for fut in as_completed(future_to_name):
            name = future_to_name[fut]
            _, timeout_s = futures[name]
            try:
                payload = fut.result(timeout=timeout_s)
                latency = (time.perf_counter() - started_at) * 1000.0
                results[name] = AgentResult(
                    agent=name,
                    status="ok",
                    latency_ms=latency,
                    payload=payload,
                    error=None,
                )
                self.stats.success += 1
            except FutureTimeout:
                latency = timeout_s * 1000.0
                results[name] = AgentResult(
                    agent=name,
                    status="timeout",
                    latency_ms=latency,
                    payload=None,
                    error="timeout",
                )
                self.stats.timeout += 1
            except Exception as exc:
                latency = (time.perf_counter() - started_at) * 1000.0
                results[name] = AgentResult(
                    agent=name,
                    status="error",
                    latency_ms=latency,
                    payload=None,
                    error=str(exc),
                )
                self.stats.error += 1
        return results
