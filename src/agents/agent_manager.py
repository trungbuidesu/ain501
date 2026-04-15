"""Parallel agent execution with per-agent timeouts (Tier 1)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable

from src.agents.face_pose_agent import FacePoseAgent
from src.agents.object_agent import ObjectAgent
from src.agents.ocr_agent import OcrAgent
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

    object_agent: ObjectAgent
    action_agent: Any | None
    ocr_agent: OcrAgent | None
    face_pose_agent: FacePoseAgent | None
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

        results: dict[str, AgentResult] = {}
        futures: dict[str, Any] = {}

        def wrap(name: str, fn: Callable[[], Any], timeout_s: float) -> None:
            fut = self._pool.submit(fn)
            futures[name] = (fut, timeout_s)

        if "object" in active:
            wrap(
                "object",
                lambda: self.object_agent.detect_instances(frame_rgb),
                self.timeout_object_s,
            )
        if "action" in active and self.action_agent is not None and action_packets:
            wrap(
                "action",
                lambda: self.action_agent.predict(action_packets),
                self.timeout_action_s,
            )
        if "ocr" in active and self.ocr_agent is not None:
            wrap(
                "ocr",
                lambda: self.ocr_agent.read_text(frame_rgb),
                self.timeout_ocr_s,
            )
        if "face_pose" in active and self.face_pose_agent is not None:
            wrap(
                "face_pose",
                lambda: self.face_pose_agent.analyze(frame_rgb),
                self.timeout_face_pose_s,
            )

        import time

        for name, (fut, timeout_s) in futures.items():
            t0 = time.perf_counter()
            try:
                payload = fut.result(timeout=timeout_s)
                latency = (time.perf_counter() - t0) * 1000.0
                results[name] = AgentResult(
                    agent=name,
                    status="ok",
                    latency_ms=latency,
                    payload=payload,
                    error=None,
                )
                self.stats.success += 1
            except FutureTimeout:
                latency = (time.perf_counter() - t0) * 1000.0
                results[name] = AgentResult(
                    agent=name,
                    status="timeout",
                    latency_ms=latency,
                    payload=None,
                    error="timeout",
                )
                self.stats.timeout += 1
            except Exception as exc:
                latency = (time.perf_counter() - t0) * 1000.0
                results[name] = AgentResult(
                    agent=name,
                    status="error",
                    latency_ms=latency,
                    payload=None,
                    error=str(exc),
                )
                self.stats.error += 1

        return results
