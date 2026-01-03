"""Progress tracking with GPU monitoring and ETA calculation."""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, List
from threading import Lock


@dataclass
class ProgressPhase:
    """A phase of work in the pipeline."""
    name: str
    weight: float  # Relative weight (all phases should sum to 1.0)
    total_steps: int = 0
    current_step: int = 0


class ProgressTracker:
    """Track progress across multiple phases with GPU monitoring and ETA.

    Usage:
        tracker = ProgressTracker(callback=lambda p, eta, gpu: update_ui(p, eta, gpu))
        tracker.set_phases([
            ("Training", 0.1),
            ("Tracking", 0.5),
            ("OCR", 0.2),
            ("Video Generation", 0.2),
        ])

        tracker.start_phase("Training", total=100)
        for i in range(100):
            do_work()
            tracker.update(1)

        tracker.start_phase("Tracking", total=1000)
        # ...
    """

    def __init__(
        self,
        callback: Optional[Callable[[float, str, float], None]] = None,
        update_interval: float = 0.5,  # Minimum seconds between updates
    ):
        """Initialize progress tracker.

        Args:
            callback: Function called with (progress_pct, eta_str, gpu_pct)
            update_interval: Minimum seconds between callback invocations
        """
        self.callback = callback
        self.update_interval = update_interval

        self._phases: List[ProgressPhase] = []
        self._phase_index = -1
        self._start_time: Optional[float] = None
        self._last_update_time = 0
        self._lock = Lock()

        # GPU monitoring
        self._pynvml_available = False
        self._nvml_handle = None
        self._init_gpu_monitoring()

    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring via nvidia-ml-py."""
        try:
            # Use nvidia-ml-py directly (pynvml is deprecated)
            from pynvml import smi
            smi.nvidia_smi.getInstance()
            self._pynvml_available = True
        except Exception:
            # Fallback to raw pynvml API
            try:
                import pynvml
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._pynvml_available = True
            except Exception:
                self._pynvml_available = False
                self._nvml_handle = None

    def get_gpu_usage(self) -> float:
        """Get current GPU utilization percentage (0-100)."""
        if not self._pynvml_available:
            return 0.0

        try:
            import pynvml
            if self._nvml_handle is None:
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            return float(util.gpu)
        except Exception:
            return 0.0

    def get_gpu_memory(self) -> tuple:
        """Get GPU memory usage (used_gb, total_gb)."""
        if not self._pynvml_available:
            return (0.0, 0.0)

        try:
            import pynvml
            if self._nvml_handle is None:
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            return (mem.used / 1e9, mem.total / 1e9)
        except Exception:
            return (0.0, 0.0)

    def set_phases(self, phases: List[tuple]):
        """Set the phases of work.

        Args:
            phases: List of (name, weight) tuples
        """
        with self._lock:
            self._phases = [
                ProgressPhase(name=name, weight=weight)
                for name, weight in phases
            ]
            self._phase_index = -1
            self._start_time = time.time()

    def start_phase(self, name: str, total: int = 0):
        """Start a new phase.

        Args:
            name: Phase name (must match one from set_phases)
            total: Total steps in this phase
        """
        with self._lock:
            # Find phase by name
            for i, phase in enumerate(self._phases):
                if phase.name == name:
                    self._phase_index = i
                    phase.total_steps = total
                    phase.current_step = 0
                    break

            if self._start_time is None:
                self._start_time = time.time()

        self._do_callback()

    def update(self, steps: int = 1):
        """Update progress by a number of steps."""
        with self._lock:
            if 0 <= self._phase_index < len(self._phases):
                phase = self._phases[self._phase_index]
                phase.current_step = min(phase.current_step + steps, phase.total_steps)

        self._do_callback()

    def set_step(self, step: int):
        """Set current step directly."""
        with self._lock:
            if 0 <= self._phase_index < len(self._phases):
                phase = self._phases[self._phase_index]
                phase.current_step = min(step, phase.total_steps)

        self._do_callback()

    def get_overall_progress(self) -> float:
        """Get overall progress as percentage (0-100)."""
        with self._lock:
            if not self._phases:
                return 0.0

            progress = 0.0
            for i, phase in enumerate(self._phases):
                if i < self._phase_index:
                    # Completed phase
                    progress += phase.weight * 100
                elif i == self._phase_index:
                    # Current phase
                    if phase.total_steps > 0:
                        phase_pct = phase.current_step / phase.total_steps
                    else:
                        phase_pct = 0
                    progress += phase.weight * 100 * phase_pct
                # Future phases contribute 0

            return min(100.0, progress)

    def get_current_phase_progress(self) -> tuple:
        """Get current phase name and progress (0-100)."""
        with self._lock:
            if 0 <= self._phase_index < len(self._phases):
                phase = self._phases[self._phase_index]
                if phase.total_steps > 0:
                    pct = (phase.current_step / phase.total_steps) * 100
                else:
                    pct = 0
                return (phase.name, pct, phase.current_step, phase.total_steps)
            return ("", 0, 0, 0)

    def get_eta(self) -> str:
        """Get estimated time remaining as formatted string."""
        if self._start_time is None:
            return "--:--"

        progress = self.get_overall_progress()
        if progress <= 0:
            return "--:--"

        elapsed = time.time() - self._start_time
        if elapsed < 1:
            return "--:--"

        # Estimate total time based on current progress
        estimated_total = elapsed / (progress / 100)
        remaining = estimated_total - elapsed

        if remaining < 0:
            return "00:00"
        elif remaining < 60:
            return f"00:{int(remaining):02d}"
        elif remaining < 3600:
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            return f"{mins:02d}:{secs:02d}"
        else:
            hours = int(remaining // 3600)
            mins = int((remaining % 3600) // 60)
            return f"{hours}:{mins:02d}:00"

    def _do_callback(self):
        """Invoke callback if enough time has passed."""
        if self.callback is None:
            return

        now = time.time()
        if now - self._last_update_time < self.update_interval:
            return

        self._last_update_time = now

        progress = self.get_overall_progress()
        eta = self.get_eta()
        gpu = self.get_gpu_usage()

        try:
            self.callback(progress, eta, gpu)
        except Exception:
            pass  # Don't let callback errors break tracking

    def force_callback(self):
        """Force a callback regardless of update interval."""
        if self.callback is None:
            return

        self._last_update_time = time.time()

        progress = self.get_overall_progress()
        eta = self.get_eta()
        gpu = self.get_gpu_usage()

        try:
            self.callback(progress, eta, gpu)
        except Exception:
            pass

    def complete(self):
        """Mark all progress as complete."""
        with self._lock:
            for phase in self._phases:
                phase.current_step = phase.total_steps
            self._phase_index = len(self._phases)

        self.force_callback()


class TqdmProgressAdapter:
    """Adapter that wraps tqdm to report to ProgressTracker."""

    def __init__(self, tracker: ProgressTracker, iterable, **tqdm_kwargs):
        """Create a tqdm-like iterator that reports to ProgressTracker.

        Args:
            tracker: ProgressTracker to report to
            iterable: Iterable to wrap
            **tqdm_kwargs: Arguments passed to tqdm (total, desc, etc.)
        """
        from tqdm import tqdm

        self.tracker = tracker
        self.total = tqdm_kwargs.get('total', len(iterable) if hasattr(iterable, '__len__') else 0)
        self.desc = tqdm_kwargs.get('desc', '')

        # Start the phase in tracker
        if self.desc:
            tracker.start_phase(self.desc, total=self.total)

        # Create actual tqdm for console output
        self._tqdm = tqdm(iterable, **tqdm_kwargs)
        self._count = 0

    def __iter__(self):
        for item in self._tqdm:
            yield item
            self._count += 1
            self.tracker.set_step(self._count)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._tqdm.close()
