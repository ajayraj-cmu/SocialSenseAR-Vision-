"""
Adaptive Quality Controller

Dynamically adjusts video quality based on measured latency to maintain
real-time performance for AR headset streaming.
"""

import time
from typing import Tuple, Optional
from dataclasses import dataclass
from collections import deque
from loguru import logger

from ..config import get_latency_config, LatencyConfig


@dataclass
class QualityLevel:
    """Video quality level configuration."""

    name: str
    width: int
    height: int
    bitrate: int
    fps: Optional[int] = None


class AdaptiveQualityController:
    """
    Controls video quality based on latency measurements.

    Uses a sliding window of latency samples to make intelligent quality
    adjustment decisions, preventing oscillation while maintaining responsiveness.
    """

    def __init__(self, config: Optional[LatencyConfig] = None):
        """
        Initialize adaptive quality controller.

        Args:
            config: Latency configuration. If None, uses default.
        """
        self.config = config or get_latency_config()

        # Quality levels (from config)
        self.levels = {
            name: QualityLevel(name=name, **params)
            for name, params in self.config.quality_levels.items()
        }

        # Current quality
        self.current_level = "high"

        # Latency tracking
        self.latency_window = deque(maxlen=10)  # Last 10 samples
        self.last_adjustment_time = time.time()
        self.min_adjustment_interval = 5.0  # Minimum 5s between adjustments

        # Statistics
        self.upgrade_count = 0
        self.downgrade_count = 0

        logger.info(f"Adaptive quality controller initialized at '{self.current_level}' quality")

    def record_latency(self, latency_ms: float):
        """
        Record a latency measurement.

        Args:
            latency_ms: Measured latency in milliseconds.
        """
        self.latency_window.append(latency_ms)

    def get_average_latency(self) -> Optional[float]:
        """
        Calculate average latency from recent samples.

        Returns:
            Average latency in ms, or None if no samples.
        """
        if not self.latency_window:
            return None

        return sum(self.latency_window) / len(self.latency_window)

    def should_adjust_quality(self) -> bool:
        """
        Determine if quality should be adjusted.

        Returns:
            True if adjustment is needed and allowed.
        """
        # Need sufficient samples
        if len(self.latency_window) < 5:
            return False

        # Respect minimum adjustment interval to prevent oscillation
        time_since_last = time.time() - self.last_adjustment_time
        if time_since_last < self.min_adjustment_interval:
            return False

        avg_latency = self.get_average_latency()
        if avg_latency is None:
            return False

        # Check if we should downgrade
        if avg_latency > self.config.quality_degradation_threshold_ms:
            if self.current_level != "low":  # Can downgrade
                return True

        # Check if we should upgrade
        if avg_latency < self.config.quality_upgrade_threshold_ms:
            if self.current_level != "high":  # Can upgrade
                return True

        return False

    def adjust_quality(self) -> Tuple[str, QualityLevel]:
        """
        Adjust quality based on current latency.

        Returns:
            Tuple of (new_level_name, new_quality_level).
        """
        avg_latency = self.get_average_latency()

        if avg_latency is None:
            return self.current_level, self.levels[self.current_level]

        old_level = self.current_level

        # Downgrade if latency too high
        if avg_latency > self.config.quality_degradation_threshold_ms:
            if self.current_level == "high":
                self.current_level = "medium"
                self.downgrade_count += 1
            elif self.current_level == "medium":
                self.current_level = "low"
                self.downgrade_count += 1

        # Upgrade if latency low enough
        elif avg_latency < self.config.quality_upgrade_threshold_ms:
            if self.current_level == "low":
                self.current_level = "medium"
                self.upgrade_count += 1
            elif self.current_level == "medium":
                self.current_level = "high"
                self.upgrade_count += 1

        if old_level != self.current_level:
            self.last_adjustment_time = time.time()
            logger.info(
                f"Quality adjusted: {old_level} → {self.current_level} "
                f"(avg latency: {avg_latency:.1f}ms)"
            )

        return self.current_level, self.levels[self.current_level]

    def get_current_quality(self) -> QualityLevel:
        """
        Get current quality level configuration.

        Returns:
            Current QualityLevel.
        """
        return self.levels[self.current_level]

    def should_drop_frame(self, current_latency_ms: float) -> bool:
        """
        Determine if current frame should be dropped to reduce latency.

        Args:
            current_latency_ms: Current frame latency.

        Returns:
            True if frame should be dropped.
        """
        if not self.config.enable_frame_dropping:
            return False

        return current_latency_ms > self.config.frame_drop_latency_threshold_ms

    def get_stats(self) -> dict:
        """
        Get controller statistics.

        Returns:
            Dictionary of statistics.
        """
        return {
            "current_level": self.current_level,
            "average_latency_ms": self.get_average_latency(),
            "latency_samples": len(self.latency_window),
            "upgrade_count": self.upgrade_count,
            "downgrade_count": self.downgrade_count,
            "time_since_last_adjustment": time.time() - self.last_adjustment_time,
        }

    def reset(self):
        """Reset controller to initial state."""
        self.current_level = "high"
        self.latency_window.clear()
        self.last_adjustment_time = time.time()
        logger.info("Quality controller reset to high quality")
