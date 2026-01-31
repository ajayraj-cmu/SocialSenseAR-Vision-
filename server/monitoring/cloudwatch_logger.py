"""
CloudWatch Logger

Publishes custom metrics and logs to AWS CloudWatch for monitoring and alerting.
"""

import time
from typing import Optional, Dict, List
from dataclasses import dataclass
import asyncio
from loguru import logger

from ..config import get_config


@dataclass
class Metric:
    """CloudWatch metric data point."""

    name: str
    value: float
    unit: str = "None"
    timestamp: Optional[float] = None
    dimensions: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class CloudWatchLogger:
    """
    CloudWatch metrics and logs publisher.

    Batches metrics for efficient publishing and provides async API.
    """

    def __init__(self):
        """Initialize CloudWatch logger."""
        self.config = get_config()
        self.enabled = self.config.ENABLE_CLOUDWATCH

        self.namespace = self.config.CLOUDWATCH_NAMESPACE
        self.log_group = self.config.CLOUDWATCH_LOG_GROUP

        # Metric buffer
        self.metric_buffer: List[Metric] = []
        self.buffer_lock = asyncio.Lock()
        self.max_buffer_size = 20  # CloudWatch max is 20 metrics per request

        # CloudWatch clients
        self.cloudwatch_client = None
        self.logs_client = None

        if self.enabled:
            self._init_clients()

    def _init_clients(self):
        """Initialize AWS CloudWatch clients."""
        try:
            import boto3

            self.cloudwatch_client = boto3.client(
                "cloudwatch", region_name=self.config.AWS_REGION
            )
            self.logs_client = boto3.client("logs", region_name=self.config.AWS_REGION)

            # Ensure log group exists
            self._create_log_group_if_not_exists()

            logger.success("CloudWatch integration initialized")

        except Exception as e:
            logger.error(f"Failed to initialize CloudWatch: {e}")
            self.enabled = False

    def _create_log_group_if_not_exists(self):
        """Create CloudWatch log group if it doesn't exist."""
        try:
            self.logs_client.create_log_group(logGroupName=self.log_group)
            logger.info(f"Created CloudWatch log group: {self.log_group}")
        except self.logs_client.exceptions.ResourceAlreadyExistsException:
            pass  # Already exists
        except Exception as e:
            logger.warning(f"Could not create log group: {e}")

    async def publish_metric(
        self,
        name: str,
        value: float,
        unit: str = "None",
        dimensions: Optional[Dict[str, str]] = None,
    ):
        """
        Publish a metric to CloudWatch.

        Args:
            name: Metric name.
            value: Metric value.
            unit: Metric unit (Count, Milliseconds, Percent, etc.).
            dimensions: Optional metric dimensions for filtering.
        """
        if not self.enabled:
            return

        metric = Metric(name=name, value=value, unit=unit, dimensions=dimensions)

        async with self.buffer_lock:
            self.metric_buffer.append(metric)

            # Flush if buffer is full
            if len(self.metric_buffer) >= self.max_buffer_size:
                await self._flush_metrics()

    async def _flush_metrics(self):
        """Flush buffered metrics to CloudWatch."""
        if not self.metric_buffer:
            return

        metrics_to_send = self.metric_buffer[: self.max_buffer_size]
        self.metric_buffer = self.metric_buffer[self.max_buffer_size :]

        # Convert to CloudWatch format
        metric_data = []
        for metric in metrics_to_send:
            data_point = {
                "MetricName": metric.name,
                "Value": metric.value,
                "Unit": metric.unit,
                "Timestamp": metric.timestamp,
            }

            if metric.dimensions:
                data_point["Dimensions"] = [
                    {"Name": k, "Value": v} for k, v in metric.dimensions.items()
                ]

            metric_data.append(data_point)

        # Send to CloudWatch
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.cloudwatch_client.put_metric_data(
                    Namespace=self.namespace, MetricData=metric_data
                ),
            )

            logger.debug(f"Published {len(metric_data)} metrics to CloudWatch")

        except Exception as e:
            logger.error(f"Failed to publish metrics to CloudWatch: {e}")

    async def flush(self):
        """Manually flush all buffered metrics."""
        async with self.buffer_lock:
            while self.metric_buffer:
                await self._flush_metrics()

    # Convenience methods for common metrics

    async def log_latency(self, latency_ms: float, session_id: Optional[str] = None):
        """Log frame processing latency."""
        dimensions = {"SessionID": session_id} if session_id else None
        await self.publish_metric("FrameLatency", latency_ms, "Milliseconds", dimensions)

    async def log_frame_count(self, count: int, session_id: Optional[str] = None):
        """Log processed frame count."""
        dimensions = {"SessionID": session_id} if session_id else None
        await self.publish_metric("FramesProcessed", count, "Count", dimensions)

    async def log_gpu_utilization(self, utilization: float):
        """Log GPU utilization percentage."""
        await self.publish_metric("GPUUtilization", utilization, "Percent")

    async def log_session_count(self, count: int):
        """Log active session count."""
        await self.publish_metric("ActiveSessions", count, "Count")

    async def log_error_rate(self, rate: float):
        """Log error rate percentage."""
        await self.publish_metric("ErrorRate", rate, "Percent")

    async def log_bandwidth(self, bandwidth_mbps: float, direction: str = "outbound"):
        """Log network bandwidth usage."""
        dimensions = {"Direction": direction}
        await self.publish_metric("Bandwidth", bandwidth_mbps, "None", dimensions)


# Global CloudWatch logger instance
_cloudwatch_logger: Optional[CloudWatchLogger] = None


def get_cloudwatch_logger() -> CloudWatchLogger:
    """Get the global CloudWatch logger instance."""
    global _cloudwatch_logger
    if _cloudwatch_logger is None:
        _cloudwatch_logger = CloudWatchLogger()
    return _cloudwatch_logger
