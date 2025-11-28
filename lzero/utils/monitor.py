"""
Performance monitoring utilities for LightZero training.

This module provides tools to monitor resource usage (CPU, GPU, memory)
during training, particularly for collecting and analyzing performance data.
"""

import logging
import os
import time
from contextlib import contextmanager
from typing import Dict, Optional, Tuple
import psutil

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor CPU, GPU, and memory usage during execution."""

    def __init__(self, name: str = "Operation"):
        """
        Initialize resource monitor.

        Args:
            name: Name of the operation being monitored (for logging)
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.start_cpu_percent = None
        self.end_cpu_percent = None
        self.start_memory = None
        self.end_memory = None
        self.start_gpu_memory = None
        self.end_gpu_memory = None

    def __enter__(self):
        """Start monitoring when entering context."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring when exiting context."""
        self.stop()

    def start(self):
        """Start monitoring resources."""
        self.start_time = time.time()

        # CPU and Memory
        process = psutil.Process()
        self.start_cpu_percent = process.cpu_percent(interval=0.1)
        self.start_memory = process.memory_info().rss / (1024 ** 3)  # GB

        # GPU (if available)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self.start_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        except ImportError:
            pass

    def stop(self):
        """Stop monitoring and collect metrics."""
        self.end_time = time.time()

        # CPU and Memory
        process = psutil.Process()
        self.end_cpu_percent = process.cpu_percent(interval=0.1)
        self.end_memory = process.memory_info().rss / (1024 ** 3)  # GB

        # GPU
        try:
            import torch
            if torch.cuda.is_available():
                self.end_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        except ImportError:
            pass

    def get_metrics(self) -> Dict[str, float]:
        """
        Get collected metrics.

        Returns:
            Dictionary containing:
                - duration: Total time in seconds
                - cpu_percent: Average CPU usage percentage
                - memory_rss_gb: Memory resident set size in GB
                - memory_delta_gb: Memory change during operation
                - gpu_memory_gb: GPU memory allocated in GB
                - gpu_memory_delta_gb: GPU memory change
        """
        if self.start_time is None or self.end_time is None:
            logger.warning("Monitor not properly started/stopped")
            return {}

        metrics = {
            "duration_sec": self.end_time - self.start_time,
            "cpu_percent": (self.start_cpu_percent + self.end_cpu_percent) / 2,
        }

        if self.start_memory is not None and self.end_memory is not None:
            metrics["memory_rss_start_gb"] = self.start_memory
            metrics["memory_rss_end_gb"] = self.end_memory
            metrics["memory_rss_delta_gb"] = self.end_memory - self.start_memory

        if self.start_gpu_memory is not None and self.end_gpu_memory is not None:
            metrics["gpu_memory_start_gb"] = self.start_gpu_memory
            metrics["gpu_memory_end_gb"] = self.end_gpu_memory
            metrics["gpu_memory_delta_gb"] = self.end_gpu_memory - self.start_gpu_memory

        return metrics

    def log_metrics(self, log_level: int = logging.INFO):
        """
        Log collected metrics.

        Args:
            log_level: Logging level (default: INFO)
        """
        metrics = self.get_metrics()
        if not metrics:
            return

        # Format message
        msg_parts = [f"\n{'='*60}", f"Performance Report: {self.name}"]
        msg_parts.append(f"{'='*60}")

        msg_parts.append(f"Duration: {metrics['duration_sec']:.2f} seconds")
        msg_parts.append(f"CPU Usage: {metrics['cpu_percent']:.1f}%")

        if "memory_rss_start_gb" in metrics:
            msg_parts.append(
                f"Memory (RSS): {metrics['memory_rss_start_gb']:.2f}GB → "
                f"{metrics['memory_rss_end_gb']:.2f}GB "
                f"(Δ {metrics['memory_rss_delta_gb']:+.2f}GB)"
            )

        if "gpu_memory_start_gb" in metrics:
            msg_parts.append(
                f"GPU Memory: {metrics['gpu_memory_start_gb']:.2f}GB → "
                f"{metrics['gpu_memory_end_gb']:.2f}GB "
                f"(Δ {metrics['gpu_memory_delta_gb']:+.2f}GB)"
            )

        msg_parts.append(f"{'='*60}\n")
        msg = "\n".join(msg_parts)
        logger.log(log_level, msg)

    def __str__(self) -> str:
        """String representation of metrics."""
        metrics = self.get_metrics()
        if not metrics:
            return f"ResourceMonitor({self.name}): Not executed"

        lines = [f"ResourceMonitor({self.name}):"]
        lines.append(f"  Duration: {metrics['duration_sec']:.2f}s")
        lines.append(f"  CPU: {metrics['cpu_percent']:.1f}%")

        if "memory_rss_delta_gb" in metrics:
            lines.append(
                f"  Memory: {metrics['memory_rss_start_gb']:.2f} → "
                f"{metrics['memory_rss_end_gb']:.2f}GB "
                f"(Δ{metrics['memory_rss_delta_gb']:+.2f}GB)"
            )

        if "gpu_memory_delta_gb" in metrics:
            lines.append(
                f"  GPU Memory: {metrics['gpu_memory_start_gb']:.2f} → "
                f"{metrics['gpu_memory_end_gb']:.2f}GB "
                f"(Δ{metrics['gpu_memory_delta_gb']:+.2f}GB)"
            )

        return "\n".join(lines)


@contextmanager
def monitor_performance(name: str = "Operation", log_level: int = logging.INFO):
    """
    Context manager for monitoring performance.

    Usage:
        with monitor_performance("collector.collect"):
            new_data = collector.collect(...)

    Args:
        name: Operation name for logging
        log_level: Logging level
    """
    monitor = ResourceMonitor(name)
    monitor.start()
    try:
        yield monitor
    finally:
        monitor.stop()
        monitor.log_metrics(log_level)


def get_gpu_info() -> Dict[str, any]:
    """
    Get detailed GPU information.

    Returns:
        Dictionary with GPU metrics if available, empty dict otherwise.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return {}

        device_count = torch.cuda.device_count()
        info = {
            "device_count": device_count,
            "current_device": torch.cuda.current_device(),
        }

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            info[f"device_{i}_name"] = props.name
            info[f"device_{i}_memory_gb"] = props.total_memory / (1024 ** 3)

        return info
    except ImportError:
        return {}


def print_system_info():
    """Print system and GPU information at startup."""
    logger.info("\n" + "="*60)
    logger.info("System Information")
    logger.info("="*60)

    # CPU
    cpu_count = psutil.cpu_count(logical=False)
    logger.info(f"Physical CPUs: {cpu_count}")
    logger.info(f"Logical CPUs: {psutil.cpu_count(logical=True)}")
    logger.info(f"CPU Frequency: {psutil.cpu_freq().current:.2f} MHz")

    # Memory
    mem = psutil.virtual_memory()
    logger.info(f"Total Memory: {mem.total / (1024**3):.2f}GB")
    logger.info(f"Available Memory: {mem.available / (1024**3):.2f}GB")

    # GPU
    gpu_info = get_gpu_info()
    if gpu_info:
        logger.info(f"GPUs Found: {gpu_info['device_count']}")
        for i in range(gpu_info['device_count']):
            logger.info(
                f"  GPU {i}: {gpu_info[f'device_{i}_name']} "
                f"({gpu_info[f'device_{i}_memory_gb']:.2f}GB)"
            )
    else:
        logger.info("No GPU available")

    logger.info("="*60 + "\n")
