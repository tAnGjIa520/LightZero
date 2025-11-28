"""
Example: How to use the performance monitoring tool with collector.collect()

This demonstrates how to integrate ResourceMonitor into train_alphazero.py
to track the performance of data collection.
"""

import logging
from lzero.utils.monitor import monitor_performance, print_system_info

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_monitoring():
    """Example 1: Basic monitoring with context manager."""
    logger.info("\n" + "="*60)
    logger.info("Example 1: Basic Monitoring")
    logger.info("="*60)

    # Print system info at startup
    print_system_info()

    # Monitor a time-consuming operation
    with monitor_performance("Data Collection", logging.INFO) as monitor:
        # Simulate collector.collect()
        import time
        time.sleep(2)  # Simulate 2 seconds of work

    # Access metrics programmatically
    metrics = monitor.get_metrics()
    logger.info(f"Metrics dict: {metrics}")


def example_2_integration_in_training():
    """
    Example 2: How to integrate into train_alphazero.py

    This shows the actual code modification needed.
    """
    logger.info("\n" + "="*60)
    logger.info("Example 2: Integration Code (Pseudocode)")
    logger.info("="*60)

    code_snippet = """
# In train_alphazero.py, around line 125:

from lzero.utils.monitor import monitor_performance

# ... in the main training loop ...

with monitor_performance(f"collector.collect(iter={learner.train_iter})", logging.DEBUG):
    new_data = collector.collect(
        train_iter=learner.train_iter,
        policy_kwargs=collect_kwargs
    )

# Metrics are automatically logged at the end of the context block
    """

    logger.info(code_snippet)


def example_3_multiple_operations():
    """Example 3: Monitor multiple operations separately."""
    logger.info("\n" + "="*60)
    logger.info("Example 3: Monitor Multiple Operations")
    logger.info("="*60)

    import time

    operations = ["data_collection", "policy_update", "evaluation"]

    for op_name in operations:
        with monitor_performance(op_name):
            time.sleep(1)  # Simulate operation
            logger.info(f"  Executing {op_name}...")


def example_4_no_context_manager():
    """Example 4: Manual start/stop (without context manager)."""
    logger.info("\n" + "="*60)
    logger.info("Example 4: Manual Monitoring (No Context Manager)")
    logger.info("="*60)

    from lzero.utils.monitor import ResourceMonitor
    import time

    monitor = ResourceMonitor("Manual Collection")
    monitor.start()

    # Simulate work
    time.sleep(1)
    logger.info("Performing work...")

    monitor.stop()
    monitor.log_metrics(logging.INFO)

    # Can also access metrics directly
    metrics = monitor.get_metrics()
    logger.info(f"Collection took {metrics['duration_sec']:.2f} seconds")


def example_5_store_metrics():
    """Example 5: Store metrics for analysis."""
    logger.info("\n" + "="*60)
    logger.info("Example 5: Store Metrics to File/Database")
    logger.info("="*60)

    import json
    import time
    from lzero.utils.monitor import ResourceMonitor

    metrics_history = []

    for iteration in range(3):
        monitor = ResourceMonitor(f"Iteration {iteration}")
        monitor.start()

        time.sleep(1)  # Simulate collection

        monitor.stop()
        metrics = monitor.get_metrics()
        metrics['iteration'] = iteration
        metrics_history.append(metrics)

    # Save to file
    output_file = "/tmp/collection_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(metrics_history, f, indent=2)

    logger.info(f"Metrics saved to {output_file}")
    logger.info(f"Sample metrics: {metrics_history[0]}")


if __name__ == "__main__":
    # Run all examples
    example_1_basic_monitoring()
    example_2_integration_in_training()
    example_3_multiple_operations()
    example_4_no_context_manager()
    example_5_store_metrics()

    logger.info("\n" + "="*60)
    logger.info("All examples completed!")
    logger.info("="*60)
