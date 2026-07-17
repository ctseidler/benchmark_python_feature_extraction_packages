"""
Script to run benchmarks for feature extraction methods on time series data.
The script records the computational time and memory usage of the feature extraction methods.

"""

import copy
import gc
import multiprocessing
import os
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend for plotting

import matplotlib.pyplot as plt
import pandas as pd
import psutil
import toml

from src.base import (
    get_dataloader,
    get_feature_extractor,
    get_sampling_frequency,
    match_configuration_to_feature_extraction_methods,
)

def main() -> None:
    """Run the benchmark for feature extraction methods."""
    # Step 1: Load the configuration file
    configuration = toml.load("feature_extraction_config.toml")

    # Create export directory if it does not exist
    os.makedirs(configuration["benchmark_settings"]["export_dir"], exist_ok=True)

    # Export the configuration to the export directory
    with open(
        Path(configuration["benchmark_settings"]["export_dir"]) / "config.toml",
        "w",
        encoding="utf-8",
    ) as f:
        toml.dump(configuration, f)

    # Step 2: Match the configuration to the feature extraction methods
    configuration = match_configuration_to_feature_extraction_methods(configuration)

    # Step 3: Load the selected dataset
    dataloader = get_dataloader(configuration["benchmark_settings"]["dataset"])
    features, targets = dataloader.load_dataset(Path(configuration["benchmark_settings"]["data_dir"]))

    # Step 4: Prepare feature extraction
    # Get the sampling frequency of the dataset
    sampling_frequency = get_sampling_frequency(configuration["benchmark_settings"]["dataset"])

    # Create export directory
    os.makedirs(configuration["benchmark_settings"]["export_dir"], exist_ok=True)

    # Export targets
    targets.to_csv(
        Path(configuration["benchmark_settings"]["export_dir"]) / "targets.csv",
        sep=";",
        decimal=",",
        index=False,
    )

    time.sleep(5)  # Wait for the system to stabilize

    # Step 5: Perform benchmarking for each package
    for trial in range(configuration["benchmark_settings"]["num_trials"]):
        for package in configuration["benchmark_settings"]["packages"]:
            print("\nExtracting features with package:", package)

            # Copy the features to avoid modifying the original data
            features_copy = copy.deepcopy(features)
            sampling_frequency = copy.deepcopy(sampling_frequency)

            # Get the correct feature extractor
            extractor = get_feature_extractor(package, features_copy, sampling_frequency)

            # Extract features while profiling resource usage
            extract_features(extractor, configuration, package, trial)
            time.sleep(5)  # Wait for the system to stabilize

def extract_features(extractor: object, configuration: dict, package: str, trial: int) -> None:
    """Extract features while profiling resource usage and CPU time measurement."""
    process = psutil.Process(os.getpid())
    memory_usages = []
    cpu_usages = []
    timestamps = []
    stop_event = threading.Event()
    sampling_interval = configuration["benchmark_settings"]["sampling_interval"]

    monitor_thread = threading.Thread(
        target=monitor_memory,
        args=(process, sampling_interval, stop_event, memory_usages, cpu_usages, timestamps),
        daemon=True,
    )
    monitor_thread.start()

    start_time = time.time()

    try:
        # Execute the target function
        extractor.extract_features(
            export_dir=Path(configuration["benchmark_settings"]["export_dir"]),
            **configuration[package],
        )
    finally:
        # Ensure the monitoring thread is stopped even if an exception occurs
        stop_event.set()
        monitor_thread.join()

    end_time = time.time()
    excecution_time = end_time - start_time
    print(f"Execution time (wall): {excecution_time:.2f} seconds")

    # Optional: Force garbage collection before final measurement
    gc.collect()

    # Calculate resource usage statistics
    peak_memory = max(memory_usages) / (1024**2)
    average_memory = (sum(memory_usages) / len(memory_usages)) / (1024**2)
    max_cpu_usage = max(cpu_usages)
    average_cpu_usage = sum(cpu_usages) / len(cpu_usages)

    print(f"Max CPU usage: {max_cpu_usage} %")
    print(f"Average CPU usage: {average_cpu_usage:.2f} %")
    print(f"Peak memory usage during execution: {peak_memory:.2f} MB")
    print(f"Average memory usage during execution: {average_memory:.2f} MB")

    # Export execution time, CPU load, and memory usage to a CSV file --> create a header if the file does not exist
    filepath = Path(configuration["benchmark_settings"]["export_dir"]) / "benchmark_results.csv"

    # Create a DataFrame for benchmark results
    benchmark_data = {
        "Package": [package],
        "Trial": [trial],
        "Execution Time (s)": [excecution_time],
        "Max CPU Load (%)": [max_cpu_usage],
        "Average CPU Load (%)": [average_cpu_usage],
        "Peak Memory (MB)": [peak_memory],
        "Average Memory (MB)": [average_memory],
    }

    benchmark_df = pd.DataFrame(benchmark_data)

    # Append to the CSV file with semicolon as separator
    benchmark_df.to_csv(filepath, mode="a", header=not filepath.exists(), sep=";", index=False, decimal=",")

    # Export the memory usage data to CSV files
    memory_usage_filepath = (
        Path(configuration["benchmark_settings"]["export_dir"]) / f"memory_usage_{package}_trial_{trial}.csv"
    )

    memory_usage_df = pd.DataFrame(
        {
            "Time (s)": timestamps,
            "Memory Usage (bytes)": memory_usages,
        }
    )

    memory_usage_df.to_csv(memory_usage_filepath, sep=";", index=False, decimal=",")

    # Export the CPU usage data to CSV files
    cpu_usage_filepath = (
        Path(configuration["benchmark_settings"]["export_dir"]) / f"cpu_usage_{package}_trial_{trial}.csv"
    )

    cpu_usage_df = pd.DataFrame(
        {
            "Time (s)": timestamps,
            "CPU Usage (%)": cpu_usages,
        }
    )

    cpu_usage_df.to_csv(cpu_usage_filepath, sep=";", index=False, decimal=",")

    # Plot memory usage over time
    plot_memory_usage(timestamps, memory_usages, configuration, package, trial)
    plot_cpu_usage(timestamps, cpu_usages, configuration, package, trial)

# Multi processing solution
def monitor_memory(process, interval, stop_event, memory_usages, cpu_usages, timestamps):
    """Continuously monitor total memory usage of the given process and its children."""
    start_time = time.time()
    while not stop_event.is_set():
        current_time = time.time() - start_time  # Time elapsed since monitoring started
        try:
            # Get current memory usage of the parent process
            memory_usage = process.memory_info().rss
            # Get child processes (recursively)
            child_processes = process.children(recursive=True)
            # Add memory usage of child processes
            for child in child_processes:
                try:
                    memory_usage += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass  # Child process terminated or access denied
            memory_usages.append(memory_usage)
            timestamps.append(current_time)
            cpu_usage = psutil.cpu_percent(interval=interval)
            cpu_usages.append(cpu_usage)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass  # Parent process terminated or access denied
        time.sleep(interval)

def plot_memory_usage(timestamps, memory_usages, configuration, package: str, trial: str) -> None:
    """Plot the memory usage over time with enhancements."""
    # Convert memory usage from bytes to MB
    memory_usages_mb = [usage / (1024**2) for usage in memory_usages]

    plt.figure(figsize=(12, 7))
    plt.plot(timestamps, memory_usages_mb, color="blue", linestyle="-", linewidth=2, label="Memory Usage")

    # Highlight the peak memory usage point
    peak_usage = max(memory_usages_mb)
    peak_time = timestamps[memory_usages_mb.index(peak_usage)]
    plt.scatter(peak_time, peak_usage, color="red", label=f"Peak Usage: {peak_usage:.2f} MB")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Optionally save the plot
    filepath = Path(configuration["benchmark_settings"]["export_dir"]) / f"memory_usage_{package}_trial_{trial}.png"
    plt.savefig(filepath)
    plt.close()

def plot_cpu_usage(timestamps, cpu_usages, configuration, package: str, trial: str) -> None:
    """Plot the CPU usage over time with enhancements."""
    plt.figure(figsize=(12, 7))
    plt.plot(timestamps, cpu_usages, color="blue", linestyle="-", linewidth=2, label="CPU Usage")

    # Highlight the peak memory usage point
    peak_usage = max(cpu_usages)
    peak_time = timestamps[cpu_usages.index(peak_usage)]
    plt.scatter(peak_time, peak_usage, color="red", label=f"Peak Usage: {peak_usage:.2f} %")

    plt.xlabel("Time (seconds)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Optionally save the plot
    filepath = Path(configuration["benchmark_settings"]["export_dir"]) / f"cpu_usage_{package}_trial_{trial}.png"
    plt.savefig(filepath)
    plt.close()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
