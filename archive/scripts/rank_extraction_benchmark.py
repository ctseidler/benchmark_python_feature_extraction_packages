"""
Script to rank the results of the feature extraction benchmark and create a critical difference diagram.

Created on: 04-08-2025
by Christian Seidler <christian.seidler@ipa.fraunhofer.de>

Last modified on: 04-08-2025
by Christian Seidler <christian.seidler@ipa.fraunhofer.de>
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from aeon.visualisation import plot_critical_difference
from scipy.stats import friedmanchisquare

EXPORT_DIR = Path(r"D:\csr\Projects\04_Benchmarking\results\12_Rankings")
METRIC = "Execution Time in s"

data = {
    "Datasets": [
        "CNC Mill Tool Wear",
        "Condition Monitoring of Hydraulic Systems",
        "Turning Dataset",
        "Bosch CNC",
    ],
    "Packages": [
        "Tsfresh__comprehensive",
        "Tsfresh__efficient",
        "TSFEL__none",
        "TSFEL__spectral",
        "TSFEL__statistical",
        "TSFEL__temporal",
        "Tsfeatures",
        "Seglearn__all",
        "Seglearn__default",
        "Pycatch22",
    ],
    # Single Processing
    "Execution Time": [
        [  # CNC Mill Tool Wear
            804.8,  # Tsfresh__comprehensive
            69.6,  # Tsfresh__efficient
            7.8,  # TSFEL__none
            3.6,  # TSFEL__spectral
            3.1,  # TSFEL__statistical
            6.8,  # TSFEL__temporal
            6.0,  # Pycatch22
            2.1,  # Seglearn__all
            3.3,  # Seglearn__default
            763.6,  # Tsfeatures
        ],
        [  # Condition Monitoring of Hydraulic Systems
            1e12,  # Tsfresh__comprehensive
            6195.6,  # Tsfresh__efficient
            377.8,  # TSFEL__none
            103.0,  # TSFEL__spectral
            81.7,  # TSFEL__statistical
            302.8,  # TSFEL__temporal
            550.4,  # Pycatch22
            70.7,  # Seglearn__all
            143.8,  # Seglearn__default
            12186.2,  # Tsfeatures
        ],
        [  # Turning Dataset
            1e12,  # Tsfresh__comprehensive
            550.1,  # Tsfresh__efficient
            30.9,  # TSFEL__none
            5.2,  # TSFEL__spectral
            4.2,  # TSFEL__statistical
            25.9,  # TSFEL__temporal
            61.6,  # Pycatch22
            2.1,  # Seglearn__all
            4.2,  # Seglearn__default
            43582.0,  # Tsfeatures
        ],
        [  # Bosch CNC
            1e12,  # Tsfresh__comprehensive
            1e12,  # Tsfresh__efficient
            1309.4,  # TSFEL__none
            99.9,  # TSFEL__spectral
            97.2,  # TSFEL__statistical
            1026.8,  # TSFEL__temporal
            16732.3,  # Pycatch22
            79.9,  # Seglearn__all
            169.2,  # Seglearn__default
            1e12,  # Tsfeatures
        ],
    ],
}

# Convert the data into a DataFrame
datasets = data["Datasets"]
algorithms = data["Packages"]
performance_data = data["Execution Time"]

# Create a list of dictionaries for each dataset
rows = []
for dataset, performance in zip(datasets, performance_data):
    row = {"Dataset": dataset}
    row.update({alg: perf for alg, perf in zip(algorithms, performance)})
    rows.append(row)

# Create the DataFrame
df = pd.DataFrame(rows)

# Calculate the ranking of each algorithm for each dataset
rankings_matrix = df[algorithms].rank(axis=1, method="min", ascending=True)

# Format the results
formatted_results = df[algorithms].copy()
for col in formatted_results.columns:
    formatted_results[col] = (
        formatted_results[col].round(3).astype(str) + " (" + rankings_matrix[col].astype(int).astype(str) + ")"
    )

# Add a row for the sum of ranks and average of ranks
sum_ranks = rankings_matrix.sum().round(3).rename("Sum Ranks")
average_ranks = rankings_matrix.mean().round(3).rename("Average Ranks")

# Add the rows to the formatted DataFrame using concat
formatted_results = pd.concat([formatted_results, sum_ranks.to_frame().T, average_ranks.to_frame().T])

# Add the 'Dataset' column to the formatted DataFrame
formatted_results.insert(0, "Dataset", df["Dataset"].tolist() + ["Sum Ranks", "Average Ranks"])

# Display the table
print(f"{METRIC} Table with Ranking:")
print(formatted_results)

# Save the formatted table as an image
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis("tight")
ax.axis("off")
table = ax.table(cellText=formatted_results.values, colLabels=formatted_results.columns, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(2.5, 2.5)
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=1, wspace=0.2, hspace=0.2)
plt.savefig(EXPORT_DIR / "table_with_rankings.png", format="png", bbox_inches="tight", dpi=300)
plt.show()
plt.close()

print("Table saved as 'table_with_rankings.png'")

# Perform the Friedman Test
friedman_stat, p_value = friedmanchisquare(*rankings_matrix.T.values)
print(f"Friedman test statistic: {friedman_stat}, p-value = {p_value}")

# Convert the accuracy matrix into a NumPy array for the critical difference diagram
scores = df[algorithms].values
classifiers = df[algorithms].columns.tolist()

print("Algorithms:", classifiers)
print("Scores:", scores)

# Set the figure size before plotting
plt.figure(figsize=(16, 12))  # Adjust the figure size as needed

# Generate the critical difference diagram
plot_critical_difference(
    scores,
    classifiers,
    lower_better=True,
    test="nemenyi",  # wilcoxon or nemenyi
    correction="holm",  # holm or bonferroni or none
    alpha=0.05,
)

# Get the current axes
ax = plt.gca()

# Adjust font size and rotation of x-axis labels
for label in ax.get_xticklabels():
    label.set_fontsize(14)
    label.set_rotation(45)
    label.set_horizontalalignment("right")

# Increase padding between labels and axis
ax.tick_params(axis="x", which="major", pad=20)

# Adjust margins to provide more space for labels
plt.subplots_adjust(bottom=0.35)

# Optionally adjust y-axis label font size
ax.tick_params(axis="y", labelsize=12)

# Save and display the plot
plt.savefig(EXPORT_DIR / "critical_difference_diagram.png", format="png", bbox_inches="tight", dpi=300)
plt.show()
