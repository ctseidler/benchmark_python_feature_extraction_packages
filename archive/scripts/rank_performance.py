"""
Script to rank the performance of the TPOT results and create a critical difference diagram.

Implementation from: https://dev.to/milenamonteiro/comparing-machine-learning-algorithms-using-friedman-test-and-critical-difference-diagrams-in-python-10a9

Created on: 24-07-2025
by Christian Seidler <christian.seidler@ipa.fraunhofer.de>

Last modified on: 24-07-2025
by Christian Seidler <christian.seidler@ipa.fraunhofer.de>
"""

import matplotlib.pyplot as plt
import pandas as pd
from aeon.visualisation import plot_critical_difference
from scipy.stats import friedmanchisquare

STD_FACTOR = 0  # Whether or not to consider the standard deviation in the ranking

data = {
    "Datasets": [
        "Bosch CNC",
        "Condition Monitoring of Hydraulic Systems",
        "CNC Mill Tool Wear",
        "Turning Dataset",
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
        "Kats",
    ],
    "Performance (Accuracy)": [
        [  # Bosch CNC
            0.0,  # Tsfresh__comprehensive
            0.99,  # Tsfresh__efficient
            0.764705882,  # TSFEL__none
            0.796470588,  # TSFEL__spectral
            0.715882353,  # TSFEL__statistical
            0.671176471,  # TSFEL__temporal
            0.0,  # Tsfeatures
            0.991764706,  # Seglearn__all
            0.941764706,  # Seglearn__default
            0.983529412,  # Pycatch22
            0.0,  # Kats
        ],
        [  # Condition Monitoring of Hydraulic Systems
            0.989115646,  # Tsfresh__comprehensive
            0.990022676,  # Tsfresh__efficient
            0.998639456,  # TSFEL__none
            0.998639456,  # TSFEL__spectral
            0.999546485,  # TSFEL__statistical
            0.999092971,  # TSFEL__temporal
            0.996825397,  # Tsfeatures
            0.999092971,  # Seglearn__all
            0.998185941,  # Seglearn__default
            0.999546485,  # Pycatch22
            0.998639456,  # Kats
        ],
        [  # CNC Mill Tool Wear
            0.7,  # Tsfresh__comprehensive
            0.55,  # Tsfresh__efficient
            0.45,  # TSFEL__none
            0.45,  # TSFEL__spectral
            0.4,  # TSFEL__statistical
            0.65,  # TSFEL__temporal
            0.75,  # Tsfeatures
            0.55,  # Seglearn__all
            0.7,  # Seglearn__default
            0.5,  # Pycatch22
            0.55,  # Kats
        ],
        [  # Turning Dataset
            0.945679012,  # Tsfresh__comprehensive
            0.949382716,  # Tsfresh__efficient
            0.871604938,  # TSFEL__none
            0.87654321,  # TSFEL__spectral
            0.680246914,  # TSFEL__statistical
            0.745679012,  # TSFEL__temporal
            0.89382716,  # Tsfeatures
            0.887654321,  # Seglearn__all
            0.827160494,  # Seglearn__default
            0.87037037,  # Pycatch22
            0.922222222,  # Kats
        ],
    ],
    "Standard Deviation (Accuracy)": [
        [  # Bosch CNC
            0.0,  # Tsfresh__comprehensive
            0.00802246,  # Tsfresh__efficient
            0.104451111,  # TSFEL__none
            0.094069841,  # TSFEL__spectral
            0.121490171,  # TSFEL__statistical
            0.166517349,  # TSFEL__temporal
            0.0,  # Tsfeatures
            0.003429972,  # Seglearn__all
            0.030869745,  # Seglearn__default
            0.013100622,  # Pycatch22
            0.0,  # Kats
        ],
        [  # Condition Monitoring of Hydraulic Systems
            0.004842212,  # Tsfresh__comprehensive
            0.00466922,  # Tsfresh__efficient
            0.00111088,  # TSFEL__none
            0.001814059,  # TSFEL__spectral
            0.000907029,  # TSFEL__statistical
            0.00111088,  # TSFEL__temporal
            0.00111088,  # Tsfeatures
            0.00111088,  # Seglearn__all
            0.001696897,  # Seglearn__default
            0.000907029,  # Pycatch22
            0.00111088,  # Kats
        ],
        [  # CNC Mill Tool Wear
            0.187082869,  # Tsfresh__comprehensive
            0.187082869,  # Tsfresh__efficient
            0.1,  # TSFEL__none
            0.187082869,  # TSFEL__spectral
            0.122474487,  # TSFEL__statistical
            0.2,  # TSFEL__temporal
            0.158113883,  # Tsfeatures
            0.1,  # Seglearn__all
            0.1,  # Seglearn__default
            0.316227766,  # Pycatch22
            0.291547595,  # Kats
        ],
        [  # Turning Dataset
            0.016285069,  # Tsfresh__comprehensive
            0.019675775,  # Tsfresh__efficient
            0.007198706,  # TSFEL__none
            0.008729713,  # TSFEL__spectral
            0.039312798,  # TSFEL__statistical
            0.013181578,  # TSFEL__temporal
            0.013181578,  # Tsfeatures
            0.025719342,  # Seglearn__all
            0.023747388,  # Seglearn__default
            0.01407624,  # Pycatch22
            0.008373247,  # Kats
        ],
    ],
}

# Convert the data into a DataFrame
datasets = data["Datasets"]
algorithms = data["Packages"]
performance_data = data["Performance (Accuracy)"]
std_data = data["Standard Deviation (Accuracy)"]

# Create a list of dictionaries for each dataset
rows = []
for dataset, performance, std in zip(datasets, performance_data, std_data):
    row = {"Dataset": dataset}
    # See: https://www.reddit.com/r/askmath/comments/1d0yx93/should_i_include_the_standard_deviation_somehow/
    row.update({alg: perf - (STD_FACTOR * s) for alg, perf, s in zip(algorithms, performance, std)})
    rows.append(row)

# Create the DataFrame
df = pd.DataFrame(rows)

# Calculate the ranking of each algorithm for each dataset
rankings_matrix = df[algorithms].rank(axis=1, method="min", ascending=False)

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
print("Accuracy Table (%) with Ranking:")
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
plt.savefig("table_with_rankings.png", format="png", bbox_inches="tight", dpi=300)
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
print("Accuracies:", scores)

# Set the figure size before plotting
plt.figure(figsize=(16, 12))  # Adjust the figure size as needed

# Generate the critical difference diagram
plot_critical_difference(
    scores,
    classifiers,
    lower_better=False,
    test="wilcoxon",  # wilcoxon or nemenyi
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
plt.savefig("critical_difference_diagram.png", format="png", bbox_inches="tight", dpi=300)
plt.show()
