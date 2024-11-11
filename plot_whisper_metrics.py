import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "results_whisper_gita.csv"  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Separate the means and standard deviations for plotting
mean_data = data[data["metrics"].str.contains("Mean")]
std_data = data[data["metrics"].str.contains("Std")]

# Set up the metrics for plotting
metrics = ["WER", "CER"]


# Function to plot each metric with mean and standard deviation error bars
def plot_metric(metric):
    mean_values = mean_data[metric].values
    std_values = std_data[metric].values
    labels = mean_data["metrics"].str.replace(" Mean", "").values

    plt.figure(figsize=(10, 6))
    plt.bar(labels, mean_values, yerr=std_values, capsize=5)
    plt.title(f"{metric} in GITA Whisper Dataset")
    plt.ylabel(metric)
    plt.ylim(0, 1)
    plt.xlabel("Case")
    plt.xticks(rotation=45)
    plt.savefig(f"{metric}_whisper_gita.png")
    plt.show()


# Plot each metric
for metric in metrics:
    plot_metric(metric)
