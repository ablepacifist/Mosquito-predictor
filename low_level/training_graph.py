#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Read the CSV file. Adjust the file path if needed.
    try:
        df = pd.read_csv("data/training_log.csv")
    except FileNotFoundError:
        print("Error: CSV log file not found. Make sure 'data/training_log.csv' exists.")
        return

    # Optionally, convert "NaN" string to actual NaN values.
    df["TrainingLoss"] = pd.to_numeric(df["TrainingLoss"], errors="coerce")

    # Create a plot with two y-axes.
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot Training Loss on the primary y-axis.
    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color=color)
    ax1.plot(df['Epoch'], df['TrainingLoss'], color=color, marker="o", label="Training Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    # Create a secondary y-axis for Validation Accuracy.
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Validation Accuracy", color=color)
    ax2.plot(df['Epoch'], df['ValidationAccuracy'], color=color, marker="x", label="Validation Accuracy")
    ax2.tick_params(axis="y", labelcolor=color)

    # Add title and grid.
    plt.title("Validation Accuracy Over Epochs")
    ax1.grid(True)

    # Combine legends.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")

    # Save and show the figure.
    plt.savefig("data/training_graph.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
