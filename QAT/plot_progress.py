# plot_progress.py

import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_progress(csv_path: str = "data.csv") -> None:
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    if df.empty:
        print("Error: CSV file is empty.")
        return

    required_columns = {
        "step",
        "loss_nats_per_token",
        "loss_bits_per_token",
        "unixtime",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(
            "Error: CSV file is missing required columns: "
            f"{sorted(missing_columns)}"
        )
        return

    # Convert columns to numeric.
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=list(required_columns))

    if df.empty:
        print("Error: CSV file contains no valid numeric rows.")
        return

    # Calculate time since beginning.
    start_time = df["unixtime"].iloc[0]
    df["time_elapsed"] = df["unixtime"] - start_time

    # Create the plot.
    plt.figure(figsize=(10, 6))

    plt.plot(
        df["time_elapsed"],
        df["loss_bits_per_token"],
        label="Loss",
        color="blue",
        linewidth=2,
    )

    plt.title("QAT Algorithm Progress: Loss vs Time", fontsize=14)
    plt.xlabel("Time Elapsed (seconds)", fontsize=12)
    plt.ylabel("Loss (bits/token)", fontsize=12)

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    print("Displaying plot...")
    plt.show()


if __name__ == "__main__":
    plot_progress()
