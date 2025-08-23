import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("machine_logs.csv", parse_dates=["timestamp"])
# Mean Cycle Time per Machine
cycle_stats = df.groupby("machine_id")["cycle_time"].mean()

# Downtime by reason
downtime_summary = df[df["downtime"] == 1].groupby("reason").size()
# Output results
threshold = df["cycle_time"].mean() + 3*df["cycle_time"].std()
anomalies = df[df["cycle_time"] > threshold]


# Cycle Time Plot
df.groupby("machine_id")["cycle_time"].plot(legend=True)
plt.title("Cycle Time per Machine")
plt.show()

# Downtime Count Plot
downtime_summary.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5),
    colormap="tab20"  # nice color scheme
)

plt.title("Downtime Reasons per Machine")
plt.xlabel("Machine ID")
plt.ylabel("Downtime Count")
plt.xticks(rotation=0)
plt.legend(title="Reason")
plt.tight_layout()
plt.show()


# Anomalies Plot
plt.figure(figsize=(10, 6))
plt.plot(anomalies["timestamp"], anomalies["cycle_time"], "ro", label="Anomaly")
plt.title("Anomalies in Cycle Time")
plt.xlabel("Time")
plt.ylabel("Cycle Time")
plt.legend()
plt.show()
