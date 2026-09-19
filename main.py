import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("machine_logs.csv", parse_dates=["timestamp"])

# Sort data for proper time-based analysis
df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

# Missing reasons are possible when downtime occurs
df["reason"] = df["reason"].fillna("Unspecified")

# Create output folder
os.makedirs("outputs", exist_ok=True)


print("\n" + "=" * 60)
print("INDUSTRIAL EQUIPMENT PREDICTIVE MAINTENANCE ANALYSIS")
print("=" * 60)

print("\nDataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nDowntime Distribution:")
print(df["downtime"].value_counts())


# ============================================================
# 2. MACHINE-WISE KPI ANALYSIS
# ============================================================

machine_summary = df.groupby("machine_id").agg(
    records=("machine_id", "size"),
    downtime_records=("downtime", "sum"),
    avg_cycle_time=("cycle_time", "mean"),
    avg_temperature=("temperature", "mean"),
    avg_vibration=("vibration", "mean")
)

machine_summary["downtime_rate_%"] = (
    machine_summary["downtime_records"]
    / machine_summary["records"] * 100
)

print("\n" + "-" * 60)
print("MACHINE-WISE PERFORMANCE SUMMARY")
print("-" * 60)

print(machine_summary.round(3))


# ============================================================
# 3. DOWNTIME ANALYSIS
# ============================================================

downtime_df = df[df["downtime"] == 1]

downtime_by_machine = (
    downtime_df.groupby("machine_id")
    .size()
    .sort_values(ascending=False)
)

downtime_by_reason = (
    downtime_df.groupby("reason")
    .size()
    .sort_values(ascending=False)
)

print("\n" + "-" * 60)
print("DOWNTIME BY MACHINE")
print("-" * 60)

print(downtime_by_machine)

print("\n" + "-" * 60)
print("DOWNTIME BY REASON")
print("-" * 60)

print(downtime_by_reason)


# Plot downtime by machine
plt.figure(figsize=(8, 5))

downtime_by_machine.plot(kind="bar")

plt.title("Downtime Records by Machine")
plt.xlabel("Machine")
plt.ylabel("Downtime Records")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("outputs/downtime_by_machine.png", dpi=300)
plt.show()


# Plot downtime reasons
plt.figure(figsize=(9, 5))

downtime_by_reason.plot(kind="bar")

plt.title("Downtime Reasons")
plt.xlabel("Reason")
plt.ylabel("Number of Downtime Records")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig("outputs/downtime_by_reason.png", dpi=300)
plt.show()


# ============================================================
# 4. MACHINE OPERATING PARAMETERS
# ============================================================

parameter_summary = df.groupby("downtime")[
    ["cycle_time", "temperature", "vibration"]
].mean()

parameter_summary.index = ["Normal Operation", "Downtime"]

print("\n" + "-" * 60)
print("OPERATING PARAMETER COMPARISON")
print("-" * 60)

print(parameter_summary.round(4))


# Cycle time by machine
plt.figure(figsize=(9, 5))

df.groupby("machine_id")["cycle_time"].mean().plot(kind="bar")

plt.title("Average Cycle Time by Machine")
plt.xlabel("Machine")
plt.ylabel("Average Cycle Time")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("outputs/average_cycle_time.png", dpi=300)
plt.show()


# Temperature by downtime status
plt.figure(figsize=(8, 5))

df.boxplot(column="temperature", by="downtime")

plt.title("Temperature Distribution: Normal vs Downtime")
plt.suptitle("")
plt.xlabel("Downtime (0 = Normal, 1 = Downtime)")
plt.ylabel("Temperature")

plt.tight_layout()

plt.savefig("outputs/temperature_downtime.png", dpi=300)
plt.show()


# Vibration by downtime status
plt.figure(figsize=(8, 5))

df.boxplot(column="vibration", by="downtime")

plt.title("Vibration Distribution: Normal vs Downtime")
plt.suptitle("")
plt.xlabel("Downtime (0 = Normal, 1 = Downtime)")
plt.ylabel("Vibration")

plt.tight_layout()

plt.savefig("outputs/vibration_downtime.png", dpi=300)
plt.show()


# ============================================================
# 5. CYCLE TIME ANOMALY DETECTION
# ============================================================

cycle_mean = df["cycle_time"].mean()
cycle_std = df["cycle_time"].std()

threshold = cycle_mean + 3 * cycle_std

df["cycle_anomaly"] = df["cycle_time"] > threshold

anomalies = df[df["cycle_anomaly"]]

print("\n" + "-" * 60)
print("CYCLE TIME ANOMALY DETECTION")
print("-" * 60)

print("Cycle Time Mean:", round(cycle_mean, 3))
print("Cycle Time Std:", round(cycle_std, 3))
print("Anomaly Threshold:", round(threshold, 3))
print("Number of Anomalies:", len(anomalies))


plt.figure(figsize=(11, 5))

plt.plot(
    df["timestamp"],
    df["cycle_time"],
    label="Cycle Time"
)

plt.scatter(
    anomalies["timestamp"],
    anomalies["cycle_time"],
    label="Anomaly"
)

plt.axhline(
    threshold,
    linestyle="--",
    label="Anomaly Threshold"
)

plt.title("Cycle Time Anomaly Detection")
plt.xlabel("Time")
plt.ylabel("Cycle Time")
plt.legend()
plt.tight_layout()

plt.savefig("outputs/cycle_time_anomalies.png", dpi=300)
plt.show()


# ============================================================
# 6. TEMPERATURE AND VIBRATION ANOMALIES
# ============================================================

temp_threshold = df["temperature"].mean() + 2 * df["temperature"].std()

vibration_mean = df["vibration"].mean()
vibration_std = df["vibration"].std()

vibration_upper = vibration_mean + 2 * vibration_std
vibration_lower = vibration_mean - 2 * vibration_std

df["temperature_anomaly"] = df["temperature"] > temp_threshold

df["vibration_anomaly"] = (
    (df["vibration"] > vibration_upper) |
    (df["vibration"] < vibration_lower)
)

print("\nTemperature anomaly records:",
      df["temperature_anomaly"].sum())

print("Vibration anomaly records:",
      df["vibration_anomaly"].sum())


# ============================================================
# 7. RELIABILITY INDICATORS
# ============================================================

# The dataset contains one-minute observations.
# Therefore, contiguous downtime records are treated as
# one downtime event.

df["previous_downtime"] = (
    df.groupby("machine_id")["downtime"]
    .shift(1)
    .fillna(0)
)

df["time_difference_min"] = (
    df.groupby("machine_id")["timestamp"]
    .diff()
    .dt.total_seconds() / 60
)

df["new_downtime_event"] = (
    (df["downtime"] == 1) &
    (
        (df["previous_downtime"] == 0) |
        (df["time_difference_min"] != 1)
    )
)

df["event_id"] = (
    df.groupby("machine_id")["new_downtime_event"]
    .cumsum()
)

events = df[df["downtime"] == 1].groupby(
    ["machine_id", "event_id"]
).agg(
    start_time=("timestamp", "min"),
    end_time=("timestamp", "max"),
    duration_records=("downtime", "size")
).reset_index()

# Each record represents approximately one minute
events["estimated_duration_min"] = events["duration_records"]

reliability = events.groupby("machine_id").agg(
    downtime_events=("event_id", "count"),
    estimated_mttr_min=("estimated_duration_min", "mean")
)

# MTBF proxy:
# number of normal operating records / downtime events
normal_records = (
    df[df["downtime"] == 0]
    .groupby("machine_id")
    .size()
)

reliability["mtbf_proxy_records"] = (
    normal_records / reliability["downtime_events"]
)

print("\n" + "-" * 60)
print("RELIABILITY INDICATORS")
print("-" * 60)

print(reliability.round(2))


# ============================================================
# 8. MACHINE LEARNING - FAILURE RISK PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("MACHINE LEARNING FAILURE-RISK MODEL")
print("=" * 60)

# Features available before/around the machine condition
features = [
    "cycle_time",
    "temperature",
    "vibration"
]

# Encode machine identity
machine_encoded = pd.get_dummies(
    df["machine_id"],
    prefix="machine"
)

X = pd.concat(
    [
        df[features],
        machine_encoded
    ],
    axis=1
)

y = df["downtime"]


# Split dataset while preserving downtime ratio
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Random Forest classifier
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# Model performance
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Downtime"]
)

disp.plot()

plt.title("Failure-Risk Model Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n" + "-" * 60)
print("FEATURE IMPORTANCE")
print("-" * 60)

print(feature_importance.round(4))


plt.figure(figsize=(9, 5))

feature_importance.sort_values().plot(kind="barh")

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
    dpi=300
)

plt.show()


# ============================================================
# 10. FAILURE-RISK SCORE
# ============================================================

# Predict risk for the complete dataset
df["failure_risk_probability"] = model.predict_proba(X)[:, 1]

high_risk_records = df.sort_values(
    "failure_risk_probability",
    ascending=False
).head(10)

print("\n" + "-" * 60)
print("TOP 10 HIGH-RISK MACHINE RECORDS")
print("-" * 60)

print(
    high_risk_records[
        [
            "machine_id",
            "timestamp",
            "cycle_time",
            "temperature",
            "vibration",
            "failure_risk_probability"
        ]
    ].round(4)
)


# ============================================================
# 11. MAINTENANCE RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 60)
print("MAINTENANCE RECOMMENDATIONS")
print("=" * 60)

print("""
1. Monitor machines with repeated downtime events.

2. High vibration observations can be investigated for:
   - Bearing condition
   - Shaft alignment
   - Mechanical looseness

3. High temperature observations can be investigated for:
   - Cooling problems
   - Lubrication problems
   - Excessive mechanical loading

4. Abnormally high cycle time can indicate:
   - Increased mechanical resistance
   - Tooling/load issues
   - Machine performance degradation

5. Use the ML failure-risk probability as a screening
   indicator for prioritizing inspection.

NOTE:
These recommendations are engineering interpretations
of the simulated dataset and should not be treated as
proof of causation.
""")


# ============================================================
# 12. SAVE ANALYZED DATA
# ============================================================

df.to_csv(
    "outputs/analyzed_machine_logs.csv",
    index=False
)

machine_summary.to_csv(
    "outputs/machine_summary.csv"
)

reliability.to_csv(
    "outputs/reliability_summary.csv"
)

print("\nAnalysis complete.")
print("All generated files are available in the 'outputs' folder.")