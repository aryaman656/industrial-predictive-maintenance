# 🏭 Predictive Maintenance & Downtime Analysis

This project analyzes **manufacturing machine logs** using **Pandas** to identify downtime patterns, calculate KPIs, and provide insights for **predictive maintenance**.

---

## 📌 Project Overview
In a manufacturing plant, machines frequently experience downtime due to various issues (e.g., overheating, bearing faults, lubrication problems).  
Traditionally, this data is logged manually in Excel or CSV files, making it difficult to analyze.

This project uses **Python (Pandas, Matplotlib)** to:
- Clean and preprocess raw machine log data  
- Calculate **KPIs** (cycle times, downtime frequency, MTBF, MTTR)  
- Visualize **downtime reasons per machine**  
- Detect anomalies in cycle time/temperature  
- Provide data-driven insights for maintenance planning  

---

## 📂 Dataset
The dataset `machine_logs.csv` contains simulated machine log data with the following columns:

| Column       | Description |
|--------------|-------------|
| machine_id   | Unique ID of machine (M1, M2, M3) |
| timestamp    | Log timestamp |
| cycle_time   | Machine cycle time (seconds) |
| temperature  | Machine temperature (°C) |
| vibration    | Vibration levels (g) |
| downtime     | 1 = downtime occurred, 0 = normal operation |
| reason       | Reason for downtime |

---

# Install required libraries
pip install pandas matplotlib
