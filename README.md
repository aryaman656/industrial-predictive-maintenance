# Industrial Equipment Predictive Maintenance & Reliability Analysis

A Python-based predictive maintenance project for analyzing industrial machine operating data, identifying abnormal behavior, studying downtime patterns, and building a machine-learning model for equipment failure-risk screening.

The project combines **mechanical engineering concepts, reliability analysis, data analytics, and machine learning** to support condition-based maintenance and maintenance planning.

---

## Project Overview

Unexpected equipment failures can cause production losses, increased maintenance costs, and downtime.

This project analyzes historical machine-condition data containing:

- Cycle time
- Temperature
- Vibration
- Downtime records
- Downtime reasons
- Machine identification
- Timestamp information

The analysis is used to identify equipment performance patterns, detect abnormal operating conditions, calculate reliability indicators, and develop a machine-learning based failure-risk screening model.

> **Dataset note:** The included dataset is a simulated manufacturing dataset created for project development and analysis. It is not SAIL plant data.

---

## Objectives

1. Analyze machine-wise production and operating performance.
2. Identify major downtime patterns and causes.
3. Compare operating parameters during normal operation and downtime.
4. Detect abnormal cycle-time, temperature, and vibration observations.
5. Estimate downtime-event based reliability indicators.
6. Develop a machine-learning model for downtime/failure-risk classification.
7. Identify important operating parameters using feature importance.
8. Generate maintenance-oriented engineering recommendations.

---

## Project Workflow

```text
Machine Logs
     ↓
Data Cleaning & Preparation
     ↓
Machine-wise KPI Analysis
     ↓
Downtime & Failure Analysis
     ↓
Temperature & Vibration Analysis
     ↓
Anomaly Detection
     ↓
Reliability Indicators
     ↓
Random Forest Failure-Risk Model
     ↓
Feature Importance
     ↓
Maintenance Recommendations