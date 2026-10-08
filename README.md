# Contextual Pitch Value Plus (cPV+)

Welcome! This repository houses the complete modeling pipeline, feature engineering scripts, and evaluation code for **Contextual Pitch Value Plus (cPV+)**—an expected run value ($x\text{RV}$) model designed to quantify MLB pitch execution.

---

### What is cPV+?
Traditional pitching metrics often conflate true execution with defense, ballpark dimensions, and sequencing luck. While physical models like Stuff+ evaluate pitches in a vacuum, cPV+ bridges the gap between raw movement and precision command. 

By modeling continuous Euclidean proximity to the strike zone perimeter ($d_{\text{edge}}$) alongside 3D trajectory physics and count leverage, cPV+ standardizes pitch execution onto an intuitive, 100-centered scale (15 points = 1 SD).

* **Model Architecture:** Histogram-based Gradient Boosting (`HistGradientBoostingRegressor`)
* **Split-Half Reliability:** $r = 0.740$ | Spearman-Brown $SB = 0.850$ ($N = 366$, min. 750 pitches)

---

### Read the Full Paper
For a comprehensive breakdown of the geometric formulation, data filtering, model benchmarks, and front-office applications:

---
*Developed using 2025 MLB Statcast tracking data via Hawk-Eye.*
