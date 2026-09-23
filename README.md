# cPV+ (Contextual Pitch Value Plus)

**cPV+** is an expected run value (xRV) pitch evaluation model designed to isolate pitcher-controlled performance from fielding, park, and sequencing noise. Built on regular-season Statcast tracking data, the model uses a gradient-boosted regression architecture (HistGradientBoostingRegressor) trained on pitch kinematics, release characteristics, and spatial plate locations across dynamic count states.

Outputs are standardized to an industry-standard 100-index distribution (mean = 100, std = 15), where higher values represent superior run prevention.

---

## Metric Stability (Split-Half Reliability)

To evaluate whether cPV+ captures true underlying talent rather than outcome noise, an odd/even split-half reliability test was conducted across qualified MLB pitchers (min. 750 pitches):

| Metric | Pearson r (Half) | Spearman-Brown (SB) |
| :--- | :---: | :---: |
| **Raw Outcomes (Delta RE)** | 0.296 | 0.457 |
| **Whiff Rate** | 0.800 | 0.889 |
| **cPV+ (xRV Index)** | **0.740** | **0.850** |

*cPV+ nearly doubles the single-season stability of raw run prevention (SB = 0.850 vs. 0.457), approaching the stability of whiff rate while preserving an expected-run scale.*

---

## Methodological Overview

- **Catcher-to-Pitcher Coordinate Inversion:** Re-indexes horizontal movement (hb_p) so arm-side and glove-side run are symmetric regardless of throwing hand.
- **Plate Boundary Distance:** Calculates 2D Euclidean distance (d_edge) from the borders of the rule-book strike zone to quantify command leverage outside heart-cut zones.
- **Platoon Interactions:** Models same-hand vs. opposite-hand matchups to account for breaking ball trajectory differentials.

---

## Project Structure

- src/ingest.py: Statcast data retrieval and initial cleaning.
- src/train_model.py: Gradient-boosted run expectancy surface training.
- src/export_ratings.py: Scores season pitches and exports 100-indexed player/arsenal leaderboards.
- src/validate_reliability.py: Odd/even pitch sequencing validation engine.
- output/pitcher_cPV_ratings_2025.csv: Overall pitcher ratings and peripheral benchmarks.
- output/pitcher_arsenal_ratings_2025.csv: Pitch-type level breakdowns.
