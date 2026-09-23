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

## Comparison: How cPV+ Differs & Adds Value

Public pitch metrics generally fall into one of two extremes: purely physical model abstractions (Stuff+) or noisy retrospective outcomes (Delta Run Expectancy / ERA-estimators). **cPV+** bridges this gap as an in-context **Expected Run Value (xRV)** framework.

| Metric Family | Example Metrics | What It Measures | In-Season Noise / Flaws | Where cPV+ Wins |
| :--- | :--- | :--- | :--- | :--- |
| **Pure Kinematic ("Stuff")** | FanGraphs Stuff+, PitchingBot (Stuff) | Movement, velo, and release geometry ignoring location | Blind to execution, count leverage, and spray command | Isolates mechanical shape while embedding count-dependent plate leverage (`d_edge`). |
| **Location Only ("Command")** | Location+, PitchingBot (Command) | Proximity to generic target heatmaps | Fails to weigh pitch physics; 98 mph up differs from 91 mph up | Unifies pitch flight kinetics with dynamic 2D edge distance in a single non-linear surface. |
| **Retrospective Results** | FanGraphs Pitch Values (wFA, wSL), Delta RE | Actual observed outcomes (hits, outs, home runs) | High sample variance ($SB = 0.457$); heavily biased by defense, BABIP, and sequencing | Replaces discrete hit outcomes with continuous expected run values ($SB = 0.850$). |

### Key Theoretical Distinctions

1. **True Run-Scale Anchor vs. Arbitrary Grades:** Unlike Stuff+ models that scale features against empirical whiff or chase rates in isolation, cPV+ optimizes directly to marginal run expectancy ($\Delta RE$). Every point of cPV+ reflects tangible expected run prevention without relying on uncoupled subjective sub-grades.
2. **Context-Sensitive Plate Leverage:** Throwing a sweeper middle-down on 0-2 carries a completely different expectation than on 3-1. cPV+ conditions both pitch physics and plate proximity on the live count state rather than evaluating ball flight in a vacuum.
3. **Defense & Batted-Ball Neutrality:** By evaluating pitch execution at the point of plate transit rather than the point of contact or defensive fielding, cPV+ prevents outfield range, wall dimensions, and batted-ball luck from polluting a pitcher's grade.
## Methodological Overview

- **Catcher-to-Pitcher Coordinate Inversion:** Re-indexes horizontal movement (hb_p) so arm-side and glove-side run are symmetric regardless of throwing hand.
- **Plate Boundary Distance:** Calculates 2D Euclidean distance (d_edge) from the borders of the rule-book strike zone to quantify command leverage outside heart-cut zones.
- **Platoon Interactions:** Models same-hand vs. opposite-hand matchups to account for breaking ball trajectory differentials.

---

## 2025 cPV+ Leaderboards

*Outputs are standardized to an industry-standard 100-index distribution (mean = 100, std = 15). Values above 100 indicate above-average run prevention; 130+ represents elite performance.*

### Top Pitchers (2025 Regular Season)

| Pitcher | Pitches | Avg Velo (mph) | Whiff% | Zone% | Total xRV | cPV+ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trevor Megill** | 753 | 94.6 | 15.4% | 67.3% | +7.30 | **151.5** |
| **Robert Suarez** | 1,076 | 96.6 | 13.0% | 68.0% | +7.65 | **137.6** |
| **Alex Vesia** | 1,000 | 89.2 | 15.5% | 65.7% | +6.93 | **136.6** |
| **Andrew Kittredge** | 768 | 91.9 | 15.9% | 70.2% | +5.11 | **135.1** |
| **Reid Detmers** | 1,083 | 88.8 | 16.2% | 66.9% | +6.70 | **132.6** |
| **Chase Burns** | 764 | 95.4 | 16.8% | 65.3% | +4.61 | **131.7** |
| **Adrián Morejón** | 994 | 94.5 | 11.5% | 68.8% | +5.91 | **131.3** |
| **Randy Rodríguez** | 808 | 92.4 | 15.5% | 66.3% | +4.78 | **131.1** |
| **Kenley Jansen** | 869 | 91.9 | 14.4% | 69.6% | +5.13 | **131.0** |
| **Garrett Crochet** | 3,150 | 92.3 | 15.2% | 67.1% | +18.46 | **130.8** |

*(Note: Total xRV reflects cumulative expected runs prevented above league average).*

---

## Project Structure

- src/ingest.py: Statcast data retrieval and initial cleaning.
- src/train_model.py: Gradient-boosted run expectancy surface training.
- src/export_ratings.py: Scores season pitches and exports 100-indexed player/arsenal leaderboards.
- src/validate_reliability.py: Odd/even pitch sequencing validation engine.
- output/pitcher_cPV_ratings_2025.csv: Overall pitcher ratings and peripheral benchmarks.
- output/pitcher_arsenal_ratings_2025.csv: Pitch-type level breakdowns.
