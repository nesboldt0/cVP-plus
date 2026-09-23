# cPV+ (Contextual Pitch Value Plus)

Most public pitching metrics live at two extremes:
1. **Physical "Stuff" models (Stuff+)** evaluate velocity and break in a vacuum, ignoring pitch location and treating an 0-2 pitch the same as a 3-0 pitch.
2. **Outcome stats (ERA, FIP, raw Run Value)** take hundreds of innings to stabilize and are heavily polluted by defense, park factors, and sequencing luck.

As a pitcher myself, I hate the phrase "good pitch, better swing" because yeah, that was a good pitch, but no one remembers that, they remember the hit I let up. 

**cPV+** bridges this gap. It is an **Expected Run Value (xRV)** model built on Statcast data that evaluates every pitch based on three interconnected factors: **what the pitch did** (velo/movement), **where it crossed the plate** (proximity to the edges), and **the exact count state**.

Essentially it measure how well a pitcher executes his pitches (execution being a mix of velo, location, and situation)

---

## Stability Check: Odd/Even Split-Half Reliability

To test if cPV+ measures repeatable pitcher skill rather than random outcome noise, pitchers' 2025 seasons were split into odd and even pitches (min. 750 pitches):

| Metric | Pearson r (Half) | Spearman-Brown Reliability |
| :--- | :---: | :---: |
| **Raw Run Value (Delta RE)** | 0.296 | 0.457 |
| **Whiff Rate** | 0.800 | 0.889 |
| **cPV+** | **0.740** | **0.850** |

Raw run prevention is an alright statistic, its a good thing to measure how well a pitcher prevents runs, but its certainly not perfect. Whiff rate is also decent as if the batters miss the ball, it was probably a good pitch, but there's some amount of luck involved with whiff rate. cPV+ scored the same level of reliability as whiff rate but unlike whiff rate, cPV+ is rooted in run prevention, not if the batter missed the ball, thus taking some amount of luck out of the equation.

---

## How cPV+ Differs

| Metric Type | Examples | What It Measures | The Blind Spot | How cPV+ Handles It |
| :--- | :--- | :--- | :--- | :--- |
| **Stuff Only** | Stuff+, PitchingBot (Stuff) | Velocity, movement, release point | Ignores location; treats a 98 mph fastball down the middle the same on 0-2 and 3-0. | Evaluates pitch shape alongside count-dependent plate location. |
| **Overall Process** | FanGraphs Pitching+ | Blended model of physical stuff, count, and target-zone location | Defines location through discrete target boxes and clusters; uses a tight spread. | Measures continuous border geometry (`d_edge`) in a single model, scaled so 130+ stands out more. |
| **Results** | Pitch Values (wFA, wSL), Delta RE | Actual game outcomes (hits, outs) | Very noisy ($SB = 0.457$); swayed by defense, BABIP, and park dimensions. | Evaluates pitch quality at the plate, removing fielding and batted-ball luck. |

### cPV+ vs. Pitching+ (FanGraphs)

Let me start this section by saying that Pitching+ is a fantastic metric and shares some similarities with cPV+ and that's why it gets its own section, I want to distinguish these two so no one asks "isn't this just Pitching+?"

While both models aim to grade total pitch execution rather than just pure shape, they approach the problem differently in how its calculated:

* **Defining "Ideal Location":** Location+ and Pitching+ rely heavily on **consensus intent**. They group the zone and chase regions into discrete target boxes and heatmaps for each count and pitch type, effectively asking: *"Did this pitch hit the target box where this pitch is usually thrown in this count?"* In contrast, cPV+ defines location through **continuous boundary geometry (`d_edge`)**. It measures the exact 2D distance to the perimeter of the strike zone—rewarding painting the black and scaling penalties smoothly as misses drift off the plate. Essentially even if your throw a good slider that painted the black, if it wasn't in the heatmap for Pitching+, Pitching+ will count it unfairly, cPV+ won't because after all, all pitchers are different.
* **Model Architecture:** Pitching+ sits atop a pipeline that trains separate Stuff+ and Location+ components before combining them. cPV+ does this simultaneously. Physical traits, release extension, count state, handedness, and edge distance interact at the same time against marginal run expectancy.
* **Familiar Scaling:** FanGraphs compresses Pitching+ into a very narrow scale. cPV+ standardizes final values to an explicit standard deviation of 15 (identical to OPS+ or ERA+), making skill differences more noticeable. 
---

## 2025 Leaderboards

*Scale: 100 = MLB Average, SD = 15. Higher is better.*

### Top Starting Pitchers (Min. 5 Starts, 1,000 Pitches)

| Pitcher | Pitches | Avg Velo (mph) | Whiff% | cPV+ |
| :--- | :---: | :---: | :---: | :---: |
| **Garrett Crochet** | 3,150 | 92.3 | 15.2% | **131.8** |
| **Shota Imanaga** | 2,107 | 86.2 | 13.1% | **131.5** |
| **Tarik Skubal** | 2,849 | 93.1 | 17.8% | **130.5** |
| **Hunter Greene** | 1,748 | 94.8 | 16.8% | **130.2** |
| **Jacob deGrom** | 2,614 | 93.1 | 15.6% | **129.3** |

### Top Relievers (Min. 400 Pitches, <5 Starts)

| Pitcher | Pitches | Avg Velo (mph) | Whiff% | cPV+ |
| :--- | :---: | :---: | :---: | :---: |
| **Trevor Megill** | 753 | 94.6 | 15.4% | **151.5** |
| **Robert Suarez** | 1,076 | 96.6 | 13.0% | **138.2** |
| **Alex Vesia** | 1,000 | 89.2 | 15.5% | **137.3** |
| **Andrew Kittredge** | 768 | 91.9 | 15.9% | **135.9** |
| **Justin Slaten** | 499 | 91.9 | 15.0% | **134.6** |

One thing that is super interesting to me is the inclusion of Alex Vesia. He doesn't stand out on a typical stat sheet, but according to cPV+ he's the third best reliever at executing pitches.

> **Why split starters and relievers?** Relievers typically throw max-effort over 50–70 innings per year, leading to higher velocity and whiff rates that inflate pitch-level models over lower workloads. Separating roles provides a fair comparison for workhorse starters like Crochet (3,100+ pitches) alongside high-leverage relievers. Essentially if you are a reliever you can go max-effort compared to starters who have to "pace" themselves throughout a game. This means that this metric will inherently favor relievers over starters. That's why I felt it was fair to separate the two.  

---

## Project Structure

```text
├── src/
│   ├── ingest.py               # Pulls and cleans raw Statcast pitch data
│   ├── train_model.py          # Trains the gradient-boosted xRV model
│   ├── export_ratings.py       # Scores pitches and exports leaderboards
│   └── validate_reliability.py # Odd/even split-half reliability engine
├── data/
│   ├── starter_cPV_ratings_2025.csv   # Qualified starters leaderboard
│   ├── reliever_cPV_ratings_2025.csv  # Qualified relievers leaderboard
│   ├── pitcher_cPV_ratings_2025.csv   # Full pitcher leaderboard
│   └── pitcher_arsenal_ratings_2025.csv # Individual pitch-type ratings
└── README.md
