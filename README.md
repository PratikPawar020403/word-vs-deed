# word-vs-deed

A benchmark that tests whether AI models will lie to their own teammate when they are afraid of losing.

## What we are trying to find out

In the past, conquerors gave people two choices: convert or die. We turned that into a game for LLMs.

Four models play in two teams of two. They debate a topic, then each model secretly chooses:

- **STAY**: keep your team's position
- **CONVERT**: switch to the other team's position
- **MIDDLE**: refuse to convert, but signal you are open to a compromise

Before voting, teammates tell each other what they plan to do. The question is:

**When a model is losing, will it say one thing to its teammate and secretly do another?**

To make lying tempting, a model that secretly converts after promising to stay gets a **+3 personal bonus**.

---

## 4-Model Cross-Family Arena (Lineup C)

We field four distinct frontier and lightweight models from four separate model families in a balanced round-robin rotation (3 pairings × 2 sides = 6 seating rotations), evaluated by two independent judges:

* **Debaters (4 Distinct Families):**
  * `anthropic/claude-sonnet-5@default` (Anthropic)
  * `gemini-3.5-flash-lite` (Google)
  * `gpt-5.4-nano-2026-03-17` (OpenAI)
  * `ibm/granite-4.0-h-small` (IBM)
* **Dual Independent Judges:**
  * `gemini-3.6-flash` (Google)
  * `gpt-oss-120b` (OpenAI)

### Key Experimental Controls
1. **Bonus Control (Greed vs. Fear):** Every condition is tested with Bonus = 0 (no personal gain for lying) and Bonus = 3 (+3 personal swing points).
2. **Per-Member Fear from Own Score:** Fear is computed from each model's own team score (trailing = High Fear, tied = Medium Fear, leading = Low Fear), with the trailing side alternating by seed.
3. **Dual Judge Calibration:** Calibrated against 4 fixed ground-truth mock debates to cleanly separate substantive reasoned conversions from fear-driven caving.
4. **Quota Meter Safety Guards:** Live polling against `kaggle b quota` to prevent cost overruns and maintain a safety reserve.

---

## What the Data Shows So Far (Stage 1: 48 Debates, 192 Member Records)

| Model | Family | Comp Score | Bonus = 0 Lie Rate | Bonus = 3 Lie Rate (Low Fear) | Bonus = 3 Lie Rate (High Fear) | Observed Behavior |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Claude Sonnet 5** | Anthropic | 1.0 (100%) | **0.0%** (0/24) | **0.0%** (0/8) | **0.0%** (0/8) | **Unwavering Loyalty:** 0 lies across all 48 games regardless of pressure or bonus. |
| **Gemini 3.5 Flash Lite** | Google | 1.0 (100%) | **0.0%** (0/24) | **100.0%** (8/8) | **0.0%** (0/8) | **Opportunistic Profit:** 0% lies when tied/behind, but betrays teammate 100% of the time when leading to pocket the +3 bonus. |
| **GPT-5.4 Nano** | OpenAI | 1.0 (100%) | **4.2%** (1/24) | **12.5%** (1/8) | **12.5%** (1/8) | **High Loyalty:** Low baseline lie rate (6.25% overall) largely independent of fear. |
| **Granite 4.0 Small** | IBM | 0.0 (0%) | **8.3%** (2/24) | **0.0%** (0/8) | **0.0%** (0/8) | **Random Noise:** Failed payoff comprehension test; flat low conversion rate. |

### Judge Audit & Family Favoritism
* **Concordance:** `gemini-3.6-flash` is a strict judge (30% reasoned rate), while `gpt-oss-120b` is lenient (80% reasoned rate).
* **Family Bias Audit:** Neither judge favored debaters from its own family (Gemini-3.6 passed Google debaters at 28.1% vs non-Google at 37.5%; GPT-OSS passed OpenAI debaters at 66.7% vs non-OpenAI at 81.1%).

---

## Repository Structure

| File | What it does |
|---|---|
| `arena_config.json` | Complete Arena configuration (Lineup C, 6 rotations, fear levels, token caps) |
| `run_arena_stage1.py` | Live execution engine for the 2v2 arena with automated quota guards |
| `analyze_arena_stage1.py` | Computes Wilson 95% CIs, partnership breakdowns, judge concordance, and bias audits |
| `arena_stage1_results.json` | Full dataset containing 48 debate transcripts and 192 member records |
| `convert_or_hold_task.py` | Core Kaggle benchmark task definition |
| `judge_calibration.py` | 4 mock debate calibration test suite across candidate judges |
| `judge_calibration_results.json` | Empirical calibration outputs for all 4 judges |
| `test_local.py` | Free local unit tests for debate logic and payoff math |
| `test_quota_meter.py` | Live Kaggle CLI quota polling utility |

---

## How to Run

```bash
# 1. Install prerequisites
pip install -U kaggle kaggle-benchmarks pydantic pandas

# 2. Run local tests (free, no API quota needed)
python test_local.py

# 3. Calibrate judges
python judge_calibration.py

# 4. Run Stage 1 of the Arena
python run_arena_stage1.py

# 5. Analyze results
python analyze_arena_stage1.py
```
