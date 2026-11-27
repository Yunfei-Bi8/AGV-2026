# Delay-Tolerant Formation Tracking for Multi-AGV Systems via Reference-Shift Coupling

Simulation code and data for the paper

> Yunfei Bi. *Delay-Tolerant Formation Tracking for Multi-AGV Systems via Reference-Shift Coupling.*
> arXiv:2611.32409, 2026.

[Paper (arXiv)](https://arxiv.org/abs/2611.32409) · [Project page](https://bi-yunfei.github.io/agv-formation/)

Version 2 of the paper (arXiv v2) is the 6-page conference paper with an appendix that contains all proofs; the tag
`arxiv-v2` marks the code that generated its figures and table. The repository contains everything needed to reproduce
its numerical results:

- **Numerical verification (E1–E7):** an event-driven simulator of the kinematic model. It checks every theorem,
  proposition and lemma of the stability analysis on the model for which it is stated.
- **Robustness study (R0–R12):** a detailed AGV simulator and 5,944 runs that compare the cascade (reference-shift)
  coupling with four baselines. The runs of R1 and R10 use the simulator in its ideal setting, which is the kinematic
  model.
- **Stored data:** the per-run results of all experiments and their time series (`results/`, about 140 MB).
- **Paper scripts:** `paper/make_paper_figures.py` and `paper/make_paper_tables.py` regenerate every figure and table
  of the paper from the stored data.

## Setup

Python 3.10 with numpy, scipy and matplotlib. Pinned versions:

```
pip install -r requirements.txt
```

All commands below are run from the repository root unless stated otherwise. The experiments use 16 to 18 worker
processes; the times below were measured on a laptop with an Intel i9-13900H (20 hardware threads).

## Quick check (about 2 minutes)

```
python3 -m unittest discover -s tests              # simulator unit tests                                   (~40 s)
python3 experiments/smoke.py figure8 nominal       # one verification run, all checks printed               (~15 s)
python3 experiments/robust/check_stored.py         # re-runs 2 stored jobs of each experiment and compares
                                                   # all their metrics with results/robust/ bit for bit      (~1 min)
```

## Figures and tables of the paper

```
python3 paper/make_paper_figures.py     # all figures -> paper/figures/   (~15 s)
python3 paper/make_paper_tables.py      # all tables  -> paper/tables/    (~15 s)
```

Both scripts only read `results/`; nothing is simulated. The paper (arXiv v2) uses three of their outputs:

| Paper | Output | Data (experiment, wall time to recompute) |
| --- | --- | --- |
| Fig. 2 (kinematic model) | `fig_kin_cdc.pdf` | `results/robust/r10_delay_margin*` (R10, 14 min), `results/e5_formation.json` (E5, 4 min), `results/e7_curved.json` (E7, 2 min) |
| Fig. 3 (detailed AGV model) | `fig_hifi.pdf` | `results/robust/r6_events*` (R6, 7 min), `summary.json` (R3, 12 min), `r12_age_gate*` (R12, 2 min), `r0_tuning_summary.json` (R0, 6 min) |
| Table I (main results on the detailed AGV model) | `tab_hifi_cdc.tex` | R0, R2, R5, R6, R7, R7b, R11 (10 min) |

Fig. 1 is drawn in the paper. The other outputs of the two scripts document the study in more detail: the worst cases
of the verification (`tab_verif.tex`), the rates of Lemmas 2 and 3 (`tab_scale.tex`, `tab_rates.tex`,
`tab_meas.tex`), the traction loss on the kinematic model (`tab_slip.tex`), the perturbation sweep (`fig_iss.pdf`), the
network sweeps, practical factors and team sizes (`fig_r3_network.pdf`, `fig_r4_simreal_steady.pdf`,
`fig_r7_scaling.pdf`), and the tables of R5, R6 and R7b. All other figures and tables of the study are produced by
`experiments/robust/make_figures_robust.py` (into `figures/robust/` and `tables/`) and `experiments/make_figures.py`.
`python3 paper/paper_numbers.py` prints every number quoted in Sections IV and V of the paper with its source (about
20 s); the numbers come from `results/e2_lemma2.json`, `results/e3_runs.json`, `results/e5_formation.json`,
`results/e7_curved.json`, `results/robust/summary.json`, the per-run results in `results/robust/*.json` and the
formulas in `agvsim/theory.py`. The figure PDFs are byte-identical on
Linux with the pinned packages and the Liberation Serif font; the tables do not depend on fonts.

## Robustness study (R0–R12)

```
python3 run_all.py            # unit tests, figures and tables;  --force re-runs R0-R9;  r3 r6 ... selects
```

An experiment is skipped when its results and time series are stored, so a plain call only regenerates the figures and
tables. `python3 run_all.py --force` re-runs all experiments (about 2 hours). R0 runs first, because the coupling
gains it selects are used by all other experiments. The figures and tables of the study go to `figures/robust/` and
`tables/`, and those of the paper to `paper/`.

| Experiment | Question | Runs | Wall time |
| --- | --- | --- | --- |
| `r0_tuning` | Coupling gain of each controller, chosen on tuning seeds 1000–1009 only | 380 | 6 min |
| `r0b_direct_divergence` | Why the direct coupling diverges: limits removed, ideal profile, 200 Hz control | 160 | 3 min |
| `r1_paper_delays` | Comparison under the heterogeneous sinusoidal delays of Peng–Ngo–Liu (2023), kinematic model | 480 | 8 min |
| `r2_realistic` | With vs. without coupling on the detailed AGV model (nominal and traction loss) | 480 (+8 trajectory recordings) | 8 min |
| `r3_network` | Latency, loss, outages (with/without watchdog), broadcast rate, scaled sinusoidal delays | 650 | 12 min |
| `r4_simreal` | Practical factors one at a time (localization, calibration, slip, payload, delays, rates, clocks, limits) | 1152 | 19 min |
| `r5_domain_rand` | Domain-randomized Monte Carlo over all factors | 600 | 11 min |
| `r6_events` | Traction loss, safety stop, relocalization jump, switching links, permanent link failure | 460 | 7 min |
| `r7_scaling` | 3–12 vehicles, five graphs, unit weights | 264 | 6 min |
| `r7b_normalised` | Complete graphs with degree-normalized weights | 48 | 2 min |
| `r8_sensitivity` | Cascade parameters λ, γ, ρ, ē | 170 | 3 min |
| `r9_numerics` | Plant step-size convergence (+12 bit-identical reruns of stored R2 jobs) | 50 | 2 min |
| `r10_delay_margin` | Kinematic model, straight reference, constant latency 0.03–4 s: direct coupling with c d = 1–8 against the linear delay condition (Proposition 2), cascade with Λ = 1, 4, 8 | 432 | 14 min |
| `r11_highgain` | Gain-matched baseline: the jobs of R0 (grid of c), R2, R5, R6 and R7/R7b (complete graphs) with `highgain` | 498 | 10 min |
| `r12_age_gate` | Latency sweep of R3 with an age gate: samples older than 1.5 s are discarded | 120 | 2 min |
| **Total** | stored jobs | **5,944** | **113 min** |

The five controllers are `uncoupled` (no coupling), `cascade` (the proposed design), `sigleak` (σ-leak filter),
`direct` (direct coupling of the delayed neighbor errors) and `highgain` (the tracking law with the gains of the
direct coupling, its error multiplied by 1 + c d_i, but without neighbor data; `tuned_ctrl('highgain')` uses the c
selected for `direct`).

### Detailed AGV simulator (`agvsim/hifi.py`)

Each run passes through the following layers:

- **Controller:** runs at 50 Hz, sampled on the common time grid; the reference time of each vehicle follows its
  local clock (offset, drift). It uses pose estimates.
- **Drives:** nominal geometry, speed and ramp limits, PI wheel loops with torque saturation, actuation delay.
- **Vehicle:** differential-drive rigid-body dynamics with payload, calibration errors, resistance and slip
  (Ornstein–Uhlenbeck).
- **Localization:** LiDAR-type fixes (rate, latency, noise, bias, outliers) fused with odometry.
- **Network:** latency models (including the sinusoidal delays of the reference paper), Gilbert–Elliott loss, outages,
  link switching, newest-sample hold.

Every layer has an ideal setting; `profile='ideal'` gives the kinematic model.

**Timing on the control grid.** Broadcasts and localization fixes are scheduled on the control grid: a broadcast rate
`fb` is realized as one broadcast every `round(fc/fb)` control periods, a fix rate as one fix every `round(fc/f_fix)`
periods and a fix latency as `round(d_fix*fc)` periods (Python rounds halves to even). At the nominal `fc = 50 Hz`, the
nominal 20 Hz broadcasts, 20 Hz fixes and 50 ms fix latency are therefore realized as **25 Hz, 25 Hz and 40 ms**; the
paper reports the realized values. `agvsim.hifi.realized_timing` computes them, every run made with the published code stores them in
`meta['realized']` (the stored results predate this key; their timing is the same), and the simulator warns when a requested value is not realized exactly. In the control-rate sweep of
R4, the timing changes with the control rate (at 10 Hz: 10 Hz broadcasts and fixes without fix latency).

## Numerical verification of the analysis (E1–E6)

```
python3 experiments/e1_prop1_algebra.py   # Lemma 2 and the perturbed inequality at 2e7 random points            (~1.5 min)
python3 experiments/e2_lemma2_network.py  # Lemma 3 on 1000 exactly integrated delayed networks, rate tightness   (~1 min)
python3 experiments/e3_cascade_mc.py      # 280 closed-loop runs, every claim of Theorems 1-2 checked on every run (~16 min)
python3 experiments/e4_iss.py             # perturbation sweep, Theorem 3 of v1 (Corollary 1 of v2), broadcast noise (~2 min)
python3 experiments/e5_formation.py       # Proposition 1: steady formation-error attenuation, traction-loss test   (~4 min)
python3 experiments/e6_direct_lk.py       # direct-coupling baseline, delay bound of v1 (not in v2)               (~3 min)
python3 experiments/e7_curved.py          # Proposition 1 on a circle against the transfer function of Remark 2    (~2 min)
python3 experiments/make_figures.py       # figures of the study (paper figures: paper/make_paper_figures.py)
```

The script names keep the numbering of arXiv v1 (see below). Results go to `results/`, figures to `figures/`.

How the numerical error is kept below the reported margins:

- Received samples are piecewise constant, so no delayed state is interpolated; discontinuities lie on step boundaries.
- Tolerances are 1e-10 (relative) and 1e-12 (absolute); the local error estimates sum to less than 3e-8 per run.
- `W_i(t) + Q_i(t) = W_i(0)` (energy balance with the integrated dissipation) is checked on every run (residual at
  most 2e-11 m²).
- The tests of Lemma 3 use the closed-form solution between events and have no discretization error.
- Every run is a deterministic function of its seed.

Not covered here: sampled control, actuator lag, wheel limits and localization noise. In E4 and E5 they enter only as
bounded perturbations; the detailed study models them explicitly.

## Numbering of the results

Version 1 of the paper (a technical report) and version 2 (the conference paper with an appendix of proofs) number
the results differently. The code and the keys of the stored results keep the v1 names.

| arXiv v2 | arXiv v1 | Code / result keys |
| --- | --- | --- |
| Lemma 2 (strict Lyapunov function) | Proposition 1 | `e1_prop1_algebra.py`, `ok_prop1`, `Vdot_ratio_*`, `propc_*` |
| Lemma 3 (delayed filter network) | Lemma 2 | `e2_lemma2_network.py`, `lemma2_ratio`, `rate_lemma2` |
| Theorem 2 (exponential estimates) | Theorem 2 | `thm2_*` |
| Remark 1 (asymptotic rate) | -- | `theory.asymptotic_rate` |
| Corollary 1, Appendix (input-to-state estimate) | Theorem 3 | `e4_iss.py`, `theory.iss_constants` |
| Proposition 1 (formation attenuation) | Proposition 2 | `e5_formation.py` |
| Remark 2 (transfer function of the filter network) | -- | `e7_curved.py`, `theory.filter_response` |
| Proposition 2 (linearized direct coupling, delay condition) | -- | `r10_delay_margin.py`, `theory.direct_delay_margin`, `theory.direct_rhp_roots` |
| -- | Proposition 3 (direct coupling, delay bound) | `e6_direct_lk.py` |

The constants of Theorem 2 in v2, given in its appendix, do not depend on the delay bound (`agvsim/theory.py`); E3 was
rerun with them, and `thm2_*_t1` reports the ratios for t ≥ 1 s.

## Repository layout

| Path | Content |
| --- | --- |
| `agvsim/references.py` | Exact references of a rigid virtual structure (circle, figure-8, warehouse route, line); speeds, turn rates and accelerations in closed form |
| `agvsim/network.py` | Packet schedule per directed link: periodic broadcast, latency with jitter and spikes, Gilbert–Elliott loss, newest-sample hold |
| `agvsim/integrator.py` | Dormand–Prince 5(4) with step-size control, integrating from event to event |
| `agvsim/cascade.py` | Closed loop of the cascade design on the kinematic model; heading error as a state; dissipation integrated for the energy balance |
| `agvsim/theory.py` | All constants of the paper: α, c1, c2, rate of Lemma 3, constants of Theorem 2 and Corollary 1, asymptotic rate, transfer function of Remark 2, delay condition of Proposition 2, delay bound of v1 |
| `agvsim/verify.py` | Evaluates every claim on a run and returns its margin |
| `agvsim/hifi.py` | Detailed AGV simulator (drives, dynamics, localization, network, clocks, events) |
| `agvsim/runner.py` | Deterministic parallel job runner; stores jobs, metrics and the software environment |
| `agvsim/stats.py` | Bootstrap confidence intervals, paired tests, Holm correction |
| `experiments/` | E1–E7 and their figures |
| `experiments/robust/` | R0–R12, `make_figures_robust.py` (figures and tables of the study), `check_stored.py` |
| `paper/` | Scripts for the figures and tables of the paper, and their output |
| `tests/` | Unit tests |
| `results/` | Stored results and time series of E1–E7 and R0–R12 |
| `figures/`, `tables/` | Figures and tables of the study |

## Reproducibility

- **Self-contained jobs.** A run is a deterministic function of its job dictionary. For each experiment,
  `results/robust/<experiment>.json` stores the job list, the metrics, the software environment and a hash of the
  simulator source.
- **Common random numbers.** Each seed gets one `SeedSequence`, spawned into independent streams (initial errors,
  network, sensors, disturbances, clocks, vehicle parameters). The number of draws does not depend on the controller,
  so all controllers see the same realization, and the comparisons between controllers are paired. The same holds
  across sweeps of latency, loss and noise magnitudes. Changing a rate (control, broadcast or fix) or the graph changes
  the number of draws and hence the realization.
- **Unit tests.** `tests/test_core.py` covers reference derivatives, the packet scheduler, common random numbers (all
  random streams end in the same state for all five controllers), the timing on the control grid, determinism, and
  the agreement with the independent event-driven simulator (0.34 mm).
- **Numerical checks.** `r9_numerics` varies the plant step size (0.5–10 ms) and reruns stored jobs bit for bit.
  `check_stored.py` and `r9_numerics` compare all metrics, including the failure flag.
- **Platform.** Bit-for-bit agreement is checked on Linux x86-64 with the pinned package versions
  (`results/robust/*.json` record the environment). Other platforms or library versions can differ in the last digits.
- **Provenance.** The stored results were computed with earlier, unpublished revisions of `agvsim/` (commit and
  `code_hash` recorded in the results); for E7 and R10–R12, that revision differs from the published code only in a
  comment. Rerunning stored jobs with the published code reproduces their metrics bit for bit (`check_stored.py`, R9);
  the changes since R0–R9 concern documentation, metadata and checks, and the added controller `highgain`, which leaves
  the other controllers unchanged.
- **Figures and tables.** `experiments/robust/make_figures_robust.py` and the scripts in `paper/` use the stored
  results and time series only. An experiment whose data are missing is skipped, and the stored summary is kept.

## Citation

```bibtex
@article{bi2026delay,
  title   = {Delay-Tolerant Formation Tracking for Multi-{AGV} Systems via Reference-Shift Coupling},
  author  = {Bi, Yunfei},
  journal = {arXiv preprint arXiv:2611.32409},
  year    = {2026}
}
```

## License

MIT License, see [LICENSE](LICENSE). Copyright (c) 2026 Yunfei Bi.
