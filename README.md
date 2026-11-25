# Delay-Tolerant Formation Tracking for Multi-AGV Systems via Reference-Shift Coupling

Simulation code for the paper

> Yunfei Bi. *Delay-Tolerant Formation Tracking for Multi-AGV Systems via Reference-Shift Coupling.*
> arXiv:2611.32409, 2026.

[Paper (arXiv)](https://arxiv.org/abs/2611.32409) · [Project page](https://bi-yunfei.github.io/agv-formation/)

The repository contains everything needed to reproduce the numerical results of the paper:

- **Numerical verification (E1–E6):** an event-driven simulator of the kinematic model. It checks every theorem,
  proposition and lemma of the stability analysis on the model for which it is stated.
- **Robustness and sim-to-real study (R0–R9):** a high-fidelity AGV simulator and 4,894 runs that compare the cascade
  (reference-shift) coupling with three baselines.
- The stored per-run results, the figures and the LaTeX tables of the paper.

## Setup

Python 3.10 with numpy, scipy and matplotlib. Pinned versions:

```
pip install -r requirements.txt
```

All commands below are run from the repository root unless stated otherwise.

## Quick check (about 2 minutes)

```
python3 -m unittest discover -s tests              # simulator unit tests                                  (~35 s)
python3 experiments/smoke.py figure8 nominal       # one verification run, all checks printed              (~15 s)
python3 experiments/robust/check_stored.py         # re-runs 2 stored jobs of each experiment and compares
                                                   # their metrics with results/robust/ bit for bit         (~1 min)
```

## Robustness and sim-to-real study (R0–R9)

```
python3 run_all.py            # unit tests, R0-R9, figures and tables;  --force recomputes;  r3 r6 ... selects
```

This takes about 1.5 hours on a 20-thread laptop CPU (Intel i9-13900H) with 18 worker processes. R0 runs first,
because the coupling gains it selects are used by all other experiments. Figures go to `figures/robust/` and tables
to `tables/`.

| Experiment | Question | Runs |
| --- | --- | --- |
| `r0_tuning` | Coupling gain of each controller, chosen on tuning seeds 1000–1009 only | 380 |
| `r0b_direct_divergence` | Why the direct coupling diverges: limits removed, ideal profile, 200 Hz control | 160 |
| `r1_paper_delays` | Comparison under the heterogeneous sinusoidal delays of Peng–Ngo–Liu (2023), model of the theorems | 480 |
| `r2_realistic` | With vs. without synchronisation on the high-fidelity AGV model (nominal and traction loss) | 480 (+8 trajectory recordings) |
| `r3_network` | Latency, loss, outages (with/without watchdog), broadcast rate, scaled sinusoidal delays | 650 |
| `r4_simreal` | Sim-to-real factors one at a time (localisation, calibration, slip, payload, delays, rates, clocks, limits) | 1152 |
| `r5_domain_rand` | Domain-randomised Monte Carlo over all factors | 600 |
| `r6_events` | Traction loss, safety stop, relocalisation jump, switching links, permanent link failure | 460 |
| `r7_scaling` | 3–12 vehicles, five graphs, unit weights | 264 |
| `r7b_normalised` | Complete graphs with degree-normalised weights | 48 |
| `r8_sensitivity` | Cascade parameters λ, γ, ρ, ē | 170 |
| `r9_numerics` | Plant step-size convergence (+12 bit-identical re-runs of stored R2 jobs) | 50 |
| **Total** | stored jobs | **4,894** |

The four controllers are `uncoupled` (no coupling), `cascade` (the proposed design), `sigleak` (σ-leak filter) and
`direct` (direct coupling of the delayed neighbour errors).

### High-fidelity simulator (`agvsim/hifi.py`)

Each run passes through the following layers:

- **Controller:** runs at 50 Hz on local clocks (offset, drift), using pose estimates.
- **Drives:** nominal geometry, speed and ramp limits, PI wheel loops with torque saturation, actuation delay.
- **Vehicle:** differential-drive rigid-body dynamics with payload, calibration errors, resistance and slip
  (Ornstein–Uhlenbeck).
- **Localisation:** LiDAR fixes (rate, latency, noise, bias, outliers) fused with odometry.
- **Network:** latency models (including the sinusoidal delays of the reference paper), Gilbert–Elliott loss, outages,
  link switching, newest-sample hold.

Every layer has an ideal setting; `profile='ideal'` gives the model of the theorems.

## Numerical verification of the analysis (E1–E6)

```
python3 experiments/e1_prop1_algebra.py   # Proposition 1 and the perturbed inequality at 2e7 random points       (~1.5 min)
python3 experiments/e2_lemma2_network.py  # Lemma 2 on 1000 exactly integrated delayed networks, rate tightness     (~1 min)
python3 experiments/e3_cascade_mc.py      # 280 closed-loop runs, every claim of Theorems 1-2 checked on every run  (~16 min)
python3 experiments/e4_iss.py             # perturbation sweep, literal check of Theorem 3, broadcast noise         (~2 min)
python3 experiments/e5_formation.py       # Proposition 2: steady formation-error attenuation, traction-loss test   (~4 min)
python3 experiments/e6_direct_lk.py       # Proposition 3: direct-coupling baseline, Lyapunov-Krasovskii functional (~3 min)
python3 experiments/make_figures.py       # figures of the paper (needs the time series written by E3 and E5)
```

The scripts use 16 to 18 processes. Results go to `results/`, figures to `figures/`.

How the numerical error is kept below the reported margins:

- Received samples are piecewise constant, so no delayed state is interpolated; discontinuities lie on step boundaries.
- Tolerances are 1e-10 (relative) and 1e-12 (absolute); the local error estimates sum to less than 3e-8 per run.
- `W_i(t) + D_i(t) = W_i(0)` is checked on every run (residual at most 2e-11 m²).
- The Lemma-2 tests use the closed-form solution between events and have no discretisation error.
- Every run is a deterministic function of its seed.

Not covered here: sampled control, actuator lag, wheel limits and localisation noise. In E4 and E5 they enter only as
bounded perturbations; the high-fidelity study above models them explicitly.

## Repository layout

| Path | Content |
| --- | --- |
| `agvsim/references.py` | Exact references of a rigid virtual structure (circle, figure-8, warehouse route, line); speeds, turn rates and accelerations in closed form |
| `agvsim/network.py` | Packet schedule per directed link: periodic broadcast, latency with jitter and spikes, Gilbert–Elliott loss, newest-sample hold |
| `agvsim/integrator.py` | Dormand–Prince 5(4) with step-size control, integrating from event to event |
| `agvsim/cascade.py` | Closed loop of the cascade design on the kinematic model; heading error as a state; dissipation integrated for the energy balance |
| `agvsim/theory.py` | All constants of the paper: α, c1, c2, Lemma-2 rate, constants of Theorems 2 and 3, delay bound of the direct design |
| `agvsim/verify.py` | Evaluates every claim on a run and returns its margin |
| `agvsim/hifi.py` | High-fidelity AGV simulator (drives, dynamics, localisation, network, clocks, events) |
| `agvsim/runner.py` | Deterministic parallel job runner; stores jobs, metrics and the software environment |
| `agvsim/stats.py` | Bootstrap confidence intervals, paired tests, Holm correction |
| `experiments/` | E1–E6 and their figures |
| `experiments/robust/` | R0–R9, `make_figures_robust.py` (figures and tables), `check_stored.py` |
| `tests/` | Unit tests |
| `results/` | Stored results: E1–E6 summaries, and for R0–R9 the job list, metrics and environment of every run |
| `figures/`, `tables/` | Figures and LaTeX tables of the paper |

## Reproducibility

- **Self-contained jobs.** A run is a deterministic function of its job dictionary. For each experiment,
  `results/robust/<experiment>.json` stores the job list, the metrics, the software environment and a hash of the
  simulator source.
- **Common random numbers.** Each seed gets one `SeedSequence`, spawned into independent streams (initial errors,
  network, sensors, disturbances, clocks, vehicle parameters). The number of draws never depends on the controller or
  on parameter values. All controllers and all sweep levels therefore see the same realisation, which makes the
  comparisons paired.
- **Unit tests.** `tests/test_core.py` covers reference derivatives, the packet scheduler, common random numbers,
  determinism, and agreement with the independent event-driven simulator (0.34 mm).
- **Numerical checks.** `r9_numerics` varies the plant step size (0.5–10 ms) and re-runs stored jobs bit for bit.
- **Figures and tables.** `experiments/robust/make_figures_robust.py` produces them from the stored results and time
  series only.

The time series of the robustness study (`results/robust/*_ts.npz`, about 110 MB) are not part of the repository;
`run_all.py` regenerates them.

The stored results were computed with earlier revisions of `agvsim/`, so the `code_hash` recorded in them differs from
that of this code. `check_stored.py` re-runs stored jobs with this code and confirms that their metrics are reproduced
bit for bit.

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
