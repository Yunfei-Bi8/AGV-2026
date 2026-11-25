"""Reproduce the whole robustness study:  python3 run_all.py [--force] [r0 r1 ...]

Runs the unit tests, the experiments R0-R9 (R0 first: the tuned gains are used by all others), then the figures and
tables.  An experiment is skipped when both its results and its time series are stored, unless --force is given.  The
repository ships the results (results/robust/*.json) but not the time series (about 110 MB), so the first call
re-runs everything.  Expected wall time on a 20-thread CPU: about 1.5 hours."""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = ['r0_tuning', 'r0b_direct_divergence', 'r1_paper_delays', 'r2_realistic', 'r3_network', 'r4_simreal', 'r5_domain_rand', 'r6_events',
       'r7_scaling', 'r7b_normalised', 'r8_sensitivity', 'r9_numerics']

NO_TS = {'r0b_direct_divergence'}                     # stores no time series
EXTRA = {'r2_realistic': ['r2_trajectories.npz']}     # recorded trajectories for the figures


def stored(e):
    files = [e + '.json'] + ([] if e in NO_TS else [e + '_ts.npz']) + EXTRA.get(e, [])
    return all(os.path.exists(os.path.join(HERE, 'results', 'robust', f)) for f in files)


def main():
    force = '--force' in sys.argv
    sel = [a for a in sys.argv[1:] if not a.startswith('--')]
    todo = [e for e in EXP if not sel or e.split('_')[0] in sel or e in sel]
    if not sel:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'], cwd=HERE, check=True)
    for e in todo:
        if stored(e) and not force:
            print('keep', e)
            continue
        print('run ', e, flush=True)
        subprocess.run([sys.executable, e + '.py'], cwd=os.path.join(HERE, 'experiments', 'robust'), check=True)
    subprocess.run([sys.executable, 'make_figures_robust.py'], cwd=os.path.join(HERE, 'experiments', 'robust'), check=True)


if __name__ == '__main__':
    main()
