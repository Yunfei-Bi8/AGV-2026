"""R2. With and without synchronisation on the high-fidelity AGV model (counterpart of the hardware experiment of
Peng-Ngo-Liu, Figs. 8-10): dynamics, PI wheel loops, limits, calibration errors, slip, LiDAR+odometry localisation,
clock offsets, Wi-Fi.  (a) nominal operation, (b) with a traction loss."""
import numpy as np
from common import job, run_jobs, tuned_ctrl, CONTROLLERS, TEST_SEEDS, traction_event
from agvsim import runner, hifi as hf

if __name__ == '__main__':
    jobs = []
    for case in ('nominal', 'event'):
        for sc in ('transport', 'warehouse'):
            for s in TEST_SEEDS:
                ev = [traction_event(s)] if case == 'event' else []
                for typ in CONTROLLERS:
                    jobs.append(job(sc, tuned_ctrl(typ), s, events=ev, case=case))
    doc = run_jobs('r2_realistic', jobs)
    # full trajectories of one seed for the figures
    keep = {}
    for sc in ('transport', 'warehouse'):
        for typ in CONTROLLERS:
            j = job(sc, tuned_ctrl(typ), 7, events=[traction_event(7)], sim=dict(keep=True))
            R = runner.make_sim(j).run()
            for k in ('t', 'p', 'P', 'evec', 'eevec', 'loc'):
                keep[f'{sc}/{typ}/{k}'] = R[k]
    np.savez_compressed(runner.RES + '/r2_trajectories.npz', **keep)
    print('wall', doc['wall_s'])
