"""R11. Gain-matched baseline ('highgain'): the tracking law with the gains of the direct coupling, that is, with the
error multiplied by 1 + c d_i (c as selected in R0 for the direct coupling), but without any neighbor data. It separates
the effect of the higher gain of the direct coupling from that of the exchanged data. The jobs repeat those of R0
(tuning grid of the direct coupling), R2, R5, R6 (traction loss, safety stop, switching links) and R7/R7b (complete
graphs) with this law and the same seeds, so that all comparisons are paired; the field 'exp' names the experiment that
a job repeats."""
from common import job, run_jobs, tuned_ctrl, TEST_SEEDS, TUNE_SEEDS, traction_event
from r0_tuning import GRID, SCEN
from r5_domain_rand import draw, SEEDS as SEEDS5
from r6_events import EVENTS, SEEDS as SEEDS6
from r7_scaling import EV as EV7, SEEDS as SEEDS7

if __name__ == '__main__':
    jobs = []
    for sc in SCEN:
        for s in TUNE_SEEDS:
            for c in GRID['direct'][1]:
                jobs.append(job(sc, dict(type='highgain', c=c), s, events=[traction_event(s)], exp='r0'))
    for case in ('nominal', 'event'):
        for sc in ('transport', 'warehouse'):
            for s in TEST_SEEDS:
                ev = [traction_event(s)] if case == 'event' else []
                jobs.append(job(sc, tuned_ctrl('highgain'), s, events=ev, case=case, exp='r2'))
    for s in SEEDS5:
        scen, cx, plant, sensor, net, clock, ev = draw(s)
        c = tuned_ctrl('highgain')
        c.update(cx)
        jobs.append(job(scen, c, s, plant=plant, sensor=sensor, net=net, clock=clock, events=ev, exp='r5'))
    for s in SEEDS6:
        for name in ('traction', 'stop', 'switching'):
            net = dict(switching=(5.0, 2.0)) if name == 'switching' else None
            jobs.append(job('warehouse', tuned_ctrl('highgain'), s, events=EVENTS[name], net=net, case=name, exp='r6'))
    for N in (3, 5, 8, 12):
        for s in SEEDS7:
            for g, e in (('complete', 'r7'), ('complete_norm', 'r7b')):
                jobs.append(job(f'loop{N}', tuned_ctrl('highgain'), s, events=EV7, N=N, graph=g, exp=e))
    doc = run_jobs('r11_highgain', jobs)
    print('wall', doc['wall_s'])
