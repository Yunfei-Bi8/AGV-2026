"""R0. Fair tuning: each controller's coupling gain is chosen on tuning seeds (never used for testing).
Objective: formation-error IAE relative to the uncoupled law, averaged over two scenarios and ten seeds, with a random
traction loss in every run (the situation that coupling is meant to handle)."""
import json, os
import numpy as np
from common import job, run_jobs, TUNE_SEEDS, traction_event, RES

GRID = {'cascade': ('Lam', [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]),
        'sigleak': ('Lam', [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]),
        'direct': ('c', [0.1, 0.25, 0.5, 1.0, 2.0, 4.0])}
SCEN = ['warehouse', 'transport']

if __name__ == '__main__':
    jobs = []
    for sc in SCEN:
        for s in TUNE_SEEDS:
            ev = [traction_event(s)]
            jobs.append(job(sc, dict(type='uncoupled'), s, events=ev))
            for typ, (par, vals) in GRID.items():
                for v in vals:
                    jobs.append(job(sc, {'type': typ, par: v}, s, events=ev))
    doc = run_jobs('r0_tuning', jobs)
    # selection
    J = {}
    base = {(j['scen'], j['seed']): m for j, m in zip(doc['jobs'], doc['metrics']) if j['ctrl']['type'] == 'uncoupled'}
    summary = {}
    for typ, (par, vals) in GRID.items():
        rows = []
        for v in vals:
            fr, tr, fail = [], [], 0
            for j, m in zip(doc['jobs'], doc['metrics']):
                if j['ctrl']['type'] == typ and j['ctrl'][par] == v:
                    if m.get('failed'):
                        fail += 1; continue
                    b = base[(j['scen'], j['seed'])]
                    fr.append(m['form_iae'] / b['form_iae']); tr.append(m['track_iae'] / b['track_iae'])
            rows.append(dict(value=v, form_ratio=float(np.mean(fr)) if fr else np.inf, track_ratio=float(np.mean(tr)) if tr else np.inf, failed=fail))
        ok = [r for r in rows if r['failed'] == 0]
        best = min(ok, key=lambda r: (r['form_ratio'], r['value']))
        summary[typ] = dict(param=par, rows=rows, best=best['value'])
    choice = {typ: {summary[typ]['param']: summary[typ]['best']} for typ in summary}
    json.dump(choice, open(os.path.join(RES, 'r0_tuning_choice.json'), 'w'), indent=1)
    json.dump(summary, open(os.path.join(RES, 'r0_tuning_summary.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))
    print('choice', choice, 'wall', doc['wall_s'])
