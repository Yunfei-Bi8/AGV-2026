"""Deterministic job runner for the robustness study.

A job is a plain dict (JSON-serialisable).  run_job(job) depends only on the job, never on the process or the order
in which jobs are executed, so every number of the study can be regenerated from the stored job list."""
import hashlib, json, os, platform, sys, time, glob
import numpy as np
from multiprocessing import Pool
from . import hifi as hf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results', 'robust')
TS_DT = 0.1                                           # stored time series: 10 Hz


def code_hash():
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(ROOT, 'agvsim', '*.py'))):
        h.update(open(f, 'rb').read())
    return h.hexdigest()[:16]


def environment():
    import scipy, matplotlib
    git = None
    try:
        import subprocess
        git = subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or None
    except Exception:
        pass
    return dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                matplotlib=matplotlib.__version__, platform=platform.platform(), cpus=os.cpu_count(),
                git_commit=git, code_hash=code_hash(), date=time.strftime('%Y-%m-%d %H:%M:%S'))


def make_sim(job):
    N = len(hf.make_scenario(job['scen'])['offs'])
    A = hf.graph(job.get('graph', 'complete'), N) if job['ctrl'].get('type', 'cascade') != 'uncoupled' else hf.graph('none', N)
    return hf.HifiSim(job['scen'], A, ctrl=job.get('ctrl'), plant=job.get('plant'), sensor=job.get('sensor'),
                      net=job.get('net'), clock=job.get('clock'), sim=job.get('sim'), profile=job.get('profile', 'realistic'),
                      netname=job.get('netname', 'wifi'), seed=job['seed'], events=job.get('events', ()), T=job.get('T'))


def run_job(job):
    t0 = time.time()
    R = make_sim(job).run()
    out = dict(metrics=R['metrics'], meta=R['meta'], wall=time.time() - t0)
    if not R['failed'] and 't' in R:
        t = R['t']
        idx = np.unique(np.searchsorted(t, np.arange(0.0, t[-1] + 1e-9, TS_DT) - 1e-9))
        idx = idx[idx < len(t)]
        ts = dict(t=t[idx], form=R['form'][idx].max(1) if R['form'].shape[1] else np.zeros(len(idx)),
                  form_est=R['form_est'][idx].max(1) if R['form_est'].shape[1] else np.zeros(len(idx)),
                  track=R['e'][idx].mean(1), track_max=R['e'][idx].max(1), sync=R['sync'][idx], tau=R['tau'][idx],
                  e=R['e'][idx], loc=R['loc'][idx].mean(1))
        out['ts'] = ts
    return out


def _worker(arg):
    i, job = arg
    return i, run_job(job)


def run_jobs(name, jobs, procs=18, keep_ts=True):
    """Runs all jobs, stores results/robust/<name>.json (jobs, metrics, environment) and <name>_ts.npz."""
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    res = [None] * len(jobs)
    with Pool(procs) as pool:
        for i, r in pool.imap_unordered(_worker, list(enumerate(jobs)), chunksize=1):
            res[i] = r
    doc = dict(name=name, environment=environment(), wall_s=time.time() - t0, n_jobs=len(jobs),
               jobs=jobs, metrics=[r['metrics'] for r in res], meta=[r['meta'] for r in res],
               wall=[r['wall'] for r in res])
    with open(os.path.join(RES, name + '.json'), 'w') as f:
        json.dump(doc, f, default=_default)
    if keep_ts:
        arrs = {}
        for i, r in enumerate(res):
            if 'ts' in r:
                for k, v in r['ts'].items():
                    arrs[f'{i}/{k}'] = np.asarray(v, dtype=np.float32 if k != 't' else np.float64)
        np.savez_compressed(os.path.join(RES, name + '_ts.npz'), **arrs)
    return doc


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def load(name):
    with open(os.path.join(RES, name + '.json')) as f:
        doc = json.load(f)
    p = os.path.join(RES, name + '_ts.npz')
    ts = np.load(p) if os.path.exists(p) else None
    return doc, ts


def series(ts, i, key):
    return ts[f'{i}/{key}']


def tuned_ctrl(typ):
    """Coupling gains selected in r0_tuning (each controller tuned on its own, on tuning seeds)."""
    base = dict(type=typ)
    p = os.path.join(RES, 'r0_tuning_choice.json')
    if typ == 'uncoupled' or not os.path.exists(p):
        return base
    choice = json.load(open(p))
    base.update(choice.get(typ, {}))
    return base
