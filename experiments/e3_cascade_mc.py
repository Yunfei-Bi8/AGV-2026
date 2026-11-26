"""E3. Cascade closed loop on the kinematic model: every claim of Theorems 1-2 and Lemmas 1-3 (arXiv v2 numbering)
is checked on every run (three references x three networks x 20 seeds, a directed chain, and very large initial
errors). Networks: 'nominal' = 60 ms latency, 'slow1' = 1 s, 'slow3' = 3 s (called the 60 ms, 1 s and 3 s networks in
the paper). Runs: 9 x 20 on the complete graph, 2 x 20 on the directed chain (figure-8), 20 without coupling
(figure-8) and 2 x 20 with large initial errors = 280."""
import json, time
import numpy as np
from multiprocessing import Pool
from common import run_case, RES, GRAPHS
from agvsim import verify as vf, cascade as cs

KEEP = ('t', 'W', 'e', 'eps', 'zeta', 'vh', 'wh', 'tau', 'p', 'Pr', 'head', 'pi', 'chi', 'g', 'z', 'w')


def job(a):
    ref, net, graph, seed, init, tag = a
    sim, run, bounds = run_case(ref, net, graph, seed, init=init)
    A = GRAPHS[graph]
    r = vf.verify_run(run, sim.par, bounds, A.sum(1).max())
    r.update(ref=ref, net=net, graph=graph, seed=seed, tag=tag, loss=run['meta']['net']['loss_ratio'] if run['meta']['net'] else 0.0)
    if seed == 7 and tag == 'std':
        np.savez_compressed(RES + f'e3_{ref}_{net}_{graph}.npz', **{k: run[k] for k in KEEP})
    return r


if __name__ == '__main__':
    t0 = time.time()
    std = (0.1, 0.5, np.deg2rad(40))
    big = (1.0, 3.0, np.deg2rad(175))
    jobs = [(ref, net, 'complete', s, std, 'std') for ref in ('circle', 'figure8', 'warehouse')
            for net in ('nominal', 'slow1', 'slow3') for s in range(20)]
    jobs += [('figure8', 'nominal', 'chain', s, std, 'std') for s in range(20)]
    jobs += [('figure8', 'slow3', 'chain', s, std, 'std') for s in range(20)]
    jobs += [('figure8', 'nominal', 'none', s, std, 'std') for s in range(20)]
    jobs += [('figure8', 'nominal', 'complete', s, big, 'large') for s in range(20)]
    jobs += [('warehouse', 'slow3', 'complete', s, big, 'large') for s in range(20)]
    with Pool(18) as pool:
        res = pool.map(job, jobs, chunksize=1)
    json.dump(res, open(RES + 'e3_runs.json', 'w'), indent=0, default=float)
    # aggregate
    groups = {}
    for r in res:
        groups.setdefault((r['ref'], r['net'], r['graph'], r['tag']), []).append(r)
    summ = []
    for key, rr in groups.items():
        g = lambda k, f: float(f([x[k] for x in rr]))
        summ.append(dict(ref=key[0], net=key[1], graph=key[2], tag=key[3], runs=len(rr), all_ok=all(x['all_ok'] for x in rr),
                         h_min=g('h', min), h_max=g('h', max), loss=g('loss', np.mean), E0_max=g('E0', max),
                         vhat_margin_min=float(min(x['vhat_min_obs'] - x['vhat_min_thm'] for x in rr)),
                         W_rise_max=g('W_max_rise_rel', max), balance_max=g('balance_resid', max),
                         identity_max=g('Wdot_identity_resid', max),
                         eps_ratio_max=g('eps_ratio', max), zeta_ratio_max=g('zeta_ratio', max), e_ratio_max=g('e_ratio', max),
                         lemma2_ratio_max=g('lemma2_ratio', max),
                         Vdot_ratio_min_run=g('Vdot_ratio_min_run', min), Vdot_ratio_min_apriori=g('Vdot_ratio_min_apriori', min),
                         propc_max_run=g('propc_ratio_max_run', max), propc_max_apriori=g('propc_ratio_max_apriori', max),
                         thm2_max=float(max(max(x['thm2_eps_ratio'], x['thm2_e_ratio'], x['thm2_head_ratio']) for x in rr)),
                         thm2_t1_max=float(max(max(x['thm2_eps_ratio_t1'], x['thm2_e_ratio_t1'], x['thm2_head_ratio_t1']) for x in rr)),
                         thm2_t1_eps=g('thm2_eps_ratio_t1', max), thm2_t1_e=g('thm2_e_ratio_t1', max),
                         thm2_t1_head=g('thm2_head_ratio_t1', max), rate_eps=g('rate_eps', np.nanmedian),
                         e_end_max=g('e_end', max), head_end_max=g('head_end', max), form_end_max=g('form_end', max),
                         rate_W=g('rate_W', np.nanmedian), rate_M=g('rate_M', np.nanmedian), rate_e=g('rate_e', np.nanmedian),
                         rate_lemma2=g('rate_lemma2', np.median), eta1_apriori=g('eta1', np.median),
                         t_form_2cm_med=g('t_form_2cm', np.median), t_form_2cm_max=g('t_form_2cm', max),
                         curv_run_max=g('curv_run', max), curv_apriori=g('curv_apriori', max),
                         err_sum_max=g('err_sum', max)))
    json.dump(dict(summary=summ, wall_s=time.time() - t0), open(RES + 'e3_summary.json', 'w'), indent=1)
    for s in summ:
        print({k: (round(v, 4) if isinstance(v, float) and abs(v) > 1e-3 else v) for k, v in s.items()})
    print('wall', time.time() - t0)
