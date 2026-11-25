"""Packet network: periodic broadcasts, random latency with jitter and spikes, bursty loss (Gilbert-Elliott),
newest-sample hold at the receiver.  The schedule is generated in advance; it does not depend on the vehicle states."""
import numpy as np

NETWORKS = {
    # Ts: broadcast period; lat: (mean, std, min, max) of the latency; spike: (probability, lo, hi);
    # ge: (loss prob. good, loss prob. bad, P(good->bad), P(bad->good)) per packet
    'nominal': dict(Ts=0.05, lat=(0.06, 0.02, 0.005, 0.2), spike=(0.03, 0.05, 0.15), ge=(0.02, 0.80, 0.03, 0.35)),
    'slow1':   dict(Ts=0.05, lat=(1.00, 0.10, 0.70, 1.40), spike=(0.03, 0.05, 0.15), ge=(0.02, 0.85, 0.04, 0.25)),
    'slow3':   dict(Ts=0.05, lat=(3.00, 0.15, 2.60, 3.30), spike=(0.03, 0.05, 0.15), ge=(0.02, 0.85, 0.04, 0.25)),
    'ideal':   dict(Ts=0.05, lat=(0.0, 0.0, 0.0, 0.0), spike=(0.0, 0.0, 0.0), ge=(0.0, 0.0, 0.0, 1.0)),
}


def const_delay_net(h, Ts=0.001):
    """Fixed latency h and fast sampling: the effective delay stays in [h, h + Ts)."""
    return dict(Ts=Ts, lat=(h, 0.0, h, h), spike=(0.0, 0.0, 0.0), ge=(0.0, 0.0, 0.0, 1.0))


class Schedule:
    """Pre-generated packet schedule.

    send_t[j][k]   : time at which vehicle j broadcasts its k-th packet
    updates        : time-sorted list of (t_arrival, i, j, k) at which receiver i replaces its held sample of j
    stats          : packets sent / lost / discarded as stale, latency statistics
    """

    def __init__(self, A, T, cfg, seed):
        rng = np.random.default_rng(seed)
        A = np.asarray(A, float)
        N = A.shape[0]
        Ts = cfg['Ts']
        lm, ls, lmin, lmax = cfg['lat']
        sp, slo, shi = cfg['spike']
        pg, pb, gb, bg = cfg['ge']
        self.N, self.T, self.cfg = N, T, cfg
        phase = rng.uniform(0, Ts, N) if ls > 0 or sp > 0 else np.zeros(N)
        self.send_t = [phase[j] + Ts * np.arange(int(np.floor((T - phase[j]) / Ts)) + 1) for j in range(N)]
        ups, sent, lost, stale, lats = [], 0, 0, 0, []
        for i in range(N):
            for j in range(N):
                if A[i, j] <= 0:
                    continue
                n = len(self.send_t[j])
                bad = False
                arr = []
                for k in range(n):
                    bad = (rng.random() >= bg) if bad else (rng.random() < gb)
                    sent += 1
                    out = cfg.get('outage')              # (period, duration): deterministic blackout windows
                    if out is not None and (self.send_t[j][k] % out[0]) < out[1]:
                        lost += 1
                        continue
                    if rng.random() < (pb if bad else pg):
                        lost += 1
                        continue
                    lat = lm + ls * rng.standard_normal() if ls > 0 else lm
                    lat = min(max(lat, lmin), lmax)
                    if sp > 0 and rng.random() < sp:
                        lat += rng.uniform(slo, shi)
                    lats.append(lat)
                    arr.append((self.send_t[j][k] + lat, k))
                arr.sort()
                newest = -1
                for (ta, k) in arr:                      # newest-sample hold: discard late and reordered packets
                    if k > newest and ta <= T:
                        ups.append((ta, i, j, k))
                        newest = k
                    elif ta <= T:
                        stale += 1
        ups.sort()
        self.updates = ups
        lats = np.array(lats) if lats else np.zeros(1)
        self.stats = dict(sent=sent, lost=lost, stale=stale, loss_ratio=lost / max(sent, 1),
                          lat_mean=float(lats.mean()), lat_std=float(lats.std()), lat_max=float(lats.max()))

    def needed_sends(self):
        """(j, k) pairs whose sample must be stored because some receiver will use it."""
        return sorted({(j, k) for (_, _, j, k) in self.updates})
