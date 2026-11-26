"""Unit tests: python3 -m unittest discover -s tests -v   (from the repository root)."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agvsim import references as rf, hifi as hf, network as nw, cascade as cs


class TestReferences(unittest.TestCase):
    def test_rotating_formation_derivatives(self):
        """Analytic velocity and acceleration of the transport references agree with finite differences."""
        ev = hf.RefEval(rf.make_scenario('transport'), 60.0)
        h = 1e-4
        for t in np.linspace(1.0, 59.0, 25):
            P, thr, vr, wr, vrd = ev.state(t)
            Pp, Pm = ev.state(t + h)[0], ev.state(t - h)[0]
            V = (Pp - Pm) / (2 * h)
            np.testing.assert_allclose(V, np.stack([vr * np.cos(thr), vr * np.sin(thr)], 1), atol=1e-6)
            vd = (ev.state(t + h)[2] - ev.state(t - h)[2]) / (2 * h)
            np.testing.assert_allclose(vd, vrd, atol=1e-5)
            th_p, th_m = ev.state(t + h)[1], ev.state(t - h)[1]
            np.testing.assert_allclose(hf.wrap(th_p - th_m) / (2 * h), wr, atol=1e-5)

    def test_reference_speed_positive(self):
        for name in ('circle', 'figure8', 'warehouse', 'transport', 'loop12'):
            sc = rf.make_scenario(name)
            ev = hf.RefEval(sc, sc['T'])
            vmin = min(ev.state(t)[2].min() for t in np.arange(0, sc['T'], 0.05))
            self.assertGreater(vmin, 0.05, name)


class TestNetwork(unittest.TestCase):
    def test_newest_sample_hold(self):
        rng = np.random.default_rng(3)
        A = hf.graph('complete', 4)
        cfg = hf.merged(hf.NET, dict(lat='lognorm', lat_mean=0.2, lat_sd=0.8, lat_min=0.01))
        net = hf.Network(A, 2000, 50.0, cfg, rng)
        for L in net.links:
            self.assertTrue(np.all(np.diff(L['ks']) > 0))                     # sequence numbers increase
            self.assertTrue(np.all(np.diff(L['arr']) >= 0))                    # arrivals ordered
            self.assertTrue(np.all(L['arr'] >= L['ks'] / 50.0 + 0.01 - 1e-12))  # never earlier than the minimum latency

    def test_common_random_numbers(self):
        """All controllers draw the same random numbers: after a run, every stream (initial errors, network, sensors,
        disturbances, clocks, vehicle parameters) is in the same state, whichever controller was used."""
        states = []
        for typ in ('uncoupled', 'cascade', 'sigleak', 'direct'):
            out = hf.HifiSim('warehouse', hf.graph('complete', 3), ctrl=dict(type=typ), sim=dict(record_rng=True),
                             seed=11, T=4.0).run()
            self.assertFalse(out['failed'])
            states.append(out['rng_state'])
        for other in states[1:]:
            self.assertEqual(states[0], other)
        # a different seed changes the streams, so the comparison above is not vacuous
        out = hf.HifiSim('warehouse', hf.graph('complete', 3), sim=dict(record_rng=True), seed=12, T=4.0).run()
        self.assertNotEqual(states[0], out['rng_state'])

    def test_timing_on_control_grid(self):
        """Broadcast and fix timing is realized on the control grid (20 Hz -> 25 Hz, 50 ms -> 40 ms at 50 Hz)."""
        r = hf.realized_timing(50.0, 20.0, 20.0, 0.05)
        self.assertAlmostEqual(r['fb'], 25.0); self.assertAlmostEqual(r['f_fix'], 25.0); self.assertAlmostEqual(r['d_fix'], 0.04)
        out = hf.HifiSim('warehouse', hf.graph('complete', 3), seed=1, T=2.0).run()
        self.assertEqual(out['meta']['realized'], dict(fb=25.0, f_fix=25.0, d_fix=0.04))


class TestSimulation(unittest.TestCase):
    def test_determinism(self):
        out = [hf.HifiSim('transport', hf.graph('complete', 3), seed=5, T=8.0).run() for _ in range(2)]
        for k in ('e', 'form', 'loc', 'v'):
            np.testing.assert_array_equal(out[0][k], out[1][k])

    def test_ideal_limit_matches_event_driven_simulator(self):
        """hifi with the ideal profile at 1 kHz reproduces the independent event-driven simulator of the theorems."""
        h, T = 0.5, 20.0
        dp0 = np.array([[0.2, -0.1], [-0.15, 0.2], [0.1, 0.25]]); vt0 = np.array([0.2, -0.15, 0.1])
        ref, _ = rf.make_reference('figure8')
        A = np.ones((3, 3)) - np.eye(3)
        sched = nw.Schedule(A, T, nw.const_delay_net(h, Ts=0.001), 1)
        sim = cs.CascadeSim(ref, rf.TRIANGLE, A, {}, sched)
        run = sim.run(sim.initial_state(dp0, vt0), T, rec_dt=0.01)
        hsim = hf.HifiSim('figure8', A, ctrl=dict(type='cascade', fc=1000.0), profile='ideal', netname='ideal',
                          net=dict(lat='const', lat_mean=h, fb=1000.0), seed=0, T=T, init_override=(dp0, vt0))
        R = hsim.run()
        e_ev = np.linalg.norm(run['e'], axis=2)
        k = np.searchsorted(R['t'], run['t'] - 1e-9)
        diff = np.abs(R['e'][k] - e_ev).max()
        print(f'  [ideal limit] max |e| difference between the two simulators: {diff:.2e} m')
        self.assertLess(diff, 2e-3, f'max difference {diff:.2e} m')


if __name__ == '__main__':
    unittest.main()
