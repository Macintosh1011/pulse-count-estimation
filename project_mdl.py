"""MDL evaluation for Michael Saleev's supplied simulation (kept in private/)."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import sys
import json
import csv
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.signal import correlate
sys.path.insert(0, str(Path(__file__).parent / 'private'))
import mdl_sim_setup as sim

OUT = Path(__file__).parent / 'results' / 'project'
MAX_COUNT = 7

def sage_cached(y, num_pulses, beta=2/3, max_iter=64, tol=1e-7):
    """Same SAGE updates, using linearity to cache correlations.

    Correlating a shifted pulse is a shifted pulse autocorrelation. This
    avoids repeating a full signal correlation for every component update.
    """
    n = len(y)
    m = n - sim.PULSE_LEN + 1
    energy = np.dot(sim.PULSE, sim.PULSE)
    ac = correlate(sim.PULSE, sim.PULSE, mode='full')
    def component(d, a):
        out = np.zeros(m)
        lo, hi = max(0, d-sim.PULSE_LEN+1), min(m, d+sim.PULSE_LEN)
        out[lo:hi] = a * ac[lo-d+sim.PULSE_LEN-1:hi-d+sim.PULSE_LEN-1]
        return out
    observed = correlate(y, sim.PULSE, mode='valid')
    fitted = np.zeros(m)
    delays = np.linspace(0, n-sim.PULSE_LEN, num_pulses).astype(int)
    amps = np.zeros(num_pulses)
    for _ in range(max_iter):
        old_d, old_a = delays.copy(), amps.copy()
        for k in range(num_pulses):
            old = component(delays[k], amps[k])
            corr = old + beta * (observed-fitted)
            delays[k] = np.argmax(corr)
            amps[k] = corr[delays[k]] / energy
            fitted += component(delays[k], amps[k]) - old
        if np.all(np.abs(delays-old_d)<tol) and np.all(np.abs(amps-old_a)<tol):
            break
    order = np.argsort(amps)[::-1]
    return delays[order], amps[order]

def estimate(y, method):
    """Fit every candidate independently; true count is never an input."""
    floor = max(float(np.mean(y*y))*1e-12, np.finfo(float).tiny)
    fits, rows = [], []
    for k in range(MAX_COUNT+1):
        if k == 0:
            d, a = np.array([], dtype=int), np.array([])
        else:
            d, a = (sim.matched_filter_delays if method == 'MF' else sage_cached)(y, k)
        rss = sim.residual_energy(y, d, a)
        penalty = (2*k+1)*np.log(len(y))
        error = len(y)/2*np.log(max(rss/len(y), floor))
        rows.append(dict(k=k, rss=rss, penalty=float(penalty), error=float(error), mdl=float(penalty+error)))
        fits.append((d,a))
    best = int(np.argmin([r['mdl'] for r in rows]))
    return best, rows, fits[best]

def trial(args):
    bounds, index = args
    rows = []
    # Reset seed per SNR: same geometry and standardized noise across SNR.
    seed = 20261121 + bounds*10000 + index
    for snr in sim.SNR_LEVELS_DB:
        env = sim.simulate(int(snr), bounds, np.random.default_rng(seed))
        for r in range(sim.NUM_RX):
            start = int(env['direct_index'][r])
            y = env['signals'][start:,r]
            assert np.all(env['true_delays'][r] >= start)
            for method in ('MF','SAGE'):
                best, diagnostics, _ = estimate(y, method)
                rows.append(dict(bounds=bounds, trial=index, seed=seed, snr=int(snr), receiver=r,
                                 method=method, estimated=best, samples=len(y),
                                 scores=[d['mdl'] for d in diagnostics]))
    return rows

def validate():
    checks = 0
    for snr,bounds in [(10,1),(20,3),(28,4)]:
        env = sim.simulate(snr,bounds,np.random.default_rng(42+bounds))
        y = env['signals'][env['direct_index'][0]:,0]
        for k in (1,3,5,7):
            d,a = sim.sage_delays(y,k)
            dc,ac = sage_cached(y,k)
            np.testing.assert_array_equal(d,dc)
            np.testing.assert_allclose(a,ac,rtol=1e-9,atol=1e-10)
            checks += 1
        for method in ('MF','SAGE'):
            best, rows, (d,a) = estimate(y,method)
            fit = sum((v*sim.shifted_pulse(int(t),len(y)) for t,v in zip(d,a)),np.zeros(len(y)))
            np.testing.assert_allclose(np.sum((y-fit)**2),rows[best]['rss'],rtol=1e-12)
    # Exact reconstruction checks sample-start convention and subtraction.
    y = sim.shifted_pulse(50,2000)+sim.shifted_pulse(800,2000)
    assert sim.residual_energy(y,np.array([50,800]),np.ones(2)) == 0
    for method in ('MF','SAGE'):
        assert estimate(y,method)[0] == 2
    rng = np.random.default_rng(19)
    y = (sim.shifted_pulse(500,2500) + .7*sim.shifted_pulse(530,2500)
         + .5*sim.shifted_pulse(1400,2500) + rng.normal(0,.1,2500))
    for k in range(1,8):
        d,a=sim.sage_delays(y,k); dc,ac=sage_cached(y,k)
        np.testing.assert_array_equal(d,dc)
        np.testing.assert_allclose(a,ac,rtol=1e-9,atol=1e-10)
    return dict(cached_sage_comparisons=checks, overlap_comparisons=7,
                residual_checks=6, noiseless_count_checks=2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trials',type=int,default=20)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--validate',action='store_true')
    args = p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.validate:
        result=validate(); print(result)
        (OUT/'validation.json').write_text(json.dumps(result,indent=2)); return
    if args.trials < 2:
        p.error('--trials must be at least 2 to estimate variation across environments')
    settings=dict(source='Michael Saleev email attachment, November 11, 2026',
                  source_sha256=hashlib.sha256(Path(sim.__file__).read_bytes()).hexdigest(),
                  trials_per_wall_count=args.trials, wall_counts=[1,2,3,4],
                  snr_db=sim.SNR_LEVELS_DB.tolist(), receivers_per_environment=sim.NUM_RX,
                  candidate_counts=list(range(MAX_COUNT+1)),
                  trial_seed_rule='20261121 + wall_count*10000 + trial_index',
                  score='(2*k+1)*ln(N) + N/2*ln(max(RSS/N, mean(y*y)*1e-12, float_tiny))',
                  snr_definition='sum(PULSE**2) / noise_variance',
                  crop='signals[direct_index[receiver]:, receiver]',
                  sage_beta=2/3, sage_max_iter=64, sage_tol=1e-7,
                  aggregation='Mean receiver accuracy within environment, then mean across environments; no voting',
                  coupling='Same geometry and standardized noise across SNR; paired methods',
                  signal_length=sim.SIGNAL_LEN, pulse_length=sim.PULSE_LEN)
    (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,result in enumerate(pool.map(trial,[(b,t) for b in range(1,5) for t in range(args.trials)])):
            rows.extend(result)
            print(f'{i+1}/{4*args.trials} environments completed',flush=True)
    (OUT/'trials.json').write_text(json.dumps(rows))
    summary=[]
    for b in range(1,5):
        for snr in sim.SNR_LEVELS_DB:
            for method in ('MF','SAGE'):
                selected=[r for r in rows if r['bounds']==b and r['snr']==snr and r['method']==method]
                est=np.array([r['estimated'] for r in selected])
                per_env=[np.mean([r['estimated']==b for r in selected if r['trial']==t]) for t in range(args.trials)]
                summary.append(dict(bounds=b,snr=int(snr),method=method,accuracy=float(np.mean(est==b)),
                                    under=float(np.mean(est<b)),over=float(np.mean(est>b)),
                                    mean_count=float(est.mean()),environments=args.trials,receivers=len(selected),
                                    se=float(np.std(per_env,ddof=1)/np.sqrt(args.trials))))
    with (OUT/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=summary[0],lineterminator='\n'); w.writeheader();w.writerows(summary)

if __name__ == '__main__':
    main()
