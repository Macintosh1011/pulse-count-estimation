"""Reproduce MDL component checks and one-factor-at-a-time experiments."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from demo import estimate_count

OUT = Path(__file__).parent / 'results'
TRIALS = 100

def signal(n=350, centers=(70,175,280), amplitudes=(1,.7,.5), width=15):
    t = np.arange(n)
    return sum(a*np.exp(-.5*((t-c)/width)**2) for c,a in zip(centers,amplitudes))

def run(n=350, centers=(70,175,280), amplitudes=(1,.7,.5), width=15, sigma=0, seed=42):
    y=signal(n,centers,amplitudes,width)+np.random.default_rng(seed).normal(0,sigma,n)
    return y, estimate_count(y,width,7,return_diagnostics=True)

def save_csv(name, rows):
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True)
    components=[]
    for sigma in [0,.03]:
        y,(count,scores,recon,rows)=run(sigma=sigma)
        for row in rows:
            components.append(dict(noise_std=sigma,**row))
        # Independently recompute RSS using a projection, not the reconstruction.
        for row in rows:
            t=np.arange(len(y)); A=np.exp(-.5*((t[:,None]-np.array(row['centers']))/15)**2)
            Q,_=np.linalg.qr(A,mode='reduced')
            residual=y-Q@(Q.T@y)
            np.testing.assert_allclose(residual@residual,row['rss'],rtol=1e-9,atol=1e-24)
            expected=(2*row['k']+1)*np.log(len(y))+len(y)/2*np.log(max(residual@residual/len(y),row['floor']))
            np.testing.assert_allclose(row['mdl'],expected,atol=1e-9)
        assert np.all(np.diff([r['rss'] for r in rows]) <= 1e-12)
    save_csv('components.csv',components)
    assert np.allclose(np.diff([r['mdl'] for r in components[2:7]]),2*np.log(350))

    # One changed setting per family; same seed yields paired noise across settings.
    cases=[]
    for sigma in [0,.03]:
        for k in [1,2,3,4,5]:
            cases.append(('count',k,dict(n=700,centers=[70+105*i for i in range(k)],amplitudes=[1]*k,sigma=sigma)))
        for d in [15,30,45,60,90,105]:
            cases.append(('spacing',d,dict(centers=[175-d,175,175+d],sigma=sigma)))
        for n in [350,700,1400]:
            cases.append(('length',n,dict(n=n,sigma=sigma)))
            cases.append(('length_overlap',n,dict(n=n,centers=[115,175,235],sigma=sigma)))
        for a in [.1,.25,.5,1]:
            cases.append(('weak_amplitude',a,dict(amplitudes=[1,.7,a],sigma=sigma)))
        for w in [5,10,15,20,30]:
            cases.append(('width',w,dict(width=w,sigma=sigma)))
    for sigma in [.01,.1,.2,.4]:
        cases.append(('noise',sigma,dict(sigma=sigma)))
    raw=[];summary=[]
    for family,value,kwargs in cases:
        sigma=kwargs.get('sigma',0);truth=len(kwargs.get('centers',[70,175,280]));counts=[];half=[]
        for seed in (range(TRIALS) if sigma else [42]):
            _,(count,score,_,rows)=run(seed=seed,**kwargs)
            half_count=int(np.argmin([r['error_term']+.5*r['penalty'] for r in rows]))+1
            counts.append(count);half.append(half_count)
            raw.append(dict(family=family,value=value,noise_std=sigma,seed=seed,true_count=truth,estimated=count,half_penalty_count=half_count))
        hist={str(i):counts.count(i) for i in sorted(set(counts))}
        summary.append(dict(family=family,value=value,noise_std=sigma,true_count=truth,trials=len(counts),correct=counts.count(truth),accuracy=counts.count(truth)/len(counts),mean_count=float(np.mean(counts)),histogram=hist,half_penalty_accuracy=half.count(truth)/len(half)))
        print(summary[-1],flush=True)
    save_csv('trials.csv',raw)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    # Numerical floor sensitivity on the same fits (no extra location optimization).
    floors=[]
    for label,kwargs in [('separated',{}),('overlap',dict(centers=[145,175,205]))]:
        for sigma in [0,.03]:
            y,(_,_,_,rows)=run(sigma=sigma,**kwargs)
            for relative in [1e-8,1e-10,1e-12,1e-14,1e-16]:
                score=[r['penalty']+len(y)/2*np.log(max(r['mse'],np.mean(y*y)*relative)) for r in rows]
                floors.append(dict(case=label,noise_std=sigma,relative_floor=relative,count=int(np.argmin(score))+1))
    save_csv('floor_sensitivity.csv',floors)
    # A diagnostic using known true locations isolates location-search error.
    y, (_, _, _, rows) = run(centers=[145,175,205])
    t = np.arange(350)
    A = np.exp(-.5*((t[:,None]-[145,175,205])/15)**2)
    oracle_rss = float(np.sum((y-A@np.linalg.lstsq(A,y,rcond=None)[0])**2))
    (OUT/'location_check.json').write_text(json.dumps(dict(
        true_centers=[145,175,205], greedy_centers=rows[2]['centers'],
        greedy_rss=rows[2]['rss'], true_center_rss=oracle_rss,
        note='Known locations are a diagnostic only, not supplied to the estimator.'),indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    for col,sigma in enumerate([0,.03]):
        rows=[r for r in components if r['noise_std']==sigma];k=np.arange(1,8)
        axs[0,col].semilogy(k,[r['mse'] for r in rows],'o-',label='Residual MSE')
        axs[0,col].axhline(rows[0]['floor'],ls='--',color='gray',label='Numerical floor')
        axs[0,col].set(title=f'Noise standard deviation: {sigma}',ylabel='Residual MSE');axs[0,col].legend()
        for field,label in [('penalty','Penalty change'),('error_term','Error-term change'),('mdl','Total score change')]:
            vals=np.array([r[field] for r in rows[2:]])
            axs[1,col].plot(k[2:],vals-vals[0],'o-',label=label)
        axs[1,col].set(xlabel='Candidate pulse count',ylabel='Change relative to k = 3');axs[1,col].legend(fontsize=8)
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.savefig(OUT/'components.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for ax,family,title in zip(axs.flat,['count','spacing','length','weak_amplitude','width','noise'],['True pulse count (N = 700)','Pulse spacing (samples)','Recording length (samples)','Third-pulse amplitude','Gaussian width (samples)','Noise standard deviation']):
        for sigma in ([0,.03] if family!='noise' else [None]):
            rows=[r for r in summary if r['family']==family and (sigma is None or r['noise_std']==sigma)]
            if family=='noise':
                rows += [r for r in summary if r['family']=='length' and r['value']==350]
            rows=sorted(rows,key=lambda r:r['noise_std'] if family=='noise' else r['value'])
            x=[r['noise_std'] if family=='noise' else r['value'] for r in rows]
            ax.plot(x,[100*r['accuracy'] for r in rows],'o-',label=f'Noise SD {sigma}' if sigma is not None else 'Separated pulses')
        ax.set(xlabel=title,ylabel='Correct count (%)',ylim=(-5,105));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/'sweeps.png',dpi=180);plt.close(fig)
    print('Component checks passed; results saved in results/.')

if __name__=='__main__':main()
