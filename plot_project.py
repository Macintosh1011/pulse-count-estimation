"""Generate figures from project_mdl.py outputs."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from project_mdl import sim, estimate, OUT

COLORS={'MF':'#2563eb','SAGE':'#c05621'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    env=sim.simulate(28,3,np.random.default_rng(20261121))
    start=int(env['direct_index'][0]); y=env['signals'][start:,0]
    t=1e3*sim.T_OBS[start:]
    fig,axes=plt.subplots(1,2,figsize=(10,3.1),layout='constrained')
    axes[0].plot(t,y,color='#cbd5e1',lw=.55,label='Received')
    clean=sum((sim.shifted_pulse(int(d),len(y)) for d in env['true_delays'][0]-start),np.zeros(len(y)))
    axes[0].plot(t,clean,color='#334155',lw=1.1,label='True signal')
    example=[]
    for method in ('MF','SAGE'):
        best, rows, (d,a)=estimate(y,method)
        fit=sum((v*sim.shifted_pulse(int(x),len(y)) for x,v in zip(d,a)),np.zeros(len(y)))
        axes[0].plot(t,fit,color=COLORS[method],ls='--' if method=='SAGE' else '-',lw=1,label=method+' fit')
        scores=np.array([r['mdl'] for r in rows]); scores-=scores.min()
        axes[1].plot(range(8),scores,'o-',ms=3,color=COLORS[method],label=f'{method}: selected {best}')
        example.append(dict(method=method,estimated=best,diagnostics=rows,delays=(d+start).tolist(),amplitudes=a.tolist()))
    axes[0].set(xlabel='Time (ms)',ylabel='Amplitude',title='Three walls, receiver 1, SNR 28 dB')
    axes[1].set(xlabel='Assumed pulse count',ylabel='MDL minus its minimum',title='Count selected without true locations',xticks=range(8))
    for ax in axes: ax.grid(alpha=.15); ax.legend(fontsize=8)
    fig.savefig(OUT/'example.png',dpi=200);plt.close(fig)
    (OUT/'example.json').write_text(json.dumps(dict(seed=20261121,snr=28,true_count=3,start=start,true_delays=env['true_delays'][0].tolist(),fits=example),indent=2))
    if not (OUT/'summary.csv').exists(): return
    rows=list(csv.DictReader((OUT/'summary.csv').open()))
    fig,axes=plt.subplots(2,2,figsize=(9,5.5),sharex=True,sharey=True,layout='constrained')
    for b,ax in enumerate(axes.flat,1):
        for method in ('MF','SAGE'):
            subset=[r for r in rows if int(r['bounds'])==b and r['method']==method]
            ax.plot([int(r['snr']) for r in subset],[100*float(r['accuracy']) for r in subset],color=COLORS[method],label=method,marker='o',ms=2.5)
        ax.set(title=f'{b} wall'+('s' if b>1 else ''),ylim=(-3,103),xticks=[6,10,14,18,22,26,28])
        ax.grid(alpha=.18)
    axes[0,0].legend()
    fig.supxlabel('SNR (dB; supplied pulse-energy definition)')
    fig.supylabel('Correct pulse count (%)')
    fig.savefig(OUT/'accuracy.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3),sharey=True,layout='constrained')
    for method,ax in zip(('MF','SAGE'),axes):
        for key,label,color in [('under','Too few','#d97706'),('accuracy','Correct','#15803d'),('over','Too many','#7c3aed')]:
            values=[100*np.mean([float(r[key]) for r in rows if int(r['snr'])==snr and r['method']==method]) for snr in sim.SNR_LEVELS_DB]
            ax.plot(sim.SNR_LEVELS_DB,values,label=label,color=color)
        ax.set(title=method,xlabel='SNR (dB)',ylim=(-3,103));ax.grid(alpha=.18)
    axes[0].set_ylabel('Receiver outcomes (%)');axes[1].legend()
    fig.savefig(OUT/'errors.png',dpi=200);plt.close(fig)

if __name__=='__main__': main()
