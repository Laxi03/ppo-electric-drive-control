"""Rebuild retrospective TDK tables/figures from the frozen 600-case experiment.

No training and no selection of successful-only cases. Run from project root:
python -m rllib_control.tdk_results
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .validate import quality

LABELS={'baseline':'PI/FOC','ppo':'PPO','icm_reduced':'ICM constant','icm_decay':'ICM decay'}
ORDER=['baseline','ppo','icm_reduced','icm_decay']
GATE=dict(max_ripple=1.,max_speed_pp=5.,max_offset=10.,max_current_ripple_fraction=.05,max_mae=30.)


def load_cases(root):
    protocol=json.loads((root/'protocol.json').read_text())
    rows=[]
    for job in protocol['jobs']:
        for profile in protocol['profiles']:
            for scenario in protocol['scenarios']:
                for seed in protocol['test_seeds']:
                    name=f"{job['id']}__{profile}__{scenario}__test{seed}"
                    path=root/'cases'/(name+'.json')
                    row=dict(case=name,motor=job['motor'],variant=job['variant'],train_seed=job['train_seed'],
                             test_seed=seed,profile=profile,scenario=scenario)
                    if not path.exists():
                        rows.append(dict(row,status='missing',engineering_pass=False,failures=['missing']));continue
                    record=json.loads(path.read_text());m=record.get('metrics',{})
                    out=Path(record.get('output',''))
                    files=list(out.glob('*_ripple_by_plateau.csv')) if out.is_dir() else []
                    plateaus=[]
                    if len(files)==1:
                        with files[0].open() as stream:
                            for r in csv.DictReader(stream):
                                plateaus.append({k:float(v) for k,v in r.items()})
                    # Every measured plateau must contain finite acceptance values.
                    required=('speed_ripple_RMS','speed_peak_to_peak','steady_offset_rad_s',
                              'i_or_id_ripple_RMS','iq_ripple_RMS')
                    finite=bool(plateaus) and all(math.isfinite(r.get(k,float('nan')))
                                                for r in plateaus for k in required)
                    assessment=quality(dict(plateaus=plateaus,metrics=m,current_limit=6. if job['motor']=='dc' else 3.),
                                       SimpleNamespace(**GATE)) if finite and m else dict(passed=False,values={},limits={})
                    vals=assessment.get('values',{});limits=assessment.get('limits',{})
                    failures=[k for k,v in vals.items() if not math.isfinite(v) or v>limits[k]]
                    if not finite:failures.append('missing_or_nonfinite_plateau')
                    if not m.get('completed_episode') or m.get('safety_violation') or m.get('time_above_current_scale_s',float('inf'))>0:
                        failures.append('completion_or_current')
                    row.update(status=record['status'],output=str(out),engineering_pass=bool(assessment['passed']),
                               failures=failures,quality_values=vals,metrics=m,plateaus=plateaus,
                               case_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    rows.append(row)
    return protocol,rows


def grouped(rows,metric,scenario='nominal'):
    result={}
    for motor in ['dc','pmsm']:
        for profile in ['steps','random']:
            for variant in ORDER:
                selected=[r for r in rows if (r['motor'],r['profile'],r['variant'],r['scenario'])==(motor,profile,variant,scenario)]
                groups=defaultdict(list)
                for r in selected:
                    value=r.get('metrics',{}).get(metric)
                    if r['status']=='complete' and isinstance(value,(int,float)) and math.isfinite(value):
                        groups[r['train_seed']].append(value)
                means=[float(np.mean(v)) for v in groups.values() if len(v)==3]
                result[motor,profile,variant]=dict(mean=float(np.mean(means)) if means else float('nan'),
                    sd=float(np.std(means,ddof=1)) if len(means)>1 else None,seeds=len(means))
    return result


def tex_table(path,header,rows,alignment):
    text=['\\begin{tabular}{'+alignment+'}','\\toprule',' & '.join(header)+r' \\',r'\midrule']
    text+=[' & '.join(str(x) for x in row)+r' \\' for row in rows]
    text += [r'\bottomrule',r'\end{tabular}']
    path.write_text('\n'.join(text)+'\n')


def build(root,output):
    protocol,rows=load_cases(root)
    output.mkdir(parents=True,exist_ok=True)
    tables=output/'tables';figures=output/'figures'
    tables.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    failures=Counter(f for r in rows for f in r['failures'])
    groups=[]
    for motor in ['dc','pmsm']:
        for variant in ORDER:
            group=[r for r in rows if r['motor']==motor and r['variant']==variant]
            groups.append(dict(motor=motor,variant=variant,total=len(group),
                passed=sum(r['engineering_pass'] for r in group),
                failed=len(group)-sum(r['engineering_pass'] for r in group),
                failures=dict(Counter(f for r in group for f in r['failures']))))
    audit=dict(protocol=str(root/'protocol.json'),status_counts=dict(Counter(r['status'] for r in rows)),
        retrospective=True,criteria=GATE,criterion_scope='Per-case plateau, current-scale and completion checks; no per-case global MAE gate',
        groups=groups,failures=dict(failures),cases=rows)
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    fields=['case','motor','variant','train_seed','test_seed','profile','scenario','status','engineering_pass',
            'MAE_speed_rad_s','RMSE_speed_rad_s','speed_ripple','speed_peak_to_peak','absolute_offset','current_ripple_fraction','failures']
    with (output/'acceptance_cases.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:
            flat={k:r.get(k) for k in fields}
            flat.update({k:r.get('metrics',{}).get(k) for k in ['MAE_speed_rad_s','RMSE_speed_rad_s']})
            flat.update(r.get('quality_values',{}));flat['failures']=';'.join(r['failures']);writer.writerow(flat)
    mae=grouped(rows,'MAE_speed_rad_s')
    def estimate(v):
        return f"{v['mean']:.2f}" if v['sd'] is None else f"${v['mean']:.2f} \\pm {v['sd']:.2f}$"
    tex_table(tables/'nominal_mae.tex',['Motor','Profile','PI/FOC','PPO','ICM fixed','ICM decay'],
        [[motor.upper(),profile,*[estimate(mae[motor,profile,v]) for v in ORDER]]
         for motor in ['dc','pmsm'] for profile in ['steps','random']],'llrrrr')
    tex_table(tables/'engineering_gate.tex',['Motor','Controller','Passed','Cases','Rate'],
        [[g['motor'].upper(),LABELS[g['variant']],g['passed'],g['total'],f"{100*g['passed']/g['total']:.1f}\\%"] for g in groups],'llrrr')
    # Per-profile means with training-seed SD, not 9 independent replicates.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.grid':True,'grid.alpha':.2,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    colors=['#555555','#0072B2','#D55E00','#009E73']
    fig,axes=plt.subplots(1,2,figsize=(6.5,4.2),layout='constrained')
    for ax,motor in zip(axes,['dc','pmsm']):
        for i,v in enumerate(ORDER):
            values=[mae[motor,p,v]['mean'] for p in ['steps','random']]
            sd=[mae[motor,p,v]['sd'] or 0. for p in ['steps','random']]
            ax.bar(np.arange(2)+(i-1.5)*.18,values,width=.18,color=colors[i],yerr=sd,capsize=2,label=LABELS[v])
        ax.set_xticks([0,1],['Steps','Random']);ax.set_title(motor.upper());ax.set_ylabel('Speed MAE [rad/s]')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.savefig(figures/'nominal_mae.pdf');fig.savefig(figures/'nominal_mae.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(6.5,4.5),layout='constrained')
    for ax,motor in zip(axes,['dc','pmsm']):
        g=[x for x in groups if x['motor']==motor]
        ax.barh(range(4),[100*x['passed']/x['total'] for x in g],color=colors)
        ax.set_yticks(range(4),[LABELS[x['variant']] for x in g]);ax.invert_yaxis()
        ax.set(xlim=(0,110),xlabel='Cases passing plateau gate [%]',title=motor.upper())
        for i,x in enumerate(g):ax.text(100*x['passed']/x['total']+1,i,f"{x['passed']}/{x['total']}",va='center',fontsize=10)
    fig.savefig(figures/'engineering_gate.pdf');fig.savefig(figures/'engineering_gate.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(6.5,4.5),layout='constrained')
    scenarios=protocol['scenarios']
    for ax,motor in zip(axes,['dc','pmsm']):
        for v,color in zip(ORDER,colors):
            points=[]
            for scenario in scenarios:
                g=grouped(rows,'MAE_speed_rad_s',scenario)
                points.append(np.mean([g[motor,p,v]['mean'] for p in ['steps','random']]))
            ax.plot(range(len(scenarios)),points,'o-',lw=1.4,color=color,label=LABELS[v])
        ax.set_xticks(range(len(scenarios)),['Nominal','Load','Params','R jump','Noise'],rotation=45,ha='right')
        ax.set(title=motor.upper(),ylabel='MAE averaged over profiles [rad/s]')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    fig.savefig(figures/'robustness_mae.pdf');fig.savefig(figures/'robustness_mae.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(6.5,7.1),sharex=True,layout='constrained')
    for ax,seed in zip(axes,[123,234,345]):
        r=next(x for x in rows if x['motor']=='pmsm' and x['variant']=='icm_decay' and x['train_seed']==seed
               and x['profile']=='random' and x['scenario']=='nominal' and x['test_seed']==901)
        ts=list(Path(r['output']).glob('*_timeseries.csv'))
        data=np.genfromtxt(ts[0],delimiter=',',names=True)
        ax.plot(data['time_s'],data['reference_rad_s'],'--',color='black',label='Reference',lw=1.)
        ax.plot(data['time_s'],data['omega_rad_s'],color='#0072B2',label='ICM decay',lw=1.)
        ax.set(title=f"Training seed {seed}; MAE {r['metrics']['MAE_speed_rad_s']:.2f} rad/s",ylabel='Speed [rad/s]')
    axes[0].legend(ncol=2);axes[-1].set_xlabel('Time [s]')
    fig.savefig(figures/'pmsm_seed_variability.pdf');fig.savefig(figures/'pmsm_seed_variability.png',dpi=180);plt.close(fig)
    lines=['# Eddigi eredmények auditja','',f"Tervezett esetek: {len(rows)}. Státuszok: {audit['status_counts']}.",
        '', 'Az alábbi arányok utólag alkalmazott plató- és határellenőrzések. Nem előre regisztrált sikerarányok.',
        'Az átlagos MAE 30 rad/s korlátja itt nincs minden egyes tesztesetre rávetítve; az a többprofilos validáció átlaga.',
        '', '| Motor | Változat | Megfelelő | Összes | Sikertelenség okai |','|---|---|---:|---:|---|']
    for g in groups:lines.append(f"|{g['motor']}|{g['variant']}|{g['passed']}|{g['total']}|{g['failures']}|")
    lines+=['','A 18 régi modell 1000 iterációs. Az új 500 iterációs PMSM-pilot más beállítású és egyseed-es.',
        'Az architektúra- és jutalomvizsgálat korábbi 1 iterációs smoke tesztje nem használható teljesítménybizonyítékként.',
        'A következő kísérletekben kizárólag tiszta külső PPO/ICM szerepel; az imént megszakított hibrid futások kizárva.',
        'Új modellek végső vizsgálatához 1901/1902/1903 tesztseedek javasoltak, mert 901/902/903 már hibadiagnózisban is szerepelt.']
    (output/'AUDIT_HU.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(output),status_counts=audit['status_counts'],groups=groups),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root',type=Path,default=Path('outputs/rllib/research/tdk_v1/main'))
    p.add_argument('--output',type=Path,default=Path('outputs/tdk_results_20261016'))
    args=p.parse_args();build(args.experiment_root,args.output)


if __name__=='__main__':main()
