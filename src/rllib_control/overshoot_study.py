"""Explicit nominal step-overshoot calibration; preserves the previous baseline."""
import csv
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path

from . import results, runner


class TuningReference:
    """Fixed reference, defined before gain search for both motors."""
    def value(self, time_s):
        return 300. if time_s < 1.-1e-10 else -300.


def measure(motor, config, gains):
    import numpy as np
    from .current_control import CurrentReferenceEnv, CurrentBaseline
    env = CurrentReferenceEnv(dict(config, profile='mixed', episode_length=3.,
        seed=501, training_mode=False, domain_randomization=False, parameter_jump=False,
        reference_scale=1.))
    controller = CurrentBaseline(motor, gains)
    rows=[]
    try:
        env.reset(seed=501)
        env.base.reference=TuningReference()
        env.base.last_reference=300.
        obs=env._obs(env.base._observation())
        while True:
            obs, _, terminated, truncated, info=env.step(controller.action(obs,env))
            rows.append([info['time'],info['reference'],info['omega'],
                         info.get('current',info.get('current_magnitude'))])
            if terminated or truncated:break
    finally:
        env.close()
    d=np.asarray(rows)
    before=d[d[:,1]>0]
    after=d[d[:,1]<0]
    overshoot=max(0.,float(-300.-after[:,2].min()))/600.*100 if len(after) else float('nan')
    pre_error=abs(float(before[-1,2])-300.) if len(before) else float('inf')
    eligible=bool(not terminated and truncated and not info['safety_violation']
                  and pre_error<=1. and np.isfinite(d).all())
    return dict(gains=gains,overshoot_percent=overshoot,pre_step_error_rad_s=pre_error,
                completed=bool(not terminated and truncated),safety_violation=bool(info['safety_violation']),
                eligible=eligible),d


def tune(motor, folder):
    from .current_control import environment_config
    args,manifest,_=results.restored(motor,'ppo',results.PRESETS[motor]['ppo'])
    config=environment_config(args)
    original_path=runner.ROOT/results.PRESETS[motor]['baseline']
    original=json.loads(original_path.read_text())
    if 'path' in original:original_path=Path(original['path']);original=json.loads(original_path.read_text())
    search_path=folder/f'{motor}_search.json'
    records=json.loads(search_path.read_text()) if search_path.exists() else []
    cache={(round(r['gains']['kp'],12),round(r['gains']['ki'],12)):(r,None) for r in records}
    target=25.;tolerance=1.
    def test(kp,ki):
        key=(round(kp,12),round(ki,12))
        if key in cache:return cache[key]
        r,d=measure(motor,config,dict(kp=kp,ki=ki))
        records.append(r);cache[key]=(r,d)
        (folder/f'{motor}_search.json').write_text(json.dumps(records,indent=2))
        print(motor,'kp',kp,'ki',round(ki,7),'overshoot',round(r['overshoot_percent'],4),
              'eligible',r['eligible'],flush=True)
        return r,d
    selected=None
    # Keep Kp fixed first; only expand to a lower/higher Kp if no Ki bracket works.
    for kp in [original['gains']['kp']*scale for scale in [1.,.5,2.,.25,.125]]:
        values=[]
        for ki in [.03,.1,.3,1.,3.,10.]:
            r,d=test(kp,ki);values.append((ki,r,d))
            if r['eligible'] and abs(r['overshoot_percent']-target)<=tolerance:
                selected=(r,d);break
        if selected:break
        for (low,a,_),(high,b,_) in zip(values[:-1],values[1:]):
            if not a['eligible'] or not b['eligible']:continue
            if (a['overshoot_percent']-target)*(b['overshoot_percent']-target)>0:continue
            fa=a['overshoot_percent']-target
            for _ in range(9):
                middle=math.sqrt(low*high);r,d=test(kp,middle)
                if not r['eligible']:break
                fm=r['overshoot_percent']-target
                if abs(fm)<=tolerance:selected=(r,d);break
                if fa*fm<=0:high=middle
                else:low=middle;fa=fm
            if selected:break
        if selected:break
    if selected is None:
        raise RuntimeError(f'{motor}: no eligible gain pair met 25 +/- 1 percentage points; inspect search, do not claim success')
    r,d=selected
    if d is None:r,d=measure(motor,config,r['gains'])
    baseline=dict(motor=motor,environment=config,gains=r['gains'],
        tuning=dict(target_overshoot_percent=25.,tolerance_percentage_points=1.,
            reference=dict(initial_rad_s=300.,final_rad_s=-300.,event_s=1.,duration_s=3.),
            normalization='600 rad/s reference step amplitude',seed=501,measured=r,
            selection='First eligible pair meeting the predeclared target; original Kp tried first',
            previous_baseline=str(original_path),previous_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest()))
    (folder/f'{motor}_baseline.json').write_text(json.dumps(baseline,indent=2))
    import numpy as np
    np.savetxt(folder/f'{motor}_tuning_trace.csv',d,delimiter=',',
        header='time_s,reference_rad_s,speed_rad_s,current_A_or_dq_magnitude',comments='')
    return baseline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motor',choices=['dc','pmsm'],default='dc')
    parser.add_argument('--resume',type=Path)
    opts=parser.parse_args()
    folder=opts.resume or (Path(os.environ['TDK_TUNING_RESUME']) if os.environ.get('TDK_TUNING_RESUME') else runner.ROOT/'outputs/overshoot_design'/datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'protocol.json').write_text(json.dumps(dict(target_percent=25,tolerance_pp=1,
        tuning_reference='+300 to -300 rad/s at 1 s; 3 s nominal episode; seed 501',
        overshoot_normalization='600 rad/s reference change',
        eligibility='complete, finite, no safety violation, pre-step error <=1 rad/s',
        scope='outer speed gains only; shared inner regulator unchanged'),indent=2))
    all_results={}
    for motor in [opts.motor]:
        saved=folder/f'{motor}_baseline.json'
        all_results[motor]=json.loads(saved.read_text()) if saved.exists() else tune(motor,folder)
    (folder/'summary.json').write_text(json.dumps(all_results,indent=2))
    (folder.parent/'latest.json').write_text(json.dumps(dict(directory=str(folder)),indent=2))
    print('CALIBRATION:',folder,flush=True)


if __name__=='__main__':main()
