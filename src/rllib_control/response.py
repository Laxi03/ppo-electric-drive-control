"""Reference-step timing measured from raw controller-rate samples."""
import numpy as np


def step_response_metrics(data,settling_band=5.):
    t,reference,speed=data[:,:3].T
    edges=np.r_[0,np.flatnonzero(abs(np.diff(reference))>1e-8)+1,len(t)]
    rows=[]
    for start,end in zip(edges[:-1],edges[1:]):
        if start==0:continue
        initial=float(speed[start-1]);target=float(reference[start]);jump=target-initial
        if abs(jump)<1e-6:continue
        ts=np.r_[t[start-1],t[start:end]]
        progress=np.r_[0.,(speed[start:end]-initial)/jump]
        def crossing(level):
            indices=np.flatnonzero(progress>=level)
            if not len(indices):return float('nan')
            i=int(indices[0]);prior=i-1
            return float(ts[prior]+(level-progress[prior])/(progress[i]-progress[prior])*(ts[i]-ts[prior]))
        a,b=crossing(.1),crossing(.9)
        errors=reference[start:end]-speed[start:end]
        outside=np.flatnonzero(abs(errors)>settling_band)
        settled=(t[start+outside[-1]+1]-t[start-1]) if len(outside) and outside[-1]+1<end-start else (0. if not len(outside) else float('nan'))
        rows.append(dict(start_s=float(t[start-1]),end_s=float(t[end-1]),reference_rad_s=target,
            initial_speed_rad_s=initial,step_rad_s=jump,rise_time_10_90_s=b-a,
            time_to_90_s=b-t[start-1],overshoot_percent=max(0.,float(progress.max()-1))*100,
            settling_time_5rad_s=float(settled),peak_current_A=float(abs(data[start:end,3]).max())))
    return rows
