"""Controller-rate diagnostics. No smoothing or removal of inconvenient samples."""
import numpy as np


def command_metrics(time, requested, limited, applied, tolerance=1e-3):
    dt = np.diff(np.r_[0., time])
    result = {}
    for name, vector in (("requested",requested),("limited",limited),("applied",applied)):
        delta = np.diff(np.vstack((np.zeros((1,vector.shape[1])),vector)),axis=0)
        sizes = np.linalg.norm(delta,axis=1)
        result[name+"_voltage_total_variation_V"] = float(sizes.sum())
        result[name+"_command_change_rate_per_s"] = float(np.count_nonzero(sizes[1:]>tolerance)/max(time[-1]-time[0],1e-12))
        result[name+"_max_slew_V_per_s"] = float(np.max(sizes/dt))
        result[name+"_rms_slew_V_per_s"] = float(np.sqrt(np.sum((sizes/dt)**2*dt)/time[-1]))
    return result


def plateau_metrics(data, fraction=.3):
    """Steady means a predefined late window, NOT a claim of mathematical settling."""
    t, ref, speed = data[:,:3].T
    boundaries = np.r_[0,np.flatnonzero(np.abs(np.diff(ref))>1e-8)+1,len(t)]
    dt = np.diff(np.r_[0.,t])
    rows = []
    for start,end in zip(boundaries[:-1],boundaries[1:]):
        if end-start < 10:
            continue
        beginning = t[start]-dt[start]
        duration = t[end-1]-beginning
        mask = np.arange(start,end)
        ss = mask[t[mask] > beginning+(1-fraction)*duration]
        tr = mask[t[mask] <= beginning+(1-fraction)*duration]
        if len(ss)<3 or len(tr)<2:
            continue
        error = ref-speed
        # Absolute 5 rad/s band, maintained to the end of this plateau.
        outside = mask[np.abs(error[mask]) > 5.]
        settling_index = int(outside[-1])+1 if len(outside) else start
        settling = float(t[settling_index]-beginning) if settling_index < end else float("nan")
        row = dict(segment=len(rows)+1, start_s=float(beginning),end_s=float(t[end-1]),
                   reference_rad_s=float(ref[start]), steady_start_s=float(t[ss[0]]),
                   steady_samples=len(ss),steady_fraction=fraction,
                   settling_time_s=settling,settling_band_rad_s=5.,
                   transient_MAE_rad_s=float(np.average(abs(error[tr]),weights=dt[tr])),
                   transient_RMSE_rad_s=float(np.sqrt(np.average(error[tr]**2,weights=dt[tr]))),
                   steady_offset_rad_s=float(np.average(error[ss],weights=dt[ss])),
                   steady_RMSE_rad_s=float(np.sqrt(np.average(error[ss]**2,weights=dt[ss]))))
        for name,column in (("speed",2),("current_magnitude",3),("i_or_id",6),("iq",7),("u_or_vd",8),("vq",9)):
            values = data[ss,column]
            row[name+"_peak_to_peak"] = float(np.ptp(values))
            mean = np.average(values,weights=dt[ss])
            row[name+"_ripple_RMS"] = float(np.sqrt(np.average((values-mean)**2,weights=dt[ss])))
        local_time=t[ss]-t[ss[0]]
        trend=np.polyfit(local_time,speed[ss],1,w=np.sqrt(dt[ss]))
        row["steady_speed_slope_rad_s2"] = float(trend[0])
        # Report drift separately; the original undetrended gate remains intact.
        residual=speed[ss]-np.polyval(trend,local_time)
        row['speed_detrended_ripple_RMS']=float(np.sqrt(np.average(residual**2,weights=dt[ss])))
        rows.append(row)
    return rows
