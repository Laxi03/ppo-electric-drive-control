"""Export thesis figures from saved, paired mixed-profile measurements.

No model loading, simulation, training or controller tuning occurs here.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path

from .workspace import workspace_root

PROJECT = workspace_root()
WORKSPACE = PROJECT.parent
LABELS = {'ppo': 'PPO', 'icm': 'PPO + ICM', 'pi': 'PI', 'foc': 'FOC'}
COLORS = {'ppo': '#0072B2', 'icm': '#D55E00', 'pi': '#009E73', 'foc': '#009E73'}
METRICS = [('MAE_speed_rad_s', 'MAE [rad/s]'),
           ('RMSE_speed_rad_s', 'RMSE [rad/s]'),
           ('RMS_current_A', 'Current RMS [A]'),
           ('max_abs_current_A', 'Peak current [A]')]


def read_bytes(path):
    """Bound the read and reject incomplete exports rather than plotting them."""
    with path.open('rb') as stream:
        data = stream.read(path.stat().st_size)
    if len(data) != path.stat().st_size or not data:
        raise ValueError(f'Empty or incomplete measurement file: {path}')
    return data


def load_case(folder, motor):
    import numpy as np
    meta = json.loads(read_bytes(folder/'evaluation_manifest.json'))
    if meta.get('motor') != motor or meta.get('profile') != 'mixed':
        raise ValueError(f'Expected {motor} mixed measurement: {folder}')
    controllers = ['ppo', 'icm', 'pi' if motor == 'dc' else 'foc']
    data, hashes = {}, {}
    for controller in controllers:
        name = f'{controller}_timeseries.csv'
        raw = read_bytes(folder/name)
        headers = next(csv.reader(io.StringIO(raw.decode())))
        required = ['time_s', 'reference_rad_s', 'omega_rad_s']
        if headers[:3] != required or len(headers) < 10:
            raise ValueError(f'Unexpected trace columns: {folder/name}')
        trace = np.loadtxt(io.StringIO(raw.decode()), delimiter=',', skiprows=1)
        if trace.ndim != 2 or trace.shape[0] < 10 or not np.isfinite(trace).all():
            raise ValueError(f'Invalid trace: {folder/name}')
        if not (np.diff(trace[:, 0]) > 0).all() or abs(trace[-1, 0]-8.) > .01:
            raise ValueError(f'Expected increasing time and eight seconds: {folder/name}')
        if data and not np.array_equal(next(iter(data.values()))[:, :2], trace[:, :2]):
            raise ValueError(f'Controllers do not share time/reference samples: {folder}')
        data[controller] = trace
        hashes[name] = hashlib.sha256(raw).hexdigest()
    metrics = json.loads(read_bytes(folder/'metrics.json'))
    for controller in controllers:
        if not metrics[controller]['completed_episode']:
            raise ValueError(f'Incomplete episode must be reviewed: {folder}/{controller}')
    return data, metrics, meta, hashes


def export_speed_current(data, motor, case, path):
    """Export all saved samples as visible vector markers for thesis-size panels."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator
    current_column = 6 if motor == 'dc' else 3
    lo = min(d[:, 2].min() for d in data.values())-25
    hi = max(d[:, 2].max() for d in data.values())+25
    imin = min(d[:, current_column].min() for d in data.values())-.15
    imax = max(d[:, current_column].max() for d in data.values())+.15
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.7), sharex=True,
                             sharey=True, layout='constrained')
    for index, (controller, d) in enumerate(data.items()):
        sc = axes[index].scatter(d[:, 2], d[:, current_column], c=d[:, 0],
            cmap='viridis', norm=Normalize(0, 8), s=4.5, alpha=.85,
            linewidths=0, edgecolors='none', rasterized=False)
        axes[index].set(xlim=(lo, hi), ylim=(imin, imax),
            xlabel='Speed [rad/s]', title=f'({chr(97+index)}) {LABELS[controller]}')
        axes[index].xaxis.set_major_locator(MaxNLocator(nbins=4))
        axes[index].tick_params(labelsize=9)
        axes[index].grid(alpha=.16)
        axes[index].set_axisbelow(True)
    axes[0].set_ylabel('Armature current [A]' if motor == 'dc' else r'$\sqrt{i_d^2+i_q^2}$ [A]')
    fig.colorbar(sc, ax=axes, label='Time [s]', ticks=[0, 2, 4, 6, 8], shrink=.88)
    if case != 'nominal':
        inductance = r'$L=0.8L_{nom}$' if motor == 'dc' else r'$L_d=0.8L_{d,nom}$, $L_q=0.8L_{q,nom}$'
        fig.suptitle(r'Altered plant for the complete episode: $R=1.3R_{nom}$, '+inductance, fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.pdf'), facecolor='white')
    fig.savefig(path.with_suffix('.png'), dpi=400, facecolor='white')
    plt.close(fig)


def export_case(folder, motor, case, dest):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy as np
    data, metrics, meta, hashes = load_case(folder, motor)
    main, appendix = dest/'main', dest/'appendix'
    main.mkdir(parents=True); appendix.mkdir()
    title = f'{motor.upper()} | mixed reference | '+('nominal' if case == 'nominal' else 'parameter mismatch')
    lo = min(d[:, 2].min() for d in data.values())-25
    hi = max(d[:, 2].max() for d in data.values())+25

    def save(fig, path):
        fig.savefig(path.with_suffix('.png'), dpi=250, facecolor='white')
        fig.savefig(path.with_suffix('.pdf'), facecolor='white')
        plt.close(fig)

    def tracking(ax, controller, d, window=None):
        ax.plot(d[:, 0], d[:, 1], color='#252525', ls='--', lw=1.15, label='Reference', zorder=3)
        ax.plot(d[:, 0], d[:, 2], color=COLORS[controller], lw=1.35, label=LABELS[controller])
        ax.set(xlim=window or (0, 8), ylabel='Speed [rad/s]')
        ax.legend(loc='best', ncol=2, fontsize=8)

    fig, axes = plt.subplots(3, 1, figsize=(8.1, 7), sharex=True, sharey=True, layout='constrained')
    for index, (controller, d) in enumerate(data.items()):
        tracking(axes[index], controller, d)
        axes[index].set(ylim=(lo, hi), title=f'({chr(97+index)}) {LABELS[controller]}')
        single, ax = plt.subplots(figsize=(8.1, 2.7), layout='constrained')
        tracking(ax, controller, d)
        ax.set(ylim=(lo, hi), xlabel='Time [s]', title=f'{title}: {LABELS[controller]}')
        save(single, main/f'tracking_{controller}')
    axes[-1].set_xlabel('Time [s]'); fig.suptitle(title)
    save(fig, main/'tracking_three_panels')

    # Fixed window around the same first reversal; identical in both plant cases.
    window = (.99, 1.15) if motor == 'dc' else (.98, 1.55)
    selected = [d[(d[:, 0] >= window[0]) & (d[:, 0] <= window[1])] for d in data.values()]
    zlo = min(min(d[:, 1].min(), d[:, 2].min()) for d in selected)-15
    zhi = max(max(d[:, 1].max(), d[:, 2].max()) for d in selected)+15
    fig, axes = plt.subplots(3, 1, figsize=(8.1, 6.7), sharex=True, sharey=True, layout='constrained')
    for index, (controller, d) in enumerate(data.items()):
        tracking(axes[index], controller, d, window)
        axes[index].set(ylim=(zlo, zhi), title=f'({chr(97+index)}) {LABELS[controller]}')
    axes[-1].set_xlabel('Time [s]'); fig.suptitle(title+' | first reversal')
    save(fig, main/'tracking_reversal_zoom')

    current_column = 6 if motor == 'dc' else 3
    export_speed_current(data, motor, case, main/'speed_current_three_panels')

    table_rows = []
    for controller, d in data.items():
        # Response numbers are computed only for the first actual reversal.
        mask = (d[:, 0] >= 1.) & (d[:, 0] < 2.)
        target = float(np.median(d[mask, 1]))
        initial_target = float(np.median(d[d[:, 0] < 1., 1]))
        direction = np.sign(target-initial_target)
        overshoot = max(0., float(np.max(direction*(d[mask, 2]-target))))
        plateaus = list(csv.DictReader(io.StringIO(read_bytes(folder/f'{controller}_ripple_by_plateau.csv').decode())))
        row = {'controller': LABELS[controller], **{label: metrics[controller][key] for key, label in METRICS}}
        row.update({'Reversal overshoot [rad/s]': overshoot,
            'Max absolute late offset [rad/s]': max(abs(float(p['steady_offset_rad_s'])) for p in plateaus),
            'Mean late speed ripple RMS [rad/s]': float(np.mean([float(p['speed_ripple_RMS']) for p in plateaus]))})
        table_rows.append(row)
        fig, axes = plt.subplots(4, 1, figsize=(8.1, 7.5), sharex=True, layout='constrained')
        axes[0].plot(d[:, 0], d[:, 1], 'k--', lw=1, label='Reference')
        voltage = d[:, 8] if motor == 'dc' else d[:, 4]
        for ax, values, ylabel in zip(axes, [d[:, 2], d[:, 1]-d[:, 2], d[:, current_column], voltage],
                ['Speed [rad/s]', 'Error [rad/s]', 'Current [A]' if motor == 'dc' else r'$|i_{dq}|$ [A]',
                 'Voltage [V]' if motor == 'dc' else r'$|v_{dq}|$ [V]']):
            ax.plot(d[:, 0], values, color=COLORS[controller], lw=1, label=LABELS[controller])
            ax.set(xlim=(0, 8), ylabel=ylabel)
        axes[0].legend(ncol=2); axes[-1].set_xlabel('Time [s]')
        fig.suptitle(title+f' | {LABELS[controller]}')
        save(fig, appendix/f'signals_{controller}')
        if motor == 'pmsm':
            fig, axes = plt.subplots(4, 1, figsize=(8.1, 7.5), sharex=True, layout='constrained')
            for ax, col, label in zip(axes, [6, 7, 8, 9], [r'$i_d$ [A]', r'$i_q$ [A]', r'$v_d$ [V]', r'$v_q$ [V]']):
                ax.plot(d[:, 0], d[:, col], color=COLORS[controller], lw=1)
                ax.set(xlim=(0, 8), ylabel=label)
            axes[-1].set_xlabel('Time [s]'); fig.suptitle(title+f' | {LABELS[controller]} | dq')
            save(fig, appendix/f'dq_signals_{controller}')
    with (main/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
        writer.writeheader(); writer.writerows(table_rows)
    lines = [r'\begin{tabular}{lrrrr}\hline',
             r'Controller & MAE [rad/s] & RMSE [rad/s] & $I_{\rm RMS}$ [A] & $I_{\rm peak}$ [A] \\ \hline']
    for row in table_rows:
        lines.append(row['controller']+' & '+' & '.join(f'{row[label]:.3f}' for _, label in METRICS)+r' \\')
    lines.append(r'\hline\end{tabular}')
    (main/'metrics.tex').write_text('\n'.join(lines)+'\n')
    (dest/'source.json').write_text(json.dumps({'directory': str(folder), 'measurement': meta,
        'sha256': hashes, 'reversal_zoom_s': window, 'late_window_fraction': .3,
        'plateau_scope': 'constant-reference segments only; no ramp or sine ripple claims'}, indent=2))
    return {'motor': motor, 'case': case, 'source': str(folder), 'output': str(dest)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Four-case batch containing evidence/<motor>/<case>')
    parser.add_argument('--output', type=Path, default=PROJECT/'TDK_Figures')
    parser.add_argument('--motor', choices=['dc', 'pmsm', 'all'], default='all')
    opts = parser.parse_args(argv)
    source = opts.input
    motor_sources = {}
    if source is None:
        pointer = WORKSPACE/'output/tdk_chapter5/latest.json'
        saved = json.loads(read_bytes(pointer))
        source = Path(saved['directory'])
        motor_sources = saved.get('motor_sources', {})
    source = source.resolve()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .18})
    output = opts.output.resolve()/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    exported = []
    specs = [('dc', 'nominal', '01_DC_nominal'), ('pmsm', 'nominal', '02_PMSM_nominal'),
             ('dc', 'parameter_mismatch', '03_DC_perturbed'), ('pmsm', 'parameter_mismatch', '04_PMSM_perturbed')]
    for motor, case, name in specs:
        if opts.motor not in ('all', motor): continue
        print(f'Exporting {motor}: {case}', flush=True)
        motor_source = Path(motor_sources.get(motor, source))
        exported.append(export_case(motor_source/'evidence'/motor/case, motor, case, output/name))
    (output/'manifest.json').write_text(json.dumps({'profile': 'mixed', 'cases': exported}, indent=2))
    (output/'README.txt').write_text('TDK FIGURES - MIXED PROFILE\n\nmain/: separate tracking, three-panel tracking, common-window reversal zoom, speed-current scatter, CSV/LaTeX metrics.\nappendix/: tracking/error/current/voltage for each controller; PMSM dq signals.\nPNG for preview; PDF for sharp Overleaf figures. Same data, no smoothing or new simulation.\nPlateau metrics cover constant-reference segments only. Sources and hashes: source.json.\nPerturbation: joint R +30%, L or Ld/Lq -20% from reset; J and controller settings unchanged.\n')
    (opts.output.resolve()/'latest.json').write_text(json.dumps({'directory': str(output)}, indent=2))
    print(f'\nFigures saved to: {output}', flush=True)


if __name__ == '__main__':
    main()
