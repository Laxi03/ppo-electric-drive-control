import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from rllib_control.validate import quality, choose
from rllib_control.showcase import selected_report
from rllib_control import runner


class ReadinessTests(unittest.TestCase):
    def test_step_rise_time_uses_measured_initial_speed(self):
        import numpy as np
        from rllib_control.response import step_response_metrics
        t=np.arange(0,2.001,.001);data=np.zeros((len(t),10));data[:,0]=t
        data[:,1]=np.where(t<1.,0.,100.)
        data[:,2]=np.where(t<1.,10.,10.+90.*np.clip((t-1.)/.5,0,1))
        row=step_response_metrics(data)[0]
        self.assertAlmostEqual(row['rise_time_10_90_s'],.4,places=6)
        self.assertEqual(row['initial_speed_rad_s'],10.)

    def test_ripple_offset_current_and_missing_plateaus(self):
        opts=SimpleNamespace(max_ripple=1.,max_speed_pp=5.,max_offset=10.,max_current_ripple_fraction=.05)
        plateau=dict(speed_ripple_RMS=.1,speed_peak_to_peak=.4,steady_offset_rad_s=1.,
                     i_or_id_ripple_RMS=.01,iq_ripple_RMS=.02)
        result=dict(plateaus=[plateau],current_limit=3.,metrics=dict(completed_episode=True,
                    safety_violation=False,time_above_current_scale_s=0.))
        self.assertTrue(quality(result,opts)['passed'])
        for key,value in [('speed_ripple_RMS',2.),('speed_peak_to_peak',6.),
                          ('steady_offset_rad_s',11.),('iq_ripple_RMS',.2)]:
            self.assertFalse(quality(dict(result,plateaus=[dict(plateau,**{key:value})]),opts)['passed'])
        self.assertFalse(quality(dict(result,plateaus=[]),opts)['passed'])
        self.assertFalse(quality(dict(result,plateaus=[plateau,dict(plateau,speed_ripple_RMS=float('nan'))]),opts)['passed'])
        self.assertFalse(quality(dict(result,plateaus=[plateau,dict(plateau,iq_ripple_RMS=float('inf'))]),opts)['passed'])

    def test_negative_step_and_unreached_response(self):
        import numpy as np
        from rllib_control.response import step_response_metrics
        t=np.arange(0,2.001,.001);data=np.zeros((len(t),10));data[:,0]=t
        data[:,1]=np.where(t<1.,100.,-100.)
        data[:,2]=np.where(t<1.,100.,100.-200.*np.clip((t-1.)/.5,0,1))
        row=step_response_metrics(data)[0]
        self.assertAlmostEqual(row['rise_time_10_90_s'],.4,places=6)
        self.assertEqual(row['overshoot_percent'],0.)
        data[:,2]=np.where(t<1.,100.,100.-100.*np.clip((t-1.)/.5,0,1))
        row=step_response_metrics(data)[0]
        self.assertTrue(np.isnan(row['rise_time_10_90_s']))
        self.assertTrue(np.isnan(row['settling_time_5rad_s']))

    def test_detrended_ripple_distinguishes_linear_drift(self):
        import numpy as np
        from rllib_control.ripple import plateau_metrics
        t=np.arange(1,1001)*.001
        data=np.zeros((len(t),10));data[:,0]=t;data[:,1]=100.;data[:,2]=100.+2*t
        row=plateau_metrics(data)[0]
        self.assertGreater(row['speed_ripple_RMS'],.1)
        self.assertAlmostEqual(row['steady_speed_slope_rad_s2'],2.)
        self.assertLess(row['speed_detrended_ripple_RMS'],1e-10)

    def test_latest_validation_per_controller_and_no_cherry_picking(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(runner,'ROOT',Path(directory)):
            root=Path(directory)/'outputs/rllib/pmsm/validation/run'
            checkpoint=str(Path(directory)/'RL_models/pmsm/ppo/run/iteration_0500')
            def save(stamp,controller,selected):
                folder=root/stamp; folder.mkdir(parents=True)
                report=dict(arguments=dict(motor='pmsm',controller=controller,run='run',
                    seeds=[501,502],profiles=['steps','random']),selected_checkpoint=selected,
                    records=[dict(checkpoint=selected,quality_pass=bool(selected))])
                (folder/'validation.json').write_text(json.dumps(report))
            save('1','ppo',checkpoint); save('2','icm','unused')
            self.assertEqual(selected_report('pmsm','ppo','run')[2],checkpoint)
            save('3','ppo',None)
            with self.assertRaises(ValueError): selected_report('pmsm','ppo','run')


if __name__=='__main__': unittest.main()
