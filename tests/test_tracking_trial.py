import copy
import json
import math
import unittest
from rllib_control import runner
from rllib_control.tracking_trial import comparison_values,progression,make_command


class TrackingTrialTests(unittest.TestCase):
    def record(self,rise=.3,offset=15.,mae=30.):
        case=dict(profile='steps',metrics=dict(completed_episode=True,safety_violation=False,
                time_above_current_scale_s=0.),response=[dict(step_rad_s=650.,rise_time_10_90_s=rise)],
                quality=dict(values=dict(speed_ripple=.2,speed_peak_to_peak=.8,
                    absolute_offset=offset,current_ripple_fraction=.01)))
        cases=[copy.deepcopy(case) for _ in range(4)]
        for c in cases[2:]:c['profile']='random';c['response']=[]
        return dict(cases=cases,mean_mae=mae)

    def history(self):
        return [dict(vf_explained_var=.6,value_prediction_std=50.,value_target_std=60.,
                     vf_normalized_loss=.1) for _ in range(20)]

    def test_only_tracking_weight_changes_and_old_run_is_not_resumed(self):
        args=runner.parser().parse_args(['train','--motor','pmsm','--stability-profile','pmsm-v1',
            '--critic-profile','normalized','--critic-return-scale','100','--current-rate-weight','.002',
            '--tracking-absolute-weight','20'])
        settings=runner.settings(args);command,actual=make_command(settings)
        self.assertEqual({k for k in settings.keys()|actual.keys() if settings.get(k)!=actual.get(k)},
                         {'tracking_weight'})
        self.assertEqual(actual['tracking_weight'],200.);self.assertNotIn('--resume',command)

    def test_progression_requires_speed_improvement_without_ripple_or_offset_regression(self):
        control=self.record(rise=.332,offset=17.,mae=33.)
        self.assertTrue(progression(self.history(),self.record(),control)['passed'])
        self.assertFalse(progression(self.history(),self.record(rise=.33),control)['passed'])
        self.assertFalse(progression(self.history(),self.record(offset=18.),control)['passed'])
        bad=self.record();bad['cases'][0]['quality']['values']['speed_ripple']=2.
        self.assertFalse(progression(self.history(),bad,control)['passed'])
        self.assertFalse(progression(self.history()[:10],self.record(),control)['passed'])

    def test_unreached_reversal_cannot_be_reported_fast(self):
        with self.assertRaises(ValueError):comparison_values(self.record(rise=float('nan')))
        self.assertFalse(progression(self.history(),self.record(rise=float('nan')),self.record())['passed'])


if __name__=='__main__':unittest.main()
