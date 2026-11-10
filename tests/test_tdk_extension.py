import hashlib
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from rllib_control import runner,research
from rllib_control.current_control import CurrentReferenceEnv,environment_config
from rllib_control.pmsm_confirmation import matched_command
from rllib_control.tdk_extension import make_protocol,verify_pairing,enrich_metrics


class ExtensionTests(unittest.TestCase):
    def fixture(self,root):
        args=runner.parser().parse_args(['train','--motor','pmsm','--stability-profile','pmsm-v1',
            '--tracking-absolute-weight','20','--current-rate-weight','.002'])
        settings=runner.settings(args);jobs=[]
        for controller in ['ppo','icm']:
            for seed in [123,234,456]:
                run=f'parent_{controller}_{seed}';command=matched_command(settings,run,controller,seed,500,2)
                parsed=runner.parser().parse_args(command)
                jobs.append(dict(id=run,variant='ppo' if controller=='ppo' else 'icm_fixed',controller=controller,
                    motor='pmsm',train_seed=seed,run=run,command=command,settings=runner.settings(parsed)))
        command=list(jobs[0]['command']);command[command.index('--run')+1]='parent_baseline'
        jobs.append(dict(id='baseline',variant='baseline',controller='baseline',motor='pmsm',train_seed=None,
                         run='parent_baseline',environment_command=command))
        parent=dict(study='pmsm_confirmation',require_frozen_baseline=True,jobs=jobs,iterations=500,
                    episode_length=8.,source_sha256={'code':'hash'})
        path=root/'protocol.json';path.write_text(json.dumps(parent))
        frozen=root/'original_baseline.json'
        frozen.write_text(json.dumps(dict(motor='pmsm',environment=environment_config(args),
                                         validation_seeds=[501,502],gains=dict(kp=.03,ki=.1))))
        (root/'pmsm_baseline.json').write_text(json.dumps(dict(path=str(frozen),
            sha256=hashlib.sha256(frozen.read_bytes()).hexdigest())))
        return path

    def test_paired_randomization_preserves_references_and_parameter_jump(self):
        for seed in [123,234,456]:
            on=CurrentReferenceEnv(dict(motor='pmsm',training_mode=True,domain_randomization=True,parameter_jump=True))
            off=CurrentReferenceEnv(dict(motor='pmsm',training_mode=True,domain_randomization=False,
                                        paired_domain_rng=True,parameter_jump=True))
            for episode in range(3):
                on.reset(seed=seed if episode==0 else None);off.reset(seed=seed if episode==0 else None)
                np.testing.assert_array_equal(on.base.reference.times,off.base.reference.times)
                np.testing.assert_array_equal(on.base.reference.values,off.base.reference.values)
                self.assertEqual(on.base.jump_time,off.base.jump_time)
                self.assertEqual(on.base.jump_factor,off.base.jump_factor)
                self.assertEqual(off.base.R,off.base.R_nom)
                self.assertEqual(off.base.J,off.base.params.J)
            on.close();off.close()

    def test_default_randomized_training_is_unchanged(self):
        config=dict(motor='pmsm',training_mode=True,domain_randomization=True)
        a=CurrentReferenceEnv(config);b=CurrentReferenceEnv(dict(config,paired_domain_rng=True))
        oa,_=a.reset(seed=234);ob,_=b.reset(seed=234);np.testing.assert_array_equal(oa,ob)
        for command in ([.5],[-.3],[.1]):
            ra=a.step(command);rb=b.step(command)
            np.testing.assert_array_equal(ra[0],rb[0]);self.assertEqual(ra[1],rb[1])
        a.close();b.close()

    @patch('rllib_control.research.fingerprints',return_value={'code':'hash'})
    def test_single_factor_plans_reuse_reference_models(self,_):
        with tempfile.TemporaryDirectory() as directory:
            parent=self.fixture(Path(directory))
            for study,changes in [('environment',{'no_domain_randomization','paired_domain_rng'}),
                                  ('reward',{'current_rate_weight'})]:
                protocol=make_protocol(parent,study,'extension')
                self.assertEqual(len(protocol['jobs']),7)
                self.assertEqual(sum(1 for _ in research.cases(protocol)),210)
                full=[j for j in protocol['jobs'] if j['variant']=='full']
                altered=[j for j in protocol['jobs'] if j['controller']=='ppo' and j['variant']!='full']
                self.assertTrue(all(j['reuse_checkpoint'] for j in full))
                for a,b in zip(full,altered):
                    difference={k for k in a['settings'].keys()|b['settings'].keys() if a['settings'].get(k)!=b['settings'].get(k)}
                    self.assertEqual(difference,changes)
            robust=make_protocol(parent,'robustness','robust')
            self.assertTrue(all(j.get('reuse_checkpoint') for j in robust['jobs'] if j['controller']!='baseline'))

    @patch('rllib_control.research.fingerprints',return_value={'code':'hash'})
    def test_frozen_baseline_is_copied_never_retuned_and_mutation_rejected(self,_):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent=self.fixture(root)
            protocol=make_protocol(parent,'robustness','robust');output=root/'output';output.mkdir()
            with patch('rllib_control.baselines.tune',side_effect=AssertionError('Retuning forbidden')):
                paths=research.freeze_baselines(protocol,output)
                self.assertNotEqual(paths['pmsm'],root/'original_baseline.json')
                paths['pmsm'].write_text('{}')
                with self.assertRaises(ValueError):research.freeze_baselines(protocol,output)

    def test_reference_or_noise_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'cases').mkdir()
            protocol=dict(jobs=[dict(id=str(i)) for i in [1,2]],profiles=['steps'],scenarios=['nominal'],test_seeds=[2901])
            for job,profile,scenario,seed in research.cases(protocol):
                path=root/'cases'/(research.case_id(job,profile,scenario,seed)+'.json')
                runner.save_json(path,dict(status='complete',paired_reference_sha256=job['id'],
                                         scenario_metadata=dict(noise_sha256='fixed')))
            with self.assertRaises(ValueError):verify_pairing(protocol,root)

    def test_held_out_evaluation_requires_complete_immutable_models(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(runner,'ROOT',Path(directory)):
            root=Path(directory);output=root/'output';output.mkdir()
            args=runner.parser().parse_args(['train','--motor','pmsm','--run','test'])
            job=dict(id='model',controller='ppo',command=['train','--motor','pmsm','--run','test'],settings=runner.settings(args))
            folder=runner.run_dir(args);checkpoint=folder/'iteration_0500';checkpoint.mkdir(parents=True)
            (checkpoint/'state').write_text('original')
            manifest=dict(settings=job['settings'],status='running',completed_iteration=450)
            runner.save_json(folder/'manifest.json',manifest)
            protocol=dict(jobs=[job],iterations=500)
            with self.assertRaises(ValueError):research.freeze_models(protocol,output)
            self.assertFalse((output/'frozen_models.json').exists())
            manifest.update(status='complete',completed_iteration=500);runner.save_json(folder/'manifest.json',manifest)
            research.freeze_models(protocol,output)
            (checkpoint/'state').write_text('modified')
            with self.assertRaises(ValueError):research.freeze_models(protocol,output)

    def test_extended_metrics_read_raw_physical_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=np.zeros((20,10));data[:,0]=np.arange(1,21)*.001
            data[10:,1]=100.;data[10:,2]=100.
            np.savetxt(root/'ppo_timeseries.csv',data,delimiter=',',header=','.join(str(i) for i in range(10)),comments='')
            row=dict(speed_ripple_RMS=.1,speed_peak_to_peak=.3,steady_offset_rad_s=2.,
                     i_or_id_ripple_RMS=.01,iq_ripple_RMS=.02)
            with (root/'ppo_ripple_by_plateau.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            metrics=dict(completed_episode=True,safety_violation=False,time_above_current_scale_s=0.,MAE_speed_rad_s=0.)
            enriched,digest=enrich_metrics(root,metrics,'pmsm','steps')
            self.assertEqual(enriched['engineering_pass'],1.)
            self.assertEqual(enriched['worst_absolute_offset_rad_s'],2.)
            self.assertAlmostEqual(enriched['largest_step_rise_s'],.0008)
            self.assertEqual(digest,hashlib.sha256(data[:,1].tobytes()).hexdigest())


if __name__=='__main__':unittest.main()
