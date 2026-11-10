import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from rllib_control import runner,research,current_suite
from rllib_control.current_control import CurrentReferenceEnv
from rllib_control.matched import MatchedCurrentEnv,MatchedVoltageEnv
from rllib_control.scenarios import Scenario


class ResearchTests(unittest.TestCase):
    def test_same_noise_prefix_and_no_state_corruption(self):
        for motor in ('dc','pmsm'):
            args=runner.parser().parse_args(['eval','--motor',motor,'--scenario','sensor_noise','--episode-length','.04'])
            envs=[CurrentReferenceEnv(dict(motor=motor,episode_length=.04,training_mode=False)) for _ in range(2)]
            scenarios=[]
            for env in envs:
                env.reset(seed=args.seed);scenarios.append(Scenario(env,args))
            a,b=scenarios
            self.assertEqual(a.metadata['noise_sha256'],b.metadata['noise_sha256'])
            state=vars(envs[0].base.state).copy()
            o=a.initial_observation(envs[0]);a.measure();a.measure()
            self.assertEqual(state,vars(envs[0].base.state))
            self.assertNotEqual(o[0],0.)
            envs[0].step([.5]);envs[1].step([-.5])
            tick=round(envs[0].base.time/envs[0].base.dt)
            self.assertAlmostEqual(a.measure()['speed']-envs[0].base.state.omega,a.noise[tick,0])
            self.assertAlmostEqual(b.measure()['speed']-envs[1].base.state.omega,a.noise[tick,0])

    def test_parameter_event_and_controller_nominals(self):
        for motor in ('dc','pmsm'):
            args=runner.parser().parse_args(['eval','--motor',motor,'--scenario','resistance_jump',
                '--episode-length','.02','--load-step-scale','.5','--resistance-scale','1.2'])
            env=CurrentReferenceEnv(dict(motor=motor,episode_length=.02,training_mode=False));env.reset(seed=1)
            b=env.base;plant=b.motor if motor=='dc' else b
            before=plant.R;load=b.load_torque
            scenario=Scenario(env,args)
            self.assertAlmostEqual(plant.R,before*1.2)
            for _ in range(b.max_steps):env.step([0.])
            self.assertTrue(scenario.applied)
            self.assertAlmostEqual(plant.R,before*1.2*1.3)
            self.assertAlmostEqual(b.load_torque,load*1.5)

    def test_matched_cost_and_penalty(self):
        for motor in ('dc','pmsm'):
            a=MatchedCurrentEnv(dict(motor=motor,training_mode=False))
            b=MatchedVoltageEnv(dict(motor=motor,training_mode=False))
            a.reset(seed=1);b.reset(seed=1)
            self.assertEqual(a._reward(),b._reward())
            for env in (a,b):
                old=env.limited.copy()
                _,_,_,_,info=env.step(np.full(env.action_space.shape,.1))
                expected=.05*np.sum(((env.limited-old)/env.vmax)**2)
                self.assertAlmostEqual(info['command_smoothness_penalty'],expected)

    def test_direct_final_without_screen(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(runner,'ROOT',Path(directory)):
            calls=[]
            def fake(command):
                args=runner.parser().parse_args(command);calls.append(args)
                if args.command=='train':
                    (runner.run_dir(args)/'iteration_1000').mkdir(parents=True)
                return Path(directory)/'evaluation'
            with patch.object(runner,'main',side_effect=fake),patch('rllib_control.baselines.tune',return_value=Path(directory)/'baseline.json'),patch('sys.argv',['suite','final','--motor','dc','--run','direct','--profiles','steps']):
                current_suite.run()
            trains=[a for a in calls if a.command=='train']
            self.assertEqual(len(trains),3)
            self.assertTrue(all(not a.resume for a in trains))
            self.assertEqual([a.run for a in trains],['direct','direct','direct_decay'])

    def test_aggregate_counts_failures_without_pseudoreplication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'cases').mkdir()
            jobs=[dict(id=str(s),motor='dc',variant='ppo',train_seed=s) for s in (1,2,3)]
            protocol=dict(jobs=jobs,profiles=['steps'],scenarios=['nominal'],test_seeds=[901,902])
            for job,profile,scenario,seed in research.cases(protocol):
                if job['train_seed']==3:continue
                value=2. if job['train_seed']==1 else 6.
                record=dict(status='complete',metrics={m:value for m in research.METRICS})
                runner.save_json(root/'cases'/(research.case_id(job,profile,scenario,seed)+'.json'),record)
            summary=research.aggregate(protocol,root)[0]
            self.assertEqual(summary['mean'],4.)
            self.assertAlmostEqual(summary['std_across_training_seeds'],np.sqrt(8.))
            self.assertEqual(summary['missing_cases'],2)
            path=root/'cases'/(research.case_id(jobs[1],'steps','nominal',901)+'.json')
            runner.save_json(path,dict(status='failed_episode',metrics={m:999. for m in research.METRICS}))
            summary=research.aggregate(protocol,root)[0]
            self.assertEqual(summary['valid_seed_groups'],1)
            self.assertEqual(summary['mean'],2.)
            self.assertEqual(summary['failed_cases'],1)


if __name__=='__main__':unittest.main()
