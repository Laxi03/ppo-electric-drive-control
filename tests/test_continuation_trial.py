import json
from pathlib import Path
import tempfile
import unittest
from rllib_control.continuation_trial import clone_run,file_hashes


class ContinuationTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'source';(source/'iteration_0500').mkdir(parents=True)
        (source/'iteration_0500/weights').write_bytes(b'original')
        (source/'training_history.csv').write_text('iteration,steps\n500,2048000\n')
        (source/'manifest.json').write_text(json.dumps(dict(status='complete',completed_iteration=500,
            target_iterations=500,settings=dict(seed=345))))
        return source

    def test_clone_is_exact_source_preserved_and_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=self.fixture(root);destination=root/'new'
            before=file_hashes(source);expected=file_hashes(source/'iteration_0500')
            clone_run(source,destination,expected)
            self.assertEqual(file_hashes(source),before)
            self.assertEqual(file_hashes(destination/'iteration_0500'),expected)
            manifest=json.loads((destination/'manifest.json').read_text())
            self.assertEqual(manifest['settings'],dict(seed=345))
            self.assertEqual(manifest['lineage']['total_iterations'],700)
            self.assertEqual(manifest['lineage']['additional_training_iterations'],200)
            with self.assertRaises(FileExistsError):clone_run(source,destination,expected)

    def test_changed_source_and_incomplete_clone_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=self.fixture(root);destination=root/'new'
            with self.assertRaises(ValueError):clone_run(source,destination,{'weights':'wrong'})
            self.assertFalse(destination.exists())
            with self.assertRaises(FileExistsError):clone_run(source,destination,file_hashes(source/'iteration_0500'))


if __name__=='__main__':unittest.main()
