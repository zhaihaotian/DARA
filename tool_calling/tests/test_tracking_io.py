"""Recover the shared-filesystem EIO seen while writing training metrics."""
import errno
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from verl.utils.tracking import _JsonlLogger


class TrackingIOTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'metrics.jsonl'
        self.logger = _JsonlLogger(self.path)
        self.logger.log({'reward': 0.25}, step=1)

    def records(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def test_open_recovers_and_keeps_both_steps(self):
        real_open = Path.open
        attempts = []

        def open_after_eio(path, *args, **kwargs):
            attempts.append(path)
            if len(attempts) <= 2:
                raise OSError(errno.EIO, 'Input/output error')
            return real_open(path, *args, **kwargs)

        with patch.object(Path, 'open', open_after_eio), patch('verl.utils.tracking.time.sleep'):
            self.logger.log({'reward': 0.5}, step=2)
        self.assertEqual(self.records(), [{'step': 1, 'reward': 0.25}, {'step': 2, 'reward': 0.5}])
        self.assertEqual(len(attempts), 3)

    def test_failed_write_is_replaced_without_a_duplicate_or_partial_line(self):
        real_open = Path.open
        for full_write in (False, True):
            with self.subTest(full_write=full_write):
                self.path.write_text('{"step": 1, "reward": 0.25}\n')
                attempts = []

                class FailingWrite:
                    def __init__(self, output):
                        self.output = output

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        self.output.close()

                    def tell(self):
                        return self.output.tell()

                    def write(self, record):
                        self.output.write(record if full_write else record[:8])
                        self.output.flush()
                        raise OSError(errno.EIO, 'Input/output error')

                def open_with_failed_write(path, *args, **kwargs):
                    output = real_open(path, *args, **kwargs)
                    attempts.append(path)
                    return FailingWrite(output) if len(attempts) == 1 else output

                with patch.object(Path, 'open', open_with_failed_write), patch('verl.utils.tracking.time.sleep'):
                    self.logger.log({'reward': 0.5}, step=2)
                self.assertEqual(self.records(), [{'step': 1, 'reward': 0.25}, {'step': 2, 'reward': 0.5}])

    def test_persistent_eio_is_reported_after_bounded_retries(self):
        with patch.object(Path, 'open', side_effect=OSError(errno.EIO, 'Input/output error')) as open_file, \
                patch('verl.utils.tracking.time.sleep') as sleep:
            with self.assertRaises(OSError) as error:
                self.logger.log({'reward': 0.5}, step=2)
        self.assertEqual(error.exception.errno, errno.EIO)
        self.assertEqual(open_file.call_count, 30)
        self.assertEqual(sleep.call_count, 29)
        self.assertEqual(self.records(), [{'step': 1, 'reward': 0.25}])

    def test_other_errors_keep_their_original_failure_behavior(self):
        with patch.object(Path, 'open', side_effect=PermissionError(errno.EACCES, 'Permission denied')), \
                patch('verl.utils.tracking.time.sleep') as sleep:
            with self.assertRaises(PermissionError):
                self.logger.log({'reward': 0.5}, step=2)
        sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
