"""Best-effort checkpoint mirroring without provider calls or model weights."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kev import mirror


class MirrorRecordTests(unittest.TestCase):
    def test_receipt_write_failure_keeps_upload_identity_and_batch_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / name / "checkpoint" for name in ("first", "second")]
            for path in paths:
                path.mkdir(parents=True)
                (path / "head.pt").write_bytes(b"test")
            for error in (OSError("disk full"), PermissionError("read-only receipt")):
                with self.subTest(error=type(error).__name__):
                    hub = Mock()
                    hub.repo_info.return_value = SimpleNamespace(private=True)
                    hub.upload_folder.side_effect = [SimpleNamespace(oid="c1"), SimpleNamespace(oid="c2")]
                    logs = []
                    with patch.object(mirror, "record", side_effect=[error, None]) as record:
                        entries = [mirror.mirror(path, "me/private", api=hub, root=root, log=logs.append) for path in paths]
                    self.assertEqual([entry["commit"] for entry in entries], ["c1", "c2"])
                    self.assertEqual(hub.upload_folder.call_count, 2)
                    self.assertEqual(record.call_count, 2)
                    self.assertTrue(any("commit c1" in line and "local record could not be saved" in line for line in logs))
                    self.assertTrue(any("later mirror may upload it again" in line for line in logs))
                    self.assertTrue(all((path / "head.pt").read_bytes() == b"test" for path in paths))
