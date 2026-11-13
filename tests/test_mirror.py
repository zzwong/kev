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

    def test_unreadable_receipt_skips_upload_and_preserves_batch_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / name / "checkpoint" for name in ("broken", "healthy")]
            for path in paths:
                path.mkdir(parents=True)
                (path / "head.pt").write_bytes(b"test")
            for error in (ValueError("invalid JSON"), PermissionError("receipt unreadable")):
                with self.subTest(error=type(error).__name__):
                    hub = Mock()
                    hub.repo_info.return_value = SimpleNamespace(private=True)
                    hub.upload_folder.return_value = SimpleNamespace(oid="c1")
                    logs = []
                    with patch.object(mirror, "recorded", side_effect=[error, None]), patch.object(mirror, "record"):
                        entries = [mirror.mirror(path, "me/private", api=hub, root=root, log=logs.append) for path in paths]
                    self.assertIsNone(entries[0])
                    self.assertEqual(entries[1]["commit"], "c1")
                    hub.upload_folder.assert_called_once()
                    self.assertEqual(hub.upload_folder.call_args.kwargs["folder_path"], str(paths[1]))
                    self.assertTrue(any("unreadable upload record" in line and "not uploaded" in line for line in logs))

    def test_force_explicitly_bypasses_unreadable_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint"
            path.mkdir()
            (path / "head.pt").write_bytes(b"test")
            hub = Mock()
            hub.repo_info.return_value = SimpleNamespace(private=True)
            hub.upload_folder.return_value = SimpleNamespace(oid="forced")
            with patch.object(mirror, "recorded", side_effect=ValueError("invalid JSON")) as recorded, patch.object(mirror, "record"):
                entry = mirror.mirror(path, "me/private", api=hub, root=tmp, force=True, log=lambda line: None)
            recorded.assert_not_called()
            self.assertEqual(entry["commit"], "forced")
            hub.upload_folder.assert_called_once()

    def test_force_corrupt_snapshot_keeps_successful_upload_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trial" / "snapshots" / "step-0000001" / "checkpoint"
            path.mkdir(parents=True)
            (path / "head.pt").write_bytes(b"test")
            metadata = path / mirror.SNAPSHOT_INFO
            metadata.write_text("{broken", encoding="utf-8")
            hub = Mock()
            hub.repo_info.return_value = SimpleNamespace(private=True)
            hub.upload_folder.return_value = SimpleNamespace(oid="forced-snapshot")
            logs = []
            entry = mirror.mirror(path, "me/private", api=hub, root=tmp, force=True, log=logs.append)
            self.assertEqual(entry["commit"], "forced-snapshot")
            hub.upload_folder.assert_called_once()
            self.assertEqual(metadata.read_text(encoding="utf-8"), "{broken")
            self.assertTrue(any("local record could not be saved" in line and "forced-snapshot" in line for line in logs))
