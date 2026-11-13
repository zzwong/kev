"""Publication orchestration tests: no Hub calls or model weights."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kev import publish


class PublishTagTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        card = root / "card.md"
        card.write_text("base_model: test\n")
        self.argv = ["kev.publish", "--run", str(root), "--repo", "me/model", "--card", str(card)]
        self.hub = Mock()
        self.hub.list_repo_files.return_value = []
        self.hub.upload_folder.return_value = SimpleNamespace(oid="uploaded-commit")
        checkpoint = SimpleNamespace(meta=SimpleNamespace(base="test"), full=True, shards=lambda: [])
        for patcher in (patch.object(publish, "HfApi", return_value=self.hub),
                        patch.object(publish, "Checkpoint", return_value=checkpoint), patch("builtins.print")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_tag_pins_uploaded_commit_even_when_branch_advances(self):
        for branch in (None, "candidate"):
            with self.subTest(branch=branch):
                self.hub.reset_mock()
                # A concurrent writer advances the mutable branch after our upload.
                self.hub.repo_info.return_value = SimpleNamespace(sha="later-commit")
                argv = self.argv + ["--tag", "v1"] + (["--revision", branch] if branch else [])
                with patch("sys.argv", argv):
                    publish.main()
                self.hub.upload_folder.assert_called_once()
                self.assertEqual(self.hub.upload_folder.call_args.kwargs["revision"], branch)
                self.hub.create_tag.assert_called_once_with(
                    "me/model", tag="v1", repo_type="model", tag_message=self.hub.upload_folder.call_args.kwargs["commit_message"],
                    exist_ok=False, revision="uploaded-commit")
                self.hub.repo_info.assert_not_called()

    def test_tag_failure_does_not_upload_again(self):
        self.hub.create_tag.side_effect = RuntimeError("tag already exists")
        with patch("sys.argv", self.argv + ["--tag", "v1"]), self.assertRaisesRegex(RuntimeError, "tag already exists"):
            publish.main()
        self.hub.upload_folder.assert_called_once()
        self.hub.create_tag.assert_called_once()

    def test_upload_failure_does_not_tag(self):
        self.hub.upload_folder.side_effect = RuntimeError("upload failed")
        with patch("sys.argv", self.argv + ["--tag", "v1"]), self.assertRaisesRegex(RuntimeError, "upload failed"):
            publish.main()
        self.hub.create_tag.assert_not_called()

    def test_no_tag_requested(self):
        with patch("sys.argv", self.argv):
            publish.main()
        self.hub.upload_folder.assert_called_once()
        self.hub.create_tag.assert_not_called()
