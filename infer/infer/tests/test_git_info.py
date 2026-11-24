"""infer_git_info 单元测试。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from infer.recording.git_info import (
    format_infer_git_info,
    get_infer_git_info,
    stamp_path,
    write_infer_git_stamp,
)


class InferGitInfoTest(unittest.TestCase):
    def test_get_infer_git_info_from_git(self) -> None:
        def fake_run(cmd, **kwargs):
            args = cmd[1:]
            out = mock.Mock()
            out.returncode = 0
            out.stderr = ""
            if args == ["rev-parse", "HEAD"]:
                out.stdout = "abcdef0123456789abcdef0123456789abcdef01\n"
            elif args == ["rev-parse", "--abbrev-ref", "HEAD"]:
                out.stdout = "feature/git-meta\n"
            elif args == ["status", "--porcelain"]:
                out.stdout = " M infer/recording/git_info.py\n"
            else:
                out.returncode = 1
                out.stdout = ""
            return out

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".git").mkdir()
            with mock.patch("infer.recording.git_info.subprocess.run", side_effect=fake_run):
                info = get_infer_git_info(root, refresh_stamp=True)

            self.assertEqual(
                info["infer_git_commit"],
                "abcdef0123456789abcdef0123456789abcdef01",
            )
            self.assertEqual(info["infer_git_commit_short"], "abcdef0")
            self.assertEqual(info["infer_git_branch"], "feature/git-meta")
            self.assertTrue(info["infer_git_dirty"])
            self.assertEqual(info["infer_git_source"], "git")
            self.assertTrue(stamp_path(root).is_file())
            self.assertIn("source=git", format_infer_git_info(info))

    def test_fallback_to_stamp_when_no_git(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_infer_git_stamp(
                {
                    "infer_git_commit": "deadbeef0123456789abcdef0123456789abcdef",
                    "infer_git_commit_short": "deadbee",
                    "infer_git_branch": "wkx-dev",
                    "infer_git_dirty": False,
                    "infer_git_source": "git",
                },
                repo_root=root,
            )
            info = get_infer_git_info(root, refresh_stamp=False)
            self.assertEqual(
                info["infer_git_commit"],
                "deadbeef0123456789abcdef0123456789abcdef",
            )
            self.assertEqual(info["infer_git_source"], "stamp")
            self.assertFalse(info["infer_git_dirty"])
            self.assertIn("source=stamp", format_infer_git_info(info))

    def test_env_overrides_git_and_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".git").mkdir()
            env = {
                "INFER_GIT_COMMIT": "1111222233334444555566667777888899990000",
                "INFER_GIT_BRANCH": "deploy",
                "INFER_GIT_DIRTY": "0",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                info = get_infer_git_info(root, refresh_stamp=False)
            self.assertEqual(info["infer_git_source"], "env")
            self.assertEqual(info["infer_git_commit_short"], "1111222")
            self.assertEqual(info["infer_git_branch"], "deploy")
            self.assertFalse(info["infer_git_dirty"])

    def test_get_infer_git_info_fail_soft(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            info = get_infer_git_info(root, refresh_stamp=False)
            self.assertIsNone(info["infer_git_commit"])
            self.assertEqual(info["infer_git_source"], "unknown")
            self.assertIn("sha=unknown", format_infer_git_info(info))


if __name__ == "__main__":
    unittest.main()
