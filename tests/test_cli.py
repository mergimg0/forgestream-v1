"""CLI entry point tests."""

import subprocess
import sys


class TestCLI:
    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "forgestream", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "forgestream" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_status_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "forgestream", "status"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "forgestream" in result.stdout.lower()

    def test_unknown_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "forgestream", "nonexistent"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
