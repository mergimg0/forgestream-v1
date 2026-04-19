"""Git worktree lifecycle management for scaffold agents."""

from __future__ import annotations

import subprocess
from pathlib import Path


class WorktreeManager:
    """Creates and manages git worktrees for scaffold agent isolation."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = Path(repo_path)
        self._parent = self.repo_path.parent

    def create(self, slug: str) -> str:
        """Create a new worktree with a dedicated branch.

        Returns the absolute path to the worktree directory.
        """
        wt_path = self._parent / f"{self.repo_path.name}-sc-{slug}"
        branch_name = f"sc/{slug}"

        subprocess.run(
            ["git", "worktree", "add", "-b", branch_name, str(wt_path)],
            cwd=self.repo_path,
            capture_output=True,
            check=True,
        )
        return str(wt_path)

    def remove(self, slug: str) -> None:
        """Remove a worktree and prune."""
        wt_path = self._parent / f"{self.repo_path.name}-sc-{slug}"
        subprocess.run(
            ["git", "worktree", "remove", str(wt_path), "--force"],
            cwd=self.repo_path,
            capture_output=True,
        )
        branch_name = f"sc/{slug}"
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=self.repo_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=self.repo_path,
            capture_output=True,
        )

    def list_worktrees(self) -> list[dict]:
        """List active scaffold worktrees."""
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        worktrees = []
        current: dict = {}
        prefix = f"{self.repo_path.name}-sc-"

        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                if current and current.get("slug"):
                    worktrees.append(current)
                path = line.split(" ", 1)[1]
                name = Path(path).name
                current = {
                    "path": path,
                    "slug": name.removeprefix(prefix) if prefix in name else "",
                }
            elif line.startswith("branch "):
                current["branch"] = line.split(" ", 1)[1]

        if current and current.get("slug"):
            worktrees.append(current)

        return worktrees
