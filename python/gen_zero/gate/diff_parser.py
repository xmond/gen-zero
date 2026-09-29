"""Diff Hunk Parser and Dual-Context Collator for Semantic Code Review Gate.

Implements Milestone 1 of Issue #22:
- Parses unified diff into structured file diffs and individual hunks.
- Distinguishes business logic changes from test code changes.
- Collates dual-context representation (file_patch vs changed_tests) for testGap evaluation.
"""

from typing import List, Dict, Any, Optional, Tuple
import dataclasses
import re


@dataclasses.dataclass
class DiffHunk:
    """Represents a single unified diff hunk with line offsets and content."""
    hunk_id: str
    file_path: str
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    header: str
    lines: List[str]
    content: str

    @property
    def added_lines(self) -> List[str]:
        return [l[1:] for l in self.lines if l.startswith("+") and not l.startswith("+++")]

    @property
    def removed_lines(self) -> List[str]:
        return [l[1:] for l in self.lines if l.startswith("-") and not l.startswith("---")]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hunk_id": self.hunk_id,
            "file_path": self.file_path,
            "old_start": self.old_start,
            "old_lines": self.old_lines,
            "new_start": self.new_start,
            "new_lines": self.new_lines,
            "header": self.header,
            "content": self.content,
        }


@dataclasses.dataclass
class FileDiff:
    """Represents changes in a single file within a unified diff."""
    file_path: str
    old_path: Optional[str]
    new_path: Optional[str]
    is_test: bool
    is_binary: bool
    hunks: List[DiffHunk]
    raw_patch: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "is_test": self.is_test,
            "is_binary": self.is_binary,
            "hunks_count": len(self.hunks),
        }


@dataclasses.dataclass
class DualContext:
    """Dual context separating production code delta from test assertion delta."""
    file_patch: str
    changed_tests: str
    business_files: List[str]
    test_files: List[str]
    all_hunks: List[DiffHunk]
    hunks_by_id: Dict[str, DiffHunk]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "business_files": self.business_files,
            "test_files": self.test_files,
            "total_hunks": len(self.all_hunks),
            "file_patch_bytes": len(self.file_patch.encode("utf-8")),
            "changed_tests_bytes": len(self.changed_tests.encode("utf-8")),
        }


class DiffHunkParser:
    """Parses standard unified diff into FileDiff instances with line-precise hunks."""

    TEST_PATTERNS = [
        re.compile(r"(^|/)tests?/"),
        re.compile(r"(^|/)test_[^/]+\.py$"),
        re.compile(r"[._]test\.[a-zA-Z0-9]+$"),
        re.compile(r"[._]spec\.[a-zA-Z0-9]+$"),
    ]

    HUNK_HEADER_REGEX = re.compile(
        r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
    )

    @classmethod
    def is_test_path(cls, path: str) -> bool:
        """Determines if a given file path is a test/spec file."""
        if not path:
            return False
        return any(p.search(path) is not None for p in cls.TEST_PATTERNS)

    @classmethod
    def parse_unified_diff(cls, diff_text: str) -> List[FileDiff]:
        """Parses a unified diff string into a list of FileDiff records."""
        if not diff_text or not diff_text.strip():
            return []

        file_diffs: List[FileDiff] = []
        lines = diff_text.splitlines(keepends=True)
        idx = 0
        n = len(lines)

        while idx < n:
            line = lines[idx]
            if line.startswith("diff --git "):
                # Extract old and new file paths
                parts = line.strip().split()
                old_p = parts[2][2:] if len(parts) > 2 and parts[2].startswith("a/") else None
                new_p = parts[3][2:] if len(parts) > 3 and parts[3].startswith("b/") else None
                current_file = new_p or old_p or "unknown"

                idx += 1
                raw_lines = [line]
                hunks: List[DiffHunk] = []
                is_binary = False

                while idx < n and not lines[idx].startswith("diff --git "):
                    subline = lines[idx]
                    raw_lines.append(subline)
                    if "Binary files" in subline:
                        is_binary = True

                    match = cls.HUNK_HEADER_REGEX.match(subline.strip())
                    if match:
                        old_s = int(match.group(1))
                        old_c = int(match.group(2)) if match.group(2) else 1
                        new_s = int(match.group(3))
                        new_c = int(match.group(4)) if match.group(4) else 1
                        header_extra = match.group(5).strip()

                        hunk_id = f"{current_file}#L{new_s}"
                        hunk_lines = []
                        idx += 1
                        while idx < n and not lines[idx].startswith("diff --git ") and not cls.HUNK_HEADER_REGEX.match(lines[idx].strip()):
                            hunk_lines.append(lines[idx])
                            raw_lines.append(lines[idx])
                            idx += 1

                        hunk_content = "".join(hunk_lines)
                        hunks.append(DiffHunk(
                            hunk_id=hunk_id,
                            file_path=current_file,
                            old_start=old_s,
                            old_lines=old_c,
                            new_start=new_s,
                            new_lines=new_c,
                            header=subline.strip(),
                            lines=hunk_lines,
                            content=hunk_content,
                        ))
                        continue

                    idx += 1

                file_diffs.append(FileDiff(
                    file_path=current_file,
                    old_path=old_p,
                    new_path=new_p,
                    is_test=cls.is_test_path(current_file),
                    is_binary=is_binary,
                    hunks=hunks,
                    raw_patch="".join(raw_lines),
                ))
            else:
                idx += 1

        return file_diffs


class DualContextCollator:
    """Collates production code and test code deltas into dual context."""

    @classmethod
    def collate(cls, diff_text: str) -> DualContext:
        """Parses diff text and splits into business logic patch and changed tests patch."""
        file_diffs = DiffHunkParser.parse_unified_diff(diff_text)

        biz_patches: List[str] = []
        test_patches: List[str] = []
        biz_files: List[str] = []
        test_files: List[str] = []
        all_hunks: List[DiffHunk] = []
        hunks_by_id: Dict[str, DiffHunk] = {}

        for fd in file_diffs:
            if fd.is_test:
                test_files.append(fd.file_path)
                test_patches.append(fd.raw_patch)
            else:
                biz_files.append(fd.file_path)
                biz_patches.append(fd.raw_patch)

            for h in fd.hunks:
                all_hunks.append(h)
                hunks_by_id[h.hunk_id] = h

        return DualContext(
            file_patch="".join(biz_patches),
            changed_tests="".join(test_patches),
            business_files=biz_files,
            test_files=test_files,
            all_hunks=all_hunks,
            hunks_by_id=hunks_by_id,
        )
