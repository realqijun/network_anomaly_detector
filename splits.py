"""Leakage-resistant, group-based dataset splits.

CIC-IDS style corpora are structured by capture day, attack window, and fixed
scenarios, so a random row split leaks nearly identical behaviour across
train and test and inflates every metric. Splits here are assigned by whole
capture group, and the resulting manifest is validated before training starts
rather than only in a test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

GROUP_KEY = "capture_day"

# Two field orders, both three numbers. Which one a corpus uses cannot be
# decided per file: over a mirror-style set, 02-14 through 02-28 resolve
# month-first while 03-01 and 03-02 are ambiguous, so per-file disambiguation
# produces two key namespaces in one directory and splits a day's shards apart.
_NUMERIC_DATE = re.compile(r"(?<!\d)(\d{2})-(\d{2})-(\d{4})(?!\d)")
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")

DAY_FIRST = "day_first"
MONTH_FIRST = "month_first"
STEM = "stem"


class SplitLeakageError(ValueError):
    """Raised when a split would place one capture group on both sides."""


def _numeric_date(path: Path) -> tuple[int, int, int] | None:
    match = _NUMERIC_DATE.search(path.stem)
    if match is None:
        return None
    first, second, year = (int(part) for part in match.groups())
    return first, second, year


def resolve_capture_convention(paths: Sequence[Path]) -> str:
    """Decide the corpus-wide day/month order from every filename at once.

    Returns "day_first", "month_first", or "stem" when no filename carries a
    numeric date. Raises when a corpus contains both orders, because no single
    reading of it can be correct and guessing would silently mis-group days.
    """

    day_first_evidence: list[Path] = []
    month_first_evidence: list[Path] = []
    for path in paths:
        parsed = _numeric_date(path)
        if parsed is None:
            continue
        first, second, _ = parsed
        if first > 12:
            day_first_evidence.append(path)
        elif second > 12:
            month_first_evidence.append(path)

    if day_first_evidence and month_first_evidence:
        raise ValueError(
            "Capture filenames mix day-first and month-first ordering, so no "
            "single reading is correct: "
            f"{month_first_evidence[0].name} is month-first while "
            f"{day_first_evidence[0].name} is day-first. Normalize the "
            "filenames before training."
        )
    if day_first_evidence:
        return DAY_FIRST
    if month_first_evidence:
        return MONTH_FIRST
    return STEM


def capture_group(path: Path, convention: str) -> str:
    """The whole-capture group key for one file, under a resolved convention.

    Grouping on the date token rather than the stem means shards of one capture
    day share a group and therefore cannot land on opposite sides of a split.
    """

    iso = _ISO_DATE.search(path.stem)
    if iso is not None:
        year, month, day = iso.groups()
        return f"{year}-{month}-{day}"

    parsed = _numeric_date(path)
    if parsed is not None and convention in (DAY_FIRST, MONTH_FIRST):
        first, second, year = parsed
        day, month = (first, second) if convention == DAY_FIRST else (second, first)
        return f"{year:04d}-{month:02d}-{day:02d}"

    return path.stem


@dataclass(frozen=True)
class GroupedSplit:
    """One assignment of whole capture groups to partitions."""

    train: list[Path]
    validation: list[Path]
    test: list[Path] = field(default_factory=list)
    external_test: list[Path] = field(default_factory=list)
    groups: dict[str, str] = field(default_factory=dict)
    group_key: str = GROUP_KEY

    def _groups_for(self, paths: Iterable[Path]) -> list[str]:
        seen: list[str] = []
        for path in paths:
            group = self.groups.get(str(path))
            if group is not None and group not in seen:
                seen.append(group)
        return seen

    @property
    def train_groups(self) -> list[str]:
        return self._groups_for(self.train)

    @property
    def validation_groups(self) -> list[str]:
        return self._groups_for(self.validation)

    @property
    def test_groups(self) -> list[str]:
        return self._groups_for(self.test)


def _matches(group: str, path: Path, patterns: Sequence[str]) -> bool:
    """Patterns match a group key or a filename.

    Matching the group key is what lets the configured defaults be normalized
    dates that are correct under either filename convention; matching the
    filename keeps corpora that carry no date usable.
    """

    return any(pattern in group or pattern in path.name for pattern in patterns)


def build_grouped_split(
    files: Sequence[Path],
    validation_patterns: Sequence[str],
    test_patterns: Sequence[str] = (),
    external_test_files: Sequence[Path] = (),
) -> GroupedSplit:
    """Assign whole capture groups to train, validation and test."""

    if not files:
        raise ValueError("No capture files to split")

    convention = resolve_capture_convention(files)
    groups = {str(path): capture_group(path, convention) for path in files}

    selected_validation = {
        groups[str(path)]
        for path in files
        if _matches(groups[str(path)], path, validation_patterns)
    }
    selected_test = {
        groups[str(path)]
        for path in files
        if test_patterns and _matches(groups[str(path)], path, test_patterns)
    }
    overlap = selected_validation & selected_test
    if overlap:
        raise SplitLeakageError(
            "Validation and test patterns select the same capture groups: "
            + ", ".join(sorted(overlap))
        )

    train, validation, test = [], [], []
    for path in files:
        group = groups[str(path)]
        if group in selected_validation:
            validation.append(path)
        elif group in selected_test:
            test.append(path)
        else:
            train.append(path)

    if not validation:
        available = "\n".join(
            f"- {group}" for group in sorted(set(groups.values()))
        )
        raise ValueError(
            "No validation_patterns matched a capture group. Choose whole "
            "capture groups from:\n" + available
        )
    if not train:
        raise ValueError(
            "Validation and test patterns selected every capture group; no "
            "training files remain"
        )

    return GroupedSplit(
        train=train,
        validation=validation,
        test=test,
        external_test=list(external_test_files),
        groups=groups,
    )


def _partition_block(paths: Sequence[Path], groups: dict[str, str]) -> dict[str, list[str]]:
    block: dict[str, list[str]] = {}
    for path in paths:
        block.setdefault(groups.get(str(path), path.stem), []).append(path.name)
    return block


def split_manifest(split: GroupedSplit) -> dict:
    """The serializable split record that goes into the model bundle."""

    return {
        "group_key": split.group_key,
        "train": _partition_block(split.train, split.groups),
        "validation": _partition_block(split.validation, split.groups),
        "test": _partition_block(split.test, split.groups),
        "external_test": [path.name for path in split.external_test],
    }


def validate_split_manifest(manifest: dict) -> None:
    """Refuse a split that could leak one capture group across partitions."""

    partitions = ("train", "validation", "test")
    seen: dict[str, str] = {}
    for partition in partitions:
        for group in manifest.get(partition, {}):
            if group in seen:
                raise SplitLeakageError(
                    f"Capture group {group!r} appears in both {seen[group]!r} and "
                    f"{partition!r}; a group must belong to exactly one partition"
                )
            seen[group] = partition

    if not manifest.get("train"):
        raise SplitLeakageError("Split manifest has an empty train partition")
    if not manifest.get("validation"):
        raise SplitLeakageError("Split manifest has an empty validation partition")


__all__ = [
    "GROUP_KEY",
    "GroupedSplit",
    "SplitLeakageError",
    "build_grouped_split",
    "capture_group",
    "resolve_capture_convention",
    "split_manifest",
    "validate_split_manifest",
]
