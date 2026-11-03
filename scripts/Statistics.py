"""CSV-based summary, distribution, join and repeat-correlation utilities."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {path}")
            return list(reader.fieldnames), list(reader)
    except OSError as exc:
        raise ValueError(f"cannot read CSV: {path}") from exc


def write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def read_group_map(path: Path) -> dict[str, str]:
    """Read a two-column ID/group CSV with an optional header row."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = [[cell.strip() for cell in row] for row in csv.reader(handle)
                    if row and any(cell.strip() for cell in row)]
    except OSError as exc:
        raise ValueError(f"cannot read group CSV: {path}") from exc
    if not rows:
        raise ValueError(f"group CSV contains no data: {path}")
    if len(rows[0]) < 2:
        raise ValueError("group CSV needs at least two columns: sample ID and group ID")

    first_id = rows[0][0].casefold().replace(" ", "").replace("_", "")
    first_group = rows[0][1].casefold().replace(" ", "").replace("_", "")
    id_headers = {"id", "sample", "sampleid", "样本", "样本编号"}
    group_headers = {"group", "groupid", "分组", "组别", "材料"}
    has_header = first_id in id_headers or first_group in group_headers
    if has_header:
        rows = rows[1:]

    groups: dict[str, str] = {}
    for line_number, row in enumerate(rows, start=2 if has_header else 1):
        if len(row) < 2 or not row[0] or not row[1]:
            raise ValueError(f"invalid group CSV row {line_number}: expected ID,group")
        sample_id, group_id = row[0], row[1]
        if sample_id in groups and groups[sample_id] != group_id:
            raise ValueError(
                f"sample ID {sample_id!r} maps to multiple groups: "
                f"{groups[sample_id]!r} and {group_id!r}"
            )
        groups[sample_id] = group_id
    if not groups:
        raise ValueError(f"group CSV contains no group mappings: {path}")
    return groups


def _numeric(value: str, column: str, row_id: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric value in {column!r} for ID {row_id!r}: {value!r}") from exc


def grouped_summary(value_file: Path, group_file: Path, output: Path) -> int:
    value_header, values = read_csv(value_file)
    if len(value_header) < 2:
        raise ValueError("value CSV needs an ID column and at least one numeric column")
    id_column, numeric_columns = value_header[0], value_header[1:]
    group_map = read_group_map(group_file)
    buckets: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    order = []
    for row in values:
        row_id = row[id_column].strip()
        if row_id not in group_map:
            raise ValueError(f"ID {row_id!r} is missing from group file")
        group = group_map[row_id]
        if group not in order:
            order.append(group)
        for column in numeric_columns:
            buckets[group][column].append(_numeric(row[column], column, row_id))

    header = (["ID"] + [f"{name}_mean" for name in numeric_columns]
              + [f"{name}_min" for name in numeric_columns]
              + [f"{name}_max" for name in numeric_columns]
              + [f"{name}_std" for name in numeric_columns] + ["repeats"])
    result = []
    for group in order:
        series = [buckets[group][name] for name in numeric_columns]
        repeat_counts = {len(item) for item in series}
        if len(repeat_counts) != 1:
            raise ValueError(f"group {group!r} has inconsistent repeat counts")
        means = [statistics.fmean(item) for item in series]
        minima = [min(item) for item in series]
        maxima = [max(item) for item in series]
        stds = [statistics.stdev(item) if len(item) > 1 else math.nan for item in series]
        result.append([group, *means, *minima, *maxima, *stds, repeat_counts.pop()])
    write_csv(output, header, ([f"{v:.2f}" if isinstance(v, float) else v for v in row]
                               for row in result))
    return len(result)


def trait_values(value_file: Path, trait: str) -> tuple[str, list[float]]:
    header, rows = read_csv(value_file)
    if trait not in header:
        raise ValueError(f"trait {trait!r} not found; available: {', '.join(header[1:])}")
    id_column = header[0]
    values = [_numeric(row[trait], trait, row[id_column]) for row in rows]
    if not values:
        raise ValueError("CSV contains no data rows")
    return id_column, values


def describe(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else math.nan
    return {"count": len(values), "minimum": min(values), "maximum": max(values),
            "mean": mean, "std": std,
            "cv": std / mean if mean != 0 else math.nan}


def join_traits(left_file: Path, right_file: Path, output: Path,
                left_trait: str, right_trait: str,
                column_names: tuple[str, str] | None = None) -> int:
    left_header, left_rows = read_csv(left_file)
    right_header, right_rows = read_csv(right_file)
    if left_trait not in left_header or right_trait not in right_header:
        raise ValueError("requested trait is not present in one of the CSV files")
    left_id, right_id = left_header[0], right_header[0]
    left = {row[left_id]: row[left_trait] for row in left_rows}
    names = column_names or (left_trait, right_trait)
    rows = [[row[right_id], left[row[right_id]], row[right_trait]]
            for row in right_rows if row[right_id] in left]
    write_csv(output, ["ID", names[0], names[1]], rows)
    return len(rows)


def repeat_pairs(value_file: Path) -> tuple[str, str, list[float], list[float]]:
    header, rows = read_csv(value_file)
    if len(header) < 3:
        raise ValueError("repeat CSV needs ID and two numeric columns")
    x_name, y_name = header[1], header[2]
    x = [_numeric(row[x_name], x_name, row[header[0]]) for row in rows]
    y = [_numeric(row[y_name], y_name, row[header[0]]) for row in rows]
    if len(x) < 2:
        raise ValueError("correlation needs at least two data rows")
    return x_name, y_name, x, y
