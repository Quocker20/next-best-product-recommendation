"""Compare two nested result dicts (e.g. a re-run against a saved reference JSON)."""

from __future__ import annotations

from numbers import Number


def flatten(d, prefix: str = "") -> dict[str, float]:
    """Numeric leaves of nested dicts/lists as {"a.b.0.c": value}; bools and strings skipped."""
    out: dict[str, float] = {}
    items = d.items() if isinstance(d, dict) else enumerate(d) if isinstance(d, list) else ()
    for k, v in items:
        path = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, (dict, list)):
            out.update(flatten(v, path))
        elif isinstance(v, Number) and not isinstance(v, bool):
            out[path] = float(v)
    return out


def compare_nested(
    ref, new, skip: tuple[str, ...] = ("seconds", "timing")
) -> list[tuple[str, float, float, float]]:
    """Numeric leaves present in both dicts, as (path, ref, new, new - ref) rows.

    Paths containing any substring in `skip` (wall-clock timings by default) are left out.
    """
    a, b = flatten(ref), flatten(new)
    return [(p, a[p], b[p], b[p] - a[p]) for p in a if p in b and not any(s in p for s in skip)]
