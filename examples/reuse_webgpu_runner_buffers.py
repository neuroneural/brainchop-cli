#!/usr/bin/env python3
"""Rewrite a generated WebGPU runner to reuse dead activation buffers.

The transformation preserves every shader, dispatch geometry, weight, and pass
in the input runner. Only ``createEmptyBuf`` temporaries named ``buf_N`` are
interval-colored into ``arena_N`` GPU buffers. Inputs and outputs stay
dedicated. The output is therefore numerically identical and retains the
runner's existing device-specific tuning.
"""
import argparse
import re
from pathlib import Path


DECL_RE = re.compile(
    r"^(?P<indent>\s*)const (?P<name>buf_\d+) = createEmptyBuf\(device, (?P<size>\d+)\);;?\s*$",
    re.MULTILINE,
)
PASS_RE = re.compile(
    r"(?P<prefix>addComputePass\([^\n]*?infinityBuf, \[)(?P<args>[^\]]*)(?P<suffix>\], \[[^\]]*\]\);)"
)


def mib(value: int) -> str:
    return f"{value / 2**20:.1f} MiB"


def rewrite(source: str) -> tuple[str, dict]:
    declarations = list(DECL_RE.finditer(source))
    if not declarations:
        raise ValueError("runner has no temporary createEmptyBuf declarations named buf_N")
    sizes = {match.group("name"): int(match.group("size")) for match in declarations}
    if len(sizes) != len(declarations):
        raise ValueError("runner declares a temporary buffer more than once")

    passes = list(PASS_RE.finditer(source))
    if not passes:
        raise ValueError("runner has no addComputePass calls")
    usage: dict[str, list[int]] = {}
    pass_args: list[list[str]] = []
    for index, match in enumerate(passes):
        args = [arg.strip() for arg in match.group("args").split(",") if arg.strip()]
        pass_args.append(args)
        for arg in args:
            if arg in sizes:
                usage.setdefault(arg, []).append(index)
    missing = set(sizes) - set(usage)
    if missing:
        raise ValueError(f"temporary buffers never used by a compute pass: {sorted(missing)}")

    # Fail closed if a temporary participates in code this parser does not own.
    for name in sizes:
        all_uses = len(re.findall(rf"\b{re.escape(name)}\b", source))
        parsed_uses = 1 + sum(args.count(name) for args in pass_args)
        if all_uses != parsed_uses:
            raise ValueError(
                f"{name} has {all_uses} source uses but only {parsed_uses} are declarations/compute arguments"
            )

    slots: list[dict[str, int]] = []
    rename: dict[str, str] = {}
    intervals = sorted((indices[0], indices[-1], name) for name, indices in usage.items())
    for first, last, name in intervals:
        size = sizes[name]
        available = [
            (max(slot["size"], size) - slot["size"], index)
            for index, slot in enumerate(slots)
            if slot["last"] < first
        ]
        if available:
            _, slot_index = min(available)
            slots[slot_index]["size"] = max(slots[slot_index]["size"], size)
            slots[slot_index]["last"] = last
        else:
            slot_index = len(slots)
            slots.append({"size": size, "last": last})
        rename[name] = f"arena_{slot_index}"

    first_decl = declarations[0]
    indent = first_decl.group("indent")
    arena_decls = "\n".join(
        f"{indent}const arena_{index} = createEmptyBuf(device, {slot['size']});"
        for index, slot in enumerate(slots)
    )
    declaration_ranges = {(match.start(), match.end()) for match in declarations}
    pieces: list[str] = []
    cursor = 0
    inserted = False
    for start, end in sorted(declaration_ranges):
        pieces.append(source[cursor:start])
        if not inserted:
            pieces.append(arena_decls)
            inserted = True
        cursor = end
    pieces.append(source[cursor:])
    output = "".join(pieces)

    def replace_pass(match: re.Match) -> str:
        args = [arg.strip() for arg in match.group("args").split(",") if arg.strip()]
        mapped = ", ".join(rename.get(arg, arg) for arg in args)
        return match.group("prefix") + mapped + match.group("suffix")

    output, changed_passes = PASS_RE.subn(replace_pass, output)
    if changed_passes != len(passes):
        raise AssertionError("compute-pass rewrite count changed unexpectedly")
    leftovers = [name for name in sizes if re.search(rf"\b{re.escape(name)}\b", output)]
    if leftovers:
        raise AssertionError(f"temporary names survived rewrite: {leftovers}")

    stats = {
        "logical_count": len(sizes),
        "logical_bytes": sum(sizes.values()),
        "arena_count": len(slots),
        "arena_bytes": sum(slot["size"] for slot in slots),
        "largest_arena": max(slot["size"] for slot in slots),
        "pass_count": len(passes),
    }
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must differ; validate the candidate before replacing a runner")
    output, stats = rewrite(args.input.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)
    print(
        f"{args.input.name}: {stats['pass_count']} passes unchanged; "
        f"{stats['logical_count']} temporaries / {mib(stats['logical_bytes'])} -> "
        f"{stats['arena_count']} arenas / {mib(stats['arena_bytes'])}; "
        f"largest {mib(stats['largest_arena'])}"
    )
    print(args.output)


if __name__ == "__main__":
    main()
