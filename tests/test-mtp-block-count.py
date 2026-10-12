#!/usr/bin/env python3
"""Regression test for the MTP/NextN block-count fix (conversion/qwen.py).

`_QwenMtpMixin` used to extend `block_count` by whatever `mtp_num_hidden_layers`
the config DECLARED, regardless of whether the checkpoint actually shipped any
`mtp.*` tensors. Releases that declare the field but export no MTP weights
(MiniCPM-V 4.7 35B-A3B) came out with `block_count = n_layer + 1` and
`nextn_predict_layers = 1` while containing only `blk.0..n_layer-1`, so every
derived artifact hard-failed at load with
`missing tensor 'blk.N.attn_norm.weight'`.

This test exercises the pure resolver `_resolve_mtp_layer_count` against the
four real-world shapes plus two extras. It deliberately avoids importing
`conversion.qwen` (torch-heavy) and instead extracts the function source, so it
runs anywhere plain python3 is available.

Run: python3 tests/test-mtp-block-count.py
"""

import re
import sys
from pathlib import Path

QWEN_PY = Path(__file__).resolve().parent.parent / "conversion" / "qwen.py"


def load_resolver():
    text = QWEN_PY.read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^def _resolve_mtp_layer_count\(.*?\n(?=\n\nclass |\nclass )", text
    )
    if match is None:
        raise AssertionError("_resolve_mtp_layer_count not found in conversion/qwen.py")
    namespace = {}
    exec(match.group(0), namespace)  # noqa: S102 - trusted in-repo source
    return namespace["_resolve_mtp_layer_count"], text


CASES = [
    # name, declared, observed_layers, observed_tensors, expected
    # MiniCPM-V 4.7 35B-A3B: declares an MTP head, ships none. The reported bug.
    ("declared-but-absent (MiniCPM-V 4.7)", 1, 0, 0, 0),
    # Qwen3.8-27B official: declares AND exports nextn tensors. Must NOT be patched --
    # blindly patching this shape corrupted real artifacts previously.
    ("declared-and-present (Qwen3.8-27B)", 1, 1, 15, 1),
    # Qwen3-Next: config omits the field; count derives from observed layer indices.
    ("undeclared-but-present (Qwen3-Next)", 0, 1, 12, 1),
    # Only the non-layered head tensors (fc/enorm/hnorm/norm), no mtp.layers.<i>.
    ("head-tensors-only", 0, 0, 4, 1),
    # No MTP anywhere: plain model.
    ("no-mtp-anywhere", 0, 0, 0, 0),
    # A two-layer head that is both declared and present.
    ("multi-layer-head", 2, 2, 30, 2),
]


def main() -> int:
    resolve, text = load_resolver()

    failures = 0
    print("=== _resolve_mtp_layer_count(declared, observed_layers, observed_tensors) ===\n")
    for name, declared, layers, tensors, expected in CASES:
        got = resolve(declared, layers, tensors)
        ok = got == expected
        failures += 0 if ok else 1
        print(
            "  %-38s declared=%d layers=%d tensors=%-3d -> %d (expect %d) %s"
            % (name, declared, layers, tensors, got, expected, "PASS" if ok else "*** FAIL ***")
        )

    print("\n=== source guards ===")
    checks = [
        ("brittle `assert opt_num_mtp_layers != 0` removed",
         "assert self.opt_num_mtp_layers != 0" not in text),
        ("resolver called from __init__",
         "_resolve_mtp_layer_count(declared" in text),
        ("observed counter reset per conversion",
         "type(self)._observed_mtp_tensors = 0" in text),
        ("observed counter incremented on kept mtp tensor",
         "cls._observed_mtp_tensors += 1" in text),
        ("declaration/weight mismatch is warned about",
         "no mtp.* tensors" in text and "logger.warning(" in text),
        ("logger is imported",
         re.search(r"^from \.base import .*logger", text, re.M) is not None),
        ("nextn key derives from block_count delta",
         'self.block_count - self.hparams["num_hidden_layers"]' in text),
    ]
    for label, ok in checks:
        failures += 0 if ok else 1
        print("  %-46s %s" % (label + ":", "PASS" if ok else "*** FAIL ***"))

    print()
    if failures:
        print("RESULT: %d check(s) FAILED" % failures)
        return 1
    print("RESULT: all %d cases + %d guards PASS" % (len(CASES), len(checks)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
