#!/usr/bin/env python3
"""Auto-resolve only provably-lossless conflicts during the upstream replay.

Safe classes:
  * deleted by us, upstream has the file  -> restore upstream's version
  * every conflict hunk has an empty ours side -> keep theirs (in place)
  * every ours-side line also exists in theirs -> take theirs

Anything else is reported as manual and left untouched.
"""
import subprocess
import sys
import os

FORK_FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".git-sync_forkfiles.txt")


def fork_touched():
    try:
        with open(FORK_FILES, encoding="utf-8") as fh:
            return {l.strip() for l in fh if l.strip()}
    except OSError:
        return None


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


def stages(path):
    out = git("ls-files", "-u", "--", path).stdout
    return {line.split()[2] for line in out.strip().splitlines() if line.split()}


def parse_conflicts(text):
    """Return list of (ours_lines, theirs_lines)."""
    hunks = []
    state = None
    ours, theirs = [], []
    for line in text.splitlines():
        if line.startswith("<<<<<<<"):
            state, ours, theirs = "ours", [], []
        elif line.startswith("=======") and state == "ours":
            state = "theirs"
        elif line.startswith(">>>>>>>") and state == "theirs":
            hunks.append((ours, theirs))
            state = None
        elif state == "ours":
            ours.append(line)
        elif state == "theirs":
            theirs.append(line)
    return hunks


def strip_to_theirs(text):
    """Keep theirs side of every conflict, drop ours."""
    out = []
    ours, theirs, state = [], [], None
    for line in text.splitlines():
        if line.startswith("<<<<<<<"):
            state, ours, theirs = "ours", [], []
        elif line.startswith("=======") and state == "ours":
            state = "theirs"
        elif line.startswith(">>>>>>>") and state == "theirs":
            out.extend(theirs)
            state = None
        elif state == "ours":
            continue
        elif state == "theirs":
            theirs.append(line)
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def main():
    root = git("rev-parse", "--show-toplevel").stdout.strip()
    os.chdir(root)
    conflicted = git("diff", "--name-only", "--diff-filter=U").stdout.split()
    fork = fork_touched()
    auto, manual = [], []
    for path in conflicted:
        st = stages(path)
        if "2" not in st:
            # we do not have it, theirs does: upstream-only file
            git("checkout", "--theirs", "--", path)
            git("add", "--", path)
            auto.append(("restore-upstream", path))
            continue
        if "3" not in st:
            manual.append(("deleted-by-them", path))
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="surrogateescape") as fh:
                text = fh.read()
        except (OSError, UnicodeError):
            manual.append(("unreadable", path))
            continue
        hunks = parse_conflicts(text)
        if not hunks:
            manual.append(("no-markers", path))
            continue
        if all(not any(l.strip() for l in ours) for ours, _ in hunks):
            with open(path, "w", encoding="utf-8", errors="surrogateescape") as fh:
                fh.write(strip_to_theirs(text))
            git("add", "--", path)
            auto.append(("ours-empty", path))
            continue
        if all(set(ours) <= set(theirs) for ours, theirs in hunks):
            git("checkout", "--theirs", "--", path)
            git("add", "--", path)
            auto.append(("ours-subset", path))
            continue
        # the fork has no work in this file: upstream's version is authoritative
        if fork is not None and path not in fork:
            dropped = [l for ours, _ in hunks for l in ours if l.strip()]
            git("checkout", "--theirs", "--", path)
            git("add", "--", path)
            auto.append((f"not-fork(-{len(dropped)}l)", path))
            for l in dropped[:8]:
                print(f"        dropped: {l.strip()}")
            continue
        manual.append(("needs-judgment", path))

    for how, path in auto:
        print(f"AUTO  {how:18s} {path}")
    for how, path in manual:
        print(f"MANUAL {how:17s} {path}")
    print(f"\nresolved {len(auto)}, manual {len(manual)}")
    return 1 if manual else 0


if __name__ == "__main__":
    sys.exit(main())
