# Rebase SOP

## Goal

Keep `main` aligned with `upstream/master` without rewriting or polluting the public work branches.

## Rules

- Never merge into `master`.
- Never push to `upstream`.
- Branch feature work from `main`.
- Rebase `main` onto `upstream/master` when syncing the fork.

## Standard Flow

### 1. Refresh remotes

```bash
git fetch upstream
git fetch origin
```

### 2. Update local `master` from upstream

```bash
git checkout master
git merge --ff-only upstream/master
```

If `master` cannot fast-forward cleanly, stop and inspect the divergence before proceeding.

### 3. Rebase `main` onto the refreshed `master`

```bash
git checkout main
git rebase master
```

Resolve conflicts in the worktree, then continue:

```bash
git rebase --continue
```

### 4. Verify the branch state

```bash
git status --short
git log --oneline --decorate -n 10
git rev-list --left-right --count upstream/master...main
```

Target:

- `master` should match `upstream/master`
- `main` should contain only the intended fork-local work

### 5. Recreate feature branches from `main`

```bash
git checkout main
git checkout -b feature/my-new-work
```

## When To Stop

Stop and ask for human review if:

- the rebase introduces widespread conflict
- a branch unexpectedly rewrites published history
- `master` diverges from `upstream/master` in a way that is not a fast-forward
- a feature branch depends on unrecovered local work in `/tmp` or another ephemeral directory

## Notes

- This SOP is the canonical branch-sync workflow for this repo.
- If a task needs a one-off exception, document it in a handover file before proceeding.
