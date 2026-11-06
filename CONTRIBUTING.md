# Contributing

Start with [the README](README.md) for this project's current scope, setup, and known limitations.

## Propose and implement a change

1. Describe the problem and expected result in an issue, or explain it in a focused pull request.
2. Work on a branch and keep unrelated changes separate.
3. Update documentation when setup, behavior, or requirements change. Preserve existing authorship, third-party notices, and licenses.

## Check your work

Check changed Python files with `python -m compileall -q` followed by their paths, using a fresh environment installed from the repository requirements. Exercise the affected behavior with local fixtures. Compilation alone does not verify external services.

Run `git diff --check` before committing. Include commands, results, and anything you could not verify in the pull request. Documentation-only changes need link and command review; do not describe unrun checks as passing.

## Keep commits portable

Leave local credentials, accounts, generated builds, dependencies, caches, and personal data out of commits. Use sanitized examples for logs and screenshots. Commit dependency lockfiles when the project uses them.

Bug reports should include the commit, operating system/toolchain, reproduction steps, expected behavior, and actual behavior. Do not put tokens, mailbox contents, health records, or other private data in public reports.
