"""Force-conflict tools: mark every difference as a conflict for manual review."""

import difflib
import hashlib
import subprocess
from pathlib import Path

import click


def _git_show(ref: str, file_path: str, repo_root: str | None = None) -> bytes | None:
    try:
        return subprocess.check_output(
            ["git", "show", f"{ref}:{file_path}"],
            cwd=repo_root,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        try:
            subprocess.check_output(
                ["git", "rev-parse", "--verify", ref],
                cwd=repo_root,
                text=True,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError:
            raise click.ClickException(f"'{ref}' is not a valid ref") from None
        return None


def _looks_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def _git_commit_info(ref: str, file_path: str, repo_root: str | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%ai|%s", ref, "--", file_path],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    if not out:
        return ""
    date, subject = out.split("|", 1)
    short_date = date[:16]
    short_subject = subject[:60] + ("..." if len(subject) > 60 else "")
    return f" | {short_date} | {short_subject}"


def _git_blob_sha(ref: str, file_path: str, repo_root: str | None = None) -> str | None:
    """Return the 40-char blob SHA for `ref:file_path`, or None if it doesn't exist."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{ref}:{file_path}"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError:
        return None


def _mark_unmerged(
    file_path: str,
    merge_base_ref: str,
    ours_ref: str,
    theirs_ref: str,
    repo_root: str | None = None,
) -> None:
    """Write stage 1/2/3 index entries so git treats `file_path` as unmerged."""
    base_sha = _git_blob_sha(merge_base_ref, file_path, repo_root)
    ours_sha = _git_blob_sha(ours_ref, file_path, repo_root)
    theirs_sha = _git_blob_sha(theirs_ref, file_path, repo_root)

    lines = []
    if base_sha:
        lines.append(f"100644 {base_sha} 1\t{file_path}\n")
    if ours_sha:
        lines.append(f"100644 {ours_sha} 2\t{file_path}\n")
    if theirs_sha:
        lines.append(f"100644 {theirs_sha} 3\t{file_path}\n")

    if not lines:
        return

    # First remove the stage-0 entry
    subprocess.run(
        ["git", "update-index", "--force-remove", "--", file_path],
        cwd=repo_root,
        check=True,
    )
    # Then write the unmerged stages
    subprocess.run(
        ["git", "update-index", "--index-info"],
        input="".join(lines),
        cwd=repo_root,
        text=True,
        check=True,
    )


def _file_hash(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def force_conflicts(
    ours_ref: str,
    theirs_ref: str,
    file_path: str,
    out_path: str,
    repo_root: str | None = None,
    force: bool = False,
) -> bool:
    """Write conflict markers for every differing hunk. Returns True if conflicts exist."""
    out = Path(out_path)
    initial_hash = _file_hash(out)

    ours_raw = _git_show(ours_ref, file_path, repo_root)
    theirs_raw = _git_show(theirs_ref, file_path, repo_root)

    if ours_raw is None and theirs_raw is None:
        raise click.ClickException(f"'{file_path}' does not exist at either ref")

    if _looks_binary(ours_raw or b"") or _looks_binary(theirs_raw or b""):
        raise click.ClickException(
            f"'{file_path}' appears to be binary — conflict markers not supported"
        )

    ours_text = (ours_raw or b"").decode()
    theirs_text = (theirs_raw or b"").decode()

    ours = ours_text.splitlines(keepends=True)
    theirs = theirs_text.splitlines(keepends=True)

    if ours and not ours[-1].endswith("\n"):
        ours[-1] += "\n"
    if theirs and not theirs[-1].endswith("\n"):
        theirs[-1] += "\n"

    if ours == theirs:
        out.write_text("".join(ours))
        return False

    ours_info = _git_commit_info(ours_ref, file_path, repo_root)
    theirs_info = _git_commit_info(theirs_ref, file_path, repo_root)

    result = []
    matcher = difflib.SequenceMatcher(None, ours, theirs)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            result.extend(ours[i1:i2])
        else:
            result.append(f"<<<<<<< {ours_ref}{ours_info}\n")
            result.extend(ours[i1:i2])
            result.append("=======\n")
            result.extend(theirs[j1:j2])
            result.append(f">>>>>>> {theirs_ref}{theirs_info}\n")

    if not force and initial_hash is not None:
        current_hash = _file_hash(out)
        if current_hash != initial_hash:
            raise click.ClickException(
                f"'{out_path}' changed on disk since we started — "
                f"refusing to overwrite. Re-run or pass --force."
            )

    out.write_text("".join(result))
    return True


def _find_repo_root() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError:
        raise click.ClickException("not inside a git repository") from None


@click.group()
def cli() -> None:
    """Force-conflict tools: mark every difference as a conflict."""


@cli.command()
@click.argument("ours_ref")
@click.argument("theirs_ref")
@click.argument("file_path")
@click.option(
    "-o", "--output", default=None, help="Output path (default: overwrite file in working tree)"
)
@click.option("--force", is_flag=True, help="Overwrite even if the file changed on disk")
def diff(ours_ref: str, theirs_ref: str, file_path: str, output: str | None, force: bool) -> None:
    """Force conflict markers on every differing hunk between OURS_REF and THEIRS_REF for FILE_PATH."""
    repo_root = _find_repo_root()
    if output is None:
        output = str(Path(repo_root) / file_path)
    has_conflicts = force_conflicts(
        ours_ref, theirs_ref, file_path, output, repo_root=repo_root, force=force
    )
    click.echo(
        f"Wrote {'conflicts' if has_conflicts else 'clean file (no differences)'} to {output}"
    )
    raise SystemExit(1 if has_conflicts else 0)


@cli.command()
@click.argument("theirs_ref")
@click.option("--force", is_flag=True, help="Overwrite even if files changed on disk")
def merge(theirs_ref: str, force: bool) -> None:
    """Merge THEIRS_REF with conflict markers on every file theirs changed.

    Runs git merge --no-commit --no-ff, then overwrites every file that
    theirs modified (relative to the merge base) with full conflict markers.
    Files only changed on our side are left as-is.
    """
    repo_root = _find_repo_root()

    merge_base = subprocess.check_output(
        ["git", "merge-base", "HEAD", theirs_ref],
        cwd=repo_root,
        text=True,
    ).strip()

    theirs_changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", merge_base, theirs_ref],
            cwd=repo_root,
            text=True,
        )
        .strip()
        .splitlines()
    )

    if not theirs_changed:
        click.echo("Nothing to merge — theirs has no changes relative to merge base.")
        return

    result = subprocess.run(
        ["git", "merge", "--no-commit", "--no-ff", theirs_ref],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise click.ClickException(f"git merge failed:\n{result.stderr}")

    all_differing = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", "ORIG_HEAD", theirs_ref],
            cwd=repo_root,
            text=True,
        )
        .strip()
        .splitlines()
    )

    conflicted = []
    skipped_binary = []
    skipped_ours_only = []
    identical = []

    for file_path in sorted(all_differing):
        if file_path not in theirs_changed:
            skipped_ours_only.append(file_path)
            continue

        ours_raw = _git_show("ORIG_HEAD", file_path, repo_root)
        theirs_raw = _git_show(theirs_ref, file_path, repo_root)

        if _looks_binary(ours_raw or b"") or _looks_binary(theirs_raw or b""):
            skipped_binary.append(file_path)
            continue

        out_path = str(Path(repo_root) / file_path)
        has_conflicts = force_conflicts(
            "ORIG_HEAD",
            theirs_ref,
            file_path,
            out_path,
            repo_root=repo_root,
            force=force,
        )
        _mark_unmerged(
            file_path=file_path,
            merge_base_ref=merge_base,
            ours_ref="ORIG_HEAD",
            theirs_ref=theirs_ref,
            repo_root=repo_root,
        )
        if has_conflicts:
            conflicted.append(file_path)
        else:
            identical.append(file_path)

    click.echo(f"\nForce-conflict merge from {theirs_ref}:")
    click.echo(f"  {len(conflicted)} files with conflict markers")
    if skipped_ours_only:
        click.echo(f"  {len(skipped_ours_only)} files skipped (only changed on our side)")
    if skipped_binary:
        click.echo(f"  {len(skipped_binary)} files skipped (binary)")
    if identical:
        click.echo(f"  {len(identical)} files identical between refs")

    if conflicted:
        click.echo("\nFiles with conflicts:")
        for f in conflicted:
            click.echo(f"  {f}")

    raise SystemExit(1 if conflicted else 0)


if __name__ == "__main__":
    cli()
