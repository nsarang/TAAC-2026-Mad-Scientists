"""Phase 1: Dependency discovery and topological sorting."""

from __future__ import annotations

import ast
from graphlib import TopologicalSorter
from pathlib import Path


def infer_root(entry_point: Path) -> Path:
    """Walk up from `entry_point` to find the project root.

    The root is the directory where top-level package imports resolve.
    For flat directories (no package imports), returns the entry point's
    parent. For package imports like ``from core.data import X``, finds
    the ancestor directory that contains the top-level package.
    """
    source = entry_point.read_text()
    tree = ast.parse(source)

    package_prefixes: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            top = node.module.split(".")[0]
            package_prefixes.append(top)
        elif isinstance(node, ast.Import):
            package_prefixes.extend(alias.name.split(".")[0] for alias in node.names)

    candidate = entry_point.parent.resolve()
    for _ in range(20):
        for prefix in package_prefixes:
            as_file = candidate / f"{prefix}.py"
            as_pkg = candidate / prefix / "__init__.py"
            if as_file.exists() or as_pkg.exists():
                return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent

    return entry_point.parent.resolve()


def resolve_import(
    module: str | None,
    name: str | None,
    importing_file: Path,
    root: Path,
    level: int = 0,
) -> Path | None:
    """Resolve a single import to a local .py file path, or None if external.

    Parameters
    ----------
    module
        The module string (e.g., ``"foo.bar"``). None for relative imports
        like ``from . import foo``.
    name
        The imported name (e.g., ``"X"``). None for ``import foo``.
    importing_file
        The file containing this import statement.
    root
        The project root directory.
    level
        Relative import level (0 = absolute, 1 = ``.``, 2 = ``..``).
    """
    if level > 0:
        anchor = importing_file.parent
        for _ in range(level - 1):
            anchor = anchor.parent
        if module:
            parts = module.split(".")
            target_dir = anchor.joinpath(*parts)
        elif name:
            target_dir = anchor / name
        else:
            return None
        return _try_resolve_path(target_dir)

    if module is None:
        return None

    parts = module.split(".")
    target = root.joinpath(*parts)
    result = _try_resolve_path(target)
    if result is not None:
        return result

    if len(parts) > 1:
        parent_target = root.joinpath(*parts[:-1])
        result = _try_resolve_path(parent_target)
        if result is not None:
            return result

    if name and len(parts) == 1:
        submod = root / parts[0] / f"{name}.py"
        if submod.exists():
            return submod

    return None


def _try_resolve_path(target: Path) -> Path | None:
    """Try ``target.py`` then ``target/__init__.py``."""
    as_file = target.with_suffix(".py")
    if as_file.exists():
        return as_file.resolve()
    as_pkg = target / "__init__.py"
    if as_pkg.exists():
        return as_pkg.resolve()
    return None


def _extract_imports(source: str) -> list[tuple[str | None, str | None, int]]:
    """Extract all imports (any depth) as ``(module, name, level)`` tuples."""
    tree = ast.parse(source)
    imports: list[tuple[str | None, str | None, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, None, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or None
            level = node.level or 0
            if node.names:
                imports.extend((module, alias.name, level) for alias in node.names)
            else:
                imports.append((module, None, level))
    return imports


def resolve_dotted_path(dotted: str, root: Path) -> Path | None:
    """Resolve a dotted Python path (e.g. ``core.models.hyformer_v2.PCVRHyFormer``)
    to the .py file that defines it. Tries progressively shorter prefixes to handle
    the case where the last segment is a class/function name rather than a module.

    Returns None if no local file matches.
    """
    parts = dotted.split(".")
    # Try full path first, then drop trailing segments (class/function names)
    for end in range(len(parts), 0, -1):
        candidate = root.joinpath(*parts[:end])
        result = _try_resolve_path(candidate)
        if result is not None:
            return result
    return None


def discover_dependencies(
    entry_point: Path,
    include_all: bool = False,
    extra_entry_points: list[Path] = None,
) -> list[tuple[Path, str]]:
    """Discover all local dependencies from an entry point.

    Parameters
    ----------
    entry_point
        The main script to bundle.
    include_all
        If True, discover every ``.py`` file under the project root
        (and their transitive deps) instead of only what the entry
        point imports.
    extra_entry_points
        Additional .py files to seed the dependency walk from. Each
        is visited (with its transitive imports) alongside the main
        entry point. Use this to pull in dynamically-loaded modules
        that static analysis of the entry point can't reach.

    Returns a topologically sorted list of ``(file_path, source_code)``
    pairs, leaves first, entry point last.

    Raises ``graphlib.CycleError`` on circular imports.
    """
    entry_point = entry_point.resolve()
    if not entry_point.exists():
        raise FileNotFoundError(f"Entry point not found: {entry_point}")

    root = infer_root(entry_point)
    graph: dict[Path, set[Path]] = {}
    sources: dict[Path, str] = {}

    def _visit(file_path: Path) -> None:
        file_path = file_path.resolve()
        if file_path in sources:
            return
        source = file_path.read_text()
        sources[file_path] = source
        graph[file_path] = set()

        for module, name, level in _extract_imports(source):
            resolved = resolve_import(module, name, file_path, root, level)
            if resolved is not None and resolved != file_path:
                graph[file_path].add(resolved)
                _visit(resolved)

    _visit(entry_point)

    if extra_entry_points:
        for ep in extra_entry_points:
            ep = ep.resolve()
            if ep.exists():
                _visit(ep)

    if include_all:
        top_packages = set()
        for p in sources:
            try:
                rel = p.relative_to(root)
            except ValueError:
                continue
            if len(rel.parts) > 1:
                top_packages.add(rel.parts[0])
        for pkg in sorted(top_packages):
            pkg_dir = root / pkg
            if (pkg_dir / "__init__.py").exists():
                for py_file in sorted(pkg_dir.rglob("*.py")):
                    _visit(py_file)

    sorter = TopologicalSorter(graph)
    try:
        ordered = list(sorter.static_order())
    except Exception as exc:
        raise type(exc)(f"Circular import detected: {exc}") from exc

    return [(p, sources[p]) for p in ordered if p in sources]
