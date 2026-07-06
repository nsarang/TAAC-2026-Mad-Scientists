"""Phase 2: Name collision detection and scope-aware rename mapping."""

from __future__ import annotations

import ast
import random
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExternalImport:
    """An external import statement to preserve in the bundle."""

    module: str
    name: str | None
    alias: str | None
    is_from: bool


@dataclass
class CollisionRename:
    """A top-level name collision resolved by renaming."""

    file_path: Path
    original_name: str
    new_name: str


@dataclass
class AnalysisResult:
    """Output of Phase 2 analysis."""

    renames: list[CollisionRename] = field(default_factory=list)
    external_imports: list[ExternalImport] = field(default_factory=list)
    local_import_modules: dict[Path, set[str]] = field(default_factory=dict)
    rename_map: dict[Path, dict[str, str]] = field(default_factory=dict)


def collect_top_level_names(source: str) -> set[str]:
    """Collect names defined at the top level of a module.

    Includes function defs, class defs, and assignments. Excludes
    imported names and names nested inside functions/classes.
    """
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_extract_assign_names(target))
        elif isinstance(node, ast.AnnAssign) and node.target:
            names.update(_extract_assign_names(node.target))
        elif isinstance(node, ast.AugAssign):
            names.update(_extract_assign_names(node.target))
    return names


def _extract_assign_names(node: ast.expr) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    elif isinstance(node, (ast.Tuple, ast.List)):
        result: list[str] = []
        for elt in node.elts:
            result.extend(_extract_assign_names(elt))
        return result
    return []


def _random_suffix(length: int = 3) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


def _classify_imports(
    source: str,
    local_modules: set[str],
) -> tuple[list[ExternalImport], set[str]]:
    """Classify top-level imports as external or local."""
    tree = ast.parse(source)
    externals: list[ExternalImport] = []
    locals_found: set[str] = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in local_modules:
                    locals_found.add(top)
                else:
                    externals.append(
                        ExternalImport(
                            module=alias.name,
                            name=None,
                            alias=alias.asname,
                            is_from=False,
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                locals_found.add(f"__relative_level{node.level}__")
                continue
            module = node.module or ""
            top = module.split(".")[0]
            if top in local_modules:
                locals_found.add(top)
            else:
                externals.extend(
                    ExternalImport(
                        module=module,
                        name=alias.name,
                        alias=alias.asname,
                        is_from=True,
                    )
                    for alias in node.names
                )

    return externals, locals_found


def _is_installed_package(name: str) -> bool:
    """Check if `name` is importable as an installed package (not local)."""
    if name in sys.stdlib_module_names:
        return True
    from importlib.util import find_spec

    try:
        spec = find_spec(name)
    except (ModuleNotFoundError, ValueError):
        return False
    return spec is not None


def validate_no_dunder_file(modules: list[tuple[Path, str]]) -> None:
    """Raise if any module references ``__file__``."""
    for file_path, source in modules:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "__file__":
                raise ValueError(
                    f"{file_path}:{node.lineno}: bundling not supported — "
                    f"module references __file__"
                )


def compute_local_module_names(modules: list[tuple[Path, str]]) -> set[str]:
    """Build the set of names that identify local modules/packages."""
    all_paths = {p for p, _ in modules}
    names: set[str] = set()
    for p in all_paths:
        stem = p.stem
        if not _is_installed_package(stem):
            names.add(stem)
        parent = p.parent
        while (parent / "__init__.py").exists():
            names.add(parent.name)
            parent = parent.parent
    return names


def compute_module_name(file_path: Path, root: Path) -> str:
    """Compute the dotted module name from a file path and project root."""
    try:
        rel = file_path.relative_to(root)
    except ValueError:
        return file_path.stem
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else file_path.stem


def collect_exported_names(source: str, local_modules: set[str]) -> set[str]:
    """Collect names a module would export: definitions + local re-exports.

    Unlike `collect_top_level_names`, this includes names brought into the
    module namespace by imports from other local modules (re-exports), and
    descends into top-level ``try``/``if`` blocks where conditional definitions
    are common (e.g. ``try: from _cext import X; except: def X(): ...``).
    """
    tree = ast.parse(source)
    names: set[str] = set()
    _collect_exported_from_body(tree.body, names, local_modules)
    return names


def _collect_exported_from_body(
    body: list[ast.stmt],
    names: set[str],
    local_modules: set[str],
) -> None:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_extract_assign_names(target))
        elif isinstance(node, ast.AnnAssign) and node.target:
            names.update(_extract_assign_names(node.target))
        elif isinstance(node, ast.AugAssign):
            names.update(_extract_assign_names(node.target))
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                for alias in node.names:
                    if alias.name != "*":
                        names.add(alias.asname or alias.name)
            elif node.module:
                top = node.module.split(".")[0]
                if top in local_modules:
                    for alias in node.names:
                        if alias.name != "*":
                            names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in local_modules:
                    names.add(alias.asname or top)
        elif isinstance(node, ast.Try):
            _collect_exported_from_body(node.body, names, local_modules)
            for handler in node.handlers:
                _collect_exported_from_body(handler.body, names, local_modules)
            _collect_exported_from_body(node.orelse, names, local_modules)
            _collect_exported_from_body(node.finalbody, names, local_modules)
        elif isinstance(node, ast.If):
            _collect_exported_from_body(node.body, names, local_modules)
            _collect_exported_from_body(node.orelse, names, local_modules)


def analyze_modules(modules: list[tuple[Path, str]]) -> AnalysisResult:
    """Analyze a toposorted module list for collisions and imports."""
    result = AnalysisResult()

    local_module_names = compute_local_module_names(modules)

    per_module_names: dict[Path, set[str]] = {}
    seen_externals: set[tuple[str, str | None]] = set()

    for file_path, source in modules:
        per_module_names[file_path] = collect_top_level_names(source)

        ext_imports, local_found = _classify_imports(source, local_module_names)
        result.local_import_modules[file_path] = local_found

        for ext in ext_imports:
            key = (ext.module, ext.name)
            if key not in seen_externals:
                seen_externals.add(key)
                result.external_imports.append(ext)

    claimed: dict[str, Path] = {}
    all_new_names: set[str] = set()

    for file_path, _ in modules:
        names = per_module_names[file_path]
        for name in sorted(names):
            if name in claimed and claimed[name] != file_path:
                new_name = name
                while new_name in claimed or new_name in all_new_names or new_name in names:
                    new_name = f"{name}_{_random_suffix()}"
                rename = CollisionRename(
                    file_path=file_path,
                    original_name=name,
                    new_name=new_name,
                )
                result.renames.append(rename)
                all_new_names.add(new_name)
                if file_path not in result.rename_map:
                    result.rename_map[file_path] = {}
                result.rename_map[file_path][name] = new_name
            else:
                claimed[name] = file_path

    _propagate_renames_to_consumers(modules, result, local_module_names)

    return result


def _propagate_renames_to_consumers(
    modules: list[tuple[Path, str]],
    result: AnalysisResult,
    local_module_names: set[str],
) -> None:
    """Add rename entries for modules that import a renamed symbol via from-import."""
    if not result.renames:
        return

    renames_by_file: dict[Path, dict[str, str]] = {}
    for rename in result.renames:
        renames_by_file.setdefault(rename.file_path, {})[rename.original_name] = rename.new_name

    stem_to_paths: dict[str, list[Path]] = {}
    for p, _ in modules:
        stem_to_paths.setdefault(p.stem, []).append(p)

    for file_path, source in modules:
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level > 0 or not node.module:
                continue
            top = node.module.split(".")[0]
            if top not in local_module_names:
                continue
            target_stem = node.module.split(".")[-1]
            target_paths = stem_to_paths.get(target_stem, [])
            for alias in node.names:
                if alias.name == "*":
                    for tp in target_paths:
                        if tp not in renames_by_file:
                            continue
                        for orig, new in renames_by_file[tp].items():
                            _add_consumer_rename(file_path, orig, new, result)
                else:
                    imported_name = alias.name
                    local_name = alias.asname or imported_name
                    for tp in target_paths:
                        if tp not in renames_by_file:
                            continue
                        if imported_name in renames_by_file[tp]:
                            _add_consumer_rename(
                                file_path,
                                local_name,
                                renames_by_file[tp][imported_name],
                                result,
                            )


def _add_consumer_rename(
    file_path: Path,
    local_name: str,
    new_name: str,
    result: AnalysisResult,
) -> None:
    if file_path not in result.rename_map:
        result.rename_map[file_path] = {}
    if local_name not in result.rename_map[file_path]:
        result.rename_map[file_path][local_name] = new_name
