"""Phase 3: CST transformation and output assembly using sys.meta_path finder.

# TODO (nsarang): ``__all__`` is not enforceable in the bundled output.
#
# ``from pkg import *`` through the synthetic module correctly respects
# ``__all__`` (``__getattr__`` returns the list, Python filters on it).
# But names excluded from ``__all__`` are still reachable as plain globals
# because every module's top-level code runs in a single shared namespace.
# For example, if ``bar.py`` defines ``SECRET`` and ``__init__.py`` sets
# ``__all__ = ["Foo", "Bar"]``, ``SECRET`` is a global after ``bar.py`` is
# inlined — before ``__init__.py`` even runs.  No transform on ``__init__``
# can fix this; the only real fix is scoping each module into its own dict
# (``exec`` per module) and only promoting exported names into globals.
# See ``test_init_all_restricts_star_import`` (xfail) for a repro.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import libcst as cst
from libcst import matchers as m
from libcst.metadata import GlobalScope, MetadataWrapper, ScopeProvider

from tools.bundler.analysis import (
    AnalysisResult,
    ExternalImport,
    collect_exported_names,
    compute_local_module_names,
    compute_module_name,
)

_PREAMBLE = textwrap.dedent("""\
    import importlib
    import importlib.abc
    import importlib.machinery
    import sys
    import types

    _BUNDLED: dict[str, types.ModuleType] = {}

    class _BundledModule(types.ModuleType):
        def __setattr__(self, name, value):
            g = self.__dict__.get('_globals')
            if g is not None and name in self.__dict__.get('_exports', {}):
                g[self.__dict__['_exports'][name]] = value
            super().__setattr__(name, value)

    class _BundledFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in _BUNDLED:
                return None
            is_pkg = getattr(_BUNDLED[fullname], '__path__', None) is not None
            return importlib.machinery.ModuleSpec(
                fullname, _BundledLoader(), is_package=is_pkg,
            )

    class _BundledLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return _BUNDLED.get(spec.name)

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _BundledFinder())
""")


def transform_module(
    source: str,
    file_path: Path,
    local_modules: set[str],
    rename_map: dict[str, str],
    is_entry: bool,
    root: Path | None = None,
) -> str:
    """Apply transformations to a single module's source.

    1. Rewrite relative imports to absolute (requires `root`).
    2. Strip local imports from non-entry modules (names already in globals).
    3. Strip ``from __future__`` imports (emitted once at top of bundle).
    4. Strip ``if __name__ == "__main__"`` block (unless entry).
    5. Apply scope-aware renames for collision resolution.
    """
    tree = cst.parse_module(source)

    if root is not None:
        module_name = compute_module_name(file_path, root)
        is_package = file_path.name == "__init__.py"
        tree = tree.visit(_RelativeToAbsoluteTransformer(module_name, is_package))

    stripper = _FutureAndMainStripper(is_entry)
    tree = tree.visit(stripper)

    if rename_map:
        wrapper = MetadataWrapper(tree)
        tree = wrapper.visit(_ScopeAwareRenamer(rename_map))

    return tree.code


def _resolve_relative_module(
    module_name: str, is_package: bool, level: int, relative_module: str | None
) -> str:
    """Compute the absolute module path for a relative import.

    Parameters
    ----------
    module_name
        Dotted name of the file containing the import (e.g. ``"pkg.sub"``).
    is_package
        True when the importing file is an ``__init__.py``.
    level
        Number of leading dots (1 = ``.``, 2 = ``..``).
    relative_module
        The module part after the dots, or None for bare ``from . import x``.
    """
    parts = module_name.split(".")
    if is_package:
        anchor = parts
    else:
        anchor = parts[:-1]
    steps_up = level - 1
    if steps_up > 0:
        anchor = anchor[:-steps_up] if steps_up < len(anchor) else []
    base = ".".join(anchor)
    if relative_module:
        return f"{base}.{relative_module}" if base else relative_module
    return base


def _dotted_name_to_cst(dotted: str) -> cst.BaseExpression:
    """Convert ``"a.b.c"`` to a libcst Attribute chain."""
    parts = dotted.split(".")
    node: cst.BaseExpression = cst.Name(parts[0])
    for part in parts[1:]:
        node = cst.Attribute(value=node, attr=cst.Name(part))
    return node


class _RelativeToAbsoluteTransformer(cst.CSTTransformer):
    """Rewrites relative imports (``from .X import Y``) to absolute form."""

    def __init__(self, module_name: str, is_package: bool) -> None:
        self.module_name = module_name
        self.is_package = is_package

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom:
        if isinstance(updated_node.relative, (list, tuple)) and len(updated_node.relative) > 0:
            level = len(updated_node.relative)
        else:
            return updated_node

        if updated_node.module is not None:
            relative_module = _get_attribute_str(updated_node.module)
        else:
            relative_module = None

        absolute = _resolve_relative_module(
            self.module_name, self.is_package, level, relative_module
        )
        if not absolute:
            return updated_node

        return updated_node.with_changes(
            relative=(),
            module=_dotted_name_to_cst(absolute),
        )


class _FutureAndMainStripper(cst.CSTTransformer):
    """Strips ``__future__`` imports and ``__name__`` guards from non-entry modules."""

    def __init__(self, is_entry: bool) -> None:
        self.is_entry = is_entry

    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine,
    ) -> cst.SimpleStatementLine | cst.RemovalSentinel:
        for stmt in updated_node.body:
            if self._is_future_import(stmt):
                return cst.RemovalSentinel.REMOVE
        return updated_node

    def leave_If(self, original_node: cst.If, updated_node: cst.If) -> cst.If | cst.RemovalSentinel:
        if self.is_entry:
            return updated_node
        if self._is_main_guard(updated_node):
            return cst.RemovalSentinel.REMOVE
        return updated_node

    def _is_future_import(self, node: cst.BaseSmallStatement) -> bool:
        if isinstance(node, cst.ImportFrom):
            if isinstance(node.module, cst.Name) and node.module.value == "__future__":
                return True
            if isinstance(node.module, cst.Attribute):
                module_str = _get_attribute_str(node.module)
                if module_str == "__future__":
                    return True
        return False

    def _is_main_guard(self, node: cst.If) -> bool:
        return m.matches(
            node,
            m.If(
                test=m.Comparison(
                    left=m.Name("__name__"),
                    comparisons=[
                        m.ComparisonTarget(
                            comparator=m.SimpleString() | m.ConcatenatedString(),
                        )
                    ],
                )
            ),
        )


class _ScopeAwareRenamer(cst.CSTTransformer):
    """Renames top-level names using ScopeProvider for accuracy."""

    METADATA_DEPENDENCIES = (ScopeProvider,)

    def __init__(self, rename_map: dict[str, str]) -> None:
        self.rename_map = rename_map

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        if updated_node.value not in self.rename_map:
            return updated_node
        try:
            scope = self.get_metadata(ScopeProvider, original_node)
        except KeyError:
            return updated_node
        for access in scope.accesses:
            if access.node is original_node:
                for referent in access.referents:
                    if isinstance(referent.scope, GlobalScope):
                        return updated_node.with_changes(value=self.rename_map[updated_node.value])
                return updated_node
        if isinstance(scope, GlobalScope):
            return updated_node.with_changes(value=self.rename_map[updated_node.value])
        return updated_node

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        if updated_node.name.value in self.rename_map:
            scope = self.get_metadata(ScopeProvider, original_node)
            if isinstance(scope, GlobalScope):
                return updated_node.with_changes(
                    name=updated_node.name.with_changes(
                        value=self.rename_map[updated_node.name.value]
                    )
                )
        return updated_node

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if updated_node.name.value in self.rename_map:
            scope = self.get_metadata(ScopeProvider, original_node)
            if isinstance(scope, GlobalScope):
                return updated_node.with_changes(
                    name=updated_node.name.with_changes(
                        value=self.rename_map[updated_node.name.value]
                    )
                )
        return updated_node


def _get_attribute_str(node: cst.BaseExpression) -> str:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return f"{_get_attribute_str(node.value)}.{node.attr.value}"
    return ""


def _format_import(ext: ExternalImport) -> str:
    if ext.is_from:
        name_part = ext.name
        if ext.alias:
            name_part = f"{ext.name} as {ext.alias}"
        return f"from {ext.module} import {name_part}"
    else:
        if ext.alias:
            return f"import {ext.module} as {ext.alias}"
        return f"import {ext.module}"


def _check_future_annotations(modules: list[tuple[Path, str]]) -> bool:
    """Return True if any module uses ``from __future__ import annotations``."""
    for _, source in modules:
        if "from __future__ import annotations" in source:
            return True
    return False


def _build_register_block(
    module_name: str,
    exported_names: set[str],
    rename_map: dict[str, str],
    package_names: list[str],
    is_package: bool = False,
) -> str:
    """Build the ``_BUNDLED[...] = ...`` registration block for a module.

    Parameters
    ----------
    module_name
        Dotted module name (e.g. ``"pkg.sub"``).
    exported_names
        Names defined in the module that should be set on the synthetic module.
    rename_map
        Collision renames applied to this module's source.
    package_names
        Intermediate package names to register (e.g. ``["pkg"]`` for ``"pkg.sub"``).
    is_package
        If True, set ``__path__`` on the module so Python treats it as a package.
    """
    lines: list[str] = []
    for pkg in package_names:
        lines.extend(
            [
                f"if {pkg!r} not in _BUNDLED:",
                f"    _pkg = _BundledModule({pkg!r})",
                "    _pkg.__path__ = []",
                f"    _BUNDLED[{pkg!r}] = _pkg",
            ]
        )
    safe = module_name.replace(".", "_")
    var = f"_mod_{safe}"
    exports = {name: rename_map.get(name, name) for name in sorted(exported_names)}
    lines.append(f"{var} = _BUNDLED.get({module_name!r}) or _BundledModule({module_name!r})")
    if is_package:
        lines.append(f"{var}.__path__ = []")
    lines.append(f"{var}.__dict__['_exports'] = {exports!r}")
    lines.append(f"{var}.__dict__['_globals'] = globals()")
    lines.append(f"def _getattr_{safe}(name):")
    lines.append(f"    e = {var}.__dict__['_exports']")
    lines.append(f"    if name in e: return {var}.__dict__['_globals'][e[name]]")
    lines.append("    raise AttributeError(name)")
    lines.append(f"{var}.__getattr__ = _getattr_{safe}")
    lines.append(f"_BUNDLED[{module_name!r}] = {var}")
    return "\n".join(lines)


def assemble_output(
    modules: list[tuple[Path, str]],
    analysis: AnalysisResult,
    entry_point: Path,
    root: Path,
) -> str:
    """Assemble the final bundled output from transformed modules.

    Parameters
    ----------
    modules
        Toposorted ``(path, source_code)`` pairs.
    analysis
        Analysis result containing external imports and rename info.
    entry_point
        The entry point file path.
    root
        Project root for computing relative paths in separators.
    """
    local_modules = compute_local_module_names(modules)

    transformed: list[tuple[Path, str]] = []
    for file_path, source in modules:
        rename_map = analysis.rename_map.get(file_path, {})
        is_entry = file_path.resolve() == entry_point.resolve()
        code = transform_module(
            source=source,
            file_path=file_path,
            local_modules=local_modules,
            rename_map=rename_map,
            is_entry=is_entry,
            root=root,
        )
        transformed.append((file_path, code))

    future_imports: list[ExternalImport] = []
    stdlib_imports: list[ExternalImport] = []
    thirdparty_imports: list[ExternalImport] = []
    for ext in analysis.external_imports:
        if ext.module == "__future__":
            future_imports.append(ext)
        elif ext.module.split(".")[0] in sys.stdlib_module_names:
            stdlib_imports.append(ext)
        else:
            thirdparty_imports.append(ext)

    future_imports.sort(key=lambda e: (e.module, e.name or ""))
    stdlib_imports.sort(key=lambda e: (e.module, e.name or ""))
    thirdparty_imports.sort(key=lambda e: (e.module, e.name or ""))

    has_future = _check_future_annotations(modules)
    seen_future_names = {ext.name for ext in future_imports}
    if has_future and "annotations" not in seen_future_names:
        future_imports.insert(
            0,
            ExternalImport(
                module="__future__",
                name="annotations",
                alias=None,
                is_from=True,
            ),
        )

    parts: list[str] = []

    parts.extend(_format_import(ext) for ext in future_imports)
    if future_imports:
        parts.append("")
    parts.extend(_format_import(ext) for ext in stdlib_imports)
    if stdlib_imports and thirdparty_imports:
        parts.append("")
    parts.extend(_format_import(ext) for ext in thirdparty_imports)
    if future_imports or stdlib_imports or thirdparty_imports:
        parts.append("")
        parts.append("")

    parts.append(_PREAMBLE)
    parts.append("")

    all_registered: set[str] = set()
    entry_code: str | None = None
    entry_rel: Path | None = None

    for file_path, code in transformed:
        is_entry = file_path.resolve() == entry_point.resolve()
        try:
            rel = file_path.relative_to(root)
        except ValueError:
            rel = file_path

        if is_entry:
            entry_code = code.strip()
            entry_rel = rel
            continue

        parts.append(f"# --- from {rel} ---")
        parts.append("")
        stripped = code.strip()
        if stripped:
            parts.append(stripped)
            parts.append("")

        module_name = compute_module_name(file_path, root)
        rename_map = analysis.rename_map.get(file_path, {})
        _, source = next((p, s) for p, s in modules if p == file_path)
        exported = collect_exported_names(source, local_modules)

        name_parts = module_name.split(".")
        package_names = [
            ".".join(name_parts[: i + 1])
            for i in range(len(name_parts) - 1)
            if ".".join(name_parts[: i + 1]) not in all_registered
        ]
        all_registered.update(package_names)
        all_registered.add(module_name)

        is_pkg = file_path.name == "__init__.py"
        reg_block = _build_register_block(module_name, exported, rename_map, package_names, is_pkg)
        parts.append(reg_block)
        parts.append("")
        parts.append("")

    if entry_code:
        parts.append(f"# --- from {entry_rel} (entry point) ---")
        parts.append("")
        parts.append(entry_code)
        parts.append("")

    return "\n".join(parts)
