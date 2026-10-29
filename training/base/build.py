"""Assemble the same scientific components for flopscope and torch.

Usage: python build.py --out <native-folder> [--torch-out <torch-kernel.py>]
The build reads only this package. L5 and extraction tools are not dependencies.
"""
import argparse
import ast
import builtins
import hashlib
import json
import shutil
import runpy
import symtable
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE / 'whest_est'
COMPONENTS = ('registry', 'params', 'config', 'state', 'array_ops', 'selection', 'moments', 'cumulants4', 'history', 'cumulants3', 'corrector', 'features', 'linalg', 'initialize', 'stages', 'runtime_api', 'estimator')
ASSETS = ('corrector.npz', 'fe31_inputs.npz', 'hadamard.npz', 'k21_v2e.npz', 'joint.npz')
OPTIONAL_ASSETS = ('linear.npz', 'feature_weights.npz')
OPTIONAL_DOCUMENTS = ('RUNTIME_CONFIG.json',)
LICENSE_DOCUMENTS = ('NOTICE.txt', 'LICENSE_CeCILL-B.txt', 'PLINOPT_AUTHORS.txt')  # terms of the kernel's derived circuits


def parameter_literals(package=HERE):
    registry = runpy.run_path(str(Path(package) / 'whest_est/registry.py'))
    return registry['parameter_defaults'](), registry['parameter_space']()


def update_parameter_defaults(package, updates):
    """Host-side export: validate the whole request, then edit registry defaults."""
    path = Path(package) / 'whest_est/registry.py'
    registry = runpy.run_path(str(path))
    params = registry['parameter_defaults']()
    changes = []
    for name, value in updates.items():
        if name not in params:
            raise ValueError('Unknown PARAMS key: ' + name)
        value = registry['normalize_parameter'](name, value)
        try:
            ast.literal_eval(repr(value))
        except (ValueError, SyntaxError) as error:
            raise ValueError('Parameter defaults require Python literals: ' + name) from error
        registry['validate_parameter'](name, value)
        if params[name] != value:
            changes.append(dict(key=name, old=params[name], new=value))
    if changes:
        source = path.read_text()
        tree = ast.parse(source)
        table = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'PARAMETER_REGISTRY' for t in n.targets))
        defaults = {ast.literal_eval(key): value.args[0] for key, value in zip(table.keys, table.values)}
        lines = source.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line.encode()))
        replacements = []
        for change in changes:
            node = defaults[change['key']]
            replacements.append((offsets[node.lineno - 1] + node.col_offset,
                                 offsets[node.end_lineno - 1] + node.end_col_offset,
                                 repr(change['new']).encode()))
        encoded = source.encode()
        for start, end, value in sorted(replacements, reverse=True):
            encoded = encoded[:start] + value + encoded[end:]
        compile(encoded, str(path), 'exec')
        path.write_bytes(encoded)
    return changes


def component_tree():
    imports, definitions, seen = [], [], set()
    for name in COMPONENTS:
        path = PACKAGE / (name + '.py')
        source = path.read_text()
        compile(source, str(path), 'exec')
        tree = ast.parse(source, filename=str(path))
        if name == 'params':
            values = dict(zip(('PARAMS', 'PARAM_SPACE'), parameter_literals()))
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    key = node.targets[0].id
                    if key in values:
                        node.value = ast.parse(repr(values[key]), mode='eval').body
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level and any(alias.asname for alias in node.names):
                raise ValueError('Unsupported relative import alias in ' + str(path))
        class BindImports(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None if node.level else node

        tree = BindImports().visit(tree)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.level:
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                key = ast.dump(node)
                if key not in seen:
                    imports.append(node)
                    seen.add(key)
            elif not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                definitions.append(node)
    imports.sort(key=lambda n: 0 if isinstance(n, ast.ImportFrom) and n.module == '__future__' else 1)
    return ast.fix_missing_locations(ast.Module(body=imports + definitions, type_ignores=[]))


class TorchMethods(ast.NodeTransformer):
    expected = ('zero_diagonal_owned', 'with_diagonal_owned', '_fast_matmul', 'reverse_vector')

    def __init__(self):
        self.matches = dict.fromkeys(self.expected, 0)
        self.scope = None

    def continuous_skip_gain(self, node):
        """Only zero-safe algebraic gains; shape/routing predicates stay literal."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'scalar':
            return self.continuous_skip_gain(node.args[0])
        if isinstance(node, ast.Name):
            if node.id in ('pair_ck3', 'cov_k3sq', 'cov_k3cross', 'split_pair_c3_cov', 'cov_adt3', '_r_k3_level', '_feed_b1', '_feed_b2', 'k31_beta'):
                return True
            if node.id in ('ratio_22c', 'ratio_31c', 'ratio_22g'):
                return self.scope in ('store_k4_vertices', 'reduce_k4_stars')
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'state':
            return node.attr in ('dress_rate', 'pair_ck3', 'cov_k3sq', 'cov_k3cross', 'cov_adt3', 't2_k3_scale', 'leg_level_gain', 'k4_22_dress', 'identity_share')
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == 'PARAMS':
                return isinstance(node.slice, ast.Constant) and node.slice.value == 'k3_carrier_mu_var'
            if isinstance(node.value, ast.Attribute) and node.value.attr == 'k4_family_ratios':
                return self.scope == 'reduce_k4_stars'
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'PARAMS' and node.func.attr == 'get':
            key = node.args[0]
            return (isinstance(key, ast.Constant) and key.value == 'edgeworth_34'
                    or isinstance(key, ast.Name) and key.id == 'flag' and self.scope in ('terminal_edgeworth', 'edgeworth_second_order'))
        return False

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.Eq, ast.NotEq)) and len(node.comparators) == 1:
            right = node.comparators[0]
            if isinstance(right, ast.Constant) and right.value == 0 and self.continuous_skip_gain(node.left):
                enabled = ast.Call(func=ast.Attribute(value=ast.Name(id='fnp', ctx=ast.Load()), attr='enabled', ctx=ast.Load()), args=[node.left], keywords=[])
                return ast.copy_location(ast.UnaryOp(op=ast.Not(), operand=enabled) if isinstance(node.ops[0], ast.Eq) else enabled, node)
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        # Request diets may hoist a cast into a bound method before calling it.
        # Lower both direct calls and aliases such as cast = tensor.astype.
        if node.attr == 'astype':
            node.attr = 'to'
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'float':
            node.func = ast.Attribute(value=ast.Name(id='fnp', ctx=ast.Load()), attr='scalar', ctx=ast.Load())
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'astype':
                node.func.attr = 'to'
            elif node.func.attr == 'copy' and isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Attribute) and node.func.value.func.attr == 'diag':
                node.func.attr = 'clone'
            elif node.func.attr == 'transpose' and not (isinstance(node.func.value, ast.Name) and node.func.value.id == 'fnp'):
                node.args.insert(0, node.func.value)
                node.func = ast.Attribute(value=ast.Name(id='fnp', ctx=ast.Load()), attr='transpose', ctx=ast.Load())
        if isinstance(node.func, ast.Name) and node.func.id == 'str' and len(node.args) == 1:
            arg = node.args[0]
            if isinstance(arg, ast.Attribute) and arg.attr == 'dtype':
                node.func = ast.Attribute(value=ast.Name(id='fnp', ctx=ast.Load()), attr='dtype_name', ctx=ast.Load())
        return node

    def visit_FunctionDef(self, node):
        previous, self.scope = self.scope, node.name
        node = self.generic_visit(node)
        self.scope = previous
        if node.name in self.matches:
            self.matches[node.name] += 1
        if node.name == 'reverse_vector':
            node.body = ast.parse('return fnp.flip(vector, axis=0)').body
        # Autograd needs an owned result before a diagonal write. This is
        # backend plumbing; covariance and moment formulas stay in components.
        if node.name in ('zero_diagonal_owned', 'with_diagonal_owned'):
            node.body.insert(0, ast.parse('matrix = matrix.clone()').body[0])
        # The native kernel writes exact fast-matmul sums into owned blocked
        # buffers in place. Torch evaluates the same product directly (float
        # roundoff only), so autograd sees one plain matmul.
        if node.name == '_fast_matmul':
            node.body = ast.parse('return left @ right').body
        return node


def source(target='native'):
    tree = component_tree()
    if target == 'torch':
        tree.body = [n for n in tree.body if not (
            isinstance(n, ast.Import) and any(a.name.startswith('flopscope') for a in n.names)
            or isinstance(n, ast.ImportFrom) and (n.module or '').startswith(('flopscope', 'whestbench')))]
        adapter = ast.parse('import torch_api as fnp\nfrom torch_api import flops, BaseEstimator, MLP, SetupContext').body
        future = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == '__future__']
        tree.body = future + adapter + [n for n in tree.body if n not in future]
        methods = TorchMethods()
        tree = methods.visit(tree)
        if any(count != 1 for count in methods.matches.values()):
            raise ValueError('Torch transforms must match exactly once: ' + repr(methods.matches))
    tree = ast.fix_missing_locations(tree)
    compile(tree, '<M1-' + target + '>', 'exec')
    return '# Generated from whest_est by build.py. Edit components and rebuild.\n' + ast.unparse(tree) + '\n'


def lint(check_assets=True):
    check_registry = runpy.run_path(str(HERE / 'registry_check.py'))['check_registry']
    check_registry(HERE,check_assets=check_assets)
    tree = component_tree()
    compile(tree, '<M1>', 'exec')
    names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    if len(names) != len(set(names)):
        raise ValueError('duplicate component definitions')
    for name in ('PARAMS', 'PARAM_SPACE'):
        value = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        ast.literal_eval(value)
    table = symtable.symtable(ast.unparse(tree), '<R1>', 'exec')
    defined = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()} | set(dir(builtins))
    missing = set()
    def check_globals(scope):
        for symbol in scope.get_symbols():
            if symbol.is_global() and symbol.is_referenced() and symbol.get_name() not in defined:
                missing.add(symbol.get_name())
        for child in scope.get_children():
            check_globals(child)
    check_globals(table)
    if missing:
        raise ValueError('undeclared component dependencies: ' + ', '.join(sorted(missing)))
    if check_assets and set(ASSETS) - {p.name for p in HERE.glob('*.npz')}:
        raise ValueError('missing model asset')
    if any(not (HERE / name).is_file() for name in LICENSE_DOCUMENTS):
        raise ValueError('missing license document')
    print('Component syntax, dependencies, parameter/asset registries and literal audit: OK')


def build(out, torch_out=None):
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('Build output must be empty: ' + str(out))
    lint()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'estimator.py').write_text(source())
    for name in ASSETS + LICENSE_DOCUMENTS + tuple(name for name in OPTIONAL_ASSETS + OPTIONAL_DOCUMENTS if (HERE / name).exists()):
        if name == 'hadamard.npz':
            import numpy as np
            with np.load(HERE / name, allow_pickle=False) as arrays:
                np.savez(out / name, production=arrays['production'])
        else:
            shutil.copy2(HERE / name, out / name)
    if torch_out is not None:
        torch_out.parent.mkdir(parents=True, exist_ok=True)
        torch_out.write_text(source('torch'))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--torch-out', type=Path)
    parser.add_argument('--lint', action='store_true')
    parser.add_argument('--no-assets',action='store_true',help='Asset-independent source lint (with --lint only)')
    args = parser.parse_args()
    if args.lint:
        lint(check_assets=not args.no_assets)
    elif args.out:
        if args.no_assets:
            parser.error('--no-assets is only supported with --lint')
        print(json.dumps(build(args.out, args.torch_out), indent=2))
    else:
        parser.error('--out or --lint is required')
