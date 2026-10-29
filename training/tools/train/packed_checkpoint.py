"""Deterministic packed matrix recomputation; opt-in training execution."""
import ast

HELPER = '"""Prototype deterministic nonreentrant checkpoint around pure packed products."""\nimport torch as _tr_torch\nimport torch.utils.checkpoint as _tr_checkpoint_module\n\n\ndef _tr_checkpoint_call(function, args, kernel):\n    if not _tr_torch.is_grad_enabled() or not any(\n            isinstance(value, _tr_torch.Tensor) and value.requires_grad for value in args):\n        return function(*args)\n    first = [True]\n    counter = kernel[\'_SL_KZ_UNPACK_REMAINING\']\n    regions = dict(kernel[\'_KZ_OUT\'])\n\n    def run(*values):\n        initial = first[0]\n        outer_counter = kernel[\'_SL_KZ_UNPACK_REMAINING\']\n        outer_regions = dict(kernel[\'_KZ_OUT\'])\n        if not initial:\n            kernel[\'_SL_KZ_UNPACK_REMAINING\'] = counter\n            kernel[\'_KZ_OUT\'].update(regions)\n        try:\n            return function(*values)\n        finally:\n            if initial:\n                first[0] = False\n            else:\n                # Recompute must not consume later layers\' route/unpack state.\n                # This also runs for checkpoint\'s deliberate early-stop error.\n                kernel[\'_SL_KZ_UNPACK_REMAINING\'] = outer_counter\n                kernel[\'_KZ_OUT\'].update(outer_regions)\n\n    return _tr_checkpoint_module.checkpoint(run, *args, use_reentrant=False,\n                                            preserve_rng_state=False)\n'

def wrap_source(source):
    tree = ast.parse(source)
    replacements = []
    for node in list(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name in ('_kz_matmul', '_kz_r40_matmul'):
            name = node.name
            args = 'left, right, schemes' if name == '_kz_matmul' else 'left, right, kind, schemes'
            node.name = '_tr_original' + name
            wrapper = ast.parse('def ' + name + '(' + args + '):\n    return _tr_checkpoint_call(' + node.name + ', (' + args + '), globals())').body[0]
            tree.body.insert(tree.body.index(node) + 1, wrapper)
            replacements.append(name)
    if set(replacements) != {'_kz_matmul', '_kz_r40_matmul'} or len(replacements) != 2:
        raise ValueError('Packed checkpoint needs the two known packed product definitions')
    future = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == '__future__']
    tree.body = future + ast.parse(HELPER).body + [node for node in tree.body if node not in future]
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'
