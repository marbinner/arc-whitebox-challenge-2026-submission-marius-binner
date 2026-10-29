"""Select trainables from runtime_schema, including future registered assets."""
import fnmatch

ALIASES = {'v2e': 'k21_v2e', 'fe31': 'fe31_inputs', 'PARAMS': 'params'}
DEFAULT_GROUPS = ('params', 'corrector', 'k21_v2e', 'fe31_inputs')


def registry_entries(schema):
    entries = {}
    for key, spec in schema['parameters'].items():
        entries['params.' + key] = dict(spec, group='params')
    for key, spec in schema['assets'].items():
        entries[key] = dict(spec, group=key.split('.', 1)[0])
    return entries


def add_joint(names=(), groups=(), owners=()):
    """Groups that add the joint/geometry heads to a selection (an empty selection means the default groups)."""
    groups = list(groups) if names or groups or owners else list(DEFAULT_GROUPS)
    return groups if 'joint' in groups else groups + ['joint']


def select(schema, *, names=(), groups=(), owners=(), default=True):
    entries = registry_entries(schema)
    eligible = {name for name, spec in entries.items()
                if spec['kind'] == 'continuous' and spec['trainable']}
    if not names and not groups and not owners and default:
        return sorted(name for name in eligible if entries[name]['group'] in DEFAULT_GROUPS)
    selected = set()
    for raw in names:
        name = raw.replace('PARAMS.', 'params.')
        if name in schema['parameters']:
            name = 'params.' + name
        matches = {key for key in entries if fnmatch.fnmatchcase(key, name)}
        if not matches:
            raise ValueError('Unknown registry name/pattern: ' + raw)
        if matches - eligible:
            raise ValueError('Frozen or discrete registry entries: ' + ', '.join(sorted(matches - eligible)))
        selected.update(matches)
    for raw in groups:
        group = ALIASES.get(raw, raw)
        if group not in {spec['group'] for spec in entries.values()} | {'assets', 'all'}:
            raise ValueError('Unknown registry group: ' + raw)
        selected.update(name for name in eligible if group == 'all' or
                        entries[name]['group'] == group or
                        group == 'assets' and entries[name]['group'] != 'params')
    for owner in owners:
        matches = {name for name, spec in entries.items() if
                   owner in [item.strip() for item in spec['owner'].split(',')]}
        if not matches:
            raise ValueError('Unknown registry owner: ' + owner)
        selected.update(matches & eligible)
    if not selected:
        raise ValueError('Selection contains no continuous trainables')
    return sorted(selected)
