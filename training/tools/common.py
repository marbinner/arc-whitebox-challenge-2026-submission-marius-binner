"""Small, host-side plumbing shared by the base tools (never submission code)."""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import runpy
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
from queue import Queue
import threading
if __package__:
    from .guards import OwnedProcesses
    track = OwnedProcesses.track
    check_cancelled = OwnedProcesses.check_cancelled
    kill = OwnedProcesses.kill
    from .guards import Paths
    normalize_paths = Paths.normalize_paths
else:
    from guards import OwnedProcesses
    track = OwnedProcesses.track
    check_cancelled = OwnedProcesses.check_cancelled
    kill = OwnedProcesses.kill
    from guards import Paths
    normalize_paths = Paths.normalize_paths

HERE = Path(__file__).resolve().parent
BASE_ROOT = HERE.parent
SHARED_ROOT = Path(os.environ.get('WHEST_BASE_SHARED_ROOT', '.'))
_default_scratch = BASE_ROOT/'runs'
SCRATCH = Path(os.environ.get('WHEST_BASE_SCRATCH', str(_default_scratch))).expanduser()
ASSETS = Path(os.environ.get('WHEST_ASSETS', str(BASE_ROOT/'assets'))).expanduser()
CLIENT_PYTHON = Path(os.environ.get('WHEST_CLIENT_PY', 'python3'))
SERVER_BIN = Path(os.environ.get('WHEST_SERVER_BIN', 'flopscope-server'))
BUDGET = 2**41


def host_paths():
    """Discover execution capabilities only when a command needs them."""
    supplied = os.environ.get('WHEST_PY')
    roots = ([Path(os.environ['WHEST_PHASE2_ROOT'])] if os.environ.get('WHEST_PHASE2_ROOT') else []) + list(HERE.parents)
    if supplied:
        roots.extend(Path(supplied).expanduser().absolute().parents)
    roots.append(SHARED_ROOT)
    root = next((parent for parent in roots if (parent/'AGENT_BRIEF.md').is_file()
                 and (parent/'whest-starterkit/.venv/bin/python').is_file()), None)
    if root is None:
        raise RuntimeError('Command requires a phase2 repository and its pinned Python; set WHEST_PHASE2_ROOT/WHEST_PY')
    python = Path(supplied).expanduser().absolute() if supplied else root/'whest-starterkit/.venv/bin/python'
    if not python.is_file():
        raise RuntimeError('Pinned Python missing: ' + str(python))
    return root, python


def phase2_root():
    return host_paths()[0]


def python_path():
    return host_paths()[1]


def __getattr__(name):
    # Compatibility for external callers; tools use explicit lazy accessors.
    if name == 'ROOT':
        return phase2_root()
    if name == 'PYTHON':
        return python_path()
    raise AttributeError(name)
def core_allocation(text, default):
    if not text:
        return tuple(default)
    try:
        maximum=max(int(token.rsplit('-',1)[-1]) for token in
                    Path('/sys/devices/system/cpu/possible').read_text().strip().split(','))
    except (OSError,ValueError):
        maximum=max((os.cpu_count() or 1)-1,max(os.sched_getaffinity(0)),max(default,default=0))
    result = []
    for token in text.split(','):
        if '-' in token:
            lo, hi = map(int, token.split('-'))
            if lo < 0 or hi < lo:
                raise ValueError('Core ranges must be nonnegative and increasing')
            if hi>maximum:
                raise ValueError('Core index is outside this host CPU range')
            result.extend(range(lo, hi + 1))
        else:
            value=int(token)
            if value>maximum:
                raise ValueError('Core index is outside this host CPU range')
            result.append(value)
    if not result or any(c < 0 for c in result) or len(set(result)) != len(result):
        raise ValueError('Cores must be nonnegative and unique')
    return tuple(result)


# Affinity is the allocation unless an explicit resource mask narrows it.
ENTRY_CORES = frozenset(os.sched_getaffinity(0))
WORK_CORES = core_allocation(os.environ.get('WHEST_BASE_WORK_CORES') or os.environ.get('WHEST_WORK_CORES'),
                             sorted(ENTRY_CORES))
QUIET_CORES = core_allocation(os.environ.get('WHEST_BASE_QUIET_CORES'), sorted(ENTRY_CORES))


def environment():
    root, python = host_paths()
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    env.update(WHEST_BASE=str(BASE_ROOT), WHEST_BASE_SCRATCH=str(SCRATCH),
               WHEST_ASSETS=str(ASSETS), WHEST_PY=str(python), WHEST_PHASE2_ROOT=str(root),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    env.setdefault('WHEST_AGENT', 'unknown')
    env['PATH'] = str(python.parent) + os.pathsep + env.get('PATH','')
    env['PYTHONPATH'] = str(HERE/'ast_cache') + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    # Compiler/temporary outputs, if any, belong to data scratch too.
    temp = bulk_dir(SCRATCH / 'tmp')
    env.update(TMPDIR=str(temp), TORCHINDUCTOR_CACHE_DIR=str(temp / 'torchinductor'),
               CUDA_CACHE_PATH=str(temp / 'cuda'))
    return env


def client_environment():
    env = environment()
    env.pop('PYTHONPATH', None)
    env['FLOPSCOPE_MAX_ARRAY_BYTES'] = str(4*1024**3)
    return env


def bulk_dir(path):
    path = Path(path).expanduser().resolve()
    if path == Path('/tmp') or Path('/tmp') in path.parents:
        raise ValueError('Bulk outputs must be outside /tmp; use /data scratch: ' + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(label):
    return bulk_dir(SCRATCH / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + label + '-' + uuid.uuid4().hex[:8]))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_nets(spec):
    """Full rows below 1000; d8b rows are explicitly namespaced or >=1000."""
    aliases = {'SCREEN8': '832:840', 'CENSUS4': '832:836', 'VAL': '832:864', 'VAL32': '832:864', 'VAL16': '832:848', 'TEST': '896:960', 'TEST64': '896:960', 'D8B1': 'd8b:3072:4096',
               'D8B2': 'd8b:13048:14048'}
    result,seen = [],set()
    for token in spec.split(','):
        token = aliases.get(token.strip().upper(), token.strip())
        d8b = token.lower().startswith('d8b:')
        if d8b:
            token = token[4:]
        pieces = token.split(':')
        try:
            numbers=list(map(int,pieces))
        except ValueError:
            raise ValueError('Invalid panel token '+repr(token)+': expected an integer index, start:stop, SCREEN8, CENSUS4, VAL, VAL16, VAL32, TEST, TEST64, D8B1 or D8B2') from None
        if len(numbers) == 1:
            values = numbers
        elif len(numbers) == 2:
            lo,hi=numbers
            if hi <= lo:
                raise ValueError('Net ranges must be nonempty and half-open: '+token)
            values = range(lo, hi)
        else:
            raise ValueError('Expected index or start:stop: ' + token)
        for index in values:
            dataset = 'd8b' if d8b or index >= 1000 else 'full'
            if index < 0 or (dataset == 'd8b' and index >= 14048):
                raise ValueError('Network index outside the dataset: ' + str(index))
            key = (dataset, index)
            if key in seen:
                raise ValueError('Duplicate network: ' + str(key))
            seen.add(key)
            result.append(key)
    if not result:
        raise ValueError('Empty network specification')
    return result


def cores(spec=None, quiet=False):
    # Retain quiet=True for callers of the old library; wb uses work cores only.
    available = ENTRY_CORES if quiet else os.sched_getaffinity(0)
    allowed = set(QUIET_CORES if quiet else WORK_CORES) & available
    if spec is None:
        result = sorted(allowed)
    else:
        result = list(core_allocation(spec, ()))
        if not set(result) <= allowed or len(set(result)) != len(result):
            raise ValueError('Cores must be unique and inside the packet allocation')
    if not result:
        raise ValueError('No allocated cores available')
    return result


def physical_cores(allocation):
    """Keep one allocated sibling per physical CPU; preserve explicit job overrides."""
    representatives, seen = [], set()
    for core in allocation:
        try:
            group = tuple(int(value) for token in Path(f'/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list').read_text().strip().split(',')
                          for value in (range(int(token.split('-')[0]),int(token.split('-')[1])+1) if '-' in token else [int(token)]))
        except (OSError, ValueError):
            group = (core,)
        if group not in seen:
            seen.add(group); representatives.append(core)
    return representatives


def parallel_jobs(items, allocation, jobs, function):
    """Give each worker a stable core, even when earlier jobs finish unevenly."""
    physical = physical_cores(allocation)
    allocation = physical + [core for core in allocation if core not in physical]
    if jobs == 0:
        jobs = len(physical)
    if jobs < 1 or jobs > len(allocation):
        raise ValueError('Jobs must fit the allocated cores')
    items = list(items)
    workers = min(jobs, len(items))
    if not workers:
        return []
    available = Queue()
    for core in allocation[:workers]:
        available.put(core)
    state = threading.local()
    def assign():
        state.core = available.get_nowait()
    def perform(item):
        check_cancelled()
        return function(item, state.core)
    with ThreadPoolExecutor(max_workers=workers, initializer=assign) as pool:
        return list(pool.map(perform, items))


@contextmanager
def exclusive_lock(path, timeout=30):
    """Timed append lock; util-linux 2.32 requires integer timeout seconds."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Lock timeout must be finite and positive')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as lock:
        seconds = max(1, math.ceil(timeout))
        acquired = subprocess.run(['flock', '--exclusive', '--timeout', str(seconds), str(lock.fileno())],
                                  pass_fds=(lock.fileno(),), capture_output=True, text=True, timeout=seconds+5)
        if acquired.returncode == 1:
            raise TimeoutError('Append lock acquisition timed out: ' + str(path))
        if acquired.returncode:
            raise RuntimeError('Append lock failed: ' + acquired.stderr.strip())
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def resolve_reference(value):
    """Resolve FINAL/BEST/FALLBACK through the live hash-checked candidate pointer."""
    if str(value).upper() not in ('FINAL', 'BEST', 'FALLBACK'):
        return Path(value).expanduser().resolve()
    name = str(value).lower()
    override = os.environ.get('WHEST_BASE_' + name.upper())
    if override:
        return Path(override).expanduser().resolve()
    pointer = SHARED_ROOT / 'research/candidates' / (name + '.json')
    module = load_module(SHARED_ROOT / 'bin/whest_tools/py/_candidate.py', 'wb_candidate_pointer')
    return module.check_pointer(json.loads(pointer.read_text()))


def execute(command, *, log, timeout=1200, core=None, env=None, cwd=None):
    check_cancelled()
    command = [str(v) for v in command]
    if core is not None:
        command = ['taskset', '-c', str(core)] + command
    started = time.monotonic()
    with Path(log).open('w') as stream:
        process = subprocess.Popen(command, env=env or environment(), cwd=phase2_root() if cwd is None else cwd,
                                   stdout=stream, stderr=stream, start_new_session=True)
        with track(process):
            finished = False
            try:
                code = process.wait(timeout=timeout)
                finished = True
            except subprocess.TimeoutExpired:
                code = 124
            finally:
                if finished:
                    # A successful/exited leader may leave work alive in the
                    # session we own. Detached jobs use their own launcher.
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                else:
                    # This new session contains only the command and its children.
                    # Stop the group we started so shell/trainer descendants cannot
                    # continue after a timeout or an interrupted wait.
                    kill(process)
                    process.wait(timeout=5)
    return dict(command=command, returncode=code, elapsed_s=time.monotonic() - started, log=str(log))


def literal(path, name):
    tree = ast.parse(Path(path).read_text())
    values = [n.value for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
              and any(isinstance(t, ast.Name) and t.id == name for t in
                      (n.targets if isinstance(n, ast.Assign) else [n.target]))]
    if len(values) != 1:
        raise ValueError('Expected one literal ' + name + ' in ' + str(path))
    try:
        return ast.literal_eval(values[0])
    except ValueError:
        path = Path(path)
        builder = path.parents[1]/'build.py'
        if path.name != 'params.py' or name not in ('PARAMS', 'PARAM_SPACE') or not builder.is_file():
            raise
        api = runpy.run_path(str(builder))
        if 'parameter_literals' not in api:
            raise
        values = api['parameter_literals'](builder.parent)
        return values[0 if name == 'PARAMS' else 1]


def asset_map(pkg, assets=None):
    """Find only declared relocation receipts; never scan the shared repository."""
    pkg = Path(pkg).resolve()
    found = {p.name: p for p in pkg.glob('*.npz')}
    if assets:
        for path in Path(assets).resolve().glob('*.npz'):
            found.setdefault(path.name, path)
    marker = pkg.parent / 'BINARIES-MOVED.txt'
    if marker.exists():
        for line in marker.read_text().splitlines():
            if ' -> ' in line:
                left, right = line.split(' -> ', 1)
                relative = Path(left)
                target = Path(right.split(' (', 1)[0])
                if relative.parent == Path(pkg.name) and target.is_file():
                    found.setdefault(relative.name, target)
    return found


def identity(folder):
    folder = Path(folder)
    paths = sorted(p for p in folder.rglob('*') if p.is_file() and
                   p.suffix != '.pyc' and '__pycache__' not in p.parts)
    files = {str(p.relative_to(folder)): digest(p) for p in paths}
    h = hashlib.sha256()
    for name, sha in files.items():
        h.update(name.encode() + b'\0' + bytes.fromhex(sha))
    return dict(path=str(folder), package_sha256=h.hexdigest(), files=files)


def copy_snapshot(inventory, target):
    """Copy a stable set of source bytes; reject a changed input inventory."""
    sources=inventory()
    before={name:digest(path) for name,path in sources.items()}
    for name,path in sources.items():
        check_cancelled()
        destination=target/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    after=inventory()
    if set(after)!=set(sources):
        raise ValueError('Input files changed during snapshot copy')
    for name,path in after.items():
        if path!=sources[name] or digest(path)!=before[name] or digest(target/name)!=before[name]:
            raise ValueError('Input bytes changed during snapshot copy: '+name)


def empty_output(pkg,target):
    target=Path(target).resolve()
    if target==pkg or (pkg.is_dir() and pkg in target.parents):
        raise ValueError('Snapshot output must be outside the input package')
    target=bulk_dir(target)
    if any(target.iterdir()):
        raise ValueError('Snapshot output must be empty: '+str(target))
    return target


def stage_source(pkg, target, assets=None):
    pkg=Path(pkg).resolve()
    if not (pkg / 'build.py').is_file() or not (pkg / 'whest_est').is_dir():
        raise ValueError('Expected M1 layout: build.py and whest_est/: ' + str(pkg))
    target=empty_output(pkg,target)
    def inventory():
        files={str(path.relative_to(pkg)):path for path in
            sorted(set(pkg.rglob('*.py')) | set(pkg.rglob('*.json')) | set(pkg.glob('*.txt'))) if '__pycache__' not in path.parts}
        files.update(asset_map(pkg,ASSETS if assets is None else assets))
        return files
    copy_snapshot(inventory,target)
    return target


def build_package(pkg, out, assets=None):
    """Build in isolated data scratch; separate native lint scope from research."""
    pkg, out = Path(pkg).resolve(), Path(out).resolve()
    if pkg == out or pkg in out.parents:
        raise ValueError('Build output must be outside the input package')
    out = bulk_dir(out)
    if any(out.iterdir()):
        raise ValueError('Build output must be empty: ' + str(out))
    source = stage_source(pkg, out / 'source', assets)
    native, twin = bulk_dir(out / 'native'), bulk_dir(out / 'twin')
    result = execute([python_path(), source / 'build.py', '--out', native, '--torch-out', twin / 'torch_kernel.py'],
                     log=out / 'build.log', timeout=120, core=cores()[0])
    if result['returncode']:
        raise RuntimeError('Package build failed; see ' + result['log'])
    provenance=source/'TRAINING-PROVENANCE.json'
    if provenance.is_file():
        for destination in (native,twin):shutil.copy2(provenance,destination/provenance.name)
    # The public wrapper consumes a minimal build.source shim, pinned to the
    # emitted kernel rather than depending on the caller's original package.
    shutil.copy2(source / 'torch_api.py', twin / 'torch_api.py')
    wrapper = ast.parse((source / 'torch_twin.py').read_text())
    for node in ast.walk(wrapper):
        if isinstance(node, ast.FunctionDef):
            for position, (arg, default) in enumerate(zip(node.args.args[-len(node.args.defaults):], node.args.defaults)):
                if arg.arg == 'assets' and isinstance(default, ast.Name) and default.id == 'HERE':
                    node.args.defaults[position] = ast.parse('HERE.parent / "native"', mode='eval').body
        if isinstance(node, ast.FunctionDef) and node.name == 'dependency_hashes':
            api=runpy.run_path(str(source/'build.py'))
            names=(*api.get('ASSETS',()),*api.get('OPTIONAL_ASSETS',()),'RUNTIME_CONFIG.json')
            node.body = ast.parse(
                'paths = {"twin/"+p.name:p for p in HERE.glob("*.py")}\n'
                'paths.update({"assets/"+name:Path(assets)/name for name in '+repr(names)+'})\n'
                'return {name:hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None '
                'for name,path in sorted(paths.items())}\n').body
    ast.fix_missing_locations(wrapper)
    (twin / 'torch_twin.py').write_text(ast.unparse(wrapper) + '\n')
    (twin / 'build.py').write_text(
        'from pathlib import Path\nHERE = Path(__file__).resolve().parent\n'
        'def source(target="torch"):\n'
        '    if target != "torch": raise ValueError("Torch-only build")\n'
        '    return (HERE / "torch_kernel.py").read_text()\n')
    receipt = dict(schema='base-build-v1', input=str(Path(pkg).resolve()),
                   source=identity(source), native=identity(native), twin=identity(twin), execution=result)
    write_json(out / 'BUILD.json', receipt)
    return receipt


def apply_reference_parameters(receipt,updates):
    """Keep both frozen reference artifacts and their sidecars in agreement."""
    from swap import change_params,typed_like
    native=Path(receipt['native']['path'])
    defaults=literal(native/'estimator.py','PARAMS')
    path=native/'RUNTIME_CONFIG.json'
    configuration=json.loads(path.read_text()) if path.exists() else {}
    before={**defaults,**configuration.get('params',{})}
    if not isinstance(updates,dict) or set(updates)-set(defaults):
        raise ValueError('Reference overrides require registered parameter names')
    values={key:typed_like(value,before[key]) for key,value in updates.items()}
    changes=[dict(key=key,old=before[key],new=value) for key,value in values.items() if before[key]!=value]
    for role,filename in (('native','estimator.py'),('twin','torch_kernel.py')):
        if not receipt.get(role): continue
        directory=Path(receipt[role]['path'])
        change_params(directory/filename,values)
        config_path=directory/'RUNTIME_CONFIG.json'
        config=json.loads(config_path.read_text()) if config_path.exists() else json.loads(json.dumps(configuration))
        config.setdefault('params',{}).update(values)
        write_json(config_path,config)
        receipt[role]=identity(directory)
    receipt['explicit_params_changes']=changes


def freeze_input(pkg, target, assets=None):
    """Always copy inputs, including frozen dirs, so evaluation pins exact bytes."""
    label = str(pkg)
    pkg = resolve_reference(pkg)
    if (pkg / 'whest_est').is_dir():
        return build_package(pkg, target, assets)
    estimator_file = pkg if pkg.is_file() and pkg.suffix == '.py' else None
    if estimator_file:
        native_src, twin_src = pkg.parent, None
    elif (pkg / 'native/estimator.py').is_file():
        native_src, twin_src = pkg / 'native', pkg / 'twin'
    else:
        native_src, twin_src = pkg, None
    if not estimator_file and not (native_src / 'estimator.py').is_file():
        raise ValueError('Expected M1 package, built output or frozen estimator directory: ' + str(pkg))
    target=empty_output(pkg,target)
    native=bulk_dir(target/'native')
    def inventory():
        files={}
        for path in ([estimator_file] if estimator_file else native_src.glob('*.py')):
            files['native/'+('estimator.py' if estimator_file else path.name)]=path
        files.update({'native/'+path.name:path for path in native_src.glob('*.json')})
        files.update({'native/'+name:path for name,path in asset_map(native_src,assets).items()})
        if twin_src and twin_src.is_dir():
            files.update({'twin/'+str(path.relative_to(twin_src)):path for path in twin_src.rglob('*')
                if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc'})
        return files
    copy_snapshot(inventory,target)
    return dict(schema='base-freeze-v1', input=str(pkg), requested_input=label, native=identity(native),
                twin=identity(target / 'twin') if (target / 'twin').is_dir() else None)
