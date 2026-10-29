"""Host-side validation, runtime identities, paths and owned processes."""

class RuntimeFingerprint:
    """Fingerprint the dependency origins selected by the execution interpreter."""
    import hashlib
    import json
    from pathlib import Path
    import shlex
    import shutil
    import subprocess
    import sys

    @staticmethod
    def digest(path):
        value = RuntimeFingerprint.hashlib.sha256()
        with RuntimeFingerprint.Path(path).open('rb') as stream:
            for block in iter(lambda : stream.read(1024 ** 2), b''):
                value.update(block)
        return value.hexdigest()

    @staticmethod
    def dependency_files(origins):
        files = set()
        for (name, spec) in origins.items():
            origin = spec.get('origin')
            if origin and origin not in ('built-in', 'frozen'):
                files.add(RuntimeFingerprint.Path(origin).resolve())
            for location in spec.get('locations', ()):
                root = RuntimeFingerprint.Path(location).resolve()
                for package in (root, root.parent / (name + '.libs')):
                    files.update((path for path in package.rglob('*') if path.is_file() and (path.suffix in ('.py', '.so', '.json') or '.so.' in path.name)))
        return files

    @staticmethod
    def interpreter(executable, env):
        """Honor console-script shebangs, preserving a selected launcher for execution."""
        with executable.open('rb') as stream:
            header = stream.readline(4096)
        if header.startswith(b'#!'):
            words = RuntimeFingerprint.shlex.split(header[2:].decode().strip())
            if words and RuntimeFingerprint.Path(words[0]).name == 'env':
                words = words[1:]
                if words and words[0] == '-S':
                    words = words[1:]
                if words:
                    selected = RuntimeFingerprint.shutil.which(words[0], path=env.get('PATH'))
                    if selected:
                        words[0] = selected
            if words and 'python' in RuntimeFingerprint.Path(words[0]).name:
                return RuntimeFingerprint.Path(words[0]).expanduser().absolute()
        candidate = executable.parent / 'python'
        return candidate if 'python' not in executable.name and candidate.is_file() else executable
    PROBE = 'import importlib.util,json,sys\nsys.path[0]=sys.argv[2]\nresult={}\nfor name in json.loads(sys.argv[1]):\n    spec=importlib.util.find_spec(name)\n    result[name]=dict(origin=spec.origin,locations=list(spec.submodule_search_locations or ())) if spec else dict(missing=True)\nprint(json.dumps(result,sort_keys=True))\n'

    @staticmethod
    def fingerprint(executable, packages, env, cwd, script_dir, cache):
        executable = RuntimeFingerprint.Path(executable).expanduser().absolute()
        python = RuntimeFingerprint.interpreter(executable, env)
        cache_id = (str(executable), str(executable.resolve()), str(python), str(python.resolve()), tuple(packages), str(RuntimeFingerprint.Path(cwd).resolve()), str(RuntimeFingerprint.Path(script_dir).resolve()), tuple(((key, env.get(key)) for key in ('PYTHONPATH', 'PYTHONHOME', 'PATH'))))
        if cache_id not in cache:
            origins = {}
            if packages:
                probe = RuntimeFingerprint.subprocess.run([str(python), '-c', RuntimeFingerprint.PROBE, RuntimeFingerprint.json.dumps(packages), str(script_dir)], env=env, cwd=cwd, capture_output=True, text=True, timeout=30)
                if probe.returncode:
                    raise RuntimeError('Cannot resolve execution import origins: ' + probe.stderr.strip())
                origins = RuntimeFingerprint.json.loads(probe.stdout)
            files = RuntimeFingerprint.dependency_files(origins) | {executable, executable.resolve(), python, python.resolve()}
            value = RuntimeFingerprint.hashlib.sha256(RuntimeFingerprint.json.dumps(origins, sort_keys=True).encode())
            for path in sorted(files):
                value.update(str(path).encode() + b'\x00' + RuntimeFingerprint.digest(path).encode())
            cache[cache_id] = value.hexdigest()
        return cache[cache_id]
    _loaded = {}

    @staticmethod
    def loaded_identity(packages=('torch', 'numpy')):
        """The trainer already imported these modules: hash those exact origins."""
        origins = {name: dict(origin=RuntimeFingerprint.sys.modules[name].__spec__.origin, locations=list(RuntimeFingerprint.sys.modules[name].__spec__.submodule_search_locations or ())) for name in packages}
        key = RuntimeFingerprint.json.dumps(origins, sort_keys=True)
        if key not in RuntimeFingerprint._loaded:
            files = RuntimeFingerprint.dependency_files(origins) | {RuntimeFingerprint.Path(RuntimeFingerprint.sys.executable).resolve()}
            value = RuntimeFingerprint.hashlib.sha256(key.encode())
            for path in sorted(files):
                value.update(str(path).encode() + b'\x00' + RuntimeFingerprint.digest(path).encode())
            RuntimeFingerprint._loaded[key] = dict(origins=origins, sha256=value.hexdigest())
        return RuntimeFingerprint._loaded[key]

class Paths:
    """Pin CLI paths in the caller's directory before launching another process."""
    from pathlib import Path
    import shutil
    PATH_OPTIONS = {'--base', '--init', '--resume', '--ckpt', '--norms', '--runtime-config', '--base-cache', '--out', '--output', '--export', '--request', '--config', '--client-python', '--server-bin'}

    @staticmethod
    def absolute(value, cwd=None):
        path = Paths.Path(value).expanduser()
        return path if path.is_absolute() else (Paths.Path(cwd) if cwd else Paths.Path.cwd()) / path

    @staticmethod
    def normalize_paths(args):
        for (name, value) in vars(args).items():
            if isinstance(value, Paths.Path) or (name == 'norms' and isinstance(value, str)):
                setattr(args, name, Paths.absolute(value))
        return args

    @staticmethod
    def normalize_arguments(arguments, cwd=None, *, command=False):
        cwd = Paths.Path(cwd) if cwd else Paths.Path.cwd()
        arguments = list(map(str, arguments))
        for (index, word) in enumerate(arguments):
            (option, equals, value) = word.partition('=')
            if option in Paths.PATH_OPTIONS:
                if equals:
                    arguments[index] = option + '=' + str(Paths.absolute(value, cwd))
                elif index + 1 < len(arguments):
                    arguments[index + 1] = str(Paths.absolute(arguments[index + 1], cwd))
        if command and arguments:
            executable = arguments[0]
            selected = Paths.shutil.which(executable) if '/' not in executable else None
            arguments[0] = selected or str(Paths.absolute(executable, cwd))
            if len(arguments) > 1 and arguments[1].endswith('.py'):
                arguments[1] = str(Paths.absolute(arguments[1], cwd))
        return arguments

    @staticmethod
    def option_value(arguments, option, default=None):
        value = default
        for (index, word) in enumerate(arguments):
            if word == option and index + 1 < len(arguments):
                value = arguments[index + 1]
            elif word.startswith(option + '='):
                value = word.split('=', 1)[1]
        return value

class OwnedProcesses:
    """Cancel only process groups started by this tool controller."""
    from contextlib import contextmanager
    import os
    from pathlib import Path
    import signal
    import subprocess
    import threading
    import time
    _processes = {}
    _lock = threading.RLock()
    _cancelled = threading.Event()
    _cancel_signum = signal.SIGINT
    _scope_depth = 0

    class Cancelled(BaseException):

        def __init__(self, signum):
            self.signum = signum

    @staticmethod
    def kill(process):
        (descendants, pending) = ([], [process.pid])
        while pending:
            pid = pending.pop()
            try:
                OwnedProcesses.os.kill(pid, OwnedProcesses.signal.SIGSTOP)
                until = OwnedProcesses.time.monotonic() + 0.1
                while True:
                    status = OwnedProcesses.Path(f'/proc/{pid}/status').read_text()
                    state = next((line.split()[1] for line in status.splitlines() if line.startswith('State:')))
                    if state in ('T', 't', 'Z', 'X') or OwnedProcesses.time.monotonic() >= until:
                        break
                    OwnedProcesses.time.sleep(0.001)
            except (OSError, StopIteration):
                pass
            try:
                tasks = list(OwnedProcesses.Path(f'/proc/{pid}/task').iterdir())
            except OSError:
                tasks = []
            children = set()
            for task in tasks:
                try:
                    children.update((int(child) for child in (task / 'children').read_text().split()))
                except (OSError, ValueError):
                    pass
            new = children - set(descendants)
            descendants.extend(new)
            pending.extend(new)
        for pid in reversed(descendants):
            try:
                if OwnedProcesses.os.getpgid(pid) == pid:
                    OwnedProcesses.os.killpg(pid, OwnedProcesses.signal.SIGKILL)
                else:
                    OwnedProcesses.os.kill(pid, OwnedProcesses.signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            OwnedProcesses.os.killpg(process.pid, OwnedProcesses.signal.SIGKILL)
        except ProcessLookupError:
            pass

    @staticmethod
    def check_cancelled():
        if OwnedProcesses._cancelled.is_set():
            raise OwnedProcesses.Cancelled(OwnedProcesses._cancel_signum)

    @staticmethod
    @contextmanager
    def track(process):
        with OwnedProcesses._lock:
            OwnedProcesses._processes[process.pid] = process
        try:
            if OwnedProcesses._cancelled.is_set():
                OwnedProcesses.kill(process)
                OwnedProcesses.check_cancelled()
            yield process
        finally:
            if OwnedProcesses._cancelled.is_set():
                OwnedProcesses.kill(process)
                try:
                    process.wait(timeout=5)
                except OwnedProcesses.subprocess.TimeoutExpired:
                    pass
            with OwnedProcesses._lock:
                OwnedProcesses._processes.pop(process.pid, None)

    @staticmethod
    @contextmanager
    def scope():
        if OwnedProcesses.threading.current_thread() is not OwnedProcesses.threading.main_thread():
            yield
            return
        if OwnedProcesses._scope_depth:
            OwnedProcesses.check_cancelled()
            yield
            return
        OwnedProcesses._scope_depth += 1
        old = {signum: OwnedProcesses.signal.getsignal(signum) for signum in (OwnedProcesses.signal.SIGINT, OwnedProcesses.signal.SIGTERM)}

        def cancel(signum, frame):
            pass
            first = not OwnedProcesses._cancelled.is_set()
            OwnedProcesses._cancel_signum = signum
            OwnedProcesses._cancelled.set()
            with OwnedProcesses._lock:
                processes = list(OwnedProcesses._processes.values())
            for process in processes:
                OwnedProcesses.kill(process)
            if first:
                raise OwnedProcesses.Cancelled(signum)
        OwnedProcesses._cancelled.clear()
        for signum in old:
            OwnedProcesses.signal.signal(signum, cancel)
        try:
            yield
        finally:
            OwnedProcesses._scope_depth -= 1
            for (signum, handler) in old.items():
                OwnedProcesses.signal.signal(signum, handler)

class Hardware:
    """Bind cached predictions to their CPU host without running a benchmark."""
    from functools import lru_cache
    from pathlib import Path
    import platform

    @staticmethod
    @lru_cache(maxsize=1)
    def identity():
        try:
            text = Hardware.Path('/proc/cpuinfo').read_text().split('\n\n', 1)[0]
        except OSError:
            return None
        fields = {}
        wanted = {'vendor_id', 'model name', 'flags', 'Features', 'CPU architecture', 'CPU implementer', 'CPU part', 'Hardware'}
        for line in text.splitlines():
            (name, separator, value) = line.partition(':')
            if separator and name.strip() in wanted:
                fields[name.strip()] = value.strip()
        if not fields:
            return None
        return dict(host=Hardware.platform.node(), architecture=Hardware.platform.machine(), cpu=fields, libc=list(Hardware.platform.libc_ver()))

class PredictionGuard:
    """Accept one captured finite float64 prediction with its recorded dimensions/hash."""
    import hashlib
    from io import BytesIO
    from pathlib import Path
    import numpy as np

    @staticmethod
    def valid(path, record, metadata=None):
        try:
            descriptor = record['prediction']
            expected = descriptor['sha256'] if isinstance(descriptor, dict) else record['prediction_sha256']
            shape = descriptor.get('shape', record.get('shape')) if isinstance(descriptor, dict) else record.get('shape')
            payload = PredictionGuard.Path(path).read_bytes()
            if PredictionGuard.hashlib.sha256(payload).hexdigest() != expected:
                return False
            values = PredictionGuard.np.load(PredictionGuard.BytesIO(payload), allow_pickle=False)
            if not isinstance(values, PredictionGuard.np.ndarray) or values.dtype != PredictionGuard.np.float64 or values.ndim != 2 or any((size < 1 for size in values.shape)) or (not PredictionGuard.np.isfinite(values).all()):
                return False
            if shape is not None and (not isinstance(shape, (list, tuple)) or len(shape) != 2 or any((type(size) is not int or size < 1 for size in shape)) or (values.shape != tuple(shape))):
                return False
            dataset = (metadata or {}).get('dataset_key', record.get('dataset_key'))
            return dataset not in ('full', 'd8b') or values.shape == (16, 1024)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError):
            return False

class FixtureGuard:
    """Validate client fixture bytes before opening a metered numerical session."""
    import hashlib
    import json
    import zipfile
    from pathlib import Path
    import numpy as np
    digest = RuntimeFingerprint.digest

    @staticmethod
    def validate(path):
        path = FixtureGuard.Path(path)
        with FixtureGuard.zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate fixture tensor names')
            if not {'weights.npy', 'target.npy', 'seed.npy'} <= set(names):
                raise ValueError('Fixture requires weights, target and seed tensors')
        before = FixtureGuard.digest(path)
        with FixtureGuard.np.load(path, allow_pickle=False) as value:
            (weights, target, seed) = (value['weights'], value['target'], value['seed'])
            metadata = FixtureGuard.json.loads(str(value['metadata'])) if 'metadata' in value.files else {}
        if not isinstance(metadata, dict):
            raise ValueError('Fixture metadata must be an object')
        if weights.ndim != 3 or weights.shape[1] != weights.shape[2] or weights.shape[0] < 1 or (weights.shape[1] < 1) or (weights.dtype != FixtureGuard.np.float32) or (target.shape != weights.shape[:2]) or (target.dtype != FixtureGuard.np.float64) or (seed.shape != ()) or (seed.dtype.kind not in 'iu') or (not FixtureGuard.np.isfinite(weights).all()) or (not FixtureGuard.np.isfinite(target).all()):
            raise ValueError('Fixture dimensions, dtype, seed or finiteness mismatch')
        if metadata.get('dataset_key') in ('full', 'd8b') and weights.shape != (16, 1024, 1024):
            raise ValueError('Production fixture dimensions mismatch')
        if metadata:
            actual = dict(seed=int(seed), weights_sha256=FixtureGuard.hashlib.sha256(weights.tobytes()).hexdigest(), target_sha256=FixtureGuard.hashlib.sha256(target.tobytes()).hexdigest())
            if any((metadata.get(name) != expected for (name, expected) in actual.items())):
                raise ValueError('Fixture metadata data identity mismatch')
        if FixtureGuard.digest(path) != before:
            raise ValueError('Fixture bytes changed during validation')
        return dict(fixture_sha256=before, metadata=metadata)
