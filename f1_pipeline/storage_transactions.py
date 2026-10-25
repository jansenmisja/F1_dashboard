"""Sandbox proposal: coordinated local-filesystem CSV transactions.
All participants must use this module's transaction context or version tokens.
Not a distributed filesystem or Google Drive durability guarantee.
"""
from pathlib import Path
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import shutil
import threading
import uuid
from .config import CONFIG

_mutex = threading.RLock()
_local = threading.local()

class StaleInputError(RuntimeError):
    pass


def _root():
    return CONFIG.root.resolve()


def _path(value):
    path = Path(value).resolve()
    if _root() not in path.parents:
        raise ValueError('Storage path must be inside configured sandbox')
    if path == _root() / '.transactions' or _root() / '.transactions' in path.parents:
        raise ValueError('Operational path must not be transaction metadata')
    return path


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _sync_file(path):
    with path.open('rb') as handle:
        os.fsync(handle.fileno())


def _sync_dir(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json(path, record):
    temporary = path.parent / ('.journal_' + uuid.uuid4().hex)
    with temporary.open('w') as handle:
        json.dump(record, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _sync_dir(path.parent)


def _metadata():
    folder = _root() / '.transactions'
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _clear(folder, record):
    for item in record['items']:
        Path(item['stage']).unlink(missing_ok=True)
    (folder / 'journal.json').unlink(missing_ok=True)
    _sync_dir(folder)
    shutil.rmtree(folder)
    _sync_dir(folder.parent)


def _recover_locked():
    # Validate all pending journals before applying any recovery mutation.
    # Project earlier rollback results in memory for later overlapping journals.
    projected = {}
    for journal in sorted(_metadata().glob('txn_*/journal.json')):
        record = json.loads(journal.read_text())
        if record.get('version') != 1 or record.get('state') not in ('prepared', 'committed'):
            raise RuntimeError('Invalid recovery journal; storage blocked')
        folder = journal.parent
        for item in record['items']:
            target = _path(item['target'])
            stage = _path(item['stage'])
            if stage.parent != target.parent or not stage.name.startswith('.' + target.name + '.'):
                raise RuntimeError('Invalid staged path')
            if item['old_hash'] is not None:
                backup = Path(item['backup']).resolve()
                if backup.parent != folder.resolve() or _hash(backup) != item['old_hash']:
                    raise RuntimeError('Recovery backup integrity failed; storage blocked')
            allowed = {item['new_hash']} if record['state'] == 'committed' else (
                {item['new_hash'], item['old_hash']} | set(item.get('allowed_hashes', [])))
            current = projected[target] if target in projected else _hash(target)
            if current not in allowed:
                raise RuntimeError('Unexpected output mutation; recovery blocked')
        for item in record['items']:
            projected[_path(item['target'])] = (
                item['old_hash'] if record['state'] == 'prepared' else item['new_hash'])
    for journal in sorted(_metadata().glob('txn_*/journal.json')):
        record = json.loads(journal.read_text())
        if record.get('version') != 1 or record.get('state') not in ('prepared', 'committed'):
            raise RuntimeError('Invalid recovery journal; storage blocked')
        folder = journal.parent
        # Validate the complete evidence set before touching any output.
        for item in record['items']:
            target = _path(item['target'])
            stage = _path(item['stage'])
            if stage.parent != target.parent or not stage.name.startswith('.' + target.name + '.'):
                raise RuntimeError('Invalid staged path')
            if item['old_hash'] is not None:
                backup = Path(item['backup']).resolve()
                if backup.parent != folder.resolve() or _hash(backup) != item['old_hash']:
                    raise RuntimeError('Recovery backup integrity failed; storage blocked')
            allowed = {item['new_hash']} if record['state'] == 'committed' else ({item['new_hash'], item['old_hash']} | set(item.get('allowed_hashes', [])))
            if _hash(target) not in allowed:
                raise RuntimeError('Unexpected output mutation; recovery blocked')
        if record['state'] == 'prepared':
            service = getattr(_local, 'service', None)
            if service is not None and service.get('folder') == folder:
                service['failed'] = True
            for item in record['items']:
                target = Path(item['target'])
                if item['old_hash'] is None:
                    target.unlink(missing_ok=True)
                elif _hash(target) != item['old_hash']:
                    temp = target.parent / ('.' + target.name + '.restore_' + uuid.uuid4().hex)
                    shutil.copyfile(item['backup'], temp)
                    _sync_file(temp)
                    os.replace(temp, target)
                _sync_dir(target.parent)
        _clear(folder, record)


@contextmanager
def transaction():
    """Hold through recovery, all related reads, decisions and final writes."""
    with _mutex:
        if getattr(_local, 'depth', 0):
            _local.depth += 1
            try:
                yield
            finally:
                _local.depth -= 1
            return
        lock_path = _metadata() / 'storage.lock'
        with lock_path.open('a+b') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            _local.depth = 1
            try:
                _recover_locked()
                yield
            finally:
                _local.depth = 0
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def recover():
    with transaction():
        pass


def versions(paths):
    with transaction():
        return {str(_path(path)): _hash(_path(path)) for path in paths}


def read(path, *, required=True, **kwargs):
    from . import io
    with transaction():
        return io._step9c_original_read(path, required=required, **kwargs)


def commit(items, *, validators=None, backup_dir=None, round_decimals=3,
           prepare_commit=None, commit_guard=None, after_commit=None,
           expected_versions=None):
    if not getattr(_local, 'depth', 0) and expected_versions is None:
        raise RuntimeError('Service transaction or complete version tokens required')
    from . import io
    validators = validators or {}
    service = getattr(_local, 'service', None)
    if service is not None and commit_guard is not None:
        guards = service.setdefault('final_guards', [])
        if not any(existing is commit_guard for existing in guards):
            guards.append(commit_guard)
    with transaction():
        normalized = {_path(path): io.round_float_columns(frame, decimals=round_decimals)
                      for path, frame in items.items()}
        if not normalized:
            raise ValueError('Empty transaction')
        if expected_versions is not None:
            expected = {str(_path(path)): value for path, value in expected_versions.items()}
            if not set(map(str, normalized)).issubset(expected):
                raise ValueError('Version tokens missing for output paths')
            for path, value in expected.items():
                if _hash(Path(path)) != value:
                    raise StaleInputError('Input changed since snapshot: ' + path)
        def validator(path):
            return validators.get(path) or validators.get(str(path))
        for path, frame in normalized.items():
            io._run_validator(validator(path), frame)
        if prepare_commit is not None:
            prepare_commit(normalized)
        if set(normalized) != {_path(p) for p in items}:
            raise ValueError('Finalization changed transaction target set')
        folder = _metadata() / ('txn_' + uuid.uuid4().hex)
        folder.mkdir()
        record = {'version': 1, 'state': 'prepared', 'items': []}
        results = {}
        try:
            for number, (path, frame) in enumerate(normalized.items()):
                path.parent.mkdir(parents=True, exist_ok=True)
                frame = io.round_float_columns(frame, decimals=round_decimals)
                normalized[path] = frame
                io._run_validator(validator(path), frame)
                old_hash = _hash(path)
                backup = folder / ('old_' + str(number) + '.csv')
                if old_hash is not None:
                    shutil.copyfile(path, backup)
                    _sync_file(backup)
                    if _hash(backup) != old_hash:
                        raise RuntimeError('Backup differs from original')
                stage = path.parent / ('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
                record['items'].append({'target': str(path), 'backup': str(backup),
                                        'stage': str(stage), 'old_hash': old_hash, 'new_hash': None})
                frame.to_csv(stage, index=False)
                _sync_file(stage)
                staged = io._step9c_original_read(stage)
                io._run_validator(validator(path), staged)
                if len(staged) != len(frame) or list(staged.columns) != list(frame.columns):
                    raise ValueError('Stage shape mismatch')
                record['items'][-1]['new_hash'] = _hash(stage)
                _service_record(path, _hash(stage))
                # Keep the public backup contract, independently of recovery copies.
                public_backup = io.backup_csv(path, backup_dir=backup_dir)
                results[path] = io.CommitResult(True, path, public_backup, len(frame), len(frame.columns),
                                                'journaled multi-file commit successful')
            _sync_dir(folder)
            _json(folder / 'journal.json', record)
            for item in record['items']:
                if commit_guard is not None:
                    commit_guard()
                os.replace(item['stage'], item['target'])
                _sync_dir(Path(item['target']).parent)
            for item in record['items']:
                path = Path(item['target'])
                if _hash(path) != item['new_hash']:
                    raise RuntimeError('Persisted bytes differ from staged payload')
                io._run_validator(validator(path), io._step9c_original_read(path))
            if after_commit is not None:
                after_commit()
            record['state'] = 'committed'
            _json(folder / 'journal.json', record)
        except Exception:
            if (folder / 'journal.json').exists():
                _recover_locked()
            else:
                _clear(folder, record)
            raise
        # A cleanup problem after committed marker must not roll back a committed run.
        _clear(folder, record)
        return results

from functools import wraps as _wraps

def transactional(function):
    @_wraps(function)
    def coordinated(*args, **kwargs):
        with transaction():
            return function(*args, **kwargs)
    return coordinated
# Service journal spans all commits made by one outer service call.
# Inner storage journals are recovered first (txn_<hex> sorts before txn_z_service).
def _service_record(path, candidate_hash):
    service = getattr(_local, 'service', None)
    if service is None:
        return
    if service.get('failed'):
        raise RuntimeError('Service transaction already rolled back')
    folder, record = service['folder'], service['record']
    if folder is None:
        folder = _metadata() / ('txn_z_service_' + uuid.uuid4().hex)
        folder.mkdir()
        record = {'version': 1, 'state': 'prepared', 'items': []}
        service.update(folder=folder, record=record)
    item = next((item for item in record['items'] if item['target'] == str(path)), None)
    if item is None:
        backup = folder / ('old_' + str(len(record['items'])) + '.csv')
        old_hash = _hash(path)
        if old_hash is not None:
            shutil.copyfile(path, backup)
            _sync_file(backup)
            if _hash(backup) != old_hash:
                raise RuntimeError('Service backup integrity failed')
        item = {'target': str(path), 'backup': str(backup), 'old_hash': old_hash,
                'new_hash': candidate_hash,
                'stage': str(path.parent / ('.' + path.name + '.service_' + uuid.uuid4().hex)),
                'allowed_hashes': [old_hash]}
        record['items'].append(item)
    item['new_hash'] = candidate_hash
    for value in (_hash(path), candidate_hash):
        if value not in item['allowed_hashes']:
            item['allowed_hashes'].append(value)
    _sync_dir(folder)
    _json(folder / 'journal.json', record)
    _sync_dir(folder.parent)



def transactional(function):
    @_wraps(function)
    def coordinated(*args, **kwargs):
        with transaction():
            if getattr(_local, 'service', None) is not None:
                return function(*args, **kwargs)
            from . import cloud_state
            cloud_state.refresh()
            service = {'folder': None, 'record': None, 'failed': False, 'final_guards': []}
            _local.service = service
            try:
                result = function(*args, **kwargs)
                if service['failed']:
                    raise RuntimeError('Service transaction already rolled back')
                if service['folder'] is not None:
                    record = service['record']
                    for item in record['items']:
                        if _hash(Path(item['target'])) != item['new_hash']:
                            raise RuntimeError('Service final storage integrity failed')
                    for gate in service['final_guards']:
                        gate()
                    record['state'] = 'committed'
                    _json(service['folder'] / 'journal.json', record)
                    try:
                        for gate in service['final_guards']:
                            gate()
                        cloud_state.publish()
                    except Exception:
                        # The process is alive and still owns the lock. Reopen the
                        # recovery record before rollback of a rejected final gate.
                        record['state'] = 'prepared'
                        _json(service['folder'] / 'journal.json', record)
                        raise
                    _clear(service['folder'], record)
                return result
            except Exception:
                cloud_state.invalidate()
                _recover_locked()
                raise
            finally:
                _local.service = None
    return coordinated
