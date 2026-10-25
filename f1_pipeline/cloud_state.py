"""Durable project snapshots on a separate GitHub branch, with optimistic concurrency."""
from pathlib import Path, PurePosixPath
import base64
import hashlib
import io
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

SCOPES = ('data', 'log', 'models')
EXTENSIONS = {'.csv', '.json', '.jsonl', '.parquet', '.pkl', '.joblib', '.cbm', '.txt'}
_version = None
_manifest = {}
REPOSITORY = 'jansenmisja/F1_dashboard'
BRANCH = 'f1-data'
STATE_PATH = 'f1-state.zip'

class CloudStorageError(RuntimeError):
    pass

class CloudConflictError(CloudStorageError):
    pass

def enabled():
    return os.environ.get('F1_CLOUD_PERSISTENCE') == 'github_v1'

def _api(method, path, payload=None, *, raw=False, allow404=False):
    token = os.environ.get('F1_GITHUB_TOKEN', '').strip()
    if not token:
        raise CloudStorageError('F1_GITHUB_TOKEN ontbreekt in Streamlit Secrets.')
    url = 'https://api.github.com/repos/' + REPOSITORY + ('/' + path if path else '')
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={
        'Authorization': 'Bearer ' + token,
        'Accept': 'application/vnd.github.raw+json' if raw else 'application/vnd.github+json',
        'Content-Type': 'application/json', 'User-Agent': 'F1-dashboard-persistence-v1',
        'X-GitHub-Api-Version': '2022-11-28',
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read(50 * 1024 * 1024 + 1)
        if len(body) > 50 * 1024 * 1024:
            raise CloudStorageError('Remote state overschrijdt de ingestelde limiet van 50 MiB.')
        return body if raw else json.loads(body)
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and allow404:
            return None
        if exc.code in (409, 422):
            raise CloudConflictError('Gelijktijdige opslag of branchconflict. Ververs de app; geen run opnieuw starten zonder statuscontrole.') from None
        raise CloudStorageError(f'GitHub-opslag HTTP {exc.code}. Controleer repositoryrechten, token en limieten.') from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise CloudStorageError('GitHub-opslag tijdelijk onbereikbaar; opslag is niet bevestigd.') from None

def _state_metadata():
    return _api('GET', f'contents/{STATE_PATH}?ref={BRANCH}', allow404=True)

def _download(meta):
    if meta.get('encoding') == 'base64' and meta.get('content'):
        data = base64.b64decode(meta['content'])
    else:
        # Address the exact immutable blob, never a branch that can change between requests.
        data = _api('GET', 'git/blobs/' + meta['sha'], raw=True)
    if hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() != meta['sha']:
        raise CloudStorageError('GitHub-state blobhash wijkt af.')
    return data

def _allowed(name):
    path = PurePosixPath(name)
    return (not path.is_absolute() and '..' not in path.parts and '\\' not in name
            and path.parts and path.parts[0] in SCOPES
            and not any(part.startswith('.') or 'backup' in part.lower() for part in path.parts)
            and path.suffix.lower() in EXTENSIONS)

def decode(data):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            files = bundle.infolist()
            if len(files) > 5000 or sum(f.file_size for f in files) > 100 * 1024 * 1024:
                raise CloudStorageError('State-archief overschrijdt de toegestane omvang.')
            names = [f.filename for f in files]
            if len(set(names)) != len(names) or not all(_allowed(name) for name in names):
                raise CloudStorageError('State-archief bevat ongeldige of dubbele paden.')
            return {f.filename: bundle.read(f) for f in files}
    except (zipfile.BadZipFile, RuntimeError, ValueError) as exc:
        if isinstance(exc, CloudStorageError):
            raise
        raise CloudStorageError('Remote state is geen geldig projectarchief.') from None

def _local_files(root):
    found = {}
    for scope in SCOPES:
        base = root / scope
        if base.exists():
            for path in base.rglob('*'):
                if path.is_file() and _allowed(path.relative_to(root).as_posix()):
                    if path.is_symlink():
                        raise CloudStorageError('Symlink in projectdata is niet toegestaan.')
                    found[path.relative_to(root).as_posix()] = path.read_bytes()
    return found

def _hashes(files):
    return {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}

def encode(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name, data in sorted(files.items()):
            if not _allowed(name):
                raise CloudStorageError('Ongeldig projectpad: ' + name)
            item = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(item, data)
    result = stream.getvalue()
    if len(result) > 45 * 1024 * 1024:
        raise CloudStorageError('Projectstate is te groot voor deze opslagconfiguratie.')
    return result

def _root():
    from .config import CONFIG
    return CONFIG.root

def _install(files):
    root = _root()
    # Decode/validate the complete archive before changing any local file.
    for name in files:
        if not _allowed(name):
            raise CloudStorageError('Ongeldig statepad.')
    existing = _local_files(root)
    for name, data in files.items():
        if existing.get(name) == data:
            continue
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.state_', dir=target.parent)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        os.replace(temporary, target)
    for name in set(existing) - set(files):
        (root / name).unlink()
    for folder in ('data/processed/backups', 'models/weights', 'models/importance', 'log'):
        (root / folder).mkdir(parents=True, exist_ok=True)

def _ensure_branch():
    branch = _api('GET', 'git/ref/heads/' + BRANCH, allow404=True)
    if branch is not None:
        return
    repo = _api('GET', '')
    main = _api('GET', 'git/ref/heads/' + urllib.parse.quote(repo['default_branch'], safe=''))
    try:
        _api('POST', 'git/refs', {'ref': 'refs/heads/' + BRANCH, 'sha': main['object']['sha']})
    except CloudConflictError:
        if _api('GET', 'git/ref/heads/' + BRANCH, allow404=True) is None:
            raise

def _put(data, sha):
    payload = {'message': 'F1: persist canonical project state', 'branch': BRANCH,
               'content': base64.b64encode(data).decode()}
    if sha is not None:
        payload['sha'] = sha
    try:
        result = _api('PUT', 'contents/' + STATE_PATH, payload)
        return result['content']['sha']
    except CloudConflictError:
        raise
    except CloudStorageError:
        # A response can be lost after a successful write. Resolve this before reporting failure.
        for _ in range(2):
            try:
                meta = _state_metadata()
                if meta and _download(meta) == data:
                    return meta['sha']
            except CloudStorageError:
                pass
            time.sleep(1)
        raise CloudStorageError('Remote opslag niet bevestigd. Ververs eerst de app om de opgeslagen status te controleren.') from None

def refresh(*, bootstrap=False):
    """Caller holds canonical storage lock; no concurrent local reads during hydration."""
    global _version, _manifest
    if not enabled():
        return
    if bootstrap:
        _ensure_branch()
    meta = _state_metadata()
    if meta is None:
        if not bootstrap:
            raise CloudStorageError('Duurzame projectstate ontbreekt; initialisatie vereist.')
        seed = _root() / 'seed_state.zip'
        files = decode(seed.read_bytes())
        try:
            _put(encode(files), None)
        except CloudConflictError:
            pass
        meta = _state_metadata()
        if meta is None:
            raise CloudStorageError('Initialisatie van duurzame opslag is niet bevestigd.')
    if _version != meta['sha']:
        files = decode(_download(meta))
        _install(files)
        _version, _manifest = meta['sha'], _hashes(files)

def publish():
    global _version, _manifest
    if not enabled():
        return
    if _version is None:
        raise CloudStorageError('Duurzame opslag niet geïnitialiseerd.')
    files = _local_files(_root())
    hashes = _hashes(files)
    if hashes == _manifest:
        return
    new_version = _put(encode(files), _version)
    _version, _manifest = new_version, hashes

def invalidate():
    global _version
    _version = None

def bootstrap():
    from .storage_transactions import transaction
    with transaction():
        refresh(bootstrap=True)
