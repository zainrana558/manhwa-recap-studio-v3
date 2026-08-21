"""Production hardening primitives for the recap pipeline.

Local-first, dependency-injection-friendly helpers used by the Python renderer
and covered by focused unit tests.  These helpers deliberately prefer explicit
FAILED/UNCERTAIN states over fabricated success.
"""
from __future__ import annotations

import hashlib, json, os, shutil, sqlite3, subprocess, time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

class Stage(str, Enum):
    JOB='JOB'; CHAPTER='CHAPTER'; PANEL='PANEL'; OCR='OCR'; NARRATION='NARRATION'; TTS='TTS'; AUDIO_ASSEMBLY='AUDIO_ASSEMBLY'; VIDEO_RENDER='VIDEO_RENDER'; MERGE='MERGE'; FINAL_QA='FINAL_QA'
class State(str, Enum):
    PENDING='PENDING'; RUNNING='RUNNING'; COMPLETE='COMPLETE'; RETRYABLE='RETRYABLE'; FAILED='FAILED'; UNCERTAIN='UNCERTAIN'
class RetryCategory(str, Enum):
    TRANSIENT='TRANSIENT'; TIMEOUT='TIMEOUT'; PROVIDER_UNAVAILABLE='PROVIDER_UNAVAILABLE'; RESOURCE='RESOURCE'; QUALITY='QUALITY'; CORRUPT_ARTIFACT='CORRUPT_ARTIFACT'; INVALID_INPUT='INVALID_INPUT'; PERMANENT_FAILURE='PERMANENT_FAILURE'

@dataclass
class QAResult:
    ok: bool; reason: str=''; duration: float=0.0; metadata: Dict[str, Any]=field(default_factory=dict)
@dataclass
class AudioResult:
    status: State; path: Optional[Path]=None; checksum: Optional[str]=None; duration: float=0.0; provider: str=''; error_code: Optional[str]=None; error_message: Optional[str]=None; retry_category: Optional[RetryCategory]=None
@dataclass
class OCRResult:
    text: str=''; confidence: float=0.0; regions: int=0; status: State=State.FAILED; quality_score: float=0.0; candidates: list=field(default_factory=list); selection_reason: str=''; provider: str='paddleocr'; model: str='unknown'
@dataclass
class ResourceStatus:
    ok: bool; free_disk_bytes: int; available_ram_bytes: int; load1: float=0.0; reason: str=''

def _enum_value(value: Any) -> str:
    return value.value if isinstance(value, Enum) else str(value)

class ProcessRunner:
    def run(self, args: Sequence[str], timeout: Optional[int]=None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, check=False)

def checksum_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

class SQLiteStateStore:
    def __init__(self, path: Path):
        self.path=path; path.parent.mkdir(parents=True, exist_ok=True); self._init()
    def _init(self):
        with sqlite3.connect(self.path) as db:
            db.execute('''CREATE TABLE IF NOT EXISTS pipeline_state(
              job_id TEXT NOT NULL, chapter_id TEXT NOT NULL DEFAULT '', panel_id TEXT NOT NULL DEFAULT '', stage TEXT NOT NULL,
              state TEXT NOT NULL, attempt_count INTEGER NOT NULL DEFAULT 0, provider TEXT, model TEXT, artifact_path TEXT,
              artifact_checksum TEXT, duration REAL, error_code TEXT, error_message TEXT, retry_category TEXT,
              metadata_json TEXT, updated_at REAL NOT NULL, PRIMARY KEY(job_id, chapter_id, panel_id, stage))''')
    def record(self, job_id: str, stage: Stage|str, state: State|str, chapter_id: str='', panel_id: str='', **kw: Any) -> None:
        metadata=kw.get('metadata') or {}
        with sqlite3.connect(self.path) as db:
            prior=db.execute('SELECT attempt_count FROM pipeline_state WHERE job_id=? AND chapter_id=? AND panel_id=? AND stage=?',(job_id,chapter_id,panel_id,_enum_value(stage))).fetchone()
            attempts=int(kw.get('attempt_count', (prior[0] if prior else 0)))
            if state in (State.RETRYABLE, 'RETRYABLE', State.RUNNING, 'RUNNING'): attempts += 1 if kw.get('increment_attempt', True) else 0
            db.execute('''INSERT OR REPLACE INTO pipeline_state VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',(
                job_id,chapter_id,panel_id,_enum_value(stage),_enum_value(state),attempts,kw.get('provider'),kw.get('model'),str(kw['artifact_path']) if kw.get('artifact_path') else None,kw.get('artifact_checksum'),kw.get('duration'),kw.get('error_code'),kw.get('error_message'),_enum_value(kw.get('retry_category')) if kw.get('retry_category') else None,json.dumps(metadata),time.time()))
    def get(self, job_id: str, stage: Stage|str, chapter_id: str='', panel_id: str='') -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.path) as db:
            db.row_factory=sqlite3.Row
            r=db.execute('SELECT * FROM pipeline_state WHERE job_id=? AND chapter_id=? AND panel_id=? AND stage=?',(job_id,chapter_id,panel_id,_enum_value(stage))).fetchone()
            return dict(r) if r else None

class ResourceGuard:
    def __init__(self, min_free_disk_bytes:int=1_000_000_000, min_available_ram_bytes:int=512_000_000, state:SQLiteStateStore|None=None):
        self.min_disk=min_free_disk_bytes; self.min_ram=min_available_ram_bytes; self.state=state
    def check(self, path: Path, job_id: str='', stage: Stage|str=Stage.JOB) -> ResourceStatus:
        du=shutil.disk_usage(path)
        ram=0
        try:
            for line in Path('/proc/meminfo').read_text().splitlines():
                if line.startswith('MemAvailable:'): ram=int(line.split()[1])*1024; break
        except Exception: ram=10**18
        load=os.getloadavg()[0] if hasattr(os,'getloadavg') else 0.0
        reason=''
        if du.free < self.min_disk: reason=f'free disk {du.free} below threshold {self.min_disk}'
        elif ram < self.min_ram: reason=f'available RAM {ram} below threshold {self.min_ram}'
        ok=not reason
        if not ok and self.state and job_id:
            self.state.record(job_id, stage, State.RETRYABLE, error_code='RESOURCE', error_message=reason, retry_category=RetryCategory.RESOURCE)
        return ResourceStatus(ok, du.free, ram, load, reason)

def classify_retry(exc: Exception|str) -> RetryCategory:
    s=str(exc).lower()
    if 'timeout' in s or 'timed out' in s: return RetryCategory.TIMEOUT
    if 'no such file' in s or 'not found' in s or 'missing binary' in s: return RetryCategory.PROVIDER_UNAVAILABLE
    if 'disk' in s or 'memory' in s or 'resource' in s or 'oom' in s: return RetryCategory.RESOURCE
    if 'quality' in s or 'silence' in s or 'uncertain' in s: return RetryCategory.QUALITY
    if 'corrupt' in s or 'ffprobe' in s: return RetryCategory.CORRUPT_ARTIFACT
    if 'invalid' in s or 'empty input' in s: return RetryCategory.INVALID_INPUT
    return RetryCategory.TRANSIENT

def atomic_promote(tmp: Path, final: Path) -> str:
    final.parent.mkdir(parents=True, exist_ok=True); digest=checksum_file(tmp); os.replace(tmp, final); return digest

class ArtifactStore:
    """Lightweight local artifact helper for tmp -> QA -> checksum -> atomic promotion."""
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
    def tmp_for(self, final: Path) -> Path:
        final.parent.mkdir(parents=True, exist_ok=True)
        return final.with_name(f"{final.stem}.tmp{final.suffix}")
    def promote(self, tmp: Path, final: Path) -> str:
        return atomic_promote(tmp, final)
    def quarantine(self, artifact: Path, reason: str = "bad") -> Path:
        qdir = artifact.parent / "quarantine"
        qdir.mkdir(parents=True, exist_ok=True)
        target = qdir / f"{artifact.name}.{int(time.time())}.{reason}"
        artifact.rename(target)
        return target

def reconcile_artifact(state: SQLiteStateStore, job_id: str, stage: Stage, artifact: Path, qa: Callable[[Path], QAResult], chapter_id: str='', panel_id: str='') -> bool:
    row=state.get(job_id, stage, chapter_id, panel_id)
    if not row or not artifact.exists():
        if row and row['state']=='COMPLETE': state.record(job_id, stage, State.RETRYABLE, chapter_id, panel_id, error_code='MISSING_ARTIFACT', retry_category=RetryCategory.CORRUPT_ARTIFACT)
        return False
    q=qa(artifact)
    if not q.ok:
        qdir=artifact.parent/'quarantine'; qdir.mkdir(exist_ok=True); artifact.rename(qdir/f'{artifact.name}.{int(time.time())}.bad')
        state.record(job_id, stage, State.RETRYABLE, chapter_id, panel_id, error_code='QA_FAILED', error_message=q.reason, retry_category=RetryCategory.CORRUPT_ARTIFACT)
        return False
    digest=checksum_file(artifact)
    if row.get('artifact_checksum') and row['artifact_checksum'] != digest: return False
    if row['state'] != 'COMPLETE': state.record(job_id, stage, State.COMPLETE, chapter_id, panel_id, artifact_path=artifact, artifact_checksum=digest, duration=q.duration)
    return True
