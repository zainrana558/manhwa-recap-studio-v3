"""Production hardening primitives for the recap pipeline.

Local-first, dependency-injection-friendly helpers used by the Python renderer
and covered by focused unit tests.  These helpers deliberately prefer explicit
FAILED/UNCERTAIN states over fabricated success.
"""
from __future__ import annotations

import hashlib, json, os, re, shutil, sqlite3, subprocess, time
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

    def wait_for_resources(self, path: Path, job_id: str='', stage: Stage|str=Stage.JOB, check_interval_sec: int=60, max_wait_sec: int=1800, logger: Any=None) -> ResourceStatus:
        status = self.check(path, job_id, stage)
        waited = 0
        while not status.ok and waited < max_wait_sec:
            msg = f"ResourceGuard threshold not met ({stage}): {status.reason}. Pausing {check_interval_sec}s for resources to recover (waited {waited}s/{max_wait_sec}s)..."
            if logger and hasattr(logger, "warning"):
                logger.warning(msg)
            else:
                print(msg)
            time.sleep(check_interval_sec)
            waited += check_interval_sec
            status = self.check(path, job_id, stage)
        return status

def classify_retry(exc: Exception|str) -> RetryCategory:
    s=str(exc).lower()
    if 'timeout' in s or 'timed out' in s: return RetryCategory.TIMEOUT
    # Check corrupt-artifact patterns BEFORE the generic provider-unavailable
    # "not found" check: ffprobe's own error text for a broken/truncated
    # media file is often literally "... not found" (e.g. "moov atom not
    # found"), which would otherwise false-match the missing-binary check
    # below and misclassify a corrupt video as a missing provider.
    if 'corrupt' in s or 'ffprobe' in s or 'moov atom' in s: return RetryCategory.CORRUPT_ARTIFACT
    if 'no such file' in s or 'not found' in s or 'missing binary' in s: return RetryCategory.PROVIDER_UNAVAILABLE
    if 'disk' in s or 'memory' in s or 'resource' in s or 'oom' in s: return RetryCategory.RESOURCE
    if 'quality' in s or 'silence' in s or 'silent' in s or 'uncertain' in s: return RetryCategory.QUALITY
    if 'invalid' in s or 'empty input' in s: return RetryCategory.INVALID_INPUT
    return RetryCategory.TRANSIENT

def atomic_promote(tmp: Path, final: Path) -> str:
    final.parent.mkdir(parents=True, exist_ok=True); digest=checksum_file(tmp); os.replace(tmp, final); return digest

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

class ArtifactStore:
    """QA-gated artifact store with temporary writes and quarantine."""
    def __init__(self, root: Path):
        self.root = root; self.tmp_dir = root/'tmp'; self.final_dir = root/'artifacts'; self.quarantine_dir = root/'quarantine'
        for d in (self.tmp_dir, self.final_dir, self.quarantine_dir): d.mkdir(parents=True, exist_ok=True)
    def temporary_path(self, name: str) -> Path:
        return self.tmp_dir / f'{name}.{os.getpid()}.{int(time.time()*1000)}.tmp'
    def final_path(self, name: str) -> Path:
        return self.final_dir / name
    def quarantine(self, path: Path, reason: str='corrupt') -> Optional[Path]:
        if not path.exists(): return None
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        dest = self.quarantine_dir / f'{path.name}.{int(time.time())}.{reason}.bad'
        path.replace(dest); return dest
    def promote(self, tmp: Path, final_name: str, qa: Callable[[Path], QAResult]) -> tuple[Path, str, QAResult]:
        q = qa(tmp)
        if not q.ok:
            self.quarantine(tmp, 'qa_failed')
            raise RuntimeError(f'artifact QA failed: {q.reason}')
        final = self.final_path(final_name)
        digest = atomic_promote(tmp, final)
        return final, digest, q


def ffprobe_json(path: Path, timeout: int=30) -> Dict[str, Any]:
    result = ProcessRunner().run(['ffprobe','-v','error','-print_format','json','-show_streams','-show_format',str(path)], timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or 'ffprobe failed')
    return json.loads(result.stdout or '{}')


def audio_qa(path: Path, narration_expected: bool=True, min_duration: float=0.05) -> QAResult:
    if not path.exists() or path.stat().st_size == 0:
        return QAResult(False, 'missing_or_empty_audio')
    try:
        meta = ffprobe_json(path)
        dur = float(meta.get('format', {}).get('duration') or 0)
        if dur < min_duration: return QAResult(False, f'audio_duration_too_short:{dur}', dur, meta)
        if not any(s.get('codec_type') == 'audio' for s in meta.get('streams', [])):
            return QAResult(False, 'no_audio_stream', dur, meta)
        if narration_expected:
            vol = ProcessRunner().run(['ffmpeg','-hide_banner','-i',str(path),'-af','volumedetect','-f','null','-'], timeout=30)
            stderr = (vol.stderr or '') + (vol.stdout or '')
            if vol.returncode != 0: return QAResult(False, 'audio_energy_probe_failed', dur, meta)
            # Lossy-encoded (MP3) silence carries quantization noise and
            # reads as roughly -90dB, not literally "-inf" — matching only
            # the exact string "mean_volume: -inf" means this check almost
            # never fires against real chapter-audio-track output (which is
            # always MP3), since raw digital silence only stays exactly
            # -inf through an uncompressed path. Use a numeric threshold on
            # max_volume instead, same approach already used and proven at
            # the per-segment level in master_pipeline.py's own _audio_qa.
            m = re.search(r'max_volume:\s*(-?\d+(?:\.\d+)?) dB', stderr)
            if m is None:
                return QAResult(False, 'audio_energy_probe_unparseable', dur, meta)
            if float(m.group(1)) <= -50.0:
                return QAResult(False, 'silent_audio', dur, meta)
        return QAResult(True, duration=dur, metadata=meta)
    except Exception as exc:
        return QAResult(False, f'ffprobe_audio_failed:{exc}')


def video_qa(path: Path, audio_expected: bool=True, expected_duration: Optional[float]=None, tolerance: float=1.0) -> QAResult:
    if not path.exists() or path.stat().st_size == 0:
        return QAResult(False, 'missing_or_empty_video')
    try:
        meta = ffprobe_json(path)
        dur = float(meta.get('format', {}).get('duration') or 0)
        streams = meta.get('streams', [])
        if not any(s.get('codec_type') == 'video' for s in streams): return QAResult(False, 'no_video_stream', dur, meta)
        if audio_expected and not any(s.get('codec_type') == 'audio' for s in streams): return QAResult(False, 'no_audio_stream', dur, meta)
        if expected_duration is not None and abs(dur - expected_duration) > tolerance:
            return QAResult(False, f'duration_out_of_tolerance:{dur}', dur, meta)
        return QAResult(True, duration=dur, metadata=meta)
    except Exception as exc:
        return QAResult(False, f'ffprobe_video_failed:{exc}')


KNOWN_SFX = {
    "BOOM", "BANG", "SWOOSH", "RUMBLE", "CRASH", "CLACK", "THUD", "SLAP",
    "WHAM", "BAM", "CLANG", "SMASH", "GASP", "UGH", "AHH", "OHH", "WHOOSH",
    "SNAP", "CRACK", "POP", "CLICK", "TAP", "ZAP", "POOF", "BLAST", "SCREECH",
    "ROAR", "GROWL", "HISS", "GRUNT", "SIGH", "RATTLE", "RUSTLE", "SQUEAK",
}

def clean_ocr_text_with_sfx(text: str) -> str:
    """Sanitize OCR text output for narration.

    - Strips non-ASCII / unprintable garbage characters.
    - Removes isolated non-word fragments and random 1-2 char letter clusters (e.g. "xz", "ll1").
    - Preserves standard sound effects (e.g. BOOM, BANG, SWOOSH, RUMBLE, CRASH, CLACK)
      and standard English words.
    - Normalizes recognized SFX words into clean readable text.
    - Performs sentence-boundary punctuation cleanup, OCR character substitutions, and end card corrections.
    """
    if not text:
        return ""

    # Sentence boundary symbol conversions & ellipsis normalization
    text = re.sub(r'\s*\b(minus|dash|underscore)\b\s*$', '...', text, flags=re.IGNORECASE)
    text = re.sub(r'\.{2,}', '...', text)

    # OCR character substitutions & mistranslation fixes
    text = re.sub(r'\bHO[0O]\b', 'HOO', text)
    text = re.sub(r'\bHO\s+O\b', 'HOO', text)
    text = re.sub(r'\bgood-curdling\b', 'blood-curdling', text, flags=re.IGNORECASE)
    text = re.sub(r'\bgood\s+curdling\b', 'blood-curdling', text, flags=re.IGNORECASE)

    # End card corrections
    text = re.sub(r'\bB\s+to\s+be\s+continued\.*', 'To Be Continued...', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*B\s+to\s+be\b(?!\s+continued)', 'To Be Continued', text, flags=re.IGNORECASE)
    text = re.sub(r'\.{2,}', '...', text)

    # Graphic logo / stylized title card filter
    text = re.sub(r'\bsouls?\s+lac(?:ing|e)\b', '', text, flags=re.IGNORECASE)

    # Strip non-ASCII / unprintable characters (keep standard ASCII printable range 32-126)
    ascii_clean = "".join(ch for ch in text if 32 <= ord(ch) <= 126)

    # Normalize whitespace
    words = ascii_clean.split()
    cleaned_words = []

    for word in words:
        # Separate trailing/leading punctuation for inspection
        match = re.match(r"^([^\w]*)([\w]+)([^\w]*)$", word)
        if not match:
            # Contains mixed characters or symbols only — skip isolated single non-word symbols
            if len(word) > 1 and any(c.isalnum() for c in word):
                cleaned_words.append(word)
            continue

        prefix, core, suffix = match.groups()
        core_upper = core.upper()

        # Check if core is a recognized SFX word
        if core_upper in KNOWN_SFX:
            # Preserve & normalize SFX word (e.g. capitalized standard word)
            normalized_core = core_upper if core.isupper() else core.capitalize()
            cleaned_words.append(f"{prefix}{normalized_core}{suffix}")
            continue

        # Filter out obvious 1-2 character random non-word fragments (e.g., "xz", "ll1", "q1")
        if len(core) <= 2:
            lower_core = core.lower()
            valid_short_words = {
                "i", "a", "in", "on", "at", "to", "do", "go", "he", "me", "my",
                "no", "so", "up", "us", "is", "it", "if", "am", "an", "as", "be",
                "by", "we", "or", "of", "ok", "oh", "ah", "ha", "hi", "ho", "um",
            }
            if lower_core not in valid_short_words and not core.isdigit():
                continue

        # Filter out clusters containing numbers mixed with letters unless digits-only or standard
        if re.search(r"[a-zA-Z]", core) and re.search(r"\d", core):
            continue

        cleaned_words.append(word)

    return " ".join(cleaned_words).strip()


def ocr_text_for_narration(result: OCRResult) -> str:
    if result.status == State.COMPLETE or str(result.status) == 'SUCCESS':
        return clean_ocr_text_with_sfx(result.text)
    return ''
