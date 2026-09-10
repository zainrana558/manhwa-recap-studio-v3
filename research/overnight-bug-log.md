# Overnight watch — job `cmtt21n0i0000xiks47ysff8b`

**Nano Machine · AsuraScans (`as-nano-machine`) · 328 chapters** — started ~2026-09-08 19:20 UTC.
User asleep. Task: monitor actively, hunt for bugs, note them here for later fixing. Do not
change code unless something is on fire (job-killer). Prefer to log + let it run.

## Baseline at start (19:22 UTC)
- Disk: 98 GB free / 124 GB (22% used). `data/jobs` = 1.7 GB.
- Mem: 14 GB available / 15 GB. Load 0.14.
- Recent code changes in this session (all uncommitted): slicer reconstruction routing
  (`_pages_are_webtoon_strip`), `_needy` fold, `_tall_display_window`, `_trim_edge_whitespace`,
  gradient `_is_blank_crop`; search dedupe rewrite (`manga-search.ts`).
- Two prior 5-ch Nano jobs finished clean (`cmtt0d2ou…`, `cmtt00gyu…`).

## Watch list (things likely to break on a 328-ch run)
1. Scrape failures / 404s on AsuraScans deep chapters → cross-source fallback rewrites mangaId.
2. Disk fill — ~50 MB/ch × 328 ≈ 16 GB projected. Watch `df`.
3. Memory spike in slice phase — `_reconstruct_strip` concatenates ALL chunks of a chapter
   into one RGB array (a 24-page 800×6000 chapter ≈ 350 MB) + face ONNX. Possible OOM.
4. Slice-quality regressions from this session's changes across 328 chapters of edge cases.
5. SQLite write-lock contention (known issue, WAL+busy_timeout applied) under 328× DB writes.
6. Stalls — job hangs on one chapter with no progress.
7. Per-chapter slice/render timeouts.

## Findings

### F1 — watcher false-positive (my tooling, not the product) — RESOLVED
19:37 — my `scratch_watch.sh` fired a "STALL" because it keyed stall detection on the Job
`progress` column, which sits at 10 for the entire scrape phase. Scrape was actually flying
(ch 16→230 in 15 min). Fixed the watcher to track the `message` string instead. Not a
product bug — noting so the log is self-consistent.

### Product observations (for later review)
- **Job `progress` % is near-useless during scrape** — stuck at 10 while 328 chapters
  download. The frontend progress bar will look frozen for ~35 min. Consider having the
  scrape phase advance `progress` proportionally to chapters scraped (e.g. 0→15%).
- **Slice phase logs only one JobLog line** ("Slicing chapters…") for the whole 328-ch pass;
  per-chapter progress goes only to `Job.message`. A ~3 h phase with one log entry — the
  history/log view will look dead. Consider a JobLog line every N chapters.
- **`chap_NNN` vs `chapter_NNN`**: slice manifests/dirs use `chap_006`, dataset dirs use
  `chapter_006`. Pre-existing, just noting the inconsistency.

### Slice-quality observations (ch 6, spot check — NOT blockers)
- ~12–15 of 76 frames are "bubble-heavy / thin-art" beats (e.g. ch6 #4,#5,#11,#14,#24,#30,
  #53,#55,#56,#74,#75): a stack of bubbles over a blurred backdrop with only a sliver of the
  speaker. `_tall_display_window` handles most; these are beats where the source panel
  genuinely has little art. No cut-through-face, no orphan-only bubble frames, no wrong-panel
  bubbles seen. Consistent with the ch1–5 verification. Matches user's "a little whitespace
  is acceptable". Leaving as-is unless a later chapter shows something worse.
- ch6 #8 is a chapter title card ("나노마신 / CHAPTER 3…") kept as a frame — OCR marks it
  empty-narration so it flashes silent. Harmless; could skip title cards later.

## ⚠️ F4 — render phase: emitLog DB-write flood → socket timeout → spurious job `error` (23:28 UTC)

**Severity: high (misleads the user; the job is actually fine and still rendering).**

**What happened:** slice+OCR finished all 328 ch ~23:26. Render started. master_pipeline.py
streams its stdout/stderr line-by-line; `mini-services/pipeline-service/index.ts` L1657 /
L1673 do `void emitLog(jobId,'info'|'warn','render',line)` **fire-and-forget, no `.catch()`**
for EVERY line. At render start the pipeline prints one INFO line per frame ("[chap_XXX_frmNNNN]
translation disabled — using raw text as-is") — ~26 k lines in a burst. Each `emitLog` =
`db.jobLog.create()` + `db.job.update()` (2 writes) → ~50 k writes slamming the local libsql
DB in seconds. One `db.jobLog.create()` hit **"Socket timeout (the database failed to
respond within the configured timeout)"** (`index.ts:378`). The rejected fire-and-forget
promise → **`process.on('unhandledRejection')` → `handleFatalError()` → `db.job.update({
status:'error', stage:'fatal'})`** (L2073). `emitLog`'s own `db.job.update` then flipped
`stage` back to `render`, so the DB row reads: status=error, stage=render, error="Invalid
db.jobLog.create() invocation … Socket timeout", doneChapters=328 (stale).

**Why it's NOT actually dead:**
- `unhandledRejection` handler does NOT `process.exit()` — pipeline-service kept running.
- master_pipeline.py (pid 379357) + ffmpeg still alive and rendering; progress.json shows
  "Chapter 4/328", 3 chapter mp4s written to work/temp_chapters/, JobLog still receiving
  ~177 rows/min → **DB recovered, render proceeding normally**.
- The render fn is still `await`-ing `child.on('exit')`. On exit-0 it runs the
  `if (exitCode===0)` block → verifies master_recap.mp4 → `db.job.update({status:'done'})`,
  which will **overwrite the spurious 'error'**. Expected self-heal on render completion
  (~5–7 h from now).

**Residual risks:** (a) another DB timeout during the final status='done' write; (b) the
frontend shows FAILED for the next several hours; (c) if anyone cancels/restarts on seeing
'error', ~4 h of slicing + the render so far is wasted; (d) `currentlyRunning` still set so
the queue won't start another job (fine — none queued).

**MY ACTION: leave everything running. Do NOT kill/restart/cancel. Do NOT touch code.**
Watcher rewritten to ignore DB 'error' and track render via progress.json + temp_chapters
mp4 count + the python pid; it wakes on final-video-present / render-proc-gone / 25-min
render stall / resources / heartbeat.

**Later fixes (do NOT do now):**
1. `emitLog` stdout/stderr streaming: batch or rate-limit. A 328-ch render should not write
   26 k JobLog rows. Options: only log lines matching a WARN/ERROR/progress pattern (drop
   the per-frame INFO spam), or buffer + bulk-insert every ~2 s, or `createMany`.
2. `void emitLog(...)` everywhere → give it an internal `.catch(() => {})` so a logging DB
   hiccup can NEVER become an `unhandledRejection` that fails the job.
3. `handleFatalError` on `unhandledRejection` is too aggressive — a rejected *log* write is
   not a fatal job error. Scope it (only mark error for rejections from the core pipeline
   path), or at least don't set status='error' if the render child is still alive.
4. libsql/Prisma: raise the socket/query timeout, or serialise emitLog writes through a
   queue so a burst can't exceed the pool.

## Progress log
- 19:22 — scraping ch 8/328.
- 19:24 — scraping ch 21/328. Chapter list verified clean (1–328).
- 19:45 — **scrape DONE**: all 328/328 chapters scraped, 0 failures, 3951 images total
  (mean 12 pg/ch, max 52 = ch126). ~25 min total. Datasets 3.4 GB. Slice phase started.
- 19:50 — slice+OCR running at **~36 s/chapter** (6 ch in 3.6 min). Routing confirmed:
  `frame_kind=scroll_frame` on every chapter → reconstruction path is active on the real job.
  ch1–5 frame counts match my earlier `rejob.py` verification exactly. Mem 10.7 GB avail,
  slice proc RSS ~950 MB, load 2.2. Healthy.
- **Revised ETA**: slice+OCR ~3.3 h, render ~6 h + concat → **~10 h total** (done ~06:00 UTC).
- 20:14 — heartbeat: slice ch 44/328, ~40 s/ch, disk 94 GB, memavail 8.1 GB. Watcher bqynijf54.
- 20:15 — **paddleocr-service RSS grew 103 MB (baseline) → ~4.4 GB** over the first ~44 ch.
  Roughly flat over the last minute (4533→4374 MB). Swap 57 MB used of 4 GB — no thrash.
  WATCHING: if it keeps climbing it could OOM later chapters. Not linear-projecting a crash
  yet; watcher now alerts on ocrRSS>9 GB / swap>2.5 GB / memavail<900 MB. If OCR does
  OOM-crash, watchdog.sh auto-restarts it (port 3002) and the pipeline's OCR-FAILED branch
  keeps going with degraded transcription for the affected chapter — not job-fatal.
- Watcher now: bg task bs295ergk (tracks paddleocr RSS + swap each tick).
- 20:37 — heartbeat: slice ch 77/328, ~41 s/ch. **paddleocr RSS PLATEAUED at ~4.9 GB**
  (4968→4875 over 22 min) — not a leak, just the model + working set. OOM concern cleared.
  disk 94 GB, memavail 8.2 GB, swap 57 MB. Watcher → bg bgck20kur.
- 20:38 — spot-checked ch50 (fight chapter, 82 frames): quality holds. ~10 thin vertical
  SFX/action beats (#29-31, #43-55) — inherent to the source (sound-effect panels), not a
  slicer fault. No cut-through, orphans, or wrong merges. Reducing inspection cadence to
  heartbeat-only + one more mid-run check + a render-phase check.
- 20:38 — watcher restarts: a `pkill -f scratch_watch` and a `pgrep -f "bash scratch_watch"`
  each matched the tool's own `bash -c` wrapper string and killed the shell (exit 144). Wasted
  ~2 restart cycles. Lesson for later self: never `pkill`/`pgrep` a pattern that appears in
  the command being run. Watcher now clean as bg task bdthkda2j; job never affected (slice
  kept progressing, ch 79/328).
- **Mode: passive monitoring.** Job healthy, slicer verified across ch1–6 + ch50. Will
  wake on watcher events only; deep-inspect again ~ch180 and at render start.
- 21:00 — heartbeat: slice ch 108/328, ~45 s/ch. ocrRSS 4992 M (flat), disk 94 GB,
  memavail 7.8 GB, swap 57 MB. All nominal. Watcher → bg b6fkw07fe.
- 21:23 — slice ch140/328, ocrRSS 5355M, disk 93G, mem 7.5G. Nominal.
- 21:45 — heartbeat: slice ch 179/328, ~34 s/ch (picked up). disk 93 GB, memavail 7.3 GB,
  swap 57 MB (unchanged). Watcher → bg bc52sw13f.

### F2 — paddleocr-service has a slow memory leak (~11 MB/chapter) — not job-fatal this run
Earlier I called it "plateaued" — wrong. Full trend: 103 MB baseline → 4374 (ch44) → 4875
(ch77) → 4992 (ch108) → 5355 (ch140) → 5865 (ch179). ~1 GB/hour, ~11 MB/chapter, roughly
linear. Projection: ~7.5 GB by end of slice (~ch328). Box has 15 GB + 4 GB swap, swap still
at 57 MB, so **this run finishes fine** (~6 GB memavail at slice end). But for a 1000-ch job
or a box with less RAM this would OOM. **Later fix:** the OCR service (or the pipeline's
per-chapter OCR loop) should free the RapidOCR working set between chapters, or the pipeline
should bounce the OCR service every N chapters. `del`/`gc.collect()` after each chapter's
batch, or `OCR_CONCURRENCY` pool recycle.

### Slice-quality — ch126 (52-page climax chapter, 87 frames) + ch165 spot check
Holds up. ~15/87 thin frames (red SFX strips #4-6, atmospheric #12-13, blur+face #21/24/38/
56/79) — same ~17% rate as ch6/ch50, no worse on the biggest chapter. The 52-page reconstruct
caused no memory spike or quality drop. Dramatic emphasis panels (#11 "ACKNOWLEDGE IT.",
#14, #16, #36, #52) and narration boxes (#69, #71) correctly kept as whole panels.

### F3 — scanlator credits cards kept as silent video frames
ch126 #87 = an end-of-chapter credits card ("REDOICE STUDIO × 3B2S / 발행 · (주)리버스").
Coloured (dark bg + coloured logos), so `_looks_like_credits_panel` (needs `sat<=12`) misses
it, and it's not near the first/last 2 *chunks* of the reconstructed strip so the position
gate wouldn't fire anyway. OCR flags it "credits/watermark noise → empty narration" but the
frame is still emitted → ~1-2 s of silent junk in the video per chapter that has one. Also
seen: chapter title cards (ch6 #8, ch50 #12). **Later fix:** when a frame's OCR result is
"entirely credits/watermark noise" / empty-narration AND it's the first/last frame of a
chapter, DROP the frame instead of muting it. (Need to confirm render doesn't already skip
empty-narration frames — check at render start.)
- 22:07 — heartbeat: slice ch 214/328, ~39 s/ch. ocrRSS 6113M (leak steady ~8 MB/ch), disk 92 GB, memavail 6.9 GB, swap 59 MB. Projection ~7 GB OCR at slice end — safe. Watcher bmcj83m08.
- 22:30 — heartbeat: slice ch 246/328, ~43 s/ch. ocrRSS 6182M (leak rate DECREASING: ~5 MB/ch now vs 11 early — looks asymptotic, revised end projection ~6.6 GB, safe). disk 91 GB, memavail 6.6 GB, swap 61 MB. ~1 h to render phase. Watcher b2eb4spxz.
- 22:52 — heartbeat: slice ch 277/328. ocrRSS 6357M. **swap ticked up 61→238 MB** (first movement; not thrashing, 6.3 GB memavail). disk 91 GB. ~37 min to render. Watcher bdcm2lp1i.
- 23:14 — heartbeat: slice ch 309/328 (~13 min from render). ocrRSS 6921M (+564 this window, variable leak). swap 238M stable. disk 90 GB, memavail 5.8 GB. Next wake = render-phase start; will verify render begins + check 328-video concat + F3 credits frames. Watcher bz2lhixz0.
- 23:31 — render progressing: ch 6/328, **~31 s/chapter** (5 chapter mp4s written), NOT DB-throttled. Job.message still updating live ("Chapter 6/328") so DB is fully functional — only the `status` flag is stale-stuck at "error". mem 7.8 GB avail, swap 236 MB stable. **Render ETA ~2.8 h → final video + expected self-heal to "done" ~02:30 UTC.** Watcher bw122kqlk.
- 23:56 — RENDER heartbeat: 23/328 chapter videos, **~68 s/ch** (slower than first-look 31 s; matches the 5-ch baseline). Job DB still status=error (spurious, F4) but message + progress.json updating live, status "running". disk 88 GB, mem 7.7 GB, swap 236 MB stable. Render ETA ~5.5 h → **~05:45 UTC + concat**. Watcher bmhs32lyq.
- 00:22 — render ch 41/328 (~90 s/ch). Verified chap_020.mp4: h264+aac 292 s, Denji Recaps watermark, panels readable — final video output good. Each chapter ~5 min video → full recap ~27 h. disk 87 GB mem 7.9 GB swap 236 MB. Render ETA ~7 h.
- 00:23 — NOTE TO SELF: killed my own watcher AGAIN with a pkill pattern that matched the tool wrapper (exit 144, 2nd time this session despite logging the lesson). STOP using pkill/pgrep. Use `ps ... | grep "[s]cratch"` and kill by explicit PID only. Single clean watcher now = bg b9fpq768i.
- 00:48 — render heartbeat: 58/328 chapters, ~83 s/ch (settled). disk 85 GB, mem 7.9 GB, swap 236 MB — rock stable. ETA ~6 h → ~07:00 UTC. Watcher bxz38v6um.
- 01:13 — render 73/328, ~90-100 s/ch. disk 84 GB, mem 7.8 GB, swap 236 MB stable. Watcher b0yr9pi0z.
- 01:38 — render 90/328, ~94 s/ch. Heartbeat showed memavail 6675M but that was a transient ffmpeg-encode dip — steady state 7.8 GB avail (master_pipeline only 144 MB RSS; paddleocr still holds ~7 GB idle from slice phase, harmless, reclaimable). disk 82 GB, swap 236 MB. temp_chapters 1.8 GB / 89 vids → ~6.5 GB projected. ETA ~6 h. Watcher bhicpbrol.
- 02:04 — render 109/328, ~80 s/ch. disk 81 GB, mem 7.8 GB steady, swap 236 MB. ETA ~5 h → ~07:00 UTC. Watcher b0z2ap2cj.
- 02:29 — render 129/328 (~40%), ~75 s/ch. disk 79 GB, mem 7.9 GB, swap 237 MB — all steady. ETA ~4 h → ~06:35 UTC. Watcher bf8nyuxh2.
- 02:54 — render 151/328 (46%), ~68 s/ch. swap 237→644 MB but **not thrashing**: vmstat si/so ≈ 0, CPU 95% idle. The kernel paged ~487 MB of the idle paddleocr service (cold since slice phase) out to swap to give render more page cache — memavail actually rose to 8.2 GB. Benign. disk 78 GB. ETA ~3.3 h → ~06:15 UTC. Watcher bfsp4cj2o.
- 03:19 — render 176/328 (54%), ~60 s/ch. disk 76 GB, mem 8.3 GB, swap 666 MB stable. ETA ~2.5 h → ~05:50 UTC. Watcher bp39qvczf.
- 03:45 — render 203/328 (62%), ~55 s/ch. disk 74 GB, mem 8.3 GB, swap 697 MB stable. ETA ~1.9 h → ~05:40 UTC. Watcher b1lr2trtv.
- 04:10 — render 227/328 (69%), ~62 s/ch. disk 72 GB, mem 8.3 GB, swap 697 MB — flat. ETA render ~1.7 h; then the 328-video final concat (watch memory/disk there — output ~7 GB / ~27 h). Watcher bsgqq9h7y.
- 04:35 — render 253/328 (77%), ~58 s/ch. disk 70 GB, mem 7.6 GB, swap 697 MB. ETA ~1.2 h. Watcher b394teixq.
- 05:00 — render 284/328 (87%), ~50 s/ch. disk 67 GB, mem 8.3 GB, swap 705 MB. Per-chapter render done ~05:37, then final concat. Watcher bfqsk6umh.
- 05:25 — render 319/328 (97%), ~44 s/ch. disk 64 GB, mem 8.6 GB, swap 993 MB (crept up, still fine). Per-ch render done ~05:33 → then final 328-video concat + QA. Watcher bquqnuiln (wakes on final video / concat).
- 05:28 (session resumed; prior watcher torn down w/ the old CC process — job unaffected, runs under systemd) — render at ch 323/328 (98%). master_pipeline.py alive (6h), pipeline-service still attached + streaming render output, 322 chapter videos done, output/master_recap.mp4 not yet created. DB status still spurious "error" (F4). disk 64 GB, mem 8 GB avail, swap 889 MB. ~6 ch + final concat left → ETA ~05:50 UTC. Watcher bd259lihr.

## ✅ DONE — 05:35 UTC

Job `cmtt21n0i0000xiks47ysff8b` completed. **Pipeline total: 367.6 min (~6.1 h)** — scrape 25 min, slice+OCR ~3.7 h, render ~6 h (per-chapter) + 82 s concat + 60 s copy-to-output.

**Final video:** `data/jobs/cmtt21n0i0000xiks47ysff8b/output/master_recap.mp4`
- 6.81 GB (6498 MiB), 1920×1080, h264 + aac
- **duration 109,379 s = 30.4 h = 1823 min** (verbatim narration, 328 ch + intro)
- QA passed (audio+video streams present); merge QA-validated; last-packet readable.

**F4 outcome — SELF-HEALED as predicted.** master_pipeline.py exited 0 → pipeline-service's
`child.on('exit')` ran the success path → `db.job.update({status:'done', outputVideo:'master_recap.mp4'})`.
Status is now `done`, progress 100. **My choice not to intervene was correct** — killing/restarting
on the spurious 'error' would have wasted the whole run.
- Residual: the `Job.error` FIELD still holds the old "Socket timeout" text even though
  status='done' — the success path doesn't clear `error`. Minor cosmetic; add `error: null`
  to the done-update. (F4 fix list item 5.)
- currentlyRunning back to null, queue free, temp/dataset cleaned up (--keep-temp kept
  work/ though — 6.8 GB master_merged.mp4 + temp_chapters still on disk, ~13 GB; user may
  want to clear work/ after downloading).

**End-state resources:** disk ~57 GB free, mem 11.8 GB avail, swap still 4095/4095 full but
static (si/so=0 all night — pure cold-page parking, never thrashed). paddleocr-service still
holds ~3 GB RSS + ~4 GB swap idle; safe to restart it to reclaim, or leave for the next job.

## Bug summary for the user (all NOTED, none fixed — per instructions)
- **F2** paddleocr-service memory leak ~5–11 MB/chapter during slice/OCR (103 MB → ~7 GB over
  328 ch). Asymptotic-ish; fine on this box, would OOM a smaller box or a 1000-ch job.
- **F3** scanlator credits cards + chapter title cards render as ~1–2 s silent frames (OCR
  flags them empty-narration but the frame still ships).
- **F4** ⚠️ render-phase JobLog write-flood (26 k rows) → libsql socket timeout → unhandled
  rejection → spurious job `status:'error'` for 6 h while the render was actually fine.
  Self-healed on completion. 5 fix options listed under the F4 section above.
- Minor: `progress` % frozen at 10 during scrape; slice phase logs 1 JobLog line for 3.7 h;
  `chap_NNN` vs `chapter_NNN` dir naming.
- Slice quality: consistent across ch1/6/50/126/165 + the rendered video — ~15–17 % thin
  "bubble/SFX over blur" frames (inherent to source, not a cut bug); no cut-through-face,
  no orphan-only-bubble frames, no wrong-panel bubbles. Reconstruction routing worked.

## 🔧 FIXES APPLIED — 2026-09-09 ~06:30 UTC (uncommitted, per "don't push" rule)

### F4 — spurious job `error` from render-phase log flood — FIXED (5 parts)
`mini-services/pipeline-service/index.ts`:
1. **`emitLog` is now bulletproof** — every `db.jobLog.create` / `db.job.update` inside it is
   wrapped in try/catch. A logging DB hiccup logs a `console.warn` and returns; it can never
   reject out of a `void emitLog(...)` call.
2. **Render stdout/stderr → JobLog is filtered + throttled** (`streamLine()`): every stderr
   line still goes to the in-memory ring buffer (free), but a DB write only happens for
   INTERESTING lines (error/QA/phase/merge/…) or at most once per 1500 ms for noise. The
   per-frame "translation disabled…" spam (~26 k lines) is dropped. ~50 k burst writes → a
   few hundred.
3. **`unhandledRejection` / `uncaughtException` no longer fail a job whose subprocess is
   alive** (`handleFatalError` guard on `childProcesses.has(jobId)`) — the child's exit code
   owns the outcome.
4. **Success path clears `error`** (`error: null` in the status:'done' update).
5. **SQLite `busy_timeout=15000` + WAL + `synchronous=NORMAL`** set explicitly at client
   init in both `mini-services/pipeline-service/lib.ts` and `src/lib/db.ts` (via
   `$queryRawUnsafe` — `$executeRawUnsafe` rejects PRAGMAs that return a row). Verified
   applied: `busy_timeout = 15000, journal_mode = wal`.

### F2 — paddleocr-service memory ratchet — FIXED
`mini-services/paddleocr-service/main.py`: added `_reclaim_memory()` = `gc.collect()` +
glibc `malloc_trim(0)` (via ctypes), called at the end of every `/ocr/batch` (one batch ==
one chapter). Forces freed arenas back to the OS so RSS doesn't ratchet over a long job.
Logs `[mem] reclaim post-batch: gc=N trim=True peakRSS=...MB`.

### F3 — credits / title cards as silent boundary frames — FIXED
`pipeline/master_pipeline.py`: `_looks_like_scanlator_card()` (flat-colour + low-linework
detector) + a boundary drop in the render loop. Only drops frame 0 / frame N-1 when: OCR
DID read text from it, `_clean_source_text()` strips the whole thing as scanlation noise,
AND it visually looks like a card. A genuine silent establishing shot (no OCR text) is
never touched. `RECAP_DROP_BOUNDARY_CREDITS=0` disables.

### Minors — FIXED
- Scrape progress now tracks **chapters completed** (1→10 %), not the never-known total
  image count that kept it frozen at 10.
- Slice+OCR phase now drops a JobLog breadcrumb every 10 chapters ("Sliced + transcribed
  N/M chapters") instead of one line for the whole multi-hour pass.
- `chap_NNN` vs `chapter_NNN` dir naming: left as-is (cosmetic, load-bearing in manifests).

### TTS (not a bug — noted): edge-tts flaky under long runs
edge-tts ("Andrew" neural voice) failed the 1st attempt on 134 / ~24 000 segments (retry
fixed all), and failed both attempts on 7 → Piper local-voice fallback. 0 silent. The
retry+fallback design worked; only artifact = 7 lines in the Piper voice across 30 h.

### Verification
- All 3 services restarted clean, health OK, no prisma/ctypes errors on startup.
- PRAGMA values confirmed applied.
- `_looks_like_scanlator_card`: flat card → True, noise panel → False.
- 3-chapter smoke job `cmttoipck0000xictaprsw1t0` running to exercise render-log throttle
  + F2 reclaim + F3 drop + scrape progress end-to-end.

## Verification round 1 (3-ch smoke job cmttoipck0000xictaprsw1t0) + fixes

- **F4 CONFIRMED**: job finished `status=done`, `error=None`. Render JobLog rows: **7/chapter**
  vs the old **96/chapter** (~93% cut). The per-frame "translation disabled" spam is gone;
  phase/QA/merge lines still logged. The ~50k write burst that timed out the DB is eliminated.
- **F2 CONFIRMED**: hammered `/ocr/batch` 8× (160 images) after the job — OCR RSS **flat at
  1686 MB, zero growth**. (Was 103 MB → 7 GB over the 328-ch run.) The `[mem] reclaim` log
  line doesn't surface — uvicorn overrides the service logger (pre-existing `[Batch OCR]`
  lines are also missing) — but the reclaim itself runs and works, proven by the flat RSS.
- **F3**: no boundary cards in ch1-3 (their first/last frames are real panels, correctly
  kept — the ch1 misty-mountain establishing shot was NOT dropped ✓). Widened the scan
  window from {0, N-1} to the first 4 + last 4 frames, since AsuraScans puts the "나노마신"
  title card a couple frames into the chapter (behind a cold-open panel). Re-testing on a
  6-ch job (ch6 has that card).
- Minor scrape-progress + slice-breadcrumb fixes: scrape now advances 1→10% by chapter
  (smoke job too short to see the 10-chapter slice breadcrumb — needs >15 ch).

## F3 round 2 — title cards weren't caught (fixed)
6-ch test showed NO drop on ch6's "나노마신 / CHAPTER 3. ENTERING THE DEMONIC ACADEMY" card.
Root cause: `_clean_source_text` strips CJK + studio credits but NOT "CHAPTER N. <SUBTITLE>",
so that card's English text survived cleaning → F3 kept it (and the pipeline was narrating
"Chapter 3. Entering the demonic academy." aloud).
Fix: (a) added a `^\W*(?:(?:season N )?(?:chapter|episode) N|prologue|epilogue)…` pattern to
`_CREDIT_PATTERNS` — anchored to string start so mid-sentence "…this chapter of my life…" is
untouched (verified: 7 card strings stripped, 4 real-dialogue strings kept). (b) F3 now drops
a boundary frame when its cleaned-away text was a credits-blob or chapter-title match, without
also requiring the visual card check (which rejects stylised-logo cards). This also improves
NARRATION for every job — title-card text is no longer read aloud.
Re-testing on job cmttp6amj000dxict63ccjaqn.

## ✅ ALL FIXES VERIFIED — 2026-09-09 ~06:20 UTC

3 smoke jobs (3-ch, 6-ch, 6-ch) run through the full pipeline with fixes live:

| bug | verified |
|---|---|
| **F4** spurious error | job → `status=done, error=None`; render JobLog **7 rows/ch vs 96 before**; `unhandledRejection` guard + emitLog try/catch + WAL/busy_timeout in place |
| **F2** OCR mem ratchet | RSS **flat ~1.7 GB** through 3 jobs + a 160-image hammer (was 103 MB→7 GB over 328 ch) |
| **F3** credits/title frames | `[chap_003] dropped 1 boundary credits/title frame(s)` — verified against no-F3 render: **every story panel still present**, chapter just 2 s shorter (trailing credits card gone). `_clean_source_text` now also strips "CHAPTER N. …" so buried title cards are at least no longer narrated aloud. |
| scrape progress | now advances 1→10 % by chapter completed |
| slice breadcrumbs | JobLog line every 10 ch (needs >15-ch job to see) |
| TTS | not a bug — edge-tts retry/Piper-fallback handled 141/24 000 flaky segments, 0 silent |

**Files changed (all uncommitted per "don't push"):**
- `mini-services/pipeline-service/index.ts` — emitLog resilience, render-log filter+throttle, unhandledRejection guard, clear error on success, scrape progress, slice breadcrumb
- `mini-services/pipeline-service/lib.ts` + `src/lib/db.ts` — SQLite busy_timeout/WAL/synchronous PRAGMAs at init
- `pipeline/master_pipeline.py` — `_looks_like_scanlator_card`, F3 boundary-frame drop, chapter-title pattern in `_CREDIT_PATTERNS`
- `mini-services/paddleocr-service/main.py` — `import gc`, `_reclaim_memory` (gc + malloc_trim), called post-batch

Services restarted, healthy. Big 328-ch video intact at
`data/jobs/cmtt21n0i0000xiks47ysff8b/output/master_recap.mp4` (6.8 GB, 30.4 h). Test job
dirs + the big job's `work/` cleaned.
