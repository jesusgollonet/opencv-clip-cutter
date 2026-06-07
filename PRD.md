# OpenCV Clip Cutter — Product Requirements Document

## Problem

Recording a skate or BMX session on a tripod produces long, unedited video files. Manually cutting individual attempts out of a 30-60 minute session is tedious. This tool automates that process.

## Target User

Skaters and BMX riders who:
- Record solo sessions on a static camera (tripod)
- Attempt the same trick repeatedly
- Want clips to review technique or share, without manual editing

**Constraints that hold:** static camera, repetitive attempts, person is the dominant motion source. Incidental background motion (wind, changing light, birds) must be tolerated. Other people in frame is a future concern, not in scope now.

---

## Phases

### Phase 1 — Robust CLI (current focus)

A well-packaged command-line tool that runs locally on the user's machine.

#### Goals
- Reliably detect motion segments across a variety of real session footage
- Handle incidental background noise without false positives or missed clips
- Fast feedback loop for tuning detection parameters

#### Detection Approach

Keep background subtraction (KNN) as the motion signal source. Improve robustness by applying signal processing to the white pixel time series:

1. **Smooth the signal** — apply a low-pass filter (Gaussian or rolling average) to eliminate transient noise spikes
2. **Peak detection** — use `scipy.signal.find_peaks` with minimum duration and minimum gap constraints to find activity bursts, replacing the current naive start/end threshold logic
3. **Hysteresis thresholding** — principled separate thresholds for segment start and end (already partially in place, to be made more robust)

**Future techniques (not in scope for Phase 1):**
- Level 2: Optical flow (Lucas-Kanade sparse) as alternative/complement to background subtraction — more robust to lighting drift
- Level 3: Template matching on the motion signal — exploit repetitive structure of trick attempts

#### Validation Tool

A static matplotlib plot (not real-time) showing:
- Full motion signal for the entire video
- Smoothed signal overlaid
- Detected segment boundaries marked
- Threshold lines visible

This is the primary development tool for tuning detection across many clips quickly.

#### CLI Interface

```bash
python main.py <video_path> [options]

Options:
  --sensitivity FLOAT       Motion detection sensitivity (default: 1.0)
  --min-clip-duration INT   Minimum clip length in seconds (default: 2)
  --min-gap INT             Minimum gap between clips in seconds (default: 3)
  --plot                    Output a detection plot instead of cutting clips
  --output-dir PATH         Where to save clips (default: alongside source video)
```

#### Output

Cut MP4 files named by timestamp offset:
```
session_001_0m32s.mp4
session_002_1m14s.mp4
...
```

No re-encoding — cuts are made on the original file via ffmpeg stream copy.

---

### Phase 2 — Hosted Web App (future)

Users upload a session video, processing runs on the server, they download their clips.

#### Key constraints inherited from Phase 1
- Source files are 30-60 min, potentially 5-15 GB — **chunked/resumable upload required**
- Detection runs on a downscaled copy; cuts are made on the original
- Processing time per video is non-trivial — async job queue needed

#### Fine Tuning UI
After detection runs, user sees:
- Motion signal graph with detected segments overlaid
- Per-segment: accept / reject / trim boundaries
- Download selected clips as a zip

#### Fine Tuning (Phase 1 CLI)
- `--sensitivity` and `--min-clip-duration` flags cover 90% of cases
- No UI needed for Phase 1

---

## Out of Scope (for now)

- Multiple people in frame / subject tracking
- Moving camera (handheld, follow shots)
- Edit Decision List (EDL/XML) export
- Template/example-based learning (Level 3 detection)
- Any auth, payments, or user accounts (Phase 2 concern)

---

## Success Criteria (Phase 1)

- Runs cleanly on a fresh machine with `pip install -r requirements.txt`
- Correctly detects >90% of attempts in a test set of 5+ real session videos
- False positive rate <10% (clips cut where nothing interesting happened)
- `--plot` mode makes it obvious why a segment was or wasn't detected
- `--sensitivity` adjustment visibly and predictably changes detection behavior
