import os
import tempfile
import shutil
import subprocess
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify

app     = Flask(__name__)
ALLOWED = {'mp3', 'wav', 'm4a', 'aac', 'ogg'}
NOTES   = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# ── Whisper 로드 (medium = 한국어 정확도 향상) ──────
from faster_whisper import WhisperModel
print("Whisper medium 로딩 중...", flush=True)
whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
print("Whisper 로딩 완료!", flush=True)

def allowed_file(f):
    return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED

# ══════════════════════════════════════════════════
# 1. librosa chroma 코드 감지
# ══════════════════════════════════════════════════
def detect_chords_chroma(audio_path, hop_length=4096, sr=22050):
    """librosa chroma 특성으로 코드 감지 (반주 트랙에 최적)"""
    import librosa

    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    # Harmonic/percussive 분리 → 화음만 추출
    y_harm, _ = librosa.effects.hpss(y)

    # Chroma 특성 추출
    chroma = librosa.feature.chroma_cqt(
        y=y_harm, sr=sr, hop_length=hop_length, bins_per_octave=36
    )

    # 코드 템플릿 정의
    chord_templates = {
        ""      : [1,0,0,0,1,0,0,1,0,0,0,0],  # Major
        "m"     : [1,0,0,1,0,0,0,1,0,0,0,0],  # Minor
        "7"     : [1,0,0,0,1,0,0,1,0,0,1,0],  # Dominant 7
        "M7"    : [1,0,0,0,1,0,0,1,0,0,0,1],  # Major 7
        "m7"    : [1,0,0,1,0,0,0,1,0,0,1,0],  # Minor 7
        "sus4"  : [1,0,0,0,0,1,0,1,0,0,0,0],  # Sus4
        "sus2"  : [1,0,1,0,0,0,0,1,0,0,0,0],  # Sus2
    }

    duration    = librosa.get_duration(y=y, sr=sr)
    frames      = chroma.shape[1]
    secs_per_fr = duration / frames

    chords_result = []
    prev_chord    = None
    window_frames = max(1, int(2.0 / secs_per_fr))  # 2초 단위

    for i in range(0, frames, window_frames):
        chunk = chroma[:, i:i+window_frames].mean(axis=1)
        best_score, best_chord = -1, "C"

        for root_idx in range(12):
            rotated = np.roll(chunk, -root_idx)
            for suffix, template in chord_templates.items():
                score = np.dot(rotated, template)
                if score > best_score:
                    best_score  = score
                    best_chord  = NOTES[root_idx] + suffix

        time = i * secs_per_fr
        if best_chord != prev_chord:
            chords_result.append({"chord": best_chord, "time": round(time, 2)})
            prev_chord = best_chord

    return chords_result, duration

# ══════════════════════════════════════════════════
# 2. pyin 멜로디 감지
# ══════════════════════════════════════════════════
def detect_melody_pyin(audio_path, sr=22050):
    """pyin 알고리즘으로 보컬 멜로디만 정확하게 감지"""
    import librosa

    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C3'),   # 130Hz - 보컬 최저음
        fmax=librosa.note_to_hz('C6'),   # 1047Hz - 보컬 최고음
        sr=sr,
        hop_length=512,
    )

    times = librosa.times_like(f0, sr=sr, hop_length=512)
    melody_notes = []
    prev_name    = None

    for i, (freq, voiced) in enumerate(zip(f0, voiced_flag)):
        if not voiced or freq is None or np.isnan(freq):
            continue

        midi  = librosa.hz_to_midi(freq)
        midi_int = int(round(midi))
        name  = NOTES[midi_int % 12]
        octave = (midi_int // 12) - 1
        time  = float(times[i])

        if name != prev_name:
            melody_notes.append({
                "name"       : name,
                "octave"     : octave,
                "midi"       : midi_int,
                "start"      : round(time, 3),
                "end"        : round(time + 0.25, 3),
                "isSharp"    : "#" in name,
                "displayName": name + str(octave),
            })
            prev_name = name

    return melody_notes[:64]

# ══════════════════════════════════════════════════
# 3. 조성 감지
# ══════════════════════════════════════════════════
def detect_key_from_chords(chords_result):
    """코드 진행에서 조성 감지"""
    major = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
    minor = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]

    note_counts = [0.0] * 12
    for c in chords_result:
        chord = c["chord"]
        root  = chord[:2] if len(chord) > 1 and chord[1] == "#" else chord[:1]
        if root in NOTES:
            note_counts[NOTES.index(root)] += 1.0

    best_key, best_mode, best_corr = "C", "major", -999
    for i in range(12):
        cm = float(np.corrcoef(note_counts, major[i:]+major[:i])[0,1])
        if cm > best_corr: best_corr, best_key, best_mode = cm, NOTES[i], "major"
        cn = float(np.corrcoef(note_counts, minor[i:]+minor[:i])[0,1])
        if cn > best_corr: best_corr, best_key, best_mode = cn, NOTES[i], "minor"

    return best_key, best_mode

# ══════════════════════════════════════════════════
# 4. BPM 감지
# ══════════════════════════════════════════════════
def detect_bpm(audio_path, sr=22050):
    """librosa beat tracking으로 BPM 감지"""
    import librosa
    y, sr = librosa.load(audio_path, sr=sr, mono=True, duration=60)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(float(tempo))
    # 찬양곡 보정 (보통 60~100 BPM)
    while bpm > 110: bpm = round(bpm / 2)
    while bpm < 50:  bpm = round(bpm * 2)
    return bpm

# ══════════════════════════════════════════════════
# 5. Demucs 음원 분리
# ══════════════════════════════════════════════════
def separate_audio(input_path, output_dir):
    try:
        print("Demucs 음원 분리 중...", flush=True)
        result = subprocess.run(
            ["python3", "-m", "demucs",
             "--two-stems", "vocals",
             "--out", output_dir,
             "--mp3",
             input_path],
            capture_output=True, text=True, timeout=300
        )
        print(f"Demucs returncode: {result.returncode}", flush=True)
        if result.stderr:
            print(f"Demucs stderr: {result.stderr[:300]}", flush=True)

        if result.returncode != 0:
            return None, None

        out             = Path(output_dir)
        vocal_files     = list(out.rglob("vocals.mp3"))
        no_vocal_files  = list(out.rglob("no_vocals.mp3"))
        vocal_path      = str(vocal_files[0])    if vocal_files    else None
        no_vocal_path   = str(no_vocal_files[0]) if no_vocal_files else None

        print(f"분리 완료! 보컬:{vocal_path is not None}", flush=True)
        return vocal_path, no_vocal_path

    except Exception as e:
        print(f"Demucs 오류: {e}", flush=True)
        return None, None

# ══════════════════════════════════════════════════
# 6. 구간 감지
# ══════════════════════════════════════════════════
def detect_sections(chords_result, lyrics_segments, total_duration):
    """코드 변화 패턴 + 가사 구간으로 섹션 감지"""
    if not chords_result: return []

    # 가사 있는 구간 파악
    vocal_times = set()
    for seg in lyrics_segments:
        t = seg["start"]
        while t < seg["end"]:
            vocal_times.add(int(t))
            t += 1.0

    # 4마디(약 8~16초) 단위로 그룹핑
    window   = 8.0
    sections = []
    prev_label = None
    vocal_count = instr_count = 0

    t = 0.0
    while t < total_duration:
        end  = min(t + window, total_duration)
        is_vocal = any(int(x) in vocal_times for x in np.arange(t, end, 1.0))

        # 이 구간의 코드
        seg_chords = []
        for c in chords_result:
            if t <= c["time"] < end:
                if not seg_chords or seg_chords[-1] != c["chord"]:
                    seg_chords.append(c["chord"])

        if is_vocal:
            vocal_count += 1
            if vocal_count == 1:   name, display = "verse1",  "1절"
            elif vocal_count == 2: name, display = "chorus",  "후렴"
            elif vocal_count == 3: name, display = "verse2",  "2절"
            elif vocal_count == 4: name, display = "chorus2", "후렴 반복"
            else:                  name, display = f"section{vocal_count}", f"섹션{vocal_count}"
        else:
            instr_count += 1
            if t < 10:             name, display = "intro",      "인트로"
            elif end >= total_duration - 10: name, display = "outro", "아웃트로"
            elif prev_label in ["vocal"]:    name, display = "interlude", "간주"
            else:                  name, display = "bridge",    "브릿지"

        if sections and sections[-1]["name"] == name:
            sections[-1]["end"] = round(end, 1)
            sections[-1]["duration"] = round(end - sections[-1]["start"], 1)
            sections[-1]["chords"] = list(dict.fromkeys(sections[-1]["chords"] + seg_chords))[:8]
        else:
            sections.append({
                "name": name, "display": display,
                "start": round(t, 1), "end": round(end, 1),
                "duration": round(end - t, 1),
                "label": "vocal" if is_vocal else "instrumental",
                "chords": list(dict.fromkeys(seg_chords))[:8],
                "hasVocal": is_vocal,
            })
            prev_label = "vocal" if is_vocal else "instrumental"

        t += window

    return sections

# ══════════════════════════════════════════════════
# 7. 정량화 음표
# ══════════════════════════════════════════════════
def quantize_notes(melody_notes, bpm):
    if not melody_notes or bpm <= 0: return []
    beat_dur = 60.0 / bpm
    result   = []
    for n in melody_notes:
        dur   = n["end"] - n["start"]
        beats = max(0.5, min(4.0, round(dur / beat_dur * 2) / 2))
        result.append({
            "pitch": n["name"], "octave": n["octave"],
            "beats": beats, "duration": beats_to_dur(beats),
            "isRest": False, "midi": n["midi"],
        })
    return result[:128]

def beats_to_dur(beats):
    if beats >= 4:   return "w"
    if beats >= 3:   return "hd"
    if beats >= 2:   return "h"
    if beats >= 1.5: return "qd"
    if beats >= 1:   return "q"
    return "8"

def chords_at_beats(chords_result, bpm):
    beat_dur = 60.0 / bpm if bpm > 0 else 0.5
    return [{"chord": c["chord"], "beat": round(c["time"] / beat_dur, 1)}
            for c in chords_result]

# ══════════════════════════════════════════════════
# API 엔드포인트
# ══════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "WorshipSheet 서버 정상 작동 중"})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "파일이 없어요"}), 400
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "MP3, WAV, M4A, AAC만 가능해요"}), 400

    suffix  = '.' + file.filename.rsplit('.', 1)[1].lower()
    tmp_dir = tempfile.mkdtemp()
    tmp_path= os.path.join(tmp_dir, "input" + suffix)
    file.save(tmp_path)

    try:
        # ── Demucs 음원 분리 시도 ──────────────────
        sep_dir        = os.path.join(tmp_dir, "separated")
        os.makedirs(sep_dir, exist_ok=True)
        vocal_path, no_vocal_path = separate_audio(tmp_path, sep_dir)
        used_demucs    = vocal_path is not None

        chord_source   = no_vocal_path if no_vocal_path else tmp_path
        vocal_source   = vocal_path    if vocal_path    else tmp_path

        # ── 1. librosa chroma 코드 감지 ───────────
        print("코드 감지 중 (librosa chroma)...", flush=True)
        chords_result, total_duration = detect_chords_chroma(chord_source)
        print(f"코드 {len(chords_result)}개 감지", flush=True)

        # ── 2. pyin 멜로디 감지 ───────────────────
        print("멜로디 감지 중 (pyin)...", flush=True)
        melody_notes = detect_melody_pyin(vocal_source)
        print(f"멜로디 음표 {len(melody_notes)}개", flush=True)

        # ── 3. BPM 감지 ───────────────────────────
        print("BPM 감지 중...", flush=True)
        bpm = detect_bpm(tmp_path)
        print(f"BPM: {bpm}", flush=True)

        # ── 4. 조성 감지 ──────────────────────────
        detected_key, mode = detect_key_from_chords(chords_result)
        key_display = detected_key + ("" if mode == "major" else "m")
        print(f"조성: {key_display}", flush=True)

        # ── 5. Whisper 가사 인식 ──────────────────
        print("Whisper medium 가사 인식 중...", flush=True)
        lyrics_segments = []
        full_lyrics     = ""
        detected_lang   = "unknown"
        try:
            segments_gen, info = whisper_model.transcribe(
                vocal_source,
                language="ko",           # 한국어 명시
                task="transcribe",
                beam_size=5,
                initial_prompt="찬양, 예배, 하나님, 주님, 예수",  # 찬양 힌트
                vad_filter=True,         # 무음 구간 필터링
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            detected_lang  = info.language
            for seg in list(segments_gen):
                text = seg.text.strip()
                if text and len(text) > 1:
                    lyrics_segments.append({
                        "text" : text,
                        "start": round(seg.start, 2),
                        "end"  : round(seg.end,   2),
                    })
            full_lyrics = "\n".join(s["text"] for s in lyrics_segments)
            print(f"가사 완료! {len(lyrics_segments)}개 세그먼트", flush=True)
        except Exception as e:
            print(f"Whisper 오류: {e}", flush=True)

        # ── 6. 구간 감지 ──────────────────────────
        sections = detect_sections(chords_result, lyrics_segments, total_duration)
        print(f"구간: {[s['display'] for s in sections]}", flush=True)

        # ── 7. 가사 + 코드 매핑 ───────────────────
        lyrics_with_chords = []
        for seg in lyrics_segments:
            seg_chords = []
            for c in chords_result:
                if seg["start"] <= c["time"] <= seg["end"]:
                    if not seg_chords or seg_chords[-1] != c["chord"]:
                        seg_chords.append(c["chord"])
            lyrics_with_chords.append({
                "text": seg["text"].strip(),
                "chords": seg_chords,
                "start": seg["start"],
                "end": seg["end"],
            })

        # ── 8. 정량화 음표 ────────────────────────
        quantized_notes = quantize_notes(melody_notes, bpm)
        chord_beats     = chords_at_beats(chords_result, bpm)

        print(f"완료! Key:{key_display} BPM:{bpm} Demucs:{used_demucs}", flush=True)

        return jsonify({
            "success"        : True,
            "key"            : key_display,
            "mode"           : mode,
            "bpm"            : bpm,
            "language"       : detected_lang,
            "duration"       : round(total_duration, 1),
            "usedDemucs"     : used_demucs,
            "noteCount"      : len(melody_notes),
            "chordCount"     : len(chords_result),
            "notes"          : melody_notes,
            "chords"         : [c["chord"] for c in chords_result],
            "chordsWithTime" : chords_result,
            "lyrics"         : full_lyrics,
            "lyricsSegments" : lyrics_with_chords,
            "sections"       : sections,
            "quantizedNotes" : quantized_notes,
            "chordBeats"     : chord_beats,
            "timeSignature"  : "4/4",
        })

    except Exception as e:
        import traceback
        print("오류:", traceback.format_exc(), flush=True)
        return jsonify({"error": "분석 실패: " + str(e)}), 500

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
