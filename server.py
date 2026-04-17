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

from faster_whisper import WhisperModel
print("Whisper medium 로딩 중...", flush=True)
whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
print("Whisper 로딩 완료!", flush=True)

def allowed_file(f):
    return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED

# ══════════════════════════════════════════════════
# 1. librosa chroma 코드 감지
# ══════════════════════════════════════════════════
def detect_chords_chroma(audio_path, sr=22050):
    import librosa
    y, sr  = librosa.load(audio_path, sr=sr, mono=True)
    y_harm, _ = librosa.effects.hpss(y)
    hop    = 4096
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop, bins_per_octave=36)

    templates = {
        ""    : [1,0,0,0,1,0,0,1,0,0,0,0],
        "m"   : [1,0,0,1,0,0,0,1,0,0,0,0],
        "7"   : [1,0,0,0,1,0,0,1,0,0,1,0],
        "M7"  : [1,0,0,0,1,0,0,1,0,0,0,1],
        "m7"  : [1,0,0,1,0,0,0,1,0,0,1,0],
        "sus4": [1,0,0,0,0,1,0,1,0,0,0,0],
        "sus2": [1,0,1,0,0,0,0,1,0,0,0,0],
    }

    duration    = librosa.get_duration(y=y, sr=sr)
    frames      = chroma.shape[1]
    secs_per_fr = duration / frames
    win_frames  = max(1, int(2.0 / secs_per_fr))

    result, prev = [], None
    for i in range(0, frames, win_frames):
        chunk = chroma[:, i:i+win_frames].mean(axis=1)
        best_score, best_chord = -1, "C"
        for root_idx in range(12):
            rotated = np.roll(chunk, -root_idx)
            for suffix, tmpl in templates.items():
                score = np.dot(rotated, tmpl)
                if score > best_score:
                    best_score = score
                    best_chord = NOTES[root_idx] + suffix
        time = i * secs_per_fr
        if best_chord != prev:
            result.append({"chord": best_chord, "time": round(time, 2)})
            prev = best_chord

    return result, duration

# ══════════════════════════════════════════════════
# 2. pyin 멜로디 + 보이스 감지
# ══════════════════════════════════════════════════
def detect_melody_pyin(audio_path, sr=22050):
    import librosa
    y, sr  = librosa.load(audio_path, sr=sr, mono=True)
    hop    = 512
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C3'),
        fmax=librosa.note_to_hz('C6'),
        sr=sr, hop_length=hop,
    )
    times        = librosa.times_like(f0, sr=sr, hop_length=hop)
    melody_notes = []
    prev_name    = None

    for i, (freq, voiced) in enumerate(zip(f0, voiced_flag)):
        if not voiced or freq is None or np.isnan(freq): continue
        midi_int = int(round(librosa.hz_to_midi(freq)))
        name     = NOTES[midi_int % 12]
        octave   = (midi_int // 12) - 1
        time     = float(times[i])
        if name != prev_name:
            melody_notes.append({
                "name": name, "octave": octave, "midi": midi_int,
                "start": round(time, 3), "end": round(time + 0.25, 3),
                "isSharp": "#" in name, "displayName": name + str(octave),
            })
            prev_name = name

    # voiced_flag 타임라인 반환 (멘트 감지용)
    voiced_timeline = [(float(times[i]), bool(voiced_flag[i]))
                       for i in range(len(times))]
    return melody_notes[:64], voiced_timeline

# ══════════════════════════════════════════════════
# 3. 멘트 vs 가사 구분
# ══════════════════════════════════════════════════
def classify_segments(lyrics_segments, voiced_timeline):
    """
    각 가사 세그먼트가 '노래(가사)'인지 '말(멘트)'인지 구분
    voiced_flag 비율로 판단:
      voiced 비율 > 55% → 가사 (노래)
      voiced 비율 <= 55% → 멘트 (선포/말)
    """
    result = []
    for seg in lyrics_segments:
        start, end = seg["start"], seg["end"]

        # 이 구간의 voiced 프레임 비율 계산
        frames_in_seg = [(t, v) for t, v in voiced_timeline if start <= t <= end]
        if not frames_in_seg:
            voiced_ratio = 0.0
        else:
            voiced_ratio = sum(1 for _, v in frames_in_seg if v) / len(frames_in_seg)

        # 분류
        if voiced_ratio > 0.55:
            seg_type = "lyrics"   # 노래 (가사)
        elif voiced_ratio > 0.15:
            seg_type = "spoken"   # 멘트 (선포)
        else:
            seg_type = "silence"  # 무음

        result.append({
            "text"        : seg["text"].strip(),
            "start"       : seg["start"],
            "end"         : seg["end"],
            "type"        : seg_type,
            "voicedRatio" : round(voiced_ratio, 2),
        })

    return result

# ══════════════════════════════════════════════════
# 4. BPM 감지
# ══════════════════════════════════════════════════
def detect_bpm(audio_path, sr=22050):
    import librosa
    y, sr  = librosa.load(audio_path, sr=sr, mono=True, duration=60)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm    = round(float(tempo))
    while bpm > 110: bpm = round(bpm / 2)
    while bpm < 50:  bpm = round(bpm * 2)
    return bpm

# ══════════════════════════════════════════════════
# 5. 조성 감지
# ══════════════════════════════════════════════════
def detect_key_from_chords(chords_result):
    major = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
    minor = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]
    counts = [0.0] * 12
    for c in chords_result:
        ch   = c["chord"]
        root = ch[:2] if len(ch) > 1 and ch[1] == "#" else ch[:1]
        if root in NOTES:
            counts[NOTES.index(root)] += 1.0
    best_key, best_mode, best_corr = "C", "major", -999
    for i in range(12):
        cm = float(np.corrcoef(counts, major[i:]+major[:i])[0,1])
        if cm > best_corr: best_corr, best_key, best_mode = cm, NOTES[i], "major"
        cn = float(np.corrcoef(counts, minor[i:]+minor[:i])[0,1])
        if cn > best_corr: best_corr, best_key, best_mode = cn, NOTES[i], "minor"
    return best_key, best_mode

# ══════════════════════════════════════════════════
# 6. Demucs 음원 분리
# ══════════════════════════════════════════════════
def separate_audio(input_path, output_dir):
    try:
        print("Demucs 음원 분리 중...", flush=True)
        result = subprocess.run(
            ["python3", "-m", "demucs", "--two-stems", "vocals",
             "--out", output_dir, "--mp3", input_path],
            capture_output=True, text=True, timeout=300
        )
        print(f"Demucs returncode: {result.returncode}", flush=True)
        if result.stderr: print(f"Demucs stderr: {result.stderr[:300]}", flush=True)
        if result.returncode != 0: return None, None

        out            = Path(output_dir)
        vocal_path     = str(list(out.rglob("vocals.mp3"))[0])    if list(out.rglob("vocals.mp3"))    else None
        no_vocal_path  = str(list(out.rglob("no_vocals.mp3"))[0]) if list(out.rglob("no_vocals.mp3")) else None
        print(f"분리 완료! 보컬:{vocal_path is not None}", flush=True)
        return vocal_path, no_vocal_path
    except Exception as e:
        print(f"Demucs 오류: {e}", flush=True)
        return None, None

# ══════════════════════════════════════════════════
# 7. 구간 감지
# ══════════════════════════════════════════════════
def detect_sections(chords_result, classified_segments, total_duration):
    vocal_times = set()
    for seg in classified_segments:
        if seg["type"] == "lyrics":
            t = seg["start"]
            while t < seg["end"]:
                vocal_times.add(int(t))
                t += 1.0

    window = 8.0
    sections = []
    prev_label = None
    vocal_count = instr_count = 0
    t = 0.0

    while t < total_duration:
        end      = min(t + window, total_duration)
        is_vocal = any(int(x) in vocal_times for x in np.arange(t, end, 1.0))

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
            if t < 10:                               name, display = "intro",      "인트로"
            elif end >= total_duration - 10:         name, display = "outro",      "아웃트로"
            elif prev_label == "vocal":              name, display = "interlude",  "간주"
            else:                                    name, display = "bridge",     "브릿지"

        if sections and sections[-1]["name"] == name:
            sections[-1]["end"]      = round(end, 1)
            sections[-1]["duration"] = round(end - sections[-1]["start"], 1)
            sections[-1]["chords"]   = list(dict.fromkeys(sections[-1]["chords"] + seg_chords))[:8]
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
# 8. 정량화
# ══════════════════════════════════════════════════
def quantize_notes(melody_notes, bpm):
    if not melody_notes or bpm <= 0: return []
    beat_dur = 60.0 / bpm
    result   = []
    for n in melody_notes:
        dur   = n["end"] - n["start"]
        beats = max(0.5, min(4.0, round(dur / beat_dur * 2) / 2))
        result.append({
            "pitch": n["name"], "octave": n["octave"], "beats": beats,
            "duration": beats_to_dur(beats), "isRest": False, "midi": n["midi"],
        })
    return result[:128]

def beats_to_dur(b):
    if b >= 4: return "w"
    if b >= 3: return "hd"
    if b >= 2: return "h"
    if b >= 1.5: return "qd"
    if b >= 1: return "q"
    return "8"

def chords_at_beats(chords_result, bpm):
    beat_dur = 60.0 / bpm if bpm > 0 else 0.5
    return [{"chord": c["chord"], "beat": round(c["time"] / beat_dur, 1)} for c in chords_result]

# ══════════════════════════════════════════════════
# API
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

    suffix   = '.' + file.filename.rsplit('.', 1)[1].lower()
    tmp_dir  = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "input" + suffix)
    file.save(tmp_path)

    try:
        # Demucs 분리
        sep_dir = os.path.join(tmp_dir, "separated")
        os.makedirs(sep_dir, exist_ok=True)
        vocal_path, no_vocal_path = separate_audio(tmp_path, sep_dir)
        used_demucs   = vocal_path is not None
        chord_source  = no_vocal_path if no_vocal_path else tmp_path
        vocal_source  = vocal_path    if vocal_path    else tmp_path

        # 1. 코드
        print("코드 감지 중...", flush=True)
        chords_result, total_duration = detect_chords_chroma(chord_source)
        print(f"코드 {len(chords_result)}개", flush=True)

        # 2. 멜로디 + voiced timeline
        print("멜로디 감지 중...", flush=True)
        melody_notes, voiced_timeline = detect_melody_pyin(vocal_source)
        print(f"멜로디 {len(melody_notes)}개", flush=True)

        # 3. BPM
        bpm = detect_bpm(tmp_path)
        print(f"BPM: {bpm}", flush=True)

        # 4. 조성
        detected_key, mode = detect_key_from_chords(chords_result)
        key_display = detected_key + ("" if mode == "major" else "m")
        print(f"조성: {key_display}", flush=True)

        # 5. Whisper 가사 인식
        print("Whisper 가사 인식 중...", flush=True)
        raw_segments  = []
        detected_lang = "unknown"
        try:
            segments_gen, info = whisper_model.transcribe(
                vocal_source,
                language="ko",
                task="transcribe",
                beam_size=5,
                initial_prompt="찬양 예배 하나님 주님 예수 그리스도 할렐루야",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            detected_lang = info.language
            for seg in list(segments_gen):
                text = seg.text.strip()
                if text and len(text) > 1:
                    raw_segments.append({
                        "text" : text,
                        "start": round(seg.start, 2),
                        "end"  : round(seg.end,   2),
                    })
            print(f"가사 {len(raw_segments)}개 세그먼트", flush=True)
        except Exception as e:
            print(f"Whisper 오류: {e}", flush=True)

        # 6. 멘트 vs 가사 분류
        classified = classify_segments(raw_segments, voiced_timeline)

        # 가사만 추출
        lyrics_only = [s for s in classified if s["type"] == "lyrics"]
        spoken_only = [s for s in classified if s["type"] == "spoken"]
        full_lyrics = "\n".join(s["text"] for s in lyrics_only)

        print(f"가사: {len(lyrics_only)}개 / 멘트: {len(spoken_only)}개", flush=True)

        # 7. 구간 감지
        sections = detect_sections(chords_result, classified, total_duration)
        print(f"구간: {[s['display'] for s in sections]}", flush=True)

        # 8. 가사+코드 매핑
        lyrics_with_chords = []
        for seg in lyrics_only:
            seg_chords = []
            for c in chords_result:
                if seg["start"] <= c["time"] <= seg["end"]:
                    if not seg_chords or seg_chords[-1] != c["chord"]:
                        seg_chords.append(c["chord"])
            lyrics_with_chords.append({
                "text": seg["text"], "chords": seg_chords,
                "start": seg["start"], "end": seg["end"],
            })

        # 멘트+코드 매핑
        spoken_with_chords = []
        for seg in spoken_only:
            seg_chords = []
            for c in chords_result:
                if seg["start"] <= c["time"] <= seg["end"]:
                    if not seg_chords or seg_chords[-1] != c["chord"]:
                        seg_chords.append(c["chord"])
            spoken_with_chords.append({
                "text": seg["text"], "chords": seg_chords,
                "start": seg["start"], "end": seg["end"],
            })

        # 9. 정량화
        quantized_notes = quantize_notes(melody_notes, bpm)
        chord_beats     = chords_at_beats(chords_result, bpm)

        print(f"완료! Key:{key_display} BPM:{bpm} Demucs:{used_demucs}", flush=True)

        return jsonify({
            "success"          : True,
            "key"              : key_display,
            "mode"             : mode,
            "bpm"              : bpm,
            "language"         : detected_lang,
            "duration"         : round(total_duration, 1),
            "usedDemucs"       : used_demucs,
            "noteCount"        : len(melody_notes),
            "chordCount"       : len(chords_result),
            "notes"            : melody_notes,
            "chords"           : [c["chord"] for c in chords_result],
            "chordsWithTime"   : chords_result,
            "lyrics"           : full_lyrics,
            "lyricsSegments"   : lyrics_with_chords,
            "spokenSegments"   : spoken_with_chords,
            "allSegments"      : classified,
            "sections"         : sections,
            "quantizedNotes"   : quantized_notes,
            "chordBeats"       : chord_beats,
            "timeSignature"    : "4/4",
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
