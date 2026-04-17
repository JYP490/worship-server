import os
import tempfile
import shutil
import subprocess
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path

os.environ["BASIC_PITCH_MODEL"] = "onnx"

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from faster_whisper import WhisperModel

app     = Flask(__name__)
ALLOWED = {'mp3', 'wav', 'm4a', 'aac', 'ogg'}
NOTES   = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

print("Whisper 모델 로딩 중...", flush=True)
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
print("Whisper 모델 로딩 완료!", flush=True)

def allowed_file(f):
    return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED

def midi_to_note(midi):
    return NOTES[midi % 12], (midi // 12) - 1

def detect_key(note_counts):
    major = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
    minor = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]
    best_key, best_mode, best_corr = "C", "major", -999
    for i in range(12):
        cm = float(np.corrcoef(note_counts, major[i:]+major[:i])[0,1])
        if cm > best_corr: best_corr, best_key, best_mode = cm, NOTES[i], "major"
        cn = float(np.corrcoef(note_counts, minor[i:]+minor[:i])[0,1])
        if cn > best_corr: best_corr, best_key, best_mode = cn, NOTES[i], "minor"
    return best_key, best_mode

def notes_to_chord(active):
    if not active: return None
    idx  = sorted(set(n % 12 for n in active))
    root = NOTES[idx[0]]
    ivs  = [(i - idx[0]) % 12 for i in idx[1:]]
    if 4 in ivs and 7 in ivs and 11 in ivs: return root + "M7"
    if 3 in ivs and 7 in ivs and 10 in ivs: return root + "m7"
    if 4 in ivs and 7 in ivs and 10 in ivs: return root + "7"
    if 4 in ivs and 7 in ivs:               return root
    if 3 in ivs and 7 in ivs:               return root + "m"
    if 4 in ivs and 6 in ivs:               return root + "dim"
    if 4 in ivs and 8 in ivs:               return root + "aug"
    if 5 in ivs and 7 in ivs:               return root + "sus4"
    if 2 in ivs and 7 in ivs:               return root + "sus2"
    return root

def extract_melody(note_events, max_notes=64):
    melody, prev = [], None
    if not note_events: return melody
    max_time = max(float(n[1]) for n in note_events)
    t = 0.0
    while t < max_time:
        active = [n for n in note_events
                  if float(n[0]) <= t+0.25 and float(n[1]) >= t
                  and (float(n[3]) if len(n) > 3 else 1.0) > 0.5]
        if active:
            top  = max(active, key=lambda n: int(n[2]))
            midi = int(top[2])
            name, octave = midi_to_note(midi)
            if name != prev:
                melody.append({
                    "name": name, "octave": octave, "midi": midi,
                    "start": round(t, 3), "end": round(float(top[1]), 3),
                    "isSharp": "#" in name, "displayName": name + str(octave)
                })
                prev = name
        t += 0.25
    return melody[:max_notes]

def separate_audio(input_path, output_dir):
    """Demucs로 보컬/반주 분리"""
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
            print(f"Demucs stderr: {result.stderr[:500]}", flush=True)
        if result.stdout:
            print(f"Demucs stdout: {result.stdout[:200]}", flush=True)
        if result.returncode != 0:
            print(f"Demucs 실패!", flush=True)
            return None, None

        # 출력 파일 찾기
        out = Path(output_dir)
        vocal_files   = list(out.rglob("vocals.mp3"))
        no_vocal_files= list(out.rglob("no_vocals.mp3"))

        vocal_path    = str(vocal_files[0])   if vocal_files    else None
        no_vocal_path = str(no_vocal_files[0]) if no_vocal_files else None

        print(f"분리 완료! 보컬: {vocal_path}", flush=True)
        return vocal_path, no_vocal_path

    except subprocess.TimeoutExpired:
        print("Demucs 타임아웃", flush=True)
        return None, None
    except Exception as e:
        print(f"Demucs 예외: {e}", flush=True)
        return None, None

def detect_sections(note_events, lyrics_segments, total_duration):
    window    = 2.0
    n_windows = int(total_duration / window) + 1
    energy    = [0.0] * n_windows
    vocal_prob= [0.0] * n_windows

    for note in note_events:
        start = float(note[0])
        midi  = int(note[2])
        conf  = float(note[3]) if len(note) > 3 else 1.0
        idx   = int(start / window)
        if idx < n_windows:
            energy[idx] += conf
            if 60 <= midi <= 84:
                vocal_prob[idx] += conf

    for i in range(n_windows):
        if energy[i] > 0:
            vocal_prob[i] /= energy[i]

    vocal_windows = set()
    for seg in lyrics_segments:
        if len(seg.get("text","").strip()) > 2:
            s = int(seg["start"] / window)
            e = int(seg["end"]   / window) + 1
            for i in range(s, min(e, n_windows)):
                vocal_windows.add(i)

    avg_energy  = sum(energy) / max(1, len([e for e in energy if e > 0]))
    low_thresh  = avg_energy * 0.3
    high_thresh = avg_energy * 1.2

    labels = []
    for i in range(n_windows):
        e     = energy[i]
        vocal = i in vocal_windows or vocal_prob[i] > 0.4
        if e < low_thresh * 0.1:     labels.append("silence")
        elif vocal:                   labels.append("vocal")
        elif e < low_thresh:         labels.append("silence")
        elif e > high_thresh:        labels.append("climax")
        else:                         labels.append("instrumental")

    raw = []
    if labels:
        cur_label, cur_start = labels[0], 0
        for i in range(1, len(labels)):
            if labels[i] != cur_label:
                raw.append((cur_start * window, i * window, cur_label))
                cur_label, cur_start = labels[i], i
        raw.append((cur_start * window, total_duration, cur_label))

    filtered = [(s, e, l) for s, e, l in raw if l != "silence" and e - s >= 4.0]

    result = []
    vocal_count, instr_count, prev_label = 0, 0, None

    for i, (s, e, label) in enumerate(filtered):
        is_first = (i == 0)
        is_last  = (i == len(filtered) - 1)

        if label == "vocal":
            vocal_count += 1
            if vocal_count == 1:   name, display = "verse1",  "1절"
            elif vocal_count == 2: name, display = "chorus",  "후렴"
            elif vocal_count == 3: name, display = "verse2",  "2절"
            elif vocal_count == 4: name, display = "chorus2", "후렴 (반복)"
            else:                  name, display = f"section{vocal_count}", f"섹션 {vocal_count}"
        elif label == "climax":
            name, display = "climax", "클라이맥스"
        else:
            instr_count += 1
            if is_first:                             name, display = "intro",     "인트로"
            elif is_last:                            name, display = "outro",     "아웃트로"
            elif prev_label in ["vocal","climax"]:   name, display = "interlude", "간주"
            else:                                    name, display = f"instrumental{instr_count}", "악기 솔로"

        section_chords = []
        t2, prev_chord = s, None
        while t2 < e:
            active = [int(n[2]) for n in note_events
                      if float(n[0]) <= t2+0.5 and float(n[1]) >= t2
                      and s <= float(n[0]) <= e]
            chord = notes_to_chord(active)
            if chord and chord != prev_chord:
                section_chords.append(chord)
                prev_chord = chord
            t2 += 0.5

        unique_chords = list(dict.fromkeys(section_chords))[:8]
        result.append({
            "name": name, "display": display,
            "start": round(s, 1), "end": round(e, 1),
            "duration": round(e-s, 1), "label": label,
            "chords": unique_chords,
            "hasVocal": label in ["vocal","climax"],
        })
        prev_label = label

    return result

def quantize_notes(note_events, bpm):
    if not note_events or bpm <= 0: return []
    beat_dur  = 60.0 / bpm
    quantized = []
    prev_name = None
    max_time  = max(float(n[1]) for n in note_events)
    t = 0.0
    while t < max_time:
        active = [n for n in note_events
                  if float(n[0]) <= t+beat_dur*0.5 and float(n[1]) >= t
                  and (float(n[3]) if len(n) > 3 else 1.0) > 0.4]
        if active:
            top  = max(active, key=lambda n: int(n[2]))
            midi = int(top[2])
            name, octave = midi_to_note(midi)
            dur   = float(top[1]) - float(top[0])
            beats = max(0.5, min(4.0, round(dur/beat_dur*2)/2))
            if name != prev_name:
                quantized.append({
                    "pitch": name, "octave": octave, "beats": beats,
                    "duration": beats_to_dur(beats), "isRest": False, "midi": midi,
                })
                prev_name = name
        else:
            quantized.append({"pitch":"r","octave":4,"beats":1.0,
                              "duration":"q","isRest":True,"midi":0})
        t += beat_dur
    return quantized[:128]

def beats_to_dur(beats):
    if beats >= 4:   return "w"
    if beats >= 3:   return "hd"
    if beats >= 2:   return "h"
    if beats >= 1.5: return "qd"
    if beats >= 1:   return "q"
    if beats >= 0.5: return "8"
    return "16"

def chords_at_beats(chords_result, bpm):
    beat_dur = 60.0 / bpm if bpm > 0 else 0.5
    return [{"chord": c["chord"], "beat": round(c["time"]/beat_dur, 1)}
            for c in chords_result]

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

    vocal_path    = None
    no_vocal_path = None

    try:
        # ── 1. Demucs 음원 분리 ────────────────────
        sep_dir = os.path.join(tmp_dir, "separated")
        os.makedirs(sep_dir, exist_ok=True)
        vocal_path, no_vocal_path = separate_audio(tmp_path, sep_dir)

        # 분리 실패 시 원본 사용
        chord_source = no_vocal_path if no_vocal_path else tmp_path
        vocal_source = vocal_path    if vocal_path    else tmp_path

        # ── 2. 반주 트랙으로 코드 분석 ────────────
        print("Basic Pitch 코드 분석 중...", flush=True)
        _, _, note_events = predict(
            chord_source, ICASSP_2022_MODEL_PATH,
            onset_threshold=0.5, frame_threshold=0.3,
            minimum_note_length=58,
            minimum_frequency=65.4, maximum_frequency=2093,
            midi_tempo=120,
        )
        print(f"음표 수: {len(note_events)}", flush=True)
        total_duration = max((float(n[1]) for n in note_events), default=0)

        # ── 3. 보컬 트랙으로 멜로디 감지 ──────────
        print("보컬 트랙 멜로디 분석 중...", flush=True)
        _, _, vocal_events = predict(
            vocal_source, ICASSP_2022_MODEL_PATH,
            onset_threshold=0.6, frame_threshold=0.4,
            minimum_note_length=80,
            minimum_frequency=130.0, maximum_frequency=1200.0,
            midi_tempo=120,
        )
        melody_notes = extract_melody(vocal_events)
        print(f"멜로디 음표: {len(melody_notes)}", flush=True)

        # ── 4. 조성 감지 ───────────────────────────
        note_count = [0] * 12
        for n in note_events:
            note_count[int(n[2]) % 12] += 1
        detected_key, mode = detect_key(note_count)
        key_display = detected_key + ("" if mode == "major" else "m")

        # ── 5. 코드 진행 ───────────────────────────
        chords_result = []
        prev_chord = None
        t = 0.0
        while t < total_duration:
            active = [int(n[2]) for n in note_events
                      if float(n[0]) <= t+0.5 and float(n[1]) >= t]
            chord = notes_to_chord(active)
            if chord and chord != prev_chord:
                chords_result.append({"chord": chord, "time": round(t, 2)})
                prev_chord = chord
            t += 0.5

        # ── 6. BPM 추정 ────────────────────────────
        starts = sorted([float(n[0]) for n in note_events])
        bpm = 0
        if len(starts) > 4:
            ivs = [starts[i+1]-starts[i] for i in range(min(20,len(starts)-1))
                   if 0.1 < starts[i+1]-starts[i] < 2.0]
            if ivs:
                raw_bpm = round(60/(sum(ivs)/len(ivs)))
                # 찬양곡은 보통 60~100BPM
                # 너무 빠르면 절반으로 보정
                if raw_bpm > 120:
                    bpm = round(raw_bpm / 2)
                else:
                    bpm = raw_bpm

        # ── 7. Whisper 가사 인식 (보컬 트랙만) ────
        print("Whisper 가사 인식 중...", flush=True)
        lyrics_segments = []
        full_lyrics     = ""
        detected_lang   = "unknown"
        try:
            segments_gen, info = whisper_model.transcribe(
                vocal_source,
                language=None,
                task="transcribe",
                beam_size=5,
            )
            detected_lang = info.language
            for seg in list(segments_gen):
                text = seg.text.strip()
                if text:
                    lyrics_segments.append({
                        "text" : text,
                        "start": round(seg.start, 2),
                        "end"  : round(seg.end,   2),
                    })
            full_lyrics = "\n".join(s["text"] for s in lyrics_segments)
            print(f"가사 완료! 언어:{detected_lang} {len(lyrics_segments)}개", flush=True)
        except Exception as e:
            print(f"Whisper 오류: {e}", flush=True)

        # ── 8. 구간 감지 ────────────────────────────
        print("구간 감지 중...", flush=True)
        sections = detect_sections(note_events, lyrics_segments, total_duration)
        print(f"구간: {[s['display'] for s in sections]}", flush=True)

        # ── 9. 가사 + 코드 매핑 ────────────────────
        lyrics_with_chords = []
        for seg in lyrics_segments:
            seg_chords = []
            for c in chords_result:
                if seg["start"] <= c["time"] <= seg["end"]:
                    if not seg_chords or seg_chords[-1] != c["chord"]:
                        seg_chords.append(c["chord"])
            lyrics_with_chords.append({
                "text"  : seg["text"].strip(),
                "chords": seg_chords,
                "start" : seg["start"],
                "end"   : seg["end"],
            })

        # ── 10. 정량화 음표 ────────────────────────
        effective_bpm   = bpm if bpm > 0 else 80  # BPM 감지 실패시 기본 80
        quantized_notes = quantize_notes(vocal_events, effective_bpm)
        chord_beats     = chords_at_beats(chords_result, effective_bpm)

        used_demucs = vocal_path is not None
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
