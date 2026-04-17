import os
import tempfile
import numpy as np
from flask import Flask, request, jsonify

os.environ["BASIC_PITCH_MODEL"] = "onnx"

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import whisper

app      = Flask(__name__)
ALLOWED  = {'mp3', 'wav', 'm4a', 'aac', 'ogg'}
NOTES    = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# Whisper 모델 미리 로드 (small = 한국어 지원, 메모리 적게 사용)
print("Whisper 모델 로딩 중...", flush=True)
whisper_model = whisper.load_model("small")
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
    idx   = sorted(set(n % 12 for n in active))
    root  = NOTES[idx[0]]
    ivs   = [(i - idx[0]) % 12 for i in idx[1:]]
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

def extract_melody(note_events):
    """각 시간 구간에서 가장 높고 자신감 있는 음만 추출 (멜로디)"""
    melody = []
    window = 0.25  # 250ms 단위
    if not note_events: return melody

    max_time = max(float(n[1]) for n in note_events)
    t = 0.0
    prev_note = None

    while t < max_time:
        # 이 구간에 울리는 음표들 중 신뢰도 높은 것만
        active = [
            n for n in note_events
            if float(n[0]) <= t + window and float(n[1]) >= t
            and (float(n[3]) if len(n) > 3 else 1.0) > 0.5
        ]
        if active:
            # 가장 높은 음 (멜로디는 보통 최고음)
            top = max(active, key=lambda n: int(n[2]))
            midi  = int(top[0+2])
            note_name, octave = midi_to_note(midi)
            if note_name != prev_note:
                melody.append({
                    "name"       : note_name,
                    "octave"     : octave,
                    "midi"       : midi,
                    "start"      : round(t, 3),
                    "end"        : round(float(top[1]), 3),
                    "isSharp"    : "#" in note_name,
                    "displayName": note_name + str(octave)
                })
                prev_note = note_name
        t += window

    return melody[:64]  # 최대 64음표

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

    suffix = '.' + file.filename.rsplit('.', 1)[1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # ── 1. Basic Pitch 채보 ────────────────────
        print("Basic Pitch 분석 시작...", flush=True)
        _, _, note_events = predict(
            tmp_path, ICASSP_2022_MODEL_PATH,
            onset_threshold=0.5, frame_threshold=0.3,
            minimum_note_length=58,
            minimum_frequency=65.4, maximum_frequency=2093,
            midi_tempo=120,
        )
        print(f"음표 수: {len(note_events)}", flush=True)

        # ── 2. 멜로디 추출 ─────────────────────────
        melody_notes = extract_melody(note_events)
        print(f"멜로디 음표: {len(melody_notes)}", flush=True)

        # ── 3. 코드 감지 ───────────────────────────
        note_count = [0] * 12
        for n in note_events:
            note_count[int(n[2]) % 12] += 1

        detected_key, mode = detect_key(note_count)
        key_display = detected_key + ("" if mode == "major" else "m")

        chords_result = []
        max_time = max((float(n[1]) for n in note_events), default=0)
        prev_chord = None
        t = 0.0
        while t < max_time:
            active = [int(n[2]) for n in note_events
                      if float(n[0]) <= t+0.5 and float(n[1]) >= t]
            chord = notes_to_chord(active)
            if chord and chord != prev_chord:
                chords_result.append({"chord": chord, "time": round(t, 2)})
                prev_chord = chord
            t += 0.5

        # ── 4. BPM 추정 ────────────────────────────
        starts = sorted([float(n[0]) for n in note_events])
        bpm = 0
        if len(starts) > 4:
            ivs = [starts[i+1]-starts[i] for i in range(min(20,len(starts)-1))
                   if 0.1 < starts[i+1]-starts[i] < 2.0]
            if ivs: bpm = round(60 / (sum(ivs)/len(ivs)))

        # ── 5. Whisper 가사 인식 ───────────────────
        print("Whisper 가사 인식 시작...", flush=True)
        lyrics_segments = []
        try:
            result = whisper_model.transcribe(
                tmp_path,
                language=None,       # 언어 자동 감지 (한국어/영어)
                task="transcribe",
                word_timestamps=True,
                fp16=False,
            )
            detected_lang = result.get("language", "unknown")
            print(f"감지된 언어: {detected_lang}", flush=True)

            for seg in result.get("segments", []):
                lyrics_segments.append({
                    "text" : seg["text"].strip(),
                    "start": round(seg["start"], 2),
                    "end"  : round(seg["end"], 2),
                })

            full_lyrics = "\n".join(s["text"] for s in lyrics_segments)
            print(f"가사 인식 완료! 세그먼트: {len(lyrics_segments)}개", flush=True)

        except Exception as e:
            print(f"Whisper 오류: {e}", flush=True)
            full_lyrics = ""
            detected_lang = "unknown"

        # ── 6. 코드 + 가사 매핑 ───────────────────
        # 각 가사 세그먼트에 해당하는 코드 찾기
        lyrics_with_chords = []
        for seg in lyrics_segments:
            # 이 구간의 코드 찾기
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

        print(f"완료! Key:{key_display} BPM:{bpm} 코드:{len(chords_result)}", flush=True)

        return jsonify({
            "success"         : True,
            "key"             : key_display,
            "mode"            : mode,
            "bpm"             : bpm,
            "language"        : detected_lang,
            "noteCount"       : len(melody_notes),
            "chordCount"      : len(chords_result),
            "notes"           : melody_notes,
            "chords"          : [c["chord"] for c in chords_result],
            "chordsWithTime"  : chords_result,
            "lyrics"          : full_lyrics,
            "lyricsSegments"  : lyrics_with_chords,
        })

    except Exception as e:
        import traceback
        print("오류:", traceback.format_exc(), flush=True)
        return jsonify({"error": "분석 실패: " + str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
