import os
import json
import tempfile
import numpy as np
from flask import Flask, request, jsonify

os.environ["BASIC_PITCH_MODEL"] = "onnx"

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

app = Flask(__name__)

ALLOWED = {'mp3', 'wav', 'm4a', 'aac', 'ogg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def midi_to_note(midi_num):
    octave   = (midi_num // 12) - 1
    note_idx = midi_num % 12
    return NOTE_NAMES[note_idx], octave

def detect_key(note_counts):
    major_profile = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
    minor_profile = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]
    best_key  = "C"
    best_mode = "major"
    best_corr = -999
    for i in range(12):
        rotated_major = major_profile[i:] + major_profile[:i]
        corr_major    = float(np.corrcoef(note_counts, rotated_major)[0,1])
        if corr_major > best_corr:
            best_corr = corr_major
            best_key  = NOTE_NAMES[i]
            best_mode = "major"
        rotated_minor = minor_profile[i:] + minor_profile[:i]
        corr_minor    = float(np.corrcoef(note_counts, rotated_minor)[0,1])
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key  = NOTE_NAMES[i]
            best_mode = "minor"
    return best_key, best_mode

def notes_to_chords(active_notes):
    if not active_notes:
        return None
    indices   = sorted(set(n % 12 for n in active_notes))
    root_idx  = indices[0]
    root      = NOTE_NAMES[root_idx]
    intervals = [(i - root_idx) % 12 for i in indices[1:]]
    if 4 in intervals and 7 in intervals and 11 in intervals: return root + "M7"
    if 3 in intervals and 7 in intervals and 10 in intervals: return root + "m7"
    if 4 in intervals and 7 in intervals and 10 in intervals: return root + "7"
    if 4 in intervals and 7 in intervals:                     return root
    if 3 in intervals and 7 in intervals:                     return root + "m"
    if 4 in intervals and 6 in intervals:                     return root + "dim"
    if 4 in intervals and 8 in intervals:                     return root + "aug"
    if 5 in intervals and 7 in intervals:                     return root + "sus4"
    if 2 in intervals and 7 in intervals:                     return root + "sus2"
    return root

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "WorshipSheet 서버 정상 작동 중"})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "파일이 없어요"}), 400
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "MP3, WAV, M4A, AAC 파일만 가능해요"}), 400
    suffix = '.' + file.filename.rsplit('.', 1)[1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        import sys
print("분석 시작...", flush=True)
model_output, midi_data, note_events = predict(
    tmp_path,
    ICASSP_2022_MODEL_PATH,
    onset_threshold    = 0.5,
    frame_threshold    = 0.3,
    minimum_note_length= 58,
    minimum_frequency  = 65.4,
    maximum_frequency  = 2093,
    midi_tempo         = 120,
)
print(f"분석 완료! 음표 수: {len(note_events)}", flush=True)
        notes_result = []
        note_count   = [0] * 12
        for note in note_events:
            start_time = float(note[0])
            end_time   = float(note[1])
            midi_pitch = int(note[2])
            confidence = float(note[3]) if len(note) > 3 else 1.0
            note_name, octave = midi_to_note(midi_pitch)
            note_count[midi_pitch % 12] += 1
            notes_result.append({
                "name": note_name, "octave": octave, "midi": midi_pitch,
                "start": round(start_time, 3), "end": round(end_time, 3),
                "duration": round(end_time - start_time, 3),
                "confidence": round(confidence, 3),
                "isSharp": "#" in note_name,
                "displayName": note_name + str(octave)
            })
        detected_key, mode = detect_key(note_count)
        key_display = detected_key + ("" if mode == "major" else "m")
        chords_result = []
        window_size   = 0.5
        max_time      = max((n["end"] for n in notes_result), default=0)
        prev_chord    = None
        t = 0.0
        while t < max_time:
            active = [n["midi"] for n in notes_result if n["start"] <= t + window_size and n["end"] >= t]
            chord  = notes_to_chords(active)
            if chord and chord != prev_chord:
                chords_result.append({"chord": chord, "time": round(t, 2)})
                prev_chord = chord
            t += window_size
        starts = sorted([n["start"] for n in notes_result])
        bpm    = 0
        if len(starts) > 4:
            intervals = [starts[i+1] - starts[i] for i in range(min(20, len(starts)-1)) if 0.1 < starts[i+1] - starts[i] < 2.0]
            if intervals:
                bpm = round(60 / (sum(intervals) / len(intervals)))
        return jsonify({
            "success": True, "key": key_display, "mode": mode, "bpm": bpm,
            "noteCount": len(notes_result), "chordCount": len(chords_result),
            "notes": notes_result[:200],
            "chords": [c["chord"] for c in chords_result],
            "chordsWithTime": chords_result,
        })
except Exception as e:
    import traceback
    print("오류 발생:", traceback.format_exc(), flush=True)
    return jsonify({"error": "분석 실패: " + str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
