import os
import tempfile
import numpy as np
from flask import Flask, request, jsonify

os.environ["BASIC_PITCH_MODEL"] = "onnx"

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import whisper

app     = Flask(__name__)
ALLOWED = {'mp3', 'wav', 'm4a', 'aac', 'ogg'}
NOTES   = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

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
    """신뢰도 높은 최고음만 추출 (멜로디)"""
    melody, prev = [], None
    max_time = max((float(n[1]) for n in note_events), default=0)
    t = 0.0
    while t < max_time:
        active = [n for n in note_events
                  if float(n[0]) <= t+0.25 and float(n[1]) >= t
                  and (float(n[3]) if len(n) > 3 else 1.0) > 0.5]
        if active:
            top = max(active, key=lambda n: int(n[2]))
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

def detect_sections(note_events, lyrics_segments, total_duration):
    """
    구간 자동 감지:
    - 에너지(음표 밀도) 분석
    - 보컬 여부 (Whisper 결과)
    - 음역대 분석 (높은음=보컬, 낮은음=악기)
    """
    window   = 2.0   # 2초 단위로 분석
    sections = []

    # 2초 단위 에너지 계산
    n_windows = int(total_duration / window) + 1
    energy    = [0.0] * n_windows
    vocal_prob= [0.0] * n_windows  # 보컬 가능성 (높은음 비율)

    for note in note_events:
        start = float(note[0])
        midi  = int(note[2])
        conf  = float(note[3]) if len(note) > 3 else 1.0
        idx   = int(start / window)
        if idx < n_windows:
            energy[idx]    += conf
            # 보컬 음역대: C4(60) ~ C6(84)
            if 60 <= midi <= 84:
                vocal_prob[idx] += conf

    # 보컬 가능성 정규화
    for i in range(n_windows):
        if energy[i] > 0:
            vocal_prob[i] /= energy[i]

    # Whisper 가사 구간에서 보컬 여부 확인
    vocal_windows = set()
    for seg in lyrics_segments:
        if len(seg.get("text","").strip()) > 2:  # 실제 가사가 있으면
            start_idx = int(seg["start"] / window)
            end_idx   = int(seg["end"]   / window) + 1
            for i in range(start_idx, min(end_idx, n_windows)):
                vocal_windows.add(i)

    # 에너지 임계값
    avg_energy = sum(energy) / max(1, len([e for e in energy if e > 0]))
    low_thresh = avg_energy * 0.3
    high_thresh= avg_energy * 1.2

    # 구간 레이블 결정
    labels = []
    for i in range(n_windows):
        t     = i * window
        e     = energy[i]
        vocal = i in vocal_windows or vocal_prob[i] > 0.4

        if e < low_thresh * 0.1:
            labels.append("silence")
        elif vocal:
            labels.append("vocal")
        elif e < low_thresh:
            labels.append("silence")
        elif e > high_thresh:
            labels.append("climax")
        else:
            labels.append("instrumental")

    # 연속 구간 병합 + 섹션 이름 부여
    raw_sections = []
    if labels:
        cur_label = labels[0]
        cur_start = 0
        for i in range(1, len(labels)):
            if labels[i] != cur_label:
                raw_sections.append((cur_start * window, i * window, cur_label))
                cur_label = labels[i]
                cur_start = i
        raw_sections.append((cur_start * window, total_duration, cur_label))

    # silence 제거 및 짧은 구간 병합 (4초 미만)
    filtered = []
    for s, e, label in raw_sections:
        if label == "silence": continue
        if e - s < 4.0: continue
        filtered.append((s, e, label))

    # 섹션 이름 자동 부여
    vocal_count = 0
    instr_count = 0
    result      = []
    prev_label  = None

    for i, (s, e, label) in enumerate(filtered):
        duration = e - s
        is_first = (i == 0)
        is_last  = (i == len(filtered) - 1)

        if label == "vocal":
            vocal_count += 1
            if vocal_count == 1:
                name = "verse1"     # 1절
                display = "1절"
            elif vocal_count == 2:
                name = "chorus"     # 후렴
                display = "후렴"
            elif vocal_count == 3:
                name = "verse2"     # 2절
                display = "2절"
            elif vocal_count == 4:
                name = "chorus2"    # 후렴 반복
                display = "후렴 (반복)"
            else:
                name = f"section{vocal_count}"
                display = f"섹션 {vocal_count}"

        elif label == "climax":
            name    = "climax"
            display = "클라이맥스"

        else:  # instrumental
            instr_count += 1
            if is_first:
                name    = "intro"
                display = "인트로"
            elif is_last:
                name    = "outro"
                display = "아웃트로"
            elif prev_label in ["vocal", "climax"]:
                name    = "interlude"
                display = "간주"
            else:
                name    = f"instrumental{instr_count}"
                display = "악기 솔로"

        # 이 구간의 코드 찾기
        section_notes = [
            int(n[2]) for n in note_events
            if s <= float(n[0]) <= e
        ]
        section_chords = []
        t2 = s
        prev_chord = None
        while t2 < e:
            active = [int(n[2]) for n in note_events
                      if float(n[0]) <= t2+0.5 and float(n[1]) >= t2
                      and s <= float(n[0]) <= e]
            chord = notes_to_chord(active)
            if chord and chord != prev_chord:
                section_chords.append(chord)
                prev_chord = chord
            t2 += 0.5

        # 중복 제거 및 핵심 코드만
        unique_chords = list(dict.fromkeys(section_chords))[:8]

        result.append({
            "name"    : name,
            "display" : display,
            "start"   : round(s, 1),
            "end"     : round(e, 1),
            "duration": round(e - s, 1),
            "label"   : label,
            "chords"  : unique_chords,
            "hasVocal": label in ["vocal", "climax"],
        })
        prev_label = label

    return result

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
        # ── 1. Basic Pitch ─────────────────────────
        print("Basic Pitch 분석 중...", flush=True)
        _, _, note_events = predict(
            tmp_path, ICASSP_2022_MODEL_PATH,
            onset_threshold=0.5, frame_threshold=0.3,
            minimum_note_length=58,
            minimum_frequency=65.4, maximum_frequency=2093,
            midi_tempo=120,
        )
        print(f"음표 수: {len(note_events)}", flush=True)

        total_duration = max((float(n[1]) for n in note_events), default=0)

        # ── 2. 멜로디 추출 ─────────────────────────
        melody_notes = extract_melody(note_events)

        # ── 3. 조성 + 코드 ─────────────────────────
        note_count = [0] * 12
        for n in note_events:
            note_count[int(n[2]) % 12] += 1

        detected_key, mode = detect_key(note_count)
        key_display = detected_key + ("" if mode == "major" else "m")

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

        # ── 4. BPM ─────────────────────────────────
        starts = sorted([float(n[0]) for n in note_events])
        bpm = 0
        if len(starts) > 4:
            ivs = [starts[i+1]-starts[i] for i in range(min(20,len(starts)-1))
                   if 0.1 < starts[i+1]-starts[i] < 2.0]
            if ivs: bpm = round(60/(sum(ivs)/len(ivs)))

        # ── 5. Whisper 가사 ─────────────────────────
        print("Whisper 가사 인식 중...", flush=True)
        lyrics_segments = []
        full_lyrics     = ""
        detected_lang   = "unknown"
        try:
            w_result = whisper_model.transcribe(
                tmp_path, language=None,
                task="transcribe", fp16=False,
            )
            detected_lang = w_result.get("language", "unknown")
            for seg in w_result.get("segments", []):
                text = seg["text"].strip()
                if text:
                    lyrics_segments.append({
                        "text" : text,
                        "start": round(seg["start"], 2),
                        "end"  : round(seg["end"],   2),
                    })
            full_lyrics = "\n".join(s["text"] for s in lyrics_segments)
            print(f"가사 완료! 언어:{detected_lang} 세그먼트:{len(lyrics_segments)}", flush=True)
        except Exception as e:
            print(f"Whisper 오류: {e}", flush=True)

        # ── 6. 구간 감지 ────────────────────────────
        print("구간 감지 중...", flush=True)
        sections = detect_sections(note_events, lyrics_segments, total_duration)
        print(f"구간: {[s['display'] for s in sections]}", flush=True)

        # ── 7. 가사 + 코드 매핑 ────────────────────
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

        # 박자 정량화
        quantized_notes = quantize_notes(note_events, bpm) if bpm > 0 else []
        chord_beats     = chords_at_beats(chords_result, bpm) if bpm > 0 else []

        print(f"완료! Key:{key_display} BPM:{bpm} 구간:{len(sections)}개", flush=True)

        return jsonify({
            "success"        : True,
            "key"            : key_display,
            "mode"           : mode,
            "bpm"            : bpm,
            "language"       : detected_lang,
            "duration"       : round(total_duration, 1),
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
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

def quantize_notes(note_events, bpm):
    """음표를 박자에 맞게 정량화 (타임스탬프 → 음표 길이)"""
    if not note_events or bpm <= 0:
        return []

    beat_duration = 60.0 / bpm  # 1박자 = 몇 초
    quantized     = []
    prev_name     = None

    # 멜로디만 추출 (최고음, 신뢰도 0.5 이상)
    sorted_events = sorted(note_events, key=lambda n: float(n[0]))
    max_time      = max(float(n[1]) for n in note_events)

    t = 0.0
    while t < max_time:
        window = beat_duration * 0.5
        active = [n for n in note_events
                  if float(n[0]) <= t + window and float(n[1]) >= t
                  and (float(n[3]) if len(n) > 3 else 1.0) > 0.4]

        if active:
            top   = max(active, key=lambda n: int(n[2]))
            midi  = int(top[2])
            name  = NOTES[midi % 12]
            oct_  = (midi // 12) - 1
            dur   = float(top[1]) - float(top[0])
            beats = max(0.5, min(4.0, round(dur / beat_duration * 2) / 2))

            if name != prev_name:
                quantized.append({
                    "pitch"   : name,
                    "octave"  : oct_,
                    "beats"   : beats,
                    "duration": beats_to_dur(beats),
                    "isRest"  : False,
                    "midi"    : midi,
                })
                prev_name = name
        else:
            quantized.append({
                "pitch": "r", "octave": 4, "beats": 1.0,
                "duration": "q", "isRest": True, "midi": 0,
            })

        t += beat_duration

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
    """코드를 박자 위치로 변환"""
    beat_dur = 60.0 / bpm if bpm > 0 else 0.5
    result   = []
    for c in chords_result:
        beat = c["time"] / beat_dur
        result.append({"chord": c["chord"], "beat": round(beat, 1)})
    return result
