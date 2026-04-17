import os, tempfile, shutil, subprocess
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
# 조성 감지 (전체 오디오 chroma 기반)
# ══════════════════════════════════════════════════
def detect_key(audio_path):
    import librosa
    y, sr    = librosa.load(audio_path, sr=22050, mono=True)
    y_harm,_ = librosa.effects.hpss(y)
    profile  = librosa.feature.chroma_cqt(
        y=y_harm, sr=sr, hop_length=2048
    ).mean(axis=1)

    major_tmpl = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
    minor_tmpl = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]
    best_key, best_mode, best_corr = "C", "major", -999

    for i in range(12):
        for tmpl, mode in [(major_tmpl,"major"),(minor_tmpl,"minor")]:
            corr = float(np.corrcoef(profile, np.roll(tmpl, i))[0,1])
            if corr > best_corr:
                best_corr, best_key, best_mode = corr, NOTES[i], mode

    return best_key, best_mode

# ══════════════════════════════════════════════════
# 키 제한 코드 감지 (핵심 정확도 향상)
# ══════════════════════════════════════════════════
def get_scale_roots(key_note, mode):
    """해당 키의 스케일 음 인덱스 반환"""
    major_steps = [0,2,4,5,7,9,11]
    minor_steps = [0,2,3,5,7,8,10]
    steps = major_steps if mode == "major" else minor_steps
    root  = NOTES.index(key_note)
    return [(root + s) % 12 for s in steps]

def scale_chord_quality(degree_idx, mode):
    """스케일 음급별 기본 화음 성질"""
    major_q = ["","m","m","","","m","dim"]
    minor_q = ["m","dim","","m","m","",""]
    return (major_q if mode == "major" else minor_q)[degree_idx]

def detect_chords(audio_path, key_note, mode, sr=22050):
    """
    키 스케일 안의 코드만 허용 → 정확도 대폭 향상
    F장조면 F,Gm,Am,Bb,C,Dm만 감지
    """
    import librosa

    y, sr    = librosa.load(audio_path, sr=sr, mono=True)
    y_harm,_ = librosa.effects.hpss(y)
    chroma   = librosa.feature.chroma_cqt(
        y=y_harm, sr=sr, hop_length=2048, bins_per_octave=36
    )

    duration    = librosa.get_duration(y=y, sr=sr)
    frames      = chroma.shape[1]
    secs_fr     = duration / frames
    win         = max(1, int(2.0 / secs_fr))

    scale_roots = get_scale_roots(key_note, mode)
    key_idx     = NOTES.index(key_note)

    # 코드 템플릿 (예배에서 자주 쓰는 것만)
    templates = {
        ""    : [1,0,0,0,1,0,0,1,0,0,0,0],
        "m"   : [1,0,0,1,0,0,0,1,0,0,0,0],
        "7"   : [1,0,0,0,1,0,0,1,0,0,1,0],
        "m7"  : [1,0,0,1,0,0,0,1,0,0,1,0],
        "sus4": [1,0,0,0,0,1,0,1,0,0,0,0],
    }

    result, prev = [], None
    for i in range(0, frames, win):
        chunk = chroma[:, i:i+win].mean(axis=1)
        best_score, best_chord = -1, NOTES[key_idx]

        for degree, root_idx in enumerate(scale_roots):
            rotated  = np.roll(chunk, -root_idx)
            expected = scale_chord_quality(degree, mode)
            for suffix, tmpl in templates.items():
                score = float(np.dot(rotated, tmpl))
                if suffix == expected:
                    score *= 1.2  # 기대 화음에 보너스
                if score > best_score:
                    best_score = score
                    best_chord = NOTES[root_idx] + suffix

        t = round(i * secs_fr, 2)
        if best_chord != prev:
            result.append({"chord": best_chord, "time": t})
            prev = best_chord

    return result, round(duration, 1)

# ══════════════════════════════════════════════════
# pyin 멜로디 + voiced 타임라인
# ══════════════════════════════════════════════════
def detect_melody(audio_path, sr=22050):
    import librosa
    y, sr   = librosa.load(audio_path, sr=sr, mono=True)
    hop     = 512
    f0, voiced, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C3'),
        fmax=librosa.note_to_hz('C6'), sr=sr, hop_length=hop
    )
    times   = librosa.times_like(f0, sr=sr, hop_length=hop)
    notes, prev = [], None

    for i, (freq, v) in enumerate(zip(f0, voiced)):
        if not v or freq is None or np.isnan(freq): continue
        midi = int(round(librosa.hz_to_midi(freq)))
        name = NOTES[midi % 12]
        if name != prev:
            notes.append({
                "name": name, "octave": (midi//12)-1, "midi": midi,
                "start": round(float(times[i]), 3),
                "end":   round(float(times[i])+0.25, 3),
                "isSharp": "#" in name,
                "displayName": name + str((midi//12)-1),
            })
            prev = name

    voiced_timeline = [(float(times[i]), bool(voiced[i])) for i in range(len(times))]
    return notes[:64], voiced_timeline

# ══════════════════════════════════════════════════
# BPM 감지
# ══════════════════════════════════════════════════
def detect_bpm(audio_path, sr=22050):
    import librosa
    y, sr  = librosa.load(audio_path, sr=sr, mono=True, duration=90)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(float(tempo))
    # 찬양 BPM 보정 (55~110)
    while bpm > 120: bpm = round(bpm / 2)
    while bpm < 55:  bpm = round(bpm * 2)
    return bpm

# ══════════════════════════════════════════════════
# 멘트 vs 가사 분류
# ══════════════════════════════════════════════════
def classify_segments(raw_segs, voiced_timeline):
    result = []
    for seg in raw_segs:
        s, e = seg["start"], seg["end"]
        frames = [(t, v) for t, v in voiced_timeline if s <= t <= e]
        ratio  = sum(1 for _, v in frames if v) / max(1, len(frames))
        seg_type = "lyrics" if ratio > 0.5 else ("spoken" if ratio > 0.1 else "silence")
        result.append({
            "text": seg["text"].strip(),
            "start": s, "end": e,
            "type": seg_type, "voicedRatio": round(ratio, 2),
        })
    return result

# ══════════════════════════════════════════════════
# 구간 감지
# ══════════════════════════════════════════════════
def detect_sections(chords, classified, total):
    vocal_times = set()
    for seg in classified:
        if seg["type"] == "lyrics":
            t = seg["start"]
            while t < seg["end"]:
                vocal_times.add(int(t)); t += 1.0

    window = 16.0
    sections, prev_label = [], None
    vocal_count = instr_count = 0
    t = 0.0

    while t < total:
        end      = min(t + window, total)
        is_vocal = any(int(x) in vocal_times for x in np.arange(t, end, 1.0))
        seg_chords = list(dict.fromkeys(
            c["chord"] for c in chords if t <= c["time"] < end
        ))[:6]

        if is_vocal:
            vocal_count += 1
            label_map = {1:"verse1",2:"chorus",3:"verse2",4:"chorus2"}
            name    = label_map.get(vocal_count, f"section{vocal_count}")
            display_map = {1:"1절",2:"후렴",3:"2절",4:"후렴 반복"}
            display = display_map.get(vocal_count, f"섹션{vocal_count}")
        else:
            instr_count += 1
            if t < 8:                        name, display = "intro",     "인트로"
            elif end >= total - 8:           name, display = "outro",     "아웃트로"
            elif prev_label == "vocal":      name, display = "interlude", "간주"
            else:                            name, display = "bridge",    "브릿지"

        lbl = "vocal" if is_vocal else "instrumental"
        if sections and sections[-1]["name"] == name:
            sections[-1]["end"]      = round(end, 1)
            sections[-1]["duration"] = round(end - sections[-1]["start"], 1)
            sections[-1]["chords"]   = list(dict.fromkeys(sections[-1]["chords"] + seg_chords))[:8]
        else:
            sections.append({
                "name": name, "display": display,
                "start": round(t,1), "end": round(end,1),
                "duration": round(end-t,1), "label": lbl,
                "chords": seg_chords, "hasVocal": is_vocal,
            })
            prev_label = lbl
        t += window

    return sections

# ══════════════════════════════════════════════════
# Demucs 음원 분리
# ══════════════════════════════════════════════════
def convert_to_wav(input_path, out_dir):
    import librosa, soundfile as sf
    wav  = os.path.join(out_dir, "input.wav")
    y, sr = librosa.load(input_path, sr=44100, mono=False)
    if y.ndim == 1: y = np.stack([y, y])
    sf.write(wav, y.T, sr)
    return wav

def separate_audio(input_path, out_dir):
    try:
        print("Demucs 시작...", flush=True)
        wav = convert_to_wav(input_path, out_dir)
        res = subprocess.run(
            ["python3", "-m", "demucs", "--two-stems", "vocals",
             "--out", out_dir, wav],
            capture_output=True, text=True, timeout=300
        )
        print(f"Demucs returncode: {res.returncode}", flush=True)
        if res.stderr: print(f"Demucs: {res.stderr[:200]}", flush=True)
        if res.returncode != 0: return None, None

        out = Path(out_dir)
        voc = list(out.rglob("vocals.wav"))
        nov = list(out.rglob("no_vocals.wav"))
        return (str(voc[0]) if voc else None), (str(nov[0]) if nov else None)
    except Exception as e:
        print(f"Demucs 오류: {e}", flush=True)
        return None, None

# ══════════════════════════════════════════════════
# 핵심 분석 파이프라인
# ══════════════════════════════════════════════════
def process_audio(audio_path, tmp_dir, hint_title="", hint_artist=""):
    try:
        # Demucs 분리
        sep_dir = os.path.join(tmp_dir, "sep")
        os.makedirs(sep_dir, exist_ok=True)
        vocal_path, instr_path = separate_audio(audio_path, sep_dir)
        used_demucs  = vocal_path is not None
        chord_src    = instr_path if instr_path else audio_path
        vocal_src    = vocal_path if vocal_path  else audio_path

        # 1. 조성 (가장 먼저!)
        print("조성 감지...", flush=True)
        key_note, mode = detect_key(chord_src)
        key_display    = key_note + ("" if mode == "major" else "m")
        print(f"조성: {key_display}", flush=True)

        # 2. 코드 (키 제한으로 정확도 향상)
        print("코드 감지 (키 제한)...", flush=True)
        chords, total_dur = detect_chords(chord_src, key_note, mode)
        print(f"코드 {len(chords)}개", flush=True)

        # 3. 멜로디
        print("멜로디 감지...", flush=True)
        melody, voiced_tl = detect_melody(vocal_src)
        print(f"멜로디 {len(melody)}개", flush=True)

        # 4. BPM
        bpm = detect_bpm(audio_path)
        print(f"BPM: {bpm}", flush=True)

        # 5. 가사
        print("가사 인식 (Whisper medium)...", flush=True)
        raw_segs, lang = [], "ko"
        try:
            segs_gen, info = whisper_model.transcribe(
                vocal_src, language="ko", task="transcribe",
                beam_size=5,
                initial_prompt="찬양 예배 하나님 주님 예수님 그리스도 성령 할렐루야",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=800, speech_pad_ms=300),
                temperature=0, no_speech_threshold=0.35,
            )
            lang = info.language
            for s in list(segs_gen):
                if s.text.strip() and len(s.text.strip()) > 1:
                    raw_segs.append({"text":s.text.strip(),"start":round(s.start,2),"end":round(s.end,2)})
            print(f"가사 {len(raw_segs)}개", flush=True)
        except Exception as e:
            print(f"Whisper 오류: {e}", flush=True)

        # 6. 멘트/가사 분류
        classified   = classify_segments(raw_segs, voiced_tl)
        lyrics_segs  = [s for s in classified if s["type"] == "lyrics"]
        spoken_segs  = [s for s in classified if s["type"] == "spoken"]
        full_lyrics  = "\n".join(s["text"] for s in lyrics_segs)
        print(f"가사:{len(lyrics_segs)}개 멘트:{len(spoken_segs)}개", flush=True)

        # 7. 구간
        sections = detect_sections(chords, classified, total_dur)
        print(f"구간: {[s['display'] for s in sections]}", flush=True)

        # 8. 가사+코드 매핑
        lyrics_with_chords = []
        for seg in lyrics_segs:
            seg_chords = list(dict.fromkeys(
                c["chord"] for c in chords if seg["start"] <= c["time"] <= seg["end"]
            ))
            lyrics_with_chords.append({
                "text":seg["text"], "chords":seg_chords,
                "start":seg["start"], "end":seg["end"],
            })

        spoken_with_chords = []
        for seg in spoken_segs:
            seg_chords = list(dict.fromkeys(
                c["chord"] for c in chords if seg["start"] <= c["time"] <= seg["end"]
            ))
            spoken_with_chords.append({
                "text":seg["text"], "chords":seg_chords,
                "start":seg["start"], "end":seg["end"],
            })

        print(f"완료! Key:{key_display} BPM:{bpm} Demucs:{used_demucs}", flush=True)

        return jsonify({
            "success"        : True,
            "key"            : key_display,
            "mode"           : mode,
            "bpm"            : bpm,
            "language"       : lang,
            "duration"       : total_dur,
            "usedDemucs"     : used_demucs,
            "chords"         : [c["chord"] for c in chords],
            "chordsWithTime" : chords,
            "lyrics"         : full_lyrics,
            "lyricsSegments" : lyrics_with_chords,
            "spokenSegments" : spoken_with_chords,
            "sections"       : sections,
            "notes"          : melody,
            "songTitle"      : hint_title,
            "songArtist"     : hint_artist,
        })

    except Exception as e:
        import traceback
        print("오류:", traceback.format_exc(), flush=True)
        return jsonify({"error": str(e)}), 500

# ══════════════════════════════════════════════════
# API 엔드포인트
# ══════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/transcribe_url', methods=['POST'])
def transcribe_url():
    """YouTube URL 채보 (메인 엔드포인트)"""
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "URL이 없어요"}), 400

    url = data['url'].strip()
    if 'youtube.com' not in url and 'youtu.be' not in url:
        return jsonify({"error": "유튜브 URL만 지원해요"}), 400

    tmp_dir = tempfile.mkdtemp()
    try:
        import yt_dlp
        print(f"유튜브 다운로드: {url}", flush=True)

        ydl_opts = {
            'format'        : 'bestaudio/best',
            'outtmpl'       : os.path.join(tmp_dir, 'audio.%(ext)s'),
            'postprocessors': [{'key':'FFmpegExtractAudio',
                                'preferredcodec':'mp3','preferredquality':'192'}],
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title  = info.get('title', '')
            artist = info.get('uploader', '')

        mp3s = [f for f in os.listdir(tmp_dir) if f.endswith('.mp3')]
        if not mp3s:
            return jsonify({"error": "다운로드 실패"}), 500

        audio_path = os.path.join(tmp_dir, mp3s[0])
        print(f"다운로드 완료: {title}", flush=True)
        return process_audio(audio_path, tmp_dir, title, artist)

    except Exception as e:
        import traceback
        print("YouTube 오류:", traceback.format_exc(), flush=True)
        return jsonify({"error": str(e)}), 500

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """파일 업로드 채보 (보조)"""
    if 'file' not in request.files:
        return jsonify({"error": "파일 없음"}), 400
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({"error": "지원하지 않는 형식"}), 400

    tmp_dir  = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "input." + file.filename.rsplit('.',1)[1].lower())
    file.save(tmp_path)
    try:
        return process_audio(tmp_path, tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

# ══════════════════════════════════════════════════
# 악보 이미지 분석 (Claude Vision API)
# ══════════════════════════════════════════════════
@app.route('/analyze_sheet', methods=['POST'])
def analyze_sheet():
    """
    악보 이미지(JPG/PNG/PDF)를 Claude Vision으로 분석해서
    코드/가사/키/박자를 추출
    """
    if 'file' not in request.files:
        return jsonify({"error": "파일이 없어요"}), 400

    file     = request.files['file']
    filename = file.filename.lower()

    try:
        import anthropic, base64

        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            return jsonify({"error": "ANTHROPIC_API_KEY가 설정되지 않았어요"}), 500

        client = anthropic.Anthropic(api_key=api_key)

        # PDF → 첫 페이지만 이미지로 변환
        if filename.endswith('.pdf'):
            import io
            pdf_bytes = file.read()
            # PDF를 이미지로 변환 (pypdf + Pillow)
            try:
                from pypdf import PdfReader
                from PIL import Image
                import io as _io

                reader = PdfReader(_io.BytesIO(pdf_bytes))
                # PDF를 PNG로 렌더링 (간단한 방법)
                # pypdf는 렌더링 미지원 → base64로 직접 전송
                img_data    = base64.b64encode(pdf_bytes).decode('utf-8')
                media_type  = "application/pdf"
            except:
                img_data   = base64.b64encode(pdf_bytes).decode('utf-8')
                media_type = "application/pdf"
        else:
            img_bytes  = file.read()
            img_data   = base64.b64encode(img_bytes).decode('utf-8')
            if filename.endswith('.png'):
                media_type = "image/png"
            elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
                media_type = "image/jpeg"
            else:
                media_type = "image/jpeg"

        print("Claude Vision으로 악보 분석 중...", flush=True)

        # Claude에게 악보 분석 요청
        prompt = """이 악보 이미지를 분석해서 아래 JSON 형식으로만 응답해주세요.
다른 설명 없이 JSON만 출력하세요.

{
  "key": "F",
  "timeSignature": "4/4",
  "bpm": 76,
  "chords": ["F", "Bb", "C", "Gm", "Dm"],
  "chordsInOrder": ["F", "C", "Bb", "F", "Gm", "C", "F"],
  "lyrics": "줄별 가사 전체",
  "lyricsLines": ["1번 줄 가사", "2번 줄 가사"],
  "sections": [
    {"name": "인트로", "chords": ["F", "C", "Bb"]},
    {"name": "1절", "chords": ["F", "Bb", "C", "Gm"]},
    {"name": "후렴", "chords": ["Bb", "F", "C", "Dm"]}
  ],
  "title": "곡 제목 (있으면)",
  "artist": "아티스트 (있으면)"
}

악보에서 읽을 수 없는 항목은 null로 처리하세요.
코드는 정확히 악보에 표시된 그대로 추출하세요 (예: F/A, Bb, Gm7 등)."""

        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }]
        )

        raw = response.content[0].text.strip()
        print(f"Claude 응답: {raw[:200]}", flush=True)

        # JSON 파싱
        import json, re
        # 마크다운 코드블록 제거
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw)
        result = json.loads(raw)

        print(f"악보 분석 완료! Key:{result.get('key')} 코드:{len(result.get('chords',[]))}개", flush=True)

        return jsonify({
            "success"       : True,
            "key"           : result.get("key"),
            "timeSignature" : result.get("timeSignature"),
            "bpm"           : result.get("bpm"),
            "chords"        : result.get("chords", []),
            "chordsInOrder" : result.get("chordsInOrder", []),
            "lyrics"        : result.get("lyrics"),
            "lyricsLines"   : result.get("lyricsLines", []),
            "sections"      : result.get("sections", []),
            "title"         : result.get("title"),
            "artist"        : result.get("artist"),
        })

    except Exception as e:
        import traceback
        print("악보 분석 오류:", traceback.format_exc(), flush=True)
        return jsonify({"error": str(e)}), 500
