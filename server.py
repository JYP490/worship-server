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

# ── 유틸 함수들 ───────────────────────────────────
def allowed_file(f):
    return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED

def detect_key(audio_path):
    import librosa
    y, sr    = librosa.load(audio_path, sr=22050, mono=True)
    y_harm,_ = librosa.effects.hpss(y)
    profile  = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=2048).mean(axis=1)
    major_t  = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
    minor_t  = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]
    best_key, best_mode, best_corr = "C", "major", -999
    for i in range(12):
        for tmpl, mode in [(major_t,"major"),(minor_t,"minor")]:
            corr = float(np.corrcoef(profile, np.roll(tmpl, i))[0,1])
            if corr > best_corr:
                best_corr, best_key, best_mode = corr, NOTES[i], mode
    return best_key, best_mode

def get_scale_roots(key_note, mode):
    steps = [0,2,4,5,7,9,11] if mode=="major" else [0,2,3,5,7,8,10]
    root  = NOTES.index(key_note)
    return [(root+s)%12 for s in steps]

def scale_chord_quality(degree_idx, mode):
    major_q = ["","m","m","","","m","dim"]
    minor_q = ["m","dim","","m","m","",""]
    return (major_q if mode=="major" else minor_q)[degree_idx]

def detect_chords(audio_path, key_note, mode, sr=22050):
    import librosa
    y, sr    = librosa.load(audio_path, sr=sr, mono=True)
    y_harm,_ = librosa.effects.hpss(y)
    chroma   = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=2048, bins_per_octave=36)
    duration = librosa.get_duration(y=y, sr=sr)
    frames   = chroma.shape[1]
    secs_fr  = duration / frames
    win      = max(1, int(2.0/secs_fr))
    scale_roots = get_scale_roots(key_note, mode)
    key_idx  = NOTES.index(key_note)
    templates = {
        "":["Major",[1,0,0,0,1,0,0,1,0,0,0,0]],
        "m":["Minor",[1,0,0,1,0,0,0,1,0,0,0,0]],
        "7":["Dom7",[1,0,0,0,1,0,0,1,0,0,1,0]],
        "m7":["Min7",[1,0,0,1,0,0,0,1,0,0,1,0]],
        "sus4":["Sus4",[1,0,0,0,0,1,0,1,0,0,0,0]],
    }
    result, prev = [], None
    for i in range(0, frames, win):
        chunk = chroma[:, i:i+win].mean(axis=1)
        best_score, best_chord = -1, NOTES[key_idx]
        for degree, root_idx in enumerate(scale_roots):
            rotated  = np.roll(chunk, -root_idx)
            expected = scale_chord_quality(degree, mode)
            for suffix, (_, tmpl) in templates.items():
                score = float(np.dot(rotated, tmpl))
                if suffix == expected: score *= 1.2
                if score > best_score:
                    best_score = score
                    best_chord = NOTES[root_idx]+suffix
        t = round(i*secs_fr, 2)
        if best_chord != prev:
            result.append({"chord":best_chord,"time":t})
            prev = best_chord
    return result, round(duration, 1)

def detect_melody(audio_path, sr=22050):
    import librosa
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    hop   = 512
    f0, voiced, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C3'),
        fmax=librosa.note_to_hz('C6'), sr=sr, hop_length=hop
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    notes, prev = [], None
    for i, (freq, v) in enumerate(zip(f0, voiced)):
        if not v or freq is None or np.isnan(freq): continue
        midi = int(round(librosa.hz_to_midi(freq)))
        name = NOTES[midi%12]
        if name != prev:
            notes.append({
                "name":name,"octave":(midi//12)-1,"midi":midi,
                "start":round(float(times[i]),3),"end":round(float(times[i])+0.25,3),
                "isSharp":"#" in name,"displayName":name+str((midi//12)-1),
            })
            prev = name
    voiced_tl = [(float(times[i]),bool(voiced[i])) for i in range(len(times))]
    return notes[:64], voiced_tl

def detect_bpm(audio_path, sr=22050):
    import librosa
    y, sr = librosa.load(audio_path, sr=sr, mono=True, duration=90)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(float(tempo))
    while bpm > 120: bpm = round(bpm/2)
    while bpm < 55:  bpm = round(bpm*2)
    return bpm

def classify_segments(raw_segs, voiced_tl):
    result = []
    for seg in raw_segs:
        s, e   = seg["start"], seg["end"]
        frames = [(t,v) for t,v in voiced_tl if s<=t<=e]
        ratio  = sum(1 for _,v in frames if v)/max(1,len(frames))
        seg_type = "lyrics" if ratio>0.5 else ("spoken" if ratio>0.1 else "silence")
        result.append({"text":seg["text"].strip(),"start":s,"end":e,
                        "type":seg_type,"voicedRatio":round(ratio,2)})
    return result

def detect_sections(chords, classified, total):
    vocal_times = set()
    for seg in classified:
        if seg["type"]=="lyrics":
            t = seg["start"]
            while t<seg["end"]: vocal_times.add(int(t)); t+=1.0
    window=16.0; sections=[]; prev_label=None
    vocal_count=instr_count=0; t=0.0
    while t<total:
        end      = min(t+window, total)
        is_vocal = any(int(x) in vocal_times for x in np.arange(t,end,1.0))
        seg_chords = list(dict.fromkeys(c["chord"] for c in chords if t<=c["time"]<end))[:6]
        if is_vocal:
            vocal_count+=1
            names    = {1:"verse1",2:"chorus",3:"verse2",4:"chorus2"}
            displays = {1:"1절",2:"후렴",3:"2절",4:"후렴 반복"}
            name    = names.get(vocal_count, f"section{vocal_count}")
            display = displays.get(vocal_count, f"섹션{vocal_count}")
        else:
            instr_count+=1
            if t<8:                   name,display="intro","인트로"
            elif end>=total-8:        name,display="outro","아웃트로"
            elif prev_label=="vocal": name,display="interlude","간주"
            else:                     name,display="bridge","브릿지"
        lbl = "vocal" if is_vocal else "instrumental"
        if sections and sections[-1]["name"]==name:
            sections[-1]["end"]=round(end,1)
            sections[-1]["duration"]=round(end-sections[-1]["start"],1)
            sections[-1]["chords"]=list(dict.fromkeys(sections[-1]["chords"]+seg_chords))[:8]
        else:
            sections.append({"name":name,"display":display,
                "start":round(t,1),"end":round(end,1),"duration":round(end-t,1),
                "label":lbl,"chords":seg_chords,"hasVocal":is_vocal})
            prev_label=lbl
        t+=window
    return sections

def convert_to_wav(input_path, out_dir):
    import librosa, soundfile as sf
    wav = os.path.join(out_dir,"input.wav")
    y, sr = librosa.load(input_path, sr=44100, mono=False)
    if y.ndim==1: y=np.stack([y,y])
    sf.write(wav, y.T, sr)
    return wav

def separate_audio(input_path, out_dir):
    try:
        print("Demucs 시작...", flush=True)
        wav = convert_to_wav(input_path, out_dir)
        res = subprocess.run(
            ["python3","-m","demucs","--two-stems","vocals","--out",out_dir,wav],
            capture_output=True, text=True, timeout=300
        )
        print(f"Demucs returncode: {res.returncode}", flush=True)
        if res.returncode!=0: return None, None
        out = Path(out_dir)
        voc = list(out.rglob("vocals.wav"))
        nov = list(out.rglob("no_vocals.wav"))
        return (str(voc[0]) if voc else None),(str(nov[0]) if nov else None)
    except Exception as e:
        print(f"Demucs 오류: {e}", flush=True)
        return None, None

def process_audio(audio_path, tmp_dir, hint_title="", hint_artist=""):
    try:
        sep_dir = os.path.join(tmp_dir,"sep")
        os.makedirs(sep_dir, exist_ok=True)
        vocal_path, instr_path = separate_audio(audio_path, sep_dir)
        used_demucs = vocal_path is not None
        chord_src   = instr_path if instr_path else audio_path
        vocal_src   = vocal_path if vocal_path  else audio_path

        print("조성 감지...", flush=True)
        key_note, mode = detect_key(chord_src)
        key_display    = key_note+("" if mode=="major" else "m")
        print(f"조성: {key_display}", flush=True)

        print("코드 감지...", flush=True)
        chords, total_dur = detect_chords(chord_src, key_note, mode)

        print("멜로디 감지...", flush=True)
        melody, voiced_tl = detect_melody(vocal_src)

        bpm = detect_bpm(audio_path)
        print(f"BPM: {bpm}", flush=True)

        print("가사 인식...", flush=True)
        raw_segs, lang = [], "ko"
        try:
            segs_gen, info = whisper_model.transcribe(
                vocal_src, language="ko", task="transcribe", beam_size=5,
                initial_prompt="찬양 예배 하나님 주님 예수님 그리스도 성령 할렐루야",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=800,speech_pad_ms=300),
                temperature=0, no_speech_threshold=0.35,
            )
            lang = info.language
            for s in list(segs_gen):
                if s.text.strip() and len(s.text.strip())>1:
                    raw_segs.append({"text":s.text.strip(),"start":round(s.start,2),"end":round(s.end,2)})
            print(f"가사 {len(raw_segs)}개", flush=True)
        except Exception as e:
            print(f"Whisper 오류: {e}", flush=True)

        classified   = classify_segments(raw_segs, voiced_tl)
        lyrics_segs  = [s for s in classified if s["type"]=="lyrics"]
        spoken_segs  = [s for s in classified if s["type"]=="spoken"]
        full_lyrics  = "\n".join(s["text"] for s in lyrics_segs)
        sections     = detect_sections(chords, classified, total_dur)

        lyrics_with_chords = []
        for seg in lyrics_segs:
            seg_chords = list(dict.fromkeys(
                c["chord"] for c in chords if seg["start"]<=c["time"]<=seg["end"]
            ))
            lyrics_with_chords.append({"text":seg["text"],"chords":seg_chords,
                "start":seg["start"],"end":seg["end"]})

        spoken_with_chords = []
        for seg in spoken_segs:
            seg_chords = list(dict.fromkeys(
                c["chord"] for c in chords if seg["start"]<=c["time"]<=seg["end"]
            ))
            spoken_with_chords.append({"text":seg["text"],"chords":seg_chords,
                "start":seg["start"],"end":seg["end"]})

        print(f"완료! Key:{key_display} BPM:{bpm} Demucs:{used_demucs}", flush=True)

        return jsonify({
            "success":"true","key":key_display,"mode":mode,"bpm":bpm,
            "language":lang,"duration":total_dur,"usedDemucs":used_demucs,
            "chords":[c["chord"] for c in chords],"chordsWithTime":chords,
            "lyrics":full_lyrics,"lyricsSegments":lyrics_with_chords,
            "spokenSegments":spoken_with_chords,"sections":sections,
            "notes":melody,"songTitle":hint_title,"songArtist":hint_artist,
        })
    except Exception as e:
        import traceback
        print("오류:", traceback.format_exc(), flush=True)
        return jsonify({"error":str(e)}), 500

# ══════════════════════════════════════════════════
# API 엔드포인트
# ══════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status":"ok"})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """MP3 파일 업로드 채보 (선택적 사용)"""
    if 'file' not in request.files:
        return jsonify({"error":"파일 없음"}), 400
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({"error":"지원하지 않는 형식"}), 400
    tmp_dir  = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir,"input."+file.filename.rsplit('.',1)[1].lower())
    file.save(tmp_path)
    try:
        return process_audio(tmp_path, tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@app.route('/analyze_sheet', methods=['POST'])
def analyze_sheet():
    """악보 이미지 → Claude Vision → 코드/가사/키 추출 (핵심!)"""
    if 'file' not in request.files:
        return jsonify({"error":"파일이 없어요"}), 400
    file     = request.files['file']
    filename = file.filename.lower()
    try:
        import anthropic, base64
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            return jsonify({"error":"ANTHROPIC_API_KEY 미설정"}), 500
        client = anthropic.Anthropic(api_key=api_key)

        img_bytes  = file.read()
        img_data   = base64.b64encode(img_bytes).decode('utf-8')
        media_type = ("image/png" if filename.endswith('.png')
                      else "application/pdf" if filename.endswith('.pdf')
                      else "image/jpeg")

        print("Claude Vision 악보 분석 중...", flush=True)
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2000,
            messages=[{"role":"user","content":[
                {"type":"image","source":{"type":"base64",
                    "media_type":media_type,"data":img_data}},
                {"type":"text","text":"""이 악보 이미지를 분석해서 JSON만 반환해주세요.

{
  "key": "F",
  "timeSignature": "4/4",
  "bpm": 76,
  "chords": ["F","Bb","C","Gm"],
  "chordsInOrder": ["F","C","Bb","F","Gm","C","F"],
  "lyrics": "전체 가사",
  "lyricsLines": ["줄1","줄2"],
  "sections": [
    {"name":"인트로","chords":["F","C"]},
    {"name":"1절","chords":["F","Bb","C","Gm"]},
    {"name":"후렴","chords":["Bb","F","C","Dm"]}
  ],
  "title": "곡 제목",
  "artist": "아티스트"
}

읽을 수 없으면 null. 코드는 악보 그대로 (F/A, Bb, Gm7 등)."""}
            ]}]
        )
        import json, re
        raw = re.sub(r'```json\s*','',response.content[0].text.strip())
        raw = re.sub(r'```\s*','',raw)
        result = json.loads(raw)
        print(f"악보 분석 완료! Key:{result.get('key')}", flush=True)
        return jsonify({"success":True,**result})
    except Exception as e:
        import traceback
        print("악보 분석 오류:", traceback.format_exc(), flush=True)
        return jsonify({"error":str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5000)))
