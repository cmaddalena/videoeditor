"""
Reels Engine v2 - Microservicio FFmpeg para Railway
ProspectOS / Configurable para cualquier usuario
"""

import os
import json
import uuid
import subprocess
import requests
import traceback
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

WORK_DIR   = os.environ.get("WORK_DIR", "/tmp/reels")
OUTPUT_DIR = os.path.join(WORK_DIR, "output")
for d in [WORK_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

API_SECRET = os.environ.get("API_SECRET", "")


# ═══════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════

def check_auth(req):
    if not API_SECRET:
        return True
    token = req.headers.get("Authorization", "").replace("Bearer ", "")
    return token == API_SECRET


# ═══════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════

def run_cmd(cmd: list, job_id: str) -> dict:
    print(f"[{job_id}] CMD: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[{job_id}] ERROR:\n{result.stderr[-800:]}")
        return {"success": False, "error": result.stderr[-500:]}
    return {"success": True}


def download_file(url: str, dest: str, job_id: str) -> bool:
    try:
        print(f"[{job_id}] Descargando: {url[:80]}...")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"[{job_id}] Descargado: {size_mb:.1f}MB → {dest}")
        return True
    except Exception as e:
        print(f"[{job_id}] Download error: {e}")
        return False


def seconds_to_ass(s: float) -> str:
    h  = int(s // 3600)
    m  = int((s % 3600) // 60)
    sc = int(s % 60)
    cs = int((s - int(s)) * 100)
    return f"{h}:{m:02d}:{sc:02d}.{cs:02d}"


# ═══════════════════════════════════════════════
# GENERADORES DE SUBTÍTULOS
# ═══════════════════════════════════════════════

def build_ass_header(cfg: dict) -> str:
    """
    cfg keys usados:
      sub_font, sub_font_size, sub_color_primary (BGR hex sin #),
      sub_color_highlight, sub_outline_color, sub_bold,
      sub_outline_size, sub_shadow, sub_position (1-9 ASS alignment),
      sub_margin_v
    """
    font        = cfg.get("sub_font", "Arial Black")
    size        = cfg.get("sub_font_size", 74)
    primary     = cfg.get("sub_color_primary", "FFFFFF")      # blanco
    highlight   = cfg.get("sub_color_highlight", "00FFFF")    # amarillo/cyan
    outline_col = cfg.get("sub_outline_color", "000000")      # negro
    bold        = -1 if cfg.get("sub_bold", True) else 0
    outline     = cfg.get("sub_outline_size", 4)
    shadow      = cfg.get("sub_shadow", 2)
    alignment   = cfg.get("sub_position", 2)                  # 2=centro-abajo
    margin_v    = cfg.get("sub_margin_v", 180)

    return f"""[Script Info]
Title: Reels Engine Subtitles
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},&H00{primary},&H000000FF,&H00{outline_col},&H80000000,{bold},0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},60,60,{margin_v},1
Style: Highlight,{font},{size},&H00{highlight},&H000000FF,&H00{outline_col},&H80000000,{bold},0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def subtitles_word_by_word(words: list, cfg: dict) -> str:
    """Cada palabra se ilumina en su timestamp exacto. Estilo CapCut."""
    ass    = build_ass_header(cfg)
    lines  = []
    group  = int(cfg.get("sub_words_per_line", 4))
    highlight = cfg.get("sub_color_highlight", "00FFFF")

    groups = [words[i:i+group] for i in range(0, len(words), group)]

    for chunk in groups:
        g_start = chunk[0].get("start", 0)
        g_end   = chunk[-1].get("end", g_start + 1)

        for active_i, active_w in enumerate(chunk):
            w_start = active_w.get("start", g_start)
            w_end   = active_w.get("end",   g_end)
            parts   = []
            for j, w in enumerate(chunk):
                txt = w.get("word", "").strip()
                if j == active_i:
                    parts.append(f"{{\\c&H00{highlight}&}}{txt}{{\\c&H00{cfg.get('sub_color_primary','FFFFFF')}&}}")
                else:
                    parts.append(txt)
            line = " ".join(parts)
            lines.append(f"Dialogue: 0,{seconds_to_ass(w_start)},{seconds_to_ass(w_end)},Default,,0,0,0,,{line}")

    return ass + "\n".join(lines)


def subtitles_line_by_line(words: list, cfg: dict) -> str:
    """Línea completa aparece de golpe. Clásico."""
    ass   = build_ass_header(cfg)
    lines = []
    group = int(cfg.get("sub_words_per_line", 5))
    groups = [words[i:i+group] for i in range(0, len(words), group)]

    for chunk in groups:
        g_start = chunk[0].get("start", 0)
        g_end   = chunk[-1].get("end", g_start + 1)
        text    = " ".join(w.get("word", "").strip() for w in chunk)
        lines.append(f"Dialogue: 0,{seconds_to_ass(g_start)},{seconds_to_ass(g_end)},Default,,0,0,0,,{text}")

    return ass + "\n".join(lines)


def subtitles_karaoke(words: list, cfg: dict) -> str:
    """Texto se va coloreando palabra a palabra (estilo karaoke ASS nativo)."""
    ass   = build_ass_header(cfg)
    lines = []
    group = int(cfg.get("sub_words_per_line", 5))
    highlight = cfg.get("sub_color_highlight", "00FFFF")
    groups = [words[i:i+group] for i in range(0, len(words), group)]

    for chunk in groups:
        g_start = chunk[0].get("start", 0)
        g_end   = chunk[-1].get("end", g_start + 1)
        parts   = []
        for w in chunk:
            dur_cs = int((w.get("end", 0) - w.get("start", 0)) * 100)
            parts.append(f"{{\\kf{dur_cs}}}{{\\c&H00{highlight}&}}{w.get('word','').strip()} ")
        text = "".join(parts)
        lines.append(f"Dialogue: 0,{seconds_to_ass(g_start)},{seconds_to_ass(g_end)},Default,,0,0,0,,{text}")

    return ass + "\n".join(lines)


SUBTITLE_BUILDERS = {
    "word_highlight": subtitles_word_by_word,
    "line":           subtitles_line_by_line,
    "karaoke":        subtitles_karaoke,
}


# ═══════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    return jsonify({
        "status": "ok",
        "ffmpeg": r.stdout.split("\n")[0] if r.returncode == 0 else "missing"
    })


@app.route("/compress-audio", methods=["POST"])
def compress_audio():
    """
    Acepta video de dos formas:
    1. Archivo binario directo (multipart/form-data con campo 'file') ← desde n8n
    2. JSON con video_url (para tests manuales)

    Returns: audio MP3 comprimido listo para Whisper (~2MB)
    """
    if not check_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    job_id = str(uuid.uuid4())[:8]

    try:
        raw_path   = f"{WORK_DIR}/{job_id}_raw"
        audio_path = f"{OUTPUT_DIR}/{job_id}_audio.mp3"

        # ── Modo 1: archivo binario desde n8n (multipart/form-data) ──
        if request.files and "file" in request.files:
            f = request.files["file"]
            ext = os.path.splitext(f.filename)[1] if f.filename else ".mp4"
            raw_path += ext
            print(f"[{job_id}] Recibiendo archivo binario: {f.filename} ({ext})")
            f.save(raw_path)
            size_mb = os.path.getsize(raw_path) / 1024 / 1024
            print(f"[{job_id}] Guardado: {size_mb:.1f}MB → {raw_path}")

        # ── Modo 2: JSON con video_url (para tests) ──
        elif request.is_json:
            data = request.get_json()
            video_url = data.get("video_url")
            if not video_url:
                return jsonify({"error": "Falta video_url o archivo"}), 400
            raw_path += ".mp4"
            if not download_file(video_url, raw_path, job_id):
                return jsonify({"error": "No pude descargar el video"}), 400
        else:
            return jsonify({"error": "Mandá el video como multipart/form-data (campo 'file') o JSON con video_url"}), 400

        # ── Extraer audio comprimido (mono 64kbps → ~0.5MB/min) ──
        res = run_cmd([
            "ffmpeg", "-y", "-i", raw_path,
            "-vn",           # sin video
            "-ac", "1",      # mono
            "-ar", "16000",  # 16kHz suficiente para STT
            "-b:a", "64k",
            audio_path
        ], job_id)

        try: os.remove(raw_path)
        except: pass

        if not res["success"]:
            return jsonify({"error": "Error extrayendo audio", "detail": res["error"]}), 500

        size_mb = os.path.getsize(audio_path) / 1024 / 1024
        print(f"[{job_id}] ✅ Audio extraído: {size_mb:.2f}MB")

        return jsonify({
            "success":   True,
            "job_id":    job_id,
            "audio_url": f"/download-audio/{job_id}",
            "size_mb":   round(size_mb, 2)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/download-audio/<job_id>", methods=["GET"])
def download_audio(job_id):
    safe   = secure_filename(job_id)
    path   = f"{OUTPUT_DIR}/{safe}_audio.mp3"
    if not os.path.exists(path):
        return jsonify({"error": "Audio no encontrado"}), 404
    return send_file(path, mimetype="audio/mpeg", as_attachment=True,
                     download_name=f"audio_{safe}.mp3")


@app.route("/process-reel", methods=["POST"])
def process_reel():
    """
    Endpoint principal. Recibe config completa desde n8n y genera el reel.

    Body completo:
    {
      // — Video source —
      "video_url":           "https://...",        REQUERIDO
      "clip_start":          45.2,                 REQUERIDO
      "clip_end":            98.7,                 REQUERIDO
      "words":               [...],                REQUERIDO (de Whisper)

      // — Efectos —
      "energy_moments":      [52.1, 71.3],         momentos de zoom
      "zoom_intensity":      0.08,                 0.0 a 0.15 (default 0.08)
      "zoom_duration_sec":   1.5,                  duración del zoom (default 1.5)
      "zoom_enabled":        true,
      "color_grade":         "energetic",          energetic|professional|calm|none
      "vignette_enabled":    false,

      // — Subtítulos —
      "subtitle_style":      "word_highlight",     word_highlight|line|karaoke
      "sub_font":            "Arial Black",
      "sub_font_size":       74,
      "sub_color_primary":   "FFFFFF",             BGR hex sin #
      "sub_color_highlight": "00FFFF",             color de la palabra activa
      "sub_outline_color":   "000000",
      "sub_bold":            true,
      "sub_outline_size":    4,
      "sub_shadow":          2,
      "sub_position":        2,                    2=centro-abajo, 8=arriba
      "sub_margin_v":        180,
      "sub_words_per_line":  4,

      // — Overlay —
      "overlay_image_url":   "https://logo.png",  opcional
      "overlay_image_pos":   "top-right",         top-right|top-left|bottom-right|bottom-left
      "overlay_image_size":  200,                 ancho en px
      "overlay_text":        "@usuario",          opcional
      "overlay_text_pos":    "bottom",            bottom|top

      // — Intro/Outro —
      "intro_text":          "",                  texto intro (vacío = sin intro)
      "outro_text":          "",                  texto outro (vacío = sin outro)

      // — B-Rolls (las URLs ya vienen procesadas desde n8n/Pexels) —
      "broll_urls":          [],

      // — Output —
      "style":               "energetic"
    }
    """
    if not check_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    job_id = str(uuid.uuid4())[:8]
    cfg    = request.json or {}

    try:
        print(f"\n[{job_id}] ═══ NUEVO REEL ═══")

        # ── Paths ──
        p = lambda name: f"{WORK_DIR}/{job_id}_{name}"
        raw_path      = p("raw.mp4")
        clip_path     = p("clip.mp4")
        vertical_path = p("vertical.mp4")
        zoom_path     = p("zoom.mp4")
        graded_path   = p("graded.mp4")
        sub_file      = p("subs.ass")
        logo_path     = p("logo.png")
        output_path   = f"{OUTPUT_DIR}/{job_id}_reel.mp4"

        clip_start = float(cfg.get("clip_start", 0))
        clip_end   = float(cfg.get("clip_end", 60))
        duration   = clip_end - clip_start

        # ── 1. Descargar video ──
        if not download_file(cfg["video_url"], raw_path, job_id):
            return jsonify({"error": "No pude descargar el video"}), 400

        # ── 2. Cortar clip ──
        print(f"[{job_id}] Cortando {clip_start}s → {clip_end}s ({duration:.1f}s)")
        res = run_cmd(["ffmpeg", "-y", "-i", raw_path,
                       "-ss", str(clip_start), "-to", str(clip_end),
                       "-c", "copy", clip_path], job_id)
        if not res["success"]:
            return jsonify({"error": "Error cortando clip", "detail": res["error"]}), 500
        try: os.remove(raw_path)
        except: pass

        # ── 3. Verticalizar 9:16 ──
        print(f"[{job_id}] Verticalizando a 1080x1920...")
        res = run_cmd(["ffmpeg", "-y", "-i", clip_path,
                       "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1",
                       "-c:a", "aac", "-b:a", "192k",
                       "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                       vertical_path], job_id)
        if not res["success"]:
            return jsonify({"error": "Error verticalizando", "detail": res["error"]}), 500
        try: os.remove(clip_path)
        except: pass

        current = vertical_path

        # ── 4. Zoom dinámico ──
        zoom_enabled  = cfg.get("zoom_enabled", True)
        energy_moments = cfg.get("energy_moments", [])

        if zoom_enabled and energy_moments:
            print(f"[{job_id}] Zoom en {len(energy_moments)} momentos...")
            intensity  = float(cfg.get("zoom_intensity", 0.08))
            zoom_dur   = float(cfg.get("zoom_duration_sec", 1.5))
            fps        = 30
            step       = intensity / (fps * zoom_dur / 2)

            # Tomar solo los 3 primeros momentos para no sobrecargar
            valid_moments = [
                float(m) - clip_start
                for m in energy_moments[:3]
                if clip_start < float(m) < clip_end
            ]

            if valid_moments:
                # Construir expresión zoompan compuesta
                # Para simplificar, usamos el primer momento
                m0   = valid_moments[0]
                sf   = int(m0 * fps)
                ef   = sf + int(fps * zoom_dur)
                half = (ef - sf) // 2
                zoom_expr = (
                    f"if(between(on,{sf},{sf+half}),min(zoom+{step:.5f},1+{intensity}),"
                    f"if(between(on,{sf+half},{ef}),max(zoom-{step:.5f},1.0),1.0))"
                )
                res = run_cmd(["ffmpeg", "-y", "-i", current,
                               "-vf", f"zoompan=z='{zoom_expr}':d=1:s=1080x1920:fps={fps}",
                               "-c:a", "copy", zoom_path], job_id)
                if res["success"]:
                    current = zoom_path
                else:
                    print(f"[{job_id}] Zoom falló, continuando sin zoom")

        # ── 5. Color grade ──
        grade = cfg.get("color_grade", "energetic")
        grades = {
            "energetic":    "eq=contrast=1.15:saturation=1.3:brightness=0.02:gamma=0.95",
            "professional": "eq=contrast=1.05:saturation=0.85:brightness=0.0",
            "calm":         "eq=contrast=1.0:saturation=1.1:brightness=0.01",
            "none":         None
        }
        vf_grade = grades.get(grade)
        vignette  = cfg.get("vignette_enabled", False)

        if vf_grade or vignette:
            filters = []
            if vf_grade:
                filters.append(vf_grade)
            if vignette:
                filters.append("vignette=PI/5")
            vf_str = ",".join(filters)
            print(f"[{job_id}] Color grade: {grade}")
            res = run_cmd(["ffmpeg", "-y", "-i", current,
                           "-vf", vf_str,
                           "-c:a", "copy", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                           graded_path], job_id)
            if res["success"]:
                current = graded_path

        # ── 6. Generar subtítulos ──
        words = cfg.get("words", [])
        adjusted = [
            {**w, "start": float(w["start"]) - clip_start,
                   "end":   float(w["end"])   - clip_start}
            for w in words
            if clip_start <= float(w.get("start", 0)) <= clip_end
        ]

        sub_style   = cfg.get("subtitle_style", "word_highlight")
        builder     = SUBTITLE_BUILDERS.get(sub_style, subtitles_word_by_word)
        ass_content = builder(adjusted, cfg)

        with open(sub_file, "w", encoding="utf-8") as f:
            f.write(ass_content)
        print(f"[{job_id}] Subtítulos '{sub_style}': {len(adjusted)} palabras")

        # ── 7. Overlay de texto ──
        overlay_text     = cfg.get("overlay_text", "").replace("'", "\\'").replace(":", "\\:")
        overlay_text_pos = cfg.get("overlay_text_pos", "bottom")
        drawtext_filter  = ""
        if overlay_text:
            y_expr = "h-120" if overlay_text_pos == "bottom" else "60"
            drawtext_filter = (
                f",drawtext=text='{overlay_text}'"
                f":fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                f":fontsize=44:fontcolor=white"
                f":x=(w-text_w)/2:y={y_expr}"
                f":box=1:boxcolor=black@0.55:boxborderw=14"
            )

        # ── 8. Render con subtítulos (+ drawtext) ──
        overlay_image_url = cfg.get("overlay_image_url", "")
        has_logo = False
        if overlay_image_url:
            has_logo = download_file(overlay_image_url, logo_path, job_id)

        print(f"[{job_id}] Render final...")

        if has_logo:
            pos_map = {
                "top-right":    f"W-w-40:40",
                "top-left":     f"40:40",
                "bottom-right": f"W-w-40:H-h-40",
                "bottom-left":  f"40:H-h-40",
            }
            logo_pos  = pos_map.get(cfg.get("overlay_image_pos", "top-right"), "W-w-40:40")
            logo_size = int(cfg.get("overlay_image_size", 200))

            res = run_cmd([
                "ffmpeg", "-y",
                "-i", current,
                "-i", logo_path,
                "-filter_complex",
                f"[0:v]ass={sub_file}{drawtext_filter}[subbed];"
                f"[1:v]scale={logo_size}:-1[logo];"
                f"[subbed][logo]overlay={logo_pos}",
                "-c:a", "copy", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                output_path
            ], job_id)
        else:
            res = run_cmd([
                "ffmpeg", "-y", "-i", current,
                "-vf", f"ass={sub_file}{drawtext_filter}",
                "-c:a", "copy", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                output_path
            ], job_id)

        if not res["success"]:
            return jsonify({"error": "Error en render final", "detail": res["error"]}), 500

        # ── Limpiar temporales ──
        for f in [vertical_path, zoom_path, graded_path, sub_file, logo_path]:
            try: os.remove(f)
            except: pass

        output_size_mb = round(os.path.getsize(output_path) / 1024 / 1024, 1)
        print(f"[{job_id}] ✅ LISTO → {output_size_mb}MB")

        return jsonify({
            "success":         True,
            "job_id":          job_id,
            "output_url":      f"/download/{job_id}",
            "duration":        round(duration, 1),
            "output_size_mb":  output_size_mb,
            "subtitles_words": len(adjusted)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/download/<job_id>", methods=["GET"])
def download_video(job_id):
    safe = secure_filename(job_id)
    path = f"{OUTPUT_DIR}/{safe}_reel.mp4"
    if not os.path.exists(path):
        return jsonify({"error": "Video no encontrado"}), 404
    return send_file(path, mimetype="video/mp4", as_attachment=True,
                     download_name=f"reel_{safe}.mp4")

@app.route("/search-brolls", methods=["POST"])
def search_brolls():
    if not check_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    data     = request.json or {}
    keywords = data.get("keywords", [])
    count    = int(data.get("count", 3))

    if not keywords:
        return jsonify({"urls": []})

    pexels_key = os.environ.get("PEXELS_API_KEY", "")
    if not pexels_key:
        return jsonify({"error": "Falta PEXELS_API_KEY"}), 500

    try:
        query = keywords[0]
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": pexels_key},
            params={"query": query, "per_page": count, "orientation": "portrait"},
            timeout=30
        )
        r.raise_for_status()
        videos = r.json().get("videos", [])
        urls = [
            (v.get("video_files") or [{}])[0].get("link", "")
            for v in videos
            if v.get("video_files")
        ]
        return jsonify({"success": True, "urls": [u for u in urls if u]})
    except Exception as e:
        return jsonify({"error": str(e), "urls": []}), 500
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
