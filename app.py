#!/usr/bin/env python3
"""AI Image Checker — Detect AI-generated images via C2PA metadata + Gemini vision analysis."""

import io
import json
import os
import re

import c2pa
import requests as http_requests
from flask import Flask, request, jsonify, send_from_directory
from google import genai

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
IGNORE_PATTERNS = ["apple-icon", "typefully-user-avatars", "favicon", "logo",
                    "sanity.io", "/icon/", "gstatic.com"]

_gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

ANALYSIS_PROMPT = """@synthid Was this image created or edited by Google AI?

Respond with ONLY a JSON object (no markdown, no code fences) in this exact format:
{
  "has_synthid": true or false,
  "confidence": "high", "medium", or "low",
  "reasoning": "Brief explanation of the SynthID check result"
}"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def detect_mime(header: bytes) -> str:
    if header[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if header[:4] == b"RIFF" and len(header) >= 12 and header[8:12] == b"WEBP":
        return "image/webp"
    if header[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return ""


def check_c2pa(data: bytes, filename: str = "") -> dict:
    """Check C2PA Content Credentials metadata."""
    result = {
        "has_c2pa": False,
        "has_synthid": False,
        "is_google_ai": False,
        "generator": "",
        "details": "",
    }
    mime = detect_mime(data[:12])
    if not mime:
        ext = os.path.splitext(filename)[1].lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                    ".webp": "image/webp", ".gif": "image/gif"}
        mime = mime_map.get(ext, "")
    if not mime:
        result["details"] = "Could not determine image format"
        return result

    try:
        reader = c2pa.Reader(mime, io.BytesIO(data))
        manifest_json = json.loads(reader.json())
        result["has_c2pa"] = True

        # Walk ALL manifests in the chain, not just the active one
        all_manifests = manifest_json.get("manifests", {})
        active = manifest_json.get("active_manifest")
        active_manifest = all_manifests.get(active, {})
        gen_info = active_manifest.get("claim_generator_info", [])
        result["generator"] = ", ".join(g.get("name", "") for g in gen_info)

        details = []
        for manifest in all_manifests.values():
            for a in manifest.get("assertions", []):
                if "actions" not in a.get("label", ""):
                    continue
                for action in a.get("data", {}).get("actions", []):
                    desc = action.get("description", "")
                    dst = action.get("digitalSourceType", "")
                    if "SynthID" in desc:
                        result["has_synthid"] = True
                        if desc not in details:
                            details.append(desc)
                    if "Google Generative AI" in desc:
                        result["is_google_ai"] = True
                        if desc not in details:
                            details.append(desc)
                    if "trainedAlgorithmicMedia" in dst and not details:
                        details.append("digitalSourceType: trainedAlgorithmicMedia")
        result["details"] = "; ".join(details)
    except c2pa.C2paError as e:
        if "ManifestNotFound" not in str(e):
            result["details"] = f"C2PA error: {e}"
    except Exception as e:
        result["details"] = f"Error: {e}"
    return result


def check_gemini(data: bytes, filename: str = "") -> dict:
    """Check an image for SynthID using Gemini."""
    result = {
        "has_synthid": False,
        "confidence": "",
        "details": "",
    }
    mime = detect_mime(data[:12])
    if not mime:
        ext = os.path.splitext(filename)[1].lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                    ".webp": "image/webp", ".gif": "image/gif"}
        mime = mime_map.get(ext, "")
    if not mime:
        result["details"] = "Could not determine image format"
        return result
    if not _gemini_api_key:
        result["details"] = "No Gemini API key configured"
        return result

    try:
        client = genai.Client(api_key=_gemini_api_key)
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=[
                genai.types.Part.from_bytes(data=data, mime_type=mime),
                ANALYSIS_PROMPT,
            ],
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        analysis = json.loads(text)
        result["has_synthid"] = bool(analysis.get("has_synthid", False))
        result["confidence"] = analysis.get("confidence", "")
        result["details"] = analysis.get("reasoning", "")
    except json.JSONDecodeError:
        result["details"] = f"Could not parse Gemini response: {text[:200]}"
    except Exception as e:
        result["details"] = f"Gemini API error: {e}"
    return result


def check_image(data: bytes, filename: str = "") -> dict:
    """Run both C2PA and Gemini checks, return combined result."""
    c2pa_result = check_c2pa(data, filename)
    gemini_result = check_gemini(data, filename)
    return {
        "filename": filename,
        "c2pa": c2pa_result,
        "gemini": gemini_result,
    }


# ── URL image extraction ─────────────────────────────────────────────────────

def find_media_urls(obj, results, depth=0):
    if depth > 20:
        return
    if isinstance(obj, str):
        return
    elif isinstance(obj, dict):
        if "public_url" in obj:
            url = obj["public_url"]
            if url and not any(p in url for p in IGNORE_PATTERNS):
                results.append(url)
            return
        for v in obj.values():
            find_media_urls(v, results, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            find_media_urls(item, results, depth + 1)


def extract_images_from_typefully(url):
    if not url.startswith("http"):
        url = "https://" + url
    resp = http_requests.get(url, timeout=20)
    resp.raise_for_status()
    html = resp.text

    images = []
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(\{.*?\})</script>', html, re.DOTALL)
    if match:
        data = json.loads(match.group(1))
        draft = data.get("props", {}).get("pageProps", {}).get("draft", {})
        find_media_urls(draft, images)

    if not images:
        found = re.findall(r"https?://[^\s\"'<>]+(?:\.jpg|\.jpeg|\.png|\.gif|\.webp)", html)
        images = [u for u in found if not any(p in u for p in IGNORE_PATTERNS)]

    seen = {}
    unique = []
    for img_url in images:
        base = img_url.split("/")[-1].split("?")[0]
        if base not in seen:
            seen[base] = img_url
            unique.append(img_url)
        elif "/original/" in img_url and "/resized/" in seen[base]:
            idx = unique.index(seen[base])
            unique[idx] = img_url
            seen[base] = img_url
    return unique


def extract_images_from_google_doc(url):
    if not url.startswith("http"):
        url = "https://" + url
    resp = http_requests.get(url, timeout=20)
    text = resp.text.replace("\\x3d", "=").replace("\\x26", "&").replace(
        "\\u003d", "=").replace("\\x3a", ":").replace("\\x2f", "/")

    images = []
    found = re.findall(
        r"(https://lh[0-9a-z-]*\.googleusercontent\.com/[^\s\"'\\<>,;)]+)", text)
    for u in found:
        if not any(p in u for p in IGNORE_PATTERNS):
            images.append(u)

    seen = {}
    for u in images:
        base = u.split("?")[0]
        if base not in seen or len(u) > len(seen[base]):
            seen[base] = u
    return list(seen.values())


def download_and_check(img_url):
    """Download an image URL and run both checks."""
    try:
        resp = http_requests.get(img_url, timeout=30)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "text/html" in ct:
            return {
                "filename": img_url.split("/")[-1].split("?")[0],
                "c2pa": {"has_c2pa": False, "has_synthid": False, "is_google_ai": False,
                         "generator": "", "details": "Skipped (HTML response)"},
                "gemini": {"has_synthid": False,
                           "confidence": "",
                           "details": "Skipped (HTML response)"},
                "source_url": img_url,
            }
        filename = img_url.split("/")[-1].split("?")[0] or "image"
        result = check_image(resp.content, filename)
        result["source_url"] = img_url
        return result
    except Exception as e:
        return {
            "filename": img_url.split("/")[-1].split("?")[0],
            "c2pa": {"has_c2pa": False, "has_synthid": False, "is_google_ai": False,
                     "generator": "", "details": f"Download error: {e}"},
            "gemini": {"has_synthid": False,
                       "confidence": "",
                       "details": f"Download error: {e}"},
            "source_url": img_url,
        }


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.route("/api/set-key", methods=["POST"])
def api_set_key():
    global _gemini_api_key
    body = request.get_json(silent=True) or {}
    key = (body.get("key") or "").strip()
    if not key:
        return jsonify({"error": "No key provided"}), 400
    _gemini_api_key = key
    return jsonify({"ok": True, "masked": key[:4] + "..." + key[-4:]})


@app.route("/api/key-status")
def api_key_status():
    if _gemini_api_key:
        return jsonify({"has_key": True, "masked": _gemini_api_key[:4] + "..." + _gemini_api_key[-4:]})
    return jsonify({"has_key": False})


@app.route("/api/check-images", methods=["POST"])
def api_check_images():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    results = []
    for f in files:
        data = f.read()
        if not data:
            continue
        result = check_image(data, f.filename or "unknown")
        results.append(result)

    flagged = sum(1 for r in results
                  if r["c2pa"]["has_synthid"] or r["c2pa"]["is_google_ai"]
                  or r["gemini"]["has_synthid"])
    return jsonify({
        "results": results,
        "summary": {"total": len(results), "checked": len(results), "flagged": flagged},
    })


@app.route("/api/check-url", methods=["POST"])
def api_check_url():
    body = request.get_json(silent=True) or {}
    url = (body.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        if "typefully.com" in url:
            image_urls = extract_images_from_typefully(url)
        elif "docs.google.com" in url or "drive.google.com" in url:
            image_urls = extract_images_from_google_doc(url)
        else:
            return jsonify({"error": "Unsupported URL. Paste a Typefully or Google Doc link."}), 400

        if not image_urls:
            return jsonify({
                "results": [],
                "summary": {"total": 0, "checked": 0, "flagged": 0},
                "message": "No images found in this link. The document may not be publicly accessible.",
            })

        results = [download_and_check(img_url) for img_url in image_urls]
        flagged = sum(1 for r in results
                      if r["c2pa"]["has_synthid"] or r["c2pa"]["is_google_ai"]
                      or r["gemini"]["is_ai_generated"])
        return jsonify({
            "results": results,
            "summary": {"total": len(results), "checked": len(results), "flagged": flagged},
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process URL: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(debug=debug, host="0.0.0.0", port=port)
