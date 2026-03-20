#!/usr/bin/env python3
"""SynthID Checker — Web app to detect Nano Banana / Google AI watermarks via C2PA."""

import io
import json
import os
import re
import tempfile

import c2pa
import requests as http_requests
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
IGNORE_PATTERNS = ["apple-icon", "typefully-user-avatars", "favicon", "logo",
                    "sanity.io", "/icon/", "gstatic.com"]


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


def check_synthid_bytes(data: bytes, filename: str = "") -> dict:
    """Check C2PA / SynthID on raw image bytes."""
    result = {
        "filename": filename,
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

        active = manifest_json.get("active_manifest")
        manifest = manifest_json["manifests"].get(active, {})
        gen_info = manifest.get("claim_generator_info", [])
        result["generator"] = ", ".join(g.get("name", "") for g in gen_info)

        details = []
        for a in manifest.get("assertions", []):
            if "actions" not in a.get("label", ""):
                continue
            for action in a.get("data", {}).get("actions", []):
                desc = action.get("description", "")
                dst = action.get("digitalSourceType", "")
                if "SynthID" in desc:
                    result["has_synthid"] = True
                    details.append(desc)
                if "Google Generative AI" in desc:
                    result["is_google_ai"] = True
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

    # Deduplicate, prefer originals
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

    # Deduplicate, keep longest URL (has key param)
    seen = {}
    for u in images:
        base = u.split("?")[0]
        if base not in seen or len(u) > len(seen[base]):
            seen[base] = u
    return list(seen.values())


def download_and_check(img_url):
    """Download an image URL and check for SynthID."""
    try:
        resp = http_requests.get(img_url, timeout=30)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "text/html" in ct:
            return {"filename": img_url.split("/")[-1].split("?")[0],
                    "has_c2pa": False, "has_synthid": False, "is_google_ai": False,
                    "generator": "", "details": "Skipped (got HTML, likely auth required)",
                    "source_url": img_url}
        filename = img_url.split("/")[-1].split("?")[0] or "image"
        result = check_synthid_bytes(resp.content, filename)
        result["source_url"] = img_url
        return result
    except Exception as e:
        return {"filename": img_url.split("/")[-1].split("?")[0],
                "has_c2pa": False, "has_synthid": False, "is_google_ai": False,
                "generator": "", "details": f"Download error: {e}",
                "source_url": img_url}


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


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
        result = check_synthid_bytes(data, f.filename or "unknown")
        results.append(result)

    flagged = sum(1 for r in results if r["has_synthid"] or r["is_google_ai"])
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
        flagged = sum(1 for r in results if r["has_synthid"] or r["is_google_ai"])
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
