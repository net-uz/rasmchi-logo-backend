
import base64
import io
import json
import os
import re
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI(title="Rasmchi Logo Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def resize_image(data: bytes, max_size: int = 1024) -> tuple[bytes, str]:
    img = Image.open(io.BytesIO(data))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail((max_size, max_size))
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue(), "image/png"


async def download_image(url: str) -> tuple[bytes, str]:
    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
        r = await client.get(url)
        r.raise_for_status()
        mime = r.headers.get("content-type", "image/jpeg").split(";")[0]
        return r.content, mime


async def gemini_analyze(image_b64: str, mime: str, prompt_uz: str) -> dict:
    if not GEMINI_KEY:
        return {"error": "GEMINI_API_KEY sozlanmagan"}

    system_text = (
        "Rasmni diqqat bilan ko'rib chiq. Quyidagi muammolarni aniqla: ortiqcha barmoqlar, ortiqcha a'zolar, buzilgan yoki noto'g'ri yozilgan matn, xato anatomiya, xira, artefaktlar, yomon kompozitsiya, keraksiz ob'ektlar, noto'g'ri ranglar, yomon yorug'lik. "
        f"Foydalanuvchi tavsifini bildirdi: '{prompt_uz}'. "
        "Faqat bitta JSON obyekt qaytar: issues_uz (muammolarni qisqa o'zbekcha, 40 ta so'zgacha), fix_en (muammolarni tuzatadigan qisqa inglizcha prompt), fix_uz (shu o'zgarishni qisqa o'zbekcha, 30 ta so'zgacha)."
    )

    url = f"{GEMINI_BASE}/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": system_text},
                    {"inline_data": {"mime_type": mime, "data": image_b64}},
                ],
            }
        ],
        "generationConfig": {"responseMimeType": "application/json"},
    }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=body)
        if r.status_code != 200:
            return {"error": f"Gemini xato: {r.status_code}", "detail": r.text[:500]}
        data = r.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "{}")
        )

    try:
        result = json.loads(text)
    except Exception:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        if m:
            try:
                result = json.loads(m.group(1))
            except Exception:
                result = None
        else:
            result = None

    if not result or not isinstance(result, dict):
        result = {
            "issues_uz": "Rasmda aniqlanadigan muammolar: sifat yoki anatomiyada noaniqliklar. Tekshirishda xatolik.",
            "fix_en": "improve overall quality, fix any anatomy or text distortions, remove artifacts and sharpen details",
            "fix_uz": "umumiy sifatni oshiring, anatomiya yoki matn buzilishlarini to'g'rilang, artifaktlarni olib tashlang",
        }

    for key in ("issues_uz", "fix_en", "fix_uz"):
        if key not in result:
            result[key] = ""
    return result


@app.get("/")
async def health():
    return {"ok": True}


@app.post("/analyze")
async def analyze(
    image_url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    prompt_uz: str = Form(""),
):
    try:
        if file:
            raw = await file.read()
            mime = file.content_type or "image/jpeg"
        elif image_url:
            raw, mime = await download_image(image_url)
        else:
            return JSONResponse(
                {"error": "image_url yoki file kerak"}, status_code=400
            )

        img_bytes, mime = resize_image(raw)
        b64 = base64.b64encode(img_bytes).decode()
        result = await gemini_analyze(b64, mime, prompt_uz)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
