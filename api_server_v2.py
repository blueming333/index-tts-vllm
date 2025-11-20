import os
import asyncio
import io
import traceback
from fastapi import FastAPI, Request, Response, File, UploadFile, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import json
import time
import soundfile as sf
from typing import Dict, List, Optional, Union
import uuid

from loguru import logger
logger.add("logs/api_server_v2.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

from indextts.infer_vllm_v2 import IndexTTS2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(CURRENT_DIR, "assets")
SPEAKER_JSON_PATH = os.path.join(ASSETS_DIR, "speaker.json")
tts = None
speaker_dict: Dict[str, List[str]] = {}

def ensure_assets_dir():
    os.makedirs(ASSETS_DIR, exist_ok=True)

def load_speaker_data() -> Dict[str, List[str]]:
    ensure_assets_dir()
    if not os.path.exists(SPEAKER_JSON_PATH):
        with open(SPEAKER_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=4, ensure_ascii=False)
    try:
        with open(SPEAKER_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if isinstance(v, list)}
    except Exception:
        pass
    return {}

def save_speaker_data():
    ensure_assets_dir()
    with open(SPEAKER_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(speaker_dict, f, indent=4, ensure_ascii=False)

def resolve_asset_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    normalized = os.path.normpath(os.path.join(CURRENT_DIR, path))
    return normalized

def get_speaker_audio_path(voice: str) -> Optional[str]:
    paths = speaker_dict.get(voice)
    if not paths:
        return None
    for candidate in paths:
        resolved = resolve_asset_path(candidate)
        if os.path.exists(resolved):
            return resolved
    # fallback to first even if file currently missing
    return resolve_asset_path(paths[0])


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS2(
        model_dir=args.model_dir,
        is_fp16=args.is_fp16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        qwenemo_gpu_memory_utilization=args.qwenemo_gpu_memory_utilization,
    )
    speaker_dict.clear()
    speaker_dict.update(load_speaker_data())
    yield


app = FastAPI(lifespan=lifespan)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if tts is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "TTS model not initialized"
            }
        )
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "Service is running",
            "timestamp": time.time()
        }
    )


@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    try:
        data = await request.json()
        emo_control_method = data.get("emo_control_method", 0)
        text = data["text"]
        spk_audio_path = data["spk_audio_path"]
        emo_ref_path = data.get("emo_ref_path", None)
        emo_weight = data.get("emo_weight", 1.0)
        emo_vec = data.get("emo_vec", [0] * 8)
        emo_text = data.get("emo_text", None)
        emo_random = data.get("emo_random", False)
        max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 120)

        global tts
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
        if emo_control_method == 1:
            emo_weight = emo_weight
        if emo_control_method == 2:
            vec = emo_vec
            vec_sum = sum(vec)
            if vec_sum > 1.5:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "error": "情感向量之和不能超过1.5，请调整后重试。"
                    }
                )
        else:
            vec = None

        # logger.info(f"Emo control mode:{emo_control_method}, vec:{vec}")
        sr, wav = await tts.infer(spk_audio_prompt=spk_audio_path, text=text,
                        output_path=None,
                        emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                        emo_vector=vec,
                        use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                        max_text_tokens_per_sentence=int(max_text_tokens_per_sentence))
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


@app.get("/audio/voices")
async def tts_voices():
    """Return the available speaker list."""
    return speaker_dict


@app.post("/audio/speech", responses={
    200: {"content": {"application/octet-stream": {}}},
    404: {"content": {"application/json": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_openai(request: Request):
    """OpenAI compatible speech endpoint."""
    try:
        data = await request.json()
        text = data["input"]
        voice = data["voice"]

        spk_audio_path = get_speaker_audio_path(voice)
        if spk_audio_path is None:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "error": f"voice `{voice}` not found, please upload or register first."
                }
            )

        global tts
        sr, wav = await tts.infer(
            spk_audio_prompt=spk_audio_path,
            text=text,
            output_path=None,
        )

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format="WAV")
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as ex:
        tb_str = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


@app.post("/audio/upload", responses={
    200: {"content": {"application/json": {}}},
    400: {"content": {"application/json": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_upload_voice(speaker: str = Form(...), file: UploadFile = File(...)):
    """Upload a WAV file and bind it to a speaker."""
    if file.content_type not in {"audio/wav", "audio/x-wav"}:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": "only WAV files are supported"}
        )

    try:
        ensure_assets_dir()
        filename = f"{speaker}_{uuid.uuid4().hex[:8]}.wav"
        target_path = os.path.join(ASSETS_DIR, filename)
        file_data = await file.read()
        with open(target_path, "wb") as out_f:
            out_f.write(file_data)

        rel_path = os.path.relpath(target_path, CURRENT_DIR).replace(os.sep, "/")
        speaker_paths = speaker_dict.setdefault(speaker, [])
        if rel_path not in speaker_paths:
            speaker_paths.append(rel_path)
            save_speaker_data()

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "speaker": speaker,
                "audio_paths": speaker_paths
            }
        )

    except Exception as ex:
        tb_str = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(tb_str)}
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model_dir", type=str, default="checkpoints/IndexTTS-2-vLLM", help="Model checkpoints directory")
    parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    parser.add_argument("--qwenemo_gpu_memory_utilization", type=float, default=0.10)
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
    args = parser.parse_args()
    
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    uvicorn.run(app=app, host=args.host, port=args.port)