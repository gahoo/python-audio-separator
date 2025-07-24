"""
Audio Separator API - Local FastAPI Deployment
A FastAPI service for separating vocals from instrumental tracks using audio-separator.
This version is decoupled from Modal for local or standard container deployment.
"""

# Standard library imports
import logging
import os
import shutil
import traceback
import uuid
import json
import typing
from typing import Optional

# Third-party imports
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response as StarletteResponse, PlainTextResponse
import filetype
import uvicorn

# Local imports
from audio_separator.separator import Separator

# --- Local Replacements for Modal-specific features ---

# In-memory dictionary to track job status.
# NOTE: This is not persistent. Job statuses will be lost on server restart.
job_status_dict = {}

# Define local paths for storage, replacing Modal Volumes
STORAGE_PATH = os.getenv("STORAGE_PATH", "./storage")
MODELS_PATH = os.getenv("AUDIO_SEPARATOR_MODEL_DIR", "./models")
UPLOADS_PATH = os.path.join(STORAGE_PATH, "uploads")
OUTPUTS_PATH = os.path.join(STORAGE_PATH, "outputs")

# Ensure local storage directories exist
os.makedirs(UPLOADS_PATH, exist_ok=True)
os.makedirs(OUTPUTS_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# --- Constants and Versioning ---
try:
    from importlib.metadata import version
    AUDIO_SEPARATOR_VERSION = version("audio-separator")
except Exception:
    AUDIO_SEPARATOR_VERSION = "unknown"


class PrettyJSONResponse(StarletteResponse):
    """Custom JSON response class for pretty-printing JSON"""
    media_type = "application/json"

    def render(self, content: typing.Any) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False, indent=4, separators=(", ", ": ")).encode("utf-8")


# --- Core Application Logic (formerly Modal Functions) ---

def separate_audio_function(
    audio_data: bytes,
    filename: str,
    models: Optional[list] = None,
    task_id: Optional[str] = None,
    # Separator parameters
    output_format: str = "flac",
    output_bitrate: Optional[str] = None,
    normalization_threshold: float = 0.9,
    amplification_threshold: float = 0.0,
    output_single_stem: Optional[str] = None,
    invert_using_spec: bool = False,
    sample_rate: int = 44100,
    use_soundfile: bool = False,
    use_autocast: bool = False,
    custom_output_names: Optional[dict] = None,
    # MDX parameters
    mdx_segment_size: int = 256,
    mdx_overlap: float = 0.25,
    mdx_batch_size: int = 1,
    mdx_hop_length: int = 1024,
    mdx_enable_denoise: bool = False,
    # VR parameters
    vr_batch_size: int = 1,
    vr_window_size: int = 512,
    vr_aggression: int = 5,
    vr_enable_tta: bool = False,
    vr_high_end_process: bool = False,
    vr_enable_post_process: bool = False,
    vr_post_process_threshold: float = 0.2,
    # Demucs parameters
    demucs_segment_size: str = "Default",
    demucs_shifts: int = 2,
    demucs_overlap: float = 0.25,
    demucs_segments_enabled: bool = True,
    # MDXC parameters
    mdxc_segment_size: int = 256,
    mdxc_override_model_segment_size: bool = False,
    mdxc_overlap: int = 8,
    mdxc_batch_size: int = 1,
    mdxc_pitch_shift: int = 0,
) -> dict:
    """
    Separate audio into stems using one or more models.
    This function is now called as a background task.
    """
    if task_id is None:
        task_id = str(uuid.uuid4())

    if models is None or len(models) == 0:
        models = [None]

    all_output_files = []
    models_used = []
    current_model_index = 0
    total_models = len(models)

    def update_job_status(status: str, progress: int = 0, error: str = None, files: list = None):
        """Update job status in the in-memory dictionary"""
        status_data = {
            "task_id": task_id,
            "status": status,
            "progress": progress,
            "original_filename": filename,
            "models_used": models_used,
            "total_models": total_models,
            "current_model_index": current_model_index,
            "files": files or [],
        }
        if error:
            status_data["error"] = error
        job_status_dict[task_id] = status_data

    output_dir = os.path.join(OUTPUTS_PATH, task_id)
    input_file_path = os.path.join(output_dir, filename)

    try:
        update_job_status("processing", 5)
        os.makedirs(output_dir, exist_ok=True)

        with open(input_file_path, "wb") as f:
            f.write(audio_data)

        update_job_status("processing", 10)

        for model_index, model_name in enumerate(models):
            current_model_index = model_index
            base_progress = 10 + (model_index * 80 // total_models)
            model_progress_range = 80 // total_models

            print(f"Processing model {model_index + 1}/{total_models}: {model_name or 'default'}")
            update_job_status("processing", base_progress + (model_progress_range // 4))

            separator = Separator(
                log_level=logging.INFO,
                model_file_dir=MODELS_PATH,
                output_dir=output_dir,
                output_format=output_format,
                # ... (rest of the parameters are passed directly)
                normalization_threshold=normalization_threshold,
                amplification_threshold=amplification_threshold,
                output_single_stem=output_single_stem,
                invert_using_spec=invert_using_spec,
                sample_rate=sample_rate,
                use_soundfile=use_soundfile,
                use_autocast=use_autocast,
                mdx_params={"hop_length": mdx_hop_length, "segment_size": mdx_segment_size, "overlap": mdx_overlap, "batch_size": mdx_batch_size, "enable_denoise": mdx_enable_denoise},
                vr_params={"batch_size": vr_batch_size, "window_size": vr_window_size, "aggression": vr_aggression, "enable_tta": vr_enable_tta, "enable_post_process": vr_enable_post_process, "post_process_threshold": vr_post_process_threshold, "high_end_process": vr_high_end_process},
                demucs_params={"segment_size": demucs_segment_size, "shifts": demucs_shifts, "overlap": demucs_overlap, "segments_enabled": demucs_segments_enabled},
                mdxc_params={"segment_size": mdxc_segment_size, "batch_size": mdxc_batch_size, "overlap": mdxc_overlap, "override_model_segment_size": mdxc_override_model_segment_size, "pitch_shift": mdxc_pitch_shift},
            )

            update_job_status("processing", base_progress + (model_progress_range // 2))
            separator.load_model(model_name if model_name else "default")
            models_used.append(model_name or "default")

            update_job_status("processing", base_progress + (3 * model_progress_range // 4))
            output_files = separator.separate(input_file_path)
            
            if not output_files:
                raise RuntimeError(f"Separation with model {models_used[-1]} produced no output files")

            all_output_files.extend([os.path.basename(f) for f in output_files])

        update_job_status("completed", 100, files=all_output_files)
        return {"task_id": task_id, "status": "completed", "files": all_output_files, "models_used": models_used, "original_filename": filename}

    except Exception as e:
        print(f"Unexpected error during separation: {str(e)}")
        traceback.print_exc()
        update_job_status("error", 0, error=str(e))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        return {"task_id": task_id, "status": "error", "error": str(e), "models_used": models_used, "original_filename": filename}


def get_job_status_function(task_id: str) -> dict:
    if task_id in job_status_dict:
        return job_status_dict[task_id]
    else:
        return {"task_id": task_id, "status": "not_found", "progress": 0, "error": "Job not found"}


def get_file_function(task_id: str, filename: str) -> bytes:
    file_path = os.path.join(OUTPUTS_PATH, task_id, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {filename}")
    with open(file_path, "rb") as f:
        return f.read()


def list_available_models() -> dict:
    separator = Separator(info_only=True, model_file_dir=MODELS_PATH)
    return separator.list_supported_model_files()


def get_simplified_models(filter_sort_by: str = None) -> dict:
    separator = Separator(info_only=True, model_file_dir=MODELS_PATH)
    return separator.get_simplified_model_list(filter_sort_by=filter_sort_by)


# --- FastAPI App Definition ---

app = FastAPI(title="Audio Separator API", description="Separate vocals from instrumental tracks using AI", version=AUDIO_SEPARATOR_VERSION)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.post("/separate")
async def separate_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file to separate"),
    model: Optional[str] = Form(None, description="Single model to use for separation"),
    models: Optional[str] = Form(None, description='JSON list of models to use, e.g. ["model1.ckpt"]'),
    # All other parameters from the original script...
    output_format: str = Form("flac", description="Output format for separated files"),
    output_bitrate: Optional[str] = Form(None, description="Output bitrate for separated files"),
    normalization_threshold: float = Form(0.9, description="Max peak amplitude to normalize audio to"),
    amplification_threshold: float = Form(0.0, description="Min peak amplitude to amplify audio to"),
    output_single_stem: Optional[str] = Form(None, description="Output only single stem (e.g. Vocals, Instrumental)"),
    invert_using_spec: bool = Form(False, description="Invert secondary stem using spectrogram"),
    sample_rate: int = Form(44100, description="Sample rate of output audio"),
    use_soundfile: bool = Form(False, description="Use soundfile for output writing"),
    use_autocast: bool = Form(False, description="Use PyTorch autocast for faster inference"),
    custom_output_names: Optional[str] = Form(None, description="JSON dict of custom output names"),
    mdx_segment_size: int = Form(256, description="MDX segment size"),
    mdx_overlap: float = Form(0.25, description="MDX overlap"),
    mdx_batch_size: int = Form(1, description="MDX batch size"),
    mdx_hop_length: int = Form(1024, description="MDX hop length"),
    mdx_enable_denoise: bool = Form(False, description="Enable MDX denoising"),
    vr_batch_size: int = Form(1, description="VR batch size"),
    vr_window_size: int = Form(512, description="VR window size"),
    vr_aggression: int = Form(5, description="VR aggression"),
    vr_enable_tta: bool = Form(False, description="Enable VR Test-Time-Augmentation"),
    vr_high_end_process: bool = Form(False, description="Enable VR high end processing"),
    vr_enable_post_process: bool = Form(False, description="Enable VR post processing"),
    vr_post_process_threshold: float = Form(0.2, description="VR post process threshold"),
    demucs_segment_size: str = Form("Default", description="Demucs segment size"),
    demucs_shifts: int = Form(2, description="Demucs shifts"),
    demucs_overlap: float = Form(0.25, description="Demucs overlap"),
    demucs_segments_enabled: bool = Form(True, description="Enable Demucs segments"),
    mdxc_segment_size: int = Form(256, description="MDXC segment size"),
    mdxc_override_model_segment_size: bool = Form(False, description="Override MDXC model segment size"),
    mdxc_overlap: int = Form(8, description="MDXC overlap"),
    mdxc_batch_size: int = Form(1, description="MDXC batch size"),
    mdxc_pitch_shift: int = Form(0, description="MDXC pitch shift"),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    models_list = None
    if models:
        try:
            models_list = json.loads(models)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in models parameter")
    elif model:
        models_list = [model]

    custom_output_names_dict = None
    if custom_output_names:
        try:
            custom_output_names_dict = json.loads(custom_output_names)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in custom_output_names parameter")

    audio_data = await file.read()
    task_id = str(uuid.uuid4())

    # Set initial status
    job_status_dict[task_id] = {"task_id": task_id, "status": "submitted", "progress": 0, "original_filename": file.filename}

    # Use FastAPI's BackgroundTasks to run the job
    background_tasks.add_task(
        separate_audio_function,
        audio_data,
        file.filename,
        models_list,
        task_id,
        output_format,
        output_bitrate,
        normalization_threshold,
        amplification_threshold,
        output_single_stem,
        invert_using_spec,
        sample_rate,
        use_soundfile,
        use_autocast,
        custom_output_names_dict,
        mdx_segment_size, mdx_overlap, mdx_batch_size, mdx_hop_length, mdx_enable_denoise,
        vr_batch_size, vr_window_size, vr_aggression, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_post_process_threshold,
        demucs_segment_size, demucs_shifts, demucs_overlap, demucs_segments_enabled,
        mdxc_segment_size, mdxc_override_model_segment_size, mdxc_overlap, mdxc_batch_size, mdxc_pitch_shift
    )

    return {"task_id": task_id, "status": "submitted"}


@app.get("/status/{task_id}")
async def get_status(task_id: str) -> dict:
    return get_job_status_function(task_id)


@app.get("/download/{task_id}/{filename}")
async def download_file(task_id: str, filename: str) -> Response:
    try:
        file_data = get_file_function(task_id, filename)
        content_type = filetype.guess_mime(file_data) or "application/octet-stream"
        return Response(content=file_data, media_type=content_type, headers={"Content-Disposition": f"attachment; filename={filename}"})
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")


@app.get("/models-json")
async def get_models_json() -> PrettyJSONResponse:
    return PrettyJSONResponse(content=list_available_models())


@app.get("/models")
async def get_models_list(filter_sort_by: str = None) -> PlainTextResponse:
    models = get_simplified_models(filter_sort_by=filter_sort_by)
    if not models:
        return PlainTextResponse("No models found")
    # Formatting logic from original script...
    filename_width = max(len("Model Filename"), max(len(filename) for filename in models.keys()))
    arch_width = max(len("Arch"), max(len(info["Type"]) for info in models.values()))
    stems_width = max(len("Output Stems (SDR)"), max(len(", ".join(info["Stems"])) for info in models.values()))
    name_width = max(len("Friendly Name"), max(len(info["Name"]) for info in models.values()))
    total_width = filename_width + arch_width + stems_width + name_width + 15
    output_lines = ["-" * total_width, f"{'Model Filename':<{filename_width}}  {'Arch':<{arch_width}}  {'Output Stems (SDR)':<{stems_width}}  {'Friendly Name'}", "-" * total_width]
    for filename, info in models.items():
        stems = ", ".join(info["Stems"])
        output_lines.append(f"{filename:<{filename_width}}  {info['Type']:<{arch_width}}  {stems:<{stems_width}}  {info['Name']}")
    return PlainTextResponse("\n".join(output_lines))


@app.get("/health")
async def health_check() -> dict:
    return {"status": "healthy", "service": "audio-separator-api", "version": AUDIO_SEPARATOR_VERSION}


@app.get("/")
async def root() -> dict:
    return {"message": "Audio Separator API (Local)", "version": AUDIO_SEPARATOR_VERSION}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
