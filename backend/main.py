import os
import io
import json
import psutil
import traceback
import gc
import torch
from pathlib import Path
from typing import List, Union, Optional, Dict
import base64
import time
from PIL import Image, ImageDraw, ImageFont

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
SERVICE_ACCOUNT_FILE = BASE_DIR / "service_account.json"
LOCAL_MODELS_DIR = BASE_DIR.parent / "downloaded_models"
HISTORY_FILE = BASE_DIR / "drive_history.json"
CONFIG_FILE = BASE_DIR / "active_model_config.json"

# --- GPU Monitoring ---
try:
    import GPUtil
    gpu_available = True
    print("INFO:     GPUtil found. GPU monitoring enabled.")
except ImportError:
    GPUtil = None
    gpu_available = False
    print("INFO:     GPUtil not found. CPU inference enabled.")
except Exception as e:
    GPUtil = None
    gpu_available = False
    print(f"WARNING:  Error importing GPUtil: {e}")

# --- Google Drive API Setup ---
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    google_drive_available = True
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    DRIVE_FOLDER_ID = "1bhx_znFbDTdIwtt9IsRv5OX0g8Ck-xEn"
except ImportError:
    print("WARNING:  Google Drive libraries not found. Drive sync disabled.")
    google_drive_available = False
    build = None

# --- Application Initialization ---
app = FastAPI(title="Shelf Detection API", version="1.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State ---
_active_model = {
    "name": None,
    "model": None,
    "class_names": []
}

# --- Helper Functions ---

def calculate_crop_parameters(width: int, height: int, zoom_level: float):
    """
    Calculates crop geometry based on zoom level.
    Returns (left, top, right, bottom) and crop_size.
    """
    min_dim = min(width, height)
    safe_zoom = max(1.0, zoom_level)
    crop_size = int(min_dim / safe_zoom)
    crop_size = max(1, min(min_dim, crop_size))

    center_x, center_y = width / 2, height / 2
    left = int(center_x - crop_size / 2)
    top = int(center_y - crop_size / 2)
    right = int(center_x + crop_size / 2)
    bottom = int(center_y + crop_size / 2)
    
    return (left, top, right, bottom), crop_size

def process_input_image(image_bytes: bytes, zoom_level: float = 1.0) -> bytes:
    """
    Crops the image to a 1:1 square based on the zoom level and returns JPEG bytes.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = img.size
        
        box_coords, _ = calculate_crop_parameters(width, height, zoom_level)
        img_to_save = img.crop(box_coords)

        output_buffer = io.BytesIO()
        img_to_save.save(output_buffer, format="JPEG", quality=90)
        return output_buffer.getvalue()

    except Exception as e:
        print(f"ERROR:    Image processing failed: {e}")
        traceback.print_exc()
        raise ValueError(f"Image processing failed: {e}")

def get_drive_service():
    """Returns an authenticated Google Drive service object."""
    if not google_drive_available or not build:
        raise ConnectionError("Google Drive unavailable.")
    if not SERVICE_ACCOUNT_FILE.exists():
        raise FileNotFoundError(f"Service account file missing at {SERVICE_ACCOUNT_FILE}")
    
    creds = service_account.Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_FILE), scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def list_drive_folder(service, folder_id: str) -> List[dict]:
    """Lists all files in a specific Drive folder."""
    if not service: return []
    query = f"'{folder_id}' in parents and trashed=false"
    items = []
    page_token = None
    
    while True:
        try:
             resp = service.files().list(
                q=query, fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token, pageSize=1000,
                supportsAllDrives=True, includeItemsFromAllDrives=True,
             ).execute()
             items.extend(resp.get("files", []))
             page_token = resp.get("nextPageToken")
             if not page_token: break
        except Exception as e:
             print(f"ERROR:    Failed listing folder '{folder_id}': {e}")
             raise
    return items

def download_drive_file(service, file_id: str, dest_path: Path):
    """Downloads a single file from Google Drive."""
    if not service: return
    request = service.files().get_media(fileId=file_id)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    
    try:
        with tmp_path.open("wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        tmp_path.replace(dest_path)
    except Exception as e:
         if tmp_path.exists():
             tmp_path.unlink()
         raise

def download_drive_folder_recursive(service, folder_id: str, local_path: Path):
    """Recursively downloads a Google Drive folder structure."""
    if not service: return
    local_path.mkdir(parents=True, exist_ok=True)
    try:
        children = list_drive_folder(service, folder_id)
    except Exception as e:
        print(f"WARNING:  Could not list folder {folder_id}: {e}")
        return

    for item in children:
        name = item["name"]
        item_id = item["id"]
        mime = item.get("mimeType", "")
        dest = local_path / name

        try:
            if mime == "application/vnd.google-apps.folder":
                download_drive_folder_recursive(service, item_id, dest)
            elif not dest.exists():
                download_drive_file(service, item_id, dest)
        except Exception as e:
            print(f"WARNING:  Skipping '{name}': {e}")

# --- API Endpoints ---

@app.get("/api/system-usage")
def get_system_usage():
    """Returns current CPU, Memory, and GPU usage stats."""
    cpu, mem, gpu, vram = None, None, None, None
    try: cpu = psutil.cpu_percent(interval=0.1)
    except Exception: pass
    try: mem = psutil.virtual_memory().percent
    except Exception: pass

    if gpu_available and GPUtil is not None:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info = gpus[0]
                gpu = round(gpu_info.load * 100, 1)
                vram = round(gpu_info.memoryUtil * 100, 1)
        except Exception: pass

    return {"cpu": cpu, "memory": mem, "gpu": gpu, "vram": vram}

@app.get("/api/history")
def get_history():
    """Reads the history JSON file."""
    if not HISTORY_FILE.exists(): return []
    try:
        if HISTORY_FILE.stat().st_size == 0: return []
        with HISTORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
         return []
    except Exception as e:
        print(f"ERROR:    Reading history file failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read history: {e}")

@app.post("/api/history")
def save_history(sessions: List[str] = Body(...)):
    """Overwrites the history JSON file with new data."""
    if not isinstance(sessions, list):
        raise HTTPException(status_code=400, detail="Invalid input: Expected list.")
    try:
        with HISTORY_FILE.open("w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2, ensure_ascii=False)
        return {"status": "saved", "count": len(sessions)}
    except Exception as e:
        print(f"ERROR:    Saving history failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save history: {e}")

@app.delete("/api/history/{index}")
def delete_history(index: int):
    """Deletes a specific history entry by index."""
    if not HISTORY_FILE.exists():
        raise HTTPException(status_code=404, detail="No history file.")
    try:
        data = []
        if HISTORY_FILE.stat().st_size > 0:
            with HISTORY_FILE.open("r", encoding="utf-8") as f:
                try: data = json.load(f)
                except json.JSONDecodeError: data = []

        if not isinstance(data, list) or not (0 <= index < len(data)):
            raise HTTPException(status_code=400, detail="Invalid index.")

        data.pop(index)
        with HISTORY_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return {"status": "deleted", "remaining": len(data)}
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Failed to delete item: {e}")

@app.get("/api/history/download")
def download_history():
    """Downloads the history file."""
    if not HISTORY_FILE.exists():
        raise HTTPException(status_code=404, detail="History file missing.")
    return FileResponse(str(HISTORY_FILE), media_type="application/json", filename=HISTORY_FILE.name)

# --- Model Management ---

def _load_model_internal(model_name: str):
    """
    Loads the specified model into memory, handling VRAM cleanup.
    """
    model_folder_path = LOCAL_MODELS_DIR / model_name 
    model_pth_path = model_folder_path / "best_model.pth"

    if not model_folder_path.is_dir():
         raise FileNotFoundError(f"Model folder not found: '{model_folder_path}'")
    if not model_pth_path.exists():
        raise FileNotFoundError(f"Model file missing in '{model_folder_path}'")
    
    try:
        ann_file_found = any(f.endswith('.json') and 'instances' in f for f in os.listdir(model_folder_path))
        if not ann_file_found:
             raise FileNotFoundError(f"Annotation file missing in '{model_folder_path}'")
    except OSError as e:
         raise OSError(f"Cannot access model folder: {e}")

    if _active_model.get("model") is not None and _active_model.get("name") != model_name:
        print(f"INFO:     Switching model. Cleaning up '{_active_model.get('name')}'...")
        try:
            model_to_delete = _active_model.pop("model", None)
            if model_to_delete is not None:
                del model_to_delete
            _active_model["name"] = None
            _active_model["class_names"] = []
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            print(f"WARNING:  Error during VRAM cleanup: {e}")
    elif _active_model.get("name") == model_name:
         print(f"INFO:     Model '{model_name}' is already loaded.")
         return

    try:
        from .inference import load_model
        print(f"INFO:     Loading model '{model_name}'...")
        model, class_names = load_model(str(model_folder_path.absolute()))

        _active_model["name"] = model_name
        _active_model["model"] = model
        _active_model["class_names"] = class_names
        print(f"INFO:     Model loaded with {len(class_names)} classes.")

        try:
            with CONFIG_FILE.open("w", encoding="utf-8") as f:
                json.dump({"active_model_name": model_name}, f, indent=2)
        except Exception:
            pass

    except Exception as e:
        _active_model["name"] = None
        _active_model["model"] = None
        _active_model["class_names"] = []
        traceback.print_exc()
        raise RuntimeError(f"Failed loading model '{model_name}': {e}")

@app.get("/api/models")
def list_models():
    """Lists available models locally and optionally syncs with Drive."""
    LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    service = None
    
    if google_drive_available:
        try:
            service = get_drive_service()
        except Exception as e:
            print(f"WARNING:  Drive initialization failed: {e}")

    if service:
        print("INFO:     Checking Google Drive for updates...")
        try:
            drive_items = list_drive_folder(service, DRIVE_FOLDER_ID)
            drive_folders = {f["name"]: f["id"] for f in drive_items if f.get("mimeType") == "application/vnd.google-apps.folder"}
            
            for folder_name, folder_id in drive_folders.items():
                local_path = LOCAL_MODELS_DIR / folder_name
                if not local_path.is_dir() or local_path.exists():
                     try:
                         download_drive_folder_recursive(service, folder_id, local_path)
                     except Exception as e:
                         print(f"ERROR:    Sync failed for '{folder_name}': {e}")
        except Exception as e:
            print(f"WARNING:  Drive sync error: {e}")

    models = []
    local_model_names = set()
    
    try:
        for iteration_path in sorted(LOCAL_MODELS_DIR.iterdir()):
            if not iteration_path.is_dir(): continue

            iteration_name = iteration_path.name
            for item_path in sorted(iteration_path.iterdir()):
                if not item_path.is_dir(): continue

                item_name = f"{iteration_name}/{item_path.name}"
                local_model_names.add(item_name)
                
                entry = {"name": item_name, "specs": [], "is_active": False}
                pth_path = item_path / "best_model.pth"
                log_path = item_path / "training_log.json"
                
                try:
                    ann_file_found = any(f.name.endswith('.json') and 'instances' in f.name for f in item_path.iterdir() if f.is_file())
                except OSError:
                    ann_file_found = False

                if not pth_path.exists(): entry["specs"].append({"label": "Status", "value": "Missing .pth"})
                if not ann_file_found: entry["specs"].append({"label": "Status", "value": "Missing JSON"})

                if log_path.exists():
                    try:
                        with log_path.open("r", encoding="utf-8") as f:
                            log_content = json.load(f)
                        epochs_data = log_content.get("epochs", [])
                        if epochs_data:
                            last_epoch = epochs_data[-1]
                            mAP5095 = last_epoch.get('mAP_0.5:0.95')
                            iou = last_epoch.get('mean_iou')
                            loss = last_epoch.get('avg_train_loss')
                            
                            entry["specs"].extend([
                                {"label": "mAP", "value": f"{mAP5095:.3f}" if isinstance(mAP5095, (int, float)) else "N/A"},
                                {"label": "IoU", "value": f"{iou:.3f}" if isinstance(iou, (int, float)) else "N/A"},
                                {"label": "Loss", "value": f"{loss:.4f}" if isinstance(loss, (int, float)) else "N/A"},
                            ])
                    except Exception:
                        pass
                
                try:
                    size_mb = round(pth_path.stat().st_size / (1024 * 1024), 1)
                    entry["specs"].append({"label": "Size", "value": f"{size_mb} MB"})
                except Exception:
                    pass

                models.append(entry)

    except Exception as e:
         print(f"ERROR:    Failed reading local models: {e}")
         raise HTTPException(status_code=500, detail=f"Error accessing models: {e}")

    active_model_name = _active_model.get("name")
    if active_model_name and active_model_name not in local_model_names:
        _active_model["name"] = None
        _active_model["model"] = None
        _active_model["class_names"] = []
    else:
        for model_entry in models:
            if model_entry["name"] == active_model_name:
                model_entry["is_active"] = True
                break

    return models

@app.get("/api/models/current")
def get_current_model():
    return {"current_model": _active_model.get("name")}

@app.post("/api/models/load")
def load_selected_model_endpoint(model_name: str = Form(...)):
    try:
        _load_model_internal(model_name)
        return {"status": "success", "message": f"Model '{model_name}' loaded."}
    except ValueError as e:
         return {"status": "success", "message": str(e)}
    except FileNotFoundError as e:
         raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

# --- Inference Endpoint ---

@app.post("/api/infer")
async def infer(
    file: UploadFile = File(...), 
    nms_method: str = Form("soft"),
    zoom: float = Form(1.0)
):
    """
    Runs inference.
    Updates:
    1. Returns 'color' for each detected item.
    2. Draws label text INSIDE the bounding box.
    3. RETURNS GLOBAL COORDINATES for dynamic frontend scaling.
    """
    if _active_model.get("model") is None:
        raise HTTPException(status_code=400, detail="No model loaded.")

    model = _active_model.get("model")
    expected_class_names = _active_model.get("class_names")
    
    try:
        from .inference import preprocess_image, run_inference
    except Exception:
        raise HTTPException(status_code=500, detail="Server error: Import failed.")

    start_time = time.time()
    try:
        original_contents = await file.read()
        if not original_contents:
             raise HTTPException(status_code=400, detail="Empty file.")

        # 1. Preprocessing & Zoom
        # We need to manually process here to get the offsets to map coords back later
        full_pil_image_orig = Image.open(io.BytesIO(original_contents)).convert("RGB")
        w_orig_full, h_orig_full = full_pil_image_orig.size
        
        # Calculate Crop Parameters based on Zoom
        (offset_left, offset_top, _, _), _ = calculate_crop_parameters(w_orig_full, h_orig_full, zoom)
        
        # Create the crop byte stream for the model
        try:
            full_image_bytes = process_input_image(original_contents, zoom_level=zoom)
        except ValueError as e:
             raise HTTPException(status_code=500, detail=str(e))

        # 2. Prepare Tensor
        full_pil_image, tensor, (processed_w, processed_h) = preprocess_image(io.BytesIO(full_image_bytes))

        # 3. Inference
        raw_detections = run_inference(model, expected_class_names, tensor, full_pil_image, nms_method=nms_method)

        # 4. Post-processing
        detected_labels = set(d["label"] for d in raw_detections)
        expected_labels_set = set(expected_class_names)
        missing_items = sorted(list(expected_labels_set - detected_labels))

        scaled_detected_items = []
        w_crop, h_crop = full_pil_image.size
        
        # Consistent color palette (Hex codes)
        CLASS_COLORS = [
            "#32CD32", "#00FFFF", "#FF0000", "#FFFF00", "#FF00FF", "#FFA500",
            "#1E90FF", "#FF1493", "#00FF7F", "#00BFFF", "#FF69B4", "#FFD700"
        ]
        class_names_list = _active_model.get("class_names", [])

        # Create drawing context for the static result image
        draw = ImageDraw.Draw(full_pil_image)

        for det in raw_detections:
            box_model = det["box"]
            
            # Local coordinates (relative to the crop)
            x1_local = max(0.0, min(float(w_crop), box_model[0]))
            y1_local = max(0.0, min(float(h_crop), box_model[1]))
            x2_local = max(x1_local, min(float(w_crop), box_model[2]))
            y2_local = max(y1_local, min(float(h_crop), box_model[3]))

            # Global coordinates (relative to the original full image)
            # Used by frontend for dynamic zoom box rendering
            x1_global = x1_local + offset_left
            y1_global = y1_local + offset_top
            x2_global = x2_local + offset_left
            y2_global = y2_local + offset_top

            if (x2_local - x1_local) > 1.0 and (y2_local - y1_local) > 1.0:
                 label = det["label"]
                 try:
                    label_index = class_names_list.index(label)
                    color = CLASS_COLORS[label_index % len(CLASS_COLORS)]
                 except (ValueError, AttributeError):
                    color = "#32CD32"

                 # Add to result list with GLOBAL coordinates
                 scaled_detected_items.append({
                    "label": label,
                    "confidence": f"{det['score']:.3f}",
                    "box": [float(x1_global), float(y1_global), float(x2_global), float(y2_global)],
                    "color": color
                 })

                 # --- Draw on the static image (using LOCAL coordinates) ---
                 draw.rectangle([x1_local, y1_local, x2_local, y2_local], outline=color, width=3)
                 
                 # Draw Text inside box
                 text_to_draw = f"{label}"
                 text_x = x1_local + 4
                 text_y = y1_local + 4
                 
                 # Boundary checks for text
                 if text_x + 50 > w_crop: text_x = w_crop - 55
                 if text_y + 15 > h_crop: text_y = h_crop - 20

                 char_width = 7
                 text_width = len(text_to_draw) * char_width
                 text_height = 14
                 
                 draw.rectangle(
                    [text_x - 2, text_y - 2, text_x + text_width, text_y + text_height],
                    fill=color
                 )
                 draw.text((text_x, text_y), text_to_draw, fill="black")

        del draw

        # Generate base64 preview (still useful for initial load)
        preview_buffer = io.BytesIO()
        full_pil_image.save(preview_buffer, format="JPEG", quality=90)
        processed_image_base64 = base64.b64encode(preview_buffer.getvalue()).decode("utf-8")

        processing_time = time.time() - start_time

        return JSONResponse(content={
            "processed_image": f"data:image/jpeg;base64,{processed_image_base64}" if processed_image_base64 else None,
            "detected_items": scaled_detected_items,
            "missing_items": missing_items,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            "processing_time_ms": int(processing_time * 1000)
        })

    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

# --- Startup ---

def startup_load_model():
    """Attempts to auto-load the previously active model on startup."""
    if not CONFIG_FILE.exists(): return
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            config = json.load(f)
        model_name = config.get("active_model_name")
        if model_name:
            _load_model_internal(model_name)
    except Exception as e:
        print(f"ERROR:    Startup load failed: {e}")

if __name__ == "__main__":
    import uvicorn
    LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    startup_load_model()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")