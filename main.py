# backend/main.py
"""
AI Image Stylizer (Cartoon-focused, OpenCV heavy with neural hooks)

Features:
- Multiple modes: cartoongan (neural, if available), faststyle (neural), opencv_cartoon (fast fallback),
  sketch, stylize (opencv stylization), cel_cartoon (improved OpenCV cartoon / color-quant + posterize).
- Optional super-resolution step using Real-ESRGAN (if installed); safe no-op if not installed.
- Extra parameters accepted via form: preset, strength, posterize_levels, palette_colors, sr (on/off)
- Robust errors and headers: X-Processing-Time, X-Used-Fallback, X-Used-SR
- Place model files in backend/models/ if you want neural paths to work.
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import io, os, time
import numpy as np
import cv2
import math

# Optional ML imports (lazy)
try:
    import torch
    from torchvision import transforms
except Exception:
    torch = None

# Optional Real-ESRGAN (super resolution). If not installed it's ignored.
REAL_ESRGAN_AVAILABLE = False
try:
    # Attempt to import known Real-ESRGAN wrappers here if you installed one.
    # Example package names vary; adapt to your env. This is a placeholder detection.
    import realesrgan  # noqa: F401
    REAL_ESRGAN_AVAILABLE = True
except Exception:
    REAL_ESRGAN_AVAILABLE = False

app = FastAPI(title="AI Image Stylizer (Neural Cartoon + Fast Style + OpenCV)")

# Allow common dev origins; change to production origins as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # during dev; lock this down for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# model paths (place weights in backend/models/)
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
CARTOONGAN_PATH = os.path.join(MODEL_DIR, "cartoongan.pth")
FASTSTYLE_PATH = os.path.join(MODEL_DIR, "faststyle.pth")

# Lazy model holders
_cartoon_model = None
_faststyle_model = None

# ----------------------
# Helpers: conversions
# ----------------------
def pil_to_cv2(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    arr = np.array(img)
    return arr[:, :, ::-1].copy()

def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    img_rgb = img_bgr[:, :, ::-1]
    return Image.fromarray(img_rgb)

# ----------------------
# Super-resolution wrapper (optional Real-ESRGAN)
# This is a best-effort wrapper; if not installed it's a no-op.
# ----------------------
def apply_superres_if_available(pil_img: Image.Image, scale: int = 2):
    used = False
    if not REAL_ESRGAN_AVAILABLE:
        return pil_img, used
    try:
        # Placeholder: implement according to the Real-ESRGAN wrapper you use.
        # Example pseudo:
        # sr = RealESRGANer(...) ; out = sr.enhance(np.array(pil_img))
        # For now, simply return pil_img to remain safe if no binding present.
        used = False
        return pil_img, used
    except Exception:
        return pil_img, used

# ----------------------
# Color quantization (k-means using cv2.kmeans)
# ----------------------
def color_quantization(img_bgr: np.ndarray, k=16):
    # img_bgr: HxWx3 (uint8)
    Z = img_bgr.reshape((-1,3)).astype(np.float32)
    # criteria, attempts
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    attempts = 2
    flags = cv2.KMEANS_PP_CENTERS
    if k <= 1:
        return img_bgr
    try:
        compactness, labels, centers = cv2.kmeans(Z, k, None, criteria, attempts, flags)
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        quant = res.reshape((img_bgr.shape))
        return quant
    except Exception:
        # fallback: return original
        return img_bgr

# ----------------------
# Posterize using PIL (safer & high-level)
# ----------------------
def posterize_pil(pil_img: Image.Image, bits: int):
    if bits < 1:
        return pil_img
    bits = max(1, min(8, bits))
    return ImageOps.posterize(pil_img, bits)

# ----------------------
# Enhanced OpenCV cartoon pipeline (cel shading + bold outlines)
# ----------------------
def opencv_cel_cartoon(img_bgr: np.ndarray, palette_colors=8, posterize_bits=4, edge_thickness=1, smoothness=2, strength=0.9):
    """
    Produces a cartoon-like cel-shaded image:
    1. Bilateral filtering (edge-preserving blur) repeated
    2. Color quantization (k-means)
    3. Posterize (PIL)
    4. Edge detection (adaptive threshold + Canny fallback)
    5. Combine with edges (bitwise)
    Parameters tuned to give 'cartoon' look rather than 'painting'
    """
    h, w = img_bgr.shape[:2]

    # 1) Smooth while preserving edges
    color = img_bgr.copy()
    for i in range(max(1, smoothness)):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)

    # 2) Color quantization
    quant = color_quantization(color, palette_colors)

    # 3) Convert to PIL and posterize for banded flat regions
    quant_pil = cv2_to_pil(quant)
    try:
        quant_post = posterize_pil(quant_pil, posterize_bits)
    except Exception:
        quant_post = quant_pil
    quant_post_bgr = pil_to_cv2(quant_post)

    # 4) Edges: combine adaptive threshold + Canny to be robust
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray_blur, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY,
                                  blockSize=9,
                                  C=2)
    # Canny edges for finer detail
    canny = cv2.Canny(gray, 100, 200)
    # combine and dilate to form thicker outlines
    combined_edges = cv2.bitwise_or(edges, canny)
    # dilate to control thickness
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + edge_thickness*2, 1 + edge_thickness*2))
    combined_edges = cv2.dilate(combined_edges, kernel)

    # convert edges to 3-channel mask
    edges_colored = cv2.cvtColor(255 - combined_edges, cv2.COLOR_GRAY2BGR)  # invert so lines are black on white
    # combine (bitwise AND will keep color where mask is white)
    cartoon = cv2.bitwise_and(quant_post_bgr, edges_colored)

    # optionally enhance details / contrast a bit
    cartoon = cv2.detailEnhance(cartoon, sigma_s=10, sigma_r=0.15)

    # blend with original based on strength (0..1)
    try:
        blended = cv2.addWeighted(cartoon, strength, img_bgr, 1 - strength, 0)
    except Exception:
        blended = cartoon

    return blended

# ----------------------
# OPENCV fast cartoonizer (older fallback)
# ----------------------
def opencv_cartoon_simple(img_bgr: np.ndarray):
    color = img_bgr.copy()
    for _ in range(2):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  blockSize=9,
                                  C=2)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges_colored)
    cartoon = cv2.detailEnhance(cartoon, sigma_s=10, sigma_r=0.15)
    return cartoon

# ----------------------
# Sketch and stylize (OpenCV)
# ----------------------
def sketch_opencv(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), sigmaX=0, sigmaY=0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    return sketch_bgr

def stylize_opencv(img_bgr: np.ndarray) -> np.ndarray:
    stylized = cv2.stylization(img_bgr, sigma_s=60, sigma_r=0.6)
    return stylized

# ----------------------
# Neural model loaders / appliers (best-effort)
# ----------------------
def load_cartoon_model():
    global _cartoon_model
    if _cartoon_model is not None:
        return _cartoon_model
    if torch is None:
        return None
    if not os.path.exists(CARTOONGAN_PATH):
        return None
    try:
        _cartoon_model = torch.jit.load(CARTOONGAN_PATH, map_location=torch.device("cpu"))
    except Exception:
        _cartoon_model = None
    return _cartoon_model

def apply_cartoon_pytorch(pil_img: Image.Image):
    model = load_cartoon_model()
    if model is None:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    img_t = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t).clamp(0,255).cpu()
    out_img = out.squeeze(0).permute(1,2,0).numpy().astype("uint8")
    return out_img[:, :, ::-1]

def load_faststyle_model():
    global _faststyle_model
    if _faststyle_model is not None:
        return _faststyle_model
    if torch is None:
        return None
    if not os.path.exists(FASTSTYLE_PATH):
        return None
    try:
        _faststyle_model = torch.jit.load(FASTSTYLE_PATH, map_location=torch.device("cpu"))
    except Exception:
        _faststyle_model = None
    return _faststyle_model

def apply_faststyle_pytorch(pil_img: Image.Image):
    model = load_faststyle_model()
    if model is None:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    img_t = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t).clamp(0,255).cpu()
    out_img = out.squeeze(0).permute(1,2,0).numpy().astype("uint8")
    return out_img[:, :, ::-1]

# ----------------------
# Main endpoint
# ----------------------
@app.post("/stylize")
async def stylize_image(
    file: UploadFile = File(...),
    mode: str = Form("cartoongan"),  # cartoongan | faststyle | opencv_cartoon | cel_cartoon | sketch | stylize
    preset: str = Form("default"),   # preset hints: family_guy, ben10, avengers, default
    strength: float = Form(0.92),    # blending between cartoon and original (0..1)
    posterize_levels: int = Form(4), # 1..8
    palette_colors: int = Form(8),   # quantization colors
    edge_thickness: int = Form(1),   # outline thickness
    sr: bool = Form(False)           # enable super-res
):
    start = time.time()
    used_fallback = False
    used_sr = False
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Optional super-res before stylize
        if sr:
            pil_img, used_sr = apply_superres_if_available(pil_img, scale=2)

        img_bgr = pil_to_cv2(pil_img)

        mode = (mode or "cartoongan").lower()
        out_cv = None

        # map preset -> parameter adjustments (simple mapping to achieve visually different results)
        preset = (preset or "default").lower()
        if preset == "family_guy":
            palette_colors = max(4, min(palette_colors, 12))
            posterize_levels = max(3, min(posterize_levels, 5))
            edge_thickness = max(1, min(edge_thickness, 2))
            strength = min(0.95, max(0.7, strength))
        elif preset == "ben10":
            palette_colors = max(6, min(palette_colors, 18))
            posterize_levels = max(2, min(posterize_levels, 5))
            edge_thickness = max(1, min(edge_thickness, 3))
        elif preset == "avengers":
            palette_colors = max(8, min(palette_colors, 24))
            posterize_levels = max(3, min(posterize_levels, 6))
            edge_thickness = max(1, min(edge_thickness, 4))

        if mode == "cartoongan":
            # prefer neural first
            if torch is not None:
                try:
                    out_cv = apply_cartoon_pytorch(pil_img)
                except Exception:
                    out_cv = None
            if out_cv is None:
                used_fallback = True
                out_cv = opencv_cel_cartoon(img_bgr, palette_colors, posterize_levels, edge_thickness, smoothness=2, strength=strength)
        elif mode == "faststyle":
            out_cv = None
            if torch is not None:
                try:
                    out_cv = apply_faststyle_pytorch(pil_img)
                except Exception:
                    out_cv = None
            if out_cv is None:
                used_fallback = True
                out_cv = stylize_opencv(img_bgr)
        elif mode == "opencv_cartoon":
            out_cv = opencv_cartoon_simple(img_bgr)
        elif mode == "cel_cartoon":
            out_cv = opencv_cel_cartoon(img_bgr, palette_colors, posterize_levels, edge_thickness, smoothness=2, strength=strength)
        elif mode == "sketch":
            out_cv = sketch_opencv(img_bgr)
        elif mode == "stylize":
            out_cv = stylize_opencv(img_bgr)
        else:
            return JSONResponse(status_code=400, content={"error": "Unknown mode"})

        # ensure result is correct type
        if isinstance(out_cv, np.ndarray) and out_cv.dtype == np.uint8:
            out_pil = cv2_to_pil(out_cv)
        else:
            out_pil = pil_img

        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")
        buf.seek(0)

        elapsed = time.time() - start
        headers = {
            "Content-Disposition": 'attachment; filename="stylized.png"',
            "X-Processing-Time": str(round(elapsed, 3)),
            "X-Used-Fallback": str(used_fallback),
            "X-Used-SR": str(bool(used_sr))
        }
        return StreamingResponse(buf, media_type="image/png", headers=headers)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})