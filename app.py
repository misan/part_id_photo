import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import cv2
import streamlit as st
from PIL import Image

try:
    from scipy.spatial import cKDTree
except Exception as e:
    cKDTree = None
    _SCIPY_IMPORT_ERROR = e
else:
    _SCIPY_IMPORT_ERROR = None


# ----------------------------
# Config
# ----------------------------
N_RESAMPLE = 256
FD_COEFFS = 20

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
ENV_DATA_DIR = os.environ.get("PART_ID_DATA_DIR", "").strip()

UPLOAD_CACHE_DIR = Path(__file__).parent / "_uploaded_data"
UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Geometry helpers
# ----------------------------
def resample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    if pts is None or len(pts) < 2:
        raise ValueError("Not enough points to resample")
    d = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    if total <= 0:
        raise ValueError("Zero-length polyline")
    t = np.linspace(0, total, n, endpoint=False)
    x = np.interp(t, s, pts[:, 0])
    y = np.interp(t, s, pts[:, 1])
    return np.stack([x, y], axis=1)


def contour_from_mask(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 800:
        return None
    return c.reshape(-1, 2).astype(np.float32)


def normalize_contour(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts.mean(axis=0, keepdims=True)
    d = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    per = float(d.sum())
    return pts if per <= 0 else (pts / per)


def fourier_descriptor(pts: np.ndarray, k: int = FD_COEFFS) -> np.ndarray:
    z = pts[:, 0] + 1j * pts[:, 1]
    z = z - z.mean()
    F = np.fft.fft(z)
    mag = np.abs(F)
    mag[0] = 0.0
    fd = mag[1 : k + 1]
    denom = fd[0] if fd[0] != 0 else (fd.mean() if fd.mean() != 0 else 1.0)
    return (fd / denom).astype(np.float32)


def rotate_pts(pts: np.ndarray, theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return pts @ R.T


def symmetric_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if cKDTree is None:
        raise RuntimeError(f"scipy not available: {_SCIPY_IMPORT_ERROR}")
    ta = cKDTree(a)
    tb = cKDTree(b)
    da, _ = tb.query(a, k=1)
    db, _ = ta.query(b, k=1)
    return float(da.mean() + db.mean())


def best_rotation_chamfer(obs: np.ndarray, templ: np.ndarray) -> float:
    best = 1e9
    for deg in range(0, 360, 10):
        r = rotate_pts(obs, np.deg2rad(deg))
        d = symmetric_chamfer(r, templ)
        if d < best:
            best = d
    return best


# ----------------------------
# Template loading
# ----------------------------
def _candidate_data_dirs():
    cands = []
    if ENV_DATA_DIR:
        cands.append(Path(ENV_DATA_DIR))
    cands.append(DEFAULT_DATA_DIR)
    cands.append(Path.cwd() / "data")
    cands.append(UPLOAD_CACHE_DIR / "data")
    return cands


def _resolve_image_path(rel_img: str, data_dir: Path):
    """
    Resolve an image path from templates.json robustly.
    Supports:
      - "data/templates/xxx.png"
      - "templates/xxx.png"
      - "xxx.png" (inside data/templates/)
      - absolute paths
    """
    if not rel_img:
        return None

    p = Path(rel_img)

    # Absolute path as-is
    if p.is_absolute() and p.exists():
        return p.resolve()

    # Normalize separators
    rel_img = rel_img.replace("\\", "/")
    p = Path(rel_img)

    # Common bases to try (in order)
    bases = [
        Path(__file__).parent,        # folder containing app.py
        data_dir,                     # .../data
        data_dir / "templates",       # .../data/templates
        data_dir.parent,              # folder containing data/
        Path.cwd(),                   # current working directory
    ]

    # Try direct join
    for base in bases:
        cand = (base / p).resolve()
        if cand.exists():
            return cand

    # If rel_img already contains "data/templates/..", try stripping leading "data/"
    parts = list(p.parts)
    if len(parts) >= 2 and parts[0].lower() == "data":
        p2 = Path(*parts[1:])
        for base in bases:
            cand = (base / p2).resolve()
            if cand.exists():
                return cand

    return None


def _load_templates_from(data_dir: Path):
    meta_path = data_dir / "templates.json"
    if not meta_path.exists():
        return [], f"Missing {meta_path}"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        return [], f"Failed to parse {meta_path}: {e}"

    if not isinstance(meta, list):
        return [], f"{meta_path} must be a JSON list of objects"

    templates = []
    missing_images = 0

    for item in meta:
        if not isinstance(item, dict):
            continue
        rel_img = item.get("image")
        img_path = _resolve_image_path(rel_img, data_dir)

        if img_path is None or not img_path.exists():
            missing_images += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            missing_images += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        pts = contour_from_mask(mask)
        if pts is None:
            continue
        try:
            pts = resample_polyline(pts, N_RESAMPLE)
        except Exception:
            continue
        pts = normalize_contour(pts)
        fd = fourier_descriptor(pts)
        templates.append(
            {
                "part_id": item.get("part_id", "UNKNOWN"),
                "image": str(img_path),
                "pts": pts,
                "fd": fd,
            }
        )

    if not templates:
        return [], f"Parsed templates.json but loaded 0 templates. Missing/unreadable images: {missing_images} of {len(meta)}"
    return templates, None


@st.cache_resource
def load_templates_auto():
    last_err = None
    last_dir = ""
    for d in _candidate_data_dirs():
        templates, err = _load_templates_from(d)
        last_dir = str(d)
        if templates:
            return templates, str(d), None
        last_err = err
    return [], last_dir, last_err or "No candidate data directories found"


def _extract_uploaded_zip_to_cache(zip_bytes: bytes):
    # Clear previous extraction
    target = UPLOAD_CACHE_DIR
    for p in list(target.glob("*")):
        if p.is_file():
            p.unlink(missing_ok=True)
        else:
            import shutil
            shutil.rmtree(p, ignore_errors=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tf:
        tf.write(zip_bytes)
        zpath = Path(tf.name)

    try:
        with zipfile.ZipFile(zpath, "r") as z:
            z.extractall(target)
    finally:
        zpath.unlink(missing_ok=True)

    if (target / "data" / "templates.json").exists():
        return target / "data"
    if (target / "templates.json").exists():
        (target / "data").mkdir(exist_ok=True)
        (target / "templates.json").replace(target / "data" / "templates.json")
        return target / "data"
    return target / "data"


# ----------------------------
# Image pipeline
# ----------------------------
def extract_observed_contour(pil_img: Image.Image, invert: bool):
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        mask = 255 - mask
    pts = contour_from_mask(mask)
    return mask, pts


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Part ID Mockup", layout="wide")
st.title("Part ID mock-up (silhouette matching)")

if cKDTree is None:
    st.error(
        "This app requires SciPy (for KDTree chamfer distance) but SciPy could not be imported.\n\n"
        f"Import error: {_SCIPY_IMPORT_ERROR}\n\n"
        "Install with: pip install scipy"
    )
    st.stop()

st.sidebar.header("Templates")

st.sidebar.write(f"Script folder: {Path(__file__).parent}")
st.sidebar.write(f"Default data folder: {DEFAULT_DATA_DIR}")

uploaded = st.sidebar.file_uploader(
    "Optional: upload a ZIP with data/templates.json + data/templates/*",
    type=["zip"],
)

if uploaded is not None:
    extracted_dir = _extract_uploaded_zip_to_cache(uploaded.getvalue())
    st.sidebar.success(f"Uploaded and extracted templates to: {extracted_dir}")
    load_templates_auto.clear()

templates, data_dir_used, load_err = load_templates_auto()

st.sidebar.write(f"Templates loaded: {len(templates)}")
st.sidebar.write(f"Data dir in use: {data_dir_used}")

if not templates:
    st.error("No templates were loaded.")
    st.write("The app looked in these locations (in order):")
    for d in _candidate_data_dirs():
        st.code(str(d))
    st.write("Error:")
    st.code(load_err or "Unknown")
    st.write(
        "Most common cause: image paths inside templates.json do not match your folder layout.\n"
        "This version tries multiple path bases (script folder, data folder, data/templates folder) to fix that."
    )
    st.stop()

st.sidebar.header("Capture settings")
invert = st.sidebar.checkbox("Invert mask (use if detection is reversed)", value=True)
show_debug = st.sidebar.checkbox("Show debug mask", value=False)

img_in = st.camera_input("Take a picture of a cut part")

if img_in is None:
    st.info("Take a photo above to identify a part.")
    st.stop()

pil_img = Image.open(img_in)
mask, pts = extract_observed_contour(pil_img, invert=invert)

if pts is None:
    st.error("Could not find a clear outer contour. Try stronger contrast and less shadow.")
    if show_debug:
        st.image(mask, caption="Binary mask", use_container_width=True)
    st.stop()

try:
    obs = resample_polyline(pts.astype(np.float32), N_RESAMPLE)
except Exception:
    st.error("Contour found, but could not be resampled. Retake the photo with a cleaner background.")
    if show_debug:
        st.image(mask, caption="Binary mask", use_container_width=True)
    st.stop()

obs = normalize_contour(obs)
obs_fd = fourier_descriptor(obs)

# Stage 1: shortlist by Fourier descriptor distance
scores = []
for t in templates:
    d0 = float(np.linalg.norm(obs_fd - t["fd"]))
    scores.append((d0, t))
scores.sort(key=lambda x: x[0])
short = scores[: min(30, len(scores))]

# Stage 2: verify by rotation-only symmetric chamfer (no mirror)
verified = []
for d0, t in short:
    d = best_rotation_chamfer(obs, t["pts"])
    verified.append((d, d0, t))
verified.sort(key=lambda x: x[0])

st.subheader("Results")
col1, col2 = st.columns([1, 1])
with col1:
    st.image(pil_img, caption="Captured image", use_container_width=True)
with col2:
    if show_debug:
        st.image(mask, caption="Binary mask", use_container_width=True)

k = 5
st.write("Top matches:")
cols = st.columns(k)
for idx in range(min(k, len(verified))):
    d, d0, t = verified[idx]
    with cols[idx]:
        st.image(t["image"], caption=f"{t['part_id']}\nscore={d:.4f}", use_container_width=True)

if verified:
    best = verified[0]
    second = verified[1] if len(verified) > 1 else None
    best_d = float(best[0])
    sep = (float(second[0]) / best_d) if (second is not None and best_d > 0) else None
    confident = (sep is None) or (sep > 1.12)

    st.markdown("---")
    if confident:
        if sep is None:
            st.success(f"Best match: **{best[2]['part_id']}**")
        else:
            st.success(f"Best match: **{best[2]['part_id']}** (separation ratio: {sep:.2f})")
    else:
        st.warning("Low confidence. Consider the top 2–3 matches or retake the photo with better alignment/contrast.")
else:
    st.error("No matches produced. Please retake the photo.")
