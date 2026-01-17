import json
from pathlib import Path

import numpy as np
import cv2
import streamlit as st
from PIL import Image
from scipy.spatial import cKDTree

DATA_DIR = Path(__file__).parent / 'data'
TEMPL_DIR = DATA_DIR / 'templates'
META_PATH = DATA_DIR / 'templates.json'

N_RESAMPLE = 256
FD_COEFFS = 20


def resample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample a 2D polyline to n points equally spaced by arc length.
    pts: (M,2) float.
    """
    if len(pts) < 2:
        raise ValueError('Not enough points to resample')
    # cumulative arc length
    d = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    if total == 0:
        raise ValueError('Zero-length polyline')
    t = np.linspace(0, total, n, endpoint=False)
    # interpolate each dimension
    x = np.interp(t, s, pts[:, 0])
    y = np.interp(t, s, pts[:, 1])
    return np.stack([x, y], axis=1)


def contour_from_mask(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 1000:
        return None
    pts = c.reshape(-1, 2).astype(np.float32)
    return pts


def normalize_contour(pts: np.ndarray) -> np.ndarray:
    # center
    pts = pts - pts.mean(axis=0, keepdims=True)
    # scale by perimeter
    d = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    per = float(d.sum())
    if per == 0:
        return pts
    return pts / per


def fourier_descriptor(pts: np.ndarray, k: int = FD_COEFFS) -> np.ndarray:
    z = pts[:, 0] + 1j * pts[:, 1]
    z = z - z.mean()
    F = np.fft.fft(z)
    mag = np.abs(F)
    mag[0] = 0.0
    # take first k coefficients from positive frequencies
    fd = mag[1 : k + 1]
    # normalize for scale
    denom = fd[0] if fd[0] != 0 else (fd.mean() if fd.mean() != 0 else 1.0)
    return (fd / denom).astype(np.float32)


def rotate_pts(pts: np.ndarray, theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return pts @ R.T


def symmetric_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    ta = cKDTree(a)
    tb = cKDTree(b)
    da, _ = tb.query(a, k=1)
    db, _ = ta.query(b, k=1)
    return float(da.mean() + db.mean())


def best_rotation_chamfer(obs: np.ndarray, templ: np.ndarray) -> float:
    # Rotation search only; no reflections (mirroring not allowed)
    best = 1e9
    for deg in range(0, 360, 10):
        r = rotate_pts(obs, np.deg2rad(deg))
        d = symmetric_chamfer(r, templ)
        if d < best:
            best = d
    return best


@st.cache_resource
def load_templates():
    meta = json.loads(META_PATH.read_text(encoding='utf-8'))
    templates = []
    for item in meta:
        img_path = (Path(__file__).parent / item['image']).resolve()
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # non-white pixels are part
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        pts = contour_from_mask(mask)
        if pts is None:
            continue
        pts = resample_polyline(pts, N_RESAMPLE)
        pts = normalize_contour(pts)
        fd = fourier_descriptor(pts)
        templates.append({
            'part_id': item['part_id'],
            'image': str(img_path),
            'pts': pts,
            'fd': fd,
        })
    return templates


def extract_observed_contour(pil_img: Image.Image, invert: bool):
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu threshold
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        mask = 255 - mask
    pts = contour_from_mask(mask)
    return bgr, mask, pts


st.set_page_config(page_title='Part ID Mockup', layout='wide')

st.title('Part ID mock-up (silhouette matching)')
st.write('Print the cutouts, cut them out, place one on a contrasting background, and take a photo. The app returns the closest matching template(s).')

templates = load_templates()
st.sidebar.header('Capture settings')
invert = st.sidebar.checkbox('Invert mask (use if detection is reversed)', value=True)
show_debug = st.sidebar.checkbox('Show debug views', value=False)

a = st.camera_input('Take a picture of a cut part')

if a is None:
    st.info('Use the camera widget above to take a photo.')
    st.stop()

pil_img = Image.open(a)

bgr, mask, pts = extract_observed_contour(pil_img, invert=invert)

if pts is None:
    st.error('Could not find a clear outer contour. Try stronger contrast, better lighting, or adjust the invert setting.')
    if show_debug:
        st.image(mask, caption='Binary mask')
    st.stop()

obs = resample_polyline(pts.astype(np.float32), N_RESAMPLE)
obs = normalize_contour(obs)
obs_fd = fourier_descriptor(obs)

# Stage 1: coarse by Fourier descriptor
scores = []
for t in templates:
    d = float(np.linalg.norm(obs_fd - t['fd']))
    scores.append((d, t))
scores.sort(key=lambda x: x[0])
short = scores[:25]

# Stage 2: chamfer verification (rotation-only)
verified = []
for d0, t in short:
    d = best_rotation_chamfer(obs, t['pts'])
    verified.append((d, d0, t))
verified.sort(key=lambda x: x[0])

best = verified[0]
second = verified[1] if len(verified) > 1 else None

# Confidence heuristic
best_d = best[0]
sep = (second[0] / best_d) if (second and best_d > 0) else None
confident = (sep is None) or (sep > 1.12)

st.subheader('Results')
col1, col2 = st.columns([1, 1])
with col1:
    st.image(pil_img, caption='Captured image', use_container_width=True)
with col2:
    if show_debug:
        st.image(mask, caption='Binary mask', use_container_width=True)

st.write(f"Templates loaded: {len(templates)}")

k = 5
st.write('Top matches:')
cols = st.columns(k)
for idx in range(k):
    if idx >= len(verified):
        break
    d, d0, t = verified[idx]
    with cols[idx]:
        st.image(t['image'], caption=f"{t['part_id']}\nscore={d:.4f}", use_container_width=True)

st.markdown('---')
if confident:
    st.success(f"Best match: **{best[2]['part_id']}** (separation ratio: {sep:.2f} if available)")
else:
    st.warning('Low confidence. Consider using the top 2-3 matches or retaking the photo with better alignment/contrast.')

