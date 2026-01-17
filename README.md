# Part ID Mock-up (PDF cutouts + camera recognition)

This is a small prototype that:
1) lets you print individual part cutouts from your provided PDF (already generated), and
2) identifies a cutout using a camera snapshot by matching its outer silhouette to the template library.

## 1) Print the cutouts
Print `cutouts_printable.pdf` and cut out the parts with scissors.

## 2) Run the app (Streamlit)

### Prerequisites
- Python 3.10+ recommended

### Install
```bash
pip install -r requirements.txt
```

### Run
```bash
streamlit run app.py
```

## Capture tips (important)
- Use a *high-contrast* background (e.g., dark/black mat behind the paper cutout).
- Keep the camera as zenithal/perpendicular as possible.
- Avoid strong shadows; use diffuse light.
- Ensure the full part is in frame.

## Output semantics
- The app returns the best matching template ID (e.g., `P1-02`).
- For this mock-up, IDs are synthetic (page + index). In your production system, the same pipeline would map to your real part IDs.
