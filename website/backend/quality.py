import numpy as np
from PIL import Image

ANALYSIS_SIZE = 512

# (warn_below, fail_below) for "higher is better"; (warn_above, fail_above) for glare.
SHARPNESS = (150.0, 60.0)
CONTRAST = (28.0, 14.0)
RESOLUTION = (400, 200)
DARK = (70.0, 40.0)
BRIGHT = (200.0, 225.0)
GLARE = (0.08, 0.20)


def _level(value, warn, fail, higher_is_better=True):
    if higher_is_better:
        return "fail" if value < fail else "warn" if value < warn else "ok"
    return "fail" if value > fail else "warn" if value > warn else "ok"


def _laplacian_variance(gray: np.ndarray) -> float:
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(lap.var())


def assess_quality(image: Image.Image, original_size: tuple[int, int] | None = None) -> dict:
    width, height = image.size
    scale = ANALYSIS_SIZE / max(width, height)
    if scale < 1:
        image = image.resize((max(3, round(width * scale)), max(3, round(height * scale))))
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    sharpness = _laplacian_variance(gray)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    glare = float((gray >= 250).mean())
    smallest_side = min(original_size or (width, height))

    brightness_status = (
        _level(brightness, *DARK)
        if brightness < 128
        else _level(brightness, *BRIGHT, higher_is_better=False)
    )

    checks = [
        {
            "name": "sharpness",
            "value": round(sharpness, 1),
            "status": _level(sharpness, *SHARPNESS),
            "tip": "Image looks blurry. Hold the phone steady and let the camera focus.",
        },
        {
            "name": "brightness",
            "value": round(brightness, 1),
            "status": brightness_status,
            "tip": "Too dark. Move to better light." if brightness < 128 else "Overexposed. Reduce direct light.",
        },
        {
            "name": "contrast",
            "value": round(contrast, 1),
            "status": _level(contrast, *CONTRAST),
            "tip": "Low contrast. Use even lighting and a plain background.",
        },
        {
            "name": "glare",
            "value": round(glare * 100, 1),
            "status": _level(glare, *GLARE, higher_is_better=False),
            "tip": "Glare on the note. Tilt it away from the light source.",
        },
        {
            "name": "resolution",
            "value": smallest_side,
            "status": _level(smallest_side, *RESOLUTION),
            "tip": "Image is too small. Move closer or use a higher-resolution photo.",
        },
    ]

    statuses = {c["status"] for c in checks}
    rating = "POOR" if "fail" in statuses else "FAIR" if "warn" in statuses else "GOOD"
    return {
        "rating": rating,
        "retake": rating == "POOR",
        "checks": checks,
        "tips": [c["tip"] for c in checks if c["status"] != "ok"],
    }
