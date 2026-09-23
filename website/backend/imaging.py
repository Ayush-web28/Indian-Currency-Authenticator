import io

from PIL import Image, ImageOps

MAX_SIDE = 1024
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 50_000_000


class ImageRejected(Exception):
    pass


def load_image(data: bytes) -> tuple[Image.Image, tuple[int, int]]:
    """Decode an upload at reduced size. Returns (RGB image, original (width, height))."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageRejected("Image is larger than 15 MB")
    try:
        image = Image.open(io.BytesIO(data))
        original_size = image.size
        image.draft("RGB", (MAX_SIDE, MAX_SIDE))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as e:
        raise ImageRejected("Invalid image file") from e
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image, original_size
