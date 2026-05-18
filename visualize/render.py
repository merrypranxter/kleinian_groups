import numpy as np
from typing import Optional, List

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def _complex_to_pixel(
    z: complex,
    bounds: tuple,
    width: int,
    height: int
) -> tuple:
    """Convert complex number to pixel coordinates.
    bounds = (x_min, x_max, y_min, y_max)
    """
    x_min, x_max, y_min, y_max = bounds
    px = int((z.real - x_min) / (x_max - x_min) * width)
    py = int((z.imag - y_min) / (y_max - y_min) * height)
    py = height - 1 - py
    return (np.clip(px, 0, width - 1), np.clip(py, 0, height - 1))


def render_limit_set(
    points: np.ndarray,
    width: int = 800,
    height: int = 800,
    color_by: str = 'depth',
    bg: str = 'black',
    output_path: Optional[str] = None
):
    """Render complex limit-set points to a PIL Image."""
    if not PIL_AVAILABLE:
        raise ImportError("Pillow is required for rendering. Install with: pip install pillow")

    if len(points) == 0:
        img = Image.new('RGB', (width, height), bg)
        if output_path:
            img.save(output_path)
        return img

    xs = np.real(points)
    ys = np.imag(points)
    pad = 0.1
    x_range = xs.max() - xs.min() if xs.max() != xs.min() else 1.0
    y_range = ys.max() - ys.min() if ys.max() != ys.min() else 1.0
    bounds = (
        xs.min() - pad * x_range,
        xs.max() + pad * x_range,
        ys.min() - pad * y_range,
        ys.max() + pad * y_range
    )

    bg_color = (0, 0, 0) if bg == 'black' else (255, 255, 255)
    img = Image.new('RGB', (width, height), bg_color)
    pixels = img.load()

    n = len(points)
    for i, z in enumerate(points):
        px, py = _complex_to_pixel(z, bounds, width, height)

        if color_by == 'depth':
            t = i / max(n - 1, 1)
            r = int(255 * (0.5 + 0.5 * np.sin(t * 6 * np.pi)))
            g = int(255 * (0.5 + 0.5 * np.sin(t * 6 * np.pi + 2)))
            b = int(255 * (0.5 + 0.5 * np.sin(t * 6 * np.pi + 4)))
            color = (r, g, b)
        elif color_by == 'argument':
            arg = (np.angle(z) + np.pi) / (2 * np.pi)
            r = int(255 * (0.5 + 0.5 * np.sin(arg * 2 * np.pi)))
            g = int(255 * (0.5 + 0.5 * np.sin(arg * 2 * np.pi + 2.094)))
            b = int(255 * (0.5 + 0.5 * np.sin(arg * 2 * np.pi + 4.189)))
            color = (r, g, b)
        elif color_by == 'density':
            color = (200, 100, 50)
        else:
            color = (255, 255, 255)

        if 0 <= px < width and 0 <= py < height:
            pixels[px, py] = color

    if output_path:
        img.save(output_path)
    return img


def render_circles(
    circles: list,
    width: int = 800,
    height: int = 800,
    output_path: Optional[str] = None
):
    """Render a list of circle dicts {center, radius, depth} to a PIL Image."""
    if not PIL_AVAILABLE:
        raise ImportError("Pillow is required for rendering. Install with: pip install pillow")

    if not circles:
        img = Image.new('RGB', (width, height), (0, 0, 0))
        if output_path:
            img.save(output_path)
        return img

    centers = np.array([c['center'] for c in circles])
    radii = np.array([c['radius'] for c in circles])

    xs = np.real(centers)
    ys = np.imag(centers)

    pad = max(radii.max(), 0.1)
    bounds = (
        xs.min() - pad,
        xs.max() + pad,
        ys.min() - pad,
        ys.max() + pad
    )

    img = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    max_depth = max((c.get('depth', 1) for c in circles), default=1)

    for c in circles:
        center = c['center']
        radius = c['radius']
        depth = c.get('depth', 1)

        t = depth / max(max_depth, 1)
        r_col = int(255 * (0.3 + 0.7 * (1 - t)))
        g_col = int(255 * 0.5 * t)
        b_col = int(255 * t)
        color = (r_col, g_col, b_col)

        cx, cy = _complex_to_pixel(center, bounds, width, height)

        x_scale = width / (bounds[1] - bounds[0])
        r_px = max(int(radius * x_scale), 1)

        draw.ellipse([cx - r_px, cy - r_px, cx + r_px, cy + r_px], outline=color)

    if output_path:
        img.save(output_path)
    return img
