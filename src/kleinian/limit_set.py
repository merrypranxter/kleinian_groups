import numpy as np
from typing import Optional
from .groups import KleinianGroup


def compute_limit_set(
    group: KleinianGroup,
    seed_points=None,
    max_depth: int = 8,
    tol: float = 1e-6
) -> np.ndarray:
    if seed_points is None:
        seed_points = [complex(0.1, 0.1), complex(-0.1, 0.05)]

    limit_points = []

    def dfs(word: str, transform, depth: int, prev_point: complex):
        if depth > max_depth:
            return
        for letter, t in group._transforms.items():
            if group._is_cancellation(word, letter):
                continue
            new_transform = transform * t
            new_word = word + letter
            for seed in seed_points:
                try:
                    p = new_transform(seed)
                    if np.isfinite(np.abs(p)):
                        limit_points.append(p)
                        if depth > 0 and np.abs(p - prev_point) < tol:
                            return
                        dfs(new_word, new_transform, depth + 1, p)
                except Exception:
                    pass

    for letter, t in group._transforms.items():
        for seed in seed_points:
            try:
                p = t(seed)
                if np.isfinite(np.abs(p)):
                    limit_points.append(p)
                    dfs(letter, t, 1, p)
            except Exception:
                pass

    if not limit_points:
        return np.array([], dtype=np.complex128)
    return np.array(limit_points, dtype=np.complex128)


def isometric_circle_limit_set(
    group: KleinianGroup,
    max_depth: int = 6
) -> list:
    circles = []
    for word, transform in group.enumerate_words(max_depth):
        if len(word) == 0:
            continue
        try:
            center, radius = transform.isometric_circle()
            circles.append({
                'center': center,
                'radius': radius,
                'depth': len(word),
                'word': word
            })
        except ValueError:
            pass
    return circles


def hausdorff_dim_estimate(
    points: np.ndarray,
    epsilon_range=None
) -> float:
    if len(points) < 10:
        return 0.0

    xs = np.real(points)
    ys = np.imag(points)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    span = max(x_max - x_min, y_max - y_min, 1e-10)

    if epsilon_range is None:
        epsilon_range = np.logspace(-1, -3, 20) * span

    log_inv_eps = []
    log_N = []

    for eps in epsilon_range:
        if eps <= 0:
            continue
        ix = np.floor((xs - x_min) / eps).astype(int)
        iy = np.floor((ys - y_min) / eps).astype(int)
        boxes = set(zip(ix, iy))
        N = len(boxes)
        if N > 1:
            log_inv_eps.append(np.log(1.0 / eps))
            log_N.append(np.log(N))

    if len(log_inv_eps) < 2:
        return 1.0

    log_inv_eps = np.array(log_inv_eps)
    log_N = np.array(log_N)
    A = np.vstack([log_inv_eps, np.ones(len(log_inv_eps))]).T
    slope, _ = np.linalg.lstsq(A, log_N, rcond=None)[0]
    return float(np.clip(slope, 0.0, 2.0))
