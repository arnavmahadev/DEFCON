"""Vectorized geometry helpers for node features (supports task 2.1).

All functions operate on numpy arrays and are written as small, independently
testable units — the triangle and corridor counts in particular are easy to get
subtly wrong, so they get dedicated tests.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "angle_sin_cos",
    "vector_angle_sin_cos",
    "dist_point_to_segments",
    "points_in_triangle",
]


def angle_sin_cos(dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sin, cos) of the angle of vector (dx, dy) w.r.t. the +x axis.

    Zero-length vectors map to angle 0 -> (sin=0, cos=1).
    """
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    norm = np.hypot(dx, dy)
    safe = norm > 1e-9
    sin = np.where(safe, dy / np.where(safe, norm, 1.0), 0.0)
    cos = np.where(safe, dx / np.where(safe, norm, 1.0), 1.0)
    return sin, cos


def vector_angle_sin_cos(
    ux: np.ndarray, uy: np.ndarray, vx: float, vy: float
) -> tuple[np.ndarray, np.ndarray]:
    """(sin, cos) of the angle between each vector (ux,uy) and a fixed vector (vx,vy).

    Uses the 2D cross and dot products. Degenerate (zero-length) vectors -> angle 0.
    """
    ux = np.asarray(ux, dtype=float)
    uy = np.asarray(uy, dtype=float)
    un = np.hypot(ux, uy)
    vn = np.hypot(vx, vy)
    denom = un * vn
    safe = denom > 1e-9
    dot = ux * vx + uy * vy
    cross = ux * vy - uy * vx
    cos = np.where(safe, dot / np.where(safe, denom, 1.0), 1.0)
    sin = np.where(safe, cross / np.where(safe, denom, 1.0), 0.0)
    return sin, cos


def dist_point_to_segments(
    px: np.ndarray, py: np.ndarray, ax: float, ay: float, bx: float, by: float
) -> np.ndarray:
    """Distance from each point (px, py) to the segment A(ax,ay)->B(bx,by)."""
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    abx, aby = bx - ax, by - ay
    seg_len2 = abx * abx + aby * aby
    if seg_len2 < 1e-12:
        # Degenerate segment -> distance to the point A.
        return np.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / seg_len2
    t = np.clip(t, 0.0, 1.0)
    projx = ax + t * abx
    projy = ay + t * aby
    return np.hypot(px - projx, py - projy)


def points_in_triangle(
    px: np.ndarray, py: np.ndarray,
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float,
) -> np.ndarray:
    """Boolean mask: is each point inside triangle ABC (inclusive of edges)?"""
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)

    def sign(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

    d1 = sign(px, py, ax, ay, bx, by)
    d2 = sign(px, py, bx, by, cx, cy)
    d3 = sign(px, py, cx, cy, ax, ay)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    # Inside if all same sign (allowing zeros on edges).
    return ~(has_neg & has_pos)
