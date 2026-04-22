# mask_editor_view.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem


@dataclass
class Stroke:
    idx: np.ndarray
    before: np.ndarray
    after: np.ndarray


class MaskEditorView(QGraphicsView):
    TOOL_ERASE = 0
    TOOL_BRUSH = 1

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setRenderHints(QPainter.RenderHint.Antialiasing |
                            QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._img_item: Optional[QGraphicsPixmapItem] = None
        self._ov_item: Optional[QGraphicsPixmapItem] = None
        self._brush_item: Optional[QGraphicsEllipseItem] = None

        self.mask_ai: Optional[np.ndarray] = None
        self.mask_edit: Optional[np.ndarray] = None

        self.edit_enabled = False
        self.tool = self.TOOL_BRUSH
        self.brush_radius = 10
        self.overlay_alpha = 120  # 0..255
        self.overlay_rgb = (0, 255, 0)

        self.undo_stack: List[Stroke] = []
        self.redo_stack: List[Stroke] = []

        self._drawing = False
        self._last_pt: Optional[Tuple[int, int]] = None
        self._stroke_before: Dict[int, int] = {}
        self._brush_cache: Dict[int, np.ndarray] = {}

        # Pan/zoom UX
        self._panning = False
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    # -------- public API --------
    def set_image(self, img: np.ndarray):
        """Set base image (RGB888 or Gray8) into scene."""
        qimg = self._numpy_to_qimage(img)
        pm = QPixmap.fromImage(qimg)

        if self._img_item is None:
            self._img_item = self._scene.addPixmap(pm)
            self._img_item.setZValue(0)
        else:
            self._img_item.setPixmap(pm)

        # ✅ overlay layer - ALWAYS transparent base pixmap (prevents black rectangle)
        if self._ov_item is None:
            empty = QPixmap(pm.size())
            empty.fill(Qt.GlobalColor.transparent)
            self._ov_item = self._scene.addPixmap(empty)
            self._ov_item.setZValue(10)
        else:
            # if image size changed, resize overlay pixmap as transparent
            if self._ov_item.pixmap().size() != pm.size():
                empty = QPixmap(pm.size())
                empty.fill(Qt.GlobalColor.transparent)
                self._ov_item.setPixmap(empty)

        # brush preview circle
        if self._brush_item is None:
            self._brush_item = self._scene.addEllipse(0, 0, 1, 1, QPen(Qt.GlobalColor.green, 1))
            self._brush_item.setZValue(20)
            self._brush_item.setVisible(False)

        self._scene.setSceneRect(QRectF(0, 0, pm.width(), pm.height()))

        # keep current zoom/rotate set by parent (do not force fit every time)
        # but on first image, a fit is nice
        if self.transform().isIdentity():
            self.resetTransform()
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        # if we already have an edited mask, rebuild overlay
        if self.mask_edit is not None:
            self._rebuild_overlay()
        else:
            self._clear_overlay()

    def set_masks(self, mask_ai: np.ndarray, mask_edit: Optional[np.ndarray] = None):
        self.mask_ai = (mask_ai.astype(np.uint8) > 0).astype(np.uint8)
        self.mask_edit = self.mask_ai.copy() if mask_edit is None else (mask_edit.astype(np.uint8) > 0).astype(np.uint8)
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._rebuild_overlay()

    def get_mask_edit(self) -> Optional[np.ndarray]:
        return None if self.mask_edit is None else self.mask_edit.copy()

    def set_edit_enabled(self, enabled: bool):
        self.edit_enabled = bool(enabled)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        if self._brush_item:
            self._brush_item.setVisible(enabled)

    def set_tool(self, tool: int):
        self.tool = int(tool)
        if self._brush_item:
            pen = QPen(Qt.GlobalColor.green if tool == self.TOOL_BRUSH else Qt.GlobalColor.red, 1)
            self._brush_item.setPen(pen)

    def set_brush_radius(self, r: int):
        self.brush_radius = max(1, int(r))

    def reset_to_ai(self):
        if self.mask_ai is None:
            return
        self.mask_edit = self.mask_ai.copy()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._rebuild_overlay()

    def undo(self):
        if not self.undo_stack or self.mask_edit is None:
            return
        s = self.undo_stack.pop()
        flat = self.mask_edit.ravel()
        flat[s.idx] = s.before
        self.redo_stack.append(s)
        self._rebuild_overlay()

    def redo(self):
        if not self.redo_stack or self.mask_edit is None:
            return
        s = self.redo_stack.pop()
        flat = self.mask_edit.ravel()
        flat[s.idx] = s.after
        self.undo_stack.append(s)
        self._rebuild_overlay()

    def smooth_edges(self, mode="light"):
        if self.mask_edit is None:
            return
        before = self.mask_edit.copy()

        try:
            import cv2  # type: ignore
            k = 3 if mode == "light" else 5
            kernel = np.ones((k, k), np.uint8)
            m = (self.mask_edit * 255).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
            self.mask_edit = (m > 127).astype(np.uint8)
        except Exception:
            self.mask_edit = self._majority_filter3x3(self.mask_edit)

        self._push_diff(before, self.mask_edit)
        self._rebuild_overlay()

    # -------- events --------
    def wheelEvent(self, e):
        if e.angleDelta().y() > 0:
            self.scale(1.15, 1.15)
        else:
            self.scale(1/1.15, 1/1.15)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            self._panning = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            super().mousePressEvent(e)
            return

        if e.button() == Qt.MouseButton.LeftButton and self.edit_enabled and self.mask_edit is not None:
            self._drawing = True
            self._stroke_before.clear()
            pt = self._to_img_pt(e.position())
            self._last_pt = pt
            if pt:
                self._stamp(pt[0], pt[1])
                self._rebuild_overlay()
            return

        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        pt = self._to_img_pt(e.position())

        # brush preview
        if pt and self._brush_item and self.edit_enabled:
            x, y = pt
            r = self.brush_radius
            self._brush_item.setRect(x - r, y - r, 2*r, 2*r)

        if self._panning:
            super().mouseMoveEvent(e)
            return

        if self._drawing and self.edit_enabled and self.mask_edit is not None and self._last_pt and pt:
            self._draw_line(self._last_pt, pt)
            self._last_pt = pt
            self._rebuild_overlay()
            return

        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton and self._panning:
            self._panning = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            super().mouseReleaseEvent(e)
            return

        if e.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            self._last_pt = None
            self._commit_stroke()
            return

        super().mouseReleaseEvent(e)

    # -------- core paint --------
    def _commit_stroke(self):
        if self.mask_edit is None or not self._stroke_before:
            return
        flat = self.mask_edit.ravel()
        idx = np.fromiter(self._stroke_before.keys(), dtype=np.int64)
        before = np.fromiter(self._stroke_before.values(), dtype=np.uint8)
        after = flat[idx].astype(np.uint8)
        ch = before != after
        if not np.any(ch):
            return
        self.undo_stack.append(Stroke(idx[ch], before[ch], after[ch]))
        self.redo_stack.clear()

    def _draw_line(self, a, b):
        x0, y0 = a
        x1, y1 = b
        n = max(abs(x1-x0), abs(y1-y0), 1)
        for i in range(n+1):
            x = int(round(x0 + (x1-x0)*i/n))
            y = int(round(y0 + (y1-y0)*i/n))
            self._stamp(x, y)

    def _stamp(self, x, y):
        m = self.mask_edit
        if m is None:
            return
        h, w = m.shape
        offsets = self._brush_offsets(self.brush_radius)
        newv = 1 if self.tool == self.TOOL_BRUSH else 0

        xs = x + offsets[:, 1]
        ys = y + offsets[:, 0]
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs = xs[ok]; ys = ys[ok]

        flat = m.ravel()
        idx = ys.astype(np.int64) * w + xs.astype(np.int64)
        old = flat[idx]
        ch = old != newv
        if not np.any(ch):
            return
        idx_ch = idx[ch]
        old_ch = old[ch]

        for ii, ov in zip(idx_ch.tolist(), old_ch.tolist()):
            if ii not in self._stroke_before:
                self._stroke_before[ii] = int(ov)

        flat[idx_ch] = newv

    # -------- overlay build --------
    def _clear_overlay(self):
        if self._ov_item is None:
            return
        empty = QPixmap(self._ov_item.pixmap().size())
        empty.fill(Qt.GlobalColor.transparent)
        self._ov_item.setPixmap(empty)

    def _rebuild_overlay(self):
        if self._ov_item is None:
            return
        if self.mask_edit is None:
            self._clear_overlay()
            return

        h, w = self.mask_edit.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        sel = self.mask_edit.astype(bool)
        r, g, b = self.overlay_rgb
        rgba[sel, 0] = r
        rgba[sel, 1] = g
        rgba[sel, 2] = b
        rgba[sel, 3] = np.uint8(self.overlay_alpha)

        qimg = QImage(rgba.data, w, h, 4*w, QImage.Format.Format_RGBA8888)
        self._ov_item.setPixmap(QPixmap.fromImage(qimg.copy()))

    # -------- helpers --------
    def _to_img_pt(self, vp: QPointF) -> Optional[Tuple[int, int]]:
        sp = self.mapToScene(int(vp.x()), int(vp.y()))
        if not self._scene.sceneRect().contains(sp):
            return None
        return int(sp.x()), int(sp.y())

    def _brush_offsets(self, r: int) -> np.ndarray:
        if r in self._brush_cache:
            return self._brush_cache[r]
        yy, xx = np.ogrid[-r:r+1, -r:r+1]
        mask = (xx*xx + yy*yy) <= r*r
        pts = np.column_stack(np.where(mask)).astype(np.int16)
        pts[:, 0] -= r
        pts[:, 1] -= r
        self._brush_cache[r] = pts
        return pts

    def _numpy_to_qimage(self, img: np.ndarray) -> QImage:
        if img.ndim == 2:
            h, w = img.shape
            img8 = img.astype(np.uint8, copy=False)
            return QImage(img8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        if img.ndim == 3 and img.shape[2] == 3:
            h, w, _ = img.shape
            img8 = img.astype(np.uint8, copy=False)
            return QImage(img8.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()
        raise ValueError("Unsupported image shape")

    def _push_diff(self, before: np.ndarray, after: np.ndarray):
        b = before.ravel()
        a = after.ravel()
        diff = b != a
        if not np.any(diff):
            return
        idx = np.where(diff)[0].astype(np.int64)
        self.undo_stack.append(Stroke(idx, b[idx].astype(np.uint8), a[idx].astype(np.uint8)))
        self.redo_stack.clear()

    def _majority_filter3x3(self, m: np.ndarray) -> np.ndarray:
        m = m.astype(np.uint8)
        pad = np.pad(m, 1, mode="edge")
        acc = np.zeros_like(m, dtype=np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc += pad[1+dy:1+dy+m.shape[0], 1+dx:1+dx+m.shape[1]]
        return (acc >= 5).astype(np.uint8)
