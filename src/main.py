# mainn.py
import sys
import math
import io
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List
from PyQt6 import sip

import cv2
import numpy as np

from PyQt6 import uic
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QListWidgetItem,
    QGraphicsTextItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsEllipseItem,
    QGraphicsSimpleTextItem,
    QGraphicsPathItem,
)
from PyQt6.QtGui import (
    QPixmap,
    QImage,
    QPen,
    QColor,
    QTransform,
    QKeySequence,
    QShortcut,
)
from PyQt6.QtCore import (
    Qt,
    QDate,
    QEvent,
    QRectF,
    QThread,
    pyqtSignal,
    QPointF,
    QByteArray,
    QBuffer,
)

from infer_engine import InferEngine
from mask_editor_view import MaskEditorView

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader


# =====================================================
# Worker thread
# =====================================================
class InferenceWorker(QThread):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, engine: InferEngine, image_path: str, task: str):
        super().__init__()
        self.engine = engine
        self.image_path = image_path
        self.task = task

    def run(self):
        try:
            img = cv2.imread(self.image_path)
            if img is None:
                raise RuntimeError("Nem tudtam beolvasni a képet cv2-vel.")

            out = {"task": self.task}

            if self.task == "cls":
                idx, name, score = self.engine.classify(img)
                out.update({"idx": int(idx), "name": str(name), "score": float(score)})

            elif self.task == "seg":
                mask, prob = self.engine.segment(img)
                out.update({"mask": mask, "prob_map": prob})

            elif self.task == "cam":
                cam01, idx, score = self.engine.gradcam(img, class_idx=None)
                out.update({"cam01": cam01, "idx": int(idx), "score": float(score)})

            else:
                raise RuntimeError(f"Ismeretlen task: {self.task}")

            self.finished.emit(out)

        except Exception as e:
            self.failed.emit(str(e))


# =====================================================
# QImage / numpy helpers
# =====================================================
def qimage_to_numpy_rgb(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.bits()
    ptr.setsize(h * w * 3)
    return np.frombuffer(ptr, np.uint8).reshape((h, w, 3)).copy()


def numpy_rgb_to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, ch = rgb.shape
    if ch != 3:
        raise ValueError("RGB array must have shape (H,W,3)")
    ptr = sip.voidptr(rgb.ctypes.data)
    qimg = QImage(ptr, w, h, w * 3, QImage.Format.Format_RGB888)
    return qimg.copy()


def numpy_gray_to_qimage(gray: np.ndarray) -> QImage:
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    h, w = gray.shape
    ptr = sip.voidptr(gray.ctypes.data)
    qimg = QImage(ptr, w, h, w, QImage.Format.Format_Grayscale8)
    return qimg.copy()


def qimage_to_png_bytes(img: QImage) -> bytes:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QBuffer.OpenModeFlag.WriteOnly)
    ok = img.save(buf, "PNG")
    buf.close()
    if not ok:
        raise RuntimeError("Nem sikerült a QImage -> PNG konvertálás.")
    return bytes(ba)


# =====================================================
# MainWindow
# =====================================================
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        uic.loadUi("windowmini.ui", self)

        if not hasattr(self, "imageLabel"):
            QMessageBox.critical(self, "UI error", "A .ui fájlban nincs 'imageLabel' (MaskEditorView).")
            raise RuntimeError("Missing imageLabel in UI")

        self.imageLabel: MaskEditorView

        self.imageLabel.setMouseTracking(True)
        self.imageLabel.viewport().setMouseTracking(True)
        self.imageLabel.viewport().installEventFilter(self)

        self.current_image_path: Optional[str] = None

        self._base_bgr: Optional[np.ndarray] = None
        self._base_rgb: Optional[np.ndarray] = None
        self._disp_rgb: Optional[np.ndarray] = None

        self.zoom_factor: float = 1.0
        self.rotation_angle: float = 0.0

        self.brightness_value: int = 0
        self.contrast_value: int = 0

        self.mm_per_pixel: float = 0.5

        self.show_overlay: bool = True
        if hasattr(self, "checkShowGrowth"):
            self.checkShowGrowth.setChecked(True)
            self.checkShowGrowth.toggled.connect(self.on_toggle_growth_overlay)
        self.overlay_alpha: float = 0.50
        self.seg_threshold: float = 0.50
        self.mask_visible: bool = True

        self.show_heatmap: bool = False

        self._last_mask: Optional[np.ndarray] = None
        self._last_prob: Optional[np.ndarray] = None
        self._last_cls: Optional[Tuple[int, str, float]] = None
        self._last_cam: Optional[np.ndarray] = None
        self._last_cam_info: Optional[Tuple[int, float]] = None
        self._growth_map: Optional[np.ndarray] = None
        self.show_growth_overlay: bool = True

        # fallback class mapping
        self.class_names = {
            0: "Glioma",
            1: "Meningioma",
            2: "No Tumor",
            3: "Pituitary tumor",
        }

        self.pan_enabled: bool = True

        self.crosshair_h: Optional[QGraphicsLineItem] = None
        self.crosshair_v: Optional[QGraphicsLineItem] = None
        self.last_scene_pos: Optional[QPointF] = None

        self._orient_items: List[QGraphicsTextItem] = []

        self.roi_mode: str = "Line"
        self.roi_start: Optional[QPointF] = None
        self.roi_item = None
        self.roi_text_item: Optional[QGraphicsTextItem] = None
        self.roi_history: List[Dict[str, Any]] = []

        self._lesions: List[Dict[str, Any]] = []
        self._selected_lesion_index: int = -1

        self._lesion_bbox_item: Optional[QGraphicsRectItem] = None
        self._lesion_label_item: Optional[QGraphicsSimpleTextItem] = None
        self._lesion_contour_item = None
        self._lesion_overlay_items: List[Any] = []
        self.show_lesion_box: bool = True

        if hasattr(self, "checkShowLesionBox"):
            self.checkShowLesionBox.setChecked(True)
            self.checkShowLesionBox.toggled.connect(self.on_toggle_lesion_box)
        if hasattr(self, "checkShowMask"):
            self.checkShowMask.setChecked(True)
            self.checkShowMask.toggled.connect(self.on_toggle_mask_visibility)
        if hasattr(self, "dateDateofBirth"):
            self.dateDateofBirth.setDisplayFormat("dd.MM.yyyy")
            self.dateDateofBirth.setCalendarPopup(True)
            self.dateDateofBirth.dateChanged.connect(self.on_date_of_birth_changed)
            self.on_date_of_birth_changed(self.dateDateofBirth.date())

        self._connect_buttons()
        self._setup_image_tools()
        self._setup_segmentation_ui_controls()
        self._setup_manual_edit_controls()

        if hasattr(self, "comboRoiMode"):
            self.comboRoiMode.currentTextChanged.connect(self.on_roi_mode_changed)
            self.roi_mode = self.comboRoiMode.currentText()
        if hasattr(self, "labelRoiStats"):
            self.labelRoiStats.setText("ROI: -")
        if hasattr(self, "listRoiHistory"):
            try:
                self.listRoiHistory.itemClicked.connect(self.on_roi_history_clicked)
            except Exception:
                pass
        if hasattr(self, "listLesions"):
            try:
                self.listLesions.itemClicked.connect(self.on_lesion_selected)
            except Exception:
                pass
        if hasattr(self, "btnRoiDelete"):
            self.btnRoiDelete.clicked.connect(self.on_roi_delete)
        if hasattr(self, "btnRoiClear"):
            self.btnRoiClear.clicked.connect(self.on_roi_clear)

        if hasattr(self, "dialRotate"):
            self.dialRotate.setRange(-180, 180)
            self.dialRotate.setValue(0)
            self.dialRotate.setTracking(True)
            self.dialRotate.valueChanged.connect(self.on_rotate_changed)

        if hasattr(self, "horizontalSliderZoom"):
            self.horizontalSliderZoom.setRange(10, 300)
            self.horizontalSliderZoom.setValue(100)
            self.horizontalSliderZoom.valueChanged.connect(self.on_zoom_changed)

        self._connect_menus_best_effort()

        if hasattr(self, "labelHUD"):
            self.labelHUD.setText("X: -  Y: -\nZoom: 100%  Rotate: 0°\nIntensity: -")
            self.labelHUD.show()
        if hasattr(self, "labelMagnifier"):
            self.labelMagnifier.clear()
            self.labelMagnifier.show()

        self.engine: Optional[InferEngine] = None
        self._worker: Optional[InferenceWorker] = None
        try:
            self.engine = InferEngine(Path.cwd(), device="cpu", debug=False)
            self.engine.load()

            # az engine-ből jövő class neveket használjuk, ha vannak
            if hasattr(self.engine, "class_names") and self.engine.class_names:
                self.class_names = {i: str(n) for i, n in enumerate(self.engine.class_names)}

        except Exception as e:
            _toggle_msg = f"{e}\n\nEllenőrizd: models/ és infer_engine.py / src/*"
            QMessageBox.critical(self, "Model load error", _toggle_msg)
            self.engine = None

        if hasattr(self, "maskLabel"):
            self.maskLabel.clear()

        self._render_results_panel()

        self._snapshot_set_default()
        self._reset_lesion_review_panel()

    def _w(self, *names):
        for n in names:
            if hasattr(self, n):
                return getattr(self, n)
        return None

    # =====================================================
    # Tumor Snapshot (2D Analysis)
    # =====================================================
    def _snapshot_set_default(self, msg: str = "Run segmentation to generate a 2D tumor snapshot."):
        lbl_size = self._w("labelSizeCategory")
        lbl_shape = self._w("labelShapeCategory")
        lbl_edge = self._w("labelEdgeCategory")
        lbl_text = self._w("labelClinicalImpression")

        if lbl_size is not None:
            lbl_size.setText("—")
        if lbl_shape is not None:
            lbl_shape.setText("—")
        if lbl_edge is not None:
            lbl_edge.setText("—")
        if lbl_text is not None:
            lbl_text.setText(msg)

    def _mask_contour_metrics(self, mask01: np.ndarray):
        m = (mask01 > 0).astype(np.uint8)
        area_px = int(m.sum())
        if area_px <= 0:
            return 0, 0.0, 0.0

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return area_px, 0.0, 0.0

        c = max(cnts, key=cv2.contourArea)
        per_px = float(cv2.arcLength(c, True))
        if per_px <= 1e-6:
            return area_px, per_px, 0.0

        circ = float(4.0 * math.pi * float(area_px) / (per_px * per_px))
        return area_px, per_px, circ

    def _edge_sharpness_score(self, mask01: np.ndarray):
        if self._disp_rgb is None:
            return None

        m = (mask01 > 0).astype(np.uint8)
        if int(m.sum()) == 0:
            return None

        gray = (0.299 * self._disp_rgb[..., 0] +
                0.587 * self._disp_rgb[..., 1] +
                0.114 * self._disp_rgb[..., 2]).astype(np.float32)

        k = np.ones((3, 3), np.uint8)
        dil = cv2.dilate(m, k, iterations=2)
        ero = cv2.erode(m, k, iterations=2)
        ring = ((dil > 0) & (ero == 0))

        if int(ring.sum()) < 20:
            return None

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        vals = mag[ring]
        if vals.size == 0:
            return None

        p95 = float(np.percentile(mag, 95)) + 1e-6
        score = float(np.clip(float(vals.mean()) / p95, 0.0, 1.0))
        return score

    def _estimate_visible_brain_mask(self) -> Optional[np.ndarray]:
        if self._disp_rgb is None:
            return None

        gray = (
            0.299 * self._disp_rgb[..., 0] +
            0.587 * self._disp_rgb[..., 1] +
            0.114 * self._disp_rgb[..., 2]
        ).astype(np.uint8)

        _, th = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)

        k = np.ones((5, 5), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
        if num <= 1:
            return None

        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        brain_mask = (labels == best).astype(np.uint8)
        return brain_mask

    def _compute_relative_tumor_size_percent(self, tumor_mask: np.ndarray) -> Optional[float]:
        if tumor_mask is None:
            return None

        brain_mask = self._estimate_visible_brain_mask()
        if brain_mask is None:
            return None

        tumor_area = int((tumor_mask > 0).sum())
        brain_area = int((brain_mask > 0).sum())

        if brain_area <= 0:
            return None

        return 100.0 * float(tumor_area) / float(brain_area)

    def _update_tumor_snapshot(self):
        lbl_size = self._w("labelSizeCategory")
        lbl_shape = self._w("labelShapeCategory")
        lbl_edge = self._w("labelEdgeCategory")
        lbl_text = self._w("labelClinicalImpression")

        mask_use = self.imageLabel.mask_edit if self.imageLabel.mask_edit is not None else self._last_mask
        if mask_use is None or mask_use.size <= 1 or int((mask_use > 0).sum()) == 0:
            self._snapshot_set_default()
            return

        area_px, per_px, circ = self._mask_contour_metrics(mask_use)
        area_mm2 = float(area_px) * (float(self.mm_per_pixel) ** 2)

        if area_mm2 < 50.0:
            size_cat = "🟢 Small"
        elif area_mm2 < 200.0:
            size_cat = "🟡 Medium"
        else:
            size_cat = "🔴 Large"

        if circ >= 0.70:
            shape_cat = "◯ Regular"
        else:
            shape_cat = "⬡ Irregular"

        edge_score = self._edge_sharpness_score(mask_use)
        if edge_score is None:
            edge_cat = "—"
            edge_line = "—"
        else:
            if edge_score >= 0.55:
                edge_cat = "Sharp"
            elif edge_score >= 0.35:
                edge_cat = "Moderate"
            else:
                edge_cat = "Diffuse"
            edge_line = f"{edge_cat} (score {edge_score:.2f})"

        if lbl_size is not None:
            lbl_size.setText(f"{size_cat} ({area_mm2:.1f} mm²)")
        if lbl_shape is not None:
            lbl_shape.setText(f"{shape_cat} (circularity {circ:.2f})")
        if lbl_edge is not None:
            lbl_edge.setText(edge_line)

        parts = []
        parts.append("irregular borders" if "Irregular" in shape_cat else "regular shape")
        if "Small" in size_cat:
            parts.append("small 2D area")
        elif "Medium" in size_cat:
            parts.append("medium 2D area")
        else:
            parts.append("large 2D area")

        if edge_score is not None:
            if edge_cat == "Diffuse":
                parts.append("diffuse boundary appearance")
            elif edge_cat == "Sharp":
                parts.append("sharp boundary appearance")

        recommend_review = ("Irregular" in shape_cat) or ("Large" in size_cat) or (edge_cat in ("Diffuse", "Moderate"))
        if recommend_review:
            msg = (
                f"The lesion shows {', '.join(parts)}, which may warrant closer review. "
                "Manual boundary review is recommended."
            )
        else:
            msg = (
                f"The lesion shows {', '.join(parts)}. "
                "Consider confirming boundaries and measurements before export."
            )

        if lbl_text is not None:
            lbl_text.setText(msg)

    # =====================================================
    # Lesion Review Panel
    # =====================================================
    def _reset_lesion_review_panel(self):
        self._lesions = []
        self._selected_lesion_index = -1

        if hasattr(self, "listLesions"):
            self.listLesions.clear()

        if hasattr(self, "labelLesionLocation"):
            self.labelLesionLocation.setText("-")
        if hasattr(self, "labelLesionArea"):
            self.labelLesionArea.setText("-")
        if hasattr(self, "labelLesionDiameter"):
            self.labelLesionDiameter.setText("-")
        if hasattr(self, "labelLesionShape"):
            self.labelLesionShape.setText("-")
        if hasattr(self, "labelLesionEdge"):
            self.labelLesionEdge.setText("-")
        if hasattr(self, "labelLesionRisk"):
            self.labelLesionRisk.setText("-")
        if hasattr(self, "labelLesionFollowup"):
            self.labelLesionFollowup.setText("-")

        if hasattr(self, "plainTextCaseSummary"):
            self.plainTextCaseSummary.setPlainText("No lesion review available.")
        self._clear_lesion_highlight()

    def _assign_location(self, centroid: Tuple[float, float], shape_hw: Tuple[int, int]) -> str:
        h, w = shape_hw
        cx, cy = centroid

        side = "Left" if cx < (w / 2.0) else "Right"

        if cy < h * 0.33:
            region = "frontal"
        elif cy < h * 0.66:
            region = "parietal"
        else:
            region = "temporal"

        return f"{side} {region}"

    def _compute_lesion_risk(self, lesion: Dict[str, Any], confidence: float) -> str:
        points = 0

        eq_diameter_mm = float(lesion.get("eq_diameter_mm", 0.0))
        circularity = float(lesion.get("circularity", 0.0))
        edge_score = lesion.get("edge_sharpness", None)

        if eq_diameter_mm >= 25.0:
            points += 2
        elif eq_diameter_mm >= 10.0:
            points += 1

        if circularity < 0.70:
            points += 1

        if edge_score is not None and edge_score > 0.60:
            points += 1

        if confidence >= 0.85:
            points += 1

        if points >= 4:
            return "HIGH"
        elif points >= 2:
            return "MEDIUM"
        return "LOW"

    def _generate_followup(self, risk_label: str) -> str:
        if risk_label == "HIGH":
            return "Specialist review recommended."
        elif risk_label == "MEDIUM":
            return "Clinical correlation and follow-up imaging advised."
        return "Routine review suggested."

    def _compute_red_flag(self, lesion: Dict[str, Any]) -> Optional[str]:
        eq_diameter_mm = float(lesion.get("eq_diameter_mm", 0.0))
        circularity = float(lesion.get("circularity", 0.0))
        risk_label = str(lesion.get("risk_label", "LOW"))
        edge_score = lesion.get("edge_sharpness", None)

        irregular = circularity < 0.70
        sharp_or_suspicious = (edge_score is not None and edge_score > 0.60)

        if eq_diameter_mm >= 30.0 and irregular:
            return "RED FLAG: Large irregular lesion."
        if risk_label == "HIGH" and irregular:
            return "RED FLAG: High-risk irregular lesion."
        if eq_diameter_mm >= 25.0 and sharp_or_suspicious and risk_label == "HIGH":
            return "RED FLAG: Suspicious lesion with possible mass effect."
        return None

    def _extract_lesions_from_mask(self, mask01: np.ndarray) -> List[Dict[str, Any]]:
        lesions: List[Dict[str, Any]] = []

        if mask01 is None or mask01.size <= 1:
            return lesions

        m = (mask01 > 0).astype(np.uint8)
        if int(m.sum()) == 0:
            return lesions

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)

        cls_conf = float(self._last_cls[2]) if self._last_cls is not None else 0.0

        for comp_id in range(1, num_labels):
            area_px = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area_px < 20:
                continue

            comp_mask = (labels == comp_id).astype(np.uint8)

            x = int(stats[comp_id, cv2.CC_STAT_LEFT])
            y = int(stats[comp_id, cv2.CC_STAT_TOP])
            w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
            h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])

            cx, cy = centroids[comp_id]
            cx = float(cx)
            cy = float(cy)

            area_px2, perimeter_px, circularity = self._mask_contour_metrics(comp_mask)
            area_mm2 = float(area_px2) * (float(self.mm_per_pixel) ** 2)
            eq_diameter_mm = 2.0 * math.sqrt(area_mm2 / math.pi) if area_mm2 > 0 else 0.0
            edge_score = self._edge_sharpness_score(comp_mask)

            location_text = self._assign_location((cx, cy), comp_mask.shape)
            risk_label = self._compute_lesion_risk(
                {
                    "eq_diameter_mm": eq_diameter_mm,
                    "circularity": circularity,
                    "edge_sharpness": edge_score,
                },
                cls_conf,
            )
            followup = self._generate_followup(risk_label)

            lesion = {
                "id": len(lesions) + 1,
                "component_id": comp_id,
                "bbox": (x, y, w, h),
                "centroid": (cx, cy),
                "area_px": area_px2,
                "area_mm2": area_mm2,
                "eq_diameter_mm": eq_diameter_mm,
                "perimeter_px": perimeter_px,
                "circularity": circularity,
                "edge_sharpness": edge_score,
                "location_text": location_text,
                "risk_label": risk_label,
                "followup": followup,
                "mask": comp_mask,
                "relative_size_percent": self._compute_relative_tumor_size_percent(comp_mask),
            }
            lesions.append(lesion)

        lesions.sort(key=lambda z: z["area_mm2"], reverse=True)
        for i, lesion in enumerate(lesions, start=1):
            lesion["id"] = i

        return lesions

    def _build_case_summary(self, lesions: List[Dict[str, Any]]) -> str:
        if not lesions:
            return "No suspicious lesion detected."

        if len(lesions) == 1:
            l = lesions[0]
            shape_text = "irregular" if l["circularity"] < 0.70 else "regular"
            edge_score = l.get("edge_sharpness", None)

            if edge_score is None:
                edge_text = "uncertain margins"
            elif edge_score > 0.60:
                edge_text = "sharp margins"
            elif edge_score > 0.35:
                edge_text = "moderately defined margins"
            else:
                edge_text = "diffuse margins"

            return (
                f"One suspicious lesion detected in the {l['location_text']} region. "
                f"Estimated equivalent diameter is {l['eq_diameter_mm']:.1f} mm with "
                f"{shape_text} morphology and {edge_text}. "
                f"{l['followup']}"
            )

        largest = lesions[0]
        return (
            f"{len(lesions)} suspicious lesions detected. "
            f"Largest lesion is located in the {largest['location_text']} region "
            f"with estimated diameter {largest['eq_diameter_mm']:.1f} mm. "
            f"{largest['followup']}"
        )

    def _generate_radiology_conclusion(self, lesions: List[Dict[str, Any]]) -> str:
        if not lesions:
            return (
                "No focal suspicious lesion identified on the current 2D AI-assisted review. "
                "Correlation with full MRI series is recommended."
            )

        l = lesions[0]

        location = l.get("location_text", "unknown region")
        diameter = float(l.get("eq_diameter_mm", 0.0))
        circularity = float(l.get("circularity", 0.0))
        edge_score = l.get("edge_sharpness", None)
        risk = l.get("risk_label", "LOW")

        shape_text = "irregular" if circularity < 0.70 else "relatively regular"

        if edge_score is None:
            margin_text = "indeterminate margins"
        elif edge_score > 0.60:
            margin_text = "relatively sharp margins"
        elif edge_score > 0.35:
            margin_text = "moderately defined margins"
        else:
            margin_text = "diffuse margins"

        rel_percent = None
        mask_use = l.get("mask", None)
        if mask_use is not None:
            rel_percent = self._compute_relative_tumor_size_percent(mask_use)

        base = (
            f"AI-assisted 2D review demonstrates a lesion in the {location} region, "
            f"with estimated equivalent diameter of {diameter:.1f} mm, "
            f"{shape_text} morphology, and {margin_text}."
        )

        if rel_percent is not None:
            base += f" The lesion occupies approximately {rel_percent:.2f}% of the visible intracranial area."

        if risk == "HIGH":
            base += " Overall imaging pattern is considered high-risk on this limited 2D assessment."
        elif risk == "MEDIUM":
            base += " Overall imaging pattern is considered intermediate-risk on this limited 2D assessment."
        else:
            base += " Overall imaging pattern is considered low-risk on this limited 2D assessment."

        base += " Final interpretation requires radiologist review and full study correlation."
        return base

    def _show_selected_lesion(self, index: int):
        if not (0 <= index < len(self._lesions)):
            return

        lesion = self._lesions[index]

        circularity = float(lesion.get("circularity", 0.0))
        edge_score = lesion.get("edge_sharpness", None)

        shape_text = "Irregular" if circularity < 0.70 else "Regular"

        if edge_score is None:
            edge_text = "Unknown"
        elif edge_score > 0.60:
            edge_text = f"Sharp ({edge_score:.2f})"
        elif edge_score > 0.35:
            edge_text = f"Moderate ({edge_score:.2f})"
        else:
            edge_text = f"Diffuse ({edge_score:.2f})"

        if hasattr(self, "labelLesionLocation"):
            self.labelLesionLocation.setText(lesion["location_text"])
        if hasattr(self, "labelLesionArea"):
            self.labelLesionArea.setText(f"{lesion['area_mm2']:.1f} mm²")
        if hasattr(self, "labelLesionDiameter"):
            self.labelLesionDiameter.setText(f"{lesion['eq_diameter_mm']:.1f} mm")
        if hasattr(self, "labelLesionShape"):
            self.labelLesionShape.setText(shape_text)
        if hasattr(self, "labelLesionEdge"):
            self.labelLesionEdge.setText(edge_text)
        if hasattr(self, "labelLesionRisk"):
            risk = lesion["risk_label"]
        if hasattr(self, "labelLesionRelativeSize"):
            rel_size = lesion.get("relative_size_percent", None)
            if rel_size is None:
                self.labelLesionRelativeSize.setText("-")
            else:
                self.labelLesionRelativeSize.setText(f"{rel_size:.2f} % of visible brain area")
        self.labelLesionRisk.setText(risk)
        self.labelLesionRisk.setStyleSheet("")

        if risk == "HIGH":
            self.labelLesionRisk.setStyleSheet("color: red; font-weight: bold;")
        elif risk == "MEDIUM":
            self.labelLesionRisk.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.labelLesionRisk.setStyleSheet("color: lightgreen;")
        if hasattr(self, "labelLesionFollowup"):
            self.labelLesionFollowup.setText(lesion["followup"])

        self._draw_lesion_highlight(lesion)
        self._center_view_on_lesion(lesion)

    def _refresh_lesion_review_panel(self):
        mask_use = self.imageLabel.mask_edit if self.imageLabel.mask_edit is not None else self._last_mask

        if mask_use is None or int((mask_use > 0).sum()) == 0:
            self._reset_lesion_review_panel()
            return

        self._lesions = self._extract_lesions_from_mask(mask_use)

       

        if self._lesions:
            self._selected_lesion_index = 0
            self._show_selected_lesion(0)
            if hasattr(self, "listLesions"):
                self.listLesions.setCurrentRow(0)
        else:
            self._selected_lesion_index = -1

        if hasattr(self, "plainTextCaseSummary"):
            summary = self._build_case_summary(self._lesions)

            if self._lesions:
                red_flag = self._compute_red_flag(self._lesions[0])
                conclusion = self._generate_radiology_conclusion(self._lesions)

                text_parts = []
                if red_flag:
                    text_parts.append(f"⚠️ {red_flag}")
                text_parts.append(summary)
                text_parts.append("")
                text_parts.append("Radiology-style conclusion:")
                text_parts.append(conclusion)

                self.plainTextCaseSummary.setPlainText("\n".join(text_parts))
            else:
                self.plainTextCaseSummary.setPlainText(summary)

    def on_lesion_selected(self, item):
        if not hasattr(self, "listLesions"):
            return

        row = self.listLesions.row(item)
        if row < 0:
            return

        self._selected_lesion_index = row
        self._show_selected_lesion(row)

        lesion = self._lesions[row]

        if hasattr(self, "labelLesionFollowup"):
            self.labelLesionFollowup.setText(lesion["followup"])

        self._draw_lesion_highlight(lesion)
        self._center_view_on_lesion(lesion)

    def on_toggle_lesion_box(self, checked: bool):
        self.show_lesion_box = bool(checked)

        if not self.show_lesion_box:
            self._clear_lesion_highlight()
            return

        if 0 <= self._selected_lesion_index < len(self._lesions):
            lesion = self._lesions[self._selected_lesion_index]
            self._draw_lesion_highlight(lesion)

    def _clear_lesion_highlight(self):
        sc = self.imageLabel.scene()
        if sc is None:
            return

        for item in (self._lesion_bbox_item, self._lesion_label_item, self._lesion_contour_item):
            if item is not None:
                try:
                    sc.removeItem(item)
                except Exception:
                    pass

        self._lesion_bbox_item = None
        self._lesion_label_item = None
        self._lesion_contour_item = None

    def _draw_lesion_highlight(self, lesion: Dict[str, Any]):
        if not self.show_lesion_box:
            return

        sc = self.imageLabel.scene()
        if sc is None:
            return

        self._clear_lesion_highlight()

        x, y, w, h = lesion["bbox"]

        pen = QPen(QColor(255, 180, 0, 230))
        pen.setWidth(2)
        self._lesion_bbox_item = sc.addRect(QRectF(float(x), float(y), float(w), float(h)), pen)
        self._lesion_bbox_item.setZValue(60)

        if self._last_cls is not None:
            _, tumor_name, _ = self._last_cls
        else:
            tumor_name = "Tumor"

        label_text = f"{tumor_name} | {lesion['eq_diameter_mm']:.1f} mm"
        self._lesion_label_item = QGraphicsSimpleTextItem(label_text)
        self._lesion_label_item.setBrush(QColor(255, 220, 120))
        self._lesion_label_item.setZValue(61)
        self._lesion_label_item.setPos(float(x), max(0.0, float(y) - 22.0))
        sc.addItem(self._lesion_label_item)

    def _center_view_on_lesion(self, lesion: Dict[str, Any]):
        x, y, w, h = lesion["bbox"]
        cx = float(x + w / 2.0)
        cy = float(y + h / 2.0)
        try:
            self.imageLabel.centerOn(cx, cy)
        except Exception:
            pass

    # =====================================================
    # Menus
    # =====================================================
    def _connect_menus_best_effort(self):
        if hasattr(self, "actionSave"):
            self.actionSave.triggered.connect(self.menu_save)
        if hasattr(self, "actionSave_as"):
            self.actionSave_as.triggered.connect(self.menu_save_as)
        if hasattr(self, "actionExport"):
            self.actionExport.triggered.connect(self.menu_export_pdf_report)
        if hasattr(self, "actionExit"):
            self.actionExit.triggered.connect(self.close)

        if hasattr(self, "actionRun_Classification"):
            self.actionRun_Classification.triggered.connect(self.run_classification)
        if hasattr(self, "actionRun_Segmentation"):
            self.actionRun_Segmentation.triggered.connect(self.run_segmentation)
        if hasattr(self, "actionGenerate_GradCAM"):
            self.actionGenerate_GradCAM.triggered.connect(self.run_gradcam)

        if hasattr(self, "actionIn"):
            self.actionIn.triggered.connect(lambda: self._menu_zoom_step(+10))
        if hasattr(self, "actionOut"):
            self.actionOut.triggered.connect(lambda: self._menu_zoom_step(-10))
        if hasattr(self, "actionReset_View"):
            self.actionReset_View.triggered.connect(self.menu_reset_view)

        if hasattr(self, "actionPan"):
            try:
                self.actionPan.setCheckable(True)
                self.actionPan.setChecked(True)
            except Exception:
                pass
            self.actionPan.triggered.connect(self.menu_toggle_pan)

        if hasattr(self, "actionShow_Heatmap"):
            try:
                self.actionShow_Heatmap.setCheckable(True)
                self.actionShow_Heatmap.setChecked(False)
            except Exception:
                pass
            self.actionShow_Heatmap.triggered.connect(self.menu_toggle_heatmap)

        if hasattr(self, "actionFlip_Horizontal"):
            self.actionFlip_Horizontal.triggered.connect(self.menu_flip_horizontal)
        if hasattr(self, "actionFlip_Vertical"):
            self.actionFlip_Vertical.triggered.connect(self.menu_flip_vertical)
        if hasattr(self, "actionRotate_90_CW"):
            self.actionRotate_90_CW.triggered.connect(lambda: self.menu_rotate_90(+90))
        if hasattr(self, "actionRotate_90_CCW"):
            self.actionRotate_90_CCW.triggered.connect(lambda: self.menu_rotate_90(-90))

    # =====================================================
    # Buttons
    # =====================================================
    def _connect_buttons(self):
        if hasattr(self, "OpenImageButton"):
            self.OpenImageButton.clicked.connect(self.open_image)
        if hasattr(self, "ClassificationButton"):
            self.ClassificationButton.clicked.connect(self.run_classification)
        if hasattr(self, "SegmentationButton"):
            self.SegmentationButton.clicked.connect(self.run_segmentation)
        if hasattr(self, "btnGradCAM"):
            self.btnGradCAM.clicked.connect(self.run_gradcam)
        if hasattr(self, "btnHeatmap"):
            self.btnHeatmap.clicked.connect(self.toggle_heatmap_button)
        if hasattr(self, "pushButtonReset"):
            self.pushButtonReset.clicked.connect(self.on_reset_image_tools)

    # =====================================================
    # Image tools
    # =====================================================
    def _setup_image_tools(self):
        if hasattr(self, "horizontalSliderBrightness"):
            self.horizontalSliderBrightness.setRange(-100, 100)
            self.horizontalSliderBrightness.setValue(0)
            self.horizontalSliderBrightness.valueChanged.connect(self.on_brightness_changed)

        if hasattr(self, "horizontalSliderContrast"):
            self.horizontalSliderContrast.setRange(-100, 100)
            self.horizontalSliderContrast.setValue(0)
            self.horizontalSliderContrast.valueChanged.connect(self.on_contrast_changed)

    def on_brightness_changed(self, value: int):
        self.brightness_value = int(value)
        self._refresh_display_image()

    def on_contrast_changed(self, value: int):
        self.contrast_value = int(value)
        self._refresh_display_image()

    def on_zoom_changed(self, value: int):
        self.zoom_factor = max(0.1, min(3.0, float(value) / 100.0))
        self._apply_view_transform()

    def on_rotate_changed(self, value: int):
        self.rotation_angle = float(value)
        self._apply_view_transform()

    def on_reset_image_tools(self):
        self.brightness_value = 0
        self.contrast_value = 0
        self.zoom_factor = 1.0
        self.rotation_angle = 0.0

        if hasattr(self, "horizontalSliderBrightness"):
            self.horizontalSliderBrightness.setValue(0)
        if hasattr(self, "horizontalSliderContrast"):
            self.horizontalSliderContrast.setValue(0)
        if hasattr(self, "horizontalSliderZoom"):
            self.horizontalSliderZoom.setValue(100)
        if hasattr(self, "dialRotate"):
            self.dialRotate.setValue(0)

        self._refresh_display_image()
        self._apply_view_transform()

    def _apply_view_transform(self):
        try:
            self.imageLabel.resetTransform()
            self.imageLabel.scale(self.zoom_factor, self.zoom_factor)
            if abs(self.rotation_angle) > 0.01:
                self.imageLabel.rotate(self.rotation_angle)
        except Exception:
            pass
        self._update_hud_text()

    # =====================================================
    # Seg UI controls
    # =====================================================
    def _setup_segmentation_ui_controls(self):
        cb = self._w("checkShowOverlay", "checkBoxShowOverlay", "chkShowOverlay")
        if cb is not None:
            cb.setChecked(True)
            cb.toggled.connect(self.on_show_overlay_toggled)

        alpha_slider = self._w("sliderOverlayAlpha", "horizontalSliderOverlayAlpha", "sliderAlpha")
        alpha_spin = self._w("doubleSpinOverlayAlpha", "spinOverlayAlpha", "doubleSpinBoxOverlayAlpha")
        if alpha_slider is not None:
            alpha_slider.setRange(0, 100)
            alpha_slider.setValue(int(self.overlay_alpha * 100))
            alpha_slider.valueChanged.connect(self.on_overlay_alpha_slider_changed)
        if alpha_spin is not None:
            alpha_spin.setRange(0.0, 1.0)
            alpha_spin.setSingleStep(0.05)
            alpha_spin.setValue(float(self.overlay_alpha))
            alpha_spin.valueChanged.connect(self.on_overlay_alpha_spin_changed)

        thr_slider = self._w("sliderSegThr", "horizontalSliderSegThr", "sliderThreshold")
        thr_spin = self._w(
            "doubleSpinSegThr",
            "spinSegThr",
            "doubleSpinBoxSegThr",
            "doubleSpinBoxThreshold",
            "spinThreshold",
        )
        if thr_slider is not None:
            thr_slider.setRange(0, 100)
            thr_slider.setValue(int(self.seg_threshold * 100))
            thr_slider.valueChanged.connect(self.on_thr_slider_changed)
        if thr_spin is not None:
            thr_spin.setRange(0.0, 1.0)
            thr_spin.setSingleStep(0.01)
            thr_spin.setValue(float(self.seg_threshold))
            thr_spin.valueChanged.connect(self.on_thr_spin_changed)

    def on_show_overlay_toggled(self, checked: bool):
        self.show_overlay = bool(checked)

        if self.show_overlay and self.mask_visible:
            self.imageLabel.overlay_alpha = int(self.overlay_alpha * 255)
        else:
            self.imageLabel.overlay_alpha = 0

        self.imageLabel._rebuild_overlay()
        self._render_results_panel()

    def on_overlay_alpha_slider_changed(self, v: int):
        self.overlay_alpha = max(0.0, min(1.0, float(v) / 100.0))
        spin = self._w("doubleSpinOverlayAlpha", "spinOverlayAlpha", "doubleSpinBoxOverlayAlpha")
        if spin is not None:
            spin.blockSignals(True)
            spin.setValue(self.overlay_alpha)
            spin.blockSignals(False)

        if self.show_overlay and self.mask_visible:
            self.imageLabel.overlay_alpha = int(self.overlay_alpha * 255)
        else:
            self.imageLabel.overlay_alpha = 0

        self.imageLabel._rebuild_overlay()

    def on_toggle_growth_overlay(self, checked: bool):
        self.show_growth_overlay = bool(checked)

        if self.statusBar() is not None:
            self.statusBar().showMessage(
                "Peritumoral risk: ON" if checked else "Peritumoral risk: OFF"
            )

        self._refresh_display_image()

    def on_overlay_alpha_spin_changed(self, v: float):
        self.overlay_alpha = max(0.0, min(1.0, float(v)))
        sl = self._w("sliderOverlayAlpha", "horizontalSliderOverlayAlpha", "sliderAlpha")
        if sl is not None:
            sl.blockSignals(True)
            sl.setValue(int(self.overlay_alpha * 100))
            sl.blockSignals(False)

        if self.show_overlay and self.mask_visible:
            self.imageLabel.overlay_alpha = int(self.overlay_alpha * 255)
        else:
            self.imageLabel.overlay_alpha = 0

        self.imageLabel._rebuild_overlay()

    def on_thr_slider_changed(self, v: int):
        self.seg_threshold = max(0.0, min(1.0, float(v) / 100.0))
        spin = self._w(
            "doubleSpinSegThr",
            "spinSegThr",
            "doubleSpinBoxSegThr",
            "doubleSpinBoxThreshold",
            "spinThreshold",
        )
        if spin is not None:
            spin.blockSignals(True)
            spin.setValue(self.seg_threshold)
            spin.blockSignals(False)

        self._apply_threshold_to_mask()

    def on_thr_spin_changed(self, v: float):
        self.seg_threshold = max(0.0, min(1.0, float(v)))
        sl = self._w("sliderSegThr", "horizontalSliderSegThr", "sliderThreshold")
        if sl is not None:
            sl.blockSignals(True)
            sl.setValue(int(self.seg_threshold * 100))
            sl.blockSignals(False)

        self._apply_threshold_to_mask()

    def _apply_threshold_to_mask(self):
        if self._last_prob is None:
            self._render_results_panel()
            return

        new_mask = self._compute_mask_from_prob(self._last_prob, self.seg_threshold)
        self._last_mask = new_mask

        edited = len(self.imageLabel.undo_stack) > 0
        if not edited:
            self.imageLabel.set_masks(new_mask)
        else:
            self.imageLabel.mask_ai = new_mask.copy()
            self.imageLabel._rebuild_overlay()

        self._show_mask_on_label(self.imageLabel.mask_edit if self.imageLabel.mask_edit is not None else new_mask)
        self._render_results_panel()
        self._update_tumor_snapshot()
        self._refresh_lesion_review_panel()

    # =====================================================
    # Manual Edit hookup
    # =====================================================
    def _setup_manual_edit_controls(self):
        view = self.imageLabel

        chk = self._w("checkEnableEditing", "chkEnableEditing", "checkBoxEnableEditing")
        rad_brush = self._w("radBrush", "radioBrush")
        rad_erase = self._w("radErase", "radioErase")
        sl_brush = self._w("sliderBrushSize", "horizontalSliderBrushSize")
        btn_undo = self._w("btnUndo")
        btn_redo = self._w("btnRedo")
        btn_smooth = self._w("btnSmooth")
        btn_reset = self._w("btnResetToAI", "btnResetAI")

        view.set_tool(MaskEditorView.TOOL_BRUSH)
        if rad_brush is not None:
            rad_brush.setChecked(True)

        def set_enabled(flag: bool):
            for w in (rad_brush, rad_erase, sl_brush, btn_undo, btn_redo, btn_smooth, btn_reset):
                if w is not None:
                    w.setEnabled(flag)

        set_enabled(False)
        view.set_edit_enabled(False)
        if chk is not None:
            chk.setChecked(False)

            def on_tog(v: bool):
                view.set_edit_enabled(bool(v))
                ok = bool(v) and (view.mask_edit is not None)
                set_enabled(ok)

            chk.toggled.connect(on_tog)

        if rad_brush is not None:
            rad_brush.toggled.connect(lambda on: on and view.set_tool(MaskEditorView.TOOL_BRUSH))
        if rad_erase is not None:
            rad_erase.toggled.connect(lambda on: on and view.set_tool(MaskEditorView.TOOL_ERASE))

        if sl_brush is not None:
            sl_brush.setRange(1, 80)
            sl_brush.setValue(view.brush_radius)
            sl_brush.valueChanged.connect(view.set_brush_radius)

        if btn_undo is not None:
            btn_undo.clicked.connect(lambda: (view.undo(), self._after_mask_edit()))
        if btn_redo is not None:
            btn_redo.clicked.connect(lambda: (view.redo(), self._after_mask_edit()))
        if btn_smooth is not None:
            btn_smooth.clicked.connect(lambda: (view.smooth_edges("light"), self._after_mask_edit()))
        if btn_reset is not None:
            btn_reset.clicked.connect(lambda: (view.reset_to_ai(), self._after_mask_edit()))

    def _after_mask_edit(self):
        m = self.imageLabel.mask_edit
        if m is not None:
            self._last_mask = m.copy()
            self._show_mask_on_label(m)
        self._render_results_panel()
        self._update_tumor_snapshot()
        self._refresh_lesion_review_panel()

    def on_toggle_mask_visibility(self, checked: bool):
        self.mask_visible = bool(checked)

        if self.mask_visible and self.show_overlay:
            self.imageLabel.overlay_alpha = int(self.overlay_alpha * 255)
        else:
            self.imageLabel.overlay_alpha = 0

        self.imageLabel._rebuild_overlay()

        if self.statusBar() is not None:
            self.statusBar().showMessage(
                "Segmentation mask: ON" if self.mask_visible else "Segmentation mask: OFF"
            )

    # =====================================================
    # Open image
    # =====================================================
    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open MRI Image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All files (*)"
        )
        if not file_path:
            return

        bgr = cv2.imread(file_path)
        if bgr is None:
            QMessageBox.warning(self, "Error", "Failed to load image with cv2.")
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self.current_image_path = file_path
        self._base_bgr = bgr
        self._base_rgb = rgb

        self.zoom_factor = 1.0
        self.rotation_angle = 0.0
        if hasattr(self, "horizontalSliderZoom"):
            self.horizontalSliderZoom.setValue(100)
        if hasattr(self, "dialRotate"):
            self.dialRotate.setValue(0)

        self.mask_visible = True
        if hasattr(self, "checkShowMask"):
            self.checkShowMask.blockSignals(True)
            self.checkShowMask.setChecked(True)
            self.checkShowMask.blockSignals(False)

        self.brightness_value = 0
        self.contrast_value = 0
        if hasattr(self, "horizontalSliderBrightness"):
            self.horizontalSliderBrightness.setValue(0)
        if hasattr(self, "horizontalSliderContrast"):
            self.horizontalSliderContrast.setValue(0)

        self._last_mask = None
        self._last_prob = None
        self._last_cls = None
        self._last_cam = None
        self._last_cam_info = None
        self._growth_map = None
        self.show_heatmap = False
        if hasattr(self, "actionShow_Heatmap"):
            try:
                self.actionShow_Heatmap.setChecked(False)
            except Exception:
                pass

        self.imageLabel.mask_ai = None
        self.imageLabel.mask_edit = None
        self.imageLabel.undo_stack.clear()
        self.imageLabel.redo_stack.clear()
        self.imageLabel._clear_overlay()
        self._clear_lesion_highlight()

        chk = self._w("checkEnableEditing", "chkEnableEditing", "checkBoxEnableEditing")
        if chk is not None:
            chk.blockSignals(True)
            chk.setChecked(False)
            chk.blockSignals(False)
        self.imageLabel.set_edit_enabled(False)

        self._refresh_display_image()

        self.imageLabel.resetTransform()
        self.imageLabel.fitInView(self.imageLabel.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom_factor = 1.0
        self.rotation_angle = 0.0

        self._draw_orientation_letters(force=True)
        self._ensure_crosshair_items()

        if hasattr(self, "maskLabel"):
            self.maskLabel.clear()

        self._render_results_panel()
        self._snapshot_set_default("Load an image and run segmentation to generate a 2D tumor snapshot.")
        self._reset_lesion_review_panel()

        if self.statusBar() is not None:
            self.statusBar().showMessage(f"Loaded: {Path(file_path).name}")

    # =====================================================
    # Display refresh pipeline
    # =====================================================
    def _refresh_display_image(self):
        if self._base_rgb is None:
            return

        rgb = self._base_rgb.astype(np.float32)

        brightness = float(self.brightness_value)
        contrast = float(self.contrast_value)
        contrast_factor = 1.0 + (contrast / 100.0)

        rgb = (rgb - 128.0) * contrast_factor + 128.0 + brightness
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        if self.show_heatmap:
            heat_src = None
            if self._last_cam is not None:
                heat_src = self._last_cam
            elif self._last_prob is not None:
                p = self._last_prob
                if p.ndim == 3:
                    p = p[..., 0]
                heat_src = p

            if heat_src is not None:
                p = np.clip(heat_src.astype(np.float32), 0.0, 1.0)
                heat = (p * 255).astype(np.uint8)
                heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                heat_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
                rgb = cv2.addWeighted(rgb, 0.80, heat_rgb, 0.20, 0)

        if self.show_growth_overlay and self._growth_map is not None:
            g = np.clip(self._growth_map.astype(np.float32), 0.0, 1.0)

            overlay = np.zeros_like(rgb, dtype=np.uint8)

            low_mask = g > 0.20
            mid_mask = g > 0.40
            high_mask = g > 0.65

            overlay[low_mask] = [170, 140, 0]
            overlay[mid_mask] = [220, 150, 0]
            overlay[high_mask] = [230, 70, 20]

            rgb = cv2.addWeighted(rgb, 0.88, overlay, 0.12, 0)

        self._disp_rgb = rgb
        self.imageLabel.set_image(rgb)

        self.imageLabel.overlay_alpha = 0 if not self.show_overlay else int(self.overlay_alpha * 255)
        self.imageLabel._rebuild_overlay()

        self._draw_orientation_letters(force=False)

    # =====================================================
    # Orientation letters
    # =====================================================
    def _clear_orientation_letters(self):
        sc = self.imageLabel.scene()
        for it in self._orient_items:
            try:
                sc.removeItem(it)
            except Exception:
                pass
        self._orient_items.clear()

    def _draw_orientation_letters(self, force: bool = False):
        sc = self.imageLabel.scene()
        if sc is None:
            return
        rect = sc.sceneRect()
        if rect.isNull():
            return

        if self._orient_items and not force:
            return

        self._clear_orientation_letters()

        pad = 8
        color = QColor(255, 255, 255, 220)

        def add(text: str, px: float, py: float):
            t = QGraphicsTextItem(text)
            t.setDefaultTextColor(color)
            t.setZValue(50)
            t.setPos(px, py)
            sc.addItem(t)
            self._orient_items.append(t)

        add("L", rect.left() + pad, rect.center().y() - 10)
        add("R", rect.right() - 18, rect.center().y() - 10)
        add("A", rect.center().x() - 8, rect.top() + pad)
        add("P", rect.center().x() - 8, rect.bottom() - 24)

    # =====================================================
    # Crosshair + HUD + Magnifier
    # =====================================================
    def _ensure_crosshair_items(self):
        if self.crosshair_h is not None and self.crosshair_v is not None:
            return
        sc = self.imageLabel.scene()
        if sc is None:
            return
        rect = sc.sceneRect()
        pen = QPen(QColor(0, 255, 255, 180))
        pen.setWidth(1)
        self.crosshair_h = sc.addLine(rect.left(), 0, rect.right(), 0, pen)
        self.crosshair_v = sc.addLine(0, rect.top(), 0, rect.bottom(), pen)
        self.crosshair_h.setZValue(40)
        self.crosshair_v.setZValue(40)

    def _update_overlays(self, scene_pos: QPointF):
        self.last_scene_pos = scene_pos
        self._ensure_crosshair_items()
        sc = self.imageLabel.scene()
        if sc is None:
            return
        rect = sc.sceneRect()

        if self.crosshair_h is not None:
            self.crosshair_h.setLine(rect.left(), scene_pos.y(), rect.right(), scene_pos.y())
        if self.crosshair_v is not None:
            self.crosshair_v.setLine(scene_pos.x(), rect.top(), scene_pos.x(), rect.bottom())

        intensity_text = "-"
        x = int(scene_pos.x())
        y = int(scene_pos.y())

        if self._disp_rgb is not None:
            h, w, _ = self._disp_rgb.shape
            if 0 <= x < w and 0 <= y < h:
                r, g, b = self._disp_rgb[y, x]
                intensity_text = f"R:{int(r)} G:{int(g)} B:{int(b)}"

                if hasattr(self, "labelMagnifier"):
                    size = 18
                    x0 = max(0, x - size)
                    y0 = max(0, y - size)
                    x1 = min(w, x + size + 1)
                    y1 = min(h, y + size + 1)
                    patch = self._disp_rgb[y0:y1, x0:x1, :]
                    if patch.size > 0:
                        qimg = numpy_rgb_to_qimage(patch)
                        pm = QPixmap.fromImage(qimg).scaled(
                            self.labelMagnifier.width(),
                            self.labelMagnifier.height(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.FastTransformation,
                        )
                        self.labelMagnifier.setPixmap(pm)
                        self.labelMagnifier.setText("")

        if hasattr(self, "labelHUD"):
            self.labelHUD.setText(
                f"X: {x}   Y: {y}\n"
                f"Zoom: {int(self.zoom_factor * 100)}%   Rotate: {int(self.rotation_angle)}°\n"
                f"Intensity: {intensity_text}"
            )

    def _update_hud_text(self):
        if hasattr(self, "labelHUD"):
            self.labelHUD.setText(
                f"X: -   Y: -\n"
                f"Zoom: {int(self.zoom_factor * 100)}%   Rotate: {int(self.rotation_angle)}°\n"
                f"Intensity: -"
            )

    # =====================================================
    # ROI helpers
    # =====================================================
    def _clear_temp_roi_items_only(self):
        sc = self.imageLabel.scene()
        if sc is None:
            return
        if self.roi_item is not None:
            try:
                sc.removeItem(self.roi_item)
            except Exception:
                pass
        self.roi_item = None

        if self.roi_text_item is not None:
            try:
                sc.removeItem(self.roi_text_item)
            except Exception:
                pass
        self.roi_text_item = None

    def _roi_pen(self):
        pen = QPen(QColor(255, 255, 0, 220))
        pen.setWidth(2)
        return pen

    def _roi_text(self, pos: QPointF, text: str):
        sc = self.imageLabel.scene()
        if sc is None:
            return
        if self.roi_text_item is not None:
            try:
                sc.removeItem(self.roi_text_item)
            except Exception:
                pass
            self.roi_text_item = None
        t = QGraphicsTextItem(text)
        t.setDefaultTextColor(QColor(255, 255, 0))
        t.setPos(pos)
        t.setZValue(45)
        sc.addItem(t)
        self.roi_text_item = t

    def _compute_roi_stats_rect(self, rect: QRectF):
        if self._disp_rgb is None:
            return None

        x0 = max(0, int(rect.left()))
        y0 = max(0, int(rect.top()))
        x1 = min(self._disp_rgb.shape[1] - 1, int(rect.right()))
        y1 = min(self._disp_rgb.shape[0] - 1, int(rect.bottom()))
        if x1 <= x0 or y1 <= y0:
            return None

        cropped = self._disp_rgb[y0 : y1 + 1, x0 : x1 + 1, :].astype(np.float32)
        gray = (0.299 * cropped[..., 0] + 0.587 * cropped[..., 1] + 0.114 * cropped[..., 2])

        w = x1 - x0 + 1
        h = y1 - y0 + 1
        mmpp = float(self.mm_per_pixel)
        w_mm = w * mmpp
        h_mm = h * mmpp
        area_px2 = w * h
        area_mm2 = area_px2 * (mmpp**2)

        return {
            "type": "rect",
            "x0": int(x0),
            "y0": int(y0),
            "x1": int(x1),
            "y1": int(y1),
            "w_px": int(w),
            "h_px": int(h),
            "w_mm": float(w_mm),
            "h_mm": float(h_mm),
            "area_px2": int(area_px2),
            "area_mm2": float(area_mm2),
            "mean": float(gray.mean()),
            "std": float(gray.std()),
            "min": float(gray.min()),
            "max": float(gray.max()),
        }

    def _compute_roi_stats_circle(self, center: QPointF, radius: float):
        if self._disp_rgb is None or radius <= 1:
            return None

        h_img, w_img = self._disp_rgb.shape[0], self._disp_rgb.shape[1]
        cx, cy = int(center.x()), int(center.y())
        r = float(radius)
        r_int = int(radius)

        x0 = max(0, cx - r_int)
        y0 = max(0, cy - r_int)
        x1 = min(w_img - 1, cx + r_int)
        y1 = min(h_img - 1, cy + r_int)
        if x1 <= x0 or y1 <= y0:
            return None

        cropped = self._disp_rgb[y0 : y1 + 1, x0 : x1 + 1, :].astype(np.float32)
        gray = (0.299 * cropped[..., 0] + 0.587 * cropped[..., 1] + 0.114 * cropped[..., 2])

        hh, ww = gray.shape
        yy, xx = np.ogrid[0:hh, 0:ww]
        mask = (xx - (cx - x0)) ** 2 + (yy - (cy - y0)) ** 2 <= (r**2)
        vals = gray[mask]
        if vals.size == 0:
            return None

        mmpp = float(self.mm_per_pixel)
        r_mm = r * mmpp
        area_px2 = float(np.pi * (r**2))
        area_mm2 = area_px2 * (mmpp**2)

        return {
            "type": "circle",
            "cx": int(cx),
            "cy": int(cy),
            "r_px": float(r),
            "r_mm": float(r_mm),
            "area_px2": float(area_px2),
            "area_mm2": float(area_mm2),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    def _start_roi_preview(self, scene_pos: QPointF):
        self._clear_temp_roi_items_only()
        sc = self.imageLabel.scene()
        if sc is None:
            return
        pen = self._roi_pen()
        mode = self.roi_mode.lower()

        if mode.startswith("line"):
            self.roi_item = sc.addLine(scene_pos.x(), scene_pos.y(), scene_pos.x(), scene_pos.y(), pen)
        elif mode.startswith("rect"):
            self.roi_item = sc.addRect(QRectF(scene_pos, scene_pos), pen)
        elif mode.startswith("circle"):
            self.roi_item = sc.addEllipse(QRectF(scene_pos, scene_pos), pen)
        else:
            self.roi_item = sc.addRect(QRectF(scene_pos, scene_pos), pen)

        if self.roi_item is not None:
            self.roi_item.setZValue(44)

    def _update_roi_preview(self, scene_pos: QPointF):
        if self.roi_item is None or self.roi_start is None:
            return

        mode = self.roi_mode.lower()
        x0, y0 = self.roi_start.x(), self.roi_start.y()
        x1, y1 = scene_pos.x(), scene_pos.y()

        if isinstance(self.roi_item, QGraphicsLineItem):
            self.roi_item.setLine(x0, y0, x1, y1)
            dist_px = math.hypot(x1 - x0, y1 - y0)
            dist_mm = dist_px * self.mm_per_pixel
            self._roi_text(scene_pos, f"{dist_px:.1f}px | {dist_mm:.1f}mm")
            if hasattr(self, "labelRoiStats"):
                self.labelRoiStats.setText(f"Line: {dist_px:.1f}px | {dist_mm:.1f} mm")
            return

        if mode.startswith("rect") and isinstance(self.roi_item, QGraphicsRectItem):
            rect = QRectF(QPointF(x0, y0), QPointF(x1, y1)).normalized()
            self.roi_item.setRect(rect)
            stats = self._compute_roi_stats_rect(rect)
            if stats and hasattr(self, "labelRoiStats"):
                self.labelRoiStats.setText(
                    f"Rect {stats['w_px']}x{stats['h_px']} px  ({stats['w_mm']:.1f}x{stats['h_mm']:.1f} mm)\n"
                    f"Area: {stats['area_px2']} px²  ({stats['area_mm2']:.1f} mm²)\n"
                    f"mean={stats['mean']:.1f} std={stats['std']:.1f}  min={stats['min']:.1f} max={stats['max']:.1f}"
                )
            return

        if mode.startswith("circle") and isinstance(self.roi_item, QGraphicsEllipseItem):
            center = self.roi_start
            r = max(1.0, math.hypot(scene_pos.x() - center.x(), scene_pos.y() - center.y()))
            rect = QRectF(center.x() - r, center.y() - r, 2 * r, 2 * r)
            self.roi_item.setRect(rect)
            stats = self._compute_roi_stats_circle(center, r)
            if stats and hasattr(self, "labelRoiStats"):
                self.labelRoiStats.setText(
                    f"Circle r={stats['r_px']:.1f} px ({stats['r_mm']:.1f} mm)\n"
                    f"Area: {stats['area_px2']:.1f} px² ({stats['area_mm2']:.1f} mm²)\n"
                    f"mean={stats['mean']:.1f} std={stats['std']:.1f}  min={stats['min']:.1f} max={stats['max']:.1f}"
                )
            return

    def _finalize_roi(self, scene_pos: QPointF):
        if self.roi_item is None or self.roi_start is None:
            self._clear_temp_roi_items_only()
            self.roi_start = None
            return

        stats = None
        mode = self.roi_mode.lower()

        if mode.startswith("line") and isinstance(self.roi_item, QGraphicsLineItem):
            x0, y0 = self.roi_start.x(), self.roi_start.y()
            x1, y1 = scene_pos.x(), scene_pos.y()
            dist_px = math.hypot(x1 - x0, y1 - y0)
            dist_mm = dist_px * self.mm_per_pixel
            stats = {
                "type": "line",
                "length_px": float(dist_px),
                "length_mm": float(dist_mm)
            }

        elif mode.startswith("rect") and isinstance(self.roi_item, QGraphicsRectItem):
            stats = self._compute_roi_stats_rect(self.roi_item.rect())

        elif mode.startswith("circle") and isinstance(self.roi_item, QGraphicsEllipseItem):
            r = self.roi_item.rect().width() / 2.0
            stats = self._compute_roi_stats_circle(self.roi_start, r)

        entry = {
            "mode": self.roi_mode,
            "start": (self.roi_start.x(), self.roi_start.y()),
            "end": (scene_pos.x(), scene_pos.y()),
            "stats": stats,
        }
        self.roi_history.append(entry)

        label = f"{len(self.roi_history)}. {self.roi_mode}"
        if stats:
            if stats.get("type") == "line":
                label = f"{len(self.roi_history)}. Line: {stats['length_mm']:.1f} mm"
            elif stats.get("type") == "rect":
                label = f"{len(self.roi_history)}. Rect {stats['w_mm']:.1f}x{stats['h_mm']:.1f} mm"
            elif stats.get("type") == "circle":
                label = f"{len(self.roi_history)}. Circle r={stats['r_mm']:.1f} mm"

        if hasattr(self, "listRoiHistory"):
            self.listRoiHistory.addItem(QListWidgetItem(label))

        self.roi_start = None

    def on_roi_history_clicked(self, item):
        try:
            if not hasattr(self, "listRoiHistory"):
                return
            row = self.listRoiHistory.row(item)
            if 0 <= row < len(self.roi_history):
                e = self.roi_history[row]
                stats = e.get("stats")

                if self.statusBar() is not None:
                    self.statusBar().showMessage(f"ROI selected: {e.get('mode','?')}")

                if stats and hasattr(self, "labelRoiStats"):
                    if stats.get("type") == "rect":
                        self.labelRoiStats.setText(
                            f"Rect {stats['w_px']}x{stats['h_px']} px  ({stats['w_mm']:.1f}x{stats['h_mm']:.1f} mm)\n"
                            f"Area: {stats['area_px2']} px²  ({stats['area_mm2']:.1f} mm²)\n"
                            f"mean={stats['mean']:.1f} std={stats['std']:.1f}  min={stats['min']:.1f} max={stats['max']:.1f}"
                        )
                    elif stats.get("type") == "circle":
                        self.labelRoiStats.setText(
                            f"Circle r={stats['r_px']:.1f} px ({stats['r_mm']:.1f} mm)\n"
                            f"Area: {stats['area_px2']:.1f} px² ({stats['area_mm2']:.1f} mm²)\n"
                            f"mean={stats['mean']:.1f} std={stats['std']:.1f}  min={stats['min']:.1f} max={stats['max']:.1f}"
                        )
        except Exception:
            pass

    def on_roi_delete(self):
        if not hasattr(self, "listRoiHistory"):
            return
        row = self.listRoiHistory.currentRow()
        if row < 0:
            return
        self.listRoiHistory.takeItem(row)
        if 0 <= row < len(self.roi_history):
            self.roi_history.pop(row)
        if hasattr(self, "labelRoiStats"):
            self.labelRoiStats.setText("ROI: -")

    def on_roi_clear(self):
        self._clear_temp_roi_items_only()
        self.roi_history.clear()
        self.roi_start = None
        if hasattr(self, "listRoiHistory"):
            self.listRoiHistory.clear()
        if hasattr(self, "labelRoiStats"):
            self.labelRoiStats.setText("ROI: -")

    def on_roi_mode_changed(self, text: str):
        self.roi_mode = text
        self._clear_temp_roi_items_only()
        self.roi_start = None
        if hasattr(self, "labelRoiStats"):
            self.labelRoiStats.setText("ROI: -")

    # =====================================================
    # Event filter
    # =====================================================
    def eventFilter(self, obj, event):
        if obj is self.imageLabel.viewport() and self._disp_rgb is not None:
            chk = self._w("checkEnableEditing", "chkEnableEditing", "checkBoxEnableEditing")
            manual_edit = bool(chk.isChecked()) if chk is not None else False

            if event.type() == QEvent.Type.MouseMove:
                scene_pos = self.imageLabel.mapToScene(int(event.position().x()), int(event.position().y()))
                self._update_overlays(scene_pos)

                if not manual_edit and self.roi_start is not None:
                    self._update_roi_preview(scene_pos)
                    return True

            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton and not manual_edit:
                    scene_pos = self.imageLabel.mapToScene(int(event.position().x()), int(event.position().y()))
                    self.roi_start = scene_pos
                    self._start_roi_preview(scene_pos)
                    return True

            if event.type() == QEvent.Type.MouseButtonRelease:
                if (
                    event.button() == Qt.MouseButton.LeftButton
                    and (not manual_edit)
                    and self.roi_start is not None
                ):
                    scene_pos = self.imageLabel.mapToScene(int(event.position().x()), int(event.position().y()))
                    self._finalize_roi(scene_pos)
                    return True

            if event.type() == QEvent.Type.Wheel:
                delta = event.angleDelta().y()
                if delta != 0 and hasattr(self, "horizontalSliderZoom"):
                    step = 10 if delta > 0 else -10
                    new_val = int(self.horizontalSliderZoom.value() + step)
                    new_val = max(self.horizontalSliderZoom.minimum(), min(self.horizontalSliderZoom.maximum(), new_val))
                    self.horizontalSliderZoom.setValue(new_val)
                return False

        return super().eventFilter(obj, event)

    # =====================================================
    # AI
    # =====================================================
    def _ensure_image_loaded(self) -> bool:
        if self.current_image_path is None or self._base_bgr is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return False
        return True

    def _set_buttons_enabled(self, enabled: bool):
        for n in ("SegmentationButton", "ClassificationButton", "OpenImageButton", "btnGradCAM"):
            if hasattr(self, n):
                getattr(self, n).setEnabled(enabled)

    def run_classification(self):
        if not self._ensure_image_loaded():
            return
        if self.engine is None:
            QMessageBox.critical(self, "Error", "Engine nincs betöltve.")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        self._set_buttons_enabled(False)
        if self.statusBar() is not None:
            self.statusBar().showMessage("Running ensemble classification...")

        self._worker = InferenceWorker(self.engine, self.current_image_path, task="cls")
        self._worker.finished.connect(self._on_infer_done)
        self._worker.failed.connect(self._on_infer_fail)
        self._worker.start()

    def run_segmentation(self):
        if not self._ensure_image_loaded():
            return
        if self.engine is None:
            QMessageBox.critical(self, "Error", "Engine nincs betöltve.")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        self._set_buttons_enabled(False)
        if self.statusBar() is not None:
            self.statusBar().showMessage("Running segmentation...")

        self._worker = InferenceWorker(self.engine, self.current_image_path, task="seg")
        self._worker.finished.connect(self._on_infer_done)
        self._worker.failed.connect(self._on_infer_fail)
        self._worker.start()

    def run_gradcam(self):
        if not self._ensure_image_loaded():
            return
        if self.engine is None:
            QMessageBox.critical(self, "Error", "Engine nincs betöltve.")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        self._set_buttons_enabled(False)
        if self.statusBar() is not None:
            self.statusBar().showMessage("Generating Grad-CAM heatmap (ResNet ensemble branch)...")

        self._worker = InferenceWorker(self.engine, self.current_image_path, task="cam")
        self._worker.finished.connect(self._on_infer_done)
        self._worker.failed.connect(self._on_infer_fail)
        self._worker.start()

    def toggle_heatmap_button(self):
        self.show_heatmap = not self.show_heatmap
        if hasattr(self, "actionShow_Heatmap"):
            try:
                self.actionShow_Heatmap.blockSignals(True)
                self.actionShow_Heatmap.setChecked(self.show_heatmap)
                self.actionShow_Heatmap.blockSignals(False)
            except Exception:
                pass
        if self.statusBar() is not None:
            self.statusBar().showMessage("Heatmap: ON" if self.show_heatmap else "Heatmap: OFF")
        self._refresh_display_image()

    def menu_toggle_heatmap(self):
        if hasattr(self, "actionShow_Heatmap"):
            try:
                self.show_heatmap = bool(self.actionShow_Heatmap.isChecked())
            except Exception:
                self.show_heatmap = not self.show_heatmap
        else:
            self.show_heatmap = not self.show_heatmap
        self._refresh_display_image()

    # =====================================================
    # Inference callbacks
    # =====================================================
    def _on_infer_done(self, result: dict):
        self._set_buttons_enabled(True)
        task = result.get("task")

        if task == "cls":
            idx = int(result.get("idx", -1))
            score = float(result.get("score", 0.0))
            raw_name = str(result.get("name", "")).strip()

            if raw_name not in ("", "?", "Unknown"):
                name = raw_name
            else:
                name = self.class_names.get(idx, "Unknown")

            self._last_cls = (idx, name, score)

            if self.statusBar() is not None:
                self.statusBar().showMessage(f"Ensemble tumor type: {name} ({score * 100:.1f}%)")

            self._render_results_panel()
            return

        if task == "seg":
            prob = result.get("prob_map", None)
            mask = result.get("mask", None)

            self._last_prob = prob

            if prob is not None:
                mask01 = self._compute_mask_from_prob(prob, self.seg_threshold)
            else:
                mask01 = (mask > 0).astype(np.uint8) if mask is not None else None

            if mask01 is None:
                QMessageBox.warning(self, "Segmentation", "Nem jött maszk.")
                return

            self._last_mask = mask01
            self._growth_map = self._compute_perilesional_risk_map(mask01)

            self.imageLabel.overlay_alpha = 0 if not self.show_overlay else int(self.overlay_alpha * 255)
            self.imageLabel.set_masks(mask01)

            self._show_mask_on_label(mask01)
            self._render_results_panel()
            self._refresh_display_image()

            self._update_tumor_snapshot()
            self._refresh_lesion_review_panel()

            cov = float((mask01 > 0).mean()) * 100.0
            if self.statusBar() is not None:
                self.statusBar().showMessage(f"Segmentation: coverage {cov:.2f}% (thr={self.seg_threshold:.2f})")
            return

        if task == "cam":
            cam01 = result.get("cam01", None)
            idx = int(result.get("idx", -1))
            score = float(result.get("score", 0.0))
            self._last_cam = cam01
            self._last_cam_info = (idx, score)

            cls_name = self.class_names.get(idx, str(idx))
            if self.statusBar() is not None:
                self.statusBar().showMessage(
                    f"Grad-CAM ready (ResNet branch, class={cls_name}, {score*100:.1f}%)."
                )

            self._refresh_display_image()
            return

        if self.statusBar() is not None:
            self.statusBar().showMessage("Done")

    def _on_infer_fail(self, msg: str):
        self._set_buttons_enabled(True)
        if self.statusBar() is not None:
            self.statusBar().showMessage("Error")
        QMessageBox.critical(self, "Inference error", msg)

    # =====================================================
    # Mask from prob
    # =====================================================
    def _compute_mask_from_prob(self, prob: np.ndarray, thr: float) -> np.ndarray:
        p = prob
        if p is None:
            return np.zeros((1, 1), np.uint8)
        if p.ndim == 3:
            p = p[..., 0]
        p = np.clip(p.astype(np.float32), 0.0, 1.0)
        m = (p >= float(thr)).astype(np.uint8)

        if m.any():
            k = np.ones((3, 3), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if num > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                best = 1 + int(np.argmax(areas))
                m = (labels == best).astype(np.uint8)

        return m

    def _compute_perilesional_risk_map(self, mask01: np.ndarray) -> Optional[np.ndarray]:
        if mask01 is None or mask01.size <= 1:
            return None
        if self._base_bgr is None:
            return None

        m = (mask01 > 0).astype(np.uint8)
        if int(m.sum()) == 0:
            return None

        outer = cv2.dilate(m, np.ones((19, 19), np.uint8), iterations=1)
        inner = cv2.dilate(m, np.ones((7, 7), np.uint8), iterations=1)
        ring = ((outer > 0) & (inner == 0)).astype(np.uint8)

        ring = cv2.morphologyEx(ring, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ring = cv2.morphologyEx(ring, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        if int(ring.sum()) == 0:
            return None

        gray = cv2.cvtColor(self._base_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        mean = cv2.GaussianBlur(gray, (9, 9), 0)
        mean2 = cv2.GaussianBlur(gray * gray, (9, 9), 0)
        local_var = np.maximum(mean2 - mean * mean, 0.0)
        local_std = np.sqrt(local_var)

        if float(local_std.max()) > 1e-6:
            local_std = local_std / float(local_std.max())
        else:
            local_std = np.zeros_like(local_std, dtype=np.float32)

        uncertainty = np.zeros_like(local_std, dtype=np.float32)
        if self._last_prob is not None:
            prob = self._last_prob
            if prob.ndim == 3:
                prob = prob[..., 0]
            prob = np.clip(prob.astype(np.float32), 0.0, 1.0)

            uncertainty = 1.0 - np.abs(prob - 0.5) * 2.0

            if uncertainty.shape != local_std.shape:
                uncertainty = cv2.resize(
                    uncertainty,
                    (local_std.shape[1], local_std.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )

        dist = cv2.distanceTransform((ring > 0).astype(np.uint8), cv2.DIST_L2, 3)
        if float(dist.max()) > 1e-6:
            dist = dist / float(dist.max())
        proximity = 1.0 - dist

        _, _, circularity = self._mask_contour_metrics(m)
        irregularity = np.clip(1.0 - circularity, 0.0, 1.0)

        risk = (
            0.50 * local_std +
            0.30 * uncertainty +
            0.20 * proximity
        )

        risk *= (1.0 + 0.8 * irregularity)
        risk *= ring.astype(np.float32)

        risk = cv2.GaussianBlur(risk, (21, 21), 0)

        if float(risk.max()) > 1e-6:
            risk = risk / float(risk.max())
        else:
            return None

        risk[risk < 0.18] = 0.0

        return risk.astype(np.float32)

    # =====================================================
    # Results panel
    # =====================================================
    def _render_results_panel(self):
        tumor_name = "-"
        conf_pct = 0
        if self._last_cls is not None:
            idx, name, score = self._last_cls
            if (name is None) or (str(name).strip() in ("", "?")):
                name = self.class_names.get(int(idx), "Unknown")
            tumor_name = str(name)
            conf_pct = int(round(max(0.0, min(1.0, float(score))) * 100))

        if hasattr(self, "labelClsResult"):
            self.labelClsResult.setText(f"Tumor type: {tumor_name}")
        if hasattr(self, "progressBar"):
            self.progressBar.setRange(0, 100)
            self.progressBar.setValue(conf_pct)

        seg_text = f"Segmentation: - (thr={self.seg_threshold:.2f})"
        tumor_text = "Tumor: -"

        mask_use = self.imageLabel.mask_edit if self.imageLabel.mask_edit is not None else self._last_mask

        if mask_use is not None:
            cov = float((mask_use > 0).mean()) * 100.0
            area_px = int((mask_use > 0).sum())
            area_mm2 = area_px * (self.mm_per_pixel**2)
            eq_diam_mm = 2.0 * math.sqrt(area_mm2 / math.pi) if area_mm2 > 0 else 0.0
            seg_text = f"Segmentation: coverage={cov:.2f}% (thr={self.seg_threshold:.2f})"
            tumor_text = f"Tumor: area={area_px}px | {area_mm2:.1f} mm² | eq.diam={eq_diam_mm:.1f} mm"

        if hasattr(self, "labelSegResult"):
            self.labelSegResult.setText(seg_text)
        if hasattr(self, "labelTumorStats"):
            self.labelTumorStats.setText(tumor_text)

        model_text = "Model: -"
        if self.engine is not None:
            dev = getattr(getattr(self.engine, "device", None), "type", "cpu")
            cls_name = getattr(self.engine, "cls_model_name", "Classifier")
            seg_name = getattr(self.engine, "seg_model_name", "Seg")
            model_text = f"Cls: {cls_name} | Seg: {seg_name} | thr={self.seg_threshold:.2f} | device={dev}"

        if hasattr(self, "labelModelInfo"):
            self.labelModelInfo.setText(model_text)

    # =====================================================
    # mask preview label
    # =====================================================
    def _show_mask_on_label(self, mask01: Optional[np.ndarray]):
        if not hasattr(self, "maskLabel"):
            return
        if mask01 is None:
            self.maskLabel.clear()
            return

        m = (mask01 > 0).astype(np.uint8) * 255
        bg = np.ones_like(m, dtype=np.uint8) * 30
        vis = np.maximum(bg, m).astype(np.uint8)

        qimg = numpy_gray_to_qimage(vis)
        pm = QPixmap.fromImage(qimg).scaled(
            self.maskLabel.width(),
            self.maskLabel.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.maskLabel.setPixmap(pm)

    # =====================================================
    # Patient DOB -> Age
    # =====================================================
    def on_date_of_birth_changed(self, qdate):
        if not isinstance(qdate, QDate):
            if hasattr(qdate, "date"):
                qdate = qdate.date()
            else:
                qdate = self.dateDateofBirth.date() if hasattr(self, "dateDateofBirth") else QDate.currentDate()

        today = QDate.currentDate()
        age = today.year() - qdate.year()
        if (today.month(), today.day()) < (qdate.month(), qdate.day()):
            age -= 1
        if age < 0:
            age = 0
        if hasattr(self, "lineEditAge"):
            self.lineEditAge.setText(str(age))

    # =====================================================
    # View menu actions
    # =====================================================
    def _menu_zoom_step(self, step_percent: int):
        if hasattr(self, "horizontalSliderZoom"):
            new_val = int(self.horizontalSliderZoom.value() + step_percent)
            new_val = max(self.horizontalSliderZoom.minimum(), min(self.horizontalSliderZoom.maximum(), new_val))
            self.horizontalSliderZoom.setValue(new_val)
        else:
            self.zoom_factor = max(0.1, min(3.0, self.zoom_factor + (step_percent / 100.0)))
            self._apply_view_transform()

    def menu_reset_view(self):
        self.zoom_factor = 1.0
        self.rotation_angle = 0.0
        if hasattr(self, "horizontalSliderZoom"):
            self.horizontalSliderZoom.setValue(100)
        if hasattr(self, "dialRotate"):
            self.dialRotate.setValue(0)
        self.imageLabel.resetTransform()
        try:
            self.imageLabel.fitInView(self.imageLabel.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        except Exception:
            pass
        self._update_hud_text()

    def menu_toggle_pan(self):
        if hasattr(self, "actionPan"):
            try:
                self.pan_enabled = bool(self.actionPan.isChecked())
            except Exception:
                self.pan_enabled = not self.pan_enabled
        else:
            self.pan_enabled = not self.pan_enabled
        if self.statusBar() is not None:
            self.statusBar().showMessage(f"Pan: {'ON' if self.pan_enabled else 'OFF'}")

    # =====================================================
    # Edit actions
    # =====================================================
    def menu_flip_horizontal(self):
        if self._base_rgb is None:
            return
        self._base_rgb = np.flip(self._base_rgb, axis=1).copy()
        if self._base_bgr is not None:
            self._base_bgr = np.flip(self._base_bgr, axis=1).copy()
        self._refresh_display_image()
        self.menu_reset_view()

    def menu_flip_vertical(self):
        if self._base_rgb is None:
            return
        self._base_rgb = np.flip(self._base_rgb, axis=0).copy()
        if self._base_bgr is not None:
            self._base_bgr = np.flip(self._base_bgr, axis=0).copy()
        self._refresh_display_image()
        self.menu_reset_view()

    def menu_rotate_90(self, deg: int):
        if self._base_rgb is None:
            return
        if deg == 90:
            self._base_rgb = np.rot90(self._base_rgb, k=3).copy()
            if self._base_bgr is not None:
                self._base_bgr = np.rot90(self._base_bgr, k=3).copy()
        elif deg == -90:
            self._base_rgb = np.rot90(self._base_rgb, k=1).copy()
            if self._base_bgr is not None:
                self._base_bgr = np.rot90(self._base_bgr, k=1).copy()
        self._refresh_display_image()
        self.menu_reset_view()

    # =====================================================
    # Save image
    # =====================================================
    def _grab_view_qimage(self) -> Optional[QImage]:
        try:
            pm = self.imageLabel.grab()
            return pm.toImage()
        except Exception:
            return None

    def menu_save(self):
        img = self._grab_view_qimage()
        if img is None:
            QMessageBox.warning(self, "Save", "Nincs mit menteni (nincs kép).")
            return
        default_name = "output.png"
        if self.current_image_path:
            p = Path(self.current_image_path)
            default_name = str(p.with_name(p.stem + "_saved.png"))
        file_path, _ = QFileDialog.getSaveFileName(self, "Save", default_name, "PNG (*.png);;JPG (*.jpg *.jpeg)")
        if not file_path:
            return
        ok = img.save(file_path)
        if not ok:
            QMessageBox.critical(self, "Save", "Mentés sikertelen.")
        else:
            if self.statusBar() is not None:
                self.statusBar().showMessage(f"Saved: {file_path}")

    def menu_save_as(self):
        img = self._grab_view_qimage()
        if img is None:
            QMessageBox.warning(self, "Save as", "Nincs mit menteni (nincs kép).")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save as", "", "PNG (*.png);;JPG (*.jpg *.jpeg)")
        if not file_path:
            return
        ok = img.save(file_path)
        if not ok:
            QMessageBox.critical(self, "Save as", "Mentés sikertelen.")
        else:
            if self.statusBar() is not None:
                self.statusBar().showMessage(f"Saved: {file_path}")

    # =====================================================
    # PDF EXPORT
    # =====================================================
    def _get_patient_info(self) -> Dict[str, str]:
        def _get_val(obj_name):
            if hasattr(self, obj_name):
                widget = getattr(self, obj_name)
                if hasattr(widget, "text"):
                    return str(widget.text()).strip() or "N/A"
                if hasattr(widget, "currentText"):
                    return str(widget.currentText()).strip() or "N/A"
            return "N/A"

        def _get_date(obj_name):
            if hasattr(self, obj_name):
                widget = getattr(self, obj_name)
                if hasattr(widget, "date"):
                    return widget.date().toString("MM/dd/yyyy")
            return "N/A"

        return {
            "Name":    _get_val("linePatientName"),
            "ID":      _get_val("linePatientID"),
            "DOB":     _get_date("dateDateofBirth"),
            "Age":     _get_val("lineEditAge"),
            "Gender":  _get_val("comboGender"),
            "Date":    _get_date("dateScanDate"),
        }

    def _results_as_text_lines(self) -> List[str]:
        lines = []
        tumor_name = "Not Detected"
        conf_pct = "0.0%"

        if self._last_cls is not None:
            idx, name, score = self._last_cls
            tumor_name = str(name) if (name and str(name).strip() not in ("", "?")) else self.class_names.get(int(idx), "Unknown")
            conf_pct = f"{float(score) * 100.0:.1f}%"

        lines.append(f"• AI Classification: {tumor_name} (Confidence: {conf_pct})")

        mask_use = self.imageLabel.mask_edit if self.imageLabel.mask_edit is not None else self._last_mask
        if mask_use is not None and mask_use.size > 1:
            area_px = int((mask_use > 0).sum())
            area_mm2 = area_px * (self.mm_per_pixel**2)
            eq_diam = 2.0 * math.sqrt(area_mm2 / math.pi) if area_mm2 > 0 else 0.0

            lines.append(f"• Lesion Surface Area: {area_mm2:.1f} mm²")
            lines.append(f"• Equivalent Diameter: ~{eq_diam:.1f} mm")

            impression = "Significant mass effect detected." if eq_diam > 20 else "Focal small-scale lesion localized."
            lines.append(f"• Impression: {impression}")

            cls_name = getattr(self.engine, "cls_model_name", "Classifier") if self.engine else "Classifier"
            seg_name = getattr(self.engine, "seg_model_name", "Seg") if self.engine else "Seg"
            lines.append(f"• Technical: Cls={cls_name} | Seg={seg_name} | Device: CPU")
        else:
            lines.append("• Segmentation: No significant lesion mask detected.")

        if self._lesions:
            top = self._lesions[0]
            lines.append(f"• Location: {top['location_text']}")

        if self._lesions:
            top = self._lesions[0]

            rel = top.get("relative_size_percent", None)
            if rel is not None:
                lines.append(f"• Relative size: {rel:.2f}% of visible brain area")

            red_flag = self._compute_red_flag(top)
            if red_flag is not None:
                lines.append(f"• Alert: {red_flag}")

            lines.append("• Conclusion: " + self._generate_radiology_conclusion(self._lesions))
        return lines

    def menu_export_pdf_report(self):
        if self._disp_rgb is None:
            QMessageBox.warning(self, "Error", "No image loaded for export.")
            return

        info = self._get_patient_info()
        filename = f"MRI_Report_{info['ID']}.pdf"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Report", filename, "PDF (*.pdf)")

        if file_path:
            try:
                self._build_pdf_report(file_path)
                if self.statusBar():
                    self.statusBar().showMessage(f"Report successfully exported: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to generate PDF: {str(e)}")

    def _build_pdf_report(self, pdf_path: str):
        from reportlab.lib import colors
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib.utils import simpleSplit

        c = canvas.Canvas(pdf_path, pagesize=A4)
        W, H = A4
        margin = 15 * mm
        med_blue = colors.HexColor("#2E5077")
        light_grey_bg = colors.HexColor("#F4F7F9")

        c.setFillColor(med_blue)
        c.rect(0, H - 35*mm, W, 35*mm, fill=True, stroke=False)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 20)
        c.drawString(margin, H - 15*mm, "NEURO-ONCOLOGY ANALYSIS REPORT")
        c.setFont("Helvetica", 10)
        c.drawString(margin, H - 23*mm, "AI-Assisted Diagnostic Suite | Radiology Department")

        c.setFont("Helvetica-Oblique", 9)
        c.drawString(margin, H - 28*mm, "Sequence: T1-weighted MRI")
        c.drawRightString(W - margin, H - 23*mm, f"Generated: {datetime.now().strftime('%Y-%m-%d | %H:%M')}")

        y_ptr = H - 42*mm
        info = self._get_patient_info()
        p_data = [
            ["PATIENT NAME:", info["Name"], "PATIENT ID:", info["ID"]],
            ["DATE OF BIRTH:", info["DOB"], "GENDER:", info["Gender"]],
            ["SCAN DATE:", info["Date"], "REPORT AGE:", info["Age"]]
        ]
        pt = Table(p_data, colWidths=[35*mm, 55*mm, 35*mm, 55*mm])
        pt.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'), ('FONTSIZE', (0,0), (-1,-1), 9),
            ('TEXTCOLOR', (0,0), (0,-1), med_blue), ('TEXTCOLOR', (2,0), (2,-1), med_blue),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'), ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        pt.wrapOn(c, margin, y_ptr)
        pt.drawOn(c, margin, y_ptr - 12*mm)

        y_ptr -= 90*mm
        img_w, img_h = 85*mm, 70*mm

        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(margin, y_ptr + img_h + 2*mm, "FIGURE 1: RENDERED MRI (AI OVERLAY)")
        c.drawString(W - margin - img_w, y_ptr + img_h + 2*mm, "FIGURE 2: SEGMENTATION MASK")

        rendered_q = self.imageLabel.grab().toImage()
        if rendered_q:
            c.setStrokeColor(med_blue)
            c.rect(margin, y_ptr, img_w, img_h, stroke=True)
            ba = QByteArray()
            buf = QBuffer(ba)
            buf.open(QBuffer.OpenModeFlag.WriteOnly)
            rendered_q.save(buf, "PNG")
            c.drawImage(ImageReader(io.BytesIO(bytes(ba))), margin+0.5*mm, y_ptr+0.5*mm, width=img_w-1*mm, height=img_h-1*mm, preserveAspectRatio=True)

            c.setFont("Helvetica-Bold", 10)
            c.setFillColor(colors.white)
            c.drawString(margin + 2*mm, y_ptr + img_h/2, "L")
            c.drawRightString(margin + img_w - 2*mm, y_ptr + img_h/2, "R")

        mask_use = self.imageLabel.mask_edit if self.imageLabel.mask_edit is not None else self._last_mask
        if mask_use is not None:
            m_vis = (mask_use > 0).astype(np.uint8) * 255
            mask_q = numpy_gray_to_qimage(np.maximum(np.ones_like(m_vis)*30, m_vis))
            ba_m = QByteArray()
            buf_m = QBuffer(ba_m)
            buf_m.open(QBuffer.OpenModeFlag.WriteOnly)
            mask_q.save(buf_m, "PNG")
            c.rect(W - margin - img_w, y_ptr, img_w, img_h, stroke=True)
            c.drawImage(ImageReader(io.BytesIO(bytes(ba_m))), W-margin-img_w+0.5*mm, y_ptr+0.5*mm, width=img_w-1*mm, height=img_h-1*mm, preserveAspectRatio=True)

        y_ptr -= 82*mm

        res_lines = self._results_as_text_lines()

        line_gap = 5 * mm
        max_text_width = W - 2 * margin - 20 * mm

        wrapped_count = 0
        for line in res_lines:
            wrapped_count += len(simpleSplit(line, "Helvetica", 10, max_text_width))

        box_height = 18 * mm + wrapped_count * line_gap
        c.setFillColor(light_grey_bg)
        c.rect(margin, y_ptr, W - 2 * margin, box_height, fill=True, stroke=False)

        c.setStrokeColor(med_blue)
        c.setLineWidth(0.4)
        c.rect(margin, y_ptr, W - 2 * margin, box_height, fill=False, stroke=True)

        c.setFillColor(med_blue)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin + 5 * mm, y_ptr + box_height - 8 * mm, "CLINICAL IMPRESSION & AI FINDINGS")

        c.setFillColor(colors.black)
        c.setFont("Helvetica", 10)

        curr_text_y = y_ptr + box_height - 16 * mm

        for line in res_lines:
            wrapped_lines = simpleSplit(line, "Helvetica", 10, max_text_width)
            for subline in wrapped_lines:
                c.drawString(margin + 10 * mm, curr_text_y, subline)
                curr_text_y -= line_gap

        table_y = y_ptr - 8 * mm
        if self.roi_history:
            row_count = len(self.roi_history) + 1
            table_height = row_count * 8 * mm
            needed_height = table_height + 25 * mm

            if y_ptr - needed_height < 20 * mm:
                c.showPage()
                y_ptr = H - 35 * mm

            y_ptr -= 7 * mm
            c.setFillColor(med_blue)
            c.setFont("Helvetica-Bold", 11)
            c.drawString(margin, y_ptr, "QUANTITATIVE ROI MEASUREMENTS")

            table_data = [["#", "TYPE", "DIMENSIONS / AREA", "INTENSITY (MEAN)"]]
            for i, e in enumerate(self.roi_history, start=1):
                st = e.get("stats")
                m_type = e.get("mode", "ROI")
                if st:
                    if st.get("type") == "line":
                        meas = f"Length: {st['length_mm']:.2f} mm"
                        val = "-"
                    elif st.get("type") == "rect":
                        meas = f"{st['w_mm']:.1f}x{st['h_mm']:.1f} mm ({st['area_mm2']:.1f} mm²)"
                        val = f"{st['mean']:.1f}"
                    elif st.get("type") == "circle":
                        meas = f"Radius: {st['r_mm']:.1f} mm ({st['area_mm2']:.1f} mm²)"
                        val = f"{st['mean']:.1f}"
                    else:
                        meas, val = "N/A", "-"
                    table_data.append([i, m_type, meas, val])

            rt = Table(table_data, colWidths=[12*mm, 35*mm, 88*mm, 45*mm])
            rt.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), med_blue),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 8),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_grey_bg]),
                ('LINEBELOW', (0,0), (-1,0), 1.5, med_blue),
            ]))

            th = len(table_data) * 7.5 * mm
            table_y = y_ptr - 3 * mm - th

            rt.wrapOn(c, margin, table_y)
            rt.drawOn(c, margin, table_y)

            c.setFont("Helvetica-Oblique", 7)
            c.setFillColor(colors.black)
            c.drawString(
                margin,
                table_y - 5 * mm,
                "*Intensity values are derived from raw T1-weighted signal units."
            )

            y_ptr = table_y - 8 * mm
        else:
            y_ptr -= 8 * mm

        sig_y = table_y - 15 * mm

        c.setStrokeColor(colors.black)
        c.setLineWidth(0.5)
        c.line(W - margin - 60*mm, sig_y + 5*mm, W - margin, sig_y + 5*mm)

        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(colors.black)
        c.drawRightString(W - margin, sig_y, "Authorized Radiologist Signature")

        c.setFont("Helvetica-Oblique", 7)
        c.setFillColor(colors.grey)
        c.drawCentredString(
            W / 2,
            10 * mm,
            "NOTICE: AI-generated research support report. Final clinical validation by a radiologist is mandatory."
        )

        c.showPage()
        c.save()

    # =====================================================
    # Close
    # =====================================================
    def closeEvent(self, event):
        if getattr(self, "engine", None) is not None:
            try:
                if hasattr(self.engine, "unload_all"):
                    self.engine.unload_all()
            except Exception:
                pass
        event.accept()


# =====================================================
# main
# =====================================================
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
