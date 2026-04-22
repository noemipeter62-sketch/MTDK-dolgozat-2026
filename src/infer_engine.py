# infer_engine.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import (
    ResNet50_Weights,
    DenseNet121_Weights,
    EfficientNet_B3_Weights,
)

from src.attention_unet import AttentionUNet


# =====================================================
# Grad-CAM helper
# =====================================================
class _GradCAM:
    """
    Minimal Grad-CAM helper for CNN classification models.
    Captures activations + gradients from a target conv block.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def close(self):
        try:
            self.h1.remove()
        except Exception:
            pass
        try:
            self.h2.remove()
        except Exception:
            pass


# =====================================================
# Fourier layer
# =====================================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1=8, modes2=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)

        self.w1_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2))
        self.w1_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2))
        self.w2_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2))
        self.w2_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2))

    def forward(self, x):
        batchsize = x.shape[0]

        w1 = torch.complex(self.w1_real, self.w1_imag)
        w2 = torch.complex(self.w2_real, self.w2_imag)

        x_f32 = x.float()
        x_ft = torch.fft.rfft2(x_f32)

        m1 = min(self.modes1, x_ft.size(-2))
        m2 = min(self.modes2, x_ft.size(-1))

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x_f32.size(-2),
            x_f32.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :m1, :m2],
            w1[:, :, :m1, :m2],
        )

        out_ft[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -m1:, :m2],
            w2[:, :, :m1, :m2],
        )

        x = torch.fft.irfft2(out_ft, s=(x_f32.size(-2), x_f32.size(-1)))
        return x.to(x_f32.dtype)


# =====================================================
# Classification models
# NOTE:
# ResNet50Fourier intentionally uses self.encoder to stay
# compatible with your saved checkpoint key names.
# =====================================================
class ResNet50Fourier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.encoder = nn.Sequential(*list(base.children())[:-2])

        self.reduce = nn.Conv2d(2048, 512, kernel_size=1)

        self.fourier = nn.Sequential(
            SpectralConv2d(512, 512, modes1=8, modes2=8),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.reduce(x)
        x = self.fourier(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class DenseNet121Fourier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.features = base.features

        self.reduce = nn.Conv2d(1024, 256, kernel_size=1)

        self.fourier = nn.Sequential(
            SpectralConv2d(256, 256, modes1=8, modes2=8),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)
        x = self.reduce(x)
        x = self.fourier(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class EfficientNetB3Fourier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.features = base.features

        self.reduce = nn.Conv2d(1536, 384, kernel_size=1)

        self.fourier = nn.Sequential(
            SpectralConv2d(384, 384, modes1=8, modes2=8),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(384, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.reduce(x)
        x = self.fourier(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# =====================================================
# Infer Engine
# =====================================================
class InferEngine:
    def __init__(self, root_dir: Path, device: str = "cpu", debug: bool = False):
        self.root_dir = Path(root_dir)
        self.device = torch.device(device)
        self.debug = bool(debug)

        # -------------------------
        # MODEL PATHS
        # -------------------------
        self.ensemble_path = self.root_dir / "models" / "fourier_ensemble_bundle.pth"
        self.att_unet_path = self.root_dir / "models" / "att_unet_best.pt"

        # -------------------------
        # MODELS
        # -------------------------
        self.resnet_model: Optional[nn.Module] = None
        self.densenet_model: Optional[nn.Module] = None
        self.efficientnet_model: Optional[nn.Module] = None
        self.seg_model: Optional[nn.Module] = None

        # UI info
        self.cls_model_name: str = "Fourier Ensemble"
        self.seg_model_name: str = "AttentionUNet"

        # Grad-CAM target layer
        self.cls_target_layer: Optional[nn.Module] = None

        # -------------------------
        # CLASSIFICATION
        # -------------------------
        self.cls_size_224 = (224, 224)
        self.cls_size_300 = (300, 300)
        self.class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

        self.cls_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # -------------------------
        # SEGMENTATION
        # -------------------------
        self.seg_size = (256, 256)
        self.threshold = 0.50
        self.keep_largest_component = True

        # overlay
        self.overlay_alpha = 0.30
        self.overlay_color_bgr = (0, 255, 0)

    # =====================================================
    # utils
    # =====================================================
    def _log(self, *args):
        if self.debug:
            print("[InferEngine]", *args)

    def _to_state_dict(self, ckpt):
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        if isinstance(ckpt, dict):
            ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        return ckpt

    def _cuda_cleanup(self):
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                self._log("CUDA cleanup error:", e)

    def gpu_memory_mb(self) -> Tuple[float, float]:
        if not torch.cuda.is_available():
            return 0.0, 0.0
        alloc = float(torch.cuda.memory_allocated() / (1024**2))
        reserv = float(torch.cuda.memory_reserved() / (1024**2))
        return alloc, reserv

    # =====================================================
    # LOAD
    # =====================================================
    def load(self):
        # ---------------- Classification ensemble ----------------
        if not self.ensemble_path.exists():
            raise FileNotFoundError(f"Missing ensemble checkpoint: {self.ensemble_path}")

        ckpt = torch.load(self.ensemble_path, map_location=self.device)

        self.class_names = ckpt.get("class_names", self.class_names)
        num_classes = int(ckpt.get("num_classes", len(self.class_names)))

        self.resnet_model = ResNet50Fourier(num_classes=num_classes)
        self.densenet_model = DenseNet121Fourier(num_classes=num_classes)
        self.efficientnet_model = EfficientNetB3Fourier(num_classes=num_classes)

        self.resnet_model.load_state_dict(self._to_state_dict(ckpt["resnet_state_dict"]), strict=True)
        self.densenet_model.load_state_dict(self._to_state_dict(ckpt["densenet_state_dict"]), strict=True)
        self.efficientnet_model.load_state_dict(self._to_state_dict(ckpt["efficientnet_state_dict"]), strict=True)

        self.resnet_model.to(self.device).eval()
        self.densenet_model.to(self.device).eval()
        self.efficientnet_model.to(self.device).eval()
        # IMPORTANT:
        # Your ResNet notebook uses Grad-CAM on model.reduce
        
        self.cls_target_layer = self.resnet_model.encoder[-1]

        # ---------------- Segmentation ----------------
        if not self.att_unet_path.exists():
            raise FileNotFoundError(
                f"Missing Attention U-Net weights: {self.att_unet_path}\n"
                f"Tip: rename your file to: models/att_unet_best.pt"
            )

        ckpt2 = torch.load(self.att_unet_path, map_location=self.device)
        state2 = self._to_state_dict(ckpt2)

        if not isinstance(state2, dict):
            raise RuntimeError("Attention U-Net checkpoint incompatible (not a state_dict).")

        self.seg_model = AttentionUNet(in_channels=1, out_channels=1)
        self.seg_model.load_state_dict(state2, strict=True)
        self.seg_model.to(self.device).eval()

        self._log("Models loaded OK.")
        if torch.cuda.is_available():
            a, r = self.gpu_memory_mb()
            self._log(f"GPU mem after load: allocated={a:.1f}MB reserved={r:.1f}MB")

    # =====================================================
    # UNLOAD / RELEASE MEMORY
    # =====================================================
    def unload_segmentation_model(self):
        if self.seg_model is not None:
            try:
                self.seg_model.to("cpu")
            except Exception:
                pass
            try:
                del self.seg_model
            except Exception:
                pass
            self.seg_model = None
        self._cuda_cleanup()
        self._log("Segmentation model unloaded.")

    def unload_classification_model(self):
        for attr in ["resnet_model", "densenet_model", "efficientnet_model"]:
            model = getattr(self, attr, None)
            if model is not None:
                try:
                    model.to("cpu")
                except Exception:
                    pass
                try:
                    del model
                except Exception:
                    pass
                setattr(self, attr, None)

        self.cls_target_layer = None
        self._cuda_cleanup()
        self._log("Classification ensemble unloaded.")

    def unload_all(self):
        self.unload_segmentation_model()
        self.unload_classification_model()
        self._cuda_cleanup()

    def release_memory(self):
        self._cuda_cleanup()
        self._log("Released cached GPU memory.")

    # =====================================================
    # CLASSIFICATION
    # =====================================================
    @torch.inference_mode()
    def classify(self, img_bgr: np.ndarray) -> Tuple[int, str, float]:
        if self.resnet_model is None or self.densenet_model is None or self.efficientnet_model is None:
            raise RuntimeError("Classification ensemble not loaded. Call load().")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_224 = cv2.resize(img_rgb, self.cls_size_224, interpolation=cv2.INTER_AREA)
        img_300 = cv2.resize(img_rgb, self.cls_size_300, interpolation=cv2.INTER_AREA)

        x224 = self.cls_tf(img_224).unsqueeze(0).to(self.device)
        x300 = self.cls_tf(img_300).unsqueeze(0).to(self.device)

        res_logits = self.resnet_model(x224)
        den_logits = self.densenet_model(x224)
        eff_logits = self.efficientnet_model(x300)

        res_probs = F.softmax(res_logits, dim=1)
        den_probs = F.softmax(den_logits, dim=1)
        eff_probs = F.softmax(eff_logits, dim=1)

        avg_probs = (res_probs + den_probs + eff_probs) / 3.0
        prob = avg_probs[0]

        idx = int(torch.argmax(prob).item())
        score = float(prob[idx].item())
        name = self.class_names[idx] if 0 <= idx < len(self.class_names) else str(idx)

        return idx, name, score

    # =====================================================
    # GRAD-CAM
    # =====================================================
    def gradcam(self, img_bgr: np.ndarray, class_idx: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        """
        Grad-CAM is computed on the ResNet50 Fourier branch.
        The chosen class index comes from the ensemble prediction unless provided.

        Returns:
            cam01: float32 (H,W) values in [0..1] resized to original image size
            idx: used class index
            score: softmax probability for idx from ResNet branch
        """
        if (
            self.resnet_model is None
            or self.densenet_model is None
            or self.efficientnet_model is None
            or self.cls_target_layer is None
        ):
            raise RuntimeError("Classification model/target layer not ready. Call load().")

        self.resnet_model.eval()
        self.densenet_model.eval()
        self.efficientnet_model.eval()

        # preprocess for ensemble
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_224 = cv2.resize(img_rgb, self.cls_size_224, interpolation=cv2.INTER_AREA)
        img_300 = cv2.resize(img_rgb, self.cls_size_300, interpolation=cv2.INTER_AREA)

        x224 = self.cls_tf(img_224).unsqueeze(0).to(self.device)
        x300 = self.cls_tf(img_300).unsqueeze(0).to(self.device)

        # ensemble class decision
        with torch.no_grad():
            res_logits_e = self.resnet_model(x224)
            den_logits_e = self.densenet_model(x224)
            eff_logits_e = self.efficientnet_model(x300)

            res_probs_e = F.softmax(res_logits_e, dim=1)
            den_probs_e = F.softmax(den_logits_e, dim=1)
            eff_probs_e = F.softmax(eff_logits_e, dim=1)

            avg_probs = (res_probs_e + den_probs_e + eff_probs_e) / 3.0
            ensemble_prob = avg_probs[0]

        if class_idx is None:
            idx = int(torch.argmax(ensemble_prob).item())
        else:
            idx = int(class_idx)

        # Grad-CAM on ResNet branch
        x = x224.clone().detach().to(self.device)

        cam_engine = _GradCAM(self.resnet_model, self.cls_target_layer)

        try:
            with torch.enable_grad():
                x.requires_grad_(True)

                logits = self.resnet_model(x)
                prob = F.softmax(logits, dim=1)[0]

                score = float(prob[idx].item())

                loss = logits[0, idx]
                self.resnet_model.zero_grad(set_to_none=True)
                loss.backward(retain_graph=False)

                A = cam_engine.activations
                G = cam_engine.gradients

                if A is None or G is None:
                    raise RuntimeError("Grad-CAM failed: missing activations/gradients from hooks.")

                weights = G.mean(dim=(2, 3), keepdim=True)
                cam = (weights * A).sum(dim=1, keepdim=False)
                cam = F.relu(cam)[0]

                cam_np = cam.detach().float().cpu().numpy()
                cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

                # cleaner heatmap
                cam_np[cam_np < 0.35] = 0.0

        finally:
            cam_engine.close()

        H0, W0 = img_bgr.shape[:2]
        cam01 = cv2.resize(cam_np, (W0, H0), interpolation=cv2.INTER_LINEAR)
        cam01 = np.clip(cam01, 0.0, 1.0).astype(np.float32)

        return cam01, idx, score

    # =====================================================
    # SEGMENTATION
    # =====================================================
    @torch.inference_mode()
    def segment(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          mask: uint8 (H,W) 0/1
          prob_map: float32 (H,W) [0..1]
        """
        if self.seg_model is None:
            raise RuntimeError("Attention U-Net not loaded. Call load().")

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        H0, W0 = gray.shape[:2]

        x = cv2.resize(gray, self.seg_size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        xt = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(self.device)

        logits = self.seg_model(xt)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
            raise RuntimeError(f"Bad seg output: {type(logits)} shape={getattr(logits, 'shape', None)}")

        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)

        prob_map = cv2.resize(prob, (W0, H0), interpolation=cv2.INTER_LINEAR)
        prob_map = np.clip(prob_map, 0.0, 1.0).astype(np.float32)

        thr = float(self.threshold)
        mask = (prob_map >= thr).astype(np.uint8)

        if mask.any():
            k = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

        if self.keep_largest_component and mask.any():
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if n > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                best = 1 + int(np.argmax(areas))
                mask = (labels == best).astype(np.uint8)

        if self.debug:
            self._log(
                "SEG(att):",
                "prob_max=", float(prob_map.max()),
                "p99=", float(np.percentile(prob_map, 99)),
                "cov%=", float(mask.mean()) * 100.0,
                "thr=", thr,
            )

        return mask, prob_map

    # =====================================================
    # OVERLAY
    # =====================================================
    def overlay_mask(self, img_bgr: np.ndarray, mask: Optional[np.ndarray], prob_map: Optional[np.ndarray] = None):
        out = img_bgr.copy()
        if mask is None:
            return out

        m = (mask > 0).astype(np.uint8)
        if float(m.mean()) <= 0.00001:
            return out

        color = np.zeros_like(out, dtype=np.uint8)
        color[m == 1] = self.overlay_color_bgr

        a = float(np.clip(self.overlay_alpha, 0.0, 1.0))
        out = cv2.addWeighted(out, 1.0 - a, color, a, 0)

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, self.overlay_color_bgr, 2)

        return out
