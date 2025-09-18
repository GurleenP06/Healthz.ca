# YOLO-Barcode Detection Module
# This will be imported into the main scanner

import torch
import torch.nn as nn
import numpy as np
import cv2
from pyzbar import pyzbar
from typing import List, Optional, Tuple
from pathlib import Path
import logging

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

# Import the classes we need
from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple
from enum import Enum
from pathlib import Path

class BarcodeType(Enum):
    """Barcode type classifications"""
    EAN13 = "EAN13"
    EAN8 = "EAN8"
    UPCA = "UPCA"
    UPCE = "UPCE"
    CODE128 = "CODE128"
    CODE39 = "CODE39"
    QR = "QR"
    DATAMATRIX = "DATAMATRIX"
    PDF417 = "PDF417"
    UNKNOWN = "UNKNOWN"

@dataclass
class BarcodeDetection:
    """Barcode detection result"""
    barcode_data: str
    barcode_type: BarcodeType
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    orientation: float  # Rotation angle
    image_quality_score: float
    timestamp: datetime = field(default_factory=datetime.now)

class Config:
    """Central configuration for the system"""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CONFIDENCE_THRESHOLD = 0.5
    INPUT_SIZE = 640

logger = logging.getLogger(__name__)

class YOLOBarcodeDetector:
    """
    State-of-the-art barcode detection using YOLO architecture
    Based on research: YOLO-Barcode and MGL-YOLO approaches
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.device = Config.DEVICE
        self.model = None
        self.transforms = self._get_transforms()
        
        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            self.initialize_yolo_barcode()
    
    def initialize_yolo_barcode(self):
        """Initialize YOLO model for barcode detection"""
        try:
            if YOLO_AVAILABLE:
                # Use YOLOv8 as base (can be customized for barcode-specific architecture)
                self.model = YOLO('yolov8n.pt')  # Start with nano version for speed
                
                # Customize for barcode detection
                self.model.model.names = {
                    0: 'barcode_1d',
                    1: 'qr_code',
                    2: 'datamatrix',
                    3: 'pdf417'
                }
                
                logger.info("Initialized YOLO-Barcode model")
            else:
                logger.warning("YOLO not available, using fallback detection")
                self.model = None
                
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            # Fallback to custom implementation
            self.model = self._build_custom_yolo_barcode()
    
    def _build_custom_yolo_barcode(self):
        """
        Build custom YOLO-Barcode architecture
        Implements improvements from MGL-YOLO paper:
        - Multi-scale feature extraction
        - Attention mechanisms for barcode patterns
        - Optimized for elongated 1D barcodes
        """
        
        class YOLOBarcodeNet(nn.Module):
            def __init__(self, num_classes=4):
                super().__init__()
                
                # Backbone: Modified CSPDarknet for barcode features
                self.backbone = torchvision.models.mobilenet_v3_small(pretrained=True)
                
                # Neck: Feature Pyramid Network with attention
                self.fpn = nn.ModuleList([
                    nn.Conv2d(576, 256, 1),
                    nn.Conv2d(256, 128, 1),
                    nn.Conv2d(128, 64, 1)
                ])
                
                # Multi-scale detection heads
                self.detection_heads = nn.ModuleList([
                    self._make_detection_head(256, num_classes),
                    self._make_detection_head(128, num_classes),
                    self._make_detection_head(64, num_classes)
                ])
                
                # Barcode-specific attention module
                self.barcode_attention = nn.Sequential(
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.Sigmoid()
                )
            
            def _make_detection_head(self, in_channels, num_classes):
                """Create detection head for specific scale"""
                return nn.Sequential(
                    nn.Conv2d(in_channels, in_channels * 2, 3, padding=1),
                    nn.BatchNorm2d(in_channels * 2),
                    nn.ReLU(),
                    nn.Conv2d(in_channels * 2, 5 + num_classes, 1)  # 5 = x,y,w,h,conf
                )
            
            def forward(self, x):
                # Extract features
                features = self.backbone.features(x)
                
                # Apply FPN
                fpn_features = []
                for fpn_layer in self.fpn:
                    features = fpn_layer(features)
                    fpn_features.append(features)
                
                # Apply attention to enhance barcode features
                fpn_features[0] = fpn_features[0] * self.barcode_attention(fpn_features[0])
                
                # Detection outputs
                outputs = []
                for feat, head in zip(fpn_features, self.detection_heads):
                    outputs.append(head(feat))
                
                return outputs
        
        return YOLOBarcodeNet().to(self.device)
    
    def _get_transforms(self):
        """Get image augmentation transforms for training and inference"""
        if ALBUMENTATIONS_AVAILABLE:
            return A.Compose([
                A.Resize(Config.INPUT_SIZE, Config.INPUT_SIZE),
                A.OneOf([
                    A.GaussNoise(p=0.5),
                    A.GaussianBlur(p=0.5),
                    A.MotionBlur(p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.OpticalDistortion(p=0.5),
                    A.GridDistortion(p=0.5),
                    A.PiecewiseAffine(p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.CLAHE(clip_limit=2, p=0.5),
                    A.Sharpen(p=0.5),
                    A.Emboss(p=0.5),
                ], p=0.3),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        return None
    
    def detect_barcodes(self, image: np.ndarray, enhance: bool = True) -> List[BarcodeDetection]:
        """
        Detect barcodes in image using YOLO-Barcode
        
        Args:
            image: Input image as numpy array
            enhance: Whether to apply image enhancement
        
        Returns:
            List of detected barcodes with locations and confidence
        """
        detections = []
        
        # Enhance image for better detection if needed
        if enhance:
            image = self._enhance_image_for_barcode(image)
        
        # YOLO detection
        if self.model and YOLO_AVAILABLE:
            try:
                results = self.model(image, conf=Config.CONFIDENCE_THRESHOLD)
                
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = box.conf[0].item()
                            cls = int(box.cls[0].item())
                            
                            # Extract barcode region
                            barcode_region = image[int(y1):int(y2), int(x1):int(x2)]
                            
                            # Decode barcode
                            barcode_data = self._decode_barcode_region(barcode_region)
                            
                            if barcode_data:
                                detection = BarcodeDetection(
                                    barcode_data=barcode_data,
                                    barcode_type=self._classify_barcode_type(barcode_data),
                                    bounding_box=(int(x1), int(y1), int(x2-x1), int(y2-y1)),
                                    confidence=conf,
                                    orientation=self._estimate_orientation(barcode_region),
                                    image_quality_score=self._assess_quality(barcode_region)
                                )
                                detections.append(detection)
            except Exception as e:
                logger.error(f"YOLO detection failed: {e}")
        
        # Fallback to traditional detection if YOLO fails
        if not detections:
            detections = self._fallback_detection(image)
        
        return detections
    
    def _enhance_image_for_barcode(self, image: np.ndarray) -> np.ndarray:
        """
        Apply advanced image enhancement for barcode detection
        Implements techniques from the research papers
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Morphological operations to enhance barcode patterns
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # Adaptive thresholding for different lighting conditions
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise while preserving edges
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # Convert back to 3-channel for YOLO
        if len(image.shape) == 3:
            enhanced_color = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        else:
            enhanced_color = denoised
        
        return enhanced_color
    
    def _decode_barcode_region(self, region: np.ndarray) -> Optional[str]:
        """Decode barcode from image region using multiple methods"""
        # Try pyzbar first
        barcodes = pyzbar.decode(region)
        if barcodes:
            return barcodes[0].data.decode('utf-8')
        
        # Try different preprocessing techniques
        for preprocess in [
            lambda x: x,  # Original
            lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
            lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),
            lambda x: cv2.flip(x, 0),  # Vertical flip
            lambda x: cv2.flip(x, 1),  # Horizontal flip
        ]:
            processed = preprocess(region)
            barcodes = pyzbar.decode(processed)
            if barcodes:
                return barcodes[0].data.decode('utf-8')
        
        return None
    
    def _classify_barcode_type(self, barcode_data: str) -> BarcodeType:
        """Classify barcode type based on data pattern"""
        if not barcode_data:
            return BarcodeType.UNKNOWN
        
        # Check length and pattern for common types
        if len(barcode_data) == 13 and barcode_data.isdigit():
            return BarcodeType.EAN13
        elif len(barcode_data) == 8 and barcode_data.isdigit():
            return BarcodeType.EAN8
        elif len(barcode_data) == 12 and barcode_data.isdigit():
            return BarcodeType.UPCA
        elif len(barcode_data) in [6, 8] and barcode_data.isdigit():
            return BarcodeType.UPCE
        else:
            return BarcodeType.CODE128  # Default for alphanumeric
    
    def _estimate_orientation(self, region: np.ndarray) -> float:
        """Estimate barcode orientation angle"""
        # Use Hough transform to detect dominant lines
        edges = cv2.Canny(region, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = theta * 180 / np.pi
                angles.append(angle)
            
            # Return median angle
            return np.median(angles)
        
        return 0.0
    
    def _assess_quality(self, region: np.ndarray) -> float:
        """Assess image quality for barcode region"""
        # Calculate focus measure using Laplacian variance
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 scale
        quality_score = min(laplacian_var / 1000, 1.0)
        
        return quality_score
    
    def _fallback_detection(self, image: np.ndarray) -> List[BarcodeDetection]:
        """Fallback to traditional barcode detection"""
        detections = []
        
        # Try multiple preprocessing techniques
        for enhanced in [
            image,
            self._enhance_image_for_barcode(image),
            cv2.GaussianBlur(image, (5, 5), 0),
        ]:
            barcodes = pyzbar.decode(enhanced)
            
            for barcode in barcodes:
                points = barcode.polygon
                if len(points) == 4:
                    x = min(p.x for p in points)
                    y = min(p.y for p in points)
                    w = max(p.x for p in points) - x
                    h = max(p.y for p in points) - y
                    
                    detection = BarcodeDetection(
                        barcode_data=barcode.data.decode('utf-8'),
                        barcode_type=self._classify_barcode_type(barcode.data.decode('utf-8')),
                        bounding_box=(x, y, w, h),
                        confidence=0.8,  # Default confidence for pyzbar
                        orientation=0.0,
                        image_quality_score=0.7
                    )
                    detections.append(detection)
        
        return detections
    
    def train(self, dataset_path: str, epochs: int = 100):
        """Train YOLO-Barcode on custom dataset"""
        if isinstance(self.model, YOLO):
            # Train using Ultralytics framework
            self.model.train(
                data=dataset_path,
                epochs=epochs,
                imgsz=Config.INPUT_SIZE,
                batch=Config.BATCH_SIZE,
                device=self.device
            )
        else:
            # Custom training loop
            logger.info("Custom training not implemented yet")
    
    def save_model(self, path: Path):
        """Save trained model"""
        if isinstance(self.model, YOLO):
            self.model.save(str(path))
        else:
            torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: Path):
        """Load trained model"""
        if path.suffix == '.pt':
            self.model = YOLO(str(path))
        else:
            # Load custom model
            self.model = self._build_custom_yolo_barcode()
            self.model.load_state_dict(torch.load(path, map_location=self.device))
