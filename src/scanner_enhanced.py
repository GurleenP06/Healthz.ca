#!/usr/bin/env python3
"""
healthz.ca - Enhanced Barcode Nutrition Scanner with State-of-the-Art Deep Learning
Integrates YOLO-Barcode, USDA FoodData Central, and Canadian sources
"""

import os
import sys
import json
import time
import hashlib
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep Learning & ML
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
from torchvision.ops import nms

# YOLO and Object Detection
try:
    import ultralytics
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. YOLO features disabled.")

# Computer Vision
try:
    import cv2
    from pyzbar import pyzbar
    from PIL import Image, ImageDraw, ImageFont
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    print("Warning: OpenCV/pyzbar not available. Computer vision features disabled.")

# Additional ML Libraries
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score, precision_recall_curve, average_precision_score
    import xgboost as xgb
    import lightgbm as lgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML libraries not available. ML features disabled.")

# Web Scraping
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from bs4 import BeautifulSoup
    import cloudscraper
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    print("Warning: Web scraping libraries not available. Scraping features disabled.")

# Data Validation
from pydantic import BaseModel, Field, validator
import re
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the system"""
    
    # API Keys (set these as environment variables)
    USDA_API_KEY = os.getenv('USDA_API_KEY', 'DEMO_KEY')
    OPENFOODFACTS_API_KEY = os.getenv('OPENFOODFACTS_API_KEY', None)
    EDAMAM_APP_ID = os.getenv('EDAMAM_APP_ID', None)
    EDAMAM_APP_KEY = os.getenv('EDAMAM_APP_KEY', None)
    
    # USDA FoodData Central API
    USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"
    USDA_SEARCH_ENDPOINT = f"{USDA_BASE_URL}/foods/search"
    USDA_FOOD_ENDPOINT = f"{USDA_BASE_URL}/food"
    USDA_LIST_ENDPOINT = f"{USDA_BASE_URL}/foods/list"
    
    # Model Paths
    MODELS_DIR = Path("./models")
    YOLO_BARCODE_PATH = MODELS_DIR / "yolo_barcode.pt"
    NUTRITION_MODEL_PATH = MODELS_DIR / "nutrition_predictor.pth"
    
    # Data Paths
    DATA_DIR = Path("./data")
    DATABASE_PATH = DATA_DIR / "products.db"
    CACHE_DIR = DATA_DIR / "cache"
    
    # Training Parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Barcode Detection Parameters
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    INPUT_SIZE = 640  # YOLO input size
    
    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ENHANCED DATA MODELS
# ============================================================================

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
class NutritionInfo:
    """Comprehensive nutrition data model with USDA fields"""
    # Basic Macros
    calories: Optional[float] = None
    total_fat: Optional[float] = None
    saturated_fat: Optional[float] = None
    trans_fat: Optional[float] = None
    polyunsaturated_fat: Optional[float] = None
    monounsaturated_fat: Optional[float] = None
    cholesterol: Optional[float] = None
    sodium: Optional[float] = None
    total_carbohydrates: Optional[float] = None
    dietary_fiber: Optional[float] = None
    soluble_fiber: Optional[float] = None
    insoluble_fiber: Optional[float] = None
    sugars: Optional[float] = None
    added_sugars: Optional[float] = None
    protein: Optional[float] = None
    
    # Vitamins
    vitamin_a: Optional[float] = None
    vitamin_c: Optional[float] = None
    vitamin_d: Optional[float] = None
    vitamin_e: Optional[float] = None
    vitamin_k: Optional[float] = None
    thiamin: Optional[float] = None
    riboflavin: Optional[float] = None
    niacin: Optional[float] = None
    vitamin_b6: Optional[float] = None
    folate: Optional[float] = None
    vitamin_b12: Optional[float] = None
    pantothenic_acid: Optional[float] = None
    biotin: Optional[float] = None
    choline: Optional[float] = None
    
    # Minerals
    calcium: Optional[float] = None
    iron: Optional[float] = None
    magnesium: Optional[float] = None
    phosphorus: Optional[float] = None
    potassium: Optional[float] = None
    zinc: Optional[float] = None
    copper: Optional[float] = None
    manganese: Optional[float] = None
    selenium: Optional[float] = None
    iodine: Optional[float] = None
    
    # Serving Information
    serving_size: Optional[str] = None
    serving_size_unit: Optional[str] = None
    servings_per_container: Optional[float] = None
    
    # Metadata
    fdc_id: Optional[str] = None  # USDA FoodData Central ID
    data_source: Optional[str] = None
    confidence_score: Optional[float] = None

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

@dataclass
class Product:
    """Enhanced product information model"""
    # Identifiers
    barcode: str
    name: str
    gtin: Optional[str] = None  # Global Trade Item Number
    fdc_id: Optional[str] = None  # USDA FoodData Central ID
    
    # Basic Info
    brand: Optional[str] = None
    manufacturer: Optional[str] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    
    # Detailed Info
    ingredients: Optional[List[str]] = None
    nutrition: Optional[NutritionInfo] = None
    allergens: Optional[List[str]] = None
    certifications: Optional[List[str]] = None  # Organic, Non-GMO, etc.
    
    # Media
    image_url: Optional[str] = None
    images: Optional[List[str]] = None  # Multiple product images
    
    # Retail Info
    price: Optional[float] = None
    store: Optional[str] = None
    availability: Optional[str] = None
    
    # Metadata
    last_updated: Optional[datetime] = None
    confidence_score: float = 0.0
    data_sources: List[str] = field(default_factory=list)
    predictions: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# SIMPLIFIED YOLO-BARCODE DETECTION MODULE
# ============================================================================

class YOLOBarcodeDetector:
    """
    Simplified barcode detection using YOLO architecture
    Falls back to traditional methods if YOLO is not available
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.device = Config.DEVICE
        self.model = None
        
        if YOLO_AVAILABLE and model_path and model_path.exists():
            self.load_model(model_path)
        else:
            self.initialize_fallback()
    
    def initialize_fallback(self):
        """Initialize fallback detection methods"""
        logger.info("Using fallback barcode detection (pyzbar)")
        self.model = None
    
    def detect_barcodes(self, image: np.ndarray, enhance: bool = True) -> List[BarcodeDetection]:
        """
        Detect barcodes in image using available methods
        
        Args:
            image: Input image as numpy array
            enhance: Whether to apply image enhancement
        
        Returns:
            List of detected barcodes with locations and confidence
        """
        if not CV_AVAILABLE:
            logger.warning("Computer vision not available")
            return []
        
        detections = []
        
        # Enhance image for better detection if needed
        if enhance:
            image = self._enhance_image_for_barcode(image)
        
        # Try YOLO if available
        if self.model and YOLO_AVAILABLE:
            detections = self._yolo_detection(image)
        
        # Fallback to traditional detection
        if not detections:
            detections = self._fallback_detection(image)
        
        return detections
    
    def _yolo_detection(self, image: np.ndarray) -> List[BarcodeDetection]:
        """YOLO-based barcode detection"""
        try:
            results = self.model(image, conf=Config.CONFIDENCE_THRESHOLD)
            
            detections = []
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
                                orientation=0.0,
                                image_quality_score=0.8
                            )
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []
    
    def _enhance_image_for_barcode(self, image: np.ndarray) -> np.ndarray:
        """Apply image enhancement for barcode detection"""
        if not CV_AVAILABLE:
            return image
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Convert back to 3-channel for YOLO
        if len(image.shape) == 3:
            enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            enhanced_color = enhanced
        
        return enhanced_color
    
    def _decode_barcode_region(self, region: np.ndarray) -> Optional[str]:
        """Decode barcode from image region using multiple methods"""
        if not CV_AVAILABLE:
            return None
        
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
    
    def _fallback_detection(self, image: np.ndarray) -> List[BarcodeDetection]:
        """Fallback to traditional barcode detection"""
        if not CV_AVAILABLE:
            return []
        
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
    
    def load_model(self, path: Path):
        """Load trained model"""
        if YOLO_AVAILABLE and path.suffix == '.pt':
            self.model = YOLO(str(path))
        else:
            logger.warning("YOLO model loading not available")

# ============================================================================
# USDA FOODDATA CENTRAL API INTEGRATION
# ============================================================================

class USDAFoodDataAPI:
    """
    Integration with USDA FoodData Central API
    Provides comprehensive nutrition data for US products
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.USDA_API_KEY
        self.session = requests.Session()
        self.cache = {}
    
    def search_by_upc(self, upc: str) -> Optional[Product]:
        """
        Search for product by UPC/barcode in USDA database
        """
        try:
            # Search using UPC as query
            params = {
                'api_key': self.api_key,
                'query': upc,
                'dataType': ['Branded'],  # Focus on branded products with UPCs
                'pageSize': 10
            }
            
            response = self.session.get(Config.USDA_SEARCH_ENDPOINT, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('foods'):
                # Find exact UPC match
                for food_item in data['foods']:
                    # Check if GTIN/UPC matches
                    if food_item.get('gtinUpc') == upc:
                        return self._parse_usda_food(food_item)
                
                # If no exact match, return first result with confidence adjustment
                if data['foods']:
                    product = self._parse_usda_food(data['foods'][0])
                    product.confidence_score *= 0.7  # Lower confidence for non-exact match
                    return product
            
        except Exception as e:
            logger.error(f"USDA API search failed for UPC {upc}: {e}")
        
        return None
    
    def _parse_usda_food(self, food_data: Dict) -> Product:
        """Parse USDA food data into Product model"""
        
        # Extract nutrition information
        nutrition = self._extract_nutrition(food_data.get('foodNutrients', []))
        
        # Build product
        product = Product(
            barcode=food_data.get('gtinUpc', ''),
            gtin=food_data.get('gtinUpc'),
            fdc_id=str(food_data.get('fdcId', '')),
            name=food_data.get('description', ''),
            brand=food_data.get('brandOwner'),
            manufacturer=food_data.get('brandOwner'),
            category=food_data.get('foodCategory'),
            ingredients=self._parse_ingredients(food_data.get('ingredients', '')),
            nutrition=nutrition,
            confidence_score=0.95,  # High confidence for USDA data
            data_sources=['USDA FoodData Central']
        )
        
        # Add serving size info
        if nutrition and food_data.get('servingSize'):
            nutrition.serving_size = str(food_data.get('servingSize'))
            nutrition.serving_size_unit = food_data.get('servingSizeUnit')
            nutrition.fdc_id = str(food_data.get('fdcId', ''))
            nutrition.data_source = 'USDA'
        
        return product
    
    def _extract_nutrition(self, nutrients: List[Dict]) -> NutritionInfo:
        """Extract nutrition info from USDA nutrient data"""
        nutrition = NutritionInfo()
        
        # Mapping of USDA nutrient IDs to our fields
        nutrient_mapping = {
            '1008': 'calories',  # Energy (kcal)
            '1003': 'protein',   # Protein
            '1004': 'total_fat', # Total lipid (fat)
            '1005': 'total_carbohydrates', # Carbohydrate, by difference
            '1079': 'dietary_fiber',  # Fiber, total dietary
            '2000': 'sugars',    # Sugars, total
            '1093': 'sodium',    # Sodium, Na
            '1087': 'calcium',   # Calcium, Ca
            '1089': 'iron',      # Iron, Fe
            '1090': 'magnesium', # Magnesium, Mg
            '1091': 'phosphorus', # Phosphorus, P
            '1092': 'potassium', # Potassium, K
            '1095': 'zinc',      # Zinc, Zn
            '1106': 'vitamin_a', # Vitamin A, RAE
            '1162': 'vitamin_c', # Vitamin C, total ascorbic acid
            '1114': 'vitamin_d', # Vitamin D (D2 + D3)
            '1109': 'vitamin_e', # Vitamin E (alpha-tocopherol)
            '1185': 'vitamin_k', # Vitamin K (phylloquinone)
            '1165': 'thiamin',   # Thiamin
            '1166': 'riboflavin', # Riboflavin
            '1167': 'niacin',    # Niacin
            '1175': 'vitamin_b6', # Vitamin B-6
            '1177': 'folate',    # Folate, total
            '1178': 'vitamin_b12', # Vitamin B-12
            '1176': 'pantothenic_acid', # Pantothenic acid
            '1180': 'choline',   # Choline, total
            '1258': 'saturated_fat', # Fatty acids, total saturated
            '1257': 'trans_fat', # Fatty acids, total trans
            '1253': 'cholesterol', # Cholesterol
        }
        
        for nutrient in nutrients:
            nutrient_id = str(nutrient.get('nutrientId', ''))
            if nutrient_id in nutrient_mapping:
                field_name = nutrient_mapping[nutrient_id]
                value = nutrient.get('value')
                if value is not None:
                    setattr(nutrition, field_name, float(value))
        
        nutrition.confidence_score = 0.95
        return nutrition
    
    def _parse_ingredients(self, ingredients_text: str) -> List[str]:
        """Parse ingredients text into list"""
        if not ingredients_text:
            return []
        
        # Clean and split ingredients
        ingredients_text = ingredients_text.strip()
        
        # Common separators
        if ',' in ingredients_text:
            ingredients = ingredients_text.split(',')
        elif ';' in ingredients_text:
            ingredients = ingredients_text.split(';')
        else:
            ingredients = [ingredients_text]
        
        # Clean each ingredient
        cleaned = []
        for ing in ingredients:
            ing = ing.strip()
            if ing and not ing.startswith('CONTAINS'):
                cleaned.append(ing)
        
        return cleaned

# ============================================================================
# MULTI-SOURCE DATA AGGREGATOR
# ============================================================================

class MultiSourceDataAggregator:
    """
    Aggregates data from multiple sources:
    - USDA FoodData Central
    - OpenFoodFacts
    - Canadian retailers
    """
    
    def __init__(self):
        self.usda_api = USDAFoodDataAPI()
        self.scraper = requests.Session()
        self.cache = {}
        
    def get_product_data(self, barcode: str, sources: List[str] = None) -> Product:
        """
        Get product data from multiple sources and aggregate
        """
        
        if sources is None:
            sources = ['usda', 'openfoodfacts']
        
        products = []
        
        # Try each source
        if 'usda' in sources:
            try:
                product = self.usda_api.search_by_upc(barcode)
                if product:
                    products.append(product)
            except Exception as e:
                logger.error(f"USDA search failed: {e}")
        
        if 'openfoodfacts' in sources:
            try:
                product = self._query_openfoodfacts(barcode)
                if product:
                    products.append(product)
            except Exception as e:
                logger.error(f"OpenFoodFacts search failed: {e}")
        
        # Aggregate results
        if products:
            return self._aggregate_products(products)
        
        # Return empty product if no data found
        return Product(
            barcode=barcode,
            name="Unknown Product",
            confidence_score=0.0
        )
    
    def _aggregate_products(self, products: List[Product]) -> Product:
        """
        Intelligently aggregate multiple product records
        Uses confidence scores and data completeness
        """
        
        # Start with the most complete product
        best_product = max(products, key=lambda p: self._calculate_completeness(p))
        
        # Merge data from other sources
        for product in products:
            if product == best_product:
                continue
            
            # Merge nutrition data
            if product.nutrition and (not best_product.nutrition or 
                                     self._nutrition_completeness(product.nutrition) > 
                                     self._nutrition_completeness(best_product.nutrition)):
                best_product.nutrition = product.nutrition
            
            # Merge ingredients
            if product.ingredients and not best_product.ingredients:
                best_product.ingredients = product.ingredients
            
            # Merge identifiers
            if product.fdc_id and not best_product.fdc_id:
                best_product.fdc_id = product.fdc_id
            
            if product.gtin and not best_product.gtin:
                best_product.gtin = product.gtin
            
            # Add all data sources
            best_product.data_sources.extend(product.data_sources)
        
        # Remove duplicates in data sources
        best_product.data_sources = list(set(best_product.data_sources))
        
        # Recalculate confidence
        best_product.confidence_score = self._calculate_completeness(best_product)
        
        return best_product
    
    def _calculate_completeness(self, product: Product) -> float:
        """Calculate product data completeness score"""
        score = 0.0
        weights = {
            'name': 0.15,
            'brand': 0.10,
            'nutrition': 0.30,
            'ingredients': 0.15,
            'category': 0.10,
            'images': 0.05,
            'fdc_id': 0.10,
            'allergens': 0.05
        }
        
        if product.name and product.name != "Unknown Product":
            score += weights['name']
        if product.brand:
            score += weights['brand']
        if product.nutrition:
            score += weights['nutrition'] * self._nutrition_completeness(product.nutrition)
        if product.ingredients:
            score += weights['ingredients']
        if product.category:
            score += weights['category']
        if product.image_url or product.images:
            score += weights['images']
        if product.fdc_id:
            score += weights['fdc_id']
        if product.allergens:
            score += weights['allergens']
        
        return min(score, 1.0)
    
    def _nutrition_completeness(self, nutrition: NutritionInfo) -> float:
        """Calculate nutrition data completeness"""
        essential_fields = [
            'calories', 'total_fat', 'saturated_fat', 
            'sodium', 'total_carbohydrates', 'protein'
        ]
        
        complete = sum(1 for field in essential_fields 
                      if getattr(nutrition, field) is not None)
        
        return complete / len(essential_fields)
    
    def _query_openfoodfacts(self, barcode: str) -> Optional[Product]:
        """Query OpenFoodFacts database"""
        try:
            url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}"
            response = self.scraper.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 1:
                    return self._parse_openfoodfacts(data['product'])
        except Exception as e:
            logger.error(f"OpenFoodFacts query failed: {e}")
        
        return None
    
    def _parse_openfoodfacts(self, data: dict) -> Product:
        """Parse OpenFoodFacts data"""
        nutrition_data = data.get('nutriments', {})
        
        nutrition = NutritionInfo(
            calories=nutrition_data.get('energy-kcal_100g'),
            total_fat=nutrition_data.get('fat_100g'),
            saturated_fat=nutrition_data.get('saturated-fat_100g'),
            trans_fat=nutrition_data.get('trans-fat_100g'),
            sodium=nutrition_data.get('sodium_100g'),
            total_carbohydrates=nutrition_data.get('carbohydrates_100g'),
            dietary_fiber=nutrition_data.get('fiber_100g'),
            sugars=nutrition_data.get('sugars_100g'),
            protein=nutrition_data.get('proteins_100g'),
            data_source='OpenFoodFacts'
        )
        
        return Product(
            barcode=data.get('code', ''),
            name=data.get('product_name', ''),
            brand=data.get('brands', ''),
            category=data.get('categories', ''),
            ingredients=data.get('ingredients_text', '').split(',') if data.get('ingredients_text') else None,
            nutrition=nutrition,
            allergens=data.get('allergens_hierarchy', []),
            image_url=data.get('image_url'),
            confidence_score=0.8,
            data_sources=['OpenFoodFacts']
        )

# ============================================================================
# MAIN PIPELINE
# ============================================================================

class HealthzScanner:
    """
    Main scanner class for healthz.ca
    Simplified version of the advanced pipeline
    """
    
    def __init__(self):
        logger.info("Initializing Healthz Scanner")
        
        # Initialize components
        self.barcode_detector = YOLOBarcodeDetector()
        self.data_aggregator = MultiSourceDataAggregator()
        
        # Check API key
        if Config.USDA_API_KEY == 'DEMO_KEY':
            logger.warning("Using DEMO_KEY - limited to 30 requests/hour")
    
    def scan_barcode(self, barcode: str) -> Optional[Product]:
        """
        Look up product information by barcode
        
        Args:
            barcode: UPC/EAN barcode string
            
        Returns:
            Product object with nutrition info or None
        """
        logger.info(f"Scanning barcode: {barcode}")
        
        # Get product data from multiple sources
        product = self.data_aggregator.get_product_data(barcode)
        
        if product and product.name != "Unknown Product":
            return product
        
        return None
    
    def scan_image(self, image_path: str) -> List[Product]:
        """
        Scan image for barcodes and return products
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of Product objects
        """
        if not CV_AVAILABLE:
            logger.error("Computer vision not available")
            return []
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return []
        
        # Detect barcodes
        detections = self.barcode_detector.detect_barcodes(image)
        
        products = []
        for detection in detections:
            product = self.scan_barcode(detection.barcode_data)
            if product:
                products.append(product)
        
        return products

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='healthz.ca Barcode Scanner')
    parser.add_argument('--barcode', type=str, help='Barcode to scan')
    parser.add_argument('--image', type=str, help='Image file to scan')
    parser.add_argument('--mode', choices=['barcode', 'image'], default='barcode', help='Scan mode')
    args = parser.parse_args()
    
    scanner = HealthzScanner()
    
    if args.mode == 'barcode':
        if args.barcode:
            barcode = args.barcode
        else:
            barcode = input("Enter barcode: ")
        
        product = scanner.scan_barcode(barcode)
        
        if product:
            print(f"\n{'='*50}")
            print(f"Product: {product.name}")
            print(f"Brand: {product.brand or 'N/A'}")
            print(f"Category: {product.category or 'N/A'}")
            
            if product.nutrition:
                print(f"\nNutrition Facts:")
                print(f"  Calories: {product.nutrition.calories or 'N/A'}")
                print(f"  Protein: {product.nutrition.protein or 'N/A'} g")
                print(f"  Fat: {product.nutrition.total_fat or 'N/A'} g")
                print(f"  Carbs: {product.nutrition.total_carbohydrates or 'N/A'} g")
                print(f"  Sodium: {product.nutrition.sodium or 'N/A'} mg")
            
            print(f"\nSources: {', '.join(product.data_sources)}")
            print(f"Confidence: {product.confidence_score:.1%}")
            print(f"{'='*50}\n")
        else:
            print("Product not found")
    
    elif args.mode == 'image':
        if not args.image:
            print("Please provide --image path")
            return
        
        products = scanner.scan_image(args.image)
        
        if products:
            print(f"\nFound {len(products)} products:")
            for i, product in enumerate(products, 1):
                print(f"\n{i}. {product.name} ({product.barcode})")
                print(f"   Brand: {product.brand or 'N/A'}")
                print(f"   Confidence: {product.confidence_score:.1%}")
        else:
            print("No products found in image")


if __name__ == "__main__":
    main()
