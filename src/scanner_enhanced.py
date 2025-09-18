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
    level=logging.INFO,
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
        
        The USDA API supports UPC codes in search queries
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
    
    def get_food_details(self, fdc_id: str) -> Optional[Dict]:
        """Get detailed food information by FDC ID"""
        try:
            url = f"{Config.USDA_FOOD_ENDPOINT}/{fdc_id}"
            params = {'api_key': self.api_key}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get food details for FDC ID {fdc_id}: {e}")
            return None
    
    def _parse_usda_food(self, food_data: Dict) -> Product:
        """Parse USDA food data into Product model"""
        
        # Extract nutrition information
        nutrition = self._extract_nutrition(food_data.get('foodNutrients', []))
        
        # Build product
        product = Product(
            barcode=food_data.get('gtinUpc', ''),
            name=food_data.get('description', ''),
            gtin=food_data.get('gtinUpc'),
            fdc_id=str(food_data.get('fdcId', '')),
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
    - Edamam (if available)
    """
    
    def __init__(self):
        self.usda_api = USDAFoodDataAPI()
        if SCRAPING_AVAILABLE:
            self.scraper = cloudscraper.create_scraper()
        else:
            self.scraper = None
        self.cache = {}
        
    def get_product_data(self, barcode: str, sources: List[str] = None) -> Product:
        """
        Get product data from multiple sources and aggregate
        
        Args:
            barcode: Product barcode/UPC
            sources: List of sources to use (defaults to all)
        
        Returns:
            Aggregated product data with confidence scores
        """
        
        if sources is None:
            sources = ['usda', 'openfoodfacts', 'canadian_retailers']
        
        products = []
        
        # Try each source in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            
            if 'usda' in sources:
                futures['usda'] = executor.submit(self.usda_api.search_by_upc, barcode)
            
            if 'openfoodfacts' in sources:
                futures['openfoodfacts'] = executor.submit(self._query_openfoodfacts, barcode)
            
            if 'canadian_retailers' in sources and SCRAPING_AVAILABLE:
                futures['loblaws'] = executor.submit(self._scrape_loblaws, barcode)
                futures['metro'] = executor.submit(self._scrape_metro, barcode)
            
            # Collect results
            for source, future in futures.items():
                try:
                    result = future.result(timeout=10)
                    if result:
                        result.data_sources.append(source)
                        products.append(result)
                except Exception as e:
                    logger.error(f"Failed to get data from {source}: {e}")
        
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
            if not self.scraper:
                return None
                
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
    
    def _scrape_loblaws(self, barcode: str) -> Optional[Product]:
        """Scrape Loblaws/PC products"""
        # Implementation for Loblaws scraping
        # This would involve web scraping with proper headers and parsing
        if not SCRAPING_AVAILABLE:
            return None
            
        try:
            # Placeholder for Loblaws scraping
            # Would use Selenium or requests with proper headers
            logger.info(f"Loblaws scraping not implemented for {barcode}")
            return None
        except Exception as e:
            logger.error(f"Loblaws scraping failed: {e}")
            return None
    
    def _scrape_metro(self, barcode: str) -> Optional[Product]:
        """Scrape Metro products"""
        # Implementation for Metro scraping
        if not SCRAPING_AVAILABLE:
            return None
            
        try:
            # Placeholder for Metro scraping
            # Would use Selenium or requests with proper headers
            logger.info(f"Metro scraping not implemented for {barcode}")
            return None
        except Exception as e:
            logger.error(f"Metro scraping failed: {e}")
            return None

# ============================================================================
# ENHANCED ML NUTRITION PREDICTOR
# ============================================================================

class EnhancedNutritionPredictor:
    """
    Advanced ML model for predicting nutrition from product features
    Uses ensemble of deep learning and gradient boosting
    """
    
    def __init__(self):
        self.device = Config.DEVICE
        if ML_AVAILABLE:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.embedding_model = None
        self.neural_model = None
        self.xgb_models = {}
        self.lgb_models = {}
        self.scalers = {}
        self.is_trained = False
        
    def train(self, products: List[Product], validation_split: float = 0.2):
        """Train all models in the ensemble"""
        
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available, skipping training")
            return
        
        # Prepare features and targets
        X, y, nutrient_names = self._prepare_training_data(products)
        
        if len(X) == 0:
            logger.error("No valid training data")
            return
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Train models for each nutrient
        for i, nutrient in enumerate(nutrient_names):
            logger.info(f"Training model for {nutrient}")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            self.scalers[nutrient] = scaler
            
            # Extract target for this nutrient
            y_train_nutrient = y_train[:, i]
            y_val_nutrient = y_val[:, i]
            
            # Train XGBoost
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                objective='reg:squarederror'
            )
            xgb_model.fit(X_train_scaled, y_train_nutrient)
            self.xgb_models[nutrient] = xgb_model
            
            # Train LightGBM
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1
            )
            lgb_model.fit(X_train_scaled, y_train_nutrient)
            self.lgb_models[nutrient] = lgb_model
            
            # Evaluate
            xgb_pred = xgb_model.predict(X_val_scaled)
            lgb_pred = lgb_model.predict(X_val_scaled)
            ensemble_pred = (xgb_pred + lgb_pred) / 2
            
            mae = mean_absolute_error(y_val_nutrient, ensemble_pred)
            logger.info(f"{nutrient} MAE: {mae:.2f}")
        
        self.is_trained = True
    
    def predict(self, product: Product) -> NutritionInfo:
        """Predict nutrition for a product"""
        
        if not self.is_trained or not ML_AVAILABLE:
            logger.warning("Model not trained or ML not available, returning empty nutrition")
            return NutritionInfo()
        
        # Extract features
        features = self._extract_features(product)
        
        # Predict each nutrient
        nutrition = NutritionInfo()
        
        for nutrient, scaler in self.scalers.items():
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Ensemble prediction
            predictions = []
            
            if nutrient in self.xgb_models:
                predictions.append(self.xgb_models[nutrient].predict(features_scaled)[0])
            
            if nutrient in self.lgb_models:
                predictions.append(self.lgb_models[nutrient].predict(features_scaled)[0])
            
            if predictions:
                value = np.mean(predictions)
                setattr(nutrition, nutrient, float(value))
        
        nutrition.data_source = 'ML Prediction'
        nutrition.confidence_score = 0.7  # ML predictions have lower confidence
        
        return nutrition
    
    def _prepare_training_data(self, products: List[Product]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data from products"""
        
        features = []
        targets = []
        valid_products = []
        
        # Filter products with nutrition data
        for product in products:
            if product.nutrition and product.nutrition.calories is not None:
                valid_products.append(product)
        
        if not valid_products:
            return np.array([]), np.array([]), []
        
        # Define target nutrients
        target_nutrients = [
            'calories', 'protein', 'total_fat', 
            'total_carbohydrates', 'sodium'
        ]
        
        # Extract features and targets
        for product in valid_products:
            # Extract features
            feature_vec = self._extract_features(product)
            features.append(feature_vec)
            
            # Extract targets
            target_vec = []
            for nutrient in target_nutrients:
                value = getattr(product.nutrition, nutrient, 0) or 0
                target_vec.append(value)
            targets.append(target_vec)
        
        return np.array(features), np.array(targets), target_nutrients
    
    def _extract_features(self, product: Product) -> np.ndarray:
        """Extract feature vector from product"""
        
        if not self.embedding_model:
            # Fallback to simple features
            text_parts = []
            if product.name:
                text_parts.append(f"Product: {product.name}")
            if product.brand:
                text_parts.append(f"Brand: {product.brand}")
            if product.category:
                text_parts.append(f"Category: {product.category}")
            if product.ingredients:
                text_parts.append(f"Ingredients: {', '.join(product.ingredients[:20])}")
            
            text = " ".join(text_parts) or "Unknown product"
            
            # Simple text features
            text_features = [
                len(text),
                text.count('organic'),
                text.count('sugar-free'),
                text.count('low-fat'),
                text.count('gluten-free'),
                text.count('natural'),
                text.count('healthy'),
            ]
            
            # Numerical features
            num_features = [
                len(product.name) if product.name else 0,
                len(product.ingredients) if product.ingredients else 0,
                1 if product.brand else 0,
                1 if product.category else 0,
            ]
            
            # Combine features
            feature_vector = np.array(text_features + num_features)
            
        else:
            # Text features using sentence embeddings
            text_parts = []
            if product.name:
                text_parts.append(f"Product: {product.name}")
            if product.brand:
                text_parts.append(f"Brand: {product.brand}")
            if product.category:
                text_parts.append(f"Category: {product.category}")
            if product.ingredients:
                text_parts.append(f"Ingredients: {', '.join(product.ingredients[:20])}")
            
            text = " ".join(text_parts) or "Unknown product"
            text_embedding = self.embedding_model.encode(text)
            
            # Numerical features
            num_features = [
                len(product.name) if product.name else 0,
                len(product.ingredients) if product.ingredients else 0,
                1 if product.brand else 0,
                1 if product.category else 0,
                1 if 'organic' in text.lower() else 0,
                1 if 'sugar-free' in text.lower() else 0,
                1 if 'low-fat' in text.lower() else 0,
                1 if 'gluten-free' in text.lower() else 0,
            ]
            
            # Combine features
            feature_vector = np.concatenate([text_embedding, num_features])
        
        return feature_vector
    
    def save_models(self, path: Path):
        """Save trained models"""
        models_data = {
            'xgb_models': self.xgb_models,
            'lgb_models': self.lgb_models,
            'scalers': self.scalers,
            'is_trained': self.is_trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(models_data, f)
    
    def load_models(self, path: Path):
        """Load trained models"""
        with open(path, 'rb') as f:
            models_data = pickle.load(f)
        
        self.xgb_models = models_data['xgb_models']
        self.lgb_models = models_data['lgb_models']
        self.scalers = models_data['scalers']
        self.is_trained = models_data['is_trained']

# ============================================================================
# MAIN PIPELINE
# ============================================================================

class AdvancedNutritionScannerPipeline:
    """
    Complete pipeline integrating:
    - YOLO-Barcode detection
    - USDA FoodData Central
    - Multi-source data aggregation
    - ML nutrition prediction
    """
    
    def __init__(self):
        logger.info("Initializing Advanced Nutrition Scanner Pipeline")
        
        # Initialize components
        if CV_AVAILABLE:
            from .yolo_detector import YOLOBarcodeDetector
            self.barcode_detector = YOLOBarcodeDetector()
        else:
            self.barcode_detector = None
            
        self.data_aggregator = MultiSourceDataAggregator()
        self.nutrition_predictor = EnhancedNutritionPredictor()
        self.database = self._init_database()
        
        # Load models if available
        self._load_models()
    
    def _init_database(self) -> sqlite3.Connection:
        """Initialize SQLite database"""
        conn = sqlite3.connect(str(Config.DATABASE_PATH), check_same_thread=False)
        
        # Create tables
        conn.execute('''
        CREATE TABLE IF NOT EXISTS products (
            barcode TEXT PRIMARY KEY,
            gtin TEXT,
            fdc_id TEXT,
            name TEXT,
            brand TEXT,
            category TEXT,
            ingredients TEXT,
            allergens TEXT,
            nutrition TEXT,
            confidence_score REAL,
            data_sources TEXT,
            last_updated TIMESTAMP
        )
        ''')
        
        conn.commit()
        return conn
    
    def _load_models(self):
        """Load pre-trained models if available"""
        
        # Load YOLO-Barcode model
        if Config.YOLO_BARCODE_PATH.exists() and self.barcode_detector:
            logger.info("Loading YOLO-Barcode model")
            self.barcode_detector.load_model(Config.YOLO_BARCODE_PATH)
        
        # Load nutrition predictor
        nutrition_model_path = Config.MODELS_DIR / "nutrition_models.pkl"
        if nutrition_model_path.exists():
            logger.info("Loading nutrition prediction models")
            self.nutrition_predictor.load_models(nutrition_model_path)
    
    def scan_and_analyze(self, 
                        image_source: Union[str, np.ndarray],
                        enhance_image: bool = True,
                        use_ml_prediction: bool = True) -> Dict[str, Any]:
        """
        Complete pipeline: scan -> detect -> fetch data -> predict
        
        Args:
            image_source: Path to image file or numpy array
            enhance_image: Whether to enhance image for better detection
            use_ml_prediction: Whether to use ML for missing nutrition data
        
        Returns:
            Dictionary with detected barcodes and product information
        """
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'detections': [],
            'products': []
        }
        
        # Load image if path provided
        if isinstance(image_source, str):
            if CV_AVAILABLE:
                image = cv2.imread(image_source)
            else:
                logger.error("OpenCV not available for image loading")
                return results
        else:
            image = image_source
        
        # Step 1: Detect barcodes
        logger.info("Detecting barcodes...")
        if self.barcode_detector:
            detections = self.barcode_detector.detect_barcodes(image, enhance=enhance_image)
        else:
            logger.warning("Barcode detector not available")
            detections = []
            
        results['detections'] = [asdict(d) for d in detections]
        
        if not detections:
            logger.warning("No barcodes detected")
            return results
        
        # Step 2: Process each detected barcode
        for detection in detections:
            logger.info(f"Processing barcode: {detection.barcode_data}")
            
            # Check cache
            cached_product = self._get_cached_product(detection.barcode_data)
            
            if cached_product:
                logger.info("Using cached product data")
                product = cached_product
            else:
                # Fetch from multiple sources
                logger.info("Fetching product data from APIs...")
                product = self.data_aggregator.get_product_data(detection.barcode_data)
                
                # Use ML prediction if needed
                if use_ml_prediction and (not product.nutrition or 
                                         product.confidence_score < 0.6):
                    logger.info("Using ML to predict nutrition...")
                    predicted_nutrition = self.nutrition_predictor.predict(product)
                    
                    if not product.nutrition or self._nutrition_completeness(predicted_nutrition) > \
                       self._nutrition_completeness(product.nutrition):
                        product.nutrition = predicted_nutrition
                        product.predictions['nutrition'] = 'ML Enhanced'
                
                # Cache the product
                self._cache_product(product)
            
            # Add detection info to product
            product.predictions['detection_confidence'] = detection.confidence
            product.predictions['barcode_type'] = detection.barcode_type.value
            product.predictions['image_quality'] = detection.image_quality_score
            
            results['products'].append(asdict(product))
        
        return results
    
    def _nutrition_completeness(self, nutrition: Optional[NutritionInfo]) -> float:
        """Calculate nutrition completeness score"""
        if not nutrition:
            return 0.0
        
        essential_fields = [
            'calories', 'total_fat', 'protein', 
            'total_carbohydrates', 'sodium'
        ]
        
        complete = sum(1 for field in essential_fields 
                      if getattr(nutrition, field) is not None)
        
        return complete / len(essential_fields)
    
    def _get_cached_product(self, barcode: str) -> Optional[Product]:
        """Get product from cache"""
        try:
            cursor = self.database.cursor()
            cursor.execute(
                "SELECT * FROM products WHERE barcode = ?",
                (barcode,)
            )
            row = cursor.fetchone()
            
            if row:
                # Deserialize product
                product = Product(
                    barcode=row[0],
                    gtin=row[1],
                    fdc_id=row[2],
                    name=row[3],
                    brand=row[4],
                    category=row[5],
                    ingredients=json.loads(row[6]) if row[6] else None,
                    allergens=json.loads(row[7]) if row[7] else None,
                    nutrition=json.loads(row[8]) if row[8] else None,
                    confidence_score=row[9],
                    data_sources=json.loads(row[10]) if row[10] else [],
                    last_updated=datetime.fromisoformat(row[11])
                )
                
                # Check if cache is fresh (less than 7 days old)
                if (datetime.now() - product.last_updated).days < 7:
                    return product
        
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
        
        return None
    
    def _cache_product(self, product: Product):
        """Cache product in database"""
        try:
            cursor = self.database.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                product.barcode,
                product.gtin,
                product.fdc_id,
                product.name,
                product.brand,
                product.category,
                json.dumps(product.ingredients),
                json.dumps(product.allergens),
                json.dumps(asdict(product.nutrition)) if product.nutrition else None,
                product.confidence_score,
                json.dumps(product.data_sources),
                datetime.now().isoformat()
            ))
            self.database.commit()
        
        except Exception as e:
            logger.error(f"Failed to cache product: {e}")
    
    def train_models(self, training_data_path: Optional[str] = None):
        """Train all ML models"""
        
        logger.info("Starting model training...")
        
        # Collect or load training data
        if training_data_path:
            with open(training_data_path, 'rb') as f:
                products = pickle.load(f)
        else:
            # Collect training data from APIs
            logger.info("Collecting training data...")
            products = self._collect_training_data()
        
        # Train nutrition predictor
        logger.info("Training nutrition prediction models...")
        self.nutrition_predictor.train(products)
        
        # Save models
        self.nutrition_predictor.save_models(Config.MODELS_DIR / "nutrition_models.pkl")
        
        logger.info("Training complete!")
    
    def _collect_training_data(self) -> List[Product]:
        """Collect training data from various sources"""
        
        products = []
        
        # Common UPC prefixes for major brands
        upc_prefixes = [
            "0681310",  # No Name
            "0660100",  # President's Choice
            "0557425",  # Compliments
            "0627843",  # Great Value
            "0380000",  # Kraft
            "0430000",  # General Mills
            "0160000",  # Coca-Cola
            "0120000",  # Pepsi
        ]
        
        # Generate sample UPCs and fetch data
        for prefix in upc_prefixes:
            for i in range(10):  # Get 10 products per prefix
                # Generate valid UPC (simplified)
                upc = f"{prefix}{str(i).zfill(5)}"
                
                try:
                    product = self.data_aggregator.get_product_data(upc)
                    if product and product.nutrition:
                        products.append(product)
                except Exception as e:
                    logger.error(f"Failed to get data for {upc}: {e}")
        
        logger.info(f"Collected {len(products)} products for training")
        return products

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Enhanced CLI for the nutrition scanner"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Advanced Barcode Nutrition Scanner with YOLO and USDA Integration'
    )
    
    parser.add_argument('--mode', 
                       choices=['scan', 'camera', 'train', 'batch', 'test'],
                       default='scan',
                       help='Operation mode')
    
    parser.add_argument('--image', 
                       type=str,
                       help='Path to image file for scanning')
    
    parser.add_argument('--barcode',
                       type=str,
                       help='Direct barcode/UPC input')
    
    parser.add_argument('--enhance',
                       action='store_true',
                       help='Enable image enhancement')
    
    parser.add_argument('--sources',
                       nargs='+',
                       choices=['usda', 'openfoodfacts', 'canadian_retailers'],
                       help='Data sources to use')
    
    parser.add_argument('--output',
                       type=str,
                       help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AdvancedNutritionScannerPipeline()
    
    if args.mode == 'scan':
        if args.barcode:
            # Direct barcode lookup
            product = pipeline.data_aggregator.get_product_data(args.barcode, args.sources)
            result = {'products': [asdict(product)]}
        elif args.image:
            # Scan from image
            result = pipeline.scan_and_analyze(args.image, enhance_image=args.enhance)
        else:
            print("Please provide --image or --barcode")
            return
        
        # Display results
        if result.get('products'):
            for product_data in result['products']:
                print("\n" + "="*60)
                print(f"PRODUCT: {product_data.get('name', 'Unknown')}")
                print(f"BRAND: {product_data.get('brand', 'N/A')}")
                print(f"BARCODE: {product_data.get('barcode')}")
                
                if product_data.get('fdc_id'):
                    print(f"USDA FDC ID: {product_data['fdc_id']}")
                
                print(f"DATA SOURCES: {', '.join(product_data.get('data_sources', []))}")
                print(f"CONFIDENCE: {product_data.get('confidence_score', 0):.1%}")
                
                nutrition = product_data.get('nutrition')
                if nutrition:
                    print("\nNUTRITION FACTS (per serving)")
                    print("-"*40)
                    
                    # Display key nutrients
                    nutrients = [
                        ('Calories', 'calories', ''),
                        ('Total Fat', 'total_fat', 'g'),
                        ('  Saturated Fat', 'saturated_fat', 'g'),
                        ('  Trans Fat', 'trans_fat', 'g'),
                        ('Cholesterol', 'cholesterol', 'mg'),
                        ('Sodium', 'sodium', 'mg'),
                        ('Total Carbohydrates', 'total_carbohydrates', 'g'),
                        ('  Dietary Fiber', 'dietary_fiber', 'g'),
                        ('  Sugars', 'sugars', 'g'),
                        ('Protein', 'protein', 'g')
                    ]
                    
                    for label, key, unit in nutrients:
                        value = nutrition.get(key)
                        if value is not None:
                            print(f"{label}: {value:.1f}{unit}")
                        else:
                            print(f"{label}: N/A")
                    
                    if nutrition.get('data_source'):
                        print(f"\nSource: {nutrition['data_source']}")
                
                if product_data.get('ingredients'):
                    print(f"\nINGREDIENTS: {', '.join(product_data['ingredients'][:10])}")
                    if len(product_data['ingredients']) > 10:
                        print(f"... and {len(product_data['ingredients']) - 10} more")
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")
    
    elif args.mode == 'camera':
        if not CV_AVAILABLE:
            print("OpenCV not available for camera mode")
            return
            
        print("Starting camera capture... Press 'q' to quit, SPACE to capture")
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect barcodes in real-time
            if pipeline.barcode_detector:
                detections = pipeline.barcode_detector.detect_barcodes(frame, enhance=False)
                
                # Draw bounding boxes
                for detection in detections:
                    x, y, w, h = detection.bounding_box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{detection.barcode_type.value}: {detection.barcode_data}"
                    cv2.putText(frame, label, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('Barcode Scanner', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar to capture and analyze
                if detections:
                    print("\nAnalyzing detected barcodes...")
                    result = pipeline.scan_and_analyze(frame)
                    for product in result.get('products', []):
                        print(f"Found: {product.get('name', 'Unknown')} - {product.get('barcode')}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    elif args.mode == 'train':
        print("Starting model training...")
        pipeline.train_models()
        print("Training complete!")
    
    elif args.mode == 'batch':
        print("Batch processing mode")
        barcodes_input = input("Enter barcodes separated by commas: ")
        barcodes = [b.strip() for b in barcodes_input.split(',')]
        
        results = []
        for barcode in barcodes:
            print(f"\nProcessing {barcode}...")
            product = pipeline.data_aggregator.get_product_data(barcode)
            results.append(asdict(product))
            
            print(f"  Name: {product.name}")
            print(f"  Brand: {product.brand}")
            print(f"  Confidence: {product.confidence_score:.1%}")
        
        # Save results
        output_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")
    
    elif args.mode == 'test':
        print("Running system tests...")
        run_tests(pipeline)

def run_tests(pipeline):
    """Run system tests"""
    
    print("\n1. Testing USDA API...")
    test_upcs = [
        "041318020007",  # Kroger Milk
        "078742022475",  # Great Value Water
        "038000845512",  # Kellogg's Cereal
    ]
    
    for upc in test_upcs:
        product = pipeline.data_aggregator.usda_api.search_by_upc(upc)
        if product:
            print(f"✓ Found: {product.name} (FDC ID: {product.fdc_id})")
        else:
            print(f"✗ Not found: {upc}")
    
    print("\n2. Testing barcode detection...")
    if CV_AVAILABLE:
        # Create test image with barcode
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "TEST BARCODE", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        if pipeline.barcode_detector:
            detections = pipeline.barcode_detector.detect_barcodes(test_image, enhance=False)
            print(f"Detected {len(detections)} barcodes")
        else:
            print("Barcode detector not available")
    else:
        print("OpenCV not available for testing")
    
    print("\n3. Testing multi-source aggregation...")
    test_barcode = "0681310084641"  # No Name product
    product = pipeline.data_aggregator.get_product_data(test_barcode)
    print(f"Sources used: {', '.join(product.data_sources)}")
    print(f"Completeness score: {product.confidence_score:.1%}")
    
    print("\nTests complete!")

if __name__ == "__main__":
    main()
