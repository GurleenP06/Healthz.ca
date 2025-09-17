#!/usr/bin/env python3
"""
healthz.ca - Barcode Nutrition Scanner
Main scanner module for barcode detection and nutrition lookup
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class NutritionInfo:
    """Nutrition information for a product"""
    calories: Optional[float] = None
    protein: Optional[float] = None
    total_fat: Optional[float] = None
    saturated_fat: Optional[float] = None
    trans_fat: Optional[float] = None
    cholesterol: Optional[float] = None
    sodium: Optional[float] = None
    total_carbohydrates: Optional[float] = None
    dietary_fiber: Optional[float] = None
    sugars: Optional[float] = None
    serving_size: Optional[str] = None


@dataclass
class Product:
    """Product information"""
    barcode: str
    name: str
    brand: Optional[str] = None
    category: Optional[str] = None
    nutrition: Optional[NutritionInfo] = None
    data_source: Optional[str] = None
    last_updated: Optional[datetime] = None


class HealthzScanner:
    """Main scanner class for healthz.ca"""
    
    def __init__(self):
        self.api_key = os.getenv('USDA_API_KEY', 'DEMO_KEY')
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        
        if self.api_key == 'DEMO_KEY':
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
        
        # Try USDA first
        product = self._search_usda(barcode)
        
        if not product:
            # Try OpenFoodFacts as fallback
            product = self._search_openfoodfacts(barcode)
        
        return product
    
    def _search_usda(self, barcode: str) -> Optional[Product]:
        """Search USDA FoodData Central"""
        try:
            url = f"{self.base_url}/foods/search"
            params = {
                'api_key': self.api_key,
                'query': barcode,
                'dataType': ['Branded'],
                'pageSize': 1
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('foods'):
                food = data['foods'][0]
                
                # Extract nutrition
                nutrition = NutritionInfo()
                for nutrient in food.get('foodNutrients', []):
                    name = nutrient.get('nutrientName', '').lower()
                    value = nutrient.get('value')
                    
                    if 'energy' in name and 'kcal' in name.lower():
                        nutrition.calories = value
                    elif name == 'protein':
                        nutrition.protein = value
                    elif 'total lipid' in name:
                        nutrition.total_fat = value
                    elif 'carbohydrate' in name:
                        nutrition.total_carbohydrates = value
                    elif 'sodium' in name:
                        nutrition.sodium = value
                
                return Product(
                    barcode=barcode,
                    name=food.get('description', 'Unknown'),
                    brand=food.get('brandOwner'),
                    category=food.get('foodCategory'),
                    nutrition=nutrition,
                    data_source='USDA',
                    last_updated=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"USDA search failed: {e}")
        
        return None
    
    def _search_openfoodfacts(self, barcode: str) -> Optional[Product]:
        """Search OpenFoodFacts as fallback"""
        try:
            url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 1:
                    product_data = data['product']
                    
                    nutrition = NutritionInfo(
                        calories=product_data.get('nutriments', {}).get('energy-kcal_100g'),
                        protein=product_data.get('nutriments', {}).get('proteins_100g'),
                        total_fat=product_data.get('nutriments', {}).get('fat_100g'),
                        sodium=product_data.get('nutriments', {}).get('sodium_100g'),
                        total_carbohydrates=product_data.get('nutriments', {}).get('carbohydrates_100g')
                    )
                    
                    return Product(
                        barcode=barcode,
                        name=product_data.get('product_name', 'Unknown'),
                        brand=product_data.get('brands'),
                        category=product_data.get('categories'),
                        nutrition=nutrition,
                        data_source='OpenFoodFacts',
                        last_updated=datetime.now()
                    )
                    
        except Exception as e:
            logger.error(f"OpenFoodFacts search failed: {e}")
        
        return None


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='healthz.ca Barcode Scanner')
    parser.add_argument('--barcode', type=str, help='Barcode to scan')
    args = parser.parse_args()
    
    scanner = HealthzScanner()
    
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
        
        print(f"\nSource: {product.data_source}")
        print(f"{'='*50}\n")
    else:
        print("Product not found")


if __name__ == "__main__":
    main()
