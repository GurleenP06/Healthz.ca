#!/usr/bin/env python3
"""
Tests for healthz.ca scanner module
"""

import pytest
import os
from unittest.mock import patch, Mock
from src.scanner import HealthzScanner, Product, NutritionInfo


class TestHealthzScanner:
    """Test cases for HealthzScanner class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.scanner = HealthzScanner()
    
    def test_scanner_initialization(self):
        """Test scanner initializes correctly"""
        assert self.scanner.api_key is not None
        assert self.scanner.base_url == "https://api.nal.usda.gov/fdc/v1"
    
    def test_scan_barcode_not_found(self):
        """Test scanning a non-existent barcode"""
        # Mock both API calls to return None
        with patch.object(self.scanner, '_search_usda', return_value=None), \
             patch.object(self.scanner, '_search_openfoodfacts', return_value=None):
            result = self.scanner.scan_barcode("000000000000")
            assert result is None
    
    def test_scan_barcode_usda_success(self):
        """Test successful USDA API response"""
        mock_product = Product(
            barcode="123456789",
            name="Test Product",
            brand="Test Brand",
            nutrition=NutritionInfo(calories=100.0, protein=5.0),
            data_source="USDA"
        )
        
        with patch.object(self.scanner, '_search_usda', return_value=mock_product), \
             patch.object(self.scanner, '_search_openfoodfacts', return_value=None):
            result = self.scanner.scan_barcode("123456789")
            
            assert result is not None
            assert result.barcode == "123456789"
            assert result.name == "Test Product"
            assert result.brand == "Test Brand"
            assert result.data_source == "USDA"
            assert result.nutrition.calories == 100.0
            assert result.nutrition.protein == 5.0


class TestProduct:
    """Test cases for Product dataclass"""
    
    def test_product_creation(self):
        """Test creating a Product instance"""
        nutrition = NutritionInfo(calories=150.0, protein=10.0)
        product = Product(
            barcode="123456789",
            name="Test Product",
            brand="Test Brand",
            nutrition=nutrition
        )
        
        assert product.barcode == "123456789"
        assert product.name == "Test Product"
        assert product.brand == "Test Brand"
        assert product.nutrition.calories == 150.0
        assert product.nutrition.protein == 10.0


class TestNutritionInfo:
    """Test cases for NutritionInfo dataclass"""
    
    def test_nutrition_info_creation(self):
        """Test creating a NutritionInfo instance"""
        nutrition = NutritionInfo(
            calories=200.0,
            protein=15.0,
            total_fat=10.0,
            sodium=500.0
        )
        
        assert nutrition.calories == 200.0
        assert nutrition.protein == 15.0
        assert nutrition.total_fat == 10.0
        assert nutrition.sodium == 500.0
        assert nutrition.sugars is None  # Not set, should be None


if __name__ == "__main__":
    pytest.main([__file__])
