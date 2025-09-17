import pytest
from src.scanner import HealthzScanner

def test_scanner_initialization():
    scanner = HealthzScanner()
    assert scanner is not None

def test_known_barcode():
    scanner = HealthzScanner()
    # Test with Coca-Cola barcode
    product = scanner.scan_barcode("049000006346")
    assert product is not None
    assert "coca" in product.name.lower() or "coke" in product.name.lower()
