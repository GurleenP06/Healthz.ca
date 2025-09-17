# 🍁 healthz.ca

**Canadian Barcode Nutrition Scanner** - Make informed nutrition choices by scanning product barcodes.

[![Tests](https://github.com/GurleenP06/healthz.ca/workflows/Tests/badge.svg)](https://github.com/GurleenP06/healthz.ca/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **Barcode Scanning**: Scan UPC/EAN barcodes to get instant nutrition information
- **Multiple Data Sources**: USDA FoodData Central + OpenFoodFacts fallback
- **Web Interface**: Beautiful Streamlit web app with Canadian theme
- **Command Line Tool**: CLI interface for quick scanning
- **Real-time API**: Live nutrition data from trusted sources
- **Canadian Focus**: Optimized for Canadian products and stores

## 📦 Installation

### Prerequisites
- Python 3.10+
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/GurleenP06/healthz.ca.git
cd healthz.ca

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your USDA API key
```

### Get API Key
1. Visit [USDA FoodData Central](https://fdc.nal.usda.gov/api-guide.html)
2. Sign up for a free API key
3. Add it to your `.env` file:
   ```
   USDA_API_KEY=your_actual_api_key_here
   ```

## 🖥️ Usage

### Web Interface (Recommended)
```bash
# Start the web app
streamlit run app.py
```
Open your browser to `http://localhost:8501`

### Command Line
```bash
# Scan a barcode
python src/scanner.py --barcode 049000006346

# Interactive mode
python src/scanner.py
```

## 🧪 Test Barcodes

Try these known barcodes:
- `049000006346` - Coca-Cola
- `041318020007` - Milk
- `038000845512` - Cereal

## 🏗️ Project Structure

```
healthz.ca/
├── .github/workflows/    # CI/CD
├── src/                  # Source code
│   ├── __init__.py
│   ├── scanner.py        # Main scanner
│   ├── config.py         # Configuration
│   └── utils.py          # Utilities
├── tests/                # Tests
├── data/                 # Data storage
├── models/               # ML models
├── docs/                 # Documentation
├── .env.example          # Example environment
├── .gitignore           # Git ignore rules
├── LICENSE              # MIT License
├── README.md            # Project description
├── requirements.txt     # Dependencies
└── app.py               # Streamlit web app
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v
```

## 🔧 Development

### Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `python -m pytest tests/`
5. Commit changes: `git commit -m "Add feature"`
6. Push to branch: `git push origin feature-name`
7. Create a Pull Request

### Code Quality
- All code is tested with pytest
- CI/CD runs automatically on push/PR
- Follow PEP 8 style guidelines
- Type hints used throughout

## 🌐 API Sources

- **Primary**: [USDA FoodData Central](https://fdc.nal.usda.gov/)
- **Fallback**: [OpenFoodFacts](https://world.openfoodfacts.org/)

## 📊 Nutrition Data

The scanner provides:
- Calories
- Protein
- Total Fat
- Carbohydrates
- Sodium
- And more detailed nutrition facts

## 🍁 Canadian Focus

Optimized for Canadian consumers with:
- Canadian store integration (Loblaws, Metro, IGA)
- Canadian product database
- Maple leaf branding
- Local nutrition standards

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/GurleenP06/healthz.ca/issues) page
2. Create a new issue with detailed information
3. Include barcode numbers and error messages

## 🚀 Roadmap

- [ ] Mobile app development
- [ ] Database caching
- [ ] Nutrition scoring
- [ ] Dietary restriction filtering
- [ ] Shopping list integration
- [ ] Multi-language support

---

**Made with ❤️ in Canada** 🇨🇦