# rise-itemizer
# RiseOCR

**RiseOCR** is an advanced OCR-based tool designed to extract and validate in-game item attributes from images. The tool is specifically built for the MMORPG game Rise Online and ensures accurate data extraction by leveraging preprocessing techniques and robust name validation.

## Features

- **OCR Extraction**: Extracts in-game item names and attributes from images using Tesseract OCR.
- **Preprocessing**: Enhances image quality for better OCR accuracy.
- **Attribute Matching**: Matches extracted item names with known names using Levenshtein distance.
- **Categorization**: Stores item data in a structured CSV format, ready for display on a PHP-based website.
- **Error Handling**: Handles common OCR errors and assigns default values where data is missing.

## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/riseocr.git
cd riseocr
pip install -r requirements.txt


