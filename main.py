import os
import sys
import subprocess
import csv
from datetime import datetime
import re
import cv2
import numpy as np
from PIL import Image
import pytesseract

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")

# Check Tesseract installation
try:
    tesseract_version = subprocess.check_output(['tesseract', '--version'], stderr=subprocess.STDOUT).decode().strip()
    print(f"Tesseract version: {tesseract_version}")
except FileNotFoundError:
    print("Error: Tesseract is not installed or not in PATH. Please install it using 'sudo apt install tesseract-ocr'")
    sys.exit(1)

# Set Tesseract command path explicitly
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Set up file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
CSV_DIR = os.path.join(BASE_DIR, 'csv')
DATA_DIR = os.path.join(BASE_DIR, 'data')

print(f"BASE_DIR: {BASE_DIR}")
print(f"IMAGES_DIR: {IMAGES_DIR}")
print(f"DATA_DIR: {DATA_DIR}")

# Ensure output directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def load_item_data(csv_file):
    item_data = {}
    if not os.path.exists(csv_file):
        print(f"Warning: CSV file {csv_file} not found.")
        return item_data

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_data[row['Name']] = row['Type']

    if not item_data:
        print(f"Warning: No data loaded from {csv_file}. The file may be empty.")

    return item_data

def load_column_names(csv_file):
    column_names = set()
    if not os.path.exists(csv_file):
        print(f"Warning: Column names CSV file {csv_file} not found.")
        return column_names
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            column_names.add(row['Name'].lower())
    
    return column_names

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def validate_item_name(name, item_data, column_names):
    if name.lower() in column_names:
        return "", ""  # Return empty name and type if it matches a column name
    
    if not item_data:
        return name, ""  # Return original name and empty type if item_data is empty

    # Check if the exact name exists in our data
    if name in item_data:
        return name, item_data[name]

    # If not, find the closest match
    closest_match = min(item_data.keys(), key=lambda x: levenshtein_distance(name, x))
    if levenshtein_distance(name, closest_match) <= 3:  # Arbitrary threshold, adjust as needed
        return closest_match, item_data[closest_match]

    return name, ""  # If no close match found, return original name and empty type

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance contrast in the top part of the image
    top_third = gray[:gray.shape[0]//3, :]
    enhanced_top = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(top_third)
    gray[:gray.shape[0]//3, :] = enhanced_top

    # Apply simple thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save preprocessed image for debugging
    debug_path = os.path.join(DATA_DIR, 'preprocessed_' + os.path.basename(image_path))
    cv2.imwrite(debug_path, binary)
    print(f"Saved preprocessed image to: {debug_path}")

    return binary

def perform_ocr(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Perform OCR on the preprocessed image
    text = pytesseract.image_to_string(preprocessed_image)

    # Print OCR output for debugging
    print("OCR Output:")
    print(text)
    print("-" * 50)

    # Additional debugging
    lines = text.split('\n')
    if lines:
        print(f"First line of OCR output: {lines[0]}")
    else:
        print("Warning: OCR output is empty")

    return text

def parse_attributes(text, item_data, column_names):
    attributes = {
        'Name': '',
        'Item Type': '',
        'Is Radiant': False,
        'Item Quality': '',
        'Grade': '',
        'Max Rune': 0,
        'Attack Power': 0,
        'Physical Defense Bonus': 0,
        'Dagger Defense': 0,
        'Sword Defense': 0,
        'Mace Defense': 0,
        'Axe Defense': 0,
        'Spear Defense': 0,
        'Bow Defense': 0,
        'Mirror Damage': 0,
        'Poison Damage': 0,
        'Fire Damage': 0,
        'Ice Damage': 0,
        'Lightning Damage': 0,
        'Holy Damage': 0,
        'HP Leech': 0,
        'Mana Burn': 0,
        'Strength Bonus': 0,
        'Health Bonus': 0,
        'Dexterity Bonus': 0,
        'Intelligence Bonus': 0,
        'Magic Bonus': 0,
        'HP Bonus': 0,
        'MP Bonus': 0,
        'Fire Resistance': 0,
        'Ice Resistance': 0,
        'Lightning Resistance': 0,
        'Holy Damage Resistance': 0,
        'Poison Damage Resistance': 0,
        'Curse Damage Resistance': 0,
        'Required Magic': 0,
        'Required Intelligence': 0,
        'Required HP': 0,
        'Required Strength': 0,
        'Required Dexterity': 0,
        'Required Level': 0,
        'Durability': 0,
        'Weight': 0,
        'Description': ''
    }

    if not text.strip():
        print("Warning: No text extracted from the image.")
        return attributes

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    print("Debug: All lines from OCR:")
    for i, line in enumerate(lines):
        print(f"Line {i + 1}: {line}")

    # Extract name from the first line that's not "Runes can't be added to this item"
    for line in lines:
        if line != "Runes can't be added to this item":
            attributes['Name'] = line
            print(f"Debug: Potential item name found: {line}")
            break

    # Validate and correct the name using item_data
    if attributes['Name']:
        closest_match = min(item_data.keys(), key=lambda x: levenshtein_distance(attributes['Name'].lower(), x.lower()))
        distance = levenshtein_distance(attributes['Name'].lower(), closest_match.lower())
        print(f"Debug: Closest match: {closest_match}, Distance: {distance}")
        if distance <= 5:  # Increased threshold
            attributes['Name'] = closest_match
            attributes['Item Type'] = item_data[closest_match]
        else:
            print(f"Debug: No close match found for {attributes['Name']}")

    print(f"Extracted Name: {attributes['Name']}")
    print(f"Extracted Item Type: {attributes['Item Type']}")

    # Extract other attributes
    for line in lines:
        if "Runes can't be added to this item" in line:
            attributes['Is Radiant'] = True
            print("Item is Radiant")

        for attr in attributes:
            if attr.lower() in line.lower():
                value = re.search(r'\d+', line)
                if value:
                    attributes[attr] = int(value.group())
                    print(f"Extracted {attr}: {attributes[attr]}")

        # Special case for Item Quality and Grade
        if 'quality' in line.lower():
            attributes['Item Quality'] = line.split(':')[-1].strip()
            print(f"Extracted Item Quality: {attributes['Item Quality']}")
        if 'grade' in line.lower():
            attributes['Grade'] = line.split(':')[-1].strip()
            print(f"Extracted Grade: {attributes['Grade']}")

    # Extract description (last line that's not a numeric attribute or special phrase)
    for line in reversed(lines):
        if not any(attr.lower() in line.lower() for attr in attributes) and line != "Runes can't be added to this item":
            attributes['Description'] = line
            print(f"Extracted Description: {attributes['Description']}")
            break

    return attributes

def main():
    # Load item data from CSV files
    item_data = {}
    csv_files = ['extracted_items.csv', 'anklets_items.csv']
    for csv_file in csv_files:
        file_path = os.path.join(CSV_DIR, csv_file)
        item_data.update(load_item_data(file_path))

    if not item_data:
        print("Error: No item data loaded. Please check the CSV files in the 'csv' directory.")
        sys.exit(1)

    # Load column names
    column_names = load_column_names(os.path.join(CSV_DIR, 'column-names.csv'))
    if not column_names:
        print("Warning: No column names loaded. Item name validation against column names will be skipped.")

    if not os.path.exists(IMAGES_DIR):
        print(f"Error: The 'images' directory does not exist in {BASE_DIR}")
        print("Please create an 'images' directory and add some images to process.")
        sys.exit(1)

    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"Error: No image files found in {IMAGES_DIR}")
        print("Please add some image files (PNG, JPG, or JPEG) to the 'images' directory.")
        sys.exit(1)

    # Create a timestamp for the output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(DATA_DIR, f"ocr_output_{timestamp}.csv")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = list(parse_attributes('', {}, set()).keys())
        fieldnames.insert(0, 'Filename')
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()

        for filename in image_files:
            image_path = os.path.join(IMAGES_DIR, filename)
            extracted_text = perform_ocr(image_path)

            # Parse attributes
            attributes = parse_attributes(extracted_text, item_data, column_names)

            # Prepare row for CSV
            row = {'Filename': filename, **attributes}

            # Write to CSV
            csvwriter.writerow(row)

            # Print to console
            print(f"Processed {filename}:")
            for key, value in attributes.items():
                print(f"{key}: {value}")
            print("-" * 50)

    print(f"OCR results have been saved to: {output_file}")

if __name__ == "__main__":
    main()