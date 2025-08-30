# app.py - Flask API for rectangle detection
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Rectangle detection API is running"})

@app.route('/detect-rectangles', methods=['POST'])
def detect_rectangles():
    try:
        # Get parameters from request
        data = request.get_json()
        
        # Extract image data (base64 encoded)
        image_data = data.get('image')
        sensitivity = int(data.get('sensitivity', 60))
        min_area = int(data.get('minArea', 1500))
        max_aspect_ratio = float(data.get('aspectRatio', 4))
        
        # Decode base64 image
        image = decode_base64_image(image_data)
        if image is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        # Perform rectangle detection
        start_time = time.time()
        rectangles = detect_rectangles_cv2(image, sensitivity, min_area, max_aspect_ratio)
        processing_time = int((time.time() - start_time) * 1000)
        
        # Return results
        return jsonify({
            "rectangles": rectangles,
            "processing_time": processing_time,
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "parameters": {
                "sensitivity": sensitivity,
                "min_area": min_area,
                "max_aspect_ratio": max_aspect_ratio
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def decode_base64_image(image_data):
    """Decode base64 image data to OpenCV format"""
    try:
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        pil_image = Image.open(BytesIO(image_bytes))
        
        # Convert to OpenCV format (BGR)
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def detect_rectangles_cv2(image, sensitivity, min_area, max_aspect_ratio):
    """Detect rectangles using OpenCV"""
    rectangles = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, sensitivity, sensitivity * 2)
    
    # Apply morphological operations to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, contour in enumerate(contours):
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        if area < min_area:
            continue
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > max_aspect_ratio:
            continue
        
        # Calculate confidence based on how well the contour fits a rectangle
        rect_area = w * h
        extent = area / rect_area
        
        # Filter based on extent (how much of bounding rectangle is filled)
        if extent > 0.5:  # At least 50% filled
            confidence = min(0.95, extent)
            
            rectangles.append({
                "id": f"RECT-{len(rectangles) + 1}",
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "area": int(area),
                "aspect_ratio": round(aspect_ratio, 2),
                "confidence": round(confidence, 2),
                "center": {
                    "x": int(x + w // 2),
                    "y": int(y + h // 2)
                }
            })
    
    # Sort by confidence (highest first)
    rectangles.sort(key=lambda r: r['confidence'], reverse=True)
    
    # Return top 10 results
    return rectangles[:10]

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
