# app.py - Simplified Flask API using only Pillow
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageOps
import base64
from io import BytesIO
import time

app = Flask(__name__)
CORS(app)

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
        rectangles = detect_rectangles_pillow(image, sensitivity, min_area, max_aspect_ratio)
        processing_time = int((time.time() - start_time) * 1000)
        
        # Return results
        return jsonify({
            "rectangles": rectangles,
            "processing_time": processing_time,
            "image_size": {"width": image.width, "height": image.height},
            "parameters": {
                "sensitivity": sensitivity,
                "min_area": min_area,
                "max_aspect_ratio": max_aspect_ratio
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def decode_base64_image(image_data):
    """Decode base64 image data to PIL format"""
    try:
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def detect_rectangles_pillow(image, sensitivity, min_area, max_aspect_ratio):
    """Detect rectangles using PIL image processing"""
    rectangles = []
    
    # Convert to grayscale
    gray = ImageOps.grayscale(image)
    
    # Apply edge detection using PIL filters
    edges = gray.filter(ImageFilter.FIND_EDGES)
    
    # Apply threshold based on sensitivity
    threshold = 255 - (sensitivity * 2)  # Convert sensitivity to threshold
    edges = edges.point(lambda p: 255 if p > threshold else 0, mode='1')
    
    # Convert back to get pixel data
    edge_pixels = list(edges.getdata())
    width, height = edges.size
    
    # Simple rectangle detection using sliding window
    step_size = 20
    rectangles_found = 0
    
    for y in range(0, height - 50, step_size):
        for x in range(0, width - 50, step_size):
            if rectangles_found >= 10:
                break
                
            # Try different rectangle sizes
            for w in range(40, min(200, width - x), 20):
                for h in range(30, min(150, height - y), 15):
                    area = w * h
                    if area < min_area:
                        continue
                    
                    aspect_ratio = max(w, h) / min(w, h)
                    if aspect_ratio > max_aspect_ratio:
                        continue
                    
                    # Count edge pixels in this rectangle
                    edge_count = 0
                    total_pixels = 0
                    
                    for dy in range(h):
                        for dx in range(w):
                            pixel_idx = (y + dy) * width + (x + dx)
                            if pixel_idx < len(edge_pixels):
                                total_pixels += 1
                                if edge_pixels[pixel_idx]:
                                    edge_count += 1
                    
                    if total_pixels == 0:
                        continue
                        
                    edge_ratio = edge_count / total_pixels
                    
                    # Check if this looks like a rectangle (reasonable edge density)
                    if 0.15 <= edge_ratio <= 0.70:  # 15-70% edges
                        confidence = min(0.95, edge_ratio * 1.5)
                        
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
                        
                        rectangles_found += 1
                        if rectangles_found >= 10:
                            break
                            
                if rectangles_found >= 10:
                    break
        if rectangles_found >= 10:
            break
    
    # Sort by confidence
    rectangles.sort(key=lambda r: r['confidence'], reverse=True)
    return rectangles

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
