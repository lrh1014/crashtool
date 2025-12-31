"""
PaddleOCR 后端服务
启动方式: python ocr_server.py
依赖安装: pip install flask flask-cors paddleocr paddlepaddle opencv-python
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from paddleocr import PaddleOCR
import base64
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

# 初始化 PaddleOCR（首次运行会自动下载模型）
# use_gpu=True 需要安装 paddlepaddle-gpu，否则设为 False
ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False)

@app.route('/ocr', methods=['POST'])
def do_ocr():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': '缺少图片数据'}), 400
        
        # 解析 base64 图片
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': '图片解析失败'}), 400
        
        # OCR 识别
        result = ocr.ocr(img, cls=True)
        
        # 提取文字
        lines = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) > 1:
                    lines.append(line[1][0])
        
        text = '\n'.join(lines)
        return jsonify({'text': text, 'success': True})
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("PaddleOCR 服务启动中...")
    print("访问地址: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
