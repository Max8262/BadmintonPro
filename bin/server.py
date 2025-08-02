from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import openai
import pandas as pd
import base64
from moviepy.editor import VideoFileClip, vfx
import os
import subprocess
import time
import json
import csv
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='processed_results')
CORS(app)  # 允許跨來源請求

# Manual CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# 設置存儲上傳文件的文件夾
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'Processed'  # 現有的處理文件夾
RESULTS_FOLDER = 'analyzed_videos'  # 存放結果的文件夾
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# OpenAI API Key - 確保在環境變數中設置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 確保文件夾存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(os.path.join(RESULTS_FOLDER, 'videos'), exist_ok=True)

def slowdown(file_id):
    folder = f"analyzed_videos/videos/{file_id}"
    slow_factor = 3.0  # 2x slower

    for filename in os.listdir(folder):
        if filename.endswith(".mp4"):
            filepath = os.path.join(folder, filename)
            print(filepath)
            clip = VideoFileClip(filepath)
            slow_clip = clip.fx(vfx.speedx, 1/slow_factor)
            slow_clip.write_videofile(os.path.join(folder, f"{filename}"))

# 檢查允許的文件擴展名
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_openai_advice(pose_type, errors):
    """使用OpenAI提供個人化的技術建議"""
    if not OPENAI_API_KEY:
        return None
    
    try:
        # 中文姿勢名稱對映
        pose_names = {
            'clear': '高遠球/長球',
            'net': '放網/網前球', 
            'drive': '平球/平抽球',
            'serve': '發球',
            'lift': '挑球',
        }
        
        pose_chinese = pose_names.get(pose_type.lower(), pose_type)
        errors_text = ', '.join(errors) if errors else '無明顯錯誤'
        
        Text_User_Settings = """
        你是一名羽球專業教練，您會根據使用者描述的姿勢和使用者的錯誤給出建議以及改善的方法，使用者是新手
        因此，您需要先研究羽毛球姿勢和常見錯誤，主要有五種姿勢：
        長球/高遠球 (clear)
            1.側身不完全 --> 身體轉動幅度不夠，無法有效利用身體力量帶動擊球。
            2.輔助手蜷縮 --> 手臂無伸展，手臂與身體夾腳<90度、手臂夾腳<70度，無法有效利用身體帶動擊球。
            3.擊球手位置過低 --> 大臂和身體夾角過小，無法在最高點擊球。
        平球 (drive)
            1.球拍未舉起 --> 擊球後，球拍隨即放下，卻未能立刻抬起以準備下一球。
            2.身體直立 --> 身體沒有稍微前傾壓低重心預備
        挑球 (lift)
            1.膝蓋超伸 --> 膝蓋超出腳尖，腿部夾腳<90度，身體壓力集中在膝蓋上。
            2.無慣用腳在前 --> 擊球手和前腳不同
            3.擊球手引拍過大 --> 擊球前向後引拍過大
        放網 (net)
            1.手臂蜷縮 --> 球拍距離身體太近沒有伸展，手臂夾角<120度。
            2.出手點過低 --> 球拍低於胸口，沒有在高點擊球。
            3.膝蓋超伸 --> 膝蓋超出腳尖學腿部夾腳<90度
            4.無慣用腳在前 --> 擊球手和前腳不同
        發球 (serve)
            1.無慣用腳在前 --> 發球時必須慣用腳在前，非慣用腳在後。
            2. 發球位置不當 --> 發球時擊球點應位於腰部附近。
        
        請為使用者提供有用的繁體中文答案(兩大點，不要長篇大論，簡潔有力：
        1. 為什麼這些錯誤會影響技術表現？
        2. 具體的改善方法和練習建議。)
        要有高清晰度、好的效果、高相關性、且高可行性，不要有廢話，且專業一點。
        """
        
        prompt = f"""{pose_chinese} --> {errors_text}。請提供具體的調整建議。"""
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": Text_User_Settings},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"OpenAI API 錯誤: {e}")
        return None

@app.route('/upload', methods=['POST'])
def upload_file():
    print(f"Upload request received from {request.remote_addr}")
    print(f"Files in request: {list(request.files.keys())}")
    
    # 檢查是否有文件在請求中
    if 'video' not in request.files:
        print("Error: No video file in request")
        return jsonify({'status': 'error', 'message': '沒有找到文件'}), 400
    
    file = request.files['video']
    print(f"File received: {file.filename}, size: {len(file.read())} bytes")
    file.seek(0)  # Reset file pointer after reading
    
    # 檢查文件是否有名稱
    if file.filename == '':
        print("Error: Empty filename")
        return jsonify({'status': 'error', 'message': '沒有選擇文件'}), 400
    
    # 檢查文件類型
    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': '不支持的文件類型'}), 400
    
    # 安全地保存文件
    filename = secure_filename(file.filename)
    # 為文件添加時間戳，避免文件名衝突
    timestamp = int(time.time())
    base_name, ext = os.path.splitext(filename)
    unique_filename = f"{base_name}_{timestamp}{ext}"
    upload_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(upload_path)
    
    # 返回文件ID（使用時間戳作為ID）
    file_id = str(timestamp)
    
    # 啟動處理流程（異步執行）
    # 實際應用中，您可能希望使用Celery或其他任務隊列系統
    process_video(upload_path, file_id)
    
    return jsonify({
        'status': 'success', 
        'message': '文件已成功上傳', 
        'file_id': file_id
    })

@app.route('/status/<file_id>', methods=['GET'])
def check_status(file_id):
    # 檢查處理狀態
    # 在實際應用中，您應該從數據庫或緩存中獲取狀態
    # 這裡我們使用一個簡單的JSON文件模擬
    status_path = os.path.join(RESULTS_FOLDER, f'status_{file_id}.json')
    
    if os.path.exists(status_path):
        with open(status_path, 'r', encoding='utf-8') as f:
            status = json.load(f)
        return jsonify(status)
    else:
        return jsonify({'status': 'pending', 'progress': 0, 'message': '等待處理'})

@app.route('/results/<file_id>', methods=['GET'])
def get_results(file_id):
    # 返回處理結果
    result_path = os.path.join(RESULTS_FOLDER, f'result_{file_id}.json')
    
    if os.path.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return jsonify(results)
    else:
        return jsonify({'status': 'error', 'message': '找不到處理結果'}), 404

@app.route('/videos/<file_id>/<video_name>', methods=['GET'])
def get_video(file_id, video_name):
    # 返回處理後的影片
    video_dir = os.path.join(RESULTS_FOLDER, 'videos', file_id)
    full_path = os.path.join(video_dir, video_name)
    
    print(f"Requesting video: {file_id}/{video_name}")
    print(f"Video directory: {video_dir}")
    print(f"Full path: {full_path}")
    print(f"Directory exists: {os.path.exists(video_dir)}")
    print(f"Video exists: {os.path.exists(full_path)}")
    
    # 列出目錄內容以便調試
    if os.path.exists(video_dir):
        print(f"Files in directory: {os.listdir(video_dir)}")
    
    if not os.path.exists(full_path):
        print(f"Video file not found: {full_path}")
        return jsonify({'error': 'Video not found', 'path': full_path}), 404
    
    try:
        return send_from_directory(video_dir, video_name, as_attachment=False, mimetype='video/mp4')
    except Exception as e:
        print(f"Error serving video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def update_status(file_id, status, progress, message):
    """更新處理狀態"""
    status_data = {
        'status': status,
        'progress': progress,
        'message': message,
        'file_id': file_id
    }
    
    status_path = os.path.join(RESULTS_FOLDER, f'status_{file_id}.json')
    with open(status_path, 'w', encoding='utf-8') as f:
        json.dump(status_data, f, ensure_ascii=False)

def process_video(video_path, file_id):
    """處理上傳的視頻文件（按照羽球分析流程）"""
    try:
        # 更新狀態：開始處理
        update_status(file_id, 'processing', 10, '正在進行羽球跟蹤...')
        
        # 複製視頻到處理文件夾，重命名為DEMO_VIDEO.mp4
        demo_path = os.path.join('DEMO_VIDEO.mp4')
        shutil.copy(video_path, demo_path)
        
        # 步驟1: 執行Badminton_Tracker_and_Auto_Editer.py
        subprocess.run(['python', 'Badminton_Tracker_and_Auto_Editer.py'], check=True)
        update_status(file_id, 'processing', 40, '羽球跟蹤完成，正在進行姿勢分類...')
        
        # 步驟2: 執行Pose_Classifier.py
        subprocess.run(['python', 'Pose_Classifier.py'], check=True)
        update_status(file_id, 'processing', 70, '姿勢分類完成，正在進行錯誤檢測...')
        
        # 步驟3: 執行Pose_Error_Detection.py
        subprocess.run(['python', 'Pose_Error_Detection.py', '--input', 'Processed', '--config', 'Processed.csv', '--output', os.path.join(RESULTS_FOLDER, 'videos'), '--report', 'Report.csv', '--no-display', '--ground-truth', 'proplayermove'], check=True)
        update_status(file_id, 'processing', 85, '錯誤檢測完成，正在生成AI建議...')
        
        # 創建一個特定於此file_id的視頻文件夾
        video_output_dir = os.path.join(RESULTS_FOLDER, 'videos', file_id)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # 移動所有處理後的影片到對應的文件夾
        analyzed_files = [f for f in os.listdir(os.path.join(RESULTS_FOLDER, 'videos')) if f.startswith('analyzed_real_time_part') and f.endswith('.mp4')]
        for analyzed_file in analyzed_files:
            original_path = os.path.join(RESULTS_FOLDER, 'videos', analyzed_file)
            new_path = os.path.join(video_output_dir, analyzed_file)
            shutil.move(original_path, new_path)

        slowdown(file_id)
        # 讀取報告數據並為每個視頻生成AI建議
        video_results = []
        processed_error_combinations = set()  # 用於追蹤已處理的錯誤組合
        
        if os.path.exists('Report.csv'):
            # 優先使用Big5編碼（台灣系統預設）
            encodings = ['big5', 'cp950', 'gbk', 'utf-8', 'latin1']
            
            for encoding in encodings:
                try:
                    with open('Report.csv', 'r', encoding=encoding) as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)  # 讀取所有行
                        
                        print(f"Successfully read Report.csv with {encoding} encoding")
                        print(f"Total rows in Report.csv: {len(rows)}")
                        
                        for row in rows:
                            video_name = row.get('影片名稱', '')
                            pose_type = row.get('姿勢類型', '未知')
                            errors_str = row.get('所有錯誤（如視頻中顯示）', '')
                            errors = [err.strip() for err in errors_str.split(';') if err.strip()]
                            
                            # 創建唯一的錯誤組合標識符（姿勢類型 + 排序後的錯誤列表）
                            error_combination = (pose_type, tuple(sorted(errors)))
                            
                            # 找到對應的analyzed視頻文件
                            # real_time_part1.mp4 -> analyzed_real_time_part1.mp4
                            base_name = os.path.splitext(video_name)[0]
                            analyzed_video_name = f"analyzed_{base_name}.mp4"
                            
                            # 確認視頻文件存在
                            video_path = os.path.join(video_output_dir, analyzed_video_name)
                            if os.path.exists(video_path):
                                # 檢查是否已經為相同的錯誤組合生成過建議
                                if error_combination not in processed_error_combinations:
                                    # 獲取OpenAI建議
                                    print(f"Getting OpenAI advice for {video_name} - pose: {pose_type}, errors: {errors}")
                                    openai_advice = get_openai_advice(pose_type, errors)
                                    processed_error_combinations.add(error_combination)
                                    
                                    # 添加短暫延遲以避免API速率限制
                                    if openai_advice:
                                        time.sleep(1)
                                else:
                                    # 使用空建議或跳過重複的錯誤組合
                                    print(f"Skipping duplicate error combination for {video_name} - pose: {pose_type}, errors: {errors}")
                                    openai_advice = None
                                
                                video_result = {
                                    'original_name': video_name,
                                    'analyzed_name': analyzed_video_name,
                                    'pose_type': pose_type,
                                    'errors': errors,
                                    'video_url': f'/videos/{file_id}/{analyzed_video_name}',
                                    'openai_advice': openai_advice
                                }
                                video_results.append(video_result)
                            else:
                                print(f"Warning: Analyzed video not found: {video_path}")
                    
                    break  # 成功讀取後跳出編碼循環
                except UnicodeDecodeError:
                    print(f"Failed to read with {encoding} encoding, trying next...")
                    continue
                except Exception as e:
                    print(f"Error reading Report.csv with {encoding}: {e}")
                    continue
            else:
                print("Could not read Report.csv with any encoding")
                video_results = []
        
        update_status(file_id, 'processing', 95, '正在整理結果...')
        
        # 創建結果JSON
        results = {
            'status': 'completed',
            'file_id': file_id,
            'total_videos': len(video_results),
            'videos': video_results,
            'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存結果
        result_path = os.path.join(RESULTS_FOLDER, f'result_{file_id}.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 更新最終狀態
        update_status(file_id, 'completed', 100, '處理完成')
        
        print(f"Processing completed. Total videos: {len(video_results)}")
        
    except Exception as e:
        # 處理錯誤
        print(f"處理視頻時出錯: {e}")
        import traceback
        traceback.print_exc()
        update_status(file_id, 'error', 0, f'處理失敗: {str(e)}')

@app.route('/test_video_simple.html')
def test_video_simple():
    return send_from_directory('.', 'test_video_simple.html')

# Add a direct static file serving route for videos
@app.route('/analyzed_videos/videos/<path:filename>')
def serve_video_direct(filename):
    """Serve video files directly from the analyzed_videos directory"""
    try:
        video_path = os.path.join(RESULTS_FOLDER, 'videos', filename)
        directory = os.path.dirname(video_path)
        file_name = os.path.basename(video_path)
        
        print(f"Direct video request: {filename}")
        print(f"Directory: {directory}")
        print(f"File: {file_name}")
        print(f"Full path: {video_path}")
        print(f"Exists: {os.path.exists(video_path)}")
        
        if os.path.exists(video_path):
            return send_from_directory(directory, file_name, mimetype='video/mp4')
        else:
            return f"File not found: {video_path}", 404
    except Exception as e:
        print(f"Error serving direct video: {e}")
        import traceback
        traceback.print_exc()
        return str(e), 500

@app.route('/video-debug.html')
def video_debug_page():
    # 提供視頻測試頁面
    return send_from_directory('.', 'video_debug.html')

@app.route('/video-test')
def video_test_page():
    # 提供影片測試頁面
    return send_from_directory('.', 'video_test.html')

@app.route('/test')
def test_page():
    # 提供測試頁面
    return send_from_directory('.', 'test_upload.html')

@app.route('/')
def index():
    # 提供主頁面 - English version as default
    return send_from_directory('.', 'index_en.html')

@app.route('/english')
def english_version():
    # English version
    return send_from_directory('.', 'index_en.html')

@app.route('/chinese')
def chinese_version():
    # Chinese version
    return send_from_directory('.', 'index_ch.html')

# 為了使手機能夠訪問，需要提供靜態文件的服務
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    # 設置環境變數
    try:
        exec(open('setup_env.py').read())
    except FileNotFoundError:
        print("未找到 setup_env.py，跳過環境設置")
    except Exception as e:
        print(f"環境設置錯誤: {e}")
    
    print("=== 羽球分析服務器啟動 ===")
    print(f"上傳資料夾: {UPLOAD_FOLDER}")
    print(f"處理資料夾: {PROCESSED_FOLDER}")
    print(f"結果資料夾: {RESULTS_FOLDER}")
    print("服務器將在以下地址啟動:")
    print("- http://localhost:5000")
    print("- http://127.0.0.1:5000")
    print("請在瀏覽器中訪問上述地址")
    
    # 檢查OpenAI設置
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OpenAI AI教練功能已啟用")
    else:
        print("⚠️ OpenAI AI教練功能未啟用（無API Key）")
    
    print("=" * 40)
    
    # 啟動Flask應用
    # 使用0.0.0.0使其能夠從外部網絡訪問
    app.run(host='0.0.0.0', port=5000, debug=True)