# ============================================================================
# INSTALLATION AND IMPORT OF LIBRARIES
# ============================================================================
print("🔄 Installing and importing libraries...")

try:
    import sys
    import subprocess
    import importlib

    required_packages = ['deepface', 'opencv-python', 'matplotlib', 'pillow', 'numpy']

    for package in required_packages:
        try:
            if package == 'deepface':
                importlib.import_module('deepface')
            elif package == 'opencv-python':
                importlib.import_module('cv2')
            else:
                importlib.import_module(package)
            print(f"✅ {package} is already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

except Exception as e:
    print(f"⚠️ Installation error: {e}")
#    !pip install deepface opencv-python matplotlib pillow numpy -q

from google.colab import output
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
from google.colab import files
from google.colab.patches import cv2_imshow
import os
from IPython.display import display, Javascript, HTML, clear_output
import time
from PIL import Image
import io
import base64
import warnings
import csv

warnings.filterwarnings('ignore')

print("✅ All libraries successfully imported!")

# -*- coding: utf-8 -*-
"""Student emotion recognition program from video (Version 2.1).ipynb

System for recognizing students' emotional states
Project No. 97 - Temporary Creative Student Group
"""

# ============================================================================
# PROGRAM CONFIGURATION
# ============================================================================
class Config:
    """Program configuration"""
    # Analysis settings
    DEFAULT_FRAME_INTERVAL = 10
    MIN_FACE_CONFIDENCE = 0.5
    EMO_MODEL = 'Emotion'

    # Visualization colors
    COLORS = {
        'Happiness': '#FFD700',     # gold
        'Sadness': '#4169E1',       # royal blue
        'Anger': '#DC143C',         # crimson
        'Surprise': '#FF8C00',      # dark orange
        'Fear': '#9370DB',          # medium purple
        'Disgust': '#32CD32',       # lime green
        'Neutral': '#808080'        # gray
    }

    # Engagement thresholds
    HIGH_ENGAGEMENT = 70
    MEDIUM_ENGAGEMENT = 40

    EMOTION_TRANSLATION = {
        'angry': 'Anger',
        'disgust': 'Disgust',
        'fear': 'Fear',
        'happy': 'Happiness',
        'sad': 'Sadness',
        'surprise': 'Surprise',
        'neutral': 'Neutral'
    }

# ============================================================================
# FILE UPLOAD FUNCTION
# ============================================================================
def upload_file(file_type='video'):
    """
    Uploads a file in Colab
    file_type: 'video' or 'image'
    """
    file_description = 'video file' if file_type == 'video' else 'image'

    print(f"📁 Please select a {file_description} for analysis...")
    print("   Supported formats:")

    if file_type == 'video':
        print("   - MP4, AVI, MOV, WMV (MP4 recommended)")
    else:
        print("   - JPG, PNG, JPEG (JPG recommended)")

    uploaded = files.upload()

    if not uploaded:
        print("❌ No file was uploaded")
        return None

    for filename in uploaded.keys():
        file_size = uploaded[filename].__len__() / (1024 * 1024)
        print(f"✅ File '{filename}' uploaded successfully ({file_size:.2f} MB)")
        return filename

    return None

# ============================================================================
# CAMERA RECORDING FUNCTION
# ============================================================================
def capture_from_camera(duration=5, fps=10):
    """
    Records video from webcam using JavaScript.
    duration: recording duration in seconds
    fps: frames per second
    """
    print(f"📷 Preparing camera recording...")
    print(f"   Duration: {duration} sec")
    print(f"   Frame rate: {fps} FPS")

    # JavaScript for video capture
    js_code = f"""
    async function captureVideo() {{
        const video = document.createElement('video');
        const canvas = document.createElement('canvas');
        const stream = await navigator.mediaDevices.getUserMedia({{
            video: {{
                width: 640,
                height: 480
            }},
            audio: false
        }});

        video.srcObject = stream;
        document.body.appendChild(video);
        await video.play();

        const frames = [];
        const totalFrames = {duration} * {fps};
        const interval = 1000 / {fps}; // delay between frames

        for (let i = 0; i < totalFrames; i++) {{
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);

            const frame = canvas.toDataURL('image/jpeg', 0.85);
            frames.push(frame);

            if (i < totalFrames - 1) {{
                await new Promise(r => setTimeout(r, interval));
            }}
        }}

        // Stop camera
        stream.getTracks().forEach(track => track.stop());
        document.body.removeChild(video);

        return frames;
    }}

    captureVideo().then(frames => {{
        window.cameraFrames = frames;
        console.log('Frames captured: ' + frames.length);
    }});
    """

    display(Javascript(js_code))  # Execute JS code for video capture

    print(f"⏳ Waiting for recording to start...")

    # Wait for recording completion and frame retrieval
    frames = None
    start_time = time.time()
    while time.time() - start_time < duration + 10:
        try:
            frames = output.eval_js("window.cameraFrames")  # Get frames
            if frames:
                print(f"✅ Recording finished! Frames received: {len(frames)}")
                break
        except Exception as e:
            print(f"⚠️ Error while retrieving frames: {e}")
            pass

        if time.time() - start_time > 5:
            print(f"   Waiting for recording to finish... ({int(time.time() - start_time)} sec)")

        time.sleep(1)

    if not frames or len(frames) == 0:
        print("❌ Failed to record video from camera")
        print("💡 Check camera permissions")
        return None

    # Save frames
    temp_dir = "/camera_frames"
    os.makedirs(temp_dir, exist_ok=True)

    frame_paths = []
    for i, frame_data in enumerate(frames):
        try:
            img_data = frame_data.split(',')[1]
            img_bytes = io.BytesIO(base64.b64decode(img_data))
            img = Image.open(img_bytes)

            frame_path = f"{temp_dir}/frame_{i:04d}.jpg"
            img.save(frame_path, quality=90)
            frame_paths.append(frame_path)

        except Exception as e:
            print(f"⚠️ Error saving frame {i}: {e}")

    print(f"📸 Saved {len(frame_paths)} frames to {temp_dir}")

    # Check if frames were saved
    if len(frame_paths) == 0:
        print("❌ Error: no frames were saved correctly.")
        return None

    return frame_paths

# ============================================================================
# SAVE FRAMES TO MP4 VIDEO
# ============================================================================
def save_frames_to_video(frame_paths, output_file="camera_recording.mp4", fps=10):
    """
    Combines saved frames into an MP4 video file
    """
    if not frame_paths:
        print("❌ No frames available to create video")
        return None

    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()

    print(f"🎥 Video saved: {output_file}")
    return output_file

# ============================================================================
# EMOTION ANALYSIS CLASS
# ============================================================================
class EmotionAnalyzer:
    """Class for emotion analysis"""

    def __init__(self):
        self.config = Config()
        print("🧠 Initializing emotion analyzer...")

    def analyze_image(self, image_path):
        """
        Analyzes emotions from a single image
        Returns: dictionary with results
        """
        try:

            analysis = DeepFace.analyze(
                img_path=image_path,
                actions=['emotion'],
                enforce_detection=True,
                detector_backend='opencv',
                silent=True
            )

            # Process results
            if isinstance(analysis, list):
                result = analysis[0]
            else:
                result = analysis

            # Emotion translation (model output -> readable label)
            dominant_emotion = result['dominant_emotion']
            dominant_emotion_ru = self.config.EMOTION_TRANSLATION.get(
                dominant_emotion, dominant_emotion
            )

            # Prepare results
            emotion_scores = result['emotion']
            scores_ru = {}

            for emotion_en, score in emotion_scores.items():
                emotion_ru = self.config.EMOTION_TRANSLATION.get(emotion_en, emotion_en)
                scores_ru[emotion_ru] = score

            return {
                'success': True,
                'dominant_emotion': dominant_emotion_ru,
                'emotion_scores': scores_ru,
                'original_emotion': dominant_emotion,
                'face_detected': True,
                'confidence': emotion_scores[dominant_emotion]
            }

        except Exception as e:
            error_msg = str(e)

            if 'Face could not be detected' in error_msg:
                return {
                    'success': False,
                    'error': 'Face not detected',
                    'face_detected': False,
                    'recommendation': 'Make sure the face is clearly visible in the image'
                }
            else:
                return {
                    'success': False,
                    'error': error_msg,
                    'face_detected': None
                }

    def analyze_video(self, video_path, frame_interval=10):
        """
        Analyzes emotions in a video file
        """
        print(f"🎥 Analyzing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {
                'success': False,
                'error': f'Could not open video file: {video_path}'
            }

        # Video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"📊 Video info:")
        print(f"   • FPS: {fps:.1f}")
        print(f"   • Total frames: {total_frames}")
        print(f"   • Duration: {duration:.1f} sec")
        print(f"   • Analysis interval: every {frame_interval} frame(s)")

        # Analysis preparation
        emotions_data = []
        emotion_counts = {}
        processed_frames = 0
        frame_idx = 0

        print("\n🔍 Starting analysis...")

        progress_bar_len = 40

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Analyze every N-th frame
            if frame_idx % frame_interval == 0:
                temp_path = f"/temp_video_frame_{processed_frames}.jpg"
                cv2.imwrite(temp_path, frame)

                result = self.analyze_image(temp_path)

                if result['success']:
                    result['timestamp'] = frame_idx / fps
                    result['frame_number'] = frame_idx

                    emotions_data.append(result)

                    emotion = result['dominant_emotion']
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

                    processed_frames += 1

                    if processed_frames % 5 == 0:
                        progress = (frame_idx / total_frames) * 100
                        filled_len = int(progress_bar_len * frame_idx // total_frames)
                        bar = '█' * filled_len + '░' * (progress_bar_len - filled_len)
                        print(f"   [{bar}] {progress:.1f}% | Frames: {processed_frames}", end='\r')

                try:
                    os.remove(temp_path)
                except:
                    pass

            frame_idx += 1

        cap.release()

        print(f"\n✅ Analysis completed!")
        print(f"   • Processed frames: {processed_frames}")
        print(f"   • Detected emotions: {len(emotion_counts)}")

        if processed_frames == 0:
            return {
                'success': False,
                'error': 'No frames were successfully analyzed',
                'recommendation': 'Try reducing frame interval or check video quality'
            }

        return {
            'success': True,
            'emotions_data': emotions_data,
            'emotion_counts': emotion_counts,
            'video_info': {
                'fps': fps,
                'total_frames': total_frames,
                'duration': duration,
                'processed_frames': processed_frames,
                'frame_interval': frame_interval
            },
            'summary': self._generate_summary(emotion_counts, processed_frames)
        }

    def analyze_frames(self, frame_paths):
        """
        Analyzes a list of frames
        """
        print(f"🔍 Analyzing {len(frame_paths)} frames...")

        emotions_data = []
        emotion_counts = {}

        for i, frame_path in enumerate(frame_paths):
            result = self.analyze_image(frame_path)

            if result['success']:
                result['frame_number'] = i
                result['timestamp'] = i * 0.1  # assuming 10 FPS

                emotions_data.append(result)

                emotion = result['dominant_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            if (i + 1) % 5 == 0 or (i + 1) == len(frame_paths):
                progress = ((i + 1) / len(frame_paths)) * 100
                print(f"   Processed: {i+1}/{len(frame_paths)} ({progress:.1f}%)", end='\r')

        print(f"\n✅ Analysis completed!")

        if len(emotions_data) == 0:
            return {
                'success': False,
                'error': 'No frames were successfully analyzed'
            }

        return {
            'success': True,
            'emotions_data': emotions_data,
            'emotion_counts': emotion_counts,
            'summary': self._generate_summary(emotion_counts, len(emotions_data))
        }

    def _generate_summary(self, emotion_counts, total_frames):
        """
        Generates statistics and calculates heuristic engagement score
        based on emotion weights.
        """
        if total_frames == 0:
            return {}

        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        dominant_percentage = (dominant_emotion[1] / total_frames) * 100

        positive_count = (
            emotion_counts.get('Happiness', 0) +
            emotion_counts.get('Surprise', 0)
        )

        negative_count = (
            emotion_counts.get('Anger', 0) +
            emotion_counts.get('Disgust', 0) +
            emotion_counts.get('Fear', 0) +
            emotion_counts.get('Sadness', 0)
        )

        neutral_count = emotion_counts.get('Neutral', 0)

        emotion_weights = {
            'Happiness': 1.0,
            'Surprise': 0.8,
            'Neutral': 0.5,
            'Sadness': 0.2,
            'Fear': 0.2,
            'Anger': 0.1,
            'Disgust': 0.0
        }

        weighted_sum = 0
        for emotion, count in emotion_counts.items():
            weight = emotion_weights.get(emotion, 0.3)
            weighted_sum += count * weight

        engagement_score = (weighted_sum / total_frames) * 100
        engagement_score = max(0, min(100, engagement_score))

        return {
            'dominant_emotion': dominant_emotion[0],
            'dominant_percentage': dominant_percentage,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'engagement_score': engagement_score,
            'total_frames': total_frames
        }

# ============================================================================
# VISUALIZATION OF RESULTS
# ============================================================================
class ResultVisualizer:
    """Class for results visualization"""

    def __init__(self):
        self.config = Config()

    def display_summary(self, results, source_type="video"):
        """
        Displays a summary of results
        """
        if not results.get('success', False):
            print("❌ No results to display")
            if 'error' in results:
                print(f"   Error: {results['error']}")
            return

        summary = results.get('summary', {})
        emotion_counts = results.get('emotion_counts', {})

        if not summary:
            print("❌ No data to display")
            return

        print("\n" + "="*60)
        print("📊 EMOTION ANALYSIS RESULTS")
        print("="*60)

        # 1. MAIN INFORMATION
        print(f"\n🎯 DOMINANT EMOTION:")
        print(f"   {summary['dominant_emotion']} - {summary['dominant_percentage']:.1f}%")

        # 2. STATISTICS
        print(f"\n📈 EMOTION STATISTICS:")
        print("-"*40)

        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)

        for emotion, count in sorted_emotions:
            percentage = (count / summary['total_frames']) * 100
            bar_length = int(percentage / 2)  # 50 chars = 100%
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"   {emotion:12} {bar} {percentage:5.1f}% ({count} frames)")

        # 3. ENGAGEMENT
        print(f"\n🎓 STUDENT ENGAGEMENT SCORE:")
        print("-"*40)

        engagement = summary['engagement_score']

        engagement_bar_len = 30
        filled_len = int(engagement_bar_len * engagement / 100)
        engagement_bar = '█' * filled_len + '░' * (engagement_bar_len - filled_len)

        print(f"   Engagement score: {engagement:.1f}/100")
        print(f"   [{engagement_bar}]")

        if engagement >= self.config.HIGH_ENGAGEMENT:
            print(f"   ✅ HIGH ENGAGEMENT")
            print(f"      Student is actively engaged in the learning process")
        elif engagement >= self.config.MEDIUM_ENGAGEMENT:
            print(f"   ⚠️  MEDIUM ENGAGEMENT")
            print(f"      Try to maintain student interest")
        else:
            print(f"   ❌ LOW ENGAGEMENT")
            print(f"      Teaching approach adjustment required")

        # 4. GRAPHS
        self._plot_results(emotion_counts, summary)

        # 5. RECOMMENDATIONS
        self._display_recommendations(summary)

        # 6. ANALYSIS INFO
        print(f"\n📋 ANALYSIS INFO:")
        print("-"*40)
        print(f"   Source: {'Camera' if source_type == 'camera' else 'Video file'}")
        print(f"   Processed frames: {summary['total_frames']}")
        print(f"   Positive emotions: {summary['positive_count']/summary['total_frames']*100:.1f}%")
        print(f"   Negative emotions: {summary['negative_count']/summary['total_frames']*100:.1f}%")
        print(f"   Neutral emotions: {summary['neutral_count']/summary['total_frames']*100:.1f}%")

    def _plot_results(self, emotion_counts, summary):
        """
        Creates result charts
        """
        try:
            emotions = list(emotion_counts.keys())
            counts = list(emotion_counts.values())
            colors = [self.config.COLORS.get(emotion, '#808080') for emotion in emotions]

            fig = plt.figure(figsize=(15, 5))

            # 1. PIE CHART
            ax1 = plt.subplot(1, 3, 1)
            ax1.pie(counts, labels=emotions, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('📊 Emotion Distribution', fontsize=14, fontweight='bold')
            ax1.axis('equal')

            # 2. BAR CHART
            ax2 = plt.subplot(1, 3, 2)
            bars = ax2.bar(emotions, counts, color=colors)
            ax2.set_title('📈 Emotion Frequency', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Frame count', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)

            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom')

            # 3. ENGAGEMENT CHART
            ax3 = plt.subplot(1, 3, 3)

            categories = ['Positive', 'Negative', 'Neutral']
            values = [
                summary['positive_count'],
                summary['negative_count'],
                summary['neutral_count']
            ]
            cat_colors = ['#4CAF50', '#F44336', '#9E9E9E']

            bars3 = ax3.bar(categories, values, color=cat_colors)
            ax3.set_title('🎓 Student Engagement', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Frame count', fontsize=12)

            total = sum(values)
            for bar, value in zip(bars3, values):
                height = bar.get_height()
                percentage = (value / total) * 100 if total > 0 else 0
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{percentage:.1f}%', ha='center', va='bottom')

            plt.tight_layout()

            timestamp = time.strftime('%Y%m%d_%H%M%S')
            plot_file = f"emotion_plot_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved: {plot_file}")

            plt.show()

        except Exception as e:
            print(f"⚠️ Error while creating plots: {e}")

    def _display_recommendations(self, summary):
        """
        Displays teacher recommendations
        """
        print(f"\n💡 TEACHER RECOMMENDATIONS:")
        print("-"*45)

        dominant = summary['dominant_emotion']
        engagement = summary['engagement_score']

        if dominant == 'Happiness' and engagement >= 70:
            print("""
   ✅ Student shows high engagement and positive attitude.
      Recommendations:
      1. Provide advanced learning materials
      2. Encourage group discussions
      3. Assign more complex tasks
      4. Reinforce positive behavior
            """)

        elif dominant == 'Neutral' or engagement < 40:
            print("""
   ⚠️  Student may be bored or tired.
      Recommendations:
      1. Change teaching format (video, interactive content)
      2. Add practical examples
      3. Take short breaks
      4. Ask engagement questions
      5. Use more visual materials
            """)

        elif dominant in ['Anger', 'Disgust', 'Fear']:
            print("""
   ❌ Student may be struggling or frustrated.
      Recommendations:
      1. Simplify explanations
      2. Offer individual support
      3. Break material into smaller parts
      4. Provide positive feedback
      5. Check understanding frequently
            """)

        elif dominant == 'Sadness':
            print("""
   😐 Student may be demotivated.
      Recommendations:
      1. Show empathy and support
      2. Explain real-world relevance
      3. Offer assistance
      4. Use positive tone
      5. Allow flexible pacing
            """)

        else:
            print("""
   📝 General recommendations:
      1. Regular comprehension checks
      2. Use varied teaching methods
      3. Adapt pace to group
      4. Collect feedback
      5. Maintain supportive environment
            """)

    def save_report(self, results, source_type):
        """
        Saves report to file
        """
        try:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            report_file = f"emotion_analysis_report_{timestamp}.txt"

            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("STUDENT EMOTION ANALYSIS REPORT\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"Date: {time.strftime('%d.%m.%Y %H:%M:%S')}\n")
                f.write(f"Data source: {source_type}\n\n")

                if results.get('success', False):
                    summary = results.get('summary', {})
                    emotion_counts = results.get('emotion_counts', {})

                    f.write("EMOTION STATISTICS:\n")
                    f.write("-" * 40 + "\n")

                    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / summary['total_frames']) * 100
                        f.write(f"{emotion:15} {count:4d} frames ({percentage:5.1f}%)\n")

                    f.write(f"\nDOMINANT EMOTION: {summary.get('dominant_emotion', 'N/A')}\n")
                    f.write(f"ENGAGEMENT SCORE: {summary.get('engagement_score', 0):.1f}/100\n")

                    if 'video_info' in results:
                        info = results['video_info']
                        f.write(f"\nVIDEO INFO:\n")
                        f.write(f"  • Duration: {info.get('duration', 0):.1f} sec\n")
                        f.write(f"  • Total frames: {info.get('total_frames', 0)}\n")
                        f.write(f"  • Processed: {info.get('processed_frames', 0)}\n")
                else:
                    f.write("ANALYSIS FAILED\n")
                    f.write(f"Error: {results.get('error', 'Unknown error')}\n")

                f.write("\n" + "=" * 60 + "\n")
                f.write("Project #97 - Student Emotion Recognition System\n")
                f.write("=" * 60 + "\n")

            print(f"📄 Report saved: {report_file}")
            return report_file

        except Exception as e:
            print(f"⚠️ Failed to save report: {e}")
            return None

# ============================================================================
# SAVE RESULTS TO CSV
# ============================================================================
def save_result_to_csv(results, mode, source_name, expected_emotion="", notes=""):
    """
    Saves a single experiment result into a CSV file.
    mode: 'online' or 'offline'
    source_name: file name or scenario name
    expected_emotion: expected emotion (if known)
    notes: additional comments
    """
    csv_file = "all_emotion_results.csv"
    file_exists = os.path.exists(csv_file)

    summary = results.get('summary', {})
    video_info = results.get('video_info', {})

    row = {
        "date_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "source_name": source_name,
        "expected_emotion": expected_emotion,
        "predicted_emotion": summary.get("dominant_emotion", ""),
        "dominant_percentage": round(summary.get("dominant_percentage", 0), 2),
        "engagement_score": round(summary.get("engagement_score", 0), 2),
        "processed_frames": summary.get("total_frames", 0),
        "video_duration_sec": round(video_info.get("duration", 0), 2) if video_info else "",
        "fps": round(video_info.get("fps", 0), 2) if video_info else "",
        "positive_count": summary.get("positive_count", 0),
        "negative_count": summary.get("negative_count", 0),
        "neutral_count": summary.get("neutral_count", 0),
        "notes": notes
    }

    with open(csv_file, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"💾 Result added to {csv_file}")

# ============================================================================
# DEMO MODE WITH REAL IMAGE
# ============================================================================
def run_demo_mode():
    """
    Runs demo mode using a real image
    """
    print("\n" + "="*60)
    print("🖼️  DEMO MODE: System Testing")
    print("="*60)

    print("\n📝 Loading test image...")

    try:
        # Using a real test image
        import requests

        # Test image URLs with faces (from open sources)
        test_urls = [
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
            "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg"
        ]

        image_data = None
        for url in test_urls:
            try:
                print(f"   Trying to download: {url}")
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    image_data = response.content
                    print("   ✅ Image successfully downloaded")
                    break
            except:
                continue

        if not image_data:
            print("❌ Failed to load test image")
            print("   Creating a simple image...")

            # Create a simple image
            img = np.zeros((300, 300, 3), dtype=np.uint8)
            img[:] = [255, 255, 255]  # White background

            # Draw a simple face
            cv2.circle(img, (150, 120), 50, (200, 200, 200), -1)  # Head
            cv2.circle(img, (130, 100), 8, (0, 0, 0), -1)  # Left eye
            cv2.circle(img, (170, 100), 8, (0, 0, 0), -1)  # Right eye
            cv2.ellipse(img, (150, 140), (30, 15), 0, 0, 180, (0, 0, 0), 3)  # Smile

            test_path = "/demo_test_face.jpg"
            cv2.imwrite(test_path, img)

        else:
            # Save downloaded image
            test_path = "/demo_real_face.jpg"
            with open(test_path, 'wb') as f:
                f.write(image_data)

        # Check image
        if os.path.exists(test_path):
            img_test = cv2.imread(test_path)
            if img_test is not None:
                print(f"   ✅ Image saved: {test_path}")
                print(f"   📏 Size: {img_test.shape[1]}x{img_test.shape[0]}")

                # Show image
                plt.figure(figsize=(6, 6))
                plt.imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))
                plt.title("Test image for analysis")
                plt.axis('off')
                plt.show()
            else:
                print("❌ Failed to load image")
                return None
        else:
            print("❌ Image file was not created")
            return None

        # Emotion analysis
        print("\n🔍 Running emotion analysis on image...")

        analyzer = EmotionAnalyzer()
        result = analyzer.analyze_image(test_path)

        if result['success']:
            print("✅ Analysis completed successfully!")
            print(f"\n📊 RESULTS:")
            print("-" * 40)
            print(f"   Dominant emotion: {result['dominant_emotion']}")
            print(f"   Confidence: {result['confidence']:.1f}%")

            print(f"\n   Emotion distribution:")
            for emotion, score in result['emotion_scores'].items():
                print(f"   • {emotion:12}: {score:.1f}%")

            # Create results structure for visualization
            demo_results = {
                'success': True,
                'emotion_counts': {result['dominant_emotion']: 1},
                'summary': {
                    'dominant_emotion': result['dominant_emotion'],
                    'dominant_percentage': result['confidence'],
                    'engagement_score': 70 if result['dominant_emotion'] == 'Happiness' else
                                       50 if result['dominant_emotion'] == 'Neutral' else 30,
                    'total_frames': 1,
                    'positive_count': 1 if result['dominant_emotion'] in ['Happiness', 'Surprise'] else 0,
                    'negative_count': 1 if result['dominant_emotion'] in ['Anger', 'Disgust', 'Fear', 'Sadness'] else 0,
                    'neutral_count': 1 if result['dominant_emotion'] == 'Neutral' else 0
                }
            }

            # Visualization
            visualizer = ResultVisualizer()
            visualizer.display_summary(demo_results, "demo")
            visualizer.save_report(demo_results, "Demo mode")

            return demo_results

        else:
            print(f"❌ Analysis error: {result.get('error', 'Unknown error')}")

            if not result.get('face_detected', True):
                print("💡 Tip: Make sure the face is clearly visible in the image")
                print("        Try using your own photo")

            return None

    except Exception as e:
        print(f"❌ Demo mode error: {e}")
        return None

# ============================================================================
# MAIN PROGRAM
# ============================================================================
def main():
    """
    Main function of the program
    """
    print("\n" + "="*70)
    print("🎓 STUDENT EMOTION RECOGNITION SYSTEM")
    print("="*70)

    print("\n" + "📋"*20)
    print("   PROJECT GOAL:")
    print("   Development of a system for recognizing emotional state")
    print("   and student engagement level with adaptive learning content")
    print("📋"*20)

    while True:
        print("\n" + "─"*50)
        print("🏠 MAIN MENU")
        print("─"*50)
        print("   1. 📁 Analyze uploaded video file")
        print("   2. 📷 Record and analyze from webcam")
        print("   3. 🖼️  Demo mode (testing)")
        print("   4. 📊 View sample report")
        print("   5. ❌ Exit program")
        print("─"*50)

        choice = input("\n👉 Choose an option (1-5): ").strip()

        if choice == "1":
            print("\n" + "📁"*20)
            print("   VIDEO FILE ANALYSIS MODE")
            print("📁"*20)

            video_path = upload_file('video')

            if video_path:
                try:
                    # Ask for analysis parameters
                    interval_input = input("\n📊 Enter frame analysis interval [10]: ").strip()
                    frame_interval = int(interval_input) if interval_input.isdigit() else 10

                    if frame_interval < 1:
                        print("⚠️  Interval must be at least 1. Set to 1")
                        frame_interval = 1

                    # Video analysis
                    analyzer = EmotionAnalyzer()
                    results = analyzer.analyze_video(video_path, frame_interval)

                    # Result visualization
                    if results.get('success', False):
                        visualizer = ResultVisualizer()
                        visualizer.display_summary(results, "video")
                        visualizer.save_report(results, f"Video file: {video_path}")

                        expected = input("Expected emotion (if known): ").strip()
                        notes = input("Notes: ").strip()

                        save_result_to_csv(
                            results=results,
                            mode="offline",
                            source_name=video_path,
                            expected_emotion=expected,
                            notes=notes
                        )
                    else:
                        print(f"❌ {results.get('error', 'Unknown error')}")
                        if 'recommendation' in results:
                            print(f"💡 {results['recommendation']}")

                except Exception as e:
                    print(f"❌ Error during video analysis: {e}")

        elif choice == "2":
            print("\n" + "📷"*20)
            print("   CAMERA RECORDING MODE")
            print("📷"*20)

            try:
                # Ask for parameters
                duration_input = input("\n⏱️  Enter recording duration in seconds [5]: ").strip()
                duration = int(duration_input) if duration_input.isdigit() else 5

                if duration > 60:
                    print("⚠️  Duration limited to 60 seconds")
                    duration = 60
                elif duration < 2:
                    print("⚠️  Minimum duration is 2 seconds")
                    duration = 2

                # Record from camera
                frame_paths = capture_from_camera(duration)

                video_file = save_frames_to_video(frame_paths, "camera_recording.mp4", fps=10)

                # allow download
                from google.colab import files
                files.download(video_file)

                if frame_paths:
                    print("\n🔍 Starting analysis of recorded frames...")

                    analyzer = EmotionAnalyzer()
                    results = analyzer.analyze_frames(frame_paths)

                    # Result visualization
                    if results.get('success', False):
                        visualizer = ResultVisualizer()
                        visualizer.display_summary(results, "camera")
                        visualizer.save_report(results, "Webcam recording")

                        scenario = input("Recording scenario (e.g. smile / neutral / surprise): ").strip()
                        notes = input("Notes: ").strip()

                        save_result_to_csv(
                            results=results,
                            mode="online",
                            source_name=scenario if scenario else "camera_recording",
                            expected_emotion=scenario,
                            notes=notes
                        )
                    else:
                        print(f"❌ {results.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"❌ Camera error: {e}")

        elif choice == "3":
            print("\n" + "🖼️"*20)
            print("   DEMO MODE")
            print("🖼️"*20)

            print("\n📝 This mode is for system testing.")
            print("   A test image is used to demonstrate functionality.")

            results = run_demo_mode()

            if results:
                print("\n✅ Demo completed successfully!")
                print("💡 Tip: Use modes 1 or 2 for real analysis")

        elif choice == "4":
            print("\n" + "📊"*20)
            print("   SAMPLE REPORT")
            print("📊"*20)

            print("""
============================================================
             SAMPLE EMOTION ANALYSIS REPORT
============================================================

Analysis date: 08.01.2026 10:30:00
Data source: Video file: lecture_sample.mp4

EMOTION STATISTICS:
----------------------------------------
Happiness        45 frames (45.0%)
Neutral          30 frames (30.0%)
Surprise         15 frames (15.0%)
Sadness          5 frames (5.0%)
Anger            3 frames (3.0%)
Fear             2 frames (2.0%)

DOMINANT EMOTION: Happiness
ENGAGEMENT SCORE: 78.5/100

VIDEO INFORMATION:
  • Duration: 120.5 sec
  • Total frames: 3615
  • Processed: 100
            """)

        elif choice == "5":
            print("\n" + "="*50)
            print("👋 Shutting down program")
            print("   Thank you for using our system!")
            print("="*50)
            break

        else:
            print("❌ Invalid choice. Please select 1–5.")

        # Pause before next action
        if choice in ['1', '2', '3']:
            input("\n↵ Press Enter to continue...")
            clear_output(wait=True)

# ============================================================================
# RUN PROGRAM
# ============================================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        print("💡 Please restart the program")