import os
import shutil
import subprocess
import requests
import yt_dlp
import librosa
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from PyQt6.QtGui import QPixmap, QImage
from duckduckgo_search import DDGS

# --- Worker Classes for background tasks ---

class AIModelLoadWorker(QObject):
    """Loads the AI model in the background."""
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, ai_engine):
        super().__init__()
        self.ai_engine = ai_engine

    def run(self):
        success = self.ai_engine.load_model()
        if success:
            self.finished.emit(True, "AI Model loaded successfully.")
        else:
            self.finished.emit(False, "Failed to load AI Model. Check logs and model path.")

class ImageSearchWorker(QObject):
    """Searches for images using DuckDuckGo."""
    image_ready = pyqtSignal(QPixmap, str)
    finished = pyqtSignal()

    def __init__(self, query):
        super().__init__()
        self.query = query

    def run(self):
        try:
            for result in DDGS().images(self.query, max_results=20):
                try:
                    response = requests.get(result.get('image'), timeout=5)
                    response.raise_for_status()
                    pixmap = QPixmap()
                    pixmap.loadFromData(response.content)
                    if not pixmap.isNull():
                        self.image_ready.emit(pixmap, result.get('image'))
                except requests.RequestException:
                    continue  # Skip images that fail to download
        finally:
            self.finished.emit()

class VideoSearchWorker(QObject):
    """Searches for videos on YouTube."""
    video_ready = pyqtSignal(str, str)
    finished = pyqtSignal(str)

    def __init__(self, query):
        super().__init__()
        self.query = query

    def run(self):
        try:
            results = DDGS().videos(f"{self.query} official audio", max_results=10)
            if not results:
                self.finished.emit("No videos found.")
                return
            for r in results:
                self.video_ready.emit(r['title'], r['content'])
        finally:
            self.finished.emit(None)

class AudioProcessingWorker(QObject):
    """Downloads audio and subtitles for a given YouTube URL."""
    finished = pyqtSignal(str, str, str)  # audio_path, lyrics_path, error_string

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        try:
            output_template = 'downloaded_media'
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': f'{output_template}.%(ext)s',
                'writesubtitles': True,
                'subtitleslangs': ['en', 'en-US'],
                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
                'overwrites': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.url])

            audio_path = f"{output_template}.mp3"
            lyrics_path = f"{output_template}.en.vtt"
            if not os.path.exists(lyrics_path):
                lyrics_path = None

            self.finished.emit(audio_path, lyrics_path, None)
        except Exception as e:
            self.finished.emit(None, None, f"Download failed: {e}")

class AIGenerationWorker(QObject):
    """Generates images using the AI Engine."""
    image_ready = pyqtSignal(QPixmap, int)
    finished = pyqtSignal()

    def __init__(self, ai_engine, prompt, num_images=4):
        super().__init__()
        self.ai_engine = ai_engine
        self.prompt = prompt
        self.num_images = num_images

    def run(self):
        images = self.ai_engine.generate_images(self.prompt, self.num_images)
        for i, img in enumerate(images):
            # Convert PIL image to QPixmap
            q_image = QImage(img.tobytes("raw", "RGB"), img.width, img.height, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_ready.emit(pixmap, i)
        self.finished.emit()

class RenderWorker(QObject):
    """Handles the entire video frame generation process."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(str, str)  # frames_path, error_string

    def __init__(self, ai_engine, char_pixmap, audio_path, trim, pose_markers, bg_markers, lyrics_data):
        super().__init__()
        self.ai_engine = ai_engine
        self.char_pixmap = char_pixmap
        self.audio_path = audio_path
        self.trim_start, self.trim_end = trim
        self.pose_markers = sorted(pose_markers.items(), key=lambda item: item[0])
        self.bg_markers = sorted(bg_markers.items(), key=lambda item: item[0])
        self.lyrics_data = lyrics_data

    def _create_mouth_shapes(self):
        """Generates simple procedural mouth shapes."""
        mouth_closed = Image.new('RGBA', (100, 50), (0, 0, 0, 0))
        draw = ImageDraw.Draw(mouth_closed)
        draw.line([(20, 25), (80, 25)], fill="black", width=5)

        mouth_open_1 = Image.new('RGBA', (100, 50), (0, 0, 0, 0))
        draw = ImageDraw.Draw(mouth_open_1)
        draw.ellipse([(20, 10), (80, 40)], fill=(0,0,0), outline="black")

        mouth_open_2 = Image.new('RGBA', (100, 50), (0, 0, 0, 0))
        draw = ImageDraw.Draw(mouth_open_2)
        draw.ellipse([(15, 0), (85, 50)], fill=(0,0,0), outline="black")

        return {"closed": mouth_closed, "open": [mouth_open_1, mouth_open_2]}

    def run(self):
        try:
            FRAME_RATE = 30
            DURATION = self.trim_end - self.trim_start
            TOTAL_FRAMES = int(DURATION * FRAME_RATE)

            if os.path.exists('temp_frames'): shutil.rmtree('temp_frames')
            os.makedirs('temp_frames')

            y, sr = librosa.load(self.audio_path, sr=None, offset=self.trim_start, duration=DURATION)
            _, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr)

            # Vocal activity detection
            y_harmonic, _ = librosa.effects.hpss(y)
            onset_env = librosa.onset.onset_detect(y=y_harmonic, sr=sr, units='time')
            is_vocal = np.zeros(TOTAL_FRAMES)
            for t_start in onset_env:
                frame_start = int(t_start * FRAME_RATE)
                # Assume vocal activity for a short duration after onset
                is_vocal[frame_start : frame_start + int(0.5 * FRAME_RATE)] = 1

            base_char_img = self._pixmap_to_pil(self.char_pixmap).resize((400, 400), Image.LANCZOS)
            mouth_shapes = self._create_mouth_shapes()
            active_bg_img = Image.new('RGBA', (512, 512), (20, 20, 20, 255))
            active_pose_img = None
            pose_end_time = -1

            try:
                font = ImageFont.truetype("arialbd.ttf", 32)
            except IOError:
                font = ImageFont.load_default()

            for i in range(TOTAL_FRAMES):
                current_time_rel = i / FRAME_RATE
                current_time_abs = self.trim_start + current_time_rel

                if self.bg_markers and current_time_abs >= self.bg_markers[0][0]:
                    prompt = self.bg_markers.pop(0)[1]['prompt']
                    active_bg_img = self.ai_engine.generate_images(prompt, 1)[0]

                if self.pose_markers and current_time_abs >= self.pose_markers[0][0]:
                    prompt = self.pose_markers.pop(0)[1]['prompt']
                    active_pose_img = self.ai_engine.generate_images(prompt, 1)[0].resize((400, 400), Image.LANCZOS)
                    pose_end_time = current_time_rel + 2.0

                if current_time_rel > pose_end_time: active_pose_img = None

                frame = active_bg_img.copy().convert("RGBA")
                char_to_draw = (active_pose_img or base_char_img).copy()

                # Lip-Sync Logic
                if is_vocal[i] and not active_pose_img:
                    mouth_img = mouth_shapes["open"][i % len(mouth_shapes["open"])]
                else:
                    mouth_img = mouth_shapes["closed"]
                # Position mouth in the center of the lower half of the character image
                mouth_x = (char_to_draw.width - mouth_img.width) // 2
                mouth_y = (char_to_draw.height // 2) + 30
                char_to_draw.paste(mouth_img, (mouth_x, mouth_y), mouth_img)

                beat_influence = max([0] + [1 - (abs(current_time_rel - bt) / 0.25) for bt in beat_times if abs(current_time_rel - bt) < 0.25])
                bounce = int(beat_influence * 20)
                sway = 0 if active_pose_img else int(np.sin(current_time_rel * np.pi * 2) * 10)

                char_x = (512 - char_to_draw.width) // 2 + sway
                char_y = (512 - char_to_draw.height) // 2 - bounce
                frame.paste(char_to_draw, (char_x, char_y), char_to_draw)

                draw = ImageDraw.Draw(frame)
                self._draw_text_with_outline(draw, (10, 460), "@Keyaruganimates", font, (255,255,255,128), (0,0,0,128))

                current_lyric = ""
                for lyric in self.lyrics_data:
                    if lyric['start'] <= current_time_abs < lyric['end']:
                        current_lyric = lyric['text']; break
                if current_lyric:
                    self._draw_text_with_outline(draw, (frame.width / 2, 400), current_lyric, font, (255,255,255), (0,0,0))

                frame.save(f'temp_frames/frame_{i:05d}.png')
                self.progress.emit(i + 1, TOTAL_FRAMES)

            self.finished.emit("temp_frames", None)
        except Exception as e:
            self.finished.emit(None, f"Render failed: {e}")

    def _pixmap_to_pil(self, pixmap):
        q_image = pixmap.toImage()
        buffer = q_image.bits()
        buffer.setsize(q_image.sizeInBytes())
        return Image.frombuffer("RGBA", (q_image.width(), q_image.height()), buffer, "raw", "RGBA", 0, 1)

    def _draw_text_with_outline(self, draw, position, text, font, fill, outline_fill):
        x, y = position
        # Draw outline by drawing text in a slightly offset position
        for dx, dy in [(-2,-2), (-2,2), (2,-2), (2,2)]:
            draw.text((x+dx, y+dy), text, font=font, fill=outline_fill, anchor="ms")
        # Draw the main text
        draw.text((x,y), text, font=font, fill=fill, anchor="ms")