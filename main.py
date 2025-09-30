import sys
import os
import shutil
import subprocess
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTabWidget, QLabel, QHBoxLayout,
    QLineEdit, QPushButton, QScrollArea, QGridLayout, QListWidget, QListWidgetItem,
    QTextEdit, QMessageBox, QFileDialog, QInputDialog, QProgressBar, QStatusBar
)
from PyQt6.QtCore import QThread, Qt
from ai_engine import AIEngine
from workers import (
    AIModelLoadWorker, ImageSearchWorker, VideoSearchWorker, AudioProcessingWorker,
    AIGenerationWorker, RenderWorker
)
from utils import parse_vtt
import numpy as np
import librosa

class AnimationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AI Animation Studio')
        self.setGeometry(100, 100, 1280, 720)

        self.ai_engine = AIEngine()
        self.threads = []
        self.state = {
            "selected_ref_urls": set(), "generated_images": {},
            "selected_generated_image": None, "audio_path": None,
            "lyrics_path": None, "audio_trim_region": (0, 15),
            "pose_markers": {}, "bg_markers": {}
        }

        self.setup_ui()
        self.load_ai_model_in_background()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.character_tab = QWidget()
        self.scene_tab = QWidget()
        self.animate_tab = QWidget()

        self.tabs.addTab(self.character_tab, "1. Character")
        self.tabs.addTab(self.scene_tab, "2. Scene")
        self.tabs.addTab(self.animate_tab, "3. Animate")

        self.setup_character_tab()
        self.setup_scene_tab()
        self.setup_animate_tab()

        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)

    def load_ai_model_in_background(self):
        self.status_bar.showMessage("Loading AI Model... Please wait.")
        self.generate_button.setEnabled(False)
        worker = AIModelLoadWorker(self.ai_engine)
        self.run_in_thread(worker, finished=self.on_model_load_finished)

    def on_model_load_finished(self, success, message):
        self.status_bar.showMessage(message, 5000) # Show for 5 seconds
        self.generate_button.setEnabled(success)
        if not success:
            QMessageBox.critical(self, "AI Error", message)

    def setup_character_tab(self):
        # --- Layouts ---
        self.char_layout = QHBoxLayout(self.character_tab)
        ref_layout = QVBoxLayout()
        gen_layout = QVBoxLayout()

        # --- Reference Search Section (Left) ---
        search_box = QHBoxLayout()
        self.char_search_input = QLineEdit(placeholderText="Search for character references...")
        self.char_search_button = QPushButton("Search")
        search_box.addWidget(self.char_search_input)
        search_box.addWidget(self.char_search_button)
        ref_layout.addLayout(search_box)

        self.ref_scroll_area = QScrollArea(widgetResizable=True)
        self.ref_results_container = QWidget()
        self.ref_results_layout = QGridLayout(self.ref_results_container)
        self.ref_scroll_area.setWidget(self.ref_results_container)
        ref_layout.addWidget(self.ref_scroll_area)

        # --- AI Generation Section (Right) ---
        self.prompt_input = QTextEdit(placeholderText="Enter prompt: e.g., 'chibi character, cute, anime style'")

        gen_button_layout = QHBoxLayout()
        self.generate_button = QPushButton("Generate Character")
        self.generate_button.setEnabled(False)
        self.regenerate_char_button = QPushButton("Regenerate")
        self.regenerate_char_button.setEnabled(False)
        gen_button_layout.addWidget(self.generate_button)
        gen_button_layout.addWidget(self.regenerate_char_button)

        self.gen_scroll_area = QScrollArea(widgetResizable=True)
        self.gen_results_container = QWidget()
        self.gen_results_layout = QGridLayout(self.gen_results_container)
        self.gen_scroll_area.setWidget(self.gen_results_container)
        self.download_button = QPushButton("Download Selected Character")
        self.download_button.setEnabled(False)

        gen_layout.addWidget(QLabel("AI Generation Prompt:"))
        gen_layout.addWidget(self.prompt_input, 1)
        gen_layout.addLayout(gen_button_layout)
        gen_layout.addWidget(self.gen_scroll_area, 3)
        gen_layout.addWidget(self.download_button)

        self.char_layout.addLayout(ref_layout, 1)
        self.char_layout.addLayout(gen_layout, 1)

        # --- Connections ---
        self.char_search_button.clicked.connect(self.start_image_search)
        self.generate_button.clicked.connect(self.start_generation)
        self.regenerate_char_button.clicked.connect(self.start_generation)
        self.download_button.clicked.connect(self.download_selected_character)

    def setup_scene_tab(self):
        layout = QVBoxLayout(self.scene_tab)
        search_layout = QHBoxLayout()
        self.song_search_input = QLineEdit(placeholderText="Enter song name...")
        self.song_search_button = QPushButton("Search Songs")
        search_layout.addWidget(self.song_search_input)
        search_layout.addWidget(self.song_search_button)
        layout.addLayout(search_layout)
        self.song_results_list = QListWidget()
        layout.addWidget(self.song_results_list)

        self.song_search_button.clicked.connect(self.start_video_search)
        self.song_results_list.itemClicked.connect(self.on_song_selected)

    def setup_animate_tab(self):
        layout = QVBoxLayout(self.animate_tab)
        self.progress_bar = QProgressBar(visible=False)
        layout.addWidget(self.progress_bar)

        controls_layout = QHBoxLayout()
        self.add_pose_button = QPushButton("Add Pose Marker")
        self.add_bg_button = QPushButton("Add Background Marker")
        self.render_button = QPushButton("Render Video")
        controls_layout.addWidget(self.add_pose_button)
        controls_layout.addWidget(self.add_bg_button)
        controls_layout.addWidget(self.render_button)
        layout.addLayout(controls_layout)

        self.timeline_widget = pg.PlotWidget()
        self.timeline_widget.setBackground('w')
        self.timeline_widget.getPlotItem().setMouseEnabled(x=True, y=False)
        layout.addWidget(self.timeline_widget)

        self.add_pose_button.clicked.connect(lambda: self.add_marker('pose'))
        self.add_bg_button.clicked.connect(lambda: self.add_marker('bg'))
        self.render_button.clicked.connect(self.start_rendering)

    # --- Thread Management ---
    def run_in_thread(self, worker, **signals):
        thread = QThread()
        worker.moveToThread(thread)
        for signal_name, slot in signals.items():
            if hasattr(worker, signal_name):
                getattr(worker, signal_name).connect(slot)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.started.connect(worker.run)
        thread.start()
        self.threads.append(thread)
        return thread

    def stop_all_threads(self):
        for thread in self.threads:
            if thread.isRunning():
                thread.requestInterruption()
                thread.quit()
                thread.wait()
        self.threads.clear()

    # --- Character Tab Logic ---
    def start_image_search(self):
        if not (query := self.char_search_input.text()): return
        self.stop_all_threads()
        for i in reversed(range(self.ref_results_layout.count())): self.ref_results_layout.itemAt(i).widget().deleteLater()
        self.state["selected_ref_urls"].clear()
        self.char_search_button.setText("Searching..."); self.char_search_button.setEnabled(False)
        worker = ImageSearchWorker(query)
        self.run_in_thread(worker, image_ready=self.add_ref_image_to_gallery, finished=lambda: (self.char_search_button.setText("Search"), self.char_search_button.setEnabled(True)))

    def add_ref_image_to_gallery(self, pixmap, image_url):
        label = QLabel(); label.setPixmap(pixmap.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        label.setFixedSize(130, 130); label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("border: 3px solid #555;"); label.mousePressEvent = lambda e, url=image_url, l=label: self.toggle_ref_selection(l, url)
        row, col = self.ref_results_layout.count() // 4, self.ref_results_layout.count() % 4
        self.ref_results_layout.addWidget(label, row, col)

    def toggle_ref_selection(self, label, url):
        if url in self.state["selected_ref_urls"]:
            self.state["selected_ref_urls"].remove(url)
            label.setStyleSheet("border: 3px solid #555;")
        else:
            self.state["selected_ref_urls"].add(url)
            label.setStyleSheet("border: 3px solid #4CAF50;")

    def start_generation(self):
        if not self.ai_engine.is_loaded:
            QMessageBox.warning(self, "AI Not Ready", "AI model is still loading or failed to load."); return
        if not (prompt := self.prompt_input.toPlainText()):
            QMessageBox.warning(self, "Prompt Missing", "Please enter a prompt."); return
        self.stop_all_threads()
        for i in reversed(range(self.gen_results_layout.count())): self.gen_results_layout.itemAt(i).widget().deleteLater()
        self.state["generated_images"].clear(); self.state["selected_generated_image"] = None; self.download_button.setEnabled(False)

        self.generate_button.setText("Generating..."); self.generate_button.setEnabled(False)
        self.regenerate_char_button.setEnabled(False)
        self.status_bar.showMessage(f"Generating images for prompt: {prompt[:50]}...")

        worker = AIGenerationWorker(self.ai_engine, prompt)
        self.run_in_thread(worker,
            image_ready=self.add_gen_image_to_gallery,
            finished=lambda: (
                self.generate_button.setText("Generate Character"),
                self.generate_button.setEnabled(True),
                self.regenerate_char_button.setEnabled(True),
                self.status_bar.showMessage("Image generation complete.", 5000)
            )
        )

    def add_gen_image_to_gallery(self, pixmap, image_id):
        self.state["generated_images"][image_id] = pixmap
        label = QLabel(); label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        label.setFixedSize(210, 210); label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("border: 3px solid #555;"); label.mousePressEvent = lambda e, p=pixmap, l=label: self.select_generated_image(l, p)
        row, col = self.gen_results_layout.count() // 2, self.gen_results_layout.count() % 2
        self.gen_results_layout.addWidget(label, row, col)

    def select_generated_image(self, label, pixmap):
        for i in range(self.gen_results_layout.count()): self.gen_results_layout.itemAt(i).widget().setStyleSheet("border: 3px solid #555;")
        label.setStyleSheet("border: 3px solid #4CAF50;")
        self.state["selected_generated_image"] = pixmap
        self.download_button.setEnabled(True)

    def download_selected_character(self):
        if not self.state["selected_generated_image"]: return
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "generated_character.png", "PNG Images (*.png)")
        if filePath: self.state["selected_generated_image"].save(filePath)

    # --- Scene Tab Logic ---
    def start_video_search(self):
        if not (query := self.song_search_input.text()): return
        self.stop_all_threads(); self.song_results_list.clear()
        self.song_search_button.setText("Searching..."); self.song_search_button.setEnabled(False)
        worker = VideoSearchWorker(query)
        self.run_in_thread(worker, video_ready=self.add_video_to_results, finished=lambda err: (self.song_search_button.setText("Search Songs"), self.song_search_button.setEnabled(True), self.song_results_list.addItem(err) if err else None))

    def add_video_to_results(self, title, url):
        item = QListWidgetItem(title); item.setData(Qt.ItemDataRole.UserRole, url); self.song_results_list.addItem(item)

    def on_song_selected(self, item):
        if not (url := item.data(Qt.ItemDataRole.UserRole)): return
        self.song_results_list.setEnabled(False)
        item.setText(f"[PROCESSING] {item.text()}")
        self.status_bar.showMessage("Downloading audio and lyrics...")
        self.stop_all_threads()
        worker = AudioProcessingWorker(url)
        self.run_in_thread(worker, finished=lambda a, l, e: self.on_audio_processing_finished(a, l, e, item))

    def on_audio_processing_finished(self, audio_path, lyrics_path, error_msg, item):
        self.song_results_list.setEnabled(True)
        item.setText(item.text().replace("[PROCESSING] ", ""))
        if error_msg:
            self.status_bar.showMessage(f"Error: {error_msg}", 5000)
            QMessageBox.critical(self, "Download Failed", error_msg)
            item.setText(f"[FAILED] {item.text()}")
            return

        self.status_bar.showMessage("Processing audio waveform...", 3000)
        self.state["audio_path"], self.state["lyrics_path"] = audio_path, lyrics_path
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            time_axis = np.linspace(0, duration, num=len(y))
            self.timeline_widget.clear()
            self.timeline_widget.plot(time_axis, y, pen=pg.mkPen('b'))
            self.timeline_widget.setYRange(np.min(y), np.max(y))
            self.timeline_widget.setXRange(0, duration)
            self.trim_region_item = pg.LinearRegionItem(values=[0, min(15, duration)], bounds=[0, duration])
            self.trim_region_item.sigRegionChanged.connect(self.on_trim_region_changed)
            self.timeline_widget.addItem(self.trim_region_item)
            self.on_trim_region_changed(self.trim_region_item)
            self.tabs.setCurrentWidget(self.animate_tab)
            self.status_bar.showMessage("Audio processed successfully. Ready to animate.", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Audio Load Failed", f"Could not load the audio file: {e}")
            self.status_bar.showMessage("Failed to load audio.", 5000)

    def on_trim_region_changed(self, region_item): self.state["audio_trim_region"] = region_item.getRegion()

    # --- Animate Tab Logic ---
    def add_marker(self, marker_type):
        if not self.state["audio_path"]: QMessageBox.warning(self, "Error", "Please select a song first."); return
        current_pos = self.state["audio_trim_region"][0]
        prompt, ok = QInputDialog.getText(self, f"New {marker_type.capitalize()} Marker", f"Enter prompt for {marker_type} at {current_pos:.2f}s:")
        if ok and prompt:
            color = 'r' if marker_type == 'pose' else 'g'
            marker = pg.InfiniteLine(pos=current_pos, angle=90, movable=True, pen=pg.mkPen(color, width=3))
            self.timeline_widget.addItem(marker)
            marker_dict = self.state["pose_markers"] if marker_type == 'pose' else self.state["bg_markers"]
            marker_dict[marker] = {'prompt': prompt}

    def start_rendering(self):
        if not self.state["selected_generated_image"] or not self.state["audio_path"]:
            QMessageBox.warning(self, "Missing Assets", "Please select a character and a song."); return
        self.render_button.setEnabled(False); self.render_button.setText("Rendering...")
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0)

        pose_markers_data = {m.value(): v for m, v in self.state["pose_markers"].items()}
        bg_markers_data = {m.value(): v for m, v in self.state["bg_markers"].items()}
        lyrics_data = parse_vtt(self.state["lyrics_path"]) if self.state["lyrics_path"] else []

        worker = RenderWorker(self.ai_engine, self.state["selected_generated_image"], self.state["audio_path"], self.state["audio_trim_region"], pose_markers_data, bg_markers_data, lyrics_data)
        self.run_in_thread(worker, progress=self.update_progress_bar, finished=self.on_rendering_finished)

    def update_progress_bar(self, value, total):
        self.progress_bar.setValue(int(value / total * 100))

    def on_rendering_finished(self, frames_path, error_msg):
        self.render_button.setEnabled(True); self.render_button.setText("Render Video"); self.progress_bar.setVisible(False)
        if error_msg: QMessageBox.critical(self, "Render Failed", error_msg); return

        output_path, _ = QFileDialog.getSaveFileName(self, "Save Video", "output.mp4", "MP4 Files (*.mp4)")
        if not output_path:
            shutil.rmtree(frames_path)
            return

        self.compile_video(frames_path, output_path)

    def compile_video(self, frames_path, output_path):
        audio_input = self.state["audio_path"]
        trim_start, trim_end = self.state["audio_trim_region"]
        duration = trim_end - trim_start
        self.status_bar.showMessage("Compiling final video with FFmpeg...")

        if not os.path.exists(audio_input):
            QMessageBox.critical(self, "Error", "Audio file not found. Cannot compile video.")
            self.status_bar.showMessage("Error: Audio file missing.", 5000)
            if os.path.exists(frames_path):
                shutil.rmtree(frames_path)
            return

        ffmpeg_command = ['ffmpeg', '-y', '-framerate', '30', '-i', f'{frames_path}/frame_%05d.png', '-ss', str(trim_start), '-i', audio_input, '-t', str(duration), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', output_path]

        try:
            result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
            QMessageBox.information(self, "Success", f"Video saved to {output_path}")
            self.status_bar.showMessage("Video compilation successful!", 5000)
        except subprocess.CalledProcessError as e:
            error_message = f"Failed to compile video:\n{e.stderr}"
            QMessageBox.critical(self, "FFmpeg Error", error_message)
            self.status_bar.showMessage("FFmpeg compilation failed.", 5000)
        finally:
            # --- Automatic Cleanup ---
            self.status_bar.showMessage("Cleaning up temporary files...", 2000)
            if os.path.exists(frames_path):
                shutil.rmtree(frames_path)
            # Use .get() to safely access keys that might not exist
            audio_path = self.state.get("audio_path")
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError as e:
                    print(f"Error removing audio file: {e}")

            lyrics_path = self.state.get("lyrics_path")
            if lyrics_path and os.path.exists(lyrics_path):
                try:
                    os.remove(lyrics_path)
                except OSError as e:
                    print(f"Error removing lyrics file: {e}")

            self.status_bar.showMessage("Cleanup complete.", 3000)

    def closeEvent(self, event):
        self.stop_all_threads()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AnimationApp()
    window.show()
    sys.exit(app.exec())