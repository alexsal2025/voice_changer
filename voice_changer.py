import tkinter as tk
from tkinter import ttk, messagebox
import pyaudio
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
import threading
import time

# === Настройки ===
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
MAX_DURATION = 60  # 1 минута
PITCH_SHIFT = 2    # По умолчанию +2 полутона

class VoiceChangerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎤 Голосовой изменятор — только тон (без ускорения)")
        self.root.geometry("550x420")
        self.root.resizable(False, False)
        self.root.configure(bg="#f0f0f0")

        # Данные
        self.recording = False
        self.audio_data = None
        self.processed_data = None
        self.pitch_shift = tk.DoubleVar(value=PITCH_SHIFT)
        self.record_seconds = 0
        self.timer_running = False

        # Создание интерфейса
        self.create_widgets()

    def create_widgets(self):
        # Заголовок
        title = tk.Label(
            self.root,
            text="🎤 Голосовой изменятор (только тон)",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0",
            fg="#333"
        )
        title.pack(pady=10)

        # Инструкция
        instr = tk.Label(
            self.root,
            text="Запиши 'траляля' — и получи голос с другим тоном!",
            font=("Arial", 11),
            bg="#f0f0f0",
            fg="#555"
        )
        instr.pack(pady=5)

        # Таймер
        timer_frame = tk.Frame(self.root, bg="#f0f0f0")
        timer_frame.pack(pady=8)

        tk.Label(timer_frame, text="⏱ Время записи: ", bg="#f0f0f0", font=("Arial", 10)).pack(side=tk.LEFT)
        self.timer_label = tk.Label(timer_frame, text="00:00", bg="#f0f0f0", font=("Arial", 10, "bold"), fg="#d32f2f")
        self.timer_label.pack(side=tk.LEFT, padx=5)

        # Слайдер тона
        pitch_frame = tk.Frame(self.root, bg="#f0f0f0")
        pitch_frame.pack(pady=10)

        tk.Label(pitch_frame, text="Тон: ", bg="#f0f0f0").pack(side=tk.LEFT)
        pitch_slider = ttk.Scale(
            pitch_frame,
            from_=-12,
            to=12,
            orient=tk.HORIZONTAL,
            length=250,
            variable=self.pitch_shift,
            command=self.update_pitch_label
        )
        pitch_slider.pack(side=tk.LEFT, padx=5)

        self.pitch_label = tk.Label(pitch_frame, text="+2.0 полутона", bg="#f0f0f0", font=("Arial", 10))
        self.pitch_label.pack(side=tk.LEFT)

        # Кнопки
        btn_frame = tk.Frame(self.root, bg="#f0f0f0")
        btn_frame.pack(pady=20)

        self.record_btn = tk.Button(
            btn_frame,
            text="⏺ Записать голос",
            font=("Arial", 12),
            bg="#4CAF50",
            fg="white",
            width=15,
            command=self.toggle_recording
        )
        self.record_btn.grid(row=0, column=0, padx=5)

        self.process_btn = tk.Button(
            btn_frame,
            text="🔄 Применить эффект",
            font=("Arial", 12),
            bg="#2196F3",
            fg="white",
            width=15,
            state=tk.DISABLED,
            command=self.process_audio
        )
        self.process_btn.grid(row=0, column=1, padx=5)

        self.play_btn = tk.Button(
            btn_frame,
            text="▶ Проиграть",
            font=("Arial", 12),
            bg="#FF9800",
            fg="white",
            width=15,
            state=tk.DISABLED,
            command=self.play_audio
        )
        self.play_btn.grid(row=1, column=0, padx=5, pady=10)

        self.save_btn = tk.Button(
            btn_frame,
            text="💾 Сохранить",
            font=("Arial", 12),
            bg="#9C27B0",
            fg="white",
            width=15,
            state=tk.DISABLED,
            command=self.save_audio
        )
        self.save_btn.grid(row=1, column=1, padx=5, pady=10)

        # Статус
        self.status_var = tk.StringVar(value="Готово")
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#666"
        )
        self.status_label.pack(pady=10)

        # Подсказка
        hint = tk.Label(
            self.root,
            text="❗ Максимум 60 секунд записи. Только тон меняется!",
            font=("Arial", 9),
            bg="#f0f0f0",
            fg="#888"
        )
        hint.pack(side=tk.BOTTOM, pady=10)

    def update_pitch_label(self, value):
        self.pitch_label.config(text=f"{float(value):+.1f} полутона")

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if self.recording:
            return

        self.recording = True
        self.record_seconds = 0
        self.timer_running = True
        self.status_var.set("🎙 Запись... (макс. 60 сек)")
        self.record_btn.config(text="⏹ Остановить", bg="#f44336")

        # Запускаем таймер
        threading.Thread(target=self._update_timer, daemon=True).start()
        # Запускаем запись
        threading.Thread(target=self._record_audio, daemon=True).start()

    def stop_recording(self):
        self.recording = False
        self.timer_running = False
        self.status_var.set("✅ Запись завершена!")
        self.record_btn.config(text="⏺ Записать голос", bg="#4CAF50")
        self.process_btn.config(state=tk.NORMAL)

    def _update_timer(self):
        while self.timer_running and self.record_seconds < MAX_DURATION:
            time.sleep(1)
            self.record_seconds += 1
            mins, secs = divmod(self.record_seconds, 60)
            self.timer_label.config(text=f"{mins:02}:{secs:02}")
        
        # Автоматическая остановка при достижении лимита
        if self.record_seconds >= MAX_DURATION and self.recording:
            self.stop_recording()

    def _record_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        frames = []
        total_chunks = int(RATE / CHUNK * MAX_DURATION)

        for i in range(total_chunks):
            if not self.recording:
                break
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        self.audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        self.timer_running = False
        self.timer_label.config(text="00:00")

    def process_audio(self):
        if self.audio_data is None:
            messagebox.showwarning("Предупреждение", "Сначала запишите голос!")
            return

        self.status_var.set("🔄 Обработка... (может занять несколько секунд)")
        self.root.update()

        # Преобразуем в float32 для librosa
        audio_float = self.audio_data.astype(np.float32) / 32768.0

        # Меняем только тон (без изменения скорости)
        # librosa.effects.pitch_shift не меняет длительность
        shifted_audio = librosa.effects.pitch_shift(
            y=audio_float,
            sr=RATE,
            n_steps=self.pitch_shift.get()
        )

        # Возвращаем в int16
        shifted_audio = np.clip(shifted_audio, -1.0, 1.0)
        shifted_audio = (shifted_audio * 32767).astype(np.int16)

        self.processed_data = shifted_audio
        self.status_var.set("✅ Эффект применён!")
        self.play_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)

    def play_audio(self):
        if self.processed_data is None:
            messagebox.showwarning("Предупреждение", "Сначала примените эффект!")
            return

        self.status_var.set("▶ Проигрывание...")
        self.root.update()

        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            frames_per_buffer=CHUNK
        )

        chunk_size = CHUNK
        for i in range(0, len(self.processed_data), chunk_size):
            chunk = self.processed_data[i:i+chunk_size]
            stream.write(chunk.tobytes())

        stream.stop_stream()
        stream.close()
        p.terminate()

        self.status_var.set("✅ Проиграно!")

    def save_audio(self):
        if self.processed_data is None:
            messagebox.showwarning("Предупреждение", "Сначала примените эффект!")
            return

        sf.write("output.wav", self.processed_data, RATE)
        self.status_var.set("💾 Сохранено как 'output.wav'")


# === Запуск приложения ===
if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceChangerApp(root)
    root.mainloop()