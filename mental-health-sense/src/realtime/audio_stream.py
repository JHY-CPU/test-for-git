"""
实时音频流采集模块

支持多种音频输入源：
    - 麦克风实时采集
    - RTSP/RTMP摄像头音频流
    - 文件模拟实时流（测试用）

功能：
    1. 持续采集音频
    2. 自动分段（按静音或固定时长）
    3. 缓存到队列供SenseVoice消费
"""

import queue
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import numpy as np


class AudioStream(ABC):
    """音频流基类"""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: int = 10,  # 秒
        buffer_size: int = 100,
    ):
        """
        Args:
            sample_rate: 采样率 (Hz)
            chunk_duration: 音频分段时长（秒）
            buffer_size: 队列缓冲区大小
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.audio_queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @abstractmethod
    def _read_audio_chunk(self) -> tuple[np.ndarray, float]:
        """
        读取一段音频数据（子类实现）

        Returns:
            (audio_data, timestamp)
            - audio_data: (n_samples,) numpy数组，float32
            - timestamp: 当前时间戳（秒）
        """
        pass

    def _capture_loop(self):
        """音频采集主循环"""
        print("[AudioStream] 开始采集...")
        while not self._stop_event.is_set():
            try:
                audio_chunk, timestamp = self._read_audio_chunk()
                if audio_chunk is not None and len(audio_chunk) > 0:
                    self.audio_queue.put((audio_chunk, timestamp), timeout=1)
            except queue.Full:
                print("[AudioStream] 队列已满，丢弃旧数据")
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
            except Exception as e:
                print(f"[AudioStream] 采集异常: {e}")
                time.sleep(0.1)

    def start(self):
        """启动音频采集线程"""
        if self._thread is not None and self._thread.is_alive():
            print("[AudioStream] 已在运行")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("[AudioStream] 采集线程已启动")

    def stop(self):
        """停止音频采集"""
        print("[AudioStream] 停止采集...")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        print("[AudioStream] 已停止")

    def get_audio(self, timeout: float = 1.0) -> Optional[tuple[np.ndarray, float]]:
        """
        从队列获取音频数据

        Args:
            timeout: 超时时间（秒）

        Returns:
            (audio_data, timestamp) 或 None
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class MicrophoneStream(AudioStream):
    """麦克风实时采集"""

    def __init__(self, device_index: Optional[int] = None, **kwargs):
        """
        Args:
            device_index: 麦克风设备索引（None=默认设备）
        """
        super().__init__(**kwargs)
        self.device_index = device_index
        self._audio_interface = None
        self._stream = None

    def _initialize_pyaudio(self):
        """延迟初始化PyAudio（避免导入时报错）"""
        try:
            import pyaudio
            self._audio_interface = pyaudio.PyAudio()
            self._stream = self._audio_interface.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.sample_rate * self.chunk_duration,
            )
            print(f"[MicrophoneStream] 已打开麦克风 (device={self.device_index})")
        except Exception as e:
            print(f"[MicrophoneStream] 初始化失败: {e}")
            raise

    def _read_audio_chunk(self) -> tuple[np.ndarray, float]:
        if self._stream is None:
            self._initialize_pyaudio()

        # 读取音频数据
        audio_bytes = self._stream.read(
            self.sample_rate * self.chunk_duration,
            exception_on_overflow=False,
        )
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
        timestamp = time.time()
        return audio_data, timestamp

    def stop(self):
        super().stop()
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._audio_interface:
            self._audio_interface.terminate()


class RTSPStream(AudioStream):
    """RTSP摄像头音频流采集"""

    def __init__(self, rtsp_url: str, **kwargs):
        """
        Args:
            rtsp_url: RTSP流地址，例如 "rtsp://192.168.1.100:554/stream"
        """
        super().__init__(**kwargs)
        self.rtsp_url = rtsp_url
        self._cap = None

    def _initialize_capture(self):
        """初始化OpenCV视频捕获"""
        try:
            import cv2
            self._cap = cv2.VideoCapture(self.rtsp_url)
            if not self._cap.isOpened():
                raise RuntimeError(f"无法打开RTSP流: {self.rtsp_url}")
            print(f"[RTSPStream] 已连接到 {self.rtsp_url}")
        except Exception as e:
            print(f"[RTSPStream] 初始化失败: {e}")
            raise

    def _read_audio_chunk(self) -> tuple[np.ndarray, float]:
        """
        从RTSP流中提取音频（需要ffmpeg支持）

        注意：OpenCV默认只支持视频流，提取音频需要额外配置
        这里提供一个框架，实际使用时需要对接ffmpeg或gstreamer
        """
        if self._cap is None:
            self._initialize_capture()

        # TODO: 实际实现需要使用ffmpeg提取音频流
        # 这里仅作为占位符
        import cv2
        ret, frame = self._cap.read()
        if not ret:
            print("[RTSPStream] 读取帧失败")
            time.sleep(0.1)
            return np.zeros(self.sample_rate * self.chunk_duration, dtype=np.float32), time.time()

        # 占位符：返回静音
        audio_data = np.zeros(self.sample_rate * self.chunk_duration, dtype=np.float32)
        timestamp = time.time()
        return audio_data, timestamp

    def stop(self):
        super().stop()
        if self._cap:
            self._cap.release()


class FileSimulatorStream(AudioStream):
    """文件模拟实时流（用于测试）"""

    def __init__(self, audio_file: str, loop: bool = True, **kwargs):
        """
        Args:
            audio_file: 音频文件路径
            loop: 是否循环播放
        """
        super().__init__(**kwargs)
        self.audio_file = Path(audio_file)
        self.loop = loop
        self._audio_data: Optional[np.ndarray] = None
        self._current_pos = 0

    def _load_audio(self):
        """加载音频文件"""
        try:
            from scipy.io import wavfile
            sr, audio = wavfile.read(self.audio_file)

            # 转换为单声道float32
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32) / 32768.0  # 归一化到[-1, 1]

            # 重采样到目标采样率
            if sr != self.sample_rate:
                from scipy.signal import resample
                target_length = int(len(audio) * self.sample_rate / sr)
                audio = resample(audio, target_length)

            self._audio_data = audio
            print(f"[FileSimulator] 已加载 {self.audio_file} ({len(audio)/sr:.1f}s)")
        except Exception as e:
            print(f"[FileSimulator] 加载失败: {e}")
            raise

    def _read_audio_chunk(self) -> tuple[np.ndarray, float]:
        if self._audio_data is None:
            self._load_audio()

        chunk_samples = self.sample_rate * self.chunk_duration

        # 循环播放
        if self._current_pos >= len(self._audio_data):
            if self.loop:
                self._current_pos = 0
            else:
                return np.zeros(chunk_samples, dtype=np.float32), time.time()

        # 提取数据块
        end_pos = min(self._current_pos + chunk_samples, len(self._audio_data))
        chunk = self._audio_data[self._current_pos:end_pos]

        # 不足时补零
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        self._current_pos = end_pos

        # 模拟实时延迟
        time.sleep(self.chunk_duration)

        return chunk, time.time()


# 使用示例
if __name__ == "__main__":
    # 示例1：麦克风采集
    print("=== 测试麦克风采集 ===")
    mic = MicrophoneStream(chunk_duration=3)
    mic.start()

    try:
        for i in range(3):
            audio, ts = mic.get_audio(timeout=5)
            if audio is not None:
                print(f"[{i+1}] 收到音频: {len(audio)} samples, timestamp={ts:.2f}")
    finally:
        mic.stop()
