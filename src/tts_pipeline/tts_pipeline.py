import os
import re
import json
import wave
import numpy as np
import librosa
import unicodedata
from piper.voice import PiperVoice

# --- CONFIGURATION ---
PIPER_BASE = os.path.abspath("./models/piper/")
VN_BASE = os.path.abspath("./models/vie/")
AUDIO_SPEED = 1.0
SAMPLE_RATE = 22050

SPEAKER_CONFIG = {
    "vivos": os.path.join(VN_BASE, "vi_VN-vivos-x_low.onnx"),
    "ngochuyen": os.path.join(VN_BASE, "ngochuyennew.onnx"),
    "minhquang": os.path.join(VN_BASE, "minhquang.onnx"),
    "chaowen": os.path.join(PIPER_BASE, "chaowen/zh_CN-chaowen-medium.onnx"),
    "huayan": os.path.join(PIPER_BASE, "huayan/zh_CN-huayan-medium.onnx"),
}

class VietnameseSpeaker:
    def __init__(self, model_key="minhquang"):
        self.model_path = SPEAKER_CONFIG.get(model_key)
        self.voice = PiperVoice.load(self.model_path) if os.path.exists(self.model_path) else None

    def clean_text(self, text):
        number_map = {"0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
                      "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"}
        for num, word in number_map.items():
            text = text.replace(num, word)
        return unicodedata.normalize('NFC', text)

    def speak(self, text):
        if not self.voice: return None
        text = self.clean_text(text)
        
        # Try synthesize with length_scale (if supported), otherwise fallback
        try:
            # We must iterate inside a list comprehension or just use it directly
            audio_generator = self.voice.synthesize(text, length_scale=1.05)
            audio_chunks = [chunk.audio_int16_array for chunk in audio_generator if hasattr(chunk, 'audio_int16_array')]
        except TypeError:
            # If length_scale is unknown, use default synthesis
            audio_generator = self.voice.synthesize(text)
            audio_chunks = [chunk.audio_int16_array for chunk in audio_generator if hasattr(chunk, 'audio_int16_array')]
            
        if not audio_chunks: return None
        return np.concatenate(audio_chunks)

class ChineseSpeaker:
    def __init__(self, model_key="chaowen"):
        self.model_path = SPEAKER_CONFIG.get(model_key)
        self.voice = PiperVoice.load(self.model_path) if os.path.exists(self.model_path) else None

    def speak(self, text):
        if not self.voice: return None
        audio_chunks = [chunk.audio_int16_array for chunk in self.voice.synthesize(text) if hasattr(chunk, 'audio_int16_array')]
        if not audio_chunks: return None
        audio_data = np.concatenate(audio_chunks)
        
        # Audio processing (Time stretch)
        audio_float = audio_data.astype(np.float32) / 32768.0
        stretched = librosa.effects.time_stretch(audio_float, rate=AUDIO_SPEED)
        return (stretched * 32767.0).astype(np.int16)

class TTS:
    def __init__(self):
        self.vn_speaker = VietnameseSpeaker("minhquang")
        self.cn_male = ChineseSpeaker("chaowen")
        self.cn_female = ChineseSpeaker("huayan")

    def create_silence(self, duration_sec):
        return np.zeros(int(duration_sec * SAMPLE_RATE), dtype=np.int16)

    def process_vocab(self, json_path, output_path, limit="all"):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_segments = []
        timestamps = []
        current_time = 0.0
        
        pause_short_val = 0.6
        pause_long_val = 1.5
        pause_short = self.create_silence(pause_short_val)
        pause_long = self.create_silence(pause_long_val)

        print(f"Generating audio for: {json_path} (Limit: {limit})")
        
        # Apply limit if it's a number
        items = list(data.items())
        if str(limit).lower() != "all":
            items = items[:int(limit)]
            
        # Process word by word
        for hanzi, info in items:
            # Mark the start of a new word
            timestamps.append(f"{current_time:.2f} - {hanzi}")
            
            vi_word = info.get("vietnamese", [""])[0]
            egs = info.get("eg", [])

            # Audio Sequence logic
            sequence = [
                (vi_word, self.vn_speaker, pause_short_val),
                (hanzi, self.cn_male, pause_short_val),
                (hanzi, self.cn_female, pause_long_val)
            ]

            for text, speaker, pause_duration in sequence:
                audio = speaker.speak(text)
                if audio is not None:
                    all_segments.append(audio)
                    current_time += len(audio) / SAMPLE_RATE
                    
                    # Add pause
                    pause_samples = int(pause_duration * SAMPLE_RATE)
                    all_segments.append(np.zeros(pause_samples, dtype=np.int16))
                    current_time += pause_duration

            # Examples
            for eg in egs:
                eg_zh = eg.get("eg_chinese", "")
                eg_vi = eg.get("eg_vietnamese", "")
                
                eg_sequence = [
                    (eg_vi, self.vn_speaker, pause_short_val),
                    (eg_zh, self.cn_male, pause_short_val),
                    (eg_zh, self.cn_female, pause_long_val)
                ]
                for text, speaker, pause_duration in eg_sequence:
                    audio = speaker.speak(text)
                    if audio is not None:
                        all_segments.append(audio)
                        current_time += len(audio) / SAMPLE_RATE
                        
                        # Add pause
                        pause_samples = int(pause_duration * SAMPLE_RATE)
                        all_segments.append(np.zeros(pause_samples, dtype=np.int16))
                        current_time += pause_duration

        if not all_segments:
            print("No audio segments generated.")
            return

        final_audio = np.concatenate(all_segments)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save Audio
        with wave.open(output_path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SAMPLE_RATE)
            f.writeframes(final_audio.tobytes())
        
        # Save Timestamps
        ts_path = output_path.replace(".wav", ".txt")
        with open(ts_path, "w", encoding="utf-8") as f:
            f.write("\n".join(timestamps))
        
        print(f"✅ Success: {output_path}")
        print(f"✅ Timestamps: {ts_path}")

def test():
    # Example usage for testing
    json_in = "./data/transcript_vocab/podcast/podcast_1.json"
    wav_out = "./data/transcript_generated_vocab_audio/podcast/podcast_1.wav"
    
    # Process limit: "all" or specific number (e.g., 3)
    limit = 10
    
    if os.path.exists(json_in):
        pipeline = TTS()
        pipeline.process_vocab(json_in, wav_out, limit=limit)
    else:
        print(f"Skipping test: {json_in} not found.")

if __name__ == "__main__":
    # test()
    pass
