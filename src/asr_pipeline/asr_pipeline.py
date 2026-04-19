import os
import torch
import omegaconf
import whisperx
import gc
from tqdm import tqdm
from pypinyin import pinyin, Style
try:
    from .utils import *
except ImportError:
    from utils import *

# Fix for tqdm hanging in some environments when used in terminal
import tqdm.std
tqdm = tqdm.std.tqdm

class ASR:
    def __init__(self, device="cpu", compute_type="int8", batch_size=4, file_type=None, type=None, file_number=None):
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size 
        
        have_to_convert = False
        if file_type == "m4a": 
            # Convert from m4a to mp4
            have_to_convert = True
        elif file_type == "mp4":
            pass
        else:
            raise NotImplementedError(f"File type {file_type} is not supported")
        input_m4a_path = os.path.join("data/m4a_audio", type, f"{type}_{file_number}.{file_type}")
        input_mp4_path = os.path.join("data/mp4_video", type, f"{type}_{file_number}.mp4")
        if have_to_convert:
            convert_m4a_to_mp4(input_file=input_m4a_path, output_file=input_mp4_path)
        
        self.audio_file_path = input_mp4_path
        self.output_path = os.path.join("data/transcript_raw", type, f"{type}_{file_number}.txt")
    
    def speech2text(self):
        torch.serialization.add_safe_globals([
            omegaconf.listconfig.ListConfig,
            omegaconf.dictconfig.DictConfig,
            omegaconf.nodes.AnyNode,
        ])
        
        # --- Step 1: Speech Recognition (Whisper V3) ---
        print("--- Step 1: Recognizing Speech (Whisper V3) ---")

        with tqdm(total=100, desc="Transcribing Audio", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            model = whisperx.load_model("base", self.device, compute_type=self.compute_type) # small / base / large-v3
            audio = whisperx.load_audio(self.audio_file_path)
            result = model.transcribe(audio, batch_size=self.batch_size)
            pbar.update(100)

        # Clear VRAM/RAM
        self.model = None
        gc.collect()
        
        return audio, result
        
    def aligin_timestamps(self, audio, result):
        # --- Step 2: Time Alignment ---
        print("\n--- Step 2: Aligning Timestamps ---")

        # Load alignment model (Wav2Vec2 for Chinese)
        model, metadata = whisperx.load_align_model(language_code="zh", device=self.device)

        with tqdm(total=len(result["segments"]), desc="Aligning Segments") as pbar:
            result = whisperx.align(
                result["segments"], 
                model, 
                metadata, 
                audio, 
                self.device, 
                return_char_alignments=False
            )
            pbar.update(len(result["segments"]))
        
        return result
            
    def print_test_results(self, result):
        # --- TEST RESULTS ---
        print("\n--- TEST RESULTS (First 5 Segments) ---")
        for segment in result["segments"][:5]:
            cn_text = segment['text']
            pinyin_text = get_pinyin(cn_text)
            print(f"[{segment['start']:.2f}s]: {cn_text}")
            print(f"PY: {pinyin_text}\n")
            
    def save_transcript(self, result):
        print(f"--- Exporting transcript to: {self.output_path} ---")

        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                for segment in result["segments"]:
                    # Get start time in MM:SS
                    start_time = format_time_simple(segment.get('start', 0.0))
                    
                    # Get Chinese text
                    chinese_text = segment.get('text', '').strip()
                    
                    # Write to file in requested format
                    f.write(f"{start_time}\n")
                    f.write(f"{chinese_text}\n")
                    
                    # Optional: If you still want Pinyin included, uncomment the line below:
                    # f.write(f"{get_pinyin(chinese_text)}\n")
                    
            print("--- Export completed successfully! ---")
        except Exception as e:
            print(f"--- Error writing to file: {e} ---")
            
    def run_pipeline(self):
        audio, result = self.speech2text()
        result = self.aligin_timestamps(audio, result)
        self.print_test_results(result)
        self.save_transcript(result)
    

def test():
    # ASR just works for mp4 files, so if input file is any other kind of file, convert them to mp4 first
    file_type = input("Enter file type: (m4a/mp4): ")
    type = input("Enter type (podcast/audiobook): ")
    file_number = input("Enter file number: ")
    
    
    asr = ASR(file_type=file_type, type=type, file_number=file_number)
    asr.run_pipeline()


if __name__ == "__main__":
    # test()
    pass

