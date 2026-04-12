import os
import sys
import json

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transcript_preprocessing_pipeline.transcript_preprocessing_pipeline import TranscriptPreprocessor
from tts_pipeline.tts_pipeline import TTS
from video_generation_pipeline.video_generation_pipeline import VideoGenerator

class EndToEndPipeline:
    def __init__(self):
        self.preprocessor = TranscriptPreprocessor()
        self.tts = TTS()
        self.video_gen = VideoGenerator()

    def run(self):
        print("\n" + "="*60)
        print("🚀 STARTING CHINESE LEARNING END-TO-END PIPELINE")
        print("="*60)

        # 1. Transcript Phase
        # Now handles its own doc_type/idx inputs once if not provided
        print("\n[PHASE 1] Analyze text and generate JSON vocab...")
        doc_type, idx = self.preprocessor.run_pipeline()
        
        if not doc_type or not idx:
            print("❌ Preprocessing cancelled or failed.")
            return
        
        vocab_path = f"./data/transcript_vocab/{doc_type}/{doc_type}_{idx}.json"
        if not os.path.exists(vocab_path):
            print(f"❌ Vocabulary file not found: {vocab_path}")
            return

        # Get range input
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        max_idx = len(vocab_data)
        print(f"\nVocabulary loaded. Found {max_idx} words.")
        
        range_input = input(f"Enter range to process (e.g. '1-{max_idx}', or 'all'): ").strip().lower()
        
        start_idx = 1
        end_idx = max_idx
        
        if range_input != 'all':
            try:
                if '-' in range_input:
                    s, e = range_input.split('-')
                    start_idx = int(s)
                    end_idx = int(e)
                else:
                    end_idx = int(range_input)
            except ValueError:
                print("⚠️ Invalid format. Processing all.")

        audio_path = f"./data/transcript_generated_vocab_audio/{doc_type}/{doc_type}_{idx}.wav"
        ts_path = f"./data/transcript_generated_vocab_audio/{doc_type}/{doc_type}_{idx}.txt"
        video_path = f"./data/transcript_generated_vocab_video/{doc_type}/{doc_type}_{idx}.mp4"

        # 2. TTS
        print(f"\n[PHASE 2] Generating VN/CN audio (Range: {start_idx} to {end_idx})...")
        self.tts.process_vocab(vocab_path, audio_path, start_idx=start_idx, end_idx=end_idx)

        # 3. Video
        print("\n[PHASE 3] Creating 16:9 YouTube video...")
        if os.path.exists(audio_path) and os.path.exists(ts_path):
            self.video_gen.generate_video(audio_path, ts_path, vocab_path, video_path)
        else:
            print("❌ Audio or Timestamps missing. Skipping video stage.")

        print("\n" + "="*60)
        print(f"✅ SUCCESS: Processed {doc_type} #{idx}")
        print(f"🎬 Video: {video_path}")
        print("="*60)
        

def test():
    pipeline = EndToEndPipeline()
    pipeline.run()

if __name__ == "__main__":
    test()
    pass
