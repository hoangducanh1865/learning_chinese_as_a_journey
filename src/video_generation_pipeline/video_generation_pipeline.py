import os
import json
import numpy as np
from moviepy import AudioFileClip, ColorClip, TextClip, CompositeVideoClip, ImageClip
from PIL import Image, ImageDraw, ImageFont

class VideoGenerator:
    def __init__(self, width=1920, height=1080, fps=24):
        self.width = width
        self.height = height
        self.fps = fps
        # Fixed font paths for macOS with better Chinese/Vietnamese support
        self.font_path = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf" 
        self.font_bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

    def create_text_image(self, word_info, size):
        """Creates a professional 16:9 Youtube-style frame"""
        # Improved Color Palette
        color_sidebar = (255, 179, 115)     # Warm Orange
        color_bg = (255, 248, 235)          # Light Cream
        color_text_main = (40, 40, 40)      # Dark Grey
        color_text_sec = (80, 80, 80)       # Medium Grey
        color_accent = (26, 83, 92)         # Deep Teal
        
        img = Image.new('RGB', size, color=color_bg)
        draw = ImageDraw.Draw(img)
        w, h = size
        
        sidebar_w = 400
        # Draw Sidebar
        draw.rectangle([0, 0, sidebar_w, h], fill=color_sidebar)

        hanzi = word_info.get('hanzi', '')
        pinyin = word_info.get('pinyin', '')
        vietnamese = ", ".join(word_info.get('vietnamese', []))
        egs = word_info.get('eg', [])
        
        # Load Fonts with huge sizes
        try:
            f_huge = ImageFont.truetype(self.font_path, 280)
            f_pinyin = ImageFont.truetype(self.font_path, 80)
            f_vi_main = ImageFont.truetype(self.font_bold, 70)
            f_eg_zh = ImageFont.truetype(self.font_path, 60)
            f_eg_vi = ImageFont.truetype(self.font_path, 45)
            f_small = ImageFont.truetype(self.font_path, 35)
        except:
            f_huge = f_pinyin = f_vi_main = f_eg_zh = f_eg_vi = f_small = ImageFont.load_default()

        # --- SIDEBAR CONTENT (Vietnamese & Info) ---
        # Main Meaning (Side)
        draw.text((sidebar_w/2, 200), "NGHĨA:", font=f_small, fill="white", anchor="mm")
        # Multi-line wrap for long Vietnamese meanings
        if len(vietnamese) > 10:
            v_words = vietnamese.split()
            vietnamese = "\n".join([" ".join(v_words[i:i+2]) for i in range(0, len(v_words), 2)])
        draw.text((sidebar_w/2, 350), vietnamese, font=f_vi_main, fill="white", anchor="mm", align="center")
        
        # ID at bottom of sidebar
        word_id = word_info.get('id', '1')
        draw.text((sidebar_w/2, h - 100), f"WORD ID: {word_id}", font=f_small, fill="white", anchor="mm")

        # --- MAIN CONTENT CONTENT ---
        main_center_x = sidebar_w + (w - sidebar_w) / 2
        
        # Big Chinese Word
        draw.text((main_center_x, 250), hanzi, font=f_huge, fill=color_text_main, anchor="mm")
        # Pinyin
        draw.text((main_center_x, 420), pinyin, font=f_pinyin, fill=color_accent, anchor="mm")
        
        # Divider
        line_y = 500
        draw.line([sidebar_w + 100, line_y, w - 100, line_y], fill=color_accent, width=4)

        # Examples Section
        if egs:
            draw.text((sidebar_w + 100, 550), "VÍ DỤ:", font=f_small, fill=color_accent)
            
            y_cursor = 620
            for eg in egs[:2]: # Show top 2 examples clearly
                eg_zh = eg.get('eg_chinese', '')
                eg_py = eg.get('eg_pinyin', '')
                eg_vi = eg.get('eg_vietnamese', '')
                
                # Chinese Example
                draw.text((sidebar_w + 120, y_cursor), eg_zh, font=f_eg_zh, fill=color_text_main)
                y_cursor += 70
                # Pinyin & Vietnamese
                draw.text((sidebar_w + 120, y_cursor), f"({eg_py})", font=f_small, fill=color_text_sec)
                y_cursor += 45
                draw.text((sidebar_w + 120, y_cursor), f"→ {eg_vi}", font=f_eg_vi, fill=color_accent)
                y_cursor += 100

        return np.array(img)

    def generate_video(self, audio_path, timestamp_path, vocab_path, output_path):
        if not os.path.exists(audio_path) or not os.path.exists(timestamp_path):
            print(f"Missing data: {audio_path} or {timestamp_path}")
            return

        # Load data
        audio = AudioFileClip(audio_path)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            
        timestamps = []
        with open(timestamp_path, 'r', encoding='utf-8') as f:
            for line in f:
                if " - " in line:
                    time_str, hanzi = line.strip().split(" - ")
                    timestamps.append((float(time_str), hanzi))

        # Create clips
        clips = []
        total_duration = audio.duration
        
        for i, (start_time, hanzi) in enumerate(timestamps):
            end_time = timestamps[i+1][0] if i+1 < len(timestamps) else total_duration
            duration = end_time - start_time
            
            if duration <= 0: continue
            
            word_info = vocab_data.get(hanzi, {})
            word_info['hanzi'] = hanzi
            
            # Generate static frame for this word
            frame_arr = self.create_text_image(word_info, (self.width, self.height))
            img_clip = ImageClip(frame_arr).with_duration(duration).with_start(start_time)
            clips.append(img_clip)

        # Background clip
        bg = ColorClip(size=(self.width, self.height), color=(30, 30, 30), duration=total_duration)
        
        # Combine everything
        final_video = CompositeVideoClip([bg] + clips)
        final_video.audio = audio
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_video.write_videofile(output_path, fps=self.fps, codec="libx264", audio_codec="aac")
        print(f"✅ Video saved: {output_path}")

def test():
    # Example paths based on previous steps
    doc_type = "podcast"
    idx = 1
    
    audio_path = f"./data/transcript_generated_vocab_audio/{doc_type}/{doc_type}_{idx}.wav"
    ts_path = f"./data/transcript_generated_vocab_audio/{doc_type}/{doc_type}_{idx}.txt"
    vocab_path = f"./data/transcript_vocab/{doc_type}/{doc_type}_{idx}.json"
    video_out = f"./data/transcript_generated_vocab_video/{doc_type}/{doc_type}_{idx}.mp4"
    
    if os.path.exists(audio_path) and os.path.exists(vocab_path):
        generator = VideoGenerator()
        generator.generate_video(audio_path, ts_path, vocab_path, video_out)
    else:
        print("Required files for video generation test not found.")

if __name__ == "__main__":
    test()
