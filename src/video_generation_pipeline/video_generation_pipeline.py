import os
import json
import numpy as np
from tqdm import tqdm
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
        # High-quality handwritten-style/textbook font for Chinese learners
        self.font_handwritten = os.path.join(os.getcwd(), "fonts/LXGWWenKai-Regular.ttf")

    def create_text_image(self, word_info, size):
        """Creates a professional 16:9 Youtube-style frame"""
        # Improved Color Palette
        color_sidebar = (255, 160, 190)     # Slightly lighter than before
        color_bg = (255, 240, 245)          # Lavender Blush (Light Pink)
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
            # Check if downloaded font exists, fallback to system font if not
            if os.path.exists(self.font_handwritten):
                f_huge = ImageFont.truetype(self.font_handwritten, 280)
                f_eg_zh = ImageFont.truetype(self.font_handwritten, 75)
            else:
                f_huge = ImageFont.truetype("/System/Library/Fonts/Supplemental/Songti.ttc", 280, index=0)
                f_eg_zh = ImageFont.truetype("/System/Library/Fonts/Supplemental/Songti.ttc", 75, index=0)
                
            f_pinyin = ImageFont.truetype(self.font_path, 80)
            f_vi_main = ImageFont.truetype(self.font_bold, 70)
            f_eg_vi = ImageFont.truetype(self.font_path, 35)
            f_small = ImageFont.truetype(self.font_path, 35)
        except:
            f_huge = f_pinyin = f_vi_main = f_eg_zh = f_eg_vi = f_small = ImageFont.load_default()

        # --- SIDEBAR CONTENT (Vietnamese & Info) ---
        # Main Meaning (Side)
        # draw.text((sidebar_w/2, 200), "NGHĨA:", font=f_small, fill="white", anchor="mm")
        
        # Multi-line wrap for long Vietnamese meanings
        if len(vietnamese) > 10:
            v_words = vietnamese.split()
            vietnamese = "\n".join([" ".join(v_words[i:i+2]) for i in range(0, len(v_words), 2)])
        draw.text((sidebar_w/2, 350), vietnamese, font=f_vi_main, fill="white", anchor="mm", align="center")
        
        # ID and HSK at bottom of sidebar
        word_id = word_info.get('id', '1')
        total_count = word_info.get('total_vocab_size', word_info.get('total_count', '?'))
        hsk_level = word_info.get('hsk_level', word_info.get('hsk', 'N/A'))
        
        # Display only the number part (e.g., hsk_1 -> 1, hsk_789 -> 789)
        if isinstance(hsk_level, str) and hsk_level.startswith('hsk_'):
            display_hsk = hsk_level.split('_')[1]
        else:
            display_hsk = hsk_level
        
        draw.text((sidebar_w/2, h - 130), f"ID: {word_id}/{total_count}", font=f_small, fill="white", anchor="mm")
        draw.text((sidebar_w/2, h - 80), f"HSK: {display_hsk}", font=f_small, fill="white", anchor="mm")
        
        # --- MAIN CONTENT CONTENT ---
        main_center_x = sidebar_w + (w - sidebar_w) / 2
        
        # Calculate Grid and Hanzi positions
        grid_size = 320
        spacing = 20
        num_chars = len(hanzi)
        total_width = (grid_size * num_chars) + (spacing * (num_chars - 1))
        start_x = main_center_x - total_width / 2
        grid_y_center = 240
        grid_color = (220, 100, 100, 150) # Soft Red for grid

        for i in range(num_chars):
            char_center_x = start_x + (grid_size / 2) + i * (grid_size + spacing)
            
            # Grid Coordinates
            gx1, gy1 = char_center_x - grid_size/2, grid_y_center - grid_size/2
            gx2, gy2 = char_center_x + grid_size/2, grid_y_center + grid_size/2
            
            # Outer Box
            draw.rectangle([gx1, gy1, gx2, gy2], outline=grid_color, width=3)
            # Vertical & Horizontal
            draw.line([char_center_x, gy1, char_center_x, gy2], fill=grid_color, width=1)
            draw.line([gx1, grid_y_center, gx2, grid_y_center], fill=grid_color, width=1)
            # Diagonals
            draw.line([gx1, gy1, gx2, gy2], fill=grid_color, width=1)
            draw.line([gx1, gy2, gx2, gy1], fill=grid_color, width=1)
            
            # Draw individual Hanzi
            draw.text((char_center_x, grid_y_center), hanzi[i], font=f_huge, fill=color_text_main, anchor="mm")

        draw.text((main_center_x, 445), pinyin, font=f_pinyin, fill=color_accent, anchor="mm")
        
        # Divider
        line_y = 500
        draw.line([sidebar_w + 100, line_y, w - 100, line_y], fill=color_accent, width=4)

        # Examples Section
        if egs:
            y_cursor = 580
            # Always show the examples provided in word_info['eg']
            # We already handled paging in generate_video()
            for eg in egs:
                eg_zh = eg.get('eg_chinese', '')
                eg_py = eg.get('eg_pinyin', '')
                eg_vi = eg.get('eg_vietnamese', '')
                
                # Chinese Example
                draw.text((sidebar_w + 120, y_cursor), eg_zh, font=f_eg_zh, fill=color_text_main)
                y_cursor += 85
                # Pinyin & Vietnamese - Upper Case first letter of pinyin
                display_eg_py = eg_py[0].upper() + eg_py[1:] if eg_py else ""
                draw.text((sidebar_w + 120, y_cursor), display_eg_py, font=f_small, fill=color_text_sec)
                y_cursor += 45
                draw.text((sidebar_w + 120, y_cursor), eg_vi, font=f_eg_vi, fill=color_accent)
                y_cursor += 110

        return np.array(img)

        # Combine everything
        final_video = CompositeVideoClip([bg] + clips)
        final_video.audio = audio
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_video.write_videofile(output_path, fps=self.fps, codec="libx264", audio_codec="aac", logger="bar")
        print(f"✅ Video saved: {output_path}")

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
                    time_str, label = line.strip().split(" - ")
                    # label might be "我" or "我_page_2"
                    timestamps.append((float(time_str), label))

        total_words = len(vocab_data)
        # Create clips
        clips = []
        total_duration = audio.duration
        
        pbar = tqdm(enumerate(timestamps), total=len(timestamps), desc="Creating Video Frames", unit="frame")
        for i, (start_time, label) in pbar:
            pbar.set_postfix({"label": label})
            end_time = timestamps[i+1][0] if i+1 < len(timestamps) else total_duration
            duration = end_time - start_time
            
            if duration <= 0: continue
            
            # Extract basic hanzi and page index
            if "_page_" in label:
                hanzi, page_str = label.split("_page_")
                page_idx = int(page_str) - 1 # 0-indexed for egs logic
            else:
                hanzi = label
                page_idx = 0
                
            word_info = vocab_data.get(hanzi, {}).copy()
            word_info['hanzi'] = hanzi
            word_info['total_vocab_size'] = total_words
            
            # Filter examples based on page index (2 per page)
            egs = word_info.get('eg', [])
            start_eg = page_idx * 2
            word_info['eg'] = egs[start_eg : start_eg + 2]
            
            # Generate static frame for this word/page
            frame_arr = self.create_text_image(word_info, (self.width, self.height))
            img_clip = ImageClip(frame_arr).with_duration(duration).with_start(start_time)
            clips.append(img_clip)

        # Background clip
        bg = ColorClip(size=(self.width, self.height), color=(30, 30, 30), duration=total_duration)
        
        # Combine everything
        final_video = CompositeVideoClip([bg] + clips)
        final_video.audio = audio
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_video.write_videofile(output_path, fps=self.fps, codec="libx264", audio_codec="aac", logger="bar")
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
    # test()
    pass