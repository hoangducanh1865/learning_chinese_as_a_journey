import subprocess
import os
import re
from tqdm import tqdm
from pypinyin import pinyin, Style


def get_pinyin(text):
    res = pinyin(text, style=Style.TONE)
    return " ".join([item[0] for item in res])

def format_time_simple(seconds):
    """
    Converts total seconds into MM:SS format.
    Example: 789.0 -> 13:09
    """
    total_seconds = int(seconds)
    minutes, seconds = divmod(total_seconds, 60)
    # If your audio is longer than 60 mins, this still works (e.g., 65:01)
    return f"{minutes:02}:{seconds:02}"

def get_duration(input_file):
    """Get the duration of the input file in seconds using ffprobe."""
    command = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', input_file
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None

def convert_m4a_to_mp4(input_file, output_file):
    """
    Converts m4a to mp4 with a black screen and displays progress.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    duration = get_duration(input_file)
    if duration is None:
        print("Could not determine file duration. Progress bar might not be accurate.")
        duration = 1  # Fallback

    command = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', 'color=c=black:s=1280x720',
        '-i', input_file,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'stillimage',
        '-c:a', 'copy',
        '-pix_fmt', 'yuv420p',
        '-shortest',
        output_file,
        '-y',
        '-stats'
    ]

    print(f"Converting: {os.path.basename(input_file)}")
    
    # Use tqdm for progress bar
    pbar = tqdm(total=100, desc="Conversion Progress", unit="%")
    
    # Run ffmpeg and parse its stderr for progress
    process = subprocess.Popen(command, stderr=subprocess.PIPE, universal_newlines=True)
    
    # Pattern to match time=HH:MM:SS.ms in ffmpeg output
    time_pattern = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")
    
    last_pct = 0
    for line in process.stderr:
        match = time_pattern.search(line)
        if match:
            hours, minutes, seconds = map(float, match.groups())
            current_time = hours * 3600 + minutes * 60 + seconds
            pct = min(99, (current_time / duration) * 100)
            pbar.update(pct - last_pct)
            last_pct = pct
            
    process.wait()
    if process.returncode == 0:
        pbar.update(100 - last_pct)
        pbar.close()
        print(f"Successfully created: {output_file}")
    else:
        pbar.close()
        print(f"Error occurred during conversion. Exit code: {process.returncode}")


def test():
    input_path = "/Users/hoangducanh/Documents/tmp_it/python/test_projects/learning_chinese_as_a_journey/notebook/data/7像素卷积核与CNN底层算法.m4a"
    output_path = "/Users/hoangducanh/Documents/tmp_it/python/test_projects/learning_chinese_as_a_journey/notebook/data/7像素卷积核与CNN底层算法.mp4"

    convert_m4a_to_mp4(input_path, output_path)


if __name__ == "__main__":
    # test()
    pass
