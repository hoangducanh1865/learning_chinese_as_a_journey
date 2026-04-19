import os
import sys
import warnings

# Suppress annoying deprecated UserWarning from third-party libs like jieba
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from end_to_end_pipeline.end_to_end_pipeline import EndToEndPipeline

def main():
    mode = input("Enter mode (asr/tts): ").strip().lower()
    pipeline = EndToEndPipeline(mode=mode)
    pipeline.run()

if __name__ == "__main__":
    main()