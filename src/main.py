import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from end_to_end_pipeline.end_to_end_pipeline import EndToEndPipeline

def main():
    pipeline = EndToEndPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()