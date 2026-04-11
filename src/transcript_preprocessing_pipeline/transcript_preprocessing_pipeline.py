import os
import re
import json
import jieba
import time
from collections import Counter, OrderedDict
from pypinyin import pinyin, Style
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MAX_API_RETRY = 5
BATCH_SIZE = 8


class TranscriptPreprocessor:
    def __init__(self):
        self.base_raw = "./data/transcript_raw"
        self.base_vocab = "./data/transcript_vocab"
        self.hsk_folder = "./data/hsk_vocab_up2date"
        self.unknown_path = os.path.join(self.hsk_folder, "hsk_unknown.json")

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # =======================
    def has_valid_eg(self, entry):
        eg = entry.get("eg", None)
        return isinstance(eg, list) and len(eg) > 0

    # =======================
    def load_hsk_lookup(self):
        lookup = {}
        level_map = {}

        if not os.path.exists(self.hsk_folder):
            return lookup, level_map

        for f in os.listdir(self.hsk_folder):
            if f.endswith(".json") and f != "hsk_unknown.json":
                level = f.replace(".json", "")
                with open(os.path.join(self.hsk_folder, f), "r", encoding="utf-8") as j:
                    data = json.load(j)
                    for word, info in data.items():
                        lookup[word] = info
                        level_map[word] = level

        return lookup, level_map

    # =======================
    def load_unknown_lookup(self):
        if os.path.exists(self.unknown_path):
            with open(self.unknown_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_unknown_lookup(self, data):
        with open(self.unknown_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    # =======================
    def get_next_unknown_id(self, unknown_lookup):
        if not unknown_lookup:
            return 1

        max_id = 0
        for entry in unknown_lookup.values():
            if "id" in entry:
                max_id = max(max_id, entry["id"])

        return max_id + 1

    # =======================
    def print_vocab_table(self, text, unknown_lookup):
        hsk_lookup, level_map = self.load_hsk_lookup()

        text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
        words = jieba.lcut(text)
        counts = Counter(words)

        vocab_list = []

        for word, count in counts.items():
            pinyin_str = "".join([i[0] for i in pinyin(word, style=Style.TONE)])

            if word in hsk_lookup:
                level = level_map[word]
                vi_list = hsk_lookup[word].get("vietnamese", [])
                en_list = hsk_lookup[word].get("english", [])
            elif word in unknown_lookup:
                level = "unknown"
                vi_list = unknown_lookup[word].get("vietnamese", [])
                en_list = unknown_lookup[word].get("english", [])
            else:
                level = "unknown"
                vi_list = []
                en_list = []

            vocab_list.append(
                {
                    "word": word,
                    "count": count,
                    "pinyin": pinyin_str,
                    "level": level,
                    "vi": "; ".join(vi_list) if vi_list else "unknown",
                    "en": "; ".join(en_list) if en_list else "unknown",
                    "vi_list": vi_list,
                    "en_list": en_list
                }
            )

        def get_hsk_num(level):
            if level == "unknown":
                return 999
            nums = re.findall(r"\d+", level)
            return int(nums[0]) if nums else 998

        vocab_list.sort(key=lambda x: (get_hsk_num(x["level"]), -x["count"]))

        print(
            f"{'No.':<4} | {'Từ':<8} | {'Pinyin':<14} | {'Level':<8} | {'Freq':<5} | {'Vietnamese':<20} | {'English'}"
        )
        print("-" * 140)

        for i, item in enumerate(vocab_list, 1):
            print(
                f"{i:<4} | {item['word']:<8} | {item['pinyin']:<14} | {item['level']:<8} | {item['count']:<5} | {item['vi']:<20} | {item['en']}"
            )

        return vocab_list

    # =======================
    def generate_batch_ai_data(self, batch_words):
        prompt = """
You are a Chinese teacher.

For each word:
- You are given Vietnamese meanings (possibly multiple meanings in ONE line).
- You may also be given English meanings.

Your task:
1. Keep ALL Vietnamese meanings (DO NOT remove or change them).
2. If Vietnamese meanings are missing → generate them.
3. Generate English meanings if missing.
4. Generate example sentences.

Rules:
- NEVER use "unknown" as a meaning.
- Each meaning MUST have at least one example.
- "meaning" field MUST EXACTLY match Vietnamese meanings.
- Keep sentences simple and natural.

Return ONLY JSON:
{
  "word": {
    "vietnamese": ["..."],
    "english": ["..."],
    "eg": [
      {
        "meaning": "...",
        "eg_chinese": "...",
        "eg_pinyin": "...",
        "eg_vietnamese": "..."
      }
    ]
  }
}
"""

        prompt += "\nDATA:\n"

        for w in batch_words:
            vi_list = w.get("vi_list", [])
            en_list = w.get("en_list", [])

            vi = ", ".join(vi_list) if vi_list else ""
            en = ", ".join(en_list) if en_list else ""

            prompt += f"- {w['word']} ({w['pinyin']}) | Vietnamese: {vi} | English: {en}\n"

        for attempt in range(1, MAX_API_RETRY + 1):
            try:
                response = self.client.responses.create(
                    model="gpt-4o-mini",
                    input=prompt
                )

                text = response.output_text.strip()
                text = text.replace("```json", "").replace("```", "").strip()

                return json.loads(text)

            except Exception as e:
                print(f"[RETRY {attempt}] batch failed: {e}")
                time.sleep(1 + attempt)

        print("[FAILED BATCH]")
        return None

    # =======================
    def process_batch(self, batch, vocab_result, unknown_lookup):
        print(f"\n🚀 PROCESS BATCH ({len(batch)} words)")

        result = self.generate_batch_ai_data(batch)
        if not result:
            return

        next_id = self.get_next_unknown_id(unknown_lookup)

        for item in batch:
            word = item["word"]
            ai_data = result.get(word, {})
            old_entry = unknown_lookup.get(word, {})

            # 🔥 merge vietnamese
            old_vi = old_entry.get("vietnamese", [])
            new_vi = ai_data.get("vietnamese", [])
            vietnamese = list(set(old_vi + new_vi))

            if len(vietnamese) > 1 and "unknown" in vietnamese:
                vietnamese = [m for m in vietnamese if m != "unknown"]

            # 🔥 merge english
            old_en = old_entry.get("english", [])
            new_en = ai_data.get("english", [])
            english = list(set(old_en + new_en))

            if len(english) > 1 and "unknown" in english:
                english = [m for m in english if m != "unknown"]

            new_entry = OrderedDict()
            new_entry["id"] = old_entry.get("id", next_id)
            new_entry["pinyin"] = item["pinyin"]
            new_entry["vietnamese"] = vietnamese if vietnamese else ["unknown"]
            new_entry["english"] = english if english else ["unknown"]
            new_entry["num_appear"] = 0

            cleaned_eg = self.clean_eg_text(ai_data.get("eg", []))
            new_entry["eg"] = cleaned_eg

            if word not in unknown_lookup:
                next_id += 1

            vocab_result[word] = dict(new_entry)
            unknown_lookup[word] = dict(new_entry)

        self.save_unknown_lookup(unknown_lookup)
        time.sleep(0.5)

    # =======================
    def clean_eg_text(self, eg_list):
        if not isinstance(eg_list, list):
            return eg_list

        for eg in eg_list:
            for key in ["eg_chinese", "eg_pinyin", "eg_vietnamese"]:
                if key in eg and isinstance(eg[key], str):
                    eg[key] = re.sub(r"\*+", "", eg[key])

        return eg_list

    # =======================
    def run_pipeline(self):
        doc_type = input("Enter type (podcast/audiobook/paper): ").strip().lower()
        idx = input("Enter file number: ").strip()

        if not idx.isdigit():
            print("❌ Invalid number")
            return

        idx = int(idx)

        raw_dir = os.path.join(self.base_raw, doc_type)
        raw_path = os.path.join(raw_dir, f"{doc_type}_{idx}.txt")

        if not os.path.exists(raw_path):
            print(f"❌ File not found: {raw_path}")
            return

        with open(raw_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        clean_text = "\n".join(
            [line for line in raw_text.splitlines() if not re.match(r"^\d+:\d+", line)]
        )

        unknown_lookup = self.load_unknown_lookup()

        print("\n📊 VOCAB ANALYSIS (BEFORE API CALL)")
        print("=" * 140)

        vocab_list = self.print_vocab_table(clean_text, unknown_lookup)

        unknown_count = sum(1 for item in vocab_list if item["level"] == "unknown")
        print(f"\n⚠️ Unknown words to process: {unknown_count}")

        print("\n🚀 START PROCESSING")
        print("=" * 140)

        hsk_lookup, level_map = self.load_hsk_lookup()

        vocab_result = {}
        batch_queue = []

        for item in vocab_list:
            word = item["word"]

            if word in hsk_lookup:
                info = dict(hsk_lookup[word])
                info["hsk_level"] = level_map[word]
                info["freq"] = item["count"]
                vocab_result[word] = info
                continue

            if word in unknown_lookup and self.has_valid_eg(unknown_lookup[word]):
                vocab_result[word] = dict(unknown_lookup[word])
                continue

            batch_queue.append(item)

            if len(batch_queue) == BATCH_SIZE:
                self.process_batch(batch_queue, vocab_result, unknown_lookup)
                batch_queue = []

        if batch_queue:
            self.process_batch(batch_queue, vocab_result, unknown_lookup)

        final_output = {}
        for i, (word, info) in enumerate(vocab_result.items(), 1):
            new_info = dict(info)
            new_info["id"] = i
            final_output[word] = new_info

        out_path = os.path.join(self.base_vocab, doc_type, f"{doc_type}_{idx}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)

        print(f"\n✅ Saved to {out_path}")


def test():
    pp = TranscriptPreprocessor()
    pp.run_pipeline()


if __name__ == "__main__":
    test()