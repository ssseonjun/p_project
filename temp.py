from pathlib import Path
import json

JSON_DIR = Path("file/json")

sample = JSON_DIR / "WST_business.json"   # 다른 애 아무나 하나
with open(sample, "r", encoding="utf-8") as f:
    data = json.load(f)

print(data.keys())
print(len(data.get("business_by_year", "")))
print(data.get("business_by_year", ""))
