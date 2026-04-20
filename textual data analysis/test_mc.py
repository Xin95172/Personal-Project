import os, json, re
from config.patterns import MAIN_PATTERNS
from utils.verdict_utils import extract_main_clause

folder = 'data/IP_Law_cases'
files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.json')]
for f in files[:2000]:
    with open(f, 'r') as file:
        data = json.load(file)
    jfull = data.get('JFULL', '')
    start = re.search(MAIN_PATTERNS["START_PATTERNS"], jfull)
    if not start: continue
    start_index = start.end()
    end = re.search(MAIN_PATTERNS["END_PATTERNS"], jfull[start_index:])
    if not end:
        print(f"File: {f} - END_PATTERN NOT FOUND. Length of rest: {len(jfull[start_index:])}")
        print(f"Preview: {jfull[start_index:start_index+200]}...")
        break
