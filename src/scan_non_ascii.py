import pathlib
import sys
skip = {'venv','__pycache__','build','dist'}
for path in pathlib.Path('.').rglob('*.py'):
    if any(part in skip for part in path.parts):
        continue
    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        text = path.read_text(encoding='gbk', errors='ignore')
    for idx, line in enumerate(text.splitlines(), 1):
        if any(ord(ch) > 127 for ch in line):
            safe = line.encode('unicode_escape', errors='backslashreplace').decode('ascii', errors='backslashreplace')
            out = f"{path}:{idx}:{safe}\n"
            sys.stdout.buffer.write(out.encode('utf-8', errors='replace'))
