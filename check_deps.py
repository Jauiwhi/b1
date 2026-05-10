import importlib.util
import sys

deps = ['yaml', 'requests', 'openai', 'anthropic', 'pypdf', 'docx', 'openpyxl', 'PIL', 'pytesseract', 'tiktoken']
missing = []

for dep in deps:
    try:
        if importlib.util.find_spec(dep) is None:
            missing.append(dep)
    except:
        missing.append(dep)

if missing:
    print('[WARNING] Missing dependencies:', ', '.join(missing))
    sys.exit(1)

print('[OK] Core dependencies installed')
print('[INFO] PaddleOCR/EasyOCR not required (optional)')
sys.exit(0)
