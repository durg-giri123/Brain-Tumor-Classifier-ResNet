import json

with open("Sample_ML_Submission_Template-2.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    content = "".join(cell.get("source", []))
    if len(content) > 100:
        content = content[:100] + "..."
    print(f"Cell {i} [{cell['cell_type']}]: {content}")
