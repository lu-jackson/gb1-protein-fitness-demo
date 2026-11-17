import re, nbformat
from nbclient import NotebookClient
src = open("src/pipeline.py", encoding="utf-8").read()
nb = nbformat.v4.new_notebook()
for chunk in re.split(r"^# %%\s*", src, flags=re.M):
    if not chunk.strip():
        continue
    if chunk.startswith("[markdown]"):
        body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in chunk.split("\n")[1:]).strip()
        nb.cells.append(nbformat.v4.new_markdown_cell(body))
    else:
        nb.cells.append(nbformat.v4.new_code_cell(chunk.strip()))
NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
nbformat.write(nb, "gb1_fitness_prediction.ipynb")
print("cells:", len(nb.cells))
