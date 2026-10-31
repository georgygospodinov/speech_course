"""Execute the seminar notebook; save outputs locally in .cache/."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['JUPYTER_CONFIG_DIR'] = str(ROOT / '.jupyter')
os.environ['JUPYTER_RUNTIME_DIR'] = str(ROOT / '.jupyter/runtime')
os.environ['IPYTHONDIR'] = str(ROOT / '.ipython')
os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')

import nbformat
from nbclient import NotebookClient

if __name__ == '__main__':
    path = ROOT / 'visualize.ipynb'
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=180, kernel_name='week08',
                            resources={'metadata': {'path': str(ROOT)}})
    client.execute()
    output = ROOT / '.cache/visualize.executed.ipynb'
    output.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, output)
    print('Notebook executed successfully:', output)
