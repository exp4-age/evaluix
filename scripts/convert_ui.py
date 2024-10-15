#%%
import pathlib
from PyQt6 import uic

# Define paths
# This is a different root than the one in the original script
root = pathlib.Path(__file__).resolve().parents[1]
ui_dir = root / 'src/evaluix/GUIs'
output_dir = ui_dir  # Output directory is the same as the input directory

# Function to convert .ui to .py
def convert_ui_to_py(ui_file, py_file):
    with open(py_file, 'w') as py_fp:
        uic.compileUi(ui_file, py_fp)

# Function to modify import statements
def modify_import_statements(py_file):
    with open(py_file, 'r') as file:
        lines = file.readlines()

    with open(py_file, 'w') as file:
        inside_import_block = False
        import_lines = []

        for line in lines:
            if line.startswith('from CustomWidgets import'):
                inside_import_block = True
                import_lines.append(line)
            elif inside_import_block and line.startswith(' '):
                import_lines.append(line)
            else:
                if inside_import_block:
                    file.write('try:\n')
                    for import_line in import_lines:
                        file.write(f'    {import_line}')
                    file.write('except ImportError:\n')
                    for import_line in import_lines:
                        file.write(f'    {import_line.replace("from CustomWidgets import", "from .CustomWidgets import")}')
                    inside_import_block = False
                    import_lines = []
                file.write(line)

        if inside_import_block:
            file.write('try:\n')
            for import_line in import_lines:
                file.write(f'    {import_line}')
            file.write('except ImportError:\n')
            for import_line in import_lines:
                file.write(f'    {import_line.replace("from CustomWidgets import", "from .CustomWidgets import")}')

# Traverse the ui_dir and convert .ui files to .py files
for ui_file in ui_dir.glob('*.ui'):
    py_file = output_dir / f"{ui_file.stem}.py"
    print(f"Converting {ui_file} to {py_file}")
    convert_ui_to_py(ui_file, py_file)
    modify_import_statements(py_file)

print("Conversion and modification completed.")