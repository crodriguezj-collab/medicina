# Código para ejecutar en el ambiente local donde están los archivos DRIAMS para dejar únicamente los de S. aureus y eliminar las demás bacterias

import os
import pandas as pd

# Rutas
folder_path = 'path/to/txt_folder'          # Folder con todos los .txt
excel_path = 'path/to/excel_file.xlsx'      # Excel con S. aureus el nombre de los archivos de S. aureus

# Cargar los filenames desde Excel
df = pd.read_excel(excel_path)
valid_names = set(df.iloc[:, 0].astype(str) + '.txt')  

# Borrar los archivos que no están en la lista
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if file_name.endswith('.txt') and file_name not in valid_names:
        os.remove(file_path)
