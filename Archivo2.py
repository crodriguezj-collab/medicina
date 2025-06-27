# Código para ejecutar en el ambiente local donde estén los .txt de S. aureus, para crear un archivo unificado con la información de todas las cepas.

import os
import pandas as pd

# === CONFIGURATION ===
folder_path = '/path/to/your/txt_folder'         # Actualizar con el path de la carpeta
output_excel = '/path/to/output.xlsx'            # Output Excel file

# === INITIALIZE CONTAINER FOR ALL ROWS ===
all_data = []

# === PROCESS EACH .TXT FILE ===
for file in os.listdir(folder_path):
    if file.endswith('.txt'):
        file_path = os.path.join(folder_path, file)
        code = os.path.splitext(file)[0]  # remove .txt

        # Read the file
        with open(file_path, 'r') as f:
            lines = f.readlines()[1:]  # skip header

        # Extract only bin_intensity (second column)
        bin_values = [float(line.strip().split()[1]) for line in lines]

        # Create row with "code" and bin_0 to bin_5999
        row = {'code': code}
        row.update({f'bin_{i}': bin_values[i] for i in range(6000)})

        all_data.append(row)

# === CREATE FINAL DATAFRAME AND EXPORT ===
df = pd.DataFrame(all_data)
df.to_excel(output_excel, index=False)
