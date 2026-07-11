import gzip
import shutil
import os
import re

# Folder where your .gz files are
input_folder = r"..\storm_events\downloads"   # ← change this path
output_folder = r"..\storm_events\data"  # ← change this path

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".gz"):
        gz_path = os.path.join(input_folder, filename)

        # Clean the output filename: remove leading numbers, add .csv
        clean_name = re.sub(r"^\d+_", "", filename)  # remove prefix like "123_"
        clean_name = clean_name.replace(".gz", "")

        out_path = os.path.join(output_folder, clean_name)

        with gzip.open(gz_path, "rb") as f_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        print(f"Extracted: {clean_name}")

print("\nDone! All files saved to:", output_folder)