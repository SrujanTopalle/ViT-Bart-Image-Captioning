import os

def save_filenames_to_txt(folder_path, output_file):

    filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    with open(output_file, "w") as f:
        for filename in filenames:
            f.write(filename + "\n")

    print(f"File names saved to {output_file}")

folder_path = "path/to/folder"  
output_file = "image_names.txt"
save_filenames_to_txt(folder_path, output_file)
