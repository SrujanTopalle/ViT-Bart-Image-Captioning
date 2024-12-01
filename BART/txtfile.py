import os

def save_filenames_to_txt(folder_path, output_file):
    # Get a list of all files in the folder
    filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Write filenames to the output file
    with open(output_file, "w") as f:
        for filename in filenames:
            f.write(filename + "\n")

    print(f"File names saved to {output_file}")

# Example usage
folder_path = r"C:\\Users\\Srujan Topalle\\Desktop\\Deep learning project\\coco2017\\train"  # Replace with the path to your folder
output_file = "image_names.txt"   # Output text file
save_filenames_to_txt(folder_path, output_file)
