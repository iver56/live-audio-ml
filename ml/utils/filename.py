import os
from pathlib import Path


def get_file_paths(image_root_path, filename_prefix=None, filename_ending='.wav'):
    """Return a list of paths to all files with the given in a directory

    Does not check subdirectories.
    """
    image_file_paths = []

    for root, dirs, filenames in os.walk(image_root_path):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)

            if filename.endswith(filename_ending):
                if filename_prefix is not None and not filename.startswith(filename_prefix):
                    continue
                image_file_paths.append(Path(file_path))

        break  # prevent descending into subfolders

    return image_file_paths
