import os

IGNORE_DIRS = {"__pycache__"}

def generate_tree(root_dir, prefix=""):
    """
    Recursively generates a directory tree structure,
    ignoring specified directories.
    """
    try:
        entries = sorted(os.listdir(root_dir))
    except PermissionError:
        return

    # filter ignored directories
    entries = [
        e for e in entries
        if not (e in IGNORE_DIRS and os.path.isdir(os.path.join(root_dir, e)))
    ]

    entries_count = len(entries)

    for index, entry in enumerate(entries):
        path = os.path.join(root_dir, entry)
        connector = "└── " if index == entries_count - 1 else "├── "

        print(prefix + connector + entry)

        if os.path.isdir(path):
            extension = "    " if index == entries_count - 1 else "│   "
            generate_tree(path, prefix + extension)


if __name__ == "__main__":
    ROOT_DIRECTORY = r"E:\Audio-and-Lyrics-Multi-Modal-Clustering-VAE\results"
    print(ROOT_DIRECTORY)
    generate_tree(ROOT_DIRECTORY)
