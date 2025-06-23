import os

def find_non_utf8_files(root_dir):
    problem_files = []
    for subdir, _, files in os.walk(root_dir):
        if '.venv' in subdir or '__pycache__' in subdir:
            continue

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(subdir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        f.read()
                except UnicodeDecodeError:
                    problem_files.append(file_path)
    return problem_files

if __name__ == "__main__":
    project_directory = '.'
    bad_files = find_non_utf8_files(project_directory)

    if bad_files:
        print("Files below are not encoded in UTF-8:")
        for f in bad_files:
            print(f"- {f}")
        print("\nPlease convert them to UTF-8 encoding.")
    else:
        print("All files are encoded in UTF-8.")