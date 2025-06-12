import os
import ast
import platform
import pkg_resources

def find_imports_in_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        try:
            tree = ast.parse(f.read(), filename=path)
        except SyntaxError:
            return set()
    
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split('.')[0])
    return names

def find_imports_in_directory(directory):
    all_libs = set()
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py"):
                full_path = os.path.join(root, filename)
                all_libs.update(find_imports_in_file(full_path))
    return sorted(all_libs)

def filter_installed_libraries(libraries):
    libs_with_versions = {}
    for name in libraries:
        try:
            dist = pkg_resources.get_distribution(name)
            libs_with_versions[dist.project_name] = dist.version
        except pkg_resources.DistributionNotFound:
            pass
    return libs_with_versions

def save_human_readable_output(libs_with_versions, output_file="requirements.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        python_version = platform.python_version()
        f.write(f"Python {python_version}\n\n")
        for name, version in sorted(libs_with_versions.items()):
            f.write(f"{name:<30} {version}\n")
    print(f"[OK] Saved to '{output_file}'.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Scanning: {base_dir}")

    used_libs = find_imports_in_directory(base_dir)
    libs_with_versions = filter_installed_libraries(used_libs)
    save_human_readable_output(libs_with_versions)
