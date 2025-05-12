import os

def remove_comments(file_path, comment_symbols):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    with open(file_path, 'w') as file:
        for line in lines:
            stripped_line = line.strip()
            if not any(stripped_line.startswith(symbol) for symbol in comment_symbols):
                file.write(line)

def process_directory(directory, extensions, comment_symbols):
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                remove_comments(file_path, comment_symbols)

directory = r"e:\github repository\physisolver"

extensions = ['.py', '.html', '.css', '.md', '.ignore']
comment_symbols = ['#', '//', '<!--']

process_directory(directory, extensions, comment_symbols)