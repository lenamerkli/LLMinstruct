#!/usr/bin/env python3
import os
import hashlib
from pathlib import Path

def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file in lowercase hex."""
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read file in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def normalize_files():
    """Process files in input folder, move to data folder with hash names, and log renames."""
    input_dir = Path('input')
    data_dir = Path('data')
    log_file = Path('rename_log.txt')

    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Clear previous log if it exists
    with open(log_file, 'w') as f:
        f.write('')

    # Process each file in input directory
    for file_path in input_dir.iterdir():
        if file_path.is_file():
            try:
                # Calculate hash
                file_hash = calculate_sha256(file_path)
                hash_filename = file_hash  # Use just the hash as filename

                # Create target path
                target_path = data_dir / hash_filename

                # Check if file with same hash already exists
                if target_path.exists():
                    # Log duplicate but don't move
                    with open(log_file, 'a') as f:
                        f.write(f'{file_path.name} -> {hash_filename}\n')
                    print(f"Duplicate found: {file_path.name} -> {hash_filename}")
                    # Remove original file from input
                    file_path.unlink()
                else:
                    # Move file to data directory
                    file_path.rename(target_path)
                    # Log the rename
                    with open(log_file, 'a') as f:
                        f.write(f'{file_path.name} -> {hash_filename}\n')
                    print(f"Moved: {file_path.name} -> {hash_filename}")

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

if __name__ == '__main__':
    normalize_files()
    print("Normalization complete!")
