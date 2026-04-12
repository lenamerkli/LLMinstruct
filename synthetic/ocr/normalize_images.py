#!/usr/bin/env python3
"""Normalize OCR images to attachments/data/ with SHA256 hash names."""
import hashlib
import os
import pathlib
from pathlib import Path


def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file in lowercase hex."""
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def normalize_ocr_images():
    """Process OCR images, copy to attachments/data/ with hash names, and log renames."""
    images_dir = Path(__file__).parent / 'images'
    data_dir = Path(__file__).parent.parent.parent / 'attachments' / 'data'
    log_file = Path(__file__).parent.parent.parent / 'attachments' / 'rename_log.txt'

    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Process each image in images directory recursively
    for image_path in images_dir.rglob('*'):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.webp']:
            continue

        try:
            # Calculate hash
            file_hash = calculate_sha256(image_path)
            hash_filename = file_hash

            # Create target path
            target_path = data_dir / hash_filename

            # Get relative path from images dir for logging
            rel_path = image_path.relative_to(images_dir)

            # Check if file with same hash already exists
            if target_path.exists():
                # Log duplicate but don't copy
                with open(log_file, 'a') as f:
                    f.write(f'{rel_path} -> {hash_filename}\n')
                print(f"Duplicate found: {rel_path} -> {hash_filename}")
            else:
                # Copy file to data directory
                import shutil
                shutil.copy2(image_path, target_path)
                # Log the rename
                with open(log_file, 'a') as f:
                    f.write(f'{rel_path} -> {hash_filename}\n')
                print(f"Copied: {rel_path} -> {hash_filename}")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")


if __name__ == '__main__':
    normalize_ocr_images()
    print("Normalization complete!")
