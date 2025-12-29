import tkinter as tk
from tkinter import ttk, messagebox
import os
import re

class TextEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("BiasBench Text Editor")
        self.root.geometry("1600x900")

        # Get directory path
        self.directory = os.path.dirname(os.path.abspath(__file__))
        self.files = self.get_txt_files()
        self.current_index = 0

        # Create GUI elements
        self.create_widgets()

        # Load first valid file if available
        if self.files:
            for filename in self.files:
                if self.load_file(filename):
                    break

        # Bind keys
        self.root.bind('<Prior>', self.prev_file)  # PageUp
        self.root.bind('<Next>', self.next_file)   # PageDown

    def get_txt_files(self):
        """Get all .txt files in the directory and sort them numerically"""
        files = [f for f in os.listdir(self.directory) if f.endswith('.txt')]
        # Sort numerically by extracting numbers from filenames
        def sort_key(filename):
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else 0
        return sorted(files, key=sort_key)

    def create_widgets(self):
        """Create the GUI widgets"""
        # Top frame for controls
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=5, pady=5)

        # File input label and entry
        ttk.Label(top_frame, text="Jump to file:").pack(side=tk.LEFT)
        self.file_entry = ttk.Entry(top_frame, width=20)
        self.file_entry.pack(side=tk.LEFT, padx=(5, 0))
        self.file_entry.bind('<Return>', self.jump_to_file)

        # Current file label
        self.current_file_label = ttk.Label(top_frame, text="Current: None")
        self.current_file_label.pack(side=tk.RIGHT)

        # Text widget
        self.text_widget = tk.Text(self.root, wrap=tk.WORD, undo=True)
        self.text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        # Status bar
        self.status_label = ttk.Label(self.root, text=f"Files: {len(self.files)} | Position: {self.current_index + 1}")
        self.status_label.pack(fill=tk.X, padx=5, pady=(0, 5))

    def load_file(self, filename):
        """Load a file into the text widget"""
        try:
            filepath = os.path.join(self.directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            self.text_widget.delete(1.0, tk.END)
            self.text_widget.insert(tk.END, content)
            self.current_file = filename
            self.current_file_label.config(text=f"Current: {filename}")
            self.update_status()
            return True
        except Exception as e:
            # Silently fail - return False to indicate failure
            return False

    def save_current_file(self):
        """Save the current file"""
        if hasattr(self, 'current_file'):
            try:
                filepath = os.path.join(self.directory, self.current_file)
                content = self.text_widget.get(1.0, tk.END).rstrip() + '\n'
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file {self.current_file}: {str(e)}")

    def prev_file(self, event=None):
        """Navigate to previous file, skipping non-existent files"""
        if self.files:
            self.save_current_file()
            original_index = self.current_index
            checked_files = 0

            while checked_files < len(self.files):
                self.current_index = (self.current_index - 1) % len(self.files)
                if self.load_file(self.files[self.current_index]):
                    break  # Successfully loaded a file
                checked_files += 1

            # If we couldn't find any valid file, stay at original position
            if checked_files >= len(self.files):
                self.current_index = original_index

    def next_file(self, event=None):
        """Navigate to next file, skipping non-existent files"""
        if self.files:
            self.save_current_file()
            original_index = self.current_index
            checked_files = 0

            while checked_files < len(self.files):
                self.current_index = (self.current_index + 1) % len(self.files)
                if self.load_file(self.files[self.current_index]):
                    break  # Successfully loaded a file
                checked_files += 1

            # If we couldn't find any valid file, stay at original position
            if checked_files >= len(self.files):
                self.current_index = original_index

    def jump_to_file(self, event=None):
        """Jump to a specific file"""
        filename = self.file_entry.get().strip()
        if not filename.endswith('.txt'):
            filename += '.txt'

        if filename in self.files:
            self.save_current_file()
            self.current_index = self.files.index(filename)
            self.load_file(filename)
            self.file_entry.delete(0, tk.END)
        else:
            messagebox.showerror("Error", f"File {filename} not found")

    def update_status(self):
        """Update the status bar"""
        self.status_label.config(text=f"Files: {len(self.files)} | Position: {self.current_index + 1}")

def main():
    root = tk.Tk()
    app = TextEditor(root)
    root.mainloop()

if __name__ == "__main__":
    main()
