import sys
sys.path.append('/home/lena/Nextcloud/apertus_data')
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from concurrent.futures import ThreadPoolExecutor
import os
from llm import LLaMaCPP

MODEL = 'Apertus-8B-Instruct-2509-Q5_K_M.gguf'


class TokenKeyboardApp:
    # Constant for number of buttons
    NUM_BUTTONS = 8

    # Constant to enable/disable continuations
    ENABLE_CONTINUATIONS = False

    # Constant for default textbox content on startup
    DEFAULT_TEXTBOX_CONTENT = '<|system_start|>You are Apertus, a helpful assistant created by the SwissAI initiative.\nKnowledge cutoff: 2024-04\nCurrent date: 2025-12-14<|system_end|><|developer_start|>Deliberation: disabled\nTool Capabilities: disabled<|developer_end|><|user_start|><|user_end|><|assistant_start|>'

    def __init__(self, root):
        self.root = root
        self.root.title("Token Keyboard")
        self.root.geometry("1000x800")
        
        # Initialize LLaMaCPP
        self.llm = LLaMaCPP()
        self.model_loaded = False
        
        # UI Variables
        self.token_buttons = []
        self.current_tokens = []  # Store original tokens with whitespace
        self.current_continuations = []  # Store continuation texts
        self.is_generating = False
        
        # Create UI
        self.create_widgets()

        # Set default textbox content
        self.textbox.insert("1.0", self.DEFAULT_TEXTBOX_CONTENT)

        # Load model in background
        self.load_model_async()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def on_close(self):
        """Called when window is closed - cleanup model"""
        if self.llm and self.model_loaded:
            print("Stopping model...")
            self.llm.stop()
        self.root.destroy()
    
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Textbox at the top
        self.textbox = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            font=('Arial', 12),
            height=10,
            width=80
        )
        self.textbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        self.textbox.bind('<KeyRelease>', self.on_text_change)

        # Status label
        self.status_label = ttk.Label(main_frame, text="Loading model...", font=('Arial', 10))
        self.status_label.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Save button
        save_button = ttk.Button(
            main_frame,
            text="Save to Moral",
            command=self.save_to_moral
        )
        save_button.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Create button frame for vertical layout
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        button_frame.columnconfigure(0, weight=1)
        
        # Create buttons in a single column
        self.token_buttons = []
        for i in range(self.NUM_BUTTONS):
            # Create frame for button and label
            button_container = ttk.Frame(button_frame)
            button_container.grid(row=i, column=0, sticky=(tk.W, tk.E), padx=2, pady=2)
            button_container.columnconfigure(0, weight=1)
            
            btn = ttk.Button(
                button_container,
                text="",
                command=lambda idx=i: self.on_token_click(idx)
            )
            btn.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
            
            # Probability label
            prob_label = ttk.Label(button_container, text="", font=('Arial', 8))
            prob_label.grid(row=0, column=1, sticky=tk.W)

            # Continuation label
            continuation_label = ttk.Label(button_container, text="", font=('Arial', 8), wraplength=300)
            continuation_label.grid(row=0, column=2, sticky=tk.W)

            self.token_buttons.append((btn, prob_label, continuation_label))
    
    def load_model_async(self):
        """Load the model in a separate thread"""
        def load_model():
            try:
                self.llm.set_model(MODEL)
                self.llm.load_model()
                self.model_loaded = True
                self.root.after(0, self.on_model_loaded)
            except Exception as e:
                self.root.after(0, lambda: self.on_model_error(str(e)))
        
        thread = threading.Thread(target=load_model)
        thread.daemon = True
        thread.start()
    
    def on_model_loaded(self):
        """Called when model is successfully loaded"""
        self.status_label.config(text="Model loaded successfully. Start typing to see token suggestions.")
        self.clear_token_buttons()
    
    def on_model_error(self, error_msg):
        """Called when model loading fails"""
        self.status_label.config(text=f"Error loading model: {error_msg}")
    
    def on_text_change(self, event=None):
        """Called when text in textbox changes"""
        if not self.model_loaded or self.is_generating:
            return
        
        # Debounce text changes
        self.root.after(300, self.generate_token_suggestions)
    
    def generate_token_suggestions(self):
        """Generate token suggestions based on current text"""
        if not self.model_loaded or self.is_generating:
            return
        
        text = self.textbox.get("1.0", "end-1c")  # -1c excludes the extra newline tkinter adds
        if text.strip() == "":
            self.clear_token_buttons()
            return
        
        self.is_generating = True
        self.status_label.config(text="Generating token suggestions...")
        
        def generate():
            try:
                # Get token probabilities
                result = self.llm.generate_probabilities(
                    text,
                    n_predict=1,
                    n_probs=self.NUM_BUTTONS,  # Get top tokens
                    temperature=0.0
                )
                print(result)
                
                # Extract tokens and probabilities
                tokens = []
                if 'steps' in result and result['steps']:
                    step = result['steps'][0]
                    if 'top_probs' in step:
                        for prob_info in step['top_probs'][:self.NUM_BUTTONS]:
                            token = prob_info.get('token', '')
                            prob = prob_info.get('prob', 0)
                            tokens.append((token, prob))

                # Generate continuations for each token concurrently (if enabled)
                continuations = []
                if self.ENABLE_CONTINUATIONS:
                    def generate_continuation(token):
                        try:
                            return self.llm.generate(
                                text + token,
                                n_predict=12,
                                temperature=1.0
                            )
                        except Exception as e:
                            print(f"Error generating continuation for token {repr(token)}: {e}")
                            return ""

                    with ThreadPoolExecutor(max_workers=self.NUM_BUTTONS) as executor:
                        futures = [executor.submit(generate_continuation, token) for token, _ in tokens]
                        continuations = [future.result() for future in futures]
                else:
                    continuations = [""] * len(tokens)

                # Update UI in main thread
                self.root.after(0, lambda: self.update_token_buttons_ui(tokens, continuations))
                
            except Exception as e:
                self.root.after(0, lambda e=e: self.status_label.config(text=f"Error generating suggestions: {str(e)}"))
            finally:
                self.is_generating = False
        
        thread = threading.Thread(target=generate)
        thread.daemon = True
        thread.start()
    
    def update_token_buttons_ui(self, tokens, continuations):
        """Update the token buttons with new tokens, probabilities, and continuations"""
        # Store original tokens and continuations
        self.current_tokens = [token for token, prob in tokens]
        self.current_continuations = continuations

        # Debug: print tokens to verify whitespace is preserved
        print(f"Stored tokens: {self.current_tokens}")

        # Update buttons with tokens - show whitespace more visibly
        for i, ((btn, prob_label, continuation_label), (token, prob), continuation) in enumerate(zip(self.token_buttons, tokens, continuations)):
            # Replace leading/trailing spaces with visible indicator
            display_token = token.replace(' ', '·')  # Middle dot for spaces
            display_token = display_token.replace('\n', '↵')  # Return symbol for newlines
            display_token = display_token.replace('\t', '→')  # Arrow for tabs
            btn.config(text=display_token)
            prob_label.config(text=f"{prob:.3f}")
            # Only display continuation if enabled
            continuation_label.config(text=continuation if self.ENABLE_CONTINUATIONS else "")

        # Clear remaining buttons
        for i, (btn, prob_label, continuation_label) in enumerate(self.token_buttons[len(tokens):]):
            btn.config(text="")
            prob_label.config(text="")
            continuation_label.config(text="")

        self.status_label.config(text="Ready - click a token to insert it")
    
    def find_smallest_available_number(self, directory):
        """Find the smallest available integer for filename in the given directory"""
        existing_numbers = set()

        # Check what files already exist
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.endswith('.txt'):
                    try:
                        num = int(filename[:-4])  # Remove .txt extension
                        existing_numbers.add(num)
                    except ValueError:
                        # Skip files that don't have integer names
                        continue

        # Find the smallest non-negative integer not in existing_numbers
        num = 0
        while num in existing_numbers:
            num += 1

        return num
    
    def save_to_moral(self):
        """Save the current text content to a new file in ./moral directory"""
        try:
            # Get the text content
            text_content = self.textbox.get("1.0", "end-1c")  # -1c excludes the extra newline

            if not text_content.strip():
                messagebox.showwarning("Empty Content", "Cannot save empty content.")
                return

            # Get the absolute path to the folder containing writer.py
            writer_dir = os.path.dirname(os.path.abspath(__file__))

            # Determine the moral directory path
            moral_dir = os.path.join(writer_dir, 'moral')

            # Create directory if it doesn't exist
            os.makedirs(moral_dir, exist_ok=True)

            # Find the smallest available number
            file_number = self.find_smallest_available_number(moral_dir)
            filename = f"{file_number}.txt"
            filepath = os.path.join(moral_dir, filename)

            # Write the content to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text_content)

            # Show success message
            messagebox.showinfo("Save Successful", f"Content saved to {filename}")
            self.status_label.config(text=f"Saved to {filename}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save: {str(e)}")
            self.status_label.config(text=f"Error saving: {str(e)}")
    
    def clear_token_buttons(self):
        """Clear all token buttons"""
        self.current_tokens = []
        self.current_continuations = []
        for btn, prob_label, continuation_label in self.token_buttons:
            btn.config(text="")
            prob_label.config(text="")
            continuation_label.config(text="")
    
    def on_token_click(self, button_index):
        """Called when a token button is clicked"""
        if button_index < len(self.current_tokens):
            token = self.current_tokens[button_index]
            print(f"Inserting token: {repr(token)}")  # Debug
            if token:
                # Insert token at cursor position (using original token with whitespace)
                self.textbox.insert(tk.INSERT, token)
                self.textbox.see(tk.END)  # Scroll to end
                self.textbox.focus_set()   # Focus back to textbox
                
                # Generate new suggestions after inserting token
                self.generate_token_suggestions()

def main():
    root = tk.Tk()
    app = TokenKeyboardApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
