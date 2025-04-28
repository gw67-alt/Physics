import tkinter as tk
from tkinter import ttk
from tkinter import messagebox  # Import messagebox for error popups
import random
import math # Import math for calculating max_attempts

def number_guessing_game():
    """
    Creates a Tkinter window for a number guessing game with a variable range,
    styled with a dark theme. The guess display updates in real-time with the slider.
    """
    # Initialize Tkinter window
    root = tk.Tk()
    root.title("Guess the Number")
    root.geometry("500x450")  # Adjusted size for styling
    root.resizable(False, False) # Prevent resizing
    root.configure(bg="#2E2E2E") # Dark background for the main window

    # --- Style Configuration ---
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except tk.TclError:
        print("Clam theme not available, using default.")

    style.configure('.', background='#2E2E2E', foreground='white') # Global style
    style.configure('TFrame', background='#2E2E2E')
    style.configure('TLabel', background='#2E2E2E', foreground='white', font=('Arial', 10))
    style.configure('TButton', background='#4A4A4A', foreground='white', font=('Arial', 11, 'bold'), borderwidth=0, padding=5) # Added padding to button
    style.map('TButton',
              background=[('active', '#6A6A6A'), ('disabled', '#3A3A3A')],
              foreground=[('disabled', '#777777')])
    style.configure('TEntry', fieldbackground='#4A4A4A', foreground='white', insertcolor='white', borderwidth=1)
    style.configure('Horizontal.TScale', background='#2E2E2E', troughcolor='#4A4A4A')
    style.map('Horizontal.TScale',
              background=[('active', '#2E2E2E')],
              troughcolor=[('active', '#5A5A5A')])
    # Style for the central display frame and label
    style.configure('Display.TFrame', background='#1A1A1A')
    style.configure('Display.TLabel', background='#1A1A1A', foreground="#FFEB3B", font=('Consolas', 36, 'bold'))


    # --- Game State Variables ---
    game_state = {
        "target_number": 1,
        "attempts": 0,
        "min_value": 1,
        "max_value": 100,
        "max_attempts": 0
    }

    # --- Game Logic ---

    # Function to update the display label based on slider value
    def update_guess_display(value):
        """Updates the central display label with the slider's current value."""
        # Slider passes value as a string, convert->float->round->int->string
        try:
            display_value = int(round(float(value)))
            current_guess_display.config(text=f"{display_value}")
        except ValueError:
             current_guess_display.config(text="--") # Handle potential conversion errors


    def check_guess():
        """ Check the player's guess against the target number. """
        target_number = game_state["target_number"]
        max_attempts = game_state["max_attempts"]

        try:
            # Get integer value from the slider
            guess = int(round(slider.get()))

            # --- Display update is now handled by the slider's command ---
            # current_guess_display.config(text=f"{guess}") # This line is removed

            game_state["attempts"] += 1
            attempts = game_state["attempts"]
            attempts_label.config(text=f"Attempts: {attempts}/{max_attempts}")

            if guess == target_number:
                result_label.config(text=f"CORRECT! You guessed {target_number} in {attempts} tries!", foreground="#4CAF50") # Green
                check_button.config(state=tk.DISABLED)
                slider.config(state=tk.DISABLED)
            elif guess < target_number:
                result_label.config(text=game_state["target_number"], foreground="#FF9800") # Orange
                game_state["target_number"] = random.randint(game_state["min_value"], game_state["max_value"])
            if attempts >= max_attempts and guess != target_number:
                result_label.config(text=f"No more attempts! The number was {target_number}.", foreground="#F44336") # Red
                check_button.config(state=tk.DISABLED)
                slider.config(state=tk.DISABLED)

        except ValueError:
            result_label.config(text="Invalid input.", foreground="#F44336") # Red

    def setup_game():
        """ Sets up/Resets the game state and range based on entry fields. """
        try:
            temp_min = int(min_entry.get())
            temp_max = int(max_entry.get())

            if temp_min >= temp_max:
                messagebox.showerror("Invalid Range", f"Min ({temp_min}) must be less than Max ({temp_max}). Using previous range.")
                min_entry.delete(0, tk.END)
                min_entry.insert(0, str(game_state["min_value"]))
                max_entry.delete(0, tk.END)
                max_entry.insert(0, str(game_state["max_value"]))
                return
            else:
                game_state["min_value"] = temp_min
                game_state["max_value"] = temp_max

            range_size = game_state["max_value"] - game_state["min_value"] + 1
            if range_size <= 1:
                 game_state["max_attempts"] = 1
            else:
                 game_state["max_attempts"] = 100

            slider.config(from_=game_state["min_value"], to=game_state["max_value"])
            slider.set(game_state["min_value"]) # Set slider position

            game_state["target_number"] = random.randint(game_state["min_value"], game_state["max_value"])
            # print(f"New target: {game_state['target_number']}") # Debugging

            game_state["attempts"] = 0
            result_label.config(text="", foreground="white")
            attempts_label.config(text=f"Attempts: {game_state['attempts']}/{game_state['max_attempts']}")
            slider.config(state=tk.NORMAL)
            check_button.config(state=tk.NORMAL)
            instruction_label.config(text=f"Guess between {game_state['min_value']} and {game_state['max_value']}")
            max_attempts_info_label.config(text=f"({game_state['max_attempts']} attempts)")
            # Reset the guess display using the update function
            update_guess_display(game_state["min_value"])


        except ValueError:
            messagebox.showerror("Invalid Input", "Min and Max must be integers. Using previous range.")
            min_entry.delete(0, tk.END)
            min_entry.insert(0, str(game_state["min_value"]))
            max_entry.delete(0, tk.END)
            max_entry.insert(0, str(game_state["max_value"]))

    # --- UI Elements ---

    # Frame for range selection (Top)
    range_frame = ttk.Frame(root, padding="10 5 10 5")
    range_frame.pack(pady=10, fill='x')

    min_label = ttk.Label(range_frame, text="Min:", font=('Arial', 10))
    min_label.pack(side=tk.LEFT, padx=(10, 2))
    min_entry = ttk.Entry(range_frame, width=8, font=('Arial', 10), justify='center')
    min_entry.pack(side=tk.LEFT, padx=(0, 10))
    min_entry.insert(0, str(game_state["min_value"]))

    max_label = ttk.Label(range_frame, text="Max:", font=('Arial', 10))
    max_label.pack(side=tk.LEFT, padx=(10, 2))
    max_entry = ttk.Entry(range_frame, width=8, font=('Arial', 10), justify='center')
    max_entry.pack(side=tk.LEFT, padx=(0, 10))
    max_entry.insert(0, str(game_state["max_value"]))

    reset_button = ttk.Button(range_frame, text="Set Range / Reset", command=setup_game, width=18)
    reset_button.pack(side=tk.RIGHT, padx=(10, 10))


    # Instruction Labels
    instruction_label = ttk.Label(root, text="Set range and click 'Set Range / Reset'", font=('Arial', 11))
    instruction_label.pack(pady=(5,0))

    max_attempts_info_label = ttk.Label(root, text="", font=('Arial', 9), foreground="#AAAAAA")
    max_attempts_info_label.pack(pady=(0,10))

    # --- Central Display Area ---
    display_frame = ttk.Frame(root, padding="10", relief="sunken", borderwidth=2, style='Display.TFrame')
    display_frame.pack(pady=10, padx=20, fill='x')

    # Label to show the guess *in real-time*
    current_guess_display = ttk.Label(display_frame, text="0", style='Display.TLabel', anchor='center')
    current_guess_display.pack(pady=10, fill='x')


    # --- Slider ---
    # ADDED command=update_guess_display back to the slider
    slider = ttk.Scale(root, from_=game_state["min_value"], to=game_state["max_value"],
                       orient=tk.HORIZONTAL, length=400,
                       command=update_guess_display) # Link slider movement to display update
    slider.pack(pady=15, padx=30)
    slider.config(state=tk.DISABLED)

    # --- Check Button ---
    check_button = ttk.Button(root, text="Check Guess", command=check_guess, width=15)
    check_button.pack(pady=10)
    check_button.config(state=tk.DISABLED)

    # --- Attempts and Result ---
    attempts_label = ttk.Label(root, text="Attempts: 0/0", font=('Arial', 10), foreground="#CCCCCC")
    attempts_label.pack(pady=(5, 0))

    result_label = ttk.Label(root, text="", font=('Arial', 14, 'bold'), anchor='center')
    result_label.pack(pady=10, fill='x')


    # --- Initial Game Setup ---
    setup_game() # Setup with default values

    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    number_guessing_game()
