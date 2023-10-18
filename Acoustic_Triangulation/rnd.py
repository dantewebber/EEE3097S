import tkinter as tk
from tkinter import ttk

# Create the main window
root = tk.Tk()
root.title("Modern Python GUI")

# Set the size of the window
window_width = 800
window_height = 600
root.geometry(f"{window_width}x{window_height}")

# Create a modern-looking frame
frame = ttk.Frame(root, padding=(10, 10, 10, 10))
frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Configure the grid to expand the frame with the window
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

# Create a label
label = ttk.Label(frame, text="Welcome to the Modern Python GUI")
label.grid(column=0, row=0, columnspan=2)

# Create an entry widget
entry = ttk.Entry(frame)
entry.grid(column=0, row=1, columnspan=2, pady=10, sticky=(tk.W, tk.E))

# Create a button
button = ttk.Button(frame, text="Click Me")
button.grid(column=0, row=2, columnspan=2)

# Create a modern-looking style
style = ttk.Style()
style.configure("TButton", foreground="white", background="#007acc")
style.configure("TLabel", foreground="black")
style.configure("TFrame", background="#f0f0f0")

# Start the main loop
root.mainloop()