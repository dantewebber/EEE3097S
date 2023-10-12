import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import Acoustic_Triangulation



# Function to start the process
def start_process():
    global positionX, positionY
    
    Acoustic_Triangulation.read_wav_files()

    # Set the new position (0.3, 0.4)
    # positionX = 0.6
    # positionY = 0.4

    # Update the graph with a red dot
    # update_graph()
    
# Function to stop the process
def stop_process():
    # Add your stop process logic here
    pass

# Function to update the graph with a red dot
def update_graph(positionX, positionY):
    plot.clear()
    plot.set_xlabel("X-axis")
    plot.set_ylabel("Y-axis")
    plot.set_xlim(0, 0.8)
    plot.set_ylim(0, 0.5)
    plot.set_title("Acoustic Triangulation")

    # Add grid lines spaced 0.1 meters apart
    plot.grid(which='both', axis='both', linestyle='--', color='gray', linewidth=0.5, alpha=0.5)
    plot.set_xticks([i / 10 for i in range(9)])
    plot.set_yticks([i / 10 for i in range(6)])  # Corrected line

    # Plot the red dot at the specified position
    plot.plot(positionX, positionY, marker='o', markersize=4, color='red')

    # Embed the Matplotlib figure in the Tkinter window
    canvas.draw()

# Initialize positionX and positionY
positionX = 0
positionY = 0

# Create a Tkinter window
root = tk.Tk()
root.configure(bg="white")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}+0+0")

# Create "Start" button
start_button = tk.Button(root, text="Start", command=start_process)
start_button.place(relx=0.1, rely=0.2, relwidth=0.2, relheight=0.1)

# Create "Stop" button
stop_button = tk.Button(root, text="Stop", command=stop_process)
stop_button.place(relx=0.1, rely=0.35, relwidth=0.2, relheight=0.1)

# Create a frame for the XY graph
graph_frame = ttk.Frame(root)
graph_frame.place(relx=0.6, rely=0, relwidth=0.4, relheight=1)

# Create a Matplotlib figure
figure = Figure(figsize=(5, 4), dpi=100)
plot = figure.add_subplot(111)

# Set labels and title for the blank graph
plot.set_xlabel("X-axis")
plot.set_ylabel("Y-axis")
plot.set_xlim(0, 0.8)
plot.set_ylim(0, 0.5)
plot.set_title("Acoustic Triangulation")

# Add grid lines spaced 0.1 meters apart
plot.grid(which='both', axis='both', linestyle='--', color='gray', linewidth=0.5, alpha=0.5)
plot.set_xticks([i / 10 for i in range(9)])
plot.set_yticks([i / 10 for i in range(6)])  # Corrected line

# Embed the Matplotlib figure in the Tkinter window
canvas = FigureCanvasTkAgg(figure, master=graph_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

# Start the Tkinter event loop
root.mainloop()
