import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.scrolledtext import ScrolledText

# Function to update the status box
def update_status(status_text):
    global status_box
    status_box.delete(1.0, "end")  # Clear existing content
    status_box.insert("end", status_text)  # Insert new status text

# Function to update the graph with a red dot
def update_graph(positionX, positionY):
    global plot
    plot.clear()
    plot.set_xlabel("X-axis")
    plot.set_ylabel("Y-axis")
    plot.set_xlim(0, 0.8)
    plot.set_ylim(0, 0.5)
    plot.set_title("Acoustic Triangulation")

    # Add grid lines spaced 0.1 meters apart
    plot.grid(which='both', axis='both', linestyle='--', color='gray', linewidth=0.5, alpha=0.5)
    plot.set_xticks([i / 10 for i in range(9)])
    plot.set_yticks([i / 10 for i in range(6)])

    # Plot the red dot at the specified position
    plot.plot(positionX, positionY, marker='o', markersize=4, color='red')

    # Embed the Matplotlib figure in the Tkinter window
    global canvas
    canvas.draw()

# Function definitions (start_process and stop_process)
def start_process():
    global positionX, positionY

    # Update the status text
    update_status("Started.")

    # Call the update_graph function to update the graph
    update_graph(positionX, positionY)

def stop_process():
    # Add your stop process logic here
    exit()

# GUI INITIALIZATION
def create_app_window():
    app = tk.Tk()
    app.title("Acoustic Triangulation")
    app.geometry("800x600")

    # Set the window background to dark
    app.configure(bg="white")

    # Create a style object for ttk widgets
    style = ttk.Style()

    # Choose a dark theme for your application
    style.theme_use("clam")  # Use the "clam" theme as an example

    # Create "Start" button with a custom style
    style.configure("Start.TButton", font=("Helvetica", 12, "bold"), foreground="white", background="#2574F4")
    start_button = ttk.Button(app, text="Start", command=start_process, style="Start.TButton")
    start_button.place(relx=0.1, rely=0.2, relwidth=0.2, relheight=0.1)

    # Create "Stop" button with a custom style
    style.configure("Stop.TButton", font=("Helvetica", 12, "bold"), foreground="white", background="red")
    stop_button = ttk.Button(app, text="Stop", command=stop_process, style="Stop.TButton")
    stop_button.place(relx=0.1, rely=0.35, relwidth=0.2, relheight=0.1)

    # Create a frame for the XY graph
    graph_frame = ttk.Frame(app)
    graph_frame.place(relx=0.6, rely=0, relwidth=0.4, relheight=1)

    # Create a Matplotlib figure
    global plot
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
    plot.set_yticks([i / 10 for i in range(6)])

    # Embed the Matplotlib figure in the Tkinter window
    global canvas
    canvas = FigureCanvasTkAgg(figure, master=graph_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)

    # Create a text status box using scrolled text
    global status_box
    status_box = ScrolledText(app, wrap=tk.WORD, height=10, width=40)
    status_box.place(relx=0.1, rely=0.5, relwidth=0.4, relheight=0.4)

    return app

if __name__ == "__main__":
    app = create_app_window()
    app.mainloop()
