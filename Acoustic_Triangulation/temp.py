import os
import pygame
import subprocess
import concurrent.futures
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import chirp
from scipy.optimize import fsolve
from scipy.signal import butter, lfilter
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Text
from tkinter.scrolledtext import ScrolledText
import threading
import paramiko

# Program Constants

# Constants
grid_width = 0.783 # 0.841 - A1 # Width of the grid [m]
grid_height = 0.49 # 0.594 - A1 # Height of the grid [m]
speed_of_sound = 343  # Speed of sound in the environment (m/s)
raspberry_pis = ["pi@raspberrypi.local", "pi@raspberrypi1.local"]
fs = 48000 # sample rate (Hz)
stop_triangulation = False

# Define positions of microphones
mic1_pos = [0, grid_height]
mic2_pos = [grid_width, grid_height]
mic3_pos = [grid_width, grid_height]
mic4_pos = [grid_width, 0]

# File paths
current_directory = os.getcwd()
audio_folder = os.path.join(current_directory, "audio")

# Target file and target location on pi
target_file = "start.txt"
destination = "/home/pi/"

# Controlling Pi's recording and sending audio files
def initializeSSH():
    
    global pi1_hostname
    global pi2_hostname
    global private_key_path
    global start_command1
    global start_command2
    global end_command1
    global end
    global mykey
    global pi1_ssh
    global pi2_ssh

    pi1_hostname = 'raspberrypi.local'
    pi2_hostname = 'raspberrypi1.local'
    private_key_path = '/home/dantewebber/.ssh/id_rsa'

    # Define the command to start recording on the Raspberry Pi
    start_command1 = 'mkdir audio && arecord -D plughw:0 -c2 -r 48000 -f S32_LE -t wav -V stereo -d 1 /home/pi/audio/raspberrypi_07:19:01.294061.wav'
    start_command2 = "mkdir audio && arecord -D plughw:0 -c2 -r 48000 -f S32_LE -t wav -V stereo -d 1 /home/pi/audio/raspberrypi1_07:19:01.292324.wav"
    end_command1 = "/home/pi/send.sh"
    end = "rm -r /home/pi/audio"

    mykey = paramiko.RSAKey(filename=private_key_path)

    # Create SSH clients for both Raspberry Pi devices
    pi1_ssh = paramiko.SSHClient()
    pi1_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pi1_ssh.connect(pi1_hostname, username='pi', pkey=mykey)

    pi2_ssh = paramiko.SSHClient()
    pi2_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pi2_ssh.connect(pi2_hostname, username="pi", pkey=mykey)

def record_send():
    
    update_status("Recording audio\n")
    
    # Send the start command to both Raspberry Pi devices
    pi1_stdin, pi1_stdout, pi1_stderr = pi1_ssh.exec_command(start_command1)
    pi2_stdin, pi2_stdout, pi2_stderr = pi2_ssh.exec_command(start_command2)

    pi1_exit_status = pi1_stdout.channel.recv_exit_status()
    pi2_exit_status = pi2_stdout.channel.recv_exit_status()

    while True:
        pi1_exit_status = pi1_stdout.channel.recv_exit_status()
        pi2_exit_status = pi2_stdout.channel.recv_exit_status()
        if pi1_exit_status == 0 and pi2_exit_status == 0:
            break
    
    update_status("Finished recording, sending files.\n")
    
    # Send files to master laptop
    pi1_stdin, pi1_stdout, pi1_stderr = pi1_ssh.exec_command(end_command1)
    pi2_stdin, pi2_stdout, pi2_stderr = pi2_ssh.exec_command(end_command1)

    while True:
        pi1_exit_status = pi1_stdout.channel.recv_exit_status()
        pi2_exit_status = pi2_stdout.channel.recv_exit_status()
        if pi1_exit_status == 0 and pi2_exit_status == 0:
            break
    
    update_status("Files sent\n")
    
    pi1_ssh.exec_command(end)
    pi2_ssh.exec_command(end)

def closeSSH():
    # Close the SSH connections
    pi1_ssh.close()
    pi2_ssh.close()

# GUI

def update_status(status_text):
    global status_box
    # status_box.delete(1.0, "end")  # Clear existing content
    status_box.insert("end", status_text)  # Insert new status text

# Function to start the process
def start_process():
    global stop_triangulation
    stop_triangulation = False
    
    # Update the status text
    update_status("Started.\n")

    start()

    # Update the graph with a red dot
    # update_graph(positionX, positionY)
    
# Function to stop the process
def stop_process():
    global stop_triangulation
    stop_triangulation = True

def exit_program():
    exit()

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
    plot.set_yticks([i / 10 for i in range(6)])  # Corrected line

    # Plot the red dot at the specified position
    plot.plot(positionX, positionY, marker='o', markersize=4, color='red')

    # Embed the Matplotlib figure in the Tkinter window
    canvas.draw()

# Start recording and get data from raspberry pis

def start():
    
    # single_triangulation()
    initializeSSH()
    
    task_thread = threading.Thread(target=continuous_triangulation)
    task_thread.start()
    
    # Plot all signals
    # signal_plot(signals[0], signals[1], signals[2], signals[3], times[0], times[1], times[2], times[3])
    
def single_triangulation():
    # Send file to both Raspberry Pis
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     executor.map(send_file_to_raspberry_pi, raspberry_pis)
    
    record_send()
    
    # Extract signals from audio files
    signals, times, first_signal, time_offset = wait_for_audio_files()
    
    # Process signals
    signals, times = process_signals2(signals, times)
    
    # Calculate the TDOAs
    tdoa1, tdoa2 = Find_TDOA_2(signals[0], signals[1], signals[2], signals[3])
    
    # Tringulate position
    source_position = Triangulation2(tdoa1, tdoa2)
    
    # Remove audio files
    command = f"rm {audio_folder}/*"
    subprocess.call(command, shell=True)
    
    # Plot triangulated position on grid
    update_graph(source_position[0], source_position[1])

def continuous_triangulation():
    while not stop_triangulation:
        single_triangulation()
    closeSSH()

def run_sim():
    # To run simulation with a live audio file uncomment the lines below:
    
    # Send file to both Raspberry Pis
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     executor.map(send_file_to_raspberry_pi, raspberry_pis)
    
    # update_status("Successfully sent start cmd to pis\n")
    # print("Successfully sent start cmd to pis")
    
    # Extract signals from audio files
    # signals, times, first_signal, time_offset = wait_for_audio_files()
    
    # SIMULATION SETUP: first parameter is a signal, 2nd is simulated position
    # Uncomment lines below and comment all lines under it in start() function to run simulation
    
    sim_pos = [grid_width/2, grid_height/2]
    sim_pos = [0.3, 0.2]
    signals, times, first_signal, time_offset = read_wav_files()
    create_sim2(signals[0], sim_pos)
    
    

# GUI INITIALIZATION
def create_app_window():
    app = tk.Tk()
    app.title("Acoustic Triangulation")
    app.geometry("800x600")
    
    app.configure(bg="white")

    # Create a style object for ttk widgets
    style = ttk.Style()

    # Choose a theme for your application (e.g., 'clam', 'winnative', 'alt', 'default', etc.)
    style.theme_use("default")

    # Create "Start" button with a custom style
    style.configure("Start.TButton", font=("Helvetica", 12, "bold"), foreground="white", background="blue")
    start_button = ttk.Button(app, text="Start", command=start_process, style="Start.TButton")
    start_button.place(relx=0.1, rely=0.1, relwidth=0.2, relheight=0.1)

    # Create "Stop" button with a custom style
    style.configure("Stop.TButton", font=("Helvetica", 12, "bold"), foreground="white", background="red")
    stop_button = ttk.Button(app, text="Stop", command=stop_process, style="Stop.TButton")
    stop_button.place(relx=0.1, rely=0.25, relwidth=0.2, relheight=0.1)
    
    # Create an "Exit" button
    style.configure("Exit.TButton", font=("Helvetica", 12, "bold"), foreground="white", background="gray")
    exit_button = ttk.Button(app, text="Exit", command=exit_program, style="Exit.TButton")
    exit_button.place(relx=0.1, rely=0.85, relwidth=0.1, relheight=0.05)

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
    status_box.place(relx=0.1, rely=0.4, relwidth=0.4, relheight=0.4)

    return app

        
def send_file_to_raspberry_pi(pi_host):
    command = f"scp {target_file} {pi_host}:{destination}"

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result, error = process.communicate()

    if process.returncode != 0:
        print(f'Failed to send file to {pi_host}. Error: {error.decode()}')
        
    else:
        print(f'Successfully sent file to {pi_host}.')
        

# Importing .wav recording files

def wait_for_audio_files():
    
    update_status("Waiting for audio signals froom raspberry Pis.\n")
    print("Waiting for audio signals froom raspberry Pis.")
    
    # time.sleep(2)
    
    # while True:
    #     if os.path.isfile("stop0.txt") and os.path.isfile("stop1.txt"):
    #         command = f"rm {current_directory}/stop0.txt && rm {current_directory}/stop1.txt"
    #         subprocess.call(command, shell=True)
    #         break
    #     time.sleep(0.05)
    
    while(True):
        num_audio_files = len(os.listdir(audio_folder))
        
        if (num_audio_files == 2):
            break
        elif (num_audio_files < 2):
            update_status("Still waiting for audio files from Pis...\n")
            time.sleep(0.05)
        elif (num_audio_files > 2):
            update_status("There are more than 2 audio files in the audio folder\nPlease remove all files except the 2 that need to be processed.\n")
            time.sleep(0.05)
    signals, times, first_signal, time_offset = read_wav_files()
    
    return signals, times, first_signal, time_offset


def read_wav_files():
    
    update_status("Reading .wav files...\n")
    
    file_list = os.listdir(audio_folder)
    n = 0
    filenames = ["",""]
    global fs

    for file_name in file_list:
        audio_file_path = os.path.join(audio_folder, file_name)
        filenames[n] = file_name.replace(".wav", "")
        n += 1
    
        sample_rate, stereo_audio_data = wav.read(audio_file_path)
        
        fs = sample_rate
        
        # Separate left and right channels
        left_channel = stereo_audio_data[:, 0]
        right_channel = stereo_audio_data[:, 1]
        
        update_status("Stereo channels seperated successfully\n")
        
        if (file_name.__contains__("pi1")):
            signal_3 = np.array(right_channel)
            num_samples = len(signal_3)
            t3 = np.linspace(0, (num_samples - 1) / sample_rate, num_samples)
            signal_4 = np.array(left_channel)
            num_samples = len(signal_4)
            t4 = np.linspace(0, (num_samples - 1) / sample_rate, num_samples)
        else:
            signal_1 = np.array(right_channel)
            num_samples = len(signal_1)
            t1 = np.linspace(0, (num_samples - 1) / sample_rate, num_samples)
            signal_2 = np.array(left_channel)
            num_samples = len(signal_2)
            t2 = np.linspace(0, (num_samples - 1) / sample_rate, num_samples)
    
    update_status(".wav files successfully read!\n")
    
    # Calculate differences in mic start times from time stamp in file name
    # time_offset_scalar = 0.6321205588 * 100
    time_offset = calculate_time_difference(filenames[0], filenames[1])
    message = "Time difference between 2 pis: " + str(time_offset) + "\n"
    # print(message)
    # update_status(message)
    
    if ((time_offset < 0) and (filenames[0].__contains__("pi_"))):
        # pi1 started before pi0
        # Therefore offset pi1
        
        first_signal = "pi1"
        time_offset = -time_offset
    elif ((time_offset<0) and (filenames[0].__contains__("pi1"))):
        # pi0 started before pi1
        # Therefore offset pi0
        
        first_signal = "pi0"
        time_offset = -time_offset
    elif ((time_offset>0) and (filenames[0].__contains__("pi_"))):
        # pi0 started before pi1
        # Therefore offset pi0
        
        first_signal = "pi0"
    elif ((time_offset>0) and (filenames[0].__contains__("pi1"))):
        # pi1 started before pi0
        # Therefore offset pi1
        
        first_signal = "pi1"
    
    return [signal_1, signal_2, signal_3, signal_4], [t1, t2, t3, t4], first_signal, time_offset

def noise_filter(signal):
    # Design a 6th order bandpass filter with a passband from 100 Hz to 20000 Hz
    lowcut = 100
    highcut = 20000
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(6, [low, high], btype='band')
    
    # Apply the bandpass filter to the edited audio data
    filtered_signal = lfilter(b, a, signal)
    
    return filtered_signal

def crop_signal(signal, crop_start_time, crop_stop_time):
    
    start_samples = int(crop_start_time*fs)
    stop_samples = int(crop_stop_time*fs)
    
    if stop_samples == 0:
        stop_samples = -1
    
    cropped_signal = signal[start_samples:stop_samples]
    
    return cropped_signal

def crop_signal(signal, time, crop_start_time, crop_stop_time):
    
    start_samples = int(crop_start_time*fs)
    stop_samples = int(crop_stop_time*fs)
    
    if stop_samples == 0:
        stop_samples = -1
        cropped_signal = signal[start_samples:-1]
        cropped_time = time[:len(signal)-1-start_samples]
    else:
        cropped_signal = signal[start_samples:len(signal)-1-stop_samples]
        cropped_time = time[:len(signal)-1-start_samples-stop_samples]
    
    return cropped_signal, cropped_time

def process_signals(signals, first_signal, time_offset):
    
    if first_signal == "pi0":
        # if the first signal is pi0 then the mics connected to pi0 started recording first and need to be offset
        pi0_offset = time_offset
        pi1_offset = 0
    elif first_signal == "pi1":
        # if the first signal is pi1 then the mics connected to pi1 started recoring first and need to be offset
        pi0_offset = 0
        pi1_offset = time_offset
    
    processed_signals = signals
    n = 0
    for signal in signals:
        # Remove 'pop' from start of mic recording
        pop_time = 0.5
        cropped_signal = crop_signal(signal, pop_time, 0)
        
        # Remove time offset
        
        # Signals 1 & 2 are from pi0 and 3 & 4 are from pi1
        if n < 2:
            offset_signal = crop_signal(cropped_signal, pi0_offset, 0)
        else:
            offset_signal = crop_signal(cropped_signal, pi1_offset, 0)
        
        processed_signals[n] = offset_signal
        
        n = n+1
        
    return processed_signals

def process_signals(signals, times, first_signal, time_offset):
    
    if first_signal == "pi0":
        # if the first signal is pi0 then the mics connected to pi0 started recording first and need to be offset
        pi0_offset = time_offset
        pi1_offset = 0
    elif first_signal == "pi1":
        # if the first signal is pi1 then the mics connected to pi1 started recoring first and need to be offset
        pi0_offset = 0
        pi1_offset = time_offset
    
    processed_signals = signals
    processed_times = times
    n = 0
    for signal in signals:
        # Remove 'pop' from start of mic recording
        pop_time = 2
        cropped_signal, cropped_time = crop_signal(signal, times[n], pop_time, 0)
        
        # Remove time offset
        
        # Signals 1 & 2 are from pi0 and 3 & 4 are from pi1
        if n < 2:
            offset_signal, offset_time = crop_signal(cropped_signal, cropped_time, pi0_offset, pi1_offset)
        else:
            offset_signal, offset_time = crop_signal(cropped_signal, cropped_time, pi1_offset, pi0_offset)
        
        time_to_sample = 3
        num_samples = len(offset_signal)
        num_samples = num_samples - time_to_sample*fs
        time_from_end = num_samples/fs
        
        cropped_signal, cropped_time = crop_signal(offset_signal, offset_time, 0, time_from_end)
        
        processed_signals[n] = cropped_signal
        processed_times[n] = cropped_time
        
        n = n+1
    
    return processed_signals, processed_times

def process_signals2(signals, times):
    processed_signals = signals
    processed_times = times
    n = 0
    for signal in signals:
        # Remove 'pop' from start of mic recording
        pop_time = 0.3
        cropped_signal, cropped_time = crop_signal(signal, times[n], pop_time, 0)

        
        time_to_sample = 0.2
        num_samples = len(cropped_signal)
        num_samples = num_samples - time_to_sample*fs
        time_from_end = num_samples/fs
        
        cropped_signal, cropped_time = crop_signal(cropped_signal, cropped_time, 0, time_from_end)
        
        processed_signals[n] = cropped_signal
        processed_times[n] = cropped_time
        
        n = n+1
        
    return processed_signals, processed_times


# Plots 4 signals
def signal_plot(sim_sig_m1, sim_sig_m2, sim_sig_m3, sim_sig_m4, t1, t2, t3, t4):
    
    # Create a figure with subplots to visualize the signals
    plt.figure()

    # # Plot the original sound source signal
    
    # plt.subplot(5, 1, 1)
    # plt.plot(t, signal)
    # plt.title('Sound Source Signal')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')

    # Plot Microphone 1 Recorded Signal with TDE line
    plt.subplot(5, 1, 2)
    plt.plot(t1, sim_sig_m1)
    # plt.axvline(x=delay_m1, color='r', linestyle='--')  # Add a red dashed line at the TDE for Microphone 1
    plt.title('Microphone 1 Recorded Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot Microphone 2 Recorded Signal with TDE line
    plt.subplot(5, 1, 3)
    plt.plot(t2, sim_sig_m2)
    # plt.axvline(x=delay_m2, color='r', linestyle='--')  # Add a red dashed line at the TDE for Microphone 2
    plt.title('Microphone 2 Recorded Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot Microphone 3 Recorded Signal with TDE line
    plt.subplot(5, 1, 4)
    plt.plot(t3, sim_sig_m3)
    # plt.axvline(x=delay_m3, color='r', linestyle='--')  # Add a red dashed line at the TDE for Microphone 3
    plt.title('Microphone 3 Recorded Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot Microphone 4 Recorded Signal with TDE line
    plt.subplot(5, 1, 5)
    plt.plot(t4, sim_sig_m4)
    # plt.axvline(x=delay_m4, color='r', linestyle='--')  # Add a red dashed line at the TDE for Microphone 4
    plt.title('Microphone 4 Recorded Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.show()

def parse_time_string(time_str):
    # Split the string by underscores and convert components to integers
    time_str = time_str.replace(":", "_")
    time_str = time_str.replace(".", "_")
    components = time_str.split('_')
    if (len(components)==4):
        seconds = int(components[-1])
        minutes = int(components[-2])
        hours = int(components[-3])
        print(hours, ":", minutes, ":", seconds)
        time = datetime(1, 1, 1, hours, minutes, seconds)
    else:
        microseconds = int(components[-1])
        seconds = int(components[-2])
        minutes = int(components[-3])
        hours = int(components[-4])
        print(hours, ":", minutes, ":", seconds, ":", microseconds)
        time = datetime(1, 1, 1, hours, minutes, seconds, microseconds)
    
    return time

def calculate_time_difference(time_str1, time_str2):
    # Parse the time strings into datetime objects
    time1 = parse_time_string(time_str1)
    time2 = parse_time_string(time_str2)
    
    print(time1)
    print(time2)

    # Calculate the time difference in seconds
    time_difference = (time2 - time1).total_seconds()
    return time_difference

# Function to find signal delay
def find_signal_delay(signal1, signal2, sampling_rate):
    # Normalize the signals
    signal1 = signal1 / np.linalg.norm(signal1)
    signal2 = signal2 / np.linalg.norm(signal2)

    # Calculate the cross-correlation
    cross_correlation = np.correlate(signal1, signal2, mode='full')

    # Find the delay corresponding to the maximum correlation
    max_index = np.argmax(cross_correlation)

    # Calculate the delay in samples (accounting for zero-based indexing)
    delay_samples = max_index - len(signal1) + 1

    # Convert delay from samples to seconds
    delay_in_seconds = delay_samples / sampling_rate

    return delay_in_seconds

def Find_TDOA(signal_1, signal_2, signal_3, signal_4):
    
    # Call the function to find the delay for sim_sig_m4
    TDOA_m4 = find_signal_delay(signal_4, signal_1, fs)

    # Display the result for sim_sig_m4
    print('Microphone 4 TDOA:')
    print(f'  Calculated Delay: {TDOA_m4:.7f} seconds')
    print()

    # Call the function to find the delay for sim_sig_m3
    TDOA_m3 = find_signal_delay(signal_3, signal_1, fs)

    # Display the result for sim_sig_m3
    print('Microphone 3 TDOA:')
    print(f'  Calculated Delay: {TDOA_m3:.7f} seconds')
    print()

    # Call the function to find the delay for sim_sig_m2
    TDOA_m2 = find_signal_delay(signal_2, signal_1, fs)

    # Display the result for sim_sig_m2
    print('Microphone 2 TDOA:')
    print(f'  Calculated Delay: {TDOA_m2:.7f} seconds')
    
    return TDOA_m2, TDOA_m3, TDOA_m4

def Find_TDOA_2(signal1, signal2, signal3, signal4):
    # Finds the tdoa between 1 & 2 with 1 as ref mic
    tdoa1 = find_signal_delay(signal2, signal1, fs)
    
    # Finds the tdoa between 3 & 4 with 4 as ref mic
    tdoa2 = find_signal_delay(signal3, signal4, fs)
    
    # Print results
    print(f"TDOA between mics 1 & 2: {tdoa1:.7f}")
    print(f"TDOA between mics 3 & 4: {tdoa2:.7f}")
    
    return tdoa1, tdoa2

# Function to calculate the intersection of hyperbolas
def paramfun(x, diff1, diff2, x1, x2, y1, y2):
    F = [
        np.sqrt(x[0] ** 2 + x[1] ** 2) - np.sqrt((x[0] - x1) ** 2 + (x[1] - y1) ** 2) - diff1,
        np.sqrt(x[0] ** 2 + x[1] ** 2) - np.sqrt((x[0] - x2) ** 2 + (x[1] - y2) ** 2) - diff2
    ]
    return F

def paramfun2(source, diff1, diff2, mic1, mic2, mic3, mic4):
    F = [
        np.sqrt((source[0] - mic1[0]) ** 2 + (source[1] - mic1[1]) ** 2) - np.sqrt((source[0] - mic2[0]) ** 2 + (source[1] - mic2[1]) ** 2) + diff1,
        np.sqrt((source[0] - mic4[0]) ** 2 + (source[1] - mic4[1]) ** 2) - np.sqrt((source[0] - mic3[0]) ** 2 + (source[1] - mic3[1]) ** 2) - diff2
    ]
    return F
    
def Triangulation(TDOA_2, TDOA_3, TDOA_4):
    # Calculate differences in distances between reference mic (mic 1) and other mics
    d_1_2 = TDOA_2 * speed_of_sound
    print('Calculated d_1_2: %.3f' % d_1_2)
    print()

    d_1_3 = TDOA_3 * speed_of_sound
    print('Calculated d_1_3: %.3f' % d_1_3)
    print()

    d_1_4 = TDOA_4 * speed_of_sound
    print('Calculated d_1_4: %.3f' % d_1_4)

    # Find points of intersection between hyperbolae tracking the difference in distance
    x1 = 0
    y1 = grid_height
    x2 = grid_width
    y2 = 0
    diff1 = d_1_2
    diff2 = d_1_4

    # Solve for the intersection
    x0 = [grid_width/2, grid_height/2]
    x_2_4 = fsolve(paramfun, x0, args=(diff1, diff2, x1, x2, y1, y2))

    # Calculate the intersection of hyperbolae from mic 2 & ref mic, and mic 3 & ref mic
    x1 = 0
    y1 = grid_height
    x2 = grid_width
    y2 = grid_height
    diff1 = d_1_2
    diff2 = d_1_3

    x_2_3 = fsolve(paramfun, x0, args=(diff1, diff2, x1, x2, y1, y2))

    # Calculate the intersection of hyperbolae from mic 3 & ref mic, and mic 4 & ref mic
    x1 = grid_width
    y1 = grid_height
    x2 = grid_width
    y2 = 0
    diff1 = d_1_3
    diff2 = d_1_4

    x_3_4 = fsolve(paramfun, x0, args=(diff1, diff2, x1, x2, y1, y2))

    # Calculate average between 3 intersection points, excluding any points that are outside of the grid
    if not (0 <= x_2_3[0] <= grid_width) or not (0 <= x_2_3[1] <= grid_height):
        x_ave = (x_2_4 + x_3_4) / 2
    elif not (0 <= x_2_4[0] <= grid_width) or not (0 <= x_2_4[1] <= grid_height):
        x_ave = (x_2_3 + x_3_4) / 2
    elif not (0 <= x_3_4[0] <= grid_width) or not (0 <= x_3_4[1] <= grid_height):
        x_ave = (x_2_3 + x_2_4) / 2
    else:
        x_ave = (x_2_3 + x_2_4 + x_3_4) / 3

    # Display the output

    print('Predicted Sound Source Position from ref mic and mics 2 and 4:', x_2_4)
    print('Predicted Sound Source Position from ref mic and mics 2 and 3:', x_2_3)
    print('Predicted Sound Source Position from ref mic and mics 3 and 4:', x_3_4)
    print('Predicted Sound Source Position average from 3 readings above:', x_ave)
    
    return x_ave

def Triangulation2(tdoa1, tdoa2):
    # Calculate differences in distances between sound source and mics 1 & 2
    d_1_2 = tdoa1 * speed_of_sound
    # Calculate differences in distances between sound source and mics 3 & 4
    d_3_4 = tdoa2 * speed_of_sound
    
    print('Calculated d_1_2: %.7f' % d_1_2)
    print()
    print('Calculated d_3_4: %.7f' % d_3_4)
    print()
    
    # Calculate the intersection of hyperbolae from mics 1 & 2 and mics 3 & 4
    start_pt = [grid_width/2, grid_height/2]
    diff1 = d_1_2
    diff2 = d_3_4

    source_position = fsolve(paramfun2, start_pt, args=(diff1, diff2, mic1_pos, mic2_pos, mic3_pos, mic4_pos)) 
    
    pos = [source_position[0], source_position[1]]
    
    print(f"Predicted Sound Source Position: {pos}")
    message = f"Predicted Sound Source Position: {pos}\n"
    update_status(message)
    
    return source_position
    
def Find_record_delay(signals):
    pop_time = 0.3
    pop_samples = int(pop_time * fs)
    
    signal1 = signals[0][:pop_samples]
    signal2 = signals[1][:pop_samples]
    signal3 = signals[2][:pop_samples]
    signal4 = signals[3][:pop_samples]
    
    del_1_2 = find_signal_delay(signal2, signal1, fs)
    del_1_3 = find_signal_delay(signal3, signal1, fs)
    del_1_4 = find_signal_delay(signal4, signal1, fs)
    
    del_2_3 = find_signal_delay(signal3, signal2, fs)
    del_2_4 = find_signal_delay(signal4, signal2, fs)
    
    del_3_4 = find_signal_delay(signal4, signal3, fs)
    
    print(f"del_1_2: {del_1_2}")
    print(f"del_1_3: {del_1_3}")
    print(f"del_1_4: {del_1_4}")
    print(f"del_2_3: {del_2_3}")
    print(f"del_2_4: {del_2_4}")
    print(f"del_3_4: {del_3_4}")
    
    
    

def create_sim(signal, position):
    
    # Shift signal based on position
    
    # Calulate distance from each of the microphones to the simulated position
    distance_to_1 = np.sqrt((mic1_pos[0] - position[0])**2 + (mic1_pos[1] - position[1])**2)
    distance_to_2 = np.sqrt((mic2_pos[0] - position[0])**2 + (mic2_pos[1] - position[1])**2)
    distance_to_3 = np.sqrt((mic3_pos[0] - position[0])**2 + (mic3_pos[1] - position[1])**2)
    distance_to_4 = np.sqrt((mic4_pos[0] - position[0])**2 + (mic4_pos[1] - position[1])**2)
    
    # Calculate time difference of arrivals between all of the microphones and the reference microphone (mic 1)
    tdoa_1_2 = (distance_to_1 - distance_to_2)/speed_of_sound
    tdoa_1_3 = (distance_to_1 - distance_to_3)/speed_of_sound
    tdoa_1_4 = (distance_to_1 - distance_to_4)/speed_of_sound
    
    offset = int(0.1*fs)
    signal1 = signal[offset:]
    
    if tdoa_1_2 < 0:
        signal2 = signal[offset-int(tdoa_1_2*fs):]
    elif tdoa_1_2==0:
        signal2 =  signal[offset:]
    else:
        signal2 = signal[offset-int(tdoa_1_2*fs):]
    
    if tdoa_1_3 < 0:
        signal3 = signal[offset-int(tdoa_1_3*fs):]
    elif tdoa_1_3==0:
        signal3 =  signal[offset:]
    else:
        signal3 = signal[offset-int(tdoa_1_3*fs):]
        
    if tdoa_1_4 < 0:
        signal4 = signal[offset-int(tdoa_1_4*fs):]
    elif tdoa_1_4==0:
        signal4 =  signal[offset:]
    else:
        signal4 = signal[offset-int(tdoa_1_4*fs):]
    
    a = int(2*fs)
    signal1 = signal1[:a]
    signal2 = signal2[:a]
    signal3 = signal3[:a]
    signal4 = signal4[:a]
    
    tdoa2, tdoa3, tdoa4 = Find_TDOA(signal1, signal2, signal3, signal4)
    sound_position = Triangulation(tdoa2, tdoa3, tdoa4)
    
    print(f"Actual tdoas (2,3,4): {tdoa_1_2}, {tdoa_1_3}, {tdoa_1_4} .......calculated (2, 3, 4): {tdoa2}, {tdoa3}, {tdoa4}")
    print(f"Actual pos: {position}..... Calculated pos: {sound_position}")
    
    update_graph(sound_position[0], sound_position[1])
    

def create_sim2(signal, position):
    
    # Shift signal based on position
    
    # Calulate distance from each of the microphones to the simulated position
    distance_to_1 = np.sqrt((mic1_pos[0] - position[0])**2 + (mic1_pos[1] - position[1])**2)
    distance_to_2 = np.sqrt((mic2_pos[0] - position[0])**2 + (mic2_pos[1] - position[1])**2)
    distance_to_3 = np.sqrt((mic3_pos[0] - position[0])**2 + (mic3_pos[1] - position[1])**2)
    distance_to_4 = np.sqrt((mic4_pos[0] - position[0])**2 + (mic4_pos[1] - position[1])**2)
    
    # Calculate time difference of arrivals between all of the microphones and the reference mics
    tdoa_2_1 = (distance_to_2 - distance_to_1)/speed_of_sound
    tdoa_3_4 = (distance_to_3 - distance_to_4)/speed_of_sound
    
    offset = int(0.1*fs)
    signal2 = signal[offset:]
    signal3 = signal[offset:]
    
    if tdoa_2_1 < 0:
        signal1 = signal[offset-int(tdoa_2_1*fs):]
    elif tdoa_2_1==0:
        signal1 =  signal[offset:]
    else:
        signal1 = signal[offset-int(tdoa_2_1*fs):]
    
    if tdoa_3_4 < 0:
        signal4 = signal[offset-int(tdoa_3_4*fs):]
    elif tdoa_3_4==0:
        signal4 =  signal[offset:]
    else:
        signal4 = signal[offset-int(tdoa_3_4*fs):]
    
    a = int(2*fs)
    signal1 = signal1[:a]
    signal2 = signal2[:a]
    signal3 = signal3[:a]
    signal4 = signal4[:a]
    
    tdoa12, tdoa34 = Find_TDOA_2(signal1, signal2, signal3, signal4)
    sound_position = Triangulation2(tdoa12, tdoa34)
    
    print(f"Actual tdoas (1-2, 3-4): {tdoa_2_1}, {tdoa_3_4}.......calculated (1-2, 3-4): {tdoa12}, {tdoa34}")
    print(f"Actual pos: {position}..... Calculated pos: {sound_position}")
    
    update_graph(sound_position[0], sound_position[1])
    

if __name__ == "__main__":
    
    # Initialize and start GUI
    app = create_app_window()
    app.mainloop()
    
    
    
    