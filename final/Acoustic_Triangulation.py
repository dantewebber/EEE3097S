import os
import time
import pygame
import subprocess
import concurrent.futures
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

# Constants
grid_width = 0.783 # 0.841 - A1 # Width of the grid [m]
grid_height = 0.49 # 0.594 - A1 # Height of the grid [m]
speed_of_sound = 343  # Speed of sound in the environment (m/s)
fs = 48000 # sample rate (Hz)

# Target file and target location
target_file = "start.txt"
destination = "/home/pi/"

# Define positions of microphones and a random position for the sound source
Microphone1_position = [0, 0]
Microphone2_position = [0, grid_height]
Microphone3_position = [grid_width, grid_height]
Microphone4_position = [grid_width, 0]

# Folders
current_directory = os.getcwd()
audio_folder = os.path.join(current_directory, "audio")
    
def send_file_to_raspberry_pi(pi_host):
    command = f"scp {target_file} {pi_host}:{destination}"

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result, error = process.communicate()

    if process.returncode != 0:
        print(f'Failed to send file to {pi_host}. Error: {error.decode()}')
    else:
        print(f'Successfully sent file to {pi_host}.')

def wait_for_audio():
    while(True):
        if (len(os.listdir(audio_folder)) == 2):
            break

    print("Audio files detected")
    time.sleep(10)
    read_wav_files()
        
# READING WAV FILE IN
def play_wav_file(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    pygame.time.wait(5000)  # You can change this if needed
    pygame.mixer.quit()
    
def read_wav_files():
    print("Reading wav files")
    seconds_to_remove_start = 0.5
    seconds_to_remove_end = 0.5
    file_list = os.listdir(audio_folder)
    n = 0
    filenames = ["",""]


    for file_name in file_list:
        audio_file_path = os.path.join(audio_folder, file_name)
        filenames[n] = file_name.replace(".wav", "")
        n += 1
    
        sample_rate, stereo_audio_data = wav.read(audio_file_path)
        
        # Calculate the number of samples to remove from the start and end
        samples_to_remove_start = int(sample_rate * seconds_to_remove_start)
        samples_to_remove_end = int(sample_rate * seconds_to_remove_end)
        
        # Separate left and right channels
        left_channel = stereo_audio_data[:, 0]
        right_channel = stereo_audio_data[:, 1]
        
        # Remove the specified number of samples from the start and end
        # chopped_left = left_channel[samples_to_remove_start:-samples_to_remove_end]
        # chopped_right = right_channel[samples_to_remove_start:-samples_to_remove_end]
        
        # Design a 6th order bandpass filter with a passband from 100 Hz to 20000 Hz
        lowcut = 100
        highcut = 20000
        nyquist = 0.5 * sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(6, [low, high], btype='band')
        
        # Apply the bandpass filter to the edited audio data
        filtered_left = lfilter(b, a, left_channel)
        filtered_right = lfilter(b, a, right_channel)
        
        if (file_name.__contains__("pi_")):
            filtered_signal_1 = np.array(filtered_right)
            filtered_signal_2 = np.array(filtered_left)
        else:
            filtered_signal_3 = np.array(filtered_left)
            filtered_signal_4 = np.array(filtered_right)
    
    print("Obtained split signals")
    # Calculate the time values
    # REMOVE MICROPHONE STARTUP 'POPS' FROM SIGNALS
    
    num_samples = len(filtered_signal_1)
    t1 = np.linspace(0, (num_samples - 1) / sample_rate, num_samples)
    filtered_signal_1 = filtered_signal_1[samples_to_remove_start:-samples_to_remove_end]
    t1 = t1[samples_to_remove_start:-samples_to_remove_end]
    
    num_samples = len(filtered_signal_2)
    t2 = np.linspace(0, (num_samples - 1) / sample_rate, num_samples)
    filtered_signal_2 = filtered_signal_2[samples_to_remove_start:-samples_to_remove_end]
    t2 = t2[samples_to_remove_start:-samples_to_remove_end]
    
    num_samples = len(filtered_signal_3)
    t3 = np.linspace(0, (num_samples - 1) / sample_rate, num_samples)
    filtered_signal_3 = filtered_signal_3[samples_to_remove_start:-samples_to_remove_end]
    t3 = t3[samples_to_remove_start:-samples_to_remove_end]
    
    num_samples = len(filtered_signal_4)
    t4 = np.linspace(0, (num_samples - 1) / sample_rate, num_samples)
    filtered_signal_4 = filtered_signal_4[samples_to_remove_start:-samples_to_remove_end]
    t4 = t4[samples_to_remove_start:-samples_to_remove_end]
    
    print(sample_rate)
    print(filenames[0])
    print(filenames[1])
    
    # REMOVE TIME OFFSET FROM LEADING SIGNAL (STARTED RECORDING FIRST)
    
    time_offset = calculate_time_difference(filenames[0], filenames[1])
    if ((time_offset < 0) and (filenames[0].__contains__("pi0"))):
        # pi1 started before pi0
        time_offset = -time_offset
        sample_offset = int(sample_rate * time_offset)
        filtered_signal_3 = filtered_signal_3[sample_offset:-1]
        t3 = t3[sample_offset:-1]
        filtered_signal_4 = filtered_signal_4[sample_offset:-1]
        t4 = t4[sample_offset:-1]
    elif ((time_offset<0) and (filenames[0].__contains__("pi1"))):
        # pi0 started before pi0
        time_offset = -time_offset
        sample_offset = int(sample_rate * time_offset)
        filtered_signal_1 = filtered_signal_1[sample_offset:-1]
        t1 = t1[sample_offset:-1]
        filtered_signal_2 = filtered_signal_2[sample_offset:-1]
        t2 = t2[sample_offset:-1]
    elif ((time_offset>0) and (filenames[0].__contains__("pi0"))):
        # pi0 started before pi1
        sample_offset = int(sample_rate * time_offset)
        filtered_signal_1 = filtered_signal_1[sample_offset:-1]
        t1 = t1[sample_offset:-1]
        filtered_signal_2 = filtered_signal_2[sample_offset:-1]
        t2 = t2[sample_offset:-1]
    elif ((time_offset>0) and (filenames[0].__contains__("pi1"))):
        # pi1 started before pi0
        sample_offset = int(sample_rate * time_offset)
        filtered_signal_3 = filtered_signal_3[sample_offset:-1]
        t3 = t3[sample_offset:-1]
        filtered_signal_4 = filtered_signal_4[sample_offset:-1]
        t4 = t4[sample_offset:-1]
    
    
    # time_offset = 3
    # sample_offset = int(sample_rate * time_offset)
    
    # filtered_signal_1 = filtered_signal_1[0:sample_offset]
    # filtered_signal_2 = filtered_signal_2[0:sample_offset]
    # filtered_signal_3 = filtered_signal_3[0:sample_offset]
    # filtered_signal_4 = filtered_signal_4[0:sample_offset]
    
    # t1 = t1[0:sample_offset]
    # t2 = t2[0:sample_offset]
    # t3 = t3[0:sample_offset]
    # t4 = t4[0:sample_offset]
         
    # time_offset = calculate_time_difference("i_05_12_09_300", "i_05_12_07_200")
    print(time_offset)
    
    # signal_plot(filtered_signal_1, filtered_signal_2, filtered_signal_3,filtered_signal_4, t1, t2, t3, t4)
    
    acoustic_localisation(filtered_signal_1, filtered_signal_2, filtered_signal_3, filtered_signal_4)
    
    # To implement continuous tracking
    

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
    print(f"The time diff is {time_difference}")
    return time_difference

    # # If milliseconds are present, round to the nearest millisecond, else round to the nearest second
    # if parse_time_string(time_str1)[-1] > 0 or parse_time_string(time_str2)[-1] > 0:
    #     return round(time_difference, 3)  # Milliseconds
    # else:
    #     return round(time_difference)  # Seconds


# Signal definition


def Simulation():
    global Sound_source_position
    Sound_source_position = [np.random.rand() * grid_width, np.random.rand() * grid_height]
    print(Sound_source_position)
    
    # Actual sound source position
    Actual_sound_source_position = Sound_source_position

    # Simulated audio signal parameters
    global fs
    fs = 48000  # Sampling frequency (Hz)
    duration = 0.1  # Duration of the signal (seconds)
    chirp_frequency = 1000  # Chirp signal frequency (Hz)

    # Generate a simulated audio signal (chirp signal)
    global t
    t = np.arange(0, duration, 1/fs)
    global signal
    signal = chirp(t, 0, duration, chirp_frequency)
    # signal = np.cos(2 * np.pi * chirp_frequency * t)

    # Calculate the distance between the sound source and each microphone
    Distance_m1_ss = np.sqrt((Sound_source_position[0] - Microphone1_position[0]) ** 2 +
                            (Sound_source_position[1] - Microphone1_position[1]) ** 2)
    Distance_m2_ss = np.sqrt((Sound_source_position[0] - Microphone2_position[0]) ** 2 +
                            (Sound_source_position[1] - Microphone2_position[1]) ** 2)
    Distance_m3_ss = np.sqrt((Sound_source_position[0] - Microphone3_position[0]) ** 2 +
                            (Sound_source_position[1] - Microphone3_position[1]) ** 2)
    Distance_m4_ss = np.sqrt((Sound_source_position[0] - Microphone4_position[0]) ** 2 +
                            (Sound_source_position[1] - Microphone4_position[1]) ** 2)

    global delay_m1 
    global delay_m2
    global delay_m3
    global delay_m4
    
    # Calculate time delays for each microphone
    delay_m1 = Distance_m1_ss / speed_of_sound
    delay_m2 = Distance_m2_ss / speed_of_sound
    delay_m3 = Distance_m3_ss / speed_of_sound
    delay_m4 = Distance_m4_ss / speed_of_sound

    # Generate random noise for each microphone with slight variations
    noise_amplitude = 0.4  # Adjust the noise amplitude as needed
    num_samples = len(t)

    noise_m1 = noise_amplitude * (np.random.randn(num_samples) + 0.4 * np.random.rand() * np.random.randn(num_samples))
    noise_m2 = noise_amplitude * (np.random.randn(num_samples) + 0.4 * np.random.rand() * np.random.randn(num_samples))
    noise_m3 = noise_amplitude * (np.random.randn(num_samples) + 0.4 * np.random.rand() * np.random.randn(num_samples))
    noise_m4 = noise_amplitude * (np.random.randn(num_samples) + 0.4 * np.random.rand() * np.random.randn(num_samples))
    
    global sim_sig_m1
    global sim_sig_m2
    global sim_sig_m3
    global sim_sig_m4

    # Add noise to the original chirp signal after applying time delays
    sim_sig_m1 = chirp(t - delay_m1, 0, duration, chirp_frequency) + noise_m1
    sim_sig_m2 = chirp(t - delay_m2, 0, duration, chirp_frequency) + noise_m2
    sim_sig_m3 = chirp(t - delay_m3, 0, duration, chirp_frequency) + noise_m3
    sim_sig_m4 = chirp(t - delay_m4, 0, duration, chirp_frequency) + noise_m4
    
    acoustic_localisation(sim_sig_m1, sim_sig_m2, sim_sig_m3, sim_sig_m4)
    
    # Actual TDOA and accuracy as a result of
    actual_TDOA_m4 = delay_m4 - delay_m1
    actual_TDOA_m3 = delay_m3 - delay_m1
    actual_TDOA_m2 = delay_m2 - delay_m1
    
    TDOA_acc(actual_TDOA_m2, actual_TDOA_m3, actual_TDOA_m4)
    
    print('Microphone 2 TDOA:')
    print(f'  Reference Delay:  {actual_TDOA_m2:.7f} seconds')
    print(f'  Accuracy: {accuracy_m2:.2f}%')
    
    print('Microphone 3 TDOA:')
    print(f'  Reference Delay:  {actual_TDOA_m3:.7f} seconds')
    print(f'  Accuracy: {accuracy_m3:.2f}%')
    
    print('Microphone 4 TDOA:')
    print(f'  Reference Delay:  {actual_TDOA_m4:.7f} seconds')
    print(f'  Accuracy: {accuracy_m4:.2f}%')
    
    # Actual difference in distances
    print('Actual d_1_2: %.3f' % (Distance_m2_ss - Distance_m1_ss))
    print()
    
    print('Actual d_1_3: %.3f' % (Distance_m3_ss - Distance_m1_ss))
    print()
    
    print('Actual d_1_4: %.3f' % (Distance_m4_ss - Distance_m1_ss))
    print()
    
    # Display actual sound source position from simulation
    print('Actual Sound Source Position:', Actual_sound_source_position)
    
    # Calculate the Euclidean distance between actual and predicted positions
    error_2_4 = np.linalg.norm(Actual_sound_source_position - x_2_4)
    error_2_3 = np.linalg.norm(Actual_sound_source_position - x_2_3)
    error_3_4 = np.linalg.norm(Actual_sound_source_position - x_3_4)
    error_ave = np.linalg.norm(Actual_sound_source_position - x_ave)

    d_max = np.linalg.norm([grid_width, grid_height])

    # Calculate the accuracy as a percentage
    acc_2_4 = (1 - error_2_4 / d_max) * 100
    acc_2_3 = (1 - error_2_3 / d_max) * 100
    acc_3_4 = (1 - error_3_4 / d_max) * 100
    acc_ave = (1 - error_ave / d_max) * 100

    print('Accuracy of ref mic and mics 2 & 4: %.3f%%' % acc_2_4)
    print('Accuracy of ref mic and mics 2 & 3: %.3f%%' % acc_2_3)
    print('Accuracy of ref mic and mics 3 & 4: %.3f%%' % acc_3_4)
    print('Accuracy of average position: %.3f%%' % acc_ave)
    print('Distance from actual sound source: %.3fm' % error_ave)

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

def TDOA_acc(tdoa2, tdoa3, tdoa4):
    # Calculate accuracy for sim_sig_m4
    global accuracy_m4
    global accuracy_m3
    global accuracy_m2
    
    accuracy_m4 = 100 * (1 - abs((tdoa4 - TDOA_m4) / tdoa4))
    accuracy_m3 = 100 * (1 - abs((tdoa3 - TDOA_m3) / tdoa3))
    accuracy_m2 = 100 * (1 - abs((tdoa2 - TDOA_m2) / tdoa2))

def Find_TDOA(signal_1, signal_2, signal_3, signal_4):

    global TDOA_m2
    global TDOA_m3
    global TDOA_m4
    
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

# Function to calculate the intersection of hyperbolas
def paramfun(x, diff1, diff2, x1, x2, y1, y2):
    F = [
        np.sqrt(x[0] ** 2 + x[1] ** 2) - np.sqrt((x[0] - x1) ** 2 + (x[1] - y1) ** 2) + diff1,
        np.sqrt(x[0] ** 2 + x[1] ** 2) - np.sqrt((x[0] - x2) ** 2 + (x[1] - y2) ** 2) + diff2
    ]
    return F

def Triangulation(TDOA_2, TDOA_3, TDOA_4):
    global x_ave
    global x_2_4
    global x_3_4
    global x_2_3
    
    
    # Calculate differences in distances between reference mic (mic 1) and other mics
    d_1_2 = TDOA_m2 * speed_of_sound
    print('Calculated d_1_2: %.3f' % d_1_2)
    print()

    d_1_3 = TDOA_m3 * speed_of_sound
    print('Calculated d_1_3: %.3f' % d_1_3)
    print()

    d_1_4 = TDOA_m4 * speed_of_sound
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

def plot_output():
    # Plot microphones, sound source, and predicted positions
    plt.figure()
    plt.plot(Microphone1_position[0], Microphone1_position[1], 'b^', label='Mic 1')
    plt.plot(Microphone2_position[0], Microphone2_position[1], 'b^', label='Mic 2')
    plt.plot(Microphone3_position[0], Microphone3_position[1], 'b^', label='Mic 3')
    plt.plot(Microphone4_position[0], Microphone4_position[1], 'b^', label='Mic 4')
    # plt.plot(Sound_source_position[0], Sound_source_position[1], 'r*', label='Actual Sound Source')
    plt.plot(x_ave[0], x_ave[1], 'k.', label='Triangulated Position (Average)')

    # Uncomment these lines to plot the intersection points of hyperbolae
    # plt.plot(x_2_4[0], x_2_4[1], 'k.')
    # plt.plot(x_2_3[0], x_2_3[1], 'm.')
    # plt.plot(x_3_4[0], x_3_4[1], 'g.')

    plt.grid(True)
    plt.legend()
    plt.xlabel('x in meters')
    plt.ylabel('y in meters')
    plt.show()

def acoustic_localisation(signal_1, signal_2, signal_3, signal_4):
    
    Find_TDOA(signal_1, signal_2, signal_3, signal_4)
    Triangulation(TDOA_m2, TDOA_m3, TDOA_m4)
    update_graph(x_ave[0], x_ave[1])
    # plot_output()
    
    # clear audio folder
    folder_path = "audio"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # call michael function again

# read_wav_files()
# Simulation()
# signal_plot()

#GUI
# Function to start the process
def start_process():
    global positionX, positionY
    
    # Pi hostnames or IP addresses
    raspberry_pis = ["pi@raspberrypi.local", "pi@raspberrypi1.local"]
    # Simulation()
    # read_wav_files()

    # Send file to both Raspberry Pis
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(send_file_to_raspberry_pi, raspberry_pis)
    
    wait_for_audio()
        
    # read_wav_files()

    # Set the new position (0.3, 0.4)
    # positionX = 0.6
    # positionY = 0.4

    # Update the graph with a red dot
    # update_graph()
    
# Function to stop the process
def stop_process():
    # Add your stop process logic here
    exit()
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