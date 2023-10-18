import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io.wavfile import read
from scipy.optimize import fsolve
from scipy.signal import butter, lfilter


sample_rate = 48000

def main():

    signal_1, signal_2, signal_3, signal_4 = read_audio_folder()
    
    print("audio files read")

    #clean up signals
    signal_1 = filter(remove_pop(signal_1))
    signal_2 = filter(remove_pop(signal_2))
    signal_3 = filter(remove_pop(signal_3))
    signal_4 = filter(remove_pop(signal_4))

    print("audio files filtered")

    # Uncomment to see signal plots
    print("Plotting signals")
    fig, ax = plt.subplots()

    ax.plot(signal_1, label="Signal 1")
    ax.plot(signal_2, label="Signal 2")
    ax.plot(signal_3, label="Signal 3")
    ax.plot(signal_4, label="Signal 4")
   
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.legend()
    # Display the plot
    plt.show()
    
    print("calling acoustic localisation")
    
    acoustic_localisation(signal_1, signal_2, signal_3, signal_4)

def plot_output(x, y):

    # Define grid size
    grid_width_m = 0.783 
    grid_height_m = 0.49

    grid_width_in = grid_width_m * 39.971
    grid_height_in = grid_height_m * 39.971

    #Plot
    plt.figure(figsize=(grid_width_in, grid_height_in))

    #plot mic positions
    mic_positions = [(0,0), (0, 0.49), (0.783, 0.49), (0.783, 0)]
    for i, mic in enumerate(mic_positions):
        a, b = mic
        plt.plot(a, b, 's', c='black', markersize=10)
        plt.annotate(f'  Mic {i+1}', mic)

    #plot triangulated poistion
    plt.plot(x, y, '.', c='red', markersize=25)
    plt.annotate("  Triangulated Position (Average)", (x, y))

    plt.grid(True)
    plt.xlabel('x in meters')
    plt.ylabel('y in meters')
    plt.show()

def read_audio_folder():

    audio_folder = os.path.join(os.getcwd(), "audio")
    files = os.listdir(audio_folder)

    time_diff = get_time_diff(files[0], files[1])

    for file in files:
        # Load stereo audio file
        full_path = audio_folder + '/' + file
        _, data = read(full_path)

        if file.__contains__("pi_"):
            signal_1 = data[:, 1]
            signal_2 = data[:, 0]

        elif file.__contains__("pi1_"):
            signal_3 = data[:, 0]
            signal_4 = data[:, 1]

    #adjusts for time difference
    diff_in_samples = abs(int(time_diff * sample_rate))

    if time_diff == 0:
        pass
    elif time_diff > 0:
        signal_1 = signal_1[diff_in_samples:]
        signal_2 = signal_2[diff_in_samples:]
        signal_3 = signal_3[:-diff_in_samples]
        signal_4 = signal_4[:-diff_in_samples]
    else:
        signal_3 = signal_3[diff_in_samples:]
        signal_4 = signal_4[diff_in_samples:]
        signal_1 = signal_1[:-diff_in_samples]
        signal_2 = signal_2[:-diff_in_samples]

    return signal_1, signal_2, signal_3, signal_4

def get_time_diff(file1, file2):

    time1 = file1.split("_")[1].replace(".wav", "")
    time2 = file2.split("_")[1].replace(".wav", "")

    FMT = '%H:%M:%S.%f'
    tdelta = datetime.strptime(time2, FMT) - datetime.strptime(time1, FMT)

    return tdelta.total_seconds()

def read_audio_file(file_path):

    sample_rate, data = read(file_path)
    return data

def acoustic_localisation(signal_1, signal_2, signal_3, signal_4):
    
    TDOA_m2, TDOA_m3, TDOA_m4  = find_TDOA(signal_1, signal_2, signal_3, signal_4)
    x, y = triangulate(TDOA_m2, TDOA_m3, TDOA_m4)
    plot_output(x,y)
    
    return x, y

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

def find_TDOA(signal_1, signal_2, signal_3, signal_4):

    fs = 48000
    
    # Call the function to find the delay for sim_sig_m4
    TDOA_m4 = find_signal_delay(signal_4, signal_1, fs)

    # Display the result for sim_sig_m4
    print('Microphone 4 TDOA:')
    print(f'  Calculated Delay: {TDOA_m4:.7f} seconds')

    # Call the function to find the delay for sim_sig_m3
    TDOA_m3 = find_signal_delay(signal_3, signal_1, fs)

    # Display the result for sim_sig_m3
    print('Microphone 3 TDOA:')
    print(f'  Calculated Delay: {TDOA_m3:.7f} seconds')

    # Call the function to find the delay for sim_sig_m2
    TDOA_m2 = find_signal_delay(signal_2, signal_1, fs)

    # Display the result for sim_sig_m2
    print('Microphone 2 TDOA:')
    print(f'  Calculated Delay: {TDOA_m2:.7f} seconds')

    return TDOA_m2, TDOA_m3, TDOA_m4

def paramfun(x, diff1, diff2, x1, x2, y1, y2):
    F = [
        np.sqrt(x[0] ** 2 + x[1] ** 2) - np.sqrt((x[0] - x1) ** 2 + (x[1] - y1) ** 2) + diff1,
        np.sqrt(x[0] ** 2 + x[1] ** 2) - np.sqrt((x[0] - x2) ** 2 + (x[1] - y2) ** 2) + diff2
    ]
    return F

def remove_pop(signal):
    #removes initial pop sound when mic starts upss
    start = int(sample_rate * 3)
    end = int(sample_rate * 1.5)

    signal = signal[start:-end]

    return signal
    
def filter(signal):
    lowcut = 100
    highcut = 20000
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(6, [low, high], btype='band')
    
    # Apply the bandpass filter to the edited audio data
    signal = lfilter(b, a, signal)

    return signal
    
def triangulate(TDOA_m2, TDOA_m3, TDOA_m4):
    speed_of_sound = 343
    grid_width = 0.783 
    grid_height = 0.49
    
    # Calculate differences in distances between reference mic (mic 1) and other mics
    d_1_2 = TDOA_m2 * speed_of_sound
    print('Calculated d_1_2: %.3f' % d_1_2)

    d_1_3 = TDOA_m3 * speed_of_sound
    print('Calculated d_1_3: %.3f' % d_1_3)

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

    # Uncomment to display the output
    #print('Predicted Sound Source Position from ref mic and mics 2 and 4:', x_2_4)
    #print('Predicted Sound Source Position from ref mic and mics 2 and 3:', x_2_3)
    #print('Predicted Sound Source Position from ref mic and mics 3 and 4:', x_3_4)
    print('Predicted Sound Source Position average from 3 readings above:', x_ave)

    x, y = x_ave
    return x,y

if __name__ == "__main__":
    main()