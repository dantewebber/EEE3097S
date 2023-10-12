import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy.optimize import fsolve

# Signal definition
# Constants
grid_width = 0.49 # 0.594  # Width of the grid
grid_height = 0.783 # 0.841  # Height of the grid
speed_of_sound = 343  # Speed of sound in the environment (m/s)

# Define positions of microphones and a random position for the sound source
Microphone1_position = [0, 0]
Microphone2_position = [0, grid_height]
Microphone3_position = [grid_width, grid_height]
Microphone4_position = [grid_width, 0]

def Simulation():
    global Sound_source_position
    Sound_source_position = [np.random.rand() * grid_width, np.random.rand() * grid_height]
    print(Sound_source_position)
    
    # Actual sound source position
    Actual_sound_source_position = Sound_source_position

    # Simulated audio signal parameters
    global fs
    fs = 44100  # Sampling frequency (Hz)
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
    
    main(sim_sig_m1, sim_sig_m2, sim_sig_m3, sim_sig_m4)
    
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

def signal_plot():
    
    # Create a figure with subplots to visualize the signals
    plt.figure()

    # Plot the original sound source signal
    plt.subplot(5, 1, 1)
    plt.plot(t, signal)
    plt.title('Sound Source Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot Microphone 1 Recorded Signal with TDE line
    plt.subplot(5, 1, 2)
    plt.plot(t, sim_sig_m1)
    plt.axvline(x=delay_m1, color='r', linestyle='--')  # Add a red dashed line at the TDE for Microphone 1
    plt.title('Microphone 1 Recorded Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot Microphone 2 Recorded Signal with TDE line
    plt.subplot(5, 1, 3)
    plt.plot(t, sim_sig_m2)
    plt.axvline(x=delay_m2, color='r', linestyle='--')  # Add a red dashed line at the TDE for Microphone 2
    plt.title('Microphone 2 Recorded Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot Microphone 3 Recorded Signal with TDE line
    plt.subplot(5, 1, 4)
    plt.plot(t, sim_sig_m3)
    plt.axvline(x=delay_m3, color='r', linestyle='--')  # Add a red dashed line at the TDE for Microphone 3
    plt.title('Microphone 3 Recorded Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot Microphone 4 Recorded Signal with TDE line
    plt.subplot(5, 1, 5)
    plt.plot(t, sim_sig_m4)
    plt.axvline(x=delay_m4, color='r', linestyle='--')  # Add a red dashed line at the TDE for Microphone 4
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
    plt.plot(Sound_source_position[0], Sound_source_position[1], 'r*', label='Actual Sound Source')
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

def main(signal_1, signal_2, signal_3, signal_4):
    Find_TDOA(signal_1, signal_2, signal_3, signal_4)
    Triangulation(TDOA_m2, TDOA_m3, TDOA_m4)
    plot_output()

# Simulation()
# signal_plot()