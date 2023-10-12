import time
import pygame
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

def play_wav_file(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    pygame.time.wait(5000)  # You can change this if needed
    pygame.mixer.quit()

def edit_wav_file(input_file_path, output_file_path, m1_output_path, m2_output_path, seconds_to_remove_start, seconds_to_remove_end):
    sample_rate, audio_data = wavfile.read(input_file_path)
    
    # Calculate the number of samples to remove from the start and end
    samples_to_remove_start = int(sample_rate * seconds_to_remove_start)
    samples_to_remove_end = int(sample_rate * seconds_to_remove_end)
    
    # Remove the specified number of samples from the start and end
    edited_audio_data = audio_data[samples_to_remove_start:-samples_to_remove_end]
    
    # Design a 6th order bandpass filter with a passband from 100 Hz to 20000 Hz
    lowcut = 100
    highcut = 20000
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(6, [low, high], btype='band')
    
    # Apply the bandpass filter to the edited audio data
    filtered_audio_data = lfilter(b, a, edited_audio_data)
    
    # Separate left and right channels
    left_channel = filtered_audio_data[:, 0]
    right_channel = filtered_audio_data[:, 1]
    
    # Save the edited and filtered audio data to the output file
    wavfile.write(output_file_path, sample_rate, filtered_audio_data)
    
    # Save the left and right channels as separate mono WAV files (M1 and M2)
    wavfile.write(m1_output_path, sample_rate, left_channel)
    wavfile.write(m2_output_path, sample_rate, right_channel)

if __name__ == "__main__":
    input_file_path12 = "C:\\Users\\Joshu\\OneDrive\\Desktop\\MICS\\soundpi0_05_12_07.wav"
    input_file_path34 = "C:\\Users\\Joshu\\OneDrive\\Desktop\\MICS\\soundpi1_05_12_09.wav"

    output_file_path12 = "C:\\Users\\Joshu\\OneDrive\\Desktop\\MICS\\edited12_file.wav"
    output_file_path34 = "C:\\Users\\Joshu\\OneDrive\\Desktop\\MICS\\edited34_file.wav"
    
    m1_L_output_path = "C:\\Users\\Joshu\\OneDrive\\Desktop\\MICS\\M1_l_pi.wav"
    m2_R_output_path = "C:\\Users\\Joshu\\OneDrive\\Desktop\\MICS\\M2_r_pi.wav"
    m3_L_output_path = "C:\\Users\\Joshu\\OneDrive\\Desktop\\MICS\\M3_l_pi1.wav"
    m4_R_output_path = "C:\\Users\\Joshu\\OneDrive\\Desktop\\MICS\\M4_r_pi1.wav"
    
    # Specify the number of seconds to remove from the start and end of the WAV file
    seconds_to_remove_start = 0.5  # Adjust this to the desired amount
    seconds_to_remove_end = 0.5  # Adjust this to the desired amount
    
    # Call the function to edit and filter the WAV file and save the left and right channels
    edit_wav_file(input_file_path12, output_file_path12, m1_L_output_path, m2_R_output_path, seconds_to_remove_start, seconds_to_remove_end)
    
    # Play the edited and filtered WAV file (you can modify this to play M1 or M2)
    play_wav_file(output_file_path12)

    #time.sleep(0.01)

    edit_wav_file(input_file_path34, output_file_path34, m3_L_output_path, m4_R_output_path, seconds_to_remove_start, seconds_to_remove_end)

    play_wav_file(output_file_path34)
