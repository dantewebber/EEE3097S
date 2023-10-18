import subprocess
import concurrent.futures

def send_file_to_raspberry_pi(pi_host):
    command = f"scp {target_file} {pi_host}:{destination}"

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result, error = process.communicate()

    if process.returncode != 0:
        print(f'Failed to send file to {pi_host}. Error: {error.decode()}')
    else:
        print(f'Successfully sent file to {pi_host}.')

# Pi hostnames or IP addresses
raspberry_pis = ["pi@raspberrypi.local", "pi@raspberrypi1.local"] 

# Target file and target location
target_file = "start.txt"
destination = "/home/pi/"

# Send file to both Raspberry Pis
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(send_file_to_raspberry_pi, raspberry_pis)