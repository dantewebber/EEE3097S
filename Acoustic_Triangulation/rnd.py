import time
import paramiko

# Replace with the actual hostnames and SSH credentials of your Raspberry Pi devices
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

# Send files to master laptop
pi1_stdin, pi1_stdout, pi1_stderr = pi1_ssh.exec_command(end_command1)
pi2_stdin, pi2_stdout, pi2_stderr = pi2_ssh.exec_command(end_command1)

while True:
    pi1_exit_status = pi1_stdout.channel.recv_exit_status()
    pi2_exit_status = pi2_stdout.channel.recv_exit_status()
    if pi1_exit_status == 0 and pi2_exit_status == 0:
        break

# Send command to start audio file reading here

pi1_ssh.exec_command(end)
pi2_ssh.exec_command(end)



# Close the SSH connections
pi1_ssh.close()
pi2_ssh.close()
