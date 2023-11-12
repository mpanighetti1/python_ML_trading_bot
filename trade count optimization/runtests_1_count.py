import os
import subprocess
import time
import win32con
import win32process

def seconds_to_hms(seconds):
    """Convert seconds to hours, minutes, and seconds."""
    h = int(seconds // 3600)
    seconds %= 3600
    m = int(seconds // 60)
    s = int(seconds % 60)
    return h, m, s

def set_priority(pid=None, priority=1):
    """ Set The Priority of a Process.  Priority is a value between 0-5 where
        2 is normal priority.  Default is 2 (normal priority). You can set it to anything:
        ABOVE_NORMAL_PRIORITY_CLASS = 32768
        BELOW_NORMAL_PRIORITY_CLASS = 16384
        HIGH_PRIORITY_CLASS = 128
        IDLE_PRIORITY_CLASS = 64
        NORMAL_PRIORITY_CLASS = 32
        REALTIME_PRIORITY_CLASS = 256
    """
    if pid == None:
        pid = os.getpid()
    priority_classes = [win32process.IDLE_PRIORITY_CLASS,
                        win32process.BELOW_NORMAL_PRIORITY_CLASS,
                        win32process.NORMAL_PRIORITY_CLASS,
                        win32process.ABOVE_NORMAL_PRIORITY_CLASS,
                        win32process.HIGH_PRIORITY_CLASS,
                        win32process.REALTIME_PRIORITY_CLASS]
    win32process.SetPriorityClass(pid, priority_classes[priority])

num_tests = 100  # Number of times you want to run the tests

for i in range(num_tests):
    
    # Get the current time before starting the process
    start_time = time.time()

    command = ["python", "lps_1_count.py"]
    # For the second run onwards, pass the saved variables as input
    if i > 0:
        command.append("saved_variables_count_1.json")

    # Here we set the creationflags to create a new process with higher priority
    process = subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | 32)
    process.communicate() # Wait for process to complete

    # Get the current time after the process has finished
    end_time = time.time()

    # Calculate the total time taken
    time_taken = end_time - start_time

    h, m, s = seconds_to_hms(time_taken)
    print(f"Total time taken: {h} hours, {m} minutes, {s} seconds")


