import threading
import time
from typing import Any

#CPU bound task
def count(n: int) -> None:
    total: int = 0
    for i in range(n):
        total +=i

# I/O bound task the GIL relleases during sleep
def download(file_id: int) -> None:
    print(f"downloading file {file_id}...")
    time.sleep(1) # simulates waiting for network
    print(f"file {file_id} done")
print()

# Sequential - no threads
print("sequential downloads: ")
start: float = time.time()
for i in range(3):
    download(i)
end: float = time.time()
print(f"sequential took {end - start: .2f} seconds\n") #f for float, .2 for 2 decimal places

#Threaded - runs simultaneously
print("threaded downloads:")
start =  time.time()
threads: list[threading.Thread] = []

for i in range(3): 
    t = threading.Thread(target=download, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

end = time.time()
print(f"threaded took {end - start: .2f}")

