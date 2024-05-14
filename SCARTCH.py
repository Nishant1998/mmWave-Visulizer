import threading

def fuc1(i):
    # Your function code here
    print("This is fuc1() running in a thread with i =", i)

# Create and start 10 threads
threads = []
for i in range(10):
    thread = threading.Thread(target=fuc1, args=(i,))
    thread.start()
    threads.append(thread)

    # print(f"{i} *** ")

# Other large code here ...
