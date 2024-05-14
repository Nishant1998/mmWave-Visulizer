import time


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.sum = None
        self.count = None
        self.avg = None
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class MaxMeter(object):
    """Computes and stores the maximum and current value"""
    def __init__(self):
        self.max = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.max = 0

    def update(self, val):
        self.val = val
        self.max = max(self.max, val)


class Meter(object):
    """Computes and stores the values over the epochs"""
    def __init__(self):
        self.values = None
        self.reset()

    def reset(self):
        self.values = []

    def update(self, val):
        self.values.append(val)


class Counter(object):
    """Keeps track of the number of times an event occurs"""
    def __init__(self):
        self.count = 0

    def reset(self):
        self.count = 0

    def increment(self):
        self.count += 1


class CountDown(object):
    """Keeps track of the number of times an event occurs"""

    def __init__(self, n):
        self.n = n
        self.count = n

    def reset(self):
        self.count = self.n

    def set(self, n):
        self.n = n
        self.count = self.n

    def decrement(self):
        self.count -= 1


class ValueList:
    def __init__(self, n=1):
        self.value_lists = [[] for _ in range(n)]

    def add_value(self, value, list_index=0):
        self.value_lists[list_index].append(value)

    def get_values(self, list_index=0):
        return self.value_lists[list_index]

class TimeInterval:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def get_elapsed_time(self):
        if self.start_time is None or self.end_time is None:
            return None

        elapsed_seconds = int(self.end_time - self.start_time)
        hours = elapsed_seconds // 3600
        minutes = (elapsed_seconds % 3600) // 60
        seconds = elapsed_seconds % 60

        time_str = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
        return time_str

