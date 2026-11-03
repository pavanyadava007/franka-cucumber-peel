"""Stand-in for rclpy.time (unused offline)."""


class Time:
    def __init__(self, seconds=0, nanoseconds=0, clock_type=None):
        self.nanoseconds = int(seconds * 1e9) + int(nanoseconds)
