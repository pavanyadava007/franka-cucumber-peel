"""Stand-in for rclpy.clock (unused offline)."""
import enum


class ClockType(enum.Enum):
    ROS_TIME = 1
    SYSTEM_TIME = 2
    STEADY_TIME = 3
