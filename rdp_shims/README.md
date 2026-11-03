Minimal stand-ins for ROS 2 message modules that RDP imports at module level (space_utils.py, time_utils.py)
but never calls during offline training or simulation. Only on PYTHONPATH for training/eval; on the real robot,
source ROS 2 instead and do not use these.
