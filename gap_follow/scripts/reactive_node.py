import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class ReactiveFollowGap(Node):
    """ 
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('reactive_node')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        self.max_range = 3.0
        self.bubble_radius = 10.0
        self.smoothing_window = 3
        self.prev_steering = 0.0

        # TODO: Subscribe to LIDAR
        self.subscription = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)

        # TODO: Publish to drive
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        self.get_logger().info('Reactive Follow Gap node initialized')

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """

        ranges = np.array(ranges, dtype = np.float32)

        #replace inf, nan, and negative values
        ranges[~np.isfinite(ranges)] = self.max_range
        ranges[ranges <= 0.0] = self.max_range

        #smooth moving average
        if self.smoothing_window > 1:
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            ranges = np.convolve(ranges, kernel, mode = 'same')

        #clip to [0, max_range]
        ranges = np.clip(ranges, 0.0, self.max_range)

        proc_ranges = ranges
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        return None
    
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        return None

    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        ranges = data.ranges
        proc_ranges = self.preprocess_lidar(ranges)
        
        # TODO:
        #Find closest point to LiDAR

        #Eliminate all points inside 'bubble' (set them to zero) 

        #Find max length gap 

        #Find the best point in the gap 

        #Publish Drive message


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()