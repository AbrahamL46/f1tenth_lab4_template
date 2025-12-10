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
        self.bubble_radius = 40
        self.smoothing_window = 3
        self.prev_steering = 0.0
        self.left_bias_angle = 0.05 #(~3 degrees left steering bias)

        # TODO: Subscribe to LIDAR
        self.subscription = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)

        # TODO: Publish to drive
        self.publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        self.get_logger().info('Reactive Follow Gap node initialized')

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """

        proc_ranges = np.array(ranges, dtype = np.float32)

        #replace inf, nan, and negative values
        proc_ranges[~np.isfinite(proc_ranges)] = self.max_range
        proc_ranges[proc_ranges <= 0.0] = self.max_range

        #smooth moving average
        if self.smoothing_window > 1:
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            proc_ranges = np.convolve(proc_ranges, kernel, mode = 'same')

        #clip to [0, max_range]
        proc_ranges = np.clip(proc_ranges, 0.0, self.max_range)
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """

        free_space_ranges = np.array(free_space_ranges, dtype = np.float32)

        max_len = 0
        max_start = 0
        curr_start = None

        for i, val in enumerate(free_space_ranges):
            if val > 0.0:
                if curr_start is None:
                    curr_start = i
            else:
                if curr_start is not None:
                    length = i - curr_start
                    if length > max_len:
                        max_len = length
                        max_start = curr_start
                    curr_start = None
        
        #gap going towards end
        if curr_start is not None:
            length = len(free_space_ranges) - curr_start
            if length > max_len:
                max_len = length
                max_start = curr_start

        #fallback aiming forward
        if max_len == 0:
            n = len(free_space_ranges)
            return n // 3, 2 * n // 3
        
        max_end = max_start + max_len - 1
        return max_start, max_end
    
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """

        ranges = np.array(ranges, dtype = np.float32)
        if end_i < start_i:
            return (start_i + end_i) // 2   #return midpoint
        
        gap = ranges[start_i:end_i + 1]
        if len(gap) == 0:
            return (start_i + end_i) // 2
        
        #index of straights in array
        mid_index = (len(ranges) - 1) / 2.0
        best_index = (start_i + end_i) // 2
        best_score = -np.inf

        angle_weight_left = 0.01
        angle_weight_right = 0.03
        
        for idx in range(start_i, end_i + 1):
            r = ranges[idx]
            if r <= 0.0:
                continue
            
            if idx < mid_index:
                angle_penalty = angle_weight_right * abs(idx - mid_index)
            else:
                angle_penalty = angle_weight_left * abs(idx - mid_index)

            score = r - angle_penalty
            if score > best_score:
                best_score = score
                best_index = idx

        return best_index

    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        ranges = np.array(data.ranges, dtype = np.float32)
        proc_ranges = self.preprocess_lidar(ranges)

        num_ranges = len(proc_ranges)

        angles = data.angle_min + np.arange(num_ranges) * data.angle_increment
        fov = np.deg2rad(110.0)
        front_mask = (angles > -fov) & (angles < fov)

        front_ranges = proc_ranges.copy()
        front_ranges[~front_mask] = 0.0

        valid = front_ranges > 0.0
        if not np.any(valid):
            drive_msg = AckermannDriveStamped()
            drive_msg.drive = AckermannDrive()
            drive_msg.drive.steering_angle = 0.0
            drive_msg.drive.speed = 1.0
            self.publisher.publish(drive_msg)
            return

        # TODO:
        #Find closest point to LiDAR
        safety_dist = 1.0
        free_space = front_ranges.copy()
        free_space[free_space < safety_dist] = 0.0

        #angle space sectors
        right_sector = (angles > -np.deg2rad(110)) & (angles < -np.deg2rad(20))
        center_sector = (angles >= -np.deg2rad(20)) & (angles <= np.deg2rad(20))
        left_sector = (angles > np.deg2rad(20)) & (angles < np.deg2rad(110))

        #apply validity mask
        right_mask = right_sector & front_mask & (free_space > 0.0)
        center_mask = center_sector & front_mask & (free_space > 0.0)
        left_mask = left_sector & front_mask & (free_space > 0.0)

        #compute total free distance for each sector
        right_score = np.sum(free_space[right_mask]) if np.any(right_mask) else 0.0
        center_score = np.sum(free_space[center_mask]) if np.any(center_mask) else 0.0
        left_score = np.sum(free_space[left_mask]) if np.any(left_mask) else 0.0

        #penalize right sector
        right_score *= 0.7

        #choose best sector
        scores = {'right': right_score, 'center': center_score, 'left': left_score}
        best_sector = max(scores, key=scores.get)

        if best_sector == 'left' and np.any(left_mask):
            candidate_indices = np.where(left_mask)[0]
        elif best_sector == 'right' and np.any(right_mask):
            candidate_indices = np.where(right_mask)[0]
        elif np.any(center_mask):
            candidate_indices = np.where(center_mask)[0]
        else:
            candidate_indices = np.where(free_space > 0.0)[0]

        #aim for furthest point chosen in sector
        if len(candidate_indices) > 0:
            if best_sector == 'left':
                best_point = int(candidate_indices[-1])
            elif best_sector == 'right':
                best_point = int(candidate_indices[0])
            else:
                sector_ranges = free_space[candidate_indices]
                best_index_in_sector = int(np.argmax(sector_ranges))
                best_point = int(candidate_indices[best_index_in_sector])
        else:
            best_point = num_ranges // 2

        best_angle = data.angle_min + best_point * data.angle_increment

        alpha = 0.3
        steering_angle = alpha * best_angle + (1.0 - alpha) * self.prev_steering
        self.prev_steering = steering_angle

        #set speed based on steering magnitude
        abs_angle = abs(steering_angle)
        if abs_angle > 0.4:
            speed = 0.8
        elif abs_angle > 0.2:
            speed = 1.4
        else:
            speed = 2.0

        #Publish Drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive = AckermannDrive()
        drive_msg.drive.steering_angle = float(steering_angle)
        drive_msg.drive.speed = float(speed)

        self.publisher.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()