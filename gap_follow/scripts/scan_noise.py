#!/usr/bin/env python3
# Simple ROS2 Foxy script:
# Subscribes to /scan
# Adds Gaussian noise to ranges
# Publishes to /scan_noisy

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import argparse
import sys


class SimpleScanNoise(Node):

    def __init__(self, range_std):
        super().__init__('scan_noise_simple')

        self.range_std = range_std
        self.rng = np.random.default_rng()

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.publisher = self.create_publisher(
            LaserScan,
            '/scan_noisy',
            10
        )

        self.get_logger().info(
            f'Noise node started. range_std = {self.range_std}'
        )

    def scan_callback(self, msg):
        noisy_msg = LaserScan()

        # Copy metadata
        noisy_msg.header = msg.header
        noisy_msg.angle_min = msg.angle_min
        noisy_msg.angle_max = msg.angle_max
        noisy_msg.angle_increment = msg.angle_increment
        noisy_msg.time_increment = msg.time_increment
        noisy_msg.scan_time = msg.scan_time
        noisy_msg.range_min = msg.range_min
        noisy_msg.range_max = msg.range_max

        # Convert ranges to numpy
        ranges = np.array(msg.ranges)

        # Add Gaussian noise
        noise = self.rng.normal(0.0, self.range_std, size=ranges.shape)
        ranges = ranges + noise

        # Clip to valid sensor limits
        ranges = np.clip(ranges, msg.range_min, msg.range_max)

        noisy_msg.ranges = ranges.tolist()
        noisy_msg.intensities = msg.intensities

        self.publisher.publish(noisy_msg)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--range-std', type=float, default=0.05,
                        help='Standard deviation of Gaussian noise (meters)')
    parsed_args = parser.parse_args()

    rclpy.init()
    node = SimpleScanNoise(parsed_args.range_std)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()