import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class ReactiveFollowGap(Node):
    """
    Implement Follow the Gap on the car (conservative version for levine_obs).
    Keeps the original skeleton structure and method names.
    """
    def __init__(self):
        super().__init__('reactive_node')

        # Topics
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # --- knobs (conservative on purpose) ---
        self.max_range = 3.0

        # Use a wide forward view, but not full 270
        self.fov_deg = 170.0

        # Smooth helps reduce “grazing” decisions
        self.smoothing_window = 7

        # Steering + smoothing
        self.steer_limit = 0.42
        self.prev_steering = 0.0
        self.alpha = 0.45 #0.30

        # Debug throttle
        self._cb_count = 0
        self._debug_every = 15  # print every N scans (watch mode=TIP/GAP/FORK)

        self.prev_best_i = None
        self.safety_dist = 0.85
        self.car_half_width = 0.22
        self._fork_lock = 0
        self._fork_dir = 0
        self._fork_cooldown = 0
        self._prev_forward_clear = 0.0
        self._force_turn = False
        self._obstacle_turn_hold = 0   # hold forced turn for N frames so the car actually executes it
        self._obstacle_turn_dir = 0.0  # -0.42 or +0.42

        # Subscribe / publish
        self.subscription = self.create_subscription(
            LaserScan, lidarscan_topic, self.lidar_callback, 10
        )
        self.publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        self.get_logger().info('Reactive Follow Gap node initialized (conservative)')

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        proc = np.array(ranges, dtype=np.float32)

        proc[~np.isfinite(proc)] = self.max_range
        proc[proc <= 0.0] = self.max_range

        proc[proc < 0.08] = 0.0
        proc = np.clip(proc, 0.0, self.max_range)

        if self.smoothing_window > 1:
            k = np.ones(self.smoothing_window, dtype=np.float32) / self.smoothing_window
            proc = np.convolve(proc, k, mode='same')

        return proc

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        free_space = np.array(free_space_ranges, dtype=np.float32)

        best_len = 0
        best_start = 0
        curr_start = None

        for i, val in enumerate(free_space):
            if val > 0.0:
                if curr_start is None:
                    curr_start = i
            else:
                if curr_start is not None:
                    length = i - curr_start
                    if length > best_len:
                        best_len = length
                        best_start = curr_start
                    curr_start = None

        # handle gap reaching the end
        if curr_start is not None:
            length = len(free_space) - curr_start
            if length > best_len:
                best_len = length
                best_start = curr_start

        if best_len == 0:
            n = len(free_space)
            return n // 3, 2 * n // 3

        return best_start, best_start + best_len - 1

    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
        Naive: Choose the furthest point within ranges and go there
        """
        if end_i <= start_i:
            return (start_i + end_i) // 2
       
        seg = np.array(ranges[start_i:end_i + 1], dtype=np.float32)
        if seg.size == 0:
            return (start_i + end_i) // 2
       
        W = int(np.clip(seg.size * 0.15, 7, 51))
        kernel = np.ones(W, dtype=np.float32) / W
        avg = np.convolve(seg, kernel, mode='same')

        best_local = int(np.argmax(avg))
        return start_i + best_local

    def lidar_callback(self, data):
        """Compact repo-style Follow-the-Gap callback (minimal heuristics)."""
        self._force_turn = False
        if self._obstacle_turn_hold > 0:
            self._obstacle_turn_hold -= 1
        proc = self.preprocess_lidar(data.ranges)
        n = len(proc)
        angles = data.angle_min + np.arange(n) * data.angle_increment

        def idx_for_angle(a_rad: float) -> int:
            i = int(round((a_rad - data.angle_min) / data.angle_increment))
            return int(np.clip(i, 0, n - 1))

        # --- FOV ---
        fov = np.deg2rad(self.fov_deg)
        front_mask = (angles > -fov / 2.0) & (angles < fov / 2.0)
        front = proc.copy()
        front[~front_mask] = 0.0

        # quick safety: drop tiny noisy points (mostly redundant w/ preprocess, but ok)
        front[front < 0.08] = 0.0

        if not np.any(front > 0.0):
            return

        # --- closest point in front + bubble ---
        valid = front > 0.0
        closest_i = int(np.argmin(np.where(valid, front, np.inf)))
        closest = float(front[closest_i])

        free = front.copy()
        bubble = int(np.clip(18 + 60 * (0.8 / max(closest, 0.2)), 22, 65))
        s = max(0, closest_i - bubble)
        e = min(n - 1, closest_i + bubble)
        free[s:e + 1] = 0.0

        # slow down if forward is not clear (uses front after bubble)
        narrow = np.deg2rad(14.0)
        forward_mask = front_mask & (angles > -narrow) & (angles < narrow)
        forward_vals = front[forward_mask & (front > 0.0)]
        forward_clear = float(np.percentile(forward_vals, 30)) if forward_vals.size else self.max_range
        # Closest in a slightly wider forward cone (±24°): used only for speed, so side walls don't limit straights.
        forward_cone = np.deg2rad(24.0)
        forward_cone_mask = front_mask & (angles > -forward_cone) & (angles < forward_cone) & (front > 0.0)
        forward_closest = float(np.min(front[forward_cone_mask])) if np.any(forward_cone_mask) else self.max_range
        left_35 = float(proc[idx_for_angle(+np.deg2rad(35.0))])
        right_35 = float(proc[idx_for_angle(-np.deg2rad(35.0))])
        left_side = float(proc[idx_for_angle(+np.deg2rad(90.0))])
        right_side = float(proc[idx_for_angle(-np.deg2rad(90.0))])

        # --- gap + best point ---
        gap_start, gap_end = self.find_max_gap(free)

        mid = (gap_start + gap_end) // 2
        if self.prev_best_i is None:
            self.prev_best_i = mid
       
        jump = abs(mid - self.prev_best_i)
        if jump > 180:
            mid = int(0.6 * mid + 0.4 * self.prev_best_i)

        best_i_smooth = self.find_best_point(gap_start, gap_end, free)

        seg_front = front[gap_start:gap_end + 1]
        best_i_far = gap_start + int(np.argmax(seg_front)) if seg_front.size else (gap_start + gap_end) // 2

        gap_len = gap_end - gap_start
        forkish = (forward_clear > 1.6) and (closest > 0.55) #gap_len > int(0.45 * np.count_nonzero(front_mask))

        if forkish:
            def local_depth(i):
                a = max(gap_start, i - 8)
                b = min(gap_end, i + 8)
                local = front[a:b+1]
                local = local[local > 0.0]
                return float(np.percentile(local, 40)) if local.size else 0.0
           
            best_i = best_i_far if local_depth(best_i_far) > local_depth(best_i_smooth) else best_i_smooth
        else:
            best_i = best_i_smooth

        # mild center bias to avoid extreme last-second swings
        mid = (gap_start + gap_end) // 2
        best_i = int(0.55 * best_i + 0.45 * mid)

        best_i = int(np.clip(best_i, gap_start, gap_end))
        a = max(gap_start, best_i - 2)
        b = min(gap_end, best_i + 2)
        if np.count_nonzero(free[a:b+1] > 0.7) <= 1:
            best_i = (gap_start + gap_end) // 2

        self.prev_best_i = best_i

        gap_angle = data.angle_min + best_i * data.angle_increment

        if forward_clear < 1.2:
            # In tight turns allow more steering to not go wide. Trigger earlier (min < 0.62).
            limit = 0.42 if min(left_35, right_35) < 0.62 else 0.35
            gap_angle = float(np.clip(gap_angle, -limit, limit))

        if closest < 0.40 and forward_clear > 1.5:
            gap_angle = float(np.clip(gap_angle, -0.25, 0.25))

        side_diff = right_side - left_side

        # --- Mode: TIP, FORK, or NORMAL. Priority avoids tip vs fork conflict. ---
        # TIP: only when BOTH sides similarly tight (point obstacle). First corner has one ~0.45 one ~0.30 -> exclude.
        tip_mode = (
            (forward_clear > 1.9) and
            (closest < 0.65) and
            (left_side < 0.62) and
            (right_side < 0.62) and
            ((max(left_side, right_side) - min(left_side, right_side)) < 0.14)
        )
        fork_ok = (forward_clear > 1.7) and (closest > 0.58) and (abs(side_diff) > 0.35)
        fork_detected = (not tip_mode) and fork_ok and (abs(gap_angle) < 0.35)
        # Approach tip: only when BOTH sides similarly tight (point obstacle). First corner = one side open -> exclude.
        approach_tip = (
            (not tip_mode) and (not fork_ok) and (forward_clear > 1.8) and (closest < 0.76) and
            (min(left_side, right_side) < 0.86) and (max(left_side, right_side) < 1.25) and
            ((max(left_side, right_side) - min(left_side, right_side)) < 0.38)
        )

        if self._fork_lock > 0:
            self._fork_lock -= 1
        if self._fork_cooldown > 0:
            self._fork_cooldown -= 1

        left_mask = front_mask & (angles >= np.deg2rad(15.0)) & (angles <= np.deg2rad(70.0)) & (front > 0.0)
        right_mask = front_mask & (angles >= np.deg2rad(-70.0)) & (angles <= np.deg2rad(-15.0)) & (front > 0.0)
        lv = front[left_mask]
        rv = front[right_mask]
        lb_lo = float(np.percentile(lv, 25)) if lv.size else 0.0
        rb_lo = float(np.percentile(rv, 25)) if rv.size else 0.0
        lb_hi = float(np.percentile(lv, 65)) if lv.size else 0.0
        rb_hi = float(np.percentile(rv, 65)) if rv.size else 0.0
        left_score = 0.75 * lb_lo + 0.25 * lb_hi
        right_score = 0.75 * rb_lo + 0.25 * rb_hi

        if tip_mode:
            # TIP only: steer away from the closer side. Commit more for margin (0.28, ±0.40).
            if left_side < right_side:
                gap_angle = float(gap_angle - 0.28)
            else:
                gap_angle = float(gap_angle + 0.28)
            gap_angle = float(np.clip(gap_angle, -0.40, 0.40))

        elif self._fork_lock > 0:
            # FORK lock active: keep committed direction (no flipping). Steer in chosen fork_dir.
            gap_angle = float(np.clip(self._fork_dir * 0.35, -0.42, 0.42))
        elif fork_detected and (self._fork_lock == 0) and (self._fork_cooldown == 0):
            # Enter fork: blend scores with side_diff (90° clearance) so choice is more consistent.
            # side_diff > 0 means right more open. Weight it so we don't get fooled by dead-end score.
            side_weight = 0.4 * np.clip(side_diff, -0.6, 0.6)
            effective_right = right_score + side_weight
            effective_left = left_score - side_weight
            self._fork_dir = -1 if effective_right > effective_left else +1
            self._fork_lock = 35
            self._fork_cooldown = 30
            gap_angle = float(np.clip(self._fork_dir * 0.35, -0.42, 0.42))
        else:
            # NORMAL: mild bias when one side is closer, open-track clip when safe.
            self._fork_dir = 0
            self._force_turn = False
            # Obstacle in front with one side open (first corner). Logs: first corner has ~0.45 and ~0.30.
            # When one side is very open (max > 0.90) we're passing (e.g. last rectangle) — skip full sharp turn
            # so we don't oversteer into the tip; clearance nudge / bias handle it gently.
            obstacle_one_side_open = (
                (not fork_ok) and (closest < 1.1) and
                (min(left_35, right_35) < 0.52) and
                (max(left_35, right_35) > 0.42) and
                (max(left_35, right_35) <= 0.90)
            )
            if obstacle_one_side_open:
                self._force_turn = True
                turn = 0.42
                self._obstacle_turn_dir = -turn if left_35 < right_35 else turn
                self._obstacle_turn_hold = 25
                gap_angle = self._obstacle_turn_dir
            elif approach_tip:
                away = 0.15 if left_side < right_side else -0.15
                gap_angle = float(gap_angle + away)
            # 2nd triangle: when one 90° side is clearly more open (wall) and one closer (hypotenuse), bias toward
            # the wall early. Run even when fork_ok so we don't get fork center bias pulling toward tip.
            second_triangle_geom = (
                (min(left_35, right_35) >= 0.48) and
                (forward_clear > 1.55) and (0.52 < closest < 1.02) and
                (max(left_side, right_side) - min(left_side, right_side) > 0.18)
            )
            if second_triangle_geom:
                toward_wall = 0.20 if left_side < right_side else -0.20
                gap_angle = float(np.clip(gap_angle + toward_wall, -0.36, 0.36))
            # Wider line in corner: only when NOT at fork, and not in a tight corner (closest > 0.68).
            elif (not fork_ok) and (forward_clear > 1.4) and (0.68 < closest < 1.15) and (abs(side_diff) > 0.45):
                gap_angle += -0.07 if right_side > left_side else 0.07
            # Clearance nudge when passing close to a large obstacle (e.g. big rectangle at end):
            # one side close, one side clearly open — nudge toward open side. Exclude tip zone (2nd triangle).
            elif (not fork_ok) and (closest < 0.55) and (min(left_35, right_35) < 0.65) and \
                 (max(left_35, right_35) - min(left_35, right_35) > 0.28) and (max(left_35, right_35) > 0.88) and \
                 (forward_clear >= 1.15 or closest >= 0.62):
                nudge = 0.08
                gap_angle += -nudge if left_35 < right_35 else nudge
            elif min(left_35, right_35) < 0.82:
                # Reduced bias to prevent wobbling on straights
                bias = 0.12 if min(left_35, right_35) < 0.60 else 0.10
                gap_angle += (-bias if left_35 < right_35 else + bias)
            # Fork approach: gentle center damping to reduce wobble before fork (exclude 2nd triangle zone).
            if fork_ok and (0.70 < closest < 0.95) and not second_triangle_geom:
                gap_angle = float(gap_angle * 0.82)
            if (forward_clear > 2.0) and (closest > 0.9):
                gap_angle = float(np.clip(gap_angle, -0.20, 0.20))
            elif (forward_clear > 1.6) and (closest > 0.72):
                gap_angle = float(np.clip(gap_angle, -0.20, 0.20))

        # --- steering ---
        if self._obstacle_turn_hold > 0:
            steering = float(np.clip(self._obstacle_turn_dir, -self.steer_limit, self.steer_limit))
        else:
            target_steer = float(np.clip(gap_angle, -self.steer_limit, self.steer_limit))
            if self._force_turn:
                steering = target_steer
            else:
                max_step = 0.06
                target_steer = float(np.clip(target_steer, self.prev_steering - max_step, self.prev_steering + max_step))
                steering = self.alpha * target_steer + (1.0 - self.alpha) * self.prev_steering
        steering = float(np.clip(steering, -self.steer_limit, self.steer_limit))
        self.prev_steering = steering

        # speed: fast on straights, slower in corners
        steering_abs = abs(steering)
        # Base speeds exactly as in your working version (conservative but stable).
        if steering_abs > 0.30:
            speed = 0.45
        elif steering_abs > 0.15:
            speed = 0.70
        else:
            speed = 1.00

        # Boost straights a bit using forward_closest ONLY when truly in a clear, low-steering straight.
        # Use the forward cone only (ignore side walls), since 'closest' is often limited by corridor width.
        # Allow some wobble: treat it as a straight as long as steering stays fairly small.
        if (steering_abs < 0.18) and (forward_closest > 1.7) and (not tip_mode):
            speed = 1.12

        if forward_clear < 0.9:
            speed = min(speed, 0.55)

        if forward_closest < 0.90:
            speed = min(speed, 0.70)
        if forward_closest < 0.70:
            speed = min(speed, 0.55)
        if forward_closest < 0.55:
            speed = min(speed, 0.45)
        if tip_mode:
            speed = min(speed, 0.28)
        if self._obstacle_turn_hold > 0:
            speed = max(speed, 0.50)  # enough speed to actually execute the turn

        # publish
        msg = AckermannDriveStamped()
        msg.drive = AckermannDrive()
        msg.drive.steering_angle = steering
        msg.drive.speed = float(speed)
        self.publisher.publish(msg)

        # debug
        self._cb_count += 1
        if self._cb_count % self._debug_every == 0:
            mode_str = "TIP" if tip_mode else ("FORK" if self._fork_lock > 0 else "GAP")
            extra = " approach_tip" if (not tip_mode and approach_tip) else ""
            self.get_logger().info(
                f"mode={mode_str}{extra} closest={closest:.2f} fwd={forward_clear:.2f} "
                f"steer={steering:.3f} spd={speed:.2f} left={left_side:.2f} right={right_side:.2f} "
                f"fork_lock={self._fork_lock}"
            )

def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    node = ReactiveFollowGap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()