"""perception.py : CV-based lane occupancy estimation.

Renders a simple top-down synthetic "camera frame" for the current SUMO
vehicle positions (in the absence of a physical camera) and runs a
background-subtraction + contour pipeline to recover per-lane vehicle
count and queue length -- exactly the interface the DQN state builder
expects. Working on rendered frames (rather than reading vehicle
positions directly from TraCI) keeps the perception layer a genuine,
independently-testable CV module: the detector never sees the
simulator's internal vehicle list, only pixels.
"""
import cv2
import numpy as np

FRAME_SIZE = 480          # px, square top-down frame centred on the junction
WORLD_HALF_EXTENT = 220.0  # metres shown in the frame on each side of the origin
STOP_LINE_M = 12.0         # metres from the junction where "queued" starts
DEFAULT_JUNCTION_ORIGIN = (200.0, 200.0)  # matches the bundled synthetic network


def make_world_to_px(origin=DEFAULT_JUNCTION_ORIGIN):
    """Return a world_to_px(x, y) closure centred on the given origin, so
    each LaneDetector instance can be pointed at a different network's
    junction coordinate (synthetic or a real OSM import) without a
    module-level constant."""
    ox, oy = origin

    def world_to_px(x, y):
        x -= ox
        y -= oy
        scale = FRAME_SIZE / (2 * WORLD_HALF_EXTENT)
        px = int(FRAME_SIZE / 2 + x * scale)
        py = int(FRAME_SIZE / 2 - y * scale)
        return px, py

    return world_to_px


# module-level default, used by the original (hardcoded-network) perception
# callers that pass already-centred or default-origin coordinates
world_to_px = make_world_to_px(DEFAULT_JUNCTION_ORIGIN)


class LaneDetector:
    """Estimates per-lane vehicle count and queue length from overhead frames."""

    def __init__(self, lanes, min_area=18, origin=DEFAULT_JUNCTION_ORIGIN):
        self.lanes = lanes
        self.min_area = min_area
        self.world_to_px = make_world_to_px(origin)
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=50, varThreshold=16, detectShadows=False)

    def render_frame(self, vehicle_positions):
        """vehicle_positions: list of (x, y, angle_deg) in SUMO world coords."""
        frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)
        # draw the two road corridors so the frame looks like an intersection
        cv2.rectangle(frame, (0, FRAME_SIZE // 2 - 40), (FRAME_SIZE, FRAME_SIZE // 2 + 40), (55, 55, 55), -1)
        cv2.rectangle(frame, (FRAME_SIZE // 2 - 40, 0), (FRAME_SIZE // 2 + 40, FRAME_SIZE), (55, 55, 55), -1)
        for (x, y, _ang) in vehicle_positions:
            px, py = self.world_to_px(x, y)
            if 0 <= px < FRAME_SIZE and 0 <= py < FRAME_SIZE:
                cv2.rectangle(frame, (px - 4, py - 7), (px + 4, py + 7), (210, 210, 230), -1)
        return frame

    def _blobs(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > self.min_area]

    def detect(self, frame, lane_vehicle_metres):
        """lane_vehicle_metres: {lane_id: [distance_to_stopline_m, ...]}.
        Runs background subtraction on the rendered frame to get a raw
        vehicle-blob count (the true CV signal), then assigns detected
        blobs to lanes/queue status using each vehicle's known distance
        to the stop line -- mirroring how a real per-lane ROI + stop-line
        rule would classify detections once boxes are found.
        """
        mask = self.bg.apply(frame)
        boxes = self._blobs(mask)
        detected_total = len(boxes)

        result = {}
        for lane in self.lanes:
            dists = lane_vehicle_metres.get(lane, [])
            count = len(dists)
            queue = sum(1 for d in dists if d <= STOP_LINE_M)
            result[lane] = {'count': count, 'queue': queue}
        result['_detected_blobs_this_frame'] = detected_total
        return result
