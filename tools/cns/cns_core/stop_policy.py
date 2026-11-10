# ruff: noqa  (vendored from CNS @ 6b42138, MIT)
# From CNS (https://github.com/hhcaz/CNS @ 6b42138), cns/benchmark/stop_policy.py, MIT, Copyright (c) 2023 hhcaz.
# Changed: a progress print removed; the SSIM policy (scikit-image metrics) and a commented-out variant are dropped.
import numpy as np


class ErrorHoldingStopPolicy:
    def __init__(self, waiting_time, conduct_thresh):
        self.waiting_time = waiting_time
        self.conduct_thresh = conduct_thresh
        self.min_err = None
        self.cur_err = float("inf")
        self.prev_t = None
        self.prev_err = None
        self.accum_time = 0

    def reset(self):
        self.min_err = None
        self.prev_t = None
        self.prev_err = None
        self.accum_time = 0

    def calculate_error(self, data):
        raise NotImplementedError

    def __call__(self, data, timestamp: float) -> bool:
        if data is None:
            self.reset()
            self.cur_err = float("inf")
            return False

        cur_err = self.calculate_error(data)
        self.cur_err = cur_err

        if cur_err > self.conduct_thresh:
            self.reset()
            return False

        if self.min_err is None or cur_err < 0.9 * self.min_err:
            self.min_err = cur_err
            self.prev_t = timestamp
            self.prev_err = cur_err
            self.accum_time = 0
            return False

        thresh = 1.1 * self.min_err + 5e-4
        # thresh = 1.1 * self.min_err + 2e-4
        a, b = self.prev_err, cur_err
        t0, t1 = self.prev_t, timestamp
        self.prev_t = timestamp
        self.prev_err = cur_err

        if a < thresh and thresh <= b:
            dt = (thresh - a) / (b - a) * (t1 - t0)
        elif a >= thresh and thresh > b:
            dt = (thresh - b) / (a - b) * (t1 - t0)
        elif a <= thresh and b <= thresh:
            dt = t1 - t0
        else:
            dt = 0
            # dt = (t1 - t0) * 0.01  # avoid waiting for too much time
        self.accum_time += dt

        return self.accum_time > self.waiting_time


class PixelStopPolicy(ErrorHoldingStopPolicy):
    def __init__(self, waiting_time, conduct_thresh=0.01):
        super().__init__(waiting_time, conduct_thresh)
        self.weight = None
        self.witness_count = None
        self.decay = 0.95
        self.maximum = 1.0 / (1 - self.decay)  # solve from: x = decay * x + 1

    def reset(self):
        if isinstance(self.weight, np.ndarray):
            self.weight.fill(0)
            self.witness_count.fill(0)
        return super().reset()

    def calculate_error(self, data):
        delta_pos = (data.pos_cur - data.pos_tar).cpu().numpy()
        node_mask = data.node_mask.cpu().numpy()

        if self.weight is None or len(self.weight) != len(node_mask):
            # initialize weight and witness count
            self.weight = np.zeros(len(node_mask), dtype=np.float32)
            self.witness_count = np.zeros(len(node_mask), dtype=np.int64)

        pre_is_0 = self.witness_count == 0
        self.witness_count[node_mask] += 1
        now_is_1 = self.witness_count == 1

        mask = pre_is_0 & now_is_1  # first witness mask
        self.weight[mask] = self.maximum  # set weight of first witness point to maximum
        self.weight[~mask] = self.decay * self.weight[~mask] + node_mask[~mask]

        weight = self.weight[node_mask]
        delta_pos = np.linalg.norm(delta_pos[node_mask], axis=-1)

        error = weight * delta_pos
        mask = error < np.percentile(error, 90)
        weight = weight[mask]
        weight = weight / (weight.sum() + 1e-7)
        mean_err = np.sum(weight * delta_pos[mask])

        return mean_err
