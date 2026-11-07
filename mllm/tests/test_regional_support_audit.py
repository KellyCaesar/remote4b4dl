import unittest

import numpy as np

from scripts.audit_regional_v2_support import surviving_outside_mask


class RegionalSupportAuditTests(unittest.TestCase):
    def test_only_outside_points_in_surviving_clamped_cells_are_counted(self):
        points = np.array([[.1, -19.9, 0., 1.], [-.2, -21., 5., 1.],
                           [40., 20., 4., 1.], [20., 21., 0., 1.]], np.float32)
        mask, ids = surviving_outside_mask(points, np.array([0, 6399]))
        np.testing.assert_array_equal(ids, [0, 0, 6399, 6360])
        np.testing.assert_array_equal(mask, [False, True, True, False])

    def test_rejects_nonfinite_or_wrong_dtype(self):
        with self.assertRaises(ValueError):
            surviving_outside_mask(np.array([[0., 0., 0., 0.]], np.float64), [0])
        with self.assertRaises(ValueError):
            surviving_outside_mask(np.array([[float("nan"), 0., 0., 0.]], np.float32), [0])


if __name__ == "__main__":
    unittest.main()
