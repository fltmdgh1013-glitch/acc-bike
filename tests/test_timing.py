import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluate_timing import binary_scores, timing_masks


class TimingTests(unittest.TestCase):
    def test_threshold_is_inclusive_and_confusion_counts(self):
        m = binary_scores([400,500,399,0], [400,399,400,0])
        self.assertEqual((m['TP'],m['FP'],m['FN'],m['TN']), (1,1,1,1))
        self.assertEqual(m['precision'], .5)
        self.assertEqual(m['recall'], .5)
        self.assertEqual(m['accuracy'], .5)

    def test_always_high_is_not_perfect_accuracy(self):
        m = binary_scores([500,500,0,0], [400]*4)
        self.assertEqual(m['recall'], 1)
        self.assertEqual(m['precision'], .5)
        self.assertEqual(m['specificity'], 0)

    def test_undefined_rates_are_not_misleading_zero(self):
        m = binary_scores([0,1], [0,1])
        self.assertTrue(np.isnan(m['precision']))
        self.assertTrue(np.isnan(m['recall']))
        self.assertEqual(m['accuracy'], 1)

    def test_target_hour_and_weekday(self):
        t = pd.to_datetime(['2018-10-08 07:00','2018-10-08 09:00','2018-10-08 10:00',
                            '2018-10-08 17:00','2018-10-08 19:00','2018-10-07 07:00'])
        masks = timing_masks(t)
        np.testing.assert_equal(masks['weekday_commute'], [True,True,False,True,True,False])
        self.assertTrue(masks['morning_all_days'][-1])

    def test_bad_alignment_rejected(self):
        with self.assertRaises(ValueError):
            binary_scores([400], [400,500])
