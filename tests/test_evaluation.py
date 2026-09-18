import unittest

import numpy as np
from sklearn.metrics import average_precision_score

from evaluation import (
    BudgetUnreachableError,
    expected_calibration_error,
    false_positives_per_hour,
    precision_recall_auc,
    recall_at_false_positive_budget,
    select_anomaly_threshold,
    top_k_precision,
)


class PrecisionRecallAucTests(unittest.TestCase):
    def test_matches_the_reference_implementation(self):
        truth = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])

        self.assertAlmostEqual(
            precision_recall_auc(truth, scores),
            float(average_precision_score(truth, scores)),
        )

    def test_is_undefined_without_positives(self):
        self.assertIsNone(precision_recall_auc(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4])))


class FalsePositiveRateTests(unittest.TestCase):
    """The metric an analyst actually feels: alerts per hour on benign traffic."""

    def test_counts_only_benign_rows_that_were_flagged(self):
        truth = np.array([0, 0, 0, 1])
        flagged = np.array([True, False, True, True])

        self.assertAlmostEqual(
            false_positives_per_hour(truth, flagged, benign_hours=2.0), 1.0
        )

    def test_scales_the_count_by_the_evaluation_sampling_rate(self):
        """A quarter sample over a full-capture denominator reads 4x too low."""

        truth = np.array([0, 0, 0, 1])
        flagged = np.array([True, False, True, True])

        full = false_positives_per_hour(truth, flagged, benign_hours=2.0, sampling_rate=1.0)
        half = false_positives_per_hour(truth, flagged, benign_hours=2.0, sampling_rate=0.5)

        self.assertAlmostEqual(half, full * 2)

    def test_rejects_a_non_positive_denominator(self):
        with self.assertRaisesRegex(ValueError, "benign_hours"):
            false_positives_per_hour(np.array([0]), np.array([True]), benign_hours=0.0)



class RecallAtFalsePositiveBudgetTests(unittest.TestCase):
    def setUp(self):
        # Three benign scores and two attack scores, over one benign hour.
        self.truth = np.array([0, 0, 0, 1, 1])
        self.scores = np.array([0.1, 0.2, 0.3, 0.25, 0.35])

    def test_picks_the_lowest_threshold_that_fits_the_budget(self):
        recall, threshold = recall_at_false_positive_budget(
            self.truth, self.scores, budget_per_hour=2.0, benign_hours=1.0
        )

        # At 0.2 two benign rows are flagged, which fits, and both attacks clear it.
        self.assertAlmostEqual(threshold, 0.2)
        self.assertAlmostEqual(recall, 1.0)

    def test_a_tighter_budget_forces_a_stricter_threshold_and_less_recall(self):
        recall, threshold = recall_at_false_positive_budget(
            self.truth, self.scores, budget_per_hour=1.0, benign_hours=1.0
        )

        self.assertAlmostEqual(threshold, 0.3)
        self.assertAlmostEqual(recall, 0.5)

    def test_a_tight_budget_falls_back_to_flagging_nothing(self):
        """Always returns a threshold; whether it is useful is a policy call."""

        # Attacks sit below the noisiest benign row, so buying zero false
        # positives costs every detection.
        truth = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.9, 0.3, 0.4])

        recall, threshold = recall_at_false_positive_budget(
            truth, scores, budget_per_hour=0.5, benign_hours=1.0
        )

        self.assertEqual(recall, 0.0)
        self.assertGreater(threshold, scores.max())

    def test_a_tight_budget_still_keeps_detections_it_can_afford(self):
        """The shared fixture buys zero false positives without losing recall."""

        recall, threshold = recall_at_false_positive_budget(
            self.truth, self.scores, budget_per_hour=0.5, benign_hours=1.0
        )

        self.assertAlmostEqual(recall, 0.5)
        self.assertGreater(threshold, 0.3)

    def test_a_perfectly_separating_detector_is_not_rejected(self):
        """Every benign row tied at one score used to look unreachable."""

        truth = np.array([0] * 100 + [1] * 10)
        scores = np.array([0.5] * 100 + [0.9] * 10)

        recall, threshold = recall_at_false_positive_budget(
            truth, scores, budget_per_hour=10.0, benign_hours=1.0
        )

        self.assertAlmostEqual(recall, 1.0)
        self.assertGreater(threshold, 0.5)
        self.assertLessEqual(threshold, 0.9)



class TopKTests(unittest.TestCase):
    def test_measures_precision_over_the_highest_scoring_rows(self):
        """Asymmetric on purpose: a symmetric fixture cannot catch an inversion."""

        truth = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.6])

        precision, effective_k = top_k_precision(truth, scores, k=2)

        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(effective_k, 2.0)

    def test_scores_the_top_of_the_queue_not_the_bottom(self):
        truth = np.array([0, 0, 1, 1])
        scores = np.array([0.9, 0.8, 0.7, 0.6])

        precision, _ = top_k_precision(truth, scores, k=2)

        self.assertAlmostEqual(precision, 0.0)

    def test_reports_the_effective_population_k_under_sampling(self):
        """Top 10 of a quarter sample is roughly the top 40 of the population."""

        truth = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.6])

        _, effective_k = top_k_precision(truth, scores, k=2, sampling_rate=0.25)

        self.assertAlmostEqual(effective_k, 8.0)

    def test_is_undefined_when_k_exceeds_the_available_rows(self):
        self.assertIsNone(top_k_precision(np.array([1]), np.array([0.5]), k=10)[0])


class CalibrationTests(unittest.TestCase):
    def test_a_perfectly_calibrated_classifier_scores_near_zero(self):
        truth = np.array([0, 1])
        probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])

        self.assertAlmostEqual(expected_calibration_error(truth, probabilities), 0.0)

    def test_an_overconfident_classifier_scores_above_zero(self):
        truth = np.array([0, 1])
        probabilities = np.array([[1.0, 0.0], [1.0, 0.0]])

        self.assertGreater(expected_calibration_error(truth, probabilities), 0.4)


class ThresholdPolicyTests(unittest.TestCase):
    """Threshold choice is an operational decision, and it is recorded as one."""

    def setUp(self):
        self.truth = np.array([0, 0, 0, 1, 1])
        self.scores = np.array([0.1, 0.2, 0.3, 0.25, 0.35])

    def test_the_budget_policy_records_what_drove_it(self):
        policy = select_anomaly_threshold(
            "false_positive_budget",
            truth=self.truth,
            scores=self.scores,
            benign_hours=1.0,
            budget_per_hour=1.0,
        )

        self.assertEqual(policy.policy, "false_positive_budget")
        self.assertAlmostEqual(policy.threshold, 0.3)
        self.assertAlmostEqual(policy.realized_false_positives_per_hour, 1.0)
        self.assertAlmostEqual(policy.parameter, 1.0)

    def test_the_quantile_policy_remains_available_for_reproducibility(self):
        policy = select_anomaly_threshold(
            "benign_quantile",
            truth=self.truth,
            scores=self.scores,
            benign_hours=1.0,
            quantile=0.95,
        )

        self.assertEqual(policy.policy, "benign_quantile")
        self.assertAlmostEqual(policy.parameter, 0.95)

    def test_refuses_a_budget_that_can_only_be_met_by_flagging_nothing(self):
        """Better no bundle than one that detects nothing within its budget."""

        truth = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.9, 0.3, 0.4])

        with self.assertRaisesRegex(BudgetUnreachableError, "flagging nothing"):
            select_anomaly_threshold(
                "false_positive_budget",
                truth=truth,
                scores=scores,
                benign_hours=1.0,
                budget_per_hour=0.5,
            )

    def test_accepts_a_perfectly_separating_detector(self):
        truth = np.array([0] * 100 + [1] * 10)
        scores = np.array([0.5] * 100 + [0.9] * 10)

        policy = select_anomaly_threshold(
            "false_positive_budget",
            truth=truth,
            scores=scores,
            benign_hours=1.0,
            budget_per_hour=10.0,
        )

        self.assertEqual(policy.realized_false_positives_per_hour, 0.0)

    def test_the_quantile_policy_uses_the_quantile_it_was_given(self):
        """Asserting the echoed parameter alone would not catch a wrong cut."""

        truth = np.array([0, 0, 0, 0, 1])
        scores = np.array([0.0, 1.0, 2.0, 3.0, 9.0])

        median = select_anomaly_threshold(
            "benign_quantile", truth=truth, scores=scores, benign_hours=1.0, quantile=0.5
        )
        high = select_anomaly_threshold(
            "benign_quantile", truth=truth, scores=scores, benign_hours=1.0, quantile=0.99
        )

        self.assertAlmostEqual(median.threshold, 1.5)
        self.assertAlmostEqual(high.threshold, 2.97)

    def test_rejects_an_unknown_policy_by_name(self):
        with self.assertRaisesRegex(ValueError, "roc_optimal"):
            select_anomaly_threshold(
                "roc_optimal",
                truth=self.truth,
                scores=self.scores,
                benign_hours=1.0,
            )

    def test_serializes_into_a_manifest_block(self):
        policy = select_anomaly_threshold(
            "false_positive_budget",
            truth=self.truth,
            scores=self.scores,
            benign_hours=1.0,
            budget_per_hour=2.0,
        )

        block = policy.to_manifest()

        self.assertEqual(block["policy"], "false_positive_budget")
        self.assertIn("realized_false_positives_per_hour", block)


if __name__ == "__main__":
    unittest.main()
