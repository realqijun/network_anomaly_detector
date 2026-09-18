import unittest

import numpy as np
import torch

from detector_runtime import HybridDetectorNetwork
from explain import (
    EXPLANATION_BASIS_ANOMALY,
    EXPLANATION_BASIS_CLASS,
    EXPLANATION_METHOD,
    EXPLANATION_VERSION,
    explain_rows,
)

FEATURES = ("Flow Duration", "Tot Fwd Pkts", "win1s_dst_unique_src")


def _zeroed_network(input_dim=3, num_classes=2):
    model = HybridDetectorNetwork(
        input_dim=input_dim, num_classes=num_classes, hidden_dims=(4,), latent_dim=2, dropout=0.0
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    model.eval()
    return model


def _network_routing_only(feature_index, input_dim=3, num_classes=2):
    """A network whose class logit can depend on exactly one input feature."""

    model = _zeroed_network(input_dim, num_classes)
    with torch.no_grad():
        model.encoder[0].weight[:, feature_index] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # The zeroed baseline also zeroes the LayerNorm affine weight, which
        # would multiply the whole encoder output by zero.
        model.encoder[1].weight.fill_(1.0)
        # Deliberately NOT proportional to the weight column above. LayerNorm
        # is scale-invariant, so a proportional bias makes the hidden vector a
        # pure rescale of a constant and the gradient is identically zero.
        model.encoder[0].bias.copy_(torch.tensor([0.5, -0.25, 0.75, -1.0]))
        # Non-uniform, because uniform weights here would sum the LayerNorm
        # outputs, which are mean-centred to zero, and the gradient would
        # vanish by construction rather than because of the attribution.
        model.encoder[4].weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]])
        )
        model.encoder[4].bias.copy_(torch.tensor([0.5, 0.5]))
        model.classifier.weight[1].fill_(1.0)
    return model


class ClassGradientBasisTests(unittest.TestCase):
    """A named family comes from a class logit, so attribute against that logit."""

    def test_ranks_the_only_feature_the_logit_can_depend_on(self):
        model = _network_routing_only(1)
        prepared = np.array([[3.0, 5.0, 7.0]], dtype=np.float32)

        explanations = explain_rows(
            model,
            prepared,
            FEATURES,
            class_indices=np.array([1]),
            use_reconstruction=np.array([False]),
        )

        self.assertEqual(explanations[0][0]["feature"], "Tot Fwd Pkts")
        self.assertEqual(explanations[0][0]["basis"], EXPLANATION_BASIS_CLASS)

    def test_features_the_logit_cannot_reach_contribute_nothing(self):
        model = _network_routing_only(1)
        prepared = np.array([[3.0, 5.0, 7.0]], dtype=np.float32)

        explanations = explain_rows(
            model,
            prepared,
            FEATURES,
            class_indices=np.array([1]),
            use_reconstruction=np.array([False]),
            top_k=3,
        )

        unreachable = {
            item["feature"]: item["contribution"]
            for item in explanations[0]
            if item["feature"] != "Tot Fwd Pkts"
        }
        for feature, contribution in unreachable.items():
            with self.subTest(feature=feature):
                self.assertAlmostEqual(contribution, 0.0, places=6)

    def test_reports_the_standardized_observation_against_the_baseline(self):
        model = _network_routing_only(1)
        prepared = np.array([[3.0, 5.0, 7.0]], dtype=np.float32)

        explanations = explain_rows(
            model,
            prepared,
            FEATURES,
            class_indices=np.array([1]),
            use_reconstruction=np.array([False]),
        )

        self.assertAlmostEqual(explanations[0][0]["observed_z"], 5.0, places=5)


class ReconstructionBasisTests(unittest.TestCase):
    """An unknown anomaly has no logit, so attribute against the score itself."""

    def test_contributions_reconstruct_the_reported_anomaly_score(self):
        """The score is the mean of these terms, so this is exactly faithful."""

        model = _zeroed_network()
        prepared = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)

        explanations = explain_rows(
            model,
            prepared,
            FEATURES,
            class_indices=np.array([0]),
            use_reconstruction=np.array([True]),
            top_k=len(FEATURES),
        )

        total = sum(item["contribution"] for item in explanations[0])
        expected_score = float(np.mean(prepared[0] ** 2))

        self.assertAlmostEqual(total, expected_score * len(FEATURES), places=4)

    def test_ranks_the_feature_furthest_from_the_benign_expectation(self):
        model = _zeroed_network()
        prepared = np.array([[1.0, 9.0, 2.0]], dtype=np.float32)

        explanations = explain_rows(
            model,
            prepared,
            FEATURES,
            class_indices=np.array([0]),
            use_reconstruction=np.array([True]),
        )

        self.assertEqual(explanations[0][0]["feature"], "Tot Fwd Pkts")
        self.assertEqual(explanations[0][0]["basis"], EXPLANATION_BASIS_ANOMALY)

    def test_direction_says_which_side_of_the_expectation_the_value_sat(self):
        """Squared terms are all positive, so direction carries the useful part."""

        model = _zeroed_network()
        prepared = np.array([[0.0, -9.0, 0.0], [0.0, 9.0, 0.0]], dtype=np.float32)

        explanations = explain_rows(
            model,
            prepared,
            FEATURES,
            class_indices=np.array([0, 0]),
            use_reconstruction=np.array([True, True]),
        )

        self.assertEqual(explanations[0][0]["direction"], -1)
        self.assertEqual(explanations[1][0]["direction"], 1)


class ExplanationShapeTests(unittest.TestCase):
    def test_returns_one_explanation_list_per_row(self):
        model = _zeroed_network()
        prepared = np.zeros((4, 3), dtype=np.float32)

        explanations = explain_rows(
            model,
            prepared,
            FEATURES,
            class_indices=np.zeros(4, dtype=int),
            use_reconstruction=np.array([True, False, True, False]),
        )

        self.assertEqual(len(explanations), 4)

    def test_respects_top_k(self):
        model = _zeroed_network()
        prepared = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        explanations = explain_rows(
            model,
            prepared,
            FEATURES,
            class_indices=np.array([0]),
            use_reconstruction=np.array([True]),
            top_k=2,
        )

        self.assertEqual(len(explanations[0]), 2)

    def test_an_empty_batch_returns_no_explanations(self):
        model = _zeroed_network()

        explanations = explain_rows(
            model,
            np.zeros((0, 3), dtype=np.float32),
            FEATURES,
            class_indices=np.zeros(0, dtype=int),
            use_reconstruction=np.zeros(0, dtype=bool),
        )

        self.assertEqual(explanations, [])

    def test_method_mapping_covers_both_bases(self):
        self.assertEqual(
            set(EXPLANATION_METHOD), {EXPLANATION_BASIS_CLASS, EXPLANATION_BASIS_ANOMALY}
        )
        self.assertEqual(EXPLANATION_VERSION, 1)

    def test_runs_even_though_scoring_uses_inference_mode(self):
        """Tensors made under inference_mode cannot take gradients later."""

        model = _network_routing_only(0)
        with torch.inference_mode():
            prepared = torch.zeros((1, 3)).cpu().numpy()

        explanations = explain_rows(
            model,
            np.ascontiguousarray(prepared, dtype=np.float32),
            FEATURES,
            class_indices=np.array([1]),
            use_reconstruction=np.array([False]),
        )

        self.assertEqual(len(explanations), 1)


class RenderedSentenceTests(unittest.TestCase):
    """The word in the sentence must match the number in the sentence.

    `direction` is the sign of the contribution and `observed_z` is the
    standardized value; their signs are unrelated. Rendering one with the
    other produces analyst-facing text like "observed -1.27 standard
    deviations above the training baseline".
    """

    def test_observed_side_always_agrees_with_the_sign_of_observed_z(self):
        model = _zeroed_network(input_dim=6, num_classes=3)
        torch.manual_seed(0)
        prepared = np.random.default_rng(0).normal(size=(5, 6)).astype(np.float32)
        names = tuple(f"f{index}" for index in range(6))

        explanations = explain_rows(
            model,
            prepared,
            names,
            class_indices=np.array([0, 1, 2, 1, 0]),
            use_reconstruction=np.array([True, False, True, False, True]),
            top_k=6,
        )

        for row in explanations:
            for item in row:
                with self.subTest(feature=item["feature"]):
                    expected = "above" if item["observed_z"] >= 0 else "below"
                    self.assertEqual(item["observed_side"], expected)

    def test_observed_side_is_independent_of_the_contribution_sign(self):
        """A negative contribution on a positive value still reads 'above'."""

        model = _network_routing_only(1)
        prepared = np.array([[0.0, 4.0, 0.0]], dtype=np.float32)

        explanations = explain_rows(
            model,
            prepared,
            FEATURES,
            class_indices=np.array([1]),
            use_reconstruction=np.array([False]),
            top_k=3,
        )
        item = next(entry for entry in explanations[0] if entry["feature"] == "Tot Fwd Pkts")

        self.assertEqual(item["observed_side"], "above")
        self.assertGreater(item["observed_z"], 0)


if __name__ == "__main__":
    unittest.main()
