# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Unit tests for hastegeo.core.utils.region_aggregate."""

import random
import unittest

from hastegeo.core.models.projects import BuildingResultStatus
from hastegeo.core.utils.region_aggregate import aggregate
from hastegeo.workflows.damage_triage_infer import DTF_CATEGORIES


class TestRegionAggregate(unittest.TestCase):
    def test_aggregation_order_invariance(self):
        # Multiple evidence rows for the same building
        ev1 = {"building_id": "b1", "unit_id": "unit_B", "probs": [0.1, 0.8, 0.1, 0.0, 0.0], "class": 1, "confidence": 0.8, "coverage_fraction": 0.6}
        ev2 = {"building_id": "b1", "unit_id": "unit_A", "probs": [0.0, 0.9, 0.1, 0.0, 0.0], "class": 1, "confidence": 0.9, "coverage_fraction": 0.9}  # Best coverage
        ev3 = {"building_id": "b1", "unit_id": "unit_C", "probs": [0.7, 0.2, 0.1, 0.0, 0.0], "class": 0, "confidence": 0.7, "coverage_fraction": 0.4}

        evidence = [ev1, ev2, ev3]
        buildings = [{"building_id": "b1", "image_layer_id": "l1"}]

        # Run with original order
        res1 = aggregate(evidence, buildings, assessment_id="ast_1")
        self.assertEqual(res1[0]["winning_unit_id"], "unit_A")
        self.assertEqual(res1[0]["confidence"], 0.9)

        # Shuffle evidence 10 times and verify identical output
        for _ in range(10):
            shuffled = list(evidence)
            random.shuffle(shuffled)
            res = aggregate(shuffled, buildings, assessment_id="ast_1")
            self.assertEqual(res[0]["winning_unit_id"], "unit_A")
            self.assertEqual(res[0]["confidence"], 0.9)
            self.assertEqual(res[0]["damage_class"], 1)

    def test_tie_breaking_rules(self):
        # Tie on coverage: higher confidence wins
        ev_low_conf = {"building_id": "b1", "unit_id": "unit_B", "class": 0, "confidence": 0.7, "coverage_fraction": 0.8}
        ev_high_conf = {"building_id": "b1", "unit_id": "unit_A", "class": 1, "confidence": 0.85, "coverage_fraction": 0.8}
        res = aggregate([ev_low_conf, ev_high_conf], [{"building_id": "b1"}])
        self.assertEqual(res[0]["winning_unit_id"], "unit_A")
        self.assertEqual(res[0]["confidence"], 0.85)

        # Tie on coverage and confidence: lower unit_id wins
        ev_unit_2 = {"building_id": "b2", "unit_id": "unit_2", "class": 0, "confidence": 0.8, "coverage_fraction": 0.8}
        ev_unit_1 = {"building_id": "b2", "unit_id": "unit_1", "class": 0, "confidence": 0.8, "coverage_fraction": 0.8}
        res2 = aggregate([ev_unit_2, ev_unit_1], [{"building_id": "b2"}])
        self.assertEqual(res2[0]["winning_unit_id"], "unit_1")

    def test_uncertain_status_threshold(self):
        ev = {"building_id": "b1", "unit_id": "u1", "class": 1, "confidence": 0.55, "coverage_fraction": 1.0}
        cfg = {"uncertain_confidence_threshold": 0.6}
        res = aggregate([ev], [{"building_id": "b1"}], config=cfg)
        self.assertEqual(res[0]["result_status"], BuildingResultStatus.UNCERTAIN.value)

    def test_no_evidence_sets_insufficient_data_with_null_class(self):
        buildings = [{"building_id": "b_nodata"}]
        res = aggregate([], buildings)
        self.assertEqual(res[0]["result_status"], BuildingResultStatus.INSUFFICIENT_DATA.value)
        self.assertIsNone(res[0]["damage_class"])
        self.assertIsNone(res[0]["damage_class_name"])
        self.assertIsNone(res[0]["confidence"])
        self.assertIsNone(res[0]["damage_percentage"])

    def test_damage_percentage_is_always_none(self):
        ev = {"building_id": "b1", "unit_id": "u1", "class": 2, "confidence": 0.95, "coverage_fraction": 1.0}
        res = aggregate([ev], [{"building_id": "b1"}])
        self.assertIsNone(res[0]["damage_percentage"])
        self.assertEqual(res[0]["damage_class_name"], DTF_CATEGORIES[2])


if __name__ == "__main__":
    unittest.main()
