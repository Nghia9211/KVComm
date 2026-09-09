import unittest
from collections import defaultdict

import torch

from segmented_kv import (
    accumulate_segment_masses,
    accumulate_segment_attention,
    logical_retention,
    segment_attention_mass_from_qk,
    select_segmented_layers,
    slice_segmented_kv,
    top_fraction_layers,
)


class SegmentedSelectionTests(unittest.TestCase):
    def test_top_70_percent_uses_floor_and_stable_ties(self):
        scores = {layer: 1.0 for layer in range(36)}
        self.assertEqual(top_fraction_layers(scores, 0.7), list(range(25)))

    def test_context_and_latent_are_ranked_independently(self):
        totals = {
            "context": defaultdict(list, {0: [1.0], 1: [0.0], 2: [0.5]}),
            "latent": defaultdict(list, {0: [0.0], 1: [1.0], 2: [0.5]}),
        }
        context, latent, _, _ = select_segmented_layers(totals, 2 / 3, 2 / 3)
        self.assertEqual(context, [0, 2])
        self.assertEqual(latent, [1, 2])

    def test_attention_segments_exclude_sink_and_receiver_keys(self):
        # Keys: sink, context, context, latent, latent, B, B.
        layer0 = torch.tensor([[[[0.90, 0.10, 0.20, 0.30, 0.40, 9.0, 9.0]]]])
        layer1 = torch.tensor([[[[0.90, 0.40, 0.50, 0.10, 0.20, 1.0, 1.0]]]])
        totals = accumulate_segment_attention({0: layer0, 1: layer1}, 3, 2)
        self.assertEqual(totals["context"][0], [0.0])
        self.assertEqual(totals["context"][1], [1.0])
        self.assertEqual(totals["latent"][0], [1.0])
        self.assertEqual(totals["latent"][1], [0.0])

    def test_chunked_qk_mass_matches_full_attention(self):
        torch.manual_seed(7)
        query = torch.randn(1, 4, 3, 2)
        key = torch.randn(1, 2, 7, 2)
        mask = torch.zeros(1, 1, 3, 7)
        mask[..., 0, -1] = torch.finfo(mask.dtype).min
        actual = segment_attention_mass_from_qk(
            query, key, mask, scaling=0.5,
            context_length=3, latent_length=2,
            num_key_value_groups=2, query_chunk_size=1,
        )

        repeated_key = key[:, :, None, :, :].expand(1, 2, 2, 7, 2).reshape(1, 4, 7, 2)
        logits = torch.matmul(query, repeated_key.transpose(2, 3)) * 0.5 + mask
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        expected_context = probabilities[..., 1:3].sum(dim=-1).mean().item()
        expected_latent = probabilities[..., 3:5].sum(dim=-1).mean().item()
        self.assertAlmostEqual(actual["context"], expected_context, places=6)
        self.assertAlmostEqual(actual["latent"], expected_latent, places=6)

    def test_scalar_masses_are_normalized_per_sample(self):
        masses = {
            0: {"context": 2.0, "latent": 4.0},
            1: {"context": 6.0, "latent": 1.0},
        }
        totals = accumulate_segment_masses(masses)
        self.assertEqual(totals["context"][0], [0.0])
        self.assertEqual(totals["context"][1], [1.0])
        self.assertEqual(totals["latent"][0], [1.0])
        self.assertEqual(totals["latent"][1], [0.0])

    def test_chunked_qk_adds_causal_mask_when_sdpa_omits_it(self):
        torch.manual_seed(9)
        query = torch.randn(1, 2, 3, 2)
        key = torch.randn(1, 1, 7, 2)
        actual = segment_attention_mass_from_qk(
            query, key, None, scaling=1.0,
            context_length=3, latent_length=1,
            num_key_value_groups=2, query_chunk_size=2,
        )
        repeated_key = key.expand(1, 2, 7, 2)
        logits = torch.matmul(query, repeated_key.transpose(2, 3))
        causal = torch.zeros(1, 1, 3, 7)
        for query_index in range(3):
            causal[..., query_index, 5 + query_index:] = torch.finfo(causal.dtype).min
        probabilities = torch.softmax(logits + causal, dim=-1, dtype=torch.float32)
        self.assertAlmostEqual(
            actual["context"], probabilities[..., 1:3].sum(dim=-1).mean().item(), places=6
        )
        self.assertAlmostEqual(
            actual["latent"], probabilities[..., 3:4].sum(dim=-1).mean().item(), places=6
        )


class SegmentedGeometryTests(unittest.TestCase):
    def setUp(self):
        self.key = torch.arange(6).view(1, 1, 6, 1)
        self.value = self.key + 10

    def _positions(self, keep_context, keep_latent):
        key, value, state = slice_segmented_kv(
            self.key, self.value, context_length=4, latent_length=2,
            keep_context=keep_context, keep_latent=keep_latent,
        )
        return key.flatten().tolist(), value.flatten().tolist(), state

    def test_all_four_cache_geometries(self):
        self.assertEqual(self._positions(True, True)[0], [0, 1, 2, 3, 4, 5])
        self.assertEqual(self._positions(True, False)[0], [0, 1, 2, 3])
        self.assertEqual(self._positions(False, True)[0], [0, 4, 5])
        self.assertEqual(self._positions(False, False)[0], [0])
        self.assertEqual(self._positions(False, True)[2], "sink+latent")

    def test_logical_retention_counts_sink_for_unselected_context(self):
        stats = logical_retention(4, [0, 1], [1, 2], 10, 2)
        # layer0=10, layer1=12, layer2=3, layer3=1
        self.assertEqual(stats["retained_kv_token_positions"], 26)
        self.assertEqual(stats["full_kv_token_positions"], 48)


if __name__ == "__main__":
    unittest.main()
