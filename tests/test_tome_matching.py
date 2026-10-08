import unittest

import torch

from utome.models.tome import merge_weighted_average, random_bipartite_matching


class RandomBipartiteMatchingTest(unittest.TestCase):
    def test_random_merge_preserves_unmerge_shape(self) -> None:
        torch.manual_seed(0)
        tokens = torch.randn(2, 8, 4)

        merge, unmerge, r_eff = random_bipartite_matching(tokens, r=2)
        merged, size = merge_weighted_average(merge, tokens)
        restored = unmerge(merged)

        self.assertEqual(r_eff, 2)
        self.assertEqual(tuple(merged.shape), (2, 6, 4))
        self.assertEqual(tuple(size.shape), (2, 6, 1))
        self.assertEqual(tuple(restored.shape), tuple(tokens.shape))


if __name__ == "__main__":
    unittest.main()
