#!/usr/bin/env python3
###############################################################
# Copyright 2026 Lawrence Livermore National Security, LLC
# (c.f. AUTHORS, NOTICE.LLNS, COPYING)
#
# This file is part of the Flux resource manager framework.
# For details, see https://github.com/flux-framework.
#
# SPDX-License-Identifier: LGPL-3.0
###############################################################

import unittest

import subflux  # noqa: F401
from flux.resource.ResourceCount import ResourceCount
from flux.resource.Rv1Pool import ResourceRequest, Rv1Pool
from flux.resource.GNNRv1Pool import GNNRv1Pool
from pycotap import TAPTestRunner

try:
    import torch
except ImportError:
    torch = None


R_3x4 = {
    "version": 1,
    "execution": {
        "R_lite": [{"rank": "0-2", "children": {"core": "0-3"}}],
        "starttime": 0,
        "expiration": 0,
        "nodelist": ["n0", "n1", "n2"],
    },
}


def rr_1n_1s():
    return ResourceRequest(
        ResourceCount(1, 1),
        ResourceCount(1, 1),
        1,
        0,
        0.0,
        None,
        False,
        None,
    )


class _ReverseIndexScoreModel:
    """Higher score for later candidates (tests GNN reordering vs worst-fit ties)."""

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, job_t, cand_t):
        n = cand_t.shape[0]
        # Ascending score with candidate index → under ties, last rank is preferred.
        return torch.arange(1, n + 1, dtype=torch.float32, device=cand_t.device)


class TestGnnRv1Pool(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_graph_tensor_shapes(self):
        from flux.gnn import graphs as gnn_graphs

        request = rr_1n_1s()
        pool = Rv1Pool(R_3x4)
        cand = []
        for rank, info in pool._ranks.items():
            if not info["up"]:
                continue
            fc = info["cores"] - info["allocated_cores"]
            fg = info["gpus"] - info["allocated_gpus"]
            cand.append((rank, info, fc, fg))
        j, c = gnn_graphs.build_job_candidate_tensors(request, cand)
        self.assertEqual(tuple(j.shape), (gnn_graphs.JOB_FEATURE_DIM,))
        self.assertEqual(
            tuple(c.shape),
            (len(cand), gnn_graphs.CAND_FEATURE_DIM),
        )

    @unittest.skipIf(torch is None, "torch not installed")
    def test_gnn_reorders_tied_worst_fit(self):
        request = rr_1n_1s()
        base = Rv1Pool(R_3x4)
        gnn = GNNRv1Pool(R_3x4)
        gnn._gnn_model = _ReverseIndexScoreModel()

        a_base = base.alloc(100, request)
        a_gnn = gnn.alloc(101, request)

        br = set(a_base._ranks)
        gr = set(a_gnn._ranks)
        self.assertEqual(len(br), 1)
        self.assertEqual(len(gr), 1)
        self.assertNotEqual(br, gr)

    def test_fallback_rank_matches_rv1pool(self):
        request = rr_1n_1s()
        plain = Rv1Pool(R_3x4)
        cand = []
        for rank, info in plain._ranks.items():
            if not info["up"]:
                continue
            fc = info["cores"] - info["allocated_cores"]
            fg = info["gpus"] - info["allocated_gpus"]
            if len(fc) < request.slot_size:
                continue
            cand.append((rank, info, fc, fg))

        class _NoGnnModelPool(GNNRv1Pool):
            def _ensure_model(self):
                return False

        out = _NoGnnModelPool(R_3x4)._rank_candidates(cand, 1, request)
        exp = plain._rank_candidates(cand, 1, request)
        self.assertEqual([t[0] for t in out], [t[0] for t in exp])


if __name__ == "__main__":
    unittest.main(testRunner=TAPTestRunner())
