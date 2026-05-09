###############################################################
# Copyright 2026 Lawrence Livermore National Security, LLC
# (c.f. AUTHORS, NOTICE.LLNS, COPYING)
#
# This file is part of the Flux resource manager framework.
# For details, see https://github.com/flux-framework.
#
# SPDX-License-Identifier: LGPL-3.0
###############################################################

"""Feature tensors for job + candidate compute graphs (Rv1 / jobspec view)."""

from __future__ import annotations

from typing import List, Tuple

# Fixed widths must match :class:`flux.gnn.model.JobCandidateGNN` defaults.
JOB_FEATURE_DIM = 16
CAND_FEATURE_DIM = 16

_MAX_DUR = 86400.0 * 7.0


def build_job_candidate_tensors(request, candidates) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Build normalized job and per-candidate feature tensors.

    Args:
        request: :class:`~flux.resource.Rv1Pool.ResourceRequest`.
        candidates: List of ``(rank, info, free_cores, free_gpus)`` tuples from
            :meth:`~flux.resource.Rv1Pool.Rv1Pool.alloc`.

    Returns:
        ``job_feat`` of shape ``(JOB_FEATURE_DIM,)`` and ``cand_feats`` of
        shape ``(len(candidates), CAND_FEATURE_DIM)``.
    """
    import torch

    j = torch.zeros(JOB_FEATURE_DIM, dtype=torch.float32)
    j[0] = float(request.nnodes) / 64.0
    j[1] = float(request.nslots) / 512.0
    j[2] = float(request.slot_size) / 128.0
    j[3] = float(request.gpu_per_slot) / 8.0
    dur = float(request.duration)
    j[4] = min(dur, _MAX_DUR) / _MAX_DUR if dur > 0.0 else 0.0
    j[5] = 1.0 if request.exclusive else 0.0
    nn_max = request.nnodes_max if request.nnodes_max else request.nnodes
    ns_max = request.nslots_max if request.nslots_max is not None else request.nslots
    j[6] = float(nn_max) / 64.0
    j[7] = float(ns_max) / 512.0
    j[8] = 1.0 if request.constraint else 0.0

    rows: List["torch.Tensor"] = []
    for rank, info, free_cores, free_gpus in candidates:
        c = torch.zeros(CAND_FEATURE_DIM, dtype=torch.float32)
        ncores = len(info["cores"])
        ngpus = len(info["gpus"])
        c[0] = len(free_cores) / 128.0
        c[1] = len(free_gpus) / 16.0
        c[2] = ncores / 128.0
        c[3] = ngpus / 16.0
        c[4] = len(info["allocated_cores"]) / max(ncores, 1)
        c[5] = len(info["allocated_gpus"]) / max(ngpus, 1)
        c[6] = float(rank) / 1024.0
        c[7] = 1.0 if info.get("up") else 0.0
        rows.append(c)

    if not rows:
        return j, torch.zeros((0, CAND_FEATURE_DIM), dtype=torch.float32)
    return j, torch.stack(rows, dim=0)
