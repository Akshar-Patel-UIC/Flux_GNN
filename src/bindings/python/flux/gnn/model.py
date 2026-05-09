###############################################################
# Copyright 2026 Lawrence Livermore National Security, LLC
# (c.f. AUTHORS, NOTICE.LLNS, COPYING)
#
# This file is part of the Flux resource manager framework.
# For details, see https://github.com/flux-framework.
#
# SPDX-License-Identifier: LGPL-3.0
###############################################################

"""Job–candidate encoder (job MLP + per-candidate head; star-graph semantics)."""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch import nn

from flux.gnn.graphs import CAND_FEATURE_DIM, JOB_FEATURE_DIM


class JobCandidateGNN(nn.Module):
    """Encode job and candidate nodes; score each candidate for placement.

    Message passing is implicit: each candidate sees the job embedding
    concatenated with its own features (bipartite star from job to ranks).
    """

    def __init__(
        self,
        job_dim: int = JOB_FEATURE_DIM,
        cand_dim: int = CAND_FEATURE_DIM,
        hidden: int = 64,
    ):
        super().__init__()
        self.job_dim = job_dim
        self.cand_dim = cand_dim
        self.hidden = hidden
        self.job_enc = nn.Sequential(
            nn.Linear(job_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(cand_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, job_feat: torch.Tensor, cand_feats: torch.Tensor) -> torch.Tensor:
        """Return a score per candidate rank.

        Args:
            job_feat: ``(job_dim,)`` or ``(1, job_dim)``.
            cand_feats: ``(C, cand_dim)``.

        Returns:
            ``(C,)`` scores (higher is preferred).
        """
        if job_feat.dim() == 2:
            job_feat = job_feat.squeeze(0)
        hj = self.job_enc(job_feat)
        c = cand_feats.shape[0]
        if c == 0:
            return torch.zeros(0, device=job_feat.device, dtype=job_feat.dtype)
        hj_exp = hj.unsqueeze(0).expand(c, -1)
        x = torch.cat([cand_feats, hj_exp], dim=1)
        return self.head(x).squeeze(-1)


def load_state_dict_optional(
    model: nn.Module,
    path: Optional[str],
    map_location: Optional[str] = None,
) -> None:
    """Load weights if *path* is set and exists; otherwise leave random init."""
    if not path:
        return
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GNN checkpoint not found: {path}")
    blob = torch.load(path, map_location=map_location or "cpu")
    if isinstance(blob, dict) and "state_dict" in blob:
        blob = blob["state_dict"]
    model.load_state_dict(blob, strict=True)
