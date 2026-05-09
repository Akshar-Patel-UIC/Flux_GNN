###############################################################
# Copyright 2026 Lawrence Livermore National Security, LLC
# (c.f. AUTHORS, NOTICE.LLNS, COPYING)
#
# This file is part of the Flux resource manager framework.
# For details, see https://github.com/flux-framework.
#
# SPDX-License-Identifier: LGPL-3.0
###############################################################

"""Rv1 pool with learned candidate ranking (GNN + MLP, optional checkpoint)."""

import syslog
from typing import List, Tuple

from flux.resource.Rv1Pool import Rv1Pool


class GNNRv1Pool(Rv1Pool):
    """Like :class:`~flux.resource.Rv1Pool.Rv1Pool` but orders candidates via a small PyTorch model.

    Falls back to worst-fit ordering if PyTorch is unavailable, the checkpoint
    is missing, or inference raises.
    """

    known_options: frozenset = frozenset({"model_path", "device", "hidden_dim"})

    def __init__(
        self,
        R,
        log=None,
        model_path=None,
        device="cpu",
        hidden_dim=64,
        **kwargs,
    ) -> None:
        super().__init__(R, log=log, **kwargs)
        self._gnn_model_path = model_path
        self._gnn_device = device
        self._gnn_hidden_dim = int(hidden_dim)
        self._gnn_model = None
        self._gnn_fallback_warned = False
        self._gnn_inference_disabled = False

    def copy(self):
        new = super().copy()
        new._gnn_model_path = self._gnn_model_path
        new._gnn_device = self._gnn_device
        new._gnn_hidden_dim = self._gnn_hidden_dim
        new._gnn_model = self._gnn_model
        new._gnn_fallback_warned = self._gnn_fallback_warned
        new._gnn_inference_disabled = self._gnn_inference_disabled
        return new

    def _copy_from_ranks(self, rank_set: set):
        new = super()._copy_from_ranks(rank_set)
        new._gnn_model_path = getattr(self, "_gnn_model_path", None)
        new._gnn_device = getattr(self, "_gnn_device", "cpu")
        new._gnn_hidden_dim = getattr(self, "_gnn_hidden_dim", 64)
        new._gnn_model = getattr(self, "_gnn_model", None)
        new._gnn_fallback_warned = getattr(self, "_gnn_fallback_warned", False)
        new._gnn_inference_disabled = getattr(self, "_gnn_inference_disabled", False)
        return new

    def _ensure_model(self) -> bool:
        if self._gnn_inference_disabled:
            return False
        if self._gnn_model is not None:
            return True
        try:
            import torch  # noqa: F401

            from flux.gnn.graphs import CAND_FEATURE_DIM, JOB_FEATURE_DIM
            from flux.gnn.model import JobCandidateGNN, load_state_dict_optional
        except ImportError:
            if not self._gnn_fallback_warned:
                self.log(
                    syslog.LOG_INFO,
                    "gnn: PyTorch or flux.gnn unavailable — using worst-fit candidate order",
                )
                self._gnn_fallback_warned = True
            self._gnn_inference_disabled = True
            return False

        self._gnn_model = JobCandidateGNN(
            job_dim=JOB_FEATURE_DIM,
            cand_dim=CAND_FEATURE_DIM,
            hidden=self._gnn_hidden_dim,
        )
        try:
            load_state_dict_optional(
                self._gnn_model,
                self._gnn_model_path,
                map_location=self._gnn_device,
            )
        except OSError as exc:
            if not self._gnn_fallback_warned:
                self.log(
                    syslog.LOG_WARNING,
                    f"gnn: could not load checkpoint {self._gnn_model_path!r}: {exc}",
                )
                self._gnn_fallback_warned = True
            self._gnn_model = None
            self._gnn_inference_disabled = True
            return False

        self._gnn_model.to(self._gnn_device)
        self._gnn_model.eval()
        return True

    def _rank_candidates(self, candidates, jobid, request) -> List[Tuple]:
        if not candidates:
            return candidates
        if not self._ensure_model():
            return super()._rank_candidates(candidates, jobid, request)
        try:
            import torch

            from flux.gnn import graphs as gnn_graphs

            job_t, cand_t = gnn_graphs.build_job_candidate_tensors(request, candidates)
            job_t = job_t.to(self._gnn_device)
            cand_t = cand_t.to(self._gnn_device)
            with torch.no_grad():
                scores = self._gnn_model(job_t, cand_t)
            score_list = scores.cpu().tolist()
            paired = list(zip(score_list, candidates))
            paired.sort(
                key=lambda x: (-x[0], -len(x[1][2]), -len(x[1][3])),
            )
            return [p[1] for p in paired]
        except Exception as exc:
            if not self._gnn_fallback_warned:
                self.log(syslog.LOG_WARNING, f"gnn rank fallback: {exc}")
                self._gnn_fallback_warned = True
            self._gnn_inference_disabled = True
            self._gnn_model = None
            return super()._rank_candidates(candidates, jobid, request)
