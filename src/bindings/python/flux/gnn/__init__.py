###############################################################
# Copyright 2026 Lawrence Livermore National Security, LLC
# (c.f. AUTHORS, NOTICE.LLNS, COPYING)
#
# This file is part of the Flux resource manager framework.
# For details, see https://github.com/flux-framework.
#
# SPDX-License-Identifier: LGPL-3.0
###############################################################

"""Graph-style encodings and PyTorch models for experimental schedulers."""

from flux.gnn.model import JobCandidateGNN, load_state_dict_optional

__all__ = ["JobCandidateGNN", "load_state_dict_optional"]
