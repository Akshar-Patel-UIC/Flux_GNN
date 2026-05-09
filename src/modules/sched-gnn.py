###############################################################
# Copyright 2026 Lawrence Livermore National Security, LLC
# (c.f. AUTHORS, NOTICE.LLNS, COPYING)
#
# This file is part of the Flux resource manager framework.
# For details, see https://github.com/flux-framework.
#
# SPDX-License-Identifier: LGPL-3.0
###############################################################

"""FIFO scheduler with GNN-ordered Rv1 candidate ranks (experimental).

Load with::

    flux module load sched-gnn \\
        [queue-depth=N|unlimited] [log-level=LEVEL] \\
        [model-path=/path/to.pt] [device=cpu|cuda:0] [hidden-dim=64]

If ``model-path`` is omitted or PyTorch is unavailable, candidate ordering
falls back to the default worst-fit policy in :class:`~flux.resource.Rv1Pool.Rv1Pool`.
"""

import importlib.util
import os

from flux.resource.GNNRv1Pool import GNNRv1Pool

_spec = importlib.util.spec_from_file_location(
    "sched_fifo",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sched-fifo.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FIFOScheduler = _mod.FIFOScheduler


class GNNFifoScheduler(FIFOScheduler):
    """Priority-FIFO scheduler using :class:`~flux.resource.GNNRv1Pool.GNNRv1Pool`."""

    pool_class = GNNRv1Pool

    def __init__(self, h, *args):
        super().__init__(h, *args)
        for arg in list(self._pending_args):
            if arg.startswith("model-path="):
                self.pool_kwargs["model_path"] = arg[len("model-path=") :]
                self._pending_args.remove(arg)
            elif arg.startswith("device="):
                self.pool_kwargs["device"] = arg[len("device=") :]
                self._pending_args.remove(arg)
            elif arg.startswith("hidden-dim="):
                val = arg[len("hidden-dim=") :]
                try:
                    self.pool_kwargs["hidden_dim"] = int(val)
                except ValueError:
                    raise ValueError(
                        f"hidden-dim={val!r} is invalid: expected a positive integer"
                    ) from None
                if self.pool_kwargs["hidden_dim"] < 1:
                    raise ValueError("hidden-dim must be at least 1")
                self._pending_args.remove(arg)

    def _reject_unknown_args(self):
        if self._pending_args:
            raise ValueError(
                f"unknown argument {self._pending_args[0]!r}: "
                "expected queue-depth, log-level, pool-class, "
                "model-path, device, or hidden-dim"
            )


def mod_main(h, *args):
    GNNFifoScheduler(h, *args).run()
