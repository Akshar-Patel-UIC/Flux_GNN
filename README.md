[![ci](https://github.com/flux-framework/flux-core/workflows/ci/badge.svg)](https://github.com/flux-framework/flux-core/actions?query=workflow%3A.github%2Fworkflows%2Fmain.yml)
[![codecov](https://codecov.io/gh/flux-framework/flux-core/branch/master/graph/badge.svg)](https://codecov.io/gh/flux-framework/flux-core)

See our [Flux Framework Documentation](https://flux-framework.readthedocs.io)
and the [Flux Core Documentation](https://flux-framework.readthedocs.io/projects/flux-core)

_NOTE: the github issue tracker is the primary way to communicate
with Flux developers._


### flux-core

flux-core implements the lowest level services and interfaces for the Flux
resource manager framework.  It is intended to be the first building block
used in the construction of a site-composed Flux resource manager.  Other
building blocks are also in development under the
[flux-framework github organization](https://github.com/flux-framework),
including a workload [scheduler](https://github.com/flux-framework/flux-sched).

Contributors: please see [CONTRIBUTING](CONTRIBUTING.md).

#### Build Requirements

For convenience, scripts that install flux-core's build dependencies
are available for [redhat](scripts/install-deps-rpm.sh) and
[debian](scripts/install-deps-deb.sh) distros.

##### Building from Source
```
./autogen.sh   # skip if building from a release tarball
./configure
make -jN
make -jN check
```

> [!TIP]
> `make check` runs many tests. Use of `-jN` recommended.
> For more details about the testsuite, see the [README](t/README.md).

##### VSCode Dev Containers

If you use VSCode we have a dev container and [instructions](vscode.md).

##### Container Images

Pre-built container images are available for a variety of linux distributions and
amd64/arm64 architectures on [Docker Hub](https://hub.docker.com/r/fluxrm/flux-core).
Tags are of the form `distribution-architecture` for the latest development version,
with `latest` mapping to `bookworm-amd64`. Specific releases are available via
`distribution-v0.XX.Y-architecture`.

#### Starting Flux

A Flux instance is composed of a set of `flux-broker` processes running as
a parallel job and can be started by most launchers that can start MPI jobs.
Doing so for a single user does not require administrator privilege.
To start a Flux instance (size = 8) on the local node for testing, use
flux's built-in test launcher:
```
src/cmd/flux start --test-size=8
```
A shell is spawned in which Flux commands can be executed.  When the shell
exits, Flux exits.

For more information on starting Flux in various environments and using it,
please refer to our [documentation](https://flux-framework.readthedocs.io/projects/flux-core/en/latest/guide/start.html).

##### Experimental GNN scheduler (`sched-gnn`)

This fork includes an optional Python scheduler module that reorders Rv1
allocation candidates using a small PyTorch model (`flux.gnn`). Load it
instead of the default `sched-simple` after starting an instance, for example:

```
flux module remove sched-simple
flux module load sched-gnn
```

Optional arguments: `model-path=/path/to.pt`, `device=cpu`, `hidden-dim=64`,
plus the usual `queue-depth` and `log-level`. If PyTorch is not installed or
inference fails, ordering falls back to the standard worst-fit policy.

To build and run in Docker with PyTorch included, see
[docker/Dockerfile.gnn](docker/Dockerfile.gnn) and optional
[requirements-gnn.txt](requirements-gnn.txt) for a local `pip install`.
The Dockerfile sets `FLUX_VERSION` (default `0.65.0`) because `autogen.sh`
needs a `major.minor.point` version when git tags are not in the build context;
override with `docker build --build-arg FLUX_VERSION=…` if you prefer.

#### Release

SPDX-License-Identifier: LGPL-3.0

LLNL-CODE-764420
