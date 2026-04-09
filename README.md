# CLVF_JAX

**Constructing Control Lyapunov-Value Functions via Hamilton-Jacobi Reachability in JAX**

This repository is a JAX-based research codebase for computing and experimenting with **Control Lyapunov-Value Functions (CLVFs)** for nonlinear systems with bounded control inputs.

It is motivated by the papers:

> **Zheng Gong, Muhan Zhao, Thomas Bewley, and Sylvia Herbert**  
> *Constructing Control Lyapunov-Value Functions using Hamilton-Jacobi Reachability Analysis*  
> IEEE Control Systems Letters, 2022.
> *Robust control lyapunov-value functions for nonlinear disturbed systems*
> arXiv preprint arXiv:2403.03455

The core idea is to modify Hamilton-Jacobi (HJ) reachability so that the resulting value function behaves like a **control Lyapunov function**, rather than only a reachability value function. In particular, the CLVF framework enables computation of value functions that can be used to:

- stabilize a system to the **origin** when a stabilizable equilibrium exists,
- stabilize a system to the **smallest control invariant set** around a point of interest when such an equilibrium does not exist,
- explicitly incorporate **input bounds** into the computation,
- characterize the **region of exponential stabilizability** under a desired decay rate `gamma`,
- and support online feedback control through a **feasibility-guaranteed quadratic program (CLVF-QP)**.

---

## Background

Classical **control Lyapunov functions (CLFs)** are a standard tool for nonlinear stabilization, but constructing them for general nonlinear systems is difficult, especially under control constraints. Standard **HJ reachability** methods, on the other hand, can handle general nonlinear dynamics and bounded inputs through dynamic programming, but they typically provide control invariance or reachability guarantees rather than asymptotic or exponential stabilization.

The CLVF framework bridges these two viewpoints.

In the paper above, the CLVF is shown to be the **viscosity solution of a modified Hamilton-Jacobi variational inequality**, making it possible to compute a Lyapunov-like object numerically on a grid while preserving the ability to reason about bounded controls and nonlinear dynamics.

---

## What this repository focuses on

This repository adapts the JAX-based `hj_reachability` workflow toward the CLVF setting, with emphasis on:

- grid-based computation of CLVFs,
- numerical study of how the exponential decay rate `gamma` changes the solution,
- estimation of the **region of exponential stabilizability (ROES)**,
- CLVF-based controller synthesis for control-affine systems,
- and simulation of trajectories under online QP controllers.

Conceptually:

- **`gamma = 0`** corresponds to the largest control-invariant behavior captured by the CLVF construction;
- **`gamma > 0`** restricts attention to states that can be stabilized with exponential decay rate `gamma`;
- larger `gamma` generally leads to a **smaller stabilizable region**, but enforces faster decay of the value along closed-loop trajectories.

---

## Repository structure

```text
CLVF_JAX/
├── hj_reachability/          # core JAX HJ reachability package
├── examples/
│   ├── quickstart.ipynb
│   ├── Example_all.ipynb
│   ├── EX_XPlusXU.ipynb
│   ├── EX_double_int.ipynb
│   ├── controller.py
│   ├── qp_controller.py
│   ├── ZG_solver.py
│   └── ZG_time_integration.py
├── requirements.txt
├── requirements-test.txt
├── setup.py
└── README.md
```

At the moment, the repository still preserves much of the upstream `hj_reachability` package structure. The CLVF-specific additions are most visible in the `examples/` directory and in the customized solver / controller utilities.

---

## Installation

Install JAX first according to your hardware platform (CPU / CUDA / etc.). Then clone and install this repository locally:

```bash
git clone https://github.com/ZG0327/CLVF_JAX.git
cd CLVF_JAX
pip install -e .
```

You may also install the listed dependencies with:

```bash
pip install -r requirements.txt
```

> **Note:** the repository currently still follows the upstream package structure, and `setup.py` retains the package name `hj_reachability`.

---

## Getting started

A practical starting path is:

1. Open `examples/quickstart.ipynb` to see the basic workflow.
2. Use `EX_XPlusXU.ipynb` and `EX_double_int.ipynb` for low-dimensional CLVF examples.
3. Inspect `ZG_solver.py` and `ZG_time_integration.py` for the PDE update and time-marching implementation.
4. Use `controller.py` and `qp_controller.py` to test CLVF-based feedback laws and QP controllers.

---

## Typical workflow

A typical CLVF workflow in this repository is:

1. Define the nonlinear dynamics and admissible control set.
2. Define an implicit target / loss function around the desired operating region.
3. Solve the modified HJ variational inequality backward until convergence.
4. Obtain the CLVF on the computational grid.
5. Evaluate gradients of the CLVF and synthesize online control actions.
6. Simulate trajectories and compare behavior for different choices of `gamma`.

---

## Why use CLVFs?

Compared with hand-crafted CLFs, the CLVF approach is attractive because it:

- applies to **general nonlinear systems**,
- incorporates **input constraints** directly into the construction,
- produces a Lyapunov-like certificate through **dynamic programming**,
- and supports online control synthesis through optimization-based controllers.

The main tradeoff is the familiar **curse of dimensionality** of grid-based HJ methods.

---

## Current status

This repository should currently be viewed as a **research codebase** rather than a polished standalone software package.

In particular:

- the top-level structure is still close to the upstream `hj_reachability` repository,
- the package metadata in `setup.py` still points to the upstream project,
- and the main entry point for understanding the CLVF-specific modifications is the `examples/` folder.

---

## Citation

If you use this repository in academic work, please cite the CLVF paper:

```bibtex
@article{gong2022clvf,
  title={Constructing Control Lyapunov-Value Functions using Hamilton-Jacobi Reachability Analysis},
  author={Gong, Zheng and Zhao, Muhan and Bewley, Thomas and Herbert, Sylvia},
  journal={IEEE Control Systems Letters},
  volume={7},
  pages={925--930},
  year={2022},
  doi={10.1109/LCSYS.2022.3228728}
}
```

---

## Acknowledgment

This repository builds on the JAX HJ reachability code structure developed in the upstream `hj_reachability` project and reorients it toward **Control Lyapunov-Value Function** computation and experimentation.
