# Electric Aircraft Charging Network Design for Regional Routes

This repository contains the implementation of the **Electric Aircraft Charging Network for Regional Routes (EACN-REG)** model and the **Kernel Search (KS)** heuristic, as proposed in the following paper:

> **Kinene, A., Birolini, S., Cattaneo, M., & Granberg, T. A. (2023).**
> *Electric aircraft charging network design for regional routes: A novel mathematical formulation and kernel search heuristic.*
> European Journal of Operational Research, 309, 1300–1315.
> [Read the paper](https://www.sciencedirect.com/science/article/pii/S037722172300125X)

## Overview

The project addresses the strategic problem of designing a charging infrastructure network for electric aviation. 
It solves a **Mixed-Integer Linear Programming (MILP)** model to balance infrastructure investment costs against 
regional connectivity and population coverage.

**Key Features:**
* **EACN-REG Model:** A novel facility location formulation enforcing multi-hop flight feasibility and range constraints.
* **Kernel Search Heuristic:** A decomposition-based heuristic to solve large-scale instances (e.g., country-wide networks) efficiently.
* **Data Processing:** Utilities to generate population grids, process airport data, and filter flight paths based on passenger utility.
* **Real-world Application:** A full case study implementation for Sweden.

---

## Installation

This project requires **Python 3.13**.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage Guide

The project is organized into three main modes of operation: Testing, Scalability Analysis, and the Sweden Case Study.

### 1. Quick Start (Synthetic Test)
Runs the model on a small, randomly generated synthetic dataset to verify installation and logic.
* **Command:** `python test.py`
* **Configuration:** `config.yml`
* **Output:** Logs to console/file and generates an HTML plot of the solution.



https://github.com/user-attachments/assets/8c4e3e2f-3b64-433c-aa29-81b4b03c40f4





### 2. Scalability Analysis (Section 4)
Replicates the computational experiments from **Section 4** of the paper, comparing the exact Branch-and-Cut (B&C) solver against the Kernel Search heuristic.
* **Command:** `python scalability.py`
* **Configuration:** `config_scalability.yml` and `scalability_tests.yml`
* **Output:** A CSV report (`EACN_REG_<timestamp>.csv`) containing solution times, objective values, and gaps for various instance sizes ($N=40, 80$) and ranges ($\tau=400, 600$).

<img width="4947" height="1835" alt="Immagine4" src="https://github.com/user-attachments/assets/c0f7a881-6988-4597-90f1-e0ef80d19c9b" />


### 3. Sweden Case Study (Section 5)
Runs the optimization on real-world data for Sweden. The script automatically downloads the required datasets (airports and population density) from the repository releases.
* **Command:** `python swe_case_study.py`
* **Configuration:** `case_study/config_case_study.yml`
* **Output:** HTML plots visualizing the optimal charging network, covered population, and flight paths.



https://github.com/user-attachments/assets/a61a28ff-6ce6-4f7c-be54-3ae6493c0cc8



### 4. Advanced Trade-off Analysis
Scripts to reproduce the sensitivity analyses discussed in the paper.
* **Pareto Analysis:** Investigate the trade-off between population coverage and the number of charging bases.
    ```bash
    python -m case_study.analysis_1
    ```
    <img width="1906" height="1068" alt="plot_analysis_1" src="https://github.com/user-attachments/assets/b26c7e7e-214f-4322-874b-ea29c662af0c" />

* **Technological Sensitivity:** Investigate the impact of aircraft range ($\tau$) and travel time limits.
    ```bash
    python -m case_study.analysis_2
    ```
    <img width="1911" height="1092" alt="plot_analysis_2" src="https://github.com/user-attachments/assets/cf818538-db9a-4581-bd24-15ceceac32d5" />

    

---

## Configuration

The behavior of the model is controlled by YAML configuration files. Key parameters include:

| Section | Parameter | Description |
| :--- | :--- | :--- |
| **`aircraft_config`** | `tau` | Maximum aircraft range on a single charge (km). |
| | `cruise_speed` | Aircraft speed (km/h). |
| **`paths_config`** | `routing_factor_thr` | Max ratio of (Path Distance / Direct Distance) for a path to be attractive. |
| | `max_total_time_travel` | Max allowed time (ground + flight) for a trip to be valid. |
| **`heuristic_config`** | `enable` | Set `True` to use Kernel Search, `False` for exact solver. |
| | `iterations` | Number of bucket iterations for the heuristic. |
| **`model_config`** | `mu_1`, `mu_2` | Weights for the objective function (Connectivity vs Cost). |

---

## Code Architecture & Paper Mapping

This codebase is structured to mirror the methodology described in the paper.

### 1. Preprocessing (Section 2)
* **Grid Generation:** `utils/init_dataset.py` creates the discrete population grid $K$ and assigns population $\pi_k$.
* **Path Generation:** `utils/preprocessing.py` implements the filtering logic:
    * **Range Constraint:** Builds graph $G$ where edges exist only if $d_{ij} \le \tau$.
    * **Routing Factor:** Discards paths where $d_{path} > 1.4 \times d_{direct}$.
    * **Time Limit:** Filters paths exceeding the 4-hour total travel time limit (Section 5).

### 2. Mathematical Model (Section 2 & 3)
* **Formulation:** `model/utils_model.py` translates the MILP formulation into Gurobi constraints.
    * Variables: $y_i$ (charging base), $z_{ij}$ (edge feasibility), $\rho_i$ (shortest path to base).
    * **Valid Inequalities:** Implements the tightened "Big-M" constraints described in **Section 3.1.1** via `get_tight_big_m()`.

### 3. Kernel Search Heuristic (Section 4)

* **Algorithm:** `model/eanc_reg_model.py` implements **Algorithm 1**.
    * `get_initial_kernel()`: Selects promising airports based on a **cost-efficiency ratio** (Unique Population Cells Served / Activation Cost).
      > **Note:** This implementation introduces cost-awareness to the initialization strategy to prioritize high-ROI airports. 
        This change transforms the initialization from a purely "centrality" based heuristic proposed in the original paper to a "greedy" heuristic.
    * `solve_eacn_model()`: Iteratively solves restricted sub-problems (buckets) to refine the solution.
      * **Early Stopping:** The algorithm includes a mechanism to terminate execution if the objective function value does not improve for a set number of consecutive buckets.
      > **Note:** This early termination criterion is an implementation detail added to reduce computational time on stagnant solutions and is not explicitly defined in the original paper's algorithm.
---

| **Inputs: sets & parameters** |                                                                                                                 |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------|
| $\mathcal{N}$                 | set of airport nodes, indexed by $i$ and $j$                                                                    |
| $\mathcal{N}_i$               | set of airport nodes adjacent to node $i$, i.e., $j \in \mathcal{N}$ such that $d_{ij} \le \tau$ and $i \neq j$ |
| $\mathcal{E}$                 | set of flight edges, indexed by $(i, j)$                                                                        |
| $\mathcal{K}$                 | set of population areas, indexed by $k$                                                                         |
| $\mathcal{P}$                 | set of paths, indexed by $p$                                                                                    |
| $\mathcal{D}$                 | set of destinations, indexed by $d$                                                                             |
| $\mathcal{P}^d_k$             | subset of paths from population area $k$ to destination $d$                                                     |
| $\mathcal{E}_p$               | subset of flight edges in path $p$                                                                              |
| $d_{ij}$                      | travel distance between $i$ and $j$                                                                             |
| $\tau$                        | maximum travel range on a single charge                                                                         |
| $c_i$                         | cost of activating node $i$ as a charging base                                                                  |
| $\pi_k$                       | population living in $k$                                                                                        |
| $\mu_1, \mu_2$                | weights of the two objectives in the objective function                                                         |

| **Variables**             |                                                        |
|---------------------------|--------------------------------------------------------|
| $y_i \in \{0, 1\}$        | = 1 if node $i$ is activated as a charging base        |
| $z_{ij} \in \{0, 1\}$     | = 1 if flight edge $(i, j)$ is feasible                |
| $\rho_i \in \mathbb{R}^+$ | value of the shortest path from $i$ to a charging base |
| $\psi_p \in \{0, 1\}$     | =1 if path $p$ can be used                             |
| $\phi^d_k \in \{0, 1\}$   | =1 if area $k$ is covered w.r.t. destination $d$       |



