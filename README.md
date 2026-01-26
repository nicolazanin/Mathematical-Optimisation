# Mathematical Optimization

Project for the **Mathematical Optimization** exam.

Implementation of a series of Mixed-Integer Linear Programming (MILP) models based on the following paper:
[**Electric aircraft charging network design for regional routes: A novel mathematical formulation and kernel search
heuristic**](https://www.sciencedirect.com/science/article/pii/S037722172300125X)

The project implements a **Mixed-Integer Linear Programming (MILP)** model to optimize the strategic location of
charging infrastructure for electric aircraft.
It addresses the fundamental problem of electric aviation: ensuring connectivity for regional routes while minimizing
infrastructure investment costs.

The solution includes:

1. **Preprocessing Logic:** To generate feasible flight paths based on range and passenger behavior.
2. **Mathematical Model:** A novel formulation (EACN-REG) to solve the charging location problem.
3. **Kernel Search Heuristic:** An efficient algorithm to solve large-scale instances (e.g., the Sweden case study).

---

## Get Started

This project uses **Python 3.13**.

#### Installing Required Packages

<ol>
<li>
Create a virtual environment:

 ```
 python3 -m venv venv
 source venv/bin/activate 
 ```

</li>
<li>
Install the required packages:

```
pip install -r requirements.txt
```

</li>

<li>
Run <em>test.py</em>:

```
python test.py
```

</li>
</ol>

# Code Architecture

This section details how the codebase implements the specific methodologies described in the paper.

## 1. Data Generation & Preprocessing (Section 2)

Before the optimization begins, the model must define the input graph. Since a fully connected graph is
computationally inefficient and unrealistic. Instead, it filters connections based on **aircraft range $\tau$** and
**passenger utility**.

### Grid Generation (`utils/init_dataset.py`)

- Implements the discrete population grid $P$ and airport set $N$.
- **Logic**:
    - `cells_generation()` creates the square grid $K$.
    - `get_pop_density()` assigns population values $\pi_k$.

### Feasible & Attractive Paths (`utils/preprocessing.py`)

- **Range Constraint**:
    - The function `get_threshold_graph()` builds the graph $G = (N, \mathcal{E})$ where edges exist only
      if $dist(i,j) ≤ \tau$.

- **Routing Factor Threshold**:
    - The paper states that passengers will not choose a route if it is significantly longer than the direct
      connection.  
      **Implementation**: `get_attractive_paths_from_rft()` calculates the
      ratio: $r_p = \frac{d_{direct}}{\sum d_{ij}}$.
    - If $r_p > 1.4$ (default config), the path is discarded.

- **Total Travel Time**:
    - As defined in the Case Study setup (Section 5), the model enforces a **4-hour limit** on total travel time (ground
      access + flight + egress).
    - **Implementation**: `get_population_cells_paths()` filters paths where:
      $T_{\text{ground start}} + T_{\text{flight}} + T_{\text{ground end}} > T_{\text{max}}$

## 2. The Mathematical Formulation (Section 2)

The core logic resides in `model/utils_model.py`. This file translates the MILP formulation into Gurobi constraints.

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

### Constraints

- **Flow & Coverage**

- **Recharging Logic**

- **Range Limit**


### Algorithmic Enhancements (Section 3.1.1)

- To improve solver performance, the paper proposes **tightening the "Big-M" constants** used in the linear relaxation.
- **Implementation**: `get_tight_big_m()` calculates specific upper bounds for `M_1, M_2, M_3` based on the graph
  topology, instead of using generic infinite values.


## 3. Kernel Search Heuristic (Section 4)

For the Sweden case study, the problem size is too large for standard Branch-and-Cut.  
The project implements **Algorithm 1** from the paper in `model/eanc_reg_model.py`.

### Initialization

- `get_initial_kernel()`: Selects a subset of "promising" airports based on the potential population they can serve.

### Bucket Iteration

- `get_buckets()`: Partitions the remaining airports into small buckets.
- `solve_eacn_model()`: Iterates through these buckets, temporarily allowing variables in the current bucket to be
  non-zero while fixing others, progressively improving the objective function value.

# Running Experiments & Replication

## 1. Scalability Analysis (Section 4)

This experiment compares the computational efficiency of the exact B&C solver against the Kernel Search heuristic on
synthetic datasets.

- **Command**: 
    ```
    python scalability.py
    ```
- **Configuration**: Reads from `config_scalability.yml` and `scalability_tests.yml` .
- **Process**:
    - Generates random instances with $N=50$ and $N=100$ airports.
    - Varies aircraft range $\tau \in {400,600,800}$.
    - Varies $K$ set.
    - Solves using B&C, KS (1 iteration), and KS (3 iterations).

- **Output**: A CSV report containing solution times (sec), objective values, and optimality gaps.

## 2. Sweden Case Study (Section 5)

This utilizes real-world data (automatically downloaded from the repository releases) to optimize the network for
Sweden.

- **Command**: 
    ```
    python swe_case_study.py
    ```
- **Data Preparation**:

    The script automatically fetches:
    - swe_airports.csv: Active Swedish airports.
    - swe_pd_2019_1km_ASCII_XYZ.csv: WorldPop population density data.
- **Configuration**: Reads from `config_case_study.yml`.
- **Process**:
    - Solves using KS (3 iterations).
- **Output**: Plot file with the dataset and solutions of the Sweden case study.

## 3. Sweden Case Study Trade-off Analysis (Section 5)

- **Command**: 
    ```
    python -m case_study.analysis_1.py.
    ```
- **Data Preparation**:

    The script automatically fetches:
    - swe_airports.csv: Active Swedish airports.
    - swe_pd_2019_1km_ASCII_XYZ.csv: WorldPop population density data.
- **Configuration**: Reads from `config_case_study.yml`.
- **Process**:
  - This script runs the optimization loop multiple times:
    - Varies charging bases limit number.
    - Varies airport network.
  - Solves using KS (3 iterations) maximizing population covered or cells covered.
- **Output**: Pickle file with the solutions of the Sweden case study.

## 4. Sweden Case Study Sensitivity Analysis (Section 5)

- **Command**: 
    ```
    python -m case_study.analysis_2.py.
    ```
- **Data Preparation**:

    The script automatically fetches:
    - swe_airports.csv: Active Swedish airports.
    - swe_pd_2019_1km_ASCII_XYZ.csv: WorldPop population density data.
- **Configuration**: Reads from `config_case_study.yml`.
- **Process**:
  - This script runs the optimization loop multiple times:
    - Varies travel time threshold.
    - Varies aircraft range $\tau.
    - Varies airport network.
  - Solves using KS (3 iterations) maximizing population covered or cells covered.
- **Output**: Pickle file with the solutions of the Sweden case study.
