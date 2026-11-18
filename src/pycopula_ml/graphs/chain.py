"""Maximum-mutual-information Hamiltonian chains.

The baseline heuristics are based on MLCOPULA 1.1.1's ``chain_graphical_model``
(double-ended greedy) and ``chain_tst`` (TSP-inspired local optimization).
Unlike a maximum spanning tree, a chain requires every vertex to have degree
at most two and connects all vertices in one simple path.

The TSP-inspired method here directly optimizes an *open* path, not a cyclic
TSP tour with the closing edge discarded. Results therefore need not coincide
with R's implementation on all data sets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pycopula_ml.dependence.weight_matrix import CopulaWeightMatrices

ChainMethod = Literal[
    "greedy",
    "tsp",
    "tsp_coin_swap",
    "greedy_random_swap",
    "greedy_inward_2opt",
    "best",
]

# The five comparable algorithms, excluding the legacy meta-selector ``best``.
CHAIN_METHODS: tuple[str, ...] = (
    "greedy",
    "tsp",
    "tsp_coin_swap",
    "greedy_random_swap",
    "greedy_inward_2opt",
)


@dataclass(frozen=True)
class ChainEdge:
    """One adjacent pair in the selected chain, with MI measured in nats."""

    source: str
    target: str
    mutual_information: float


@dataclass(frozen=True)
class ChainGraph:
    """Result of a mutual-information chain selection.

    Attributes
    ----------
    order : tuple[int, ...]
        Permutation of the feature indices (zero-based).
    feature_names : tuple[str, ...]
        Labels corresponding to the columns of the original MI matrix.
    edges : tuple[ChainEdge, ...]
        Consecutive, undirected edges in traversal order.
    total_mi : float
        Sum of the ``n_features - 1`` selected edge weights, in nats.
    method : str
        Heuristic that produced the chain; ``best`` reports its winner.
    """

    order: tuple[int, ...]
    feature_names: tuple[str, ...]
    edges: tuple[ChainEdge, ...]
    total_mi: float
    method: str

    @property
    def permutation(self) -> tuple[str, ...]:
        """Return the ordered feature labels, matching the R chain table."""
        return tuple(self.feature_names[i] for i in self.order)

    @property
    def edge_weights(self) -> NDArray[np.float64]:
        """Return the ordered MI values of the selected edges."""
        return np.array([edge.mutual_information for edge in self.edges], dtype=float)

    def adjacency_matrix(self, *, weighted: bool = True) -> NDArray[np.float64]:
        """Return a symmetric adjacency matrix without self-loops.

        With ``weighted=False``, edges have value 1 even if their MI is zero.
        This preserves graph topology for independent feature pairs.
        """
        n_features = len(self.feature_names)
        adjacency = np.zeros((n_features, n_features), dtype=float)
        for i, j, edge in zip(self.order[:-1], self.order[1:], self.edges, strict=True):
            weight = edge.mutual_information if weighted else 1.0
            adjacency[i, j] = weight
            adjacency[j, i] = weight
        return adjacency

    def edge_attributes(self, *, bidirectional: bool = True) -> NDArray[np.float64]:
        """Return MI edge attributes shaped ``(E, 1)`` aligned to edge indices."""
        weights = self.edge_weights
        if bidirectional:
            weights = np.tile(weights, 2)
        return weights.reshape(-1, 1)

    def edge_index(self, *, bidirectional: bool = True) -> NDArray[np.int64]:
        """Return edge indices shaped ``(2, E)`` for GNN data pipelines.

        Undirected edges are repeated in both orientations by default. Edge
        weights are provided separately through :attr:`edge_weights`.
        The result is a NumPy array; no PyTorch dependency is required.
        """
        pairs = [(i, j) for i, j in zip(self.order[:-1], self.order[1:], strict=True)]
        if bidirectional:
            pairs += [(j, i) for i, j in pairs]
        return np.asarray(pairs, dtype=np.int64).T.copy()


def _validate_mi_matrix(
    matrix: ArrayLike,
    names: Sequence[str] | None,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Validate symmetric, nonnegative MI values and optional feature names."""
    try:
        mi = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("mi_matrix must contain numeric values.") from exc
    if mi.ndim != 2 or mi.shape[0] != mi.shape[1] or mi.shape[0] < 2:
        raise ValueError("mi_matrix must be square with at least two features.")
    if not np.isfinite(mi).all():
        raise ValueError("mi_matrix must contain only finite values.")
    if not np.allclose(mi, mi.T, rtol=1e-10, atol=1e-10):
        raise ValueError("mi_matrix must be symmetric.")
    if np.any(mi < -1e-10):
        raise ValueError("Mutual information cannot be negative.")
    if not np.allclose(np.diag(mi), 0.0, rtol=0.0, atol=1e-10):
        raise ValueError("mi_matrix must have a zero diagonal (no self-edges).")

    n = mi.shape[0]
    if names is None:
        labels = tuple(f"X{i + 1}" for i in range(n))
    else:
        labels = tuple(names)
        if len(labels) != n or any(not isinstance(x, str) or not x for x in labels):
            raise ValueError(
                "feature_names must be nonempty strings matching the matrix."
            )
        if len(set(labels)) != n:
            raise ValueError("feature_names must be unique.")
    cleaned = np.maximum(0.0, (mi + mi.T) / 2.0)
    np.fill_diagonal(cleaned, 0.0)
    return cleaned, labels


def _canonical(order: Sequence[int]) -> tuple[int, ...]:
    """Remove path-reversal ambiguity for reproducible tie resolution."""
    forward = tuple(order)
    reverse = forward[::-1]
    return min(forward, reverse)


def _score(mi: NDArray[np.float64], order: Sequence[int]) -> float:
    """Sum mutual information across adjacent vertices in the path."""
    return float(sum(mi[i, j] for i, j in zip(order[:-1], order[1:], strict=True)))


def _double_ended_greedy(mi: NDArray[np.float64]) -> tuple[int, ...]:
    """Start at the strongest pair and extend the stronger available end."""
    n = len(mi)
    # Only the strict upper triangle represents distinct feature pairs.
    candidates = ((-mi[i, j], i, j) for i in range(n) for j in range(i + 1, n))
    _, first, second = min(candidates)
    order = [first, second]
    visited = set(order)

    while len(order) < n:
        # A candidate is (-MI, added feature, side); ties are deterministic.
        options = []
        for vertex in range(n):
            if vertex in visited:
                continue
            options.append((-mi[order[0], vertex], vertex, 0))
            options.append((-mi[order[-1], vertex], vertex, 1))
        _, vertex, side = min(options)
        if side == 0:
            order.insert(0, vertex)
        else:
            order.append(vertex)
        visited.add(vertex)
    return _canonical(order)


def _nearest_neighbor(mi: NDArray[np.float64], start: int) -> tuple[int, ...]:
    """Grow an open Hamiltonian path by maximum-MI nearest neighbors."""
    unvisited = set(range(len(mi)))
    unvisited.remove(start)
    order = [start]
    while unvisited:
        next_vertex = min(unvisited, key=lambda j: (-mi[order[-1], j], j))
        order.append(next_vertex)
        unvisited.remove(next_vertex)
    return tuple(order)


def _two_opt_gain(
    mi: NDArray[np.float64], order: Sequence[int], i: int, j: int
) -> float:
    """Return the MI gain from reversing the closed interval ``[i, j]``."""
    before = after = 0.0
    if i > 0:
        before += mi[order[i - 1], order[i]]
        after += mi[order[i - 1], order[j]]
    if j < len(order) - 1:
        before += mi[order[j], order[j + 1]]
        after += mi[order[i], order[j + 1]]
    return float(after - before)


def _two_opt(mi: NDArray[np.float64], initial: Sequence[int]) -> tuple[int, ...]:
    """Improve an open path by first-improvement 2-opt segment reversals."""
    order = list(initial)
    n = len(order)
    while True:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                if _two_opt_gain(mi, order, i, j) > 1e-12:
                    order[i : j + 1] = reversed(order[i : j + 1])
                    improved = True
                    break
            if improved:
                break
        if not improved:
            return _canonical(order)


def _inward_two_opt(mi: NDArray[np.float64], initial: Sequence[int]) -> tuple[int, ...]:
    """First-improvement 2-opt, testing outermost segments before inner ones.

    All admissible reversals are considered, so a pass ends at a 2-opt local
    optimum. Only the *order* of candidate moves differs from regular 2-opt.
    """
    order = list(initial)
    n = len(order)
    # Prioritize large reversals touching the outer parts of the chain.
    candidates = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
    candidates.sort(
        key=lambda pair: (
            min(pair[0], n - 1 - pair[1]),
            -(pair[1] - pair[0]),
            pair[0],
        )
    )
    while True:
        for i, j in candidates:
            if _two_opt_gain(mi, order, i, j) > 1e-12:
                order[i : j + 1] = reversed(order[i : j + 1])
                break
        else:
            return _canonical(order)


def _swap_gain(mi: NDArray[np.float64], order: Sequence[int], i: int, j: int) -> float:
    """Compute the exact MI change of swapping two positions in O(1) time."""
    if i == j:
        return 0.0
    if i > j:
        i, j = j, i
    n = len(order)
    affected = {edge for edge in (i - 1, i, j - 1, j) if 0 <= edge < n - 1}

    def new_vertex(position: int) -> int:
        if position == i:
            return order[j]
        if position == j:
            return order[i]
        return order[position]

    old = sum(mi[order[k], order[k + 1]] for k in affected)
    new = sum(mi[new_vertex(k), new_vertex(k + 1)] for k in affected)
    return float(new - old)


def _greedy_random_swap(
    mi: NDArray[np.float64], trials: int, rng: np.random.Generator
) -> tuple[int, ...]:
    """Accept only improving random position swaps after double-ended greedy."""
    order = list(_double_ended_greedy(mi))
    for _ in range(trials):
        i = int(rng.integers(0, len(order)))
        j = int(rng.integers(0, len(order) - 1))
        if j >= i:
            j += 1
        if _swap_gain(mi, order, i, j) > 1e-12:
            order[i], order[j] = order[j], order[i]
    return _canonical(order)


def _tsp_coin_swap(
    mi: NDArray[np.float64],
    rng: np.random.Generator,
    probability: float,
    passes: int,
) -> tuple[int, ...]:
    """Replace 2-opt with optional mirrored-position swaps in TSP starts.

    Every starting vertex gets a nearest-neighbor path. Within that path,
    perform ``passes`` scans of mirrored internal positions (1, n-2),
    (2, n-3), etc. A coin toss determines whether to test each move; a
    proposed swap is accepted only if it increases the chain's total MI.

    This is not 2-opt; unlike tsp, it need not find a 2-opt local optimum.
    Its main potential benefit is reduced computation, not a guarantee of
    beating the original tsp quality.
    """
    best_order: tuple[int, ...] | None = None
    best_score = -np.inf
    n = len(mi)
    for start in range(n):
        order = list(_nearest_neighbor(mi, start))
        for _ in range(passes):
            for i in range(1, n // 2):
                j = n - 1 - i
                if rng.random() < probability and _swap_gain(mi, order, i, j) > 1e-12:
                    order[i], order[j] = order[j], order[i]
        candidate = _canonical(order)
        score = _score(mi, candidate)
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12
            and (best_order is None or candidate < best_order)
        ):
            best_order, best_score = candidate, score
    assert best_order is not None
    return best_order


def _tsp_heuristic(mi: NDArray[np.float64]) -> tuple[int, ...]:
    """Run nearest-neighbor and open-path 2-opt from every starting vertex."""
    best_order: tuple[int, ...] | None = None
    best_score = -np.inf
    for start in range(len(mi)):
        order = _two_opt(mi, _nearest_neighbor(mi, start))
        score = _score(mi, order)
        if (score > best_score + 1e-12) or (
            abs(score - best_score) <= 1e-12
            and (best_order is None or order < best_order)
        ):
            best_order, best_score = order, score
    assert best_order is not None  # At least two vertices were validated.
    return best_order


def build_chain_graph(
    mi_matrix: ArrayLike | CopulaWeightMatrices,
    *,
    method: ChainMethod = "best",
    feature_names: Sequence[str] | None = None,
    random_seed: int | None = 0,
    random_trials: int | None = None,
    coin_probability: float = 0.5,
    coin_passes: int = 3,
) -> ChainGraph:
    """Build an undirected maximum-MI chain with one of five heuristics.

    Parameters
    ----------
    mi_matrix : array_like or CopulaWeightMatrices
        Symmetric nonnegative ``(d,d)`` MI matrix in nats, diagonal zero.
    method : {'greedy', 'tsp', 'tsp_coin_swap', 'greedy_random_swap',
              'greedy_inward_2opt', 'best'}, default 'best'
        ``greedy`` grows from the highest-MI edge at both endpoints.
        ``tsp`` uses nearest-neighbor multi-start plus open-path 2-opt.
        ``tsp_coin_swap`` uses the same multi-start nearest-neighbor seeds
        as ``tsp``, but replaces 2-opt with probabilistic mirrored swaps.
        ``greedy_random_swap`` improves greedy by random swaps.
        ``greedy_inward_2opt`` improves greedy using outer-to-inner 2-opt.
        ``best`` retains legacy behavior: the better of greedy and tsp.
    feature_names : sequence of str, optional
        Labels for a matrix; inferred from CopulaWeightMatrices if omitted.
    random_seed : int or None, default 0
        Seed for randomized variants. Set None to use nondeterministic entropy.
    random_trials : int or None, optional
        Number of attempted swaps in ``greedy_random_swap``. By default,
        ``2*d*d``; zero disables attempts. More trials increase cost.
    coin_probability : float, default 0.5
        Bernoulli probability of trying each mirrored swap in
        ``tsp_coin_swap``; must lie in [0,1].
    coin_passes : int, default 3
        Number of outside-to-inside scans in ``tsp_coin_swap``.

    Returns
    -------
    ChainGraph
        Selected path, MI edge weights and adjacency/GNN representations.

    Notes
    -----
    These are heuristics, not exact Hamiltonian-path optimizers. The two
    greedy refinements never decrease Greedy's score. The TSP coin variant
    replaces, rather than appends to, exhaustive 2-opt, and can score worse.
    The ``best`` option deliberately compares only the two original baselines
    for backward compatibility. Benchmark all five with ``CHAIN_METHODS``.

    Raises
    ------
    ValueError
        For an invalid MI matrix, labels, method, or algorithm parameters.
    """
    if method not in (*CHAIN_METHODS, "best"):
        raise ValueError(f"method must be one of {(*CHAIN_METHODS, 'best')}.")
    if random_seed is not None and (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, (int, np.integer))
        or random_seed < 0
    ):
        raise ValueError("random_seed must be a nonnegative integer or None.")
    if random_trials is not None and (
        isinstance(random_trials, bool)
        or not isinstance(random_trials, (int, np.integer))
        or random_trials < 0
    ):
        raise ValueError("random_trials must be a nonnegative integer or None.")
    if not np.isfinite(coin_probability) or not 0.0 <= coin_probability <= 1.0:
        raise ValueError("coin_probability must be between zero and one.")
    if (
        isinstance(coin_passes, bool)
        or not isinstance(coin_passes, (int, np.integer))
        or coin_passes < 0
    ):
        raise ValueError("coin_passes must be a nonnegative integer.")

    if isinstance(mi_matrix, CopulaWeightMatrices):
        if feature_names is None:
            feature_names = mi_matrix.feature_names
        values = mi_matrix.mutual_information
    else:
        values = mi_matrix
    mi, names = _validate_mi_matrix(values, feature_names)

    rng = np.random.default_rng(random_seed)
    if method == "greedy":
        order = _double_ended_greedy(mi)
        winner = "greedy"
    elif method == "tsp":
        order = _tsp_heuristic(mi)
        winner = "tsp"
    elif method == "tsp_coin_swap":
        order = _tsp_coin_swap(mi, rng, coin_probability, coin_passes)
        winner = "tsp_coin_swap"
    elif method == "greedy_random_swap":
        attempts = 2 * mi.shape[0] ** 2 if random_trials is None else int(random_trials)
        order = _greedy_random_swap(mi, attempts, rng)
        winner = "greedy_random_swap"
    elif method == "greedy_inward_2opt":
        order = _inward_two_opt(mi, _double_ended_greedy(mi))
        winner = "greedy_inward_2opt"
    else:
        greedy = _double_ended_greedy(mi)
        tsp = _tsp_heuristic(mi)
        if _score(mi, greedy) > _score(mi, tsp) + 1e-12:
            order, winner = greedy, "greedy"
        else:
            order, winner = tsp, "tsp"

    edges = tuple(
        ChainEdge(names[i], names[j], float(mi[i, j]))
        for i, j in zip(order[:-1], order[1:], strict=True)
    )
    return ChainGraph(
        order=order,
        feature_names=names,
        edges=edges,
        total_mi=float(sum(edge.mutual_information for edge in edges)),
        method=winner,
    )
