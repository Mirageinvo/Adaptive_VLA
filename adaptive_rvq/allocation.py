"""Allocation algorithms for per-position RVQ depth maps."""

from __future__ import annotations

import heapq
import itertools

import numpy as np
import torch


def legal_moves(depth: torch.Tensor, max_depth: int = 3) -> list[int]:
    """Return positions where one nested level can still be added."""

    if depth.ndim != 1:
        raise ValueError(f"Expected 1D depth map, got {tuple(depth.shape)}")
    return [int(i) for i in range(len(depth)) if int(depth[i]) < max_depth]


def round_robin_depth_map(n_positions: int, budget: int, max_depth: int = 3) -> torch.Tensor:
    depth = torch.ones(n_positions, dtype=torch.long)
    total = int(depth.sum())
    pos = 0
    while total < budget:
        if depth[pos] < max_depth:
            depth[pos] += 1
            total += 1
        pos = (pos + 1) % n_positions
    return depth


def random_depth_map(
    n_positions: int,
    budget: int,
    rng: np.random.Generator,
    max_depth: int = 3,
) -> torch.Tensor:
    depth = torch.ones(n_positions, dtype=torch.long)
    total = int(depth.sum())
    while total < budget:
        legal = np.flatnonzero(depth.numpy() < max_depth)
        pos = int(rng.choice(legal))
        depth[pos] += 1
        total += 1
    return depth


def enumerate_depths_for_budget(
    n_positions: int,
    budget: int,
    max_depth: int = 3,
) -> list[tuple[int, ...]]:
    """Enumerate all legal depth maps with exact token budget."""

    out: list[tuple[int, ...]] = []

    def rec(pos: int, remaining: int, cur: list[int]) -> None:
        if pos == n_positions:
            if remaining == 0:
                out.append(tuple(cur))
            return
        min_left = n_positions - pos - 1
        max_left = (n_positions - pos - 1) * max_depth
        for depth in range(1, max_depth + 1):
            rem = remaining - depth
            if rem < min_left or rem > max_left:
                continue
            cur.append(depth)
            rec(pos + 1, rem, cur)
            cur.pop()

    rec(0, budget, [])
    return out


def greedy_oracle_single(score_depth, n_positions: int, budgets: list[int], max_depth: int = 3):
    """Greedy exact-budget path for one chunk.

    score_depth(depth) -> scalar error, lower is better.
    """

    depth = torch.ones(n_positions, dtype=torch.long)
    current_err = float(score_depth(depth))
    history = [{"budget": int(depth.sum()), "depth": depth.clone(), "error": current_err, "gain": 0.0}]
    out = {int(depth.sum()): {"depth": depth.clone(), "error": current_err, "gain": 0.0}}

    while int(depth.sum()) < max(budgets):
        best = None
        for pos in legal_moves(depth, max_depth=max_depth):
            cand = depth.clone()
            cand[pos] += 1
            err = float(score_depth(cand))
            gain = current_err - err
            if best is None or gain > best["gain"]:
                best = {"pos": pos, "depth": cand, "error": err, "gain": gain}
        if best is None:
            break
        depth = best["depth"]
        current_err = best["error"]
        budget = int(depth.sum())
        history.append({"budget": budget, "depth": depth.clone(), "error": current_err, "gain": best["gain"]})
        out[budget] = {"depth": depth.clone(), "error": current_err, "gain": best["gain"], "last_pos": best["pos"]}
    return out, history


def greedy_oracle_single_mode(
    score_depth,
    n_positions: int,
    budgets: list[int],
    mode: str = "exact-budget",
    max_depth: int = 3,
):
    """Greedy path with exact-budget or at-most-budget stopping."""

    if mode not in {"exact-budget", "at-most-budget"}:
        raise ValueError(f"Unknown mode: {mode}")
    depth = torch.ones(n_positions, dtype=torch.long)
    current_err = float(score_depth(depth))
    history = [{"budget": int(depth.sum()), "depth": depth.clone(), "error": current_err, "gain": 0.0}]
    out = {int(depth.sum()): {"depth": depth.clone(), "error": current_err, "gain": 0.0}}
    while int(depth.sum()) < max(budgets):
        best = None
        for pos in legal_moves(depth, max_depth=max_depth):
            cand = depth.clone()
            cand[pos] += 1
            err = float(score_depth(cand))
            gain = current_err - err
            if best is None or gain > best["gain"]:
                best = {"pos": pos, "depth": cand, "error": err, "gain": gain}
        if best is None:
            break
        if mode == "at-most-budget" and best["gain"] <= 0:
            break
        depth = best["depth"]
        current_err = best["error"]
        budget = int(depth.sum())
        history.append({"budget": budget, "depth": depth.clone(), "error": current_err, "gain": best["gain"]})
        out[budget] = {"depth": depth.clone(), "error": current_err, "gain": best["gain"], "last_pos": best["pos"]}
    return out, history


def validate_depth_map(depth: torch.Tensor, budget: int | None = None, max_depth: int = 3) -> None:
    if depth.ndim != 1:
        raise ValueError(f"Depth must be 1D, got {tuple(depth.shape)}")
    if int(depth.min().item()) < 1 or int(depth.max().item()) > max_depth:
        raise ValueError(f"Depth values must be in [1,{max_depth}]")
    if budget is not None and int(depth.sum().item()) != int(budget):
        raise ValueError(f"Budget mismatch: expected {budget}, got {int(depth.sum().item())}")


def beam_search_depth(score_depth, n_positions: int, budget: int, beam_width: int = 64, max_depth: int = 3):
    """Beam search over legal nested additions up to exact budget."""

    start = tuple([1] * n_positions)
    beams: dict[tuple[int, ...], float] = {start: float(score_depth(torch.tensor(start, dtype=torch.long)))}
    while sum(next(iter(beams))) < budget:
        cand_heap = []
        for state, _ in beams.items():
            state_t = torch.tensor(state, dtype=torch.long)
            for pos in legal_moves(state_t, max_depth=max_depth):
                nxt = list(state)
                nxt[pos] += 1
                nxt_tup = tuple(nxt)
                err = float(score_depth(torch.tensor(nxt_tup, dtype=torch.long)))
                cand_heap.append((err, nxt_tup))
        next_beams = {}
        for err, state in heapq.nsmallest(beam_width, cand_heap):
            if state not in next_beams or err < next_beams[state]:
                next_beams[state] = err
        beams = next_beams
    best_state = min(beams, key=beams.get)
    return torch.tensor(best_state, dtype=torch.long), float(beams[best_state])


def exact_search_subset(
    score_subset_depth,
    subset_positions: list[int],
    full_depth: torch.Tensor,
    budget: int,
    max_depth: int = 3,
):
    """Exact search on a subset of positions while keeping others fixed."""

    subset_budget = budget - int(full_depth.sum().item()) + len(subset_positions)
    best_depth = None
    best_err = None
    for state in enumerate_depths_for_budget(len(subset_positions), subset_budget, max_depth=max_depth):
        cand = full_depth.clone()
        for pos, value in zip(subset_positions, state):
            cand[pos] = int(value)
        err = float(score_subset_depth(cand))
        if best_err is None or err < best_err:
            best_err = err
            best_depth = cand.clone()
    if best_depth is None:
        raise RuntimeError("Exact search found no candidates")
    return best_depth, float(best_err)
