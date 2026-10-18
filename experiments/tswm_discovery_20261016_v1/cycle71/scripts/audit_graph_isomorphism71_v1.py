"""Exact five-node structure-orbit audit; TRAIN/VAL metadata only, no forecasts.

This does not change C71's frozen assessment or claim a new research gap.
Physical edge labels are preserved: graph complementation is not an equivalence.
"""
import sys
import pathlib
import itertools
import collections
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
import numpy as np


def main():
    begun = time.perf_counter()
    acq = read(OUT / 'ACQUISITION71_V1.json')
    data = pathlib.Path(acq['data_root'])
    config = pathlib.Path(acq['source_root']) / 'neurips2019' / 'nri_structure.json'
    source = next(x for x in acq['sources'] if x['path'] == 'neurips2019/nri_structure.json')
    assert sha(config) == source['sha256']
    train_file = data / 'edges_springs5_train.npy'
    val_file = data / 'edges_springs5_test.npy'
    assert sha(train_file) == 'a99f886f4b057fb95372218d78e0902c51eab8f6a6c11d643311f53ab0805f12'
    assert sha(val_file) == '9b7d18a19fa6ad8c37a9d1b30b83712c77cb440466901409bd7d619c154fcfc3'
    train = np.load(train_file, mmap_mode='r', allow_pickle=False)[:2048].copy()
    val = np.load(val_file, mmap_mode='r', allow_pickle=False)[:256].copy()
    assert train.shape == (2048, 20) and val.shape == (256, 20)
    assert set(np.unique(train)) <= {0, 1} and set(np.unique(val)) <= {0, 1}
    edges = [tuple(x) for x in read(config)['edges']]
    assert len(edges) == 20 and set(edges) == {(i, j) for i in range(5) for j in range(5) if i != j}
    reverse = [edges.index((j, i)) for i, j in edges]
    assert np.array_equal(train, train[:, reverse]) and np.array_equal(val, val[:, reverse])
    pairs = list(itertools.combinations(range(5), 2))
    pair_index = {p: k for k, p in enumerate(pairs)}
    source_index = [edges.index(p) for p in pairs]
    powers = np.left_shift(np.int64(1), np.arange(10, dtype=np.int64))
    train_id = np.asarray(train[:, source_index], dtype=np.int64) @ powers
    val_id = np.asarray(val[:, source_index], dtype=np.int64) @ powers
    ids = np.arange(1024, dtype=np.int64)
    bits = ((ids[:, None] >> np.arange(10)) & 1).astype(np.int64)
    permutations = list(itertools.permutations(range(5)))
    transformed = []
    for perm in permutations:
        selectors = [pair_index[tuple(sorted((perm[i], perm[j])))] for i, j in pairs]
        transformed.append(bits[:, selectors] @ powers)
    transformed = np.stack(transformed, axis=1)
    canonical = np.min(transformed, axis=1)
    representatives = np.unique(canonical)
    assert len(representatives) == 34
    assert np.all(canonical[canonical] == canonical)
    assert np.all(canonical[transformed] == canonical[:, None])
    assert np.all(bits[canonical].sum(axis=1) == bits.sum(axis=1))

    # An independent adjacency-matrix enumeration checks every representative.
    orbit_check = []
    for rep in representatives:
        adj = np.zeros((5, 5), dtype=np.int64)
        for k, (i, j) in enumerate(pairs):
            adj[i, j] = adj[j, i] = bits[rep, k]
        orbit = set()
        for p in permutations:
            moved = adj[np.ix_(p, p)]
            code = sum(int(moved[i, j]) << k for k, (i, j) in enumerate(pairs))
            orbit.add(code)
        expected = set(np.flatnonzero(canonical == rep).tolist())
        assert orbit == expected
        orbit_check.append(len(orbit))
    assert sum(orbit_check) == 1024

    seen_train = set(train_id.tolist())
    train_orbit = canonical[train_id]
    val_orbit = canonical[val_id]
    seen_orbits = set(train_orbit.tolist())
    labeled_seen = np.array([int(x) in seen_train for x in val_id])
    isomorphic_seen = np.array([int(x) in seen_orbits for x in val_orbit])
    assert not np.any(labeled_seen & ~isomorphic_seen)
    tcounts = collections.Counter(map(int, train_orbit))
    vcounts = collections.Counter(map(int, val_orbit))
    rows = []
    for rep in representatives:
        n_edges = int(bits[rep].sum())
        degrees = np.zeros(5, dtype=int)
        for k, (i, j) in enumerate(pairs):
            degrees[i] += bits[rep, k]
            degrees[j] += bits[rep, k]
        rows.append(dict(canonical_code=int(rep), orbit_size=int(np.sum(canonical == rep)),
                         edges=n_edges, sorted_degrees=sorted(map(int, degrees)),
                         train_rows=tcounts[int(rep)], val_rows=vcounts[int(rep)],
                         train_distinct_labeled_graphs=int(np.sum(canonical[np.unique(train_id)] == rep)),
                         val_distinct_labeled_graphs=int(np.sum(canonical[np.unique(val_id)] == rep))))
    artifact = storage('artifact') / 'GRAPH_ISOMORPHISM71_V1.npz'
    assert not artifact.exists()
    np.savez_compressed(artifact, train_labeled_graph_id=train_id, val_labeled_graph_id=val_id,
                        train_canonical_orbit=train_orbit, val_canonical_orbit=val_orbit,
                        canonical_for_all_labeled_graphs=canonical,
                        val_labeled_seen=labeled_seen, val_isomorphic_seen=isomorphic_seen)
    result = dict(
        utc=utc(), status='EXACT_TRAIN_VAL_STRUCTURE_ORBIT_AUDIT_CLOSED',
        script_sha256=sha(__file__), source_config_sha256=sha(config),
        source_train_edges_sha256=sha(train_file), source_val_container_edges_sha256=sha(val_file),
        train_rows=2048, val_rows=256, reserved_eval_rows_semantically_accessed=0,
        forecasts_computed=0, checkpoint_files_accessed=0,
        equivalence='All 120 node bijections preserving physical binary edge labels; no edge-color swap',
        n_nodes=5, possible_labeled_undirected_graphs=1024, possible_unlabeled_orbits=34,
        train_distinct_labeled_graphs=len(seen_train), train_orbits=len(seen_orbits),
        val_distinct_labeled_graphs=len(np.unique(val_id)), val_orbits=len(np.unique(val_orbit)),
        val_rows_labeled_seen=int(labeled_seen.sum()),
        val_rows_isomorphic_seen=int(isomorphic_seen.sum()),
        val_rows_labeled_unseen_but_isomorphic_seen=int(np.sum(~labeled_seen & isomorphic_seen)),
        val_rows_genuinely_unseen_orbit=int(np.sum(~isomorphic_seen)),
        val_isomorphic_seen_fraction=float(isomorphic_seen.mean()),
        independent_adjacency_enumeration_matches=True,
        complete_universe_partition_verified=True, orbit_counts=rows,
        raw_artifact=dict(storage='server artifact_root/cycle71/GRAPH_ISOMORPHISM71_V1.npz', sha256=sha(artifact)),
        interpretation='Structure coverage only: node permutation does not equate continuous initial conditions, inference difficulty or forecast accuracy.',
        frozen_c71_forecast_protocol_changed=False, accepted_final_directions=0,
        elapsed_seconds=time.perf_counter() - begun,
    )
    dump(OUT / 'GRAPH_ISOMORPHISM71_V1.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'orbit_counts'}), flush=True)


if __name__ == '__main__':
    main()
