"""Close the finite structure audit and append bounded findings on the server."""
import sys
import pathlib
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *

a = read(OUT / 'GRAPH_ISOMORPHISM71_V1.json')
assert a['status'] == 'EXACT_TRAIN_VAL_STRUCTURE_ORBIT_AUDIT_CLOSED'
assert a['independent_adjacency_enumeration_matches'] and a['complete_universe_partition_verified']
assert a['forecasts_computed'] == a['checkpoint_files_accessed'] == a['reserved_eval_rows_semantically_accessed'] == 0
assert a['train_orbits'] == a['possible_unlabeled_orbits'] == 34
assert a['val_rows_genuinely_unseen_orbit'] == 0
assert sum(row['train_rows'] for row in a['orbit_counts']) == a['train_rows']
assert sum(row['val_rows'] for row in a['orbit_counts']) == a['val_rows']
assert sha(ROOT / 'scripts' / 'audit_graph_isomorphism71_v1.py') == a['script_sha256']
raw_artifact = storage('artifact') / 'GRAPH_ISOMORPHISM71_V1.npz'
assert sha(raw_artifact) == a['raw_artifact']['sha256']

report = f'''# Exact structure-orbit coverage in the current native baseline

## Observed result

The declared TRAIN contains {a['train_rows']} trajectories with {a['train_distinct_labeled_graphs']} distinct labeled graphs. Its graphs cover all {a['train_orbits']} non-isomorphic simple undirected binary graph structures on five nodes. The {a['val_rows']}-trajectory VAL contains {a['val_distinct_labeled_graphs']} labeled graphs and {a['val_orbits']} structure orbits. Of these trajectories, {a['val_rows_labeled_seen']} have an exactly matching labeled TRAIN graph. All remaining {a['val_rows_labeled_unseen_but_isomorphic_seen']} have a TRAIN graph with the same topology after a node bijection. Genuinely unseen VAL structure orbits: {a['val_rows_genuinely_unseen_orbit']}. Thus label-hash exclusion alone would not establish unseen-topology generalization for these already-trained weights.

## What was computed

`scripts/audit_graph_isomorphism71_v1.py` verifies the pinned source edge order and both canonical label-file hashes. It reads TRAIN0:2048 and development VAL0:256; reserved EVAL labels are not semantically accessed. The already-available label container is hashed for integrity only. No state trajectory, checkpoint or prediction is read. The algorithm enumerates all120 node permutations of every one of the1024 possible five-node undirected binary graphs, preserving physical edge colors. Complementation or swapping absent/present edge meanings is not an allowed equivalence. The canonical representative is the minimum ten-bit code in an orbit. A separate adjacency-matrix enumeration checks every orbit; the orbit sizes partition all1024 graphs.

The complete per-orbit counts and script/source/artifact hashes are in `outputs/GRAPH_ISOMORPHISM71_V1.json`. Per-row codes and the complete canonical lookup remain in the server artifact store. This is an exhaustive metadata result, not a sampled numerical forecast comparison.

## Consequence for the research decision

The baseline still tests prediction on new continuous trajectories and permits the already frozen free-simulation competence assessment. Structure isomorphism does not make initial conditions identical, prove the proposal network equivariant, prove that the model memorizes, or determine forecast accuracy. No claim about those behaviors follows from graph coverage. This audit does not show that the original paper's stated reproduction results are wrong; the current2048-task development protocol is distinct from its full experiment.

The proposed shortcut of discarding the214 labeled-overlap VAL trajectories and calling the remaining42 an unseen-composition test is invalid for the specific unseen-topology claim. Moving those42 rows into another partition after this model has learned all34 orbits does not fix it. Acquiring more five-node graphs from the same complete binary structural family does not create a topology absent from this training set.

If a later experiment specifically claims transfer to unseen topology, it must assign complete graph-isomorphism orbits to TRAIN/development/EVAL before fitting any shared library, representation, proposal or calibrator. If nodes have different physical roles, equivalence must preserve those roles instead of permuting all nodes. The protocol must report primitive coverage, topology coverage, initial-state and parameter distributions, observation interfaces and permitted adaptation separately. A topology shift is not automatically a new physical law. Selection must use development data and keep independent EVAL closed.

Such a protocol is necessary evidence hygiene, not an accepted new method or benchmark contribution. A new mechanism still needs a residual beyond competent ModularNRI plus observation/state inference, ordinary alignment and same-information capacity/compute controls. Current source-based relational training and its frozen forecast assessment continue unchanged. There is no new GPU fit, tuning against held-out forecasts or accepted final research direction in this audit.

## Decision and next discriminator

1. Finish all three already registered native fits and their frozen free-VAL assessment. Do not change the forecast arms or checkpoint rule from this metadata finding.
2. Do not use the current split or relabeled leftovers to claim unseen structural composition.
3. Only if the complete-baseline evidence warrants this route, design a new orbit-disjoint training cohort before a new model fit. Failure caused only by a harder split, extra sensor uncertainty, insufficient data or inference compute does not establish the sought novel gap.

Accepted final directions:0. Full research objective:ACTIVE / UNACHIEVED. Raw arrays and weights remain on the server; GitHub receives authored code, compact evidence and reports.
'''

updates = {
    'literature_review.md': 'An exhaustive metadata audit supplies a local protocol boundary, not a literature novelty claim. STRUCTURE_ORBIT_FINDING71_V1.md separates graph topology from continuous initial-state generalization. The original full ModularNRI experiment is not being declared invalid by this subset audit.',
    'baseline_registry.md': f'Exact TRAIN/VAL structure audit closed: TRAIN covers all{a["train_orbits"]} five-node binary undirected graph orbits; VAL has{a["val_orbits"]}, all seen structurally. The existing native fits remain new-trajectory baseline development, not an unseen-topology experiment. No fit, checkpoint selector or frozen forecast arm changed.',
    'experimental_results.md': f'GRAPH_ISOMORPHISM71_V1.json: {a["train_distinct_labeled_graphs"]} labeled TRAIN graphs cover34/34 possible node-permutation orbits. VAL has{a["val_distinct_labeled_graphs"]} labeled graphs/{a["val_orbits"]} orbits; {a["val_rows_labeled_seen"]}/{a["val_rows"]} exact labeled overlaps, all{a["val_rows_labeled_unseen_but_isomorphic_seen"]} remaining rows isomorphic to TRAIN. Unseen VAL orbits0. Exhaustive120-permutation and independent adjacency enumeration agree. No forecasts or EVAL rows were semantically accessed.',
    'negative_evidence.md': 'Rejected protocol shortcut: selecting the42 label-hash-unseen VAL rows does not yield an unseen-topology test; all42 are node permutations of TRAIN graphs. A post-training row repartition cannot undo exposure to every one of the34 structural orbits. This rejects that evidence interpretation, not new-trajectory prediction or the general potential of modular models.',
    'hypothesis_space.md': 'The hypothesis that labeled-graph exclusion in this already-trained five-node cohort creates an unseen-topology discriminator is contradicted by the exhaustive orbit audit. New continuous-state generalization, proposal equivariance, memorization and forecast quality remain separate untested explanations. An orbit-disjoint fit would be a new experiment requiring pre-training partitioning.',
    'research_ideas.md': 'CandidateC cannot be rescued by graph-hash exclusion alone: all34 structural orbit types are already in TRAIN. For candidateA, a later claim about unseen topology requires a new fit with complete orbit exclusion before any shared representation/library/proposal training. This metadata rule is ordinary evaluation hygiene and supplies no distinct new learning mechanism. Continue the frozen competence assessment before committing to another composition experiment.',
    'final_research_direction.md': 'No final direction accepted. The new exact orbit audit rules out interpreting the current split or its42 label-hash-unseen leftovers as unseen-topology transfer: every VAL topology is already represented in TRAIN up to node permutation. The next live discriminator remains the unchanged complete-model free-simulation assessment. Any later topology claim needs an independently defined orbit-disjoint fit, not retrospective relabeling of this baseline.',
}
heading = '\n\n## Exact structure-orbit audit, C71\n\n'
previous = OUT / 'publication_recovery_v1' / 'files'
planned = {}
for name, text in updates.items():
    original = (previous / name).read_bytes()
    assert original == (ROOT / name).read_bytes(), (name, 'Ledger differs from last verified publication')
    assert heading.encode() not in original
    planned[name] = (original, (heading + text + '\n').encode())
assert not (ROOT / 'STRUCTURE_ORBIT_FINDING71_V1.md').exists()
assert not (OUT / 'STRUCTURE_ORBIT_CLOSURE71_V1.json').exists()
assert not (OUT / 'publication_structure_orbit_v1').exists()
with (ROOT / 'STRUCTURE_ORBIT_FINDING71_V1.md').open('x', encoding='utf-8') as f:
    f.write(report)
ledger_proofs = {}
for name, (original, suffix) in planned.items():
    with (ROOT / name).open('ab') as f:
        f.write(suffix)
    after = (ROOT / name).read_bytes()
    assert after == original + suffix
    ledger_proofs[name] = dict(before_sha256=hashlib.sha256(original).hexdigest(),
                              after_sha256=hashlib.sha256(after).hexdigest(),
                              original_bytes=len(original), appended_bytes=len(suffix), prefix_preserved=True)
closure = dict(utc=utc(), status='EXACT_ORBIT_AUDIT_CLOSED_PROTOCOL_SHORTCUT_REJECTED',
               audit_sha256=sha(OUT / 'GRAPH_ISOMORPHISM71_V1.json'),
               report_sha256=sha(ROOT / 'STRUCTURE_ORBIT_FINDING71_V1.md'),
               ledgers=ledger_proofs, forecasts_computed=0,
               reserved_eval_rows_semantically_accessed=0, new_method=False,
               native_training_protocol_changed=False, frozen_forecast_protocol_changed=False,
               accepted_final_directions=0, overall_goal='ACTIVE_UNACHIEVED')
dump(OUT / 'STRUCTURE_ORBIT_CLOSURE71_V1.json', closure)
status = '''# C71: topology coverage corrected; baseline training and research remain open

The exact metadata audit finds TRAIN already covers all34 non-isomorphic five-node binary undirected graph structures. All42 VAL rows without an exact labeled-graph match are node permutations of TRAIN graphs; unseen VAL topology count0. See cycle71/STRUCTURE_ORBIT_FINDING71_V1.md and its closed machine-readable evidence. This does not measure prediction accuracy or prove memorization. The frozen three-fit competence/forecast protocol is unchanged; unseen-topology evidence would require a new pre-training orbit-disjoint split and new fits.

The previous driver recovery and its exact continuation qualification remain documented in CURRENT_RESEARCH_STATUS_V71_RECOVERY.md. Live training state must be checked from actual process identities and current terminal receipts; neither status file is a completion claim. No final research direction accepted; full goal ACTIVE/UNACHIEVED. Reserved EVAL remains closed. Raw data, arrays and weights stay on the server; GitHub receives source, reports and compact closed evidence.
'''
with (PROJECT / 'CURRENT_RESEARCH_STATUS_V71_STRUCTURE.md').open('x', encoding='utf-8') as f:
    f.write(status)

dest = OUT / 'publication_structure_orbit_v1'
dest.mkdir(exist_ok=False)
selected = ([ROOT / x for x in updates]
            + [ROOT / 'STRUCTURE_ORBIT_FINDING71_V1.md', ROOT / 'scripts' / 'audit_graph_isomorphism71_v1.py',
               ROOT / 'scripts' / 'close_structure_orbit71_v1.py', OUT / 'GRAPH_ISOMORPHISM71_V1.json',
               OUT / 'STRUCTURE_ORBIT_CLOSURE71_V1.json'])
patterns = json.loads(os.environ['C71_PRIVATE_EXPORT_PATTERNS_JSON'])
rows = []
for p in sorted(selected):
    raw = p.read_bytes()
    rel = p.relative_to(ROOT).as_posix()
    for pattern in patterns:
        assert pattern.encode() not in raw, (rel, 'Private export pattern')
    text = raw.decode('utf-8-sig')
    b = text.encode('utf-8')
    target = dest / 'files' / rel
    target.parent.mkdir(exist_ok=True, parents=True)
    target.write_bytes(b)
    rows.append(dict(path=rel, bytes=len(b), sha256=hashlib.sha256(b).hexdigest(),
                     git_blob_sha1=hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest(),
                     server_snapshot_file=str(target), parts=math.ceil(len(text) / 3500)))
manifest = dict(utc=utc(), status=closure['status'], files=rows, file_count=len(rows),
                bytes=sum(r['bytes'] for r in rows), raw_arrays_weights_or_live_logs_included=False,
                forecasts_computed=0, reserved_eval_rows_semantically_accessed=0,
                accepted_final_directions=0, overall_goal='ACTIVE_UNACHIEVED',
                local_result_data_or_model_files_written=0)
dump(dest / 'manifest.json', manifest)
print(json.dumps(dict(status=manifest['status'], files=len(rows), bytes=manifest['bytes'],
                      manifest_sha256=sha(dest / 'manifest.json'), ledgers_preserved=len(ledger_proofs))), flush=True)
