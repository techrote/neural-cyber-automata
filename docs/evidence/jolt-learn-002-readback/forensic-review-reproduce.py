#!/usr/bin/env python3
"""Read-only stdlib forensic audit of retained LEARN-JOLT-002 bytes.

Usage: python forensic-review-reproduce.py ORIGINAL_PARENT SOURCE_REPO OUTPUT_JSON
No repository modules, Torch, model loaders, host or training commands are used.
"""
import collections
import datetime
import hashlib
import json
import pathlib
import sys

base, repo, destination = map(pathlib.Path, sys.argv[1:])
run = base / 'jolt-learn-002-20261021'
lifecycle = base / 'jolt-learn-002-20261021-lifecycle'

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def identity(path):
    digest = hashlib.sha256()
    count = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
            count += len(block)
    return {'bytes': count, 'sha256': digest.hexdigest()}

def read(name):
    return json.loads((run / name).read_bytes())

worker = read('worker-result.json')
result = read('result.json')
manifest = read('manifest.json')
audit = read('independent-readback.json')
inv = read('inventory.json')['files']
actual_paths = {p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file()}
missing = sorted(set(inv) - actual_paths)
surplus = sorted(actual_paths - set(inv) - {'inventory.json'})
mismatches = []
for name, expected in inv.items():
    path = run / name
    if path.is_file() and identity(path) != expected:
        mismatches.append({'path': name, 'expected': expected, 'actual': identity(path)})
assert not missing and not surplus and not mismatches
source_mismatches = [name for name, sha in manifest['source_files'].items()
                     if identity(repo / name)['sha256'] != sha]
assert not source_mismatches
assert hashlib.sha256(canonical(manifest['source_files'])).hexdigest() == manifest['source_sha256']

count = 0
previous = '0' * 64
events = collections.Counter()
requests = collections.Counter()
started = []
finished = []
pending = {}
ticks = 0
response_count = 0
last_request = None
last_record = None
with (run / 'owner-journal.jsonl').open('rb') as stream:
    for line in stream:
        count += 1
        assert line.endswith(b'\n') and len(line) <= 1048576
        record = json.loads(line)
        meta = record['_journal']
        claimed = meta.pop('record_sha256')
        assert meta['sequence'] == count and meta['previous_sha256'] == previous
        assert hashlib.sha256(canonical(record)).hexdigest() == claimed
        previous = claimed
        event = record['event']
        events[event] += 1
        if event == 'episode_started':
            started.append(record['id'])
        elif event == 'episode_finished':
            finished.append(record['row']['id'])
        elif event == 'owner_request':
            assert record['serial'] not in pending
            pending[record['serial']] = record
            requests[record['operation']] += 1
            last_request = {key: record[key] for key in ('serial', 'operation', 'stage', 'elapsed_seconds')}
        elif event == 'owner_response':
            request = pending.pop(record['serial'])
            response_count += 1
            if request['operation'] == 'step':
                ticks += record['response'].get('actual_ticks', 0)
        last_record = {key: value for key, value in record.items() if key in ('event', 'serial')}

reservations = read('closed-budget-reservations.json')['reservations']
counters = collections.Counter()
last_elapsed = -1.0
for index, row in enumerate(reservations):
    assert row['index'] == index and row['elapsed_seconds'] >= last_elapsed
    last_elapsed = row['elapsed_seconds']
    counters.update(row['increments'])
assert all(counters[key] == value for key, value in result['costs']['counts'].items())
assert hashlib.sha256(canonical(reservations)).hexdigest() == result['costs']['reservations_sha256']
assert identity(run / 'budget-reservations.json') == identity(run / 'closed-budget-reservations.json')
assert result['worker_result_sha256'] == identity(run / 'worker-result.json')['sha256']
assert result['audit_sha256'] == identity(run / 'independent-readback.json')['sha256']

bound = {name: identity(run / name) for name in (
    'manifest.json', 'declaration.json', 'config.json', 'worker-result.json', 'result.json',
    'independent-readback.json', 'inventory.json', 'owner-journal.jsonl',
    'budget-reservations.json', 'closed-budget-reservations.json')}
outer = {f'lifecycle/{name}': identity(lifecycle / name) for name in ('invocation.json', 'stdout.log', 'stderr.log')}
for name in ('jolt-learn-002-20261021-terminal.json', 'jolt-learn-002-20261021-command-stderr.log',
             'jolt-learn-002-training-postmortem-review-20261021.json'):
    outer[name] = identity(base / name)
source_paths = ('src/nca/evaluation/durable_journal.py', 'src/nca/evaluation/jolt_learning_audit.py',
                'src/nca/train/jolt_screen.py', 'tools/learn_jolt.py', 'tools/execute_jolt_diagnostic.py')
missing_outer = [name for name in ('executor.json', 'finalization.json') if not (lifecycle / name).exists()]
finals = [row['id'] for row in worker['checkpoints'] if row['id'].endswith('-final')]
report = {
 'schema': 'nca-learn-jolt-002-forensic-review/v1',
 'reviewer': 'independent-forensic-source-and-retained-data-review',
 'reviewed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'scope': 'Independent standard-library byte, JSON, hash-chain, reservation and frozen-source examination. No experimental replay or physical verification.',
 'accepted_scope': 'Faithful publication of the original failed, incomplete, nonqualifying allocation and its evidence limitations.',
 'original_run': {
   'source_revision': manifest['source_revision'], 'source_sha256': manifest['source_sha256'],
   'source_inventory_files': len(manifest['source_files']), 'source_inventory_matches': True,
   'declaration_canonical_sha256': manifest['declaration_sha256'],
   'created_utc': manifest['created_utc'], 'whole_process_started_monotonic': manifest['whole_process_started_monotonic'],
   'status': result['status'], 'error': result['error'],
   'learned_qualified': False, 'evidence_complete': False, 'owner_feel_accepted': False, 'morphology_transfer_admitted': False,
   'required_finals': 6, 'recorded_finals': finals,
   'recorded_open_rows': sum(row['stage'] == 'open' for row in worker['results']),
   'recorded_sealed_rows': sum(row['stage'] == 'sealed' for row in worker['results'])
 },
 'original_files': bound,
 'outer_files': outer,
 'reviewed_source_files': {name: identity(repo / name)['sha256'] for name in source_paths},
 'inventory_readback': {'indexed_files': len(inv), 'retained_files_including_inventory': len(actual_paths),
   'missing': missing, 'surplus': surplus, 'identity_mismatches': mismatches, 'passed': True,
   'meaning': 'Every indexed retained file matches the original index. This does not establish journal, physical, paired-evaluation, or executor completeness.'},
 'journal': {
   'prefix_chain_valid': True, 'records': count, 'chain_sha256': previous,
   'events': dict(events), 'requests': dict(requests), 'responses': response_count,
   'response_reported_ticks': ticks, 'pending_requests': len(pending),
   'episode_starts': len(started), 'episode_finishes': len(finished),
   'last_episode_started': started[-1], 'last_episode_finished': finished[-1],
   'last_request': last_request, 'last_record': last_record,
   'worker_claimed_durability': worker['closure']['durability'],
   'matches_worker_receipt': False,
   'complete_lifetime': False,
   'cause_of_missing_suffix': 'unknown'
 },
 'reservation_readback': {
   'sequential_and_monotonic': True, 'reservation_count': len(reservations),
   'canonical_sha256': result['costs']['reservations_sha256'],
   'before_and_after_readback_bytes_identical': True,
   'summed_increments_match_recorded_counts': True,
   'counts': result['costs']['counts'],
   'last_reservation_elapsed_seconds': last_elapsed,
   'last_reservation': reservations[-1],
   'scope': 'Internal accounting is consistent. The missing owner-journal suffix prevents a complete original-request/receipt reconciliation.'
 },
 'time_and_closure': {
   'worker_reported_wall_seconds': result['costs']['wall_seconds'],
   'worker_measurement_boundary': 'BudgetLedger.close before result.json publication, inventory sealing, external executor verification and final receipt publication.',
   'maximum_declared_whole_process_seconds': result['costs']['maximum_wall_seconds'],
   'authoritative_terminal_bytes': identity(base / 'jolt-learn-002-20261021-terminal.json')['bytes'],
   'missing_lifecycle_files': missing_outer,
   'whole_process_elapsed_seconds': None, 'original_executor_returncode': None,
   'original_executor_timeout': None, 'independent_owned_process_tree_cleanup_verified': False,
   'native_close_confirmed_by_worker': worker['closure']['native_close_confirmed'],
   'native_close_independently_authenticated': False,
   'within_whole_process_ceiling_proven': False,
   'initial_invocation_physical_execution_started_false_is_final_status': False
 },
 'original_independent_readback': audit,
 'findings': [
  {'id': 'consumed_allocation', 'classification': 'direct retained evidence',
   'assessment': 'A prospective declaration and initial invocation exist alongside 64 teacher records, learned checkpoints, training artifacts and 229 worker result rows. The one allocation has been consumed. The initial physical_execution_started=false field precedes worker construction and is never a final no-execution certificate.'},
  {'id': 'journal_suffix_missing_before_seal', 'classification': 'direct retained evidence plus source ordering',
   'assessment': 'The original independent auditor rejects the journal before any checkpoint load or neural audit forward. The later original inventory binds the same 6720-record prefix. Therefore the discrepancy was already present during the original readback and inventory sealing; it is not explained solely by a later change to the indexed file.'},
  {'id': 'journal_cause_unknown', 'classification': 'source establishes mechanism limits, not a causal diagnosis',
   'assessment': 'DurableJournal writes one linked record, flushes and fsyncs before advancing its in-memory counters. Its durability_receipt reports those counters and does not independently reread the pathname. The runner contains no intended truncation or rewrite of the live journal. The retained bytes do not establish how the missing suffix arose; neither a particular filesystem fault, path replacement, manual change nor transport failure is proven.'},
  {'id': 'inventory_not_lifetime_proof', 'classification': 'direct byte comparison',
   'assessment': 'The inventory exactly matches all retained indexed files, including the incomplete journal. A passing archive or inventory comparison authenticates the saved package, not the full owner lifetime or the claimed 24195-record closure.'},
  {'id': 'missing_external_completion', 'classification': 'direct absence plus source control-flow analysis',
   'assessment': 'The outer executor source normally publishes executor.json, finalization.json and terminal JSON even when complete=false. All three final receipts are absent or empty. The worker printed status=failed but this is not the executor completion verdict. Cause and exit state of the outer executor are unestablished; a worker ValueError alone does not explain why the outer failure receipts are missing.'},
  {'id': 'partial_time_accounting', 'classification': 'direct metadata with explicit limit',
   'assessment': '963.0812728359988 seconds is the worker ledger close interval from the common original start. It precedes subsequent finalization work. No authoritative final whole-process wall interval, timeout value or owned-tree cleanup receipt is retained, so a completed under-1800-second invocation cannot be claimed.'},
  {'id': 'qualification_unavailable', 'classification': 'direct contract failure',
   'assessment': 'Five of six required final checkpoints exist and neither open nor sealed comparisons began. There are no complete paired outcomes or confidence bounds. The original failed readback and failed result remain authoritative. Learned reliability, own-initial gain, authored-relative gain, character and held-out runtime qualification are unmeasured; no learned family, owner feel or morphology is accepted.'}
 ],
 'disposition': {
   'publish_original_bytes_unchanged': True,
   'publish_original_failed_verdict_unchanged': True,
   'additional_physical_invocation_admitted': False,
   'resume_or_retry_admitted': False,
   'retuning_reseeding_extension_or_checkpoint_selection_admitted': False,
   'merge_scope': 'Reviewed source plus truthful failed-allocation evidence and reconciliation, subject to final-head CI and independent publication review; no successful experiment or learned qualification claim.',
   'future_source_work': 'Separately review the diagnostics and durability evidence paths. Do not repair old receipts, extend the consumed allocation or claim a diagnosed persistence cause without evidence.'
 },
 'review_operations': {'stdlib_only': True, 'torch_imports': 0, 'host_starts': 0, 'physical_ticks': 0,
   'checkpoint_deserializations': 0, 'model_constructions': 0, 'neural_forwards': 0,
   'optimizer_calls': 0, 'original_artifact_mutations': 0, 'original_auditor_replays': 0,
   'original_executable_source_mutations': 0, 'github_mutations': 0, 'retry_started': False}
}
destination.write_text(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
print(json.dumps({'output': destination.name, 'identity': identity(destination), 'inventory_files': len(inv),
                 'journal_records': count, 'reservation_records': len(reservations), 'qualified': False}))
