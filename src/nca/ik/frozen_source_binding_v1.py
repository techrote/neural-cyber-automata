"""Narrow provenance amendment for the already sealed, unopened e2aec1f suite.

The only production edits admitted here are provenance-validation functions.
AST equality enforces unchanged generation, isolation, inference and scoring.
Two independent admissions must still bind the exact qualification revision.
"""
import ast
from pathlib import Path
import subprocess
from nca.contracts import sha256_json

PREPARED_HEAD='e2aec1f2a6f8365193c17cfa3cd14549f4b44693'
PREPARED_CLOSURE='be8028af4c8b8f517fa86c5eb6408d94d18a9cd477656c7f0ca49069d9c835e3'
SELF='src/nca/ik/frozen_source_binding_v1.py'
ALLOWED={'src/nca/train/ik_frozen_seal_v1.py':{'load'},
         'src/nca/evaluation/ik54_frozen_final_v1.py':{'verify_proof','prepare'}}
APPROVED_FILES={
    'src/nca/train/ik_frozen_seal_v1.py':'4330ffe03b4e578ea5e4e00ea688eb1a4b69ba56819aa9170c37b88f327924b1',
    'src/nca/evaluation/ik54_frozen_final_v1.py':'3a16f608000ba532deb230e1183e62229c497e235d1d731352292de7bee0356e'}


def nonbinding_ast(source,allowed):
    result=[];binding_imports=0
    for node in ast.parse(source).body:
        if isinstance(node,ast.ImportFrom) and node.module=='nca.ik.frozen_source_binding_v1':
            binding_imports+=1
            if binding_imports!=1 or [(v.name,v.asname) for v in node.names]!=[('source_binding',None)]:
                raise ValueError('only one unaliased provenance import is permitted')
            continue
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in allowed:continue
        result.append(ast.dump(node,include_attributes=False))
    return result


def source_binding(prepared,current):
    if prepared==current:return {'mode':'exact_source','prepared_head':prepared['revision'],'qualification_head':current['revision']}
    if (prepared.get('revision')!=PREPARED_HEAD or prepared.get('source_sha256')!=PREPARED_CLOSURE
        or sha256_json(prepared.get('files'))!=PREPARED_CLOSURE):raise ValueError('unknown prepared source amendment')
    before=prepared['files'];after=current['files']
    changed={k for k in before if before[k]!=after.get(k)}
    if set(after)-set(before)!={SELF} or set(before)-set(after) or changed!=set(ALLOWED):
        raise ValueError('amendment changes production files beyond provenance binding')
    if any(after[path]!=expected for path,expected in APPROVED_FILES.items()):
        raise ValueError('amendment differs from independently reviewed provenance file bytes')
    checkout=Path(__file__).resolve().parents[3]
    for path,functions in ALLOWED.items():
        original=subprocess.check_output(['git','show',f'{PREPARED_HEAD}:{path}'],cwd=checkout).decode('utf-8')
        revised=(checkout/path).read_text(encoding='utf-8')
        if nonbinding_ast(original,functions)!=nonbinding_ast(revised,functions):
            raise ValueError('amendment changes generation, isolation, inference or scoring AST')
    return {'mode':'test_and_provenance_only','prepared_head':PREPARED_HEAD,
        'prepared_source_sha256':PREPARED_CLOSURE,'qualification_head':current['revision'],
        'qualification_source_sha256':current['source_sha256'],
        'allowed_provenance_functions':{k:sorted(v) for k,v in ALLOWED.items()},
        'unchanged_generation_isolation_inference_scoring':True}


# Copyright 2026 techrote. All rights reserved.
