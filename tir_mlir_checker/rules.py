"""Stdlib-only capability inventory and conservative cross-stage rules.

This module observes data, never imports/executes IR. No operator-name dispatch.
Unknown semantic constructs are reported even in branches not taken by a case.
"""
from __future__ import annotations

import ast
import re

from .capabilities import (AFFINE_MATMUL_MAP, MAP_ALIAS, TIR_CALLS, TIR_METADATA, TIR_SCALARS, MLIR_MATMUL,
                           MLIR_OPS, LOW_PRECISION, TIR_BATCH_INFIX, TIR_BATCH2_TILES, TIR_INTRINSICS)
from .capabilities import TIR_REDUCE_COMBINE
from .diagnostics import finding, locate
from .trace_io import STAGES
from .tir_tail_contracts import script_tail_issues


def name(n):
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        return name(n.value) + '.' + n.attr
    return ''


def runtime_nodes(nodes):
    for n in nodes:
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and name(n.value.func) in TIR_METADATA:
            continue
        yield n
        yield from runtime_nodes(ast.iter_child_nodes(n))


def tir_inventory(source, stage, *, backend='ast'):
    issues = []
    facts = {'loops': [], 'reductions': [], 'conversions': [], 'operations': [], 'allocations': []}
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return facts, [finding('CAP-TIR-SYNTAX', 'unsupported', 'static', str(exc), stage=stage, line=exc.lineno)]
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
             and any(name(d) == 'T.prim_func' for d in n.decorator_list)]
    if len(funcs) != 1:
        return facts, [finding('CAP-TIR-FUNCTION', 'unsupported', 'static', '需要一个 @T.prim_func', stage=stage)]

    def issue(code, node, message, status="unsupported", **details):
        issues.append(finding(code, status, 'static', message, stage=stage,
                              line=node.lineno, details=details))

    object_calls = {'T.grid','T.init','T.axis.spatial','T.axis.reduce','T.axis.remap'} if backend=='tvm' else set()

    def walk(nodes, loops=()):
        for n in nodes:
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and name(n.value.func) in TIR_METADATA:
                continue
            if isinstance(n, ast.For):
                facts['loops'].append({'line': n.lineno, 'variable': ast.unparse(n.target),
                    'iterator': ast.unparse(n.iter), 'enclosing_loops': list(loops)})
                walk([n.target, n.iter], loops)
                walk(n.body + n.orelse, loops + (n.lineno,))
                continue
            if isinstance(n, ast.Call):
                op = name(n.func)
                facts['operations'].append({'line': n.lineno, 'operation': op})
                if op not in TIR_CALLS | TIR_SCALARS | object_calls:
                    issue('CAP-TIR-CALL', n, f'未支持调用 {op}', operation=op)
                if op == 'T.call_intrin' and (
                        len(n.args)<2
                        or not all(isinstance(a,ast.Constant) and isinstance(a.value,str) for a in n.args[:2])
                        or n.args[1].value not in TIR_INTRINSICS
                        or (n.keywords and (n.args[1].value not in TIR_BATCH2_TILES or
                                            any(k.arg!='annotations' for k in n.keywords)))):
                    issue('CAP-TIR-CALL', n, 'T.call_intrin 仅支持明确列出的字面量标量操作与 TileOp', operation=op)
                if op == 'T.reduce':
                    args = [ast.unparse(a) for a in n.args]
                    facts['reductions'].append({'line': n.lineno, 'arguments': args,
                        'enclosing_loops': list(loops), 'text': ast.unparse(n)})
                    if len(n.args) != 5 or n.keywords:
                        issue('CAP-REDUCE-FORM', n, '当前支持 reduce(src,dst,kind,axis,clear) 五参数形式')
                    elif not isinstance(n.args[2], ast.Constant) or not isinstance(n.args[2].value,str) or n.args[2].value not in TIR_REDUCE_COMBINE:
                        issue('CAP-REDUCE-KIND', n, '归约类型须为受支持的静态字符串', kind=args[2])
                if op == 'T.Cast':
                    facts['conversions'].append({'line': n.lineno, 'text': ast.unparse(n),
                        'target': ast.unparse(n.args[0]) if n.args else '?', 'enclosing_loops': list(loops)})
                if op in TIR_SCALARS and n.args:
                    facts['conversions'].append({'line': n.lineno, 'text': ast.unparse(n),
                        'target': op[2:], 'enclosing_loops': list(loops)})
                if op == 'T.copy' and (len(n.args) != 2 or any(k.arg!='disable_tma' for k in n.keywords)):
                    issue('CAP-COPY-FORM', n, 'copy 附加参数/属性尚未支持', text=ast.unparse(n))
                if op in {'T.match_buffer', 'T.alloc_buffer'}:
                    facts['allocations'].append({'line': n.lineno, 'text': ast.unparse(n)})
                    # Dtype is a string in saved TVMScript; annotations are not traversed.
                    for a in list(n.args) + [k.value for k in n.keywords if k.arg == 'dtype']:
                        if isinstance(a, ast.Constant) and a.value in LOW_PRECISION:
                            facts.setdefault('low_precision', []).append({'line':n.lineno,'dtype':a.value,'requires':'ml_dtypes for bf16'})
                if op == 'T.Cast' and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value in LOW_PRECISION:
                    facts.setdefault('low_precision', []).append({'line':n.lineno,'dtype':n.args[0].value})
            if (isinstance(n, ast.BinOp) and type(n.op).__name__ not in TIR_BATCH_INFIX
                    and type(n.op) not in {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod}):
                issue('CAP-TIR-EXPRESSION', n, '此二元运算尚未支持', operation=type(n.op).__name__)
            if isinstance(n, (ast.Assign, ast.AnnAssign)) and isinstance(n.value, ast.Call):
                call = n.value
                if name(call.func) == 'T.launch_thread' and len(call.args) == 2:
                    target = n.targets[0] if isinstance(n, ast.Assign) else n.target
                    axis, extent = call.args
                    if isinstance(axis, ast.Constant) and str(axis.value).startswith('threadIdx.'):
                        used = [v for v in runtime_nodes(funcs[0].body) if isinstance(v, ast.Name)
                                and isinstance(v.ctx, ast.Load) and v.id == name(target)]
                        if used and not (isinstance(extent, ast.Constant) and extent.value == 1):
                            issue('CAP-PHYSICAL-THREAD', n, '物理线程索引参与计算；需执行一维协作调度并验证布局与 barrier', variable=name(target), status='uncertain')
            walk(ast.iter_child_nodes(n), loops)
    walk(funcs[0].body)
    for node,op,message in script_tail_issues(funcs[0],object_backend=backend=='tvm'):
        issue('CAP-TIR-CALL',node,message,operation=op)
    return facts, issues


def mlir_inventory(source, stage, observed, *, backend='text'):
    facts = {'loops': [], 'reductions': [], 'conversions': [], 'operations': observed['events'], 'allocations': []}
    issues = []
    for event in observed['events']:
        op, line, text = event['operation'], event['line'], event['text']
        if op not in MLIR_OPS and op != 'func.func':
            issues.append(finding('CAP-MLIR-OP', 'unsupported', 'static', f'未支持操作 {op}',
                                 stage=stage, line=line, details={'operation': op}))
        if op in {'scf.for', 'scf.parallel', 'scf.while'}:
            facts['loops'].append({'line': line, 'text': text})
            if op != 'scf.while' and (('iter_args' in text and backend != 'official') or ' init ' in text):
                issues.append(finding('CAP-LOOP-ARGS', 'unsupported', 'static', '携带循环值/归约的形式尚未支持', stage=stage, line=line))
        if op == 'scf.reduce' and re.search(r'scf.reduce\s*\(', text):
            issues.append(finding('CAP-SCF-REDUCE', 'unsupported', 'static', '带值 scf.reduce 尚未支持', stage=stage, line=line))
        if op in {'arith.extf', 'arith.truncf', 'arith.index_cast', 'arith.index_castui', 'arith.sitofp', 'arith.uitofp',
                  'arith.extsi', 'arith.extui', 'arith.trunci', 'arith.fptosi', 'arith.fptoui', 'arith.bitcast'}:
            facts['conversions'].append({'line': line, 'text': text})
        if op in {'memref.alloca', 'memref.alloc', 'linalg.fill'}:
            facts['allocations'].append({'line': line, 'text': text})
    # Headers and types must be scanned as well as executable operation names.
    for line, raw in enumerate(source.splitlines(), 1):
        text = raw.split('//', 1)[0].strip()
        if 'affine_map<' in text:
            if re.fullmatch(rf'{MAP_ALIAS}\s*=\s*{AFFINE_MATMUL_MAP}',text):
                continue
            if (text.split(' ',1)[0] in MLIR_MATMUL
                    and not re.search(r'memref<[^>]*affine_map<',text)):
                continue  # Interpreter checks the exact projections and order.
            issues.append(finding('CAP-LAYOUT', 'unsupported', 'static', 'affine layout 尚未支持', stage=stage,line=line))
    return facts, issues


def analyze_rules(group, observed, *, mlir_backend='text', tir_backend='ast'):
    all_facts, findings, support = {}, [], {}
    for stage in STAGES:
        facts, issues = (tir_inventory(group.text[stage], stage, backend=tir_backend) if stage in STAGES[:3]
                         else mlir_inventory(group.text[stage], stage, observed[stage], backend=mlir_backend))
        all_facts[stage] = facts
        support[stage] = {'status': 'unsupported' if any(i['status']=='unsupported' for i in issues) else 'requires_execution' if issues else 'no_known_gap',
            'known_gap_count': len(issues),
            'scope': '操作名、类型和部分属性扫描；no_known_gap 不代表语法/执行全部支持。'}
        findings.extend(issues)
    for left, right in zip(STAGES[:3], STAGES[1:3]):
        a, b = all_facts[left]['reductions'], all_facts[right]['reductions']
        if [r['arguments'] for r in a] != [r['arguments'] for r in b]:
            findings.append(finding('REDUCE-CONTRACT-DIFFERENCE', 'uncertain', 'static',
                'TIR 阶段的归约描述有变化，需要核对轴、初始化和区域映射',
                locations=[{'stage': s, 'line': r['line']} for s, rs in [(left, a), (right, b)] for r in rs],
                details={'left': a, 'right': b}, scope='归约描述差异不自动表示数学不等价。'))
    return {'facts': all_facts, 'support': support, 'findings': locate(findings, group)}
