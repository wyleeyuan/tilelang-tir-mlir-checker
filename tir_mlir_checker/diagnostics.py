"""Stable diagnostic records. Legacy diagnostic strings remain compatible."""
from __future__ import annotations


def finding(rule_id, status, method, message, *, stage=None, line=None,
            details=None, locations=None, scope=None, next_action=None):
    return {"rule_id": rule_id, "status": status, "method": method,
            "message": message, "stage": stage,
            "locations": locations or ([{"stage": stage, "line": line}] if stage else []),
            "details": details or {},
            "scope": scope or "当前输入、已支持的 IR 子集和明确列出的条件。",
            "next_action": next_action or "根据原始行号与前提复核；不由局部结果推断整体等价。"}


def locate(items, group):
    """Map worker snapshots back to original stage filenames and excerpts."""
    records = {r['id']: r for r in group.stages}
    source_lines = {s:text.splitlines() for s,text in group.text.items()}
    for item in items:
        for loc in item['locations']:
            stage, line = loc.get('stage'), loc.get('line')
            if stage not in records:
                continue
            loc['file'] = records[stage]['file']
            lines = source_lines[stage]
            if isinstance(line, int) and 1 <= line <= len(lines):
                loc['excerpt'] = lines[line - 1]
    return items


def agent_diagnostics(result):
    """MLIR-style presentation; original rule statuses remain unchanged."""
    output = []
    for index, item in enumerate(result.get("findings", [])):
        status = item["status"]
        level = ("error" if status in {"mismatch", "input_error", "tool_error"} else
                 "warning" if status == "uncertain" else "remark")
        locations = item.get("locations", [])
        mlir_location=item.get('details',{}).get('mlir_location')
        node_location = item.get('details',{}).get('tvm_location') or mlir_location
        if node_location and locations:
            locations = [dict(locations[0],node_path=node_location.get('node_path'),
                              original_span=node_location.get('original_span'),
                              ir_location=node_location.get('ir_location'),
                              node_frontend='mlir' if mlir_location else 'tvm'), *locations[1:]]
        output.append({"id": f"F{index + 1:03d}", "severity": level,
                       "rule_id": item["rule_id"], "status": status, "method": item["method"],
                       "message": item["message"], "location": locations[0] if locations else None,
                       "notes": [{"message": "相关位置", "location": p} for p in locations[1:]],
                       "details": item.get("details", {}), "scope": item.get("scope"),
                       "next_action": item.get("next_action"),
                       "evidence_ref": f"result.json#/findings/{index}"})
    return output


def diagnostic_text(items):
    lines = []
    for item in items:
        loc = item["location"] or {}
        site = f"{loc.get('file', loc.get('stage', '<checker>'))}:{loc.get('line') or '?'}"
        lines.append(f"{site}: {item['severity']}: {item['message']} [{item['rule_id']}; {item['status']}]")
        if loc.get("excerpt"):
            lines.append("  " + loc["excerpt"])
        if loc.get('node_path'):
            lines.append('  note: '+('MLIR operation ' if loc.get('node_frontend')=='mlir' else 'TVM node ') + loc['node_path'] +
                         ('; exported-text line unavailable' if not loc.get('line') else ''))
        if loc.get('ir_location'):
            lines.append('  note: official MLIR location ' + loc['ir_location'])
        if loc.get('original_span'):
            span=loc['original_span']
            lines.append(f"  note: original source {span['source']}:{span['line']}:{span['column']} (not a Trace line)")
        for note in item["notes"]:
            p = note["location"]
            lines.append(f"{p.get('file', p.get('stage', '?'))}:{p.get('line') or '?'}: note: {note['message']}")
    return "\n".join(lines) + ("\n" if lines else "")
