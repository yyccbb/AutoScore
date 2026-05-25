import json
import sys
from collections import Counter

MAX_DEPTH = 6
MAX_KEYS = 30
MAX_LIST_ITEMS = 3
MAX_STR_LEN = 80


def short_value(x):
    if isinstance(x, str):
        s = x.replace("\n", "\\n")
        if len(s) > MAX_STR_LEN:
            s = s[:MAX_STR_LEN] + "..."
        return f'str({len(x)}): "{s}"'
    if isinstance(x, (int, float, bool)) or x is None:
        return repr(x)
    return type(x).__name__


def inspect(obj, path="$", depth=0):
    indent = "  " * depth

    if depth > MAX_DEPTH:
        print(f"{indent}{path}: ... max depth reached")
        return

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{indent}{path}: dict, {len(keys)} keys")

        for k in keys[:MAX_KEYS]:
            v = obj[k]
            print(f"{indent}  - {k}: {type(v).__name__}")
            inspect(v, f"{path}.{k}", depth + 1)

        if len(keys) > MAX_KEYS:
            print(f"{indent}  ... {len(keys) - MAX_KEYS} more keys")

    elif isinstance(obj, list):
        print(f"{indent}{path}: list, len={len(obj)}")

        if not obj:
            return

        type_counter = Counter(type(x).__name__ for x in obj)
        print(f"{indent}  item types: {dict(type_counter)}")

        for i, item in enumerate(obj[:MAX_LIST_ITEMS]):
            inspect(item, f"{path}[{i}]", depth + 1)

        if len(obj) > MAX_LIST_ITEMS:
            print(f"{indent}  ... {len(obj) - MAX_LIST_ITEMS} more items")

    else:
        print(f"{indent}{path}: {short_value(obj)}")


def try_load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def try_load_jsonl(path, max_lines=5):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_json.py path/to/sample.json")
        sys.exit(1)

    path = sys.argv[1]

    try:
        data = try_load_json(path)
        print("Detected format: normal JSON")
        inspect(data)
    except json.JSONDecodeError:
        print("Normal JSON parse failed. Trying JSONL / NDJSON...")
        data = try_load_jsonl(path)
        print("Detected format: JSONL / NDJSON sample")
        inspect(data)


if __name__ == "__main__":
    main()
