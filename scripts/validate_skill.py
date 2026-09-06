#!/usr/bin/env python3
from pathlib import Path
import re
import sys

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("SKILL.md")
text = path.read_text(encoding="utf-8")

if not text.startswith("---\n"):
    raise SystemExit("ERROR: SKILL.md must begin with YAML frontmatter.")

parts = text.split("---", 2)
if len(parts) < 3:
    raise SystemExit("ERROR: YAML frontmatter is not closed.")

frontmatter = parts[1]
for key in ("name:", "description:"):
    if key not in frontmatter:
        raise SystemExit(f"ERROR: missing required field: {key[:-1]}")

name_match = re.search(r"^name:\s*([^\n]+)$", frontmatter, re.MULTILINE)
if not name_match:
    raise SystemExit("ERROR: invalid or missing skill name.")

name = name_match.group(1).strip()
if not re.fullmatch(r"[a-z0-9-]+", name):
    raise SystemExit("ERROR: name must use lowercase letters, numbers, and hyphens only.")

if path.parent.name != name:
    print(f"WARNING: folder '{path.parent.name}' does not match skill name '{name}'.")

print("SKILL.md basic validation passed.")
