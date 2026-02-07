#!/usr/bin/env python3
"""Patch function_call_parser.py to add glm47 parser"""
import sys

filepath = sys.argv[1]
with open(filepath, 'r') as f:
    content = f.read()

if 'glm47' in content:
    print("Already has glm47")
    sys.exit(0)

# Add import
old_import = "from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector"
new_import = old_import + "\nfrom sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector"
content = content.replace(old_import, new_import)

# Add to parser enum
old_enum = '"glm45": Glm4MoeDetector,'
new_enum = old_enum + '\n        "glm47": Glm47MoeDetector,'
content = content.replace(old_enum, new_enum)

with open(filepath, 'w') as f:
    f.write(content)
print("Patched successfully")
