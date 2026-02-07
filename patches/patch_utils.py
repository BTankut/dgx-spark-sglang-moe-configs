#!/usr/bin/env python3
"""Patch utils.py to add infer_type_from_json_schema"""
import sys

FUNC_CODE = '''

def infer_type_from_json_schema(schema):
    """Infer the primary type of a parameter from JSON Schema."""
    if not isinstance(schema, dict):
        return None
    if "type" in schema:
        type_value = schema["type"]
        if isinstance(type_value, str):
            return type_value
        elif isinstance(type_value, list) and type_value:
            non_null_types = [t for t in type_value if t != "null"]
            if non_null_types:
                return non_null_types[0]
            return "string"
    if "anyOf" in schema or "oneOf" in schema:
        schemas = schema.get("anyOf") or schema.get("oneOf")
        types = []
        if isinstance(schemas, list):
            for sub_schema in schemas:
                inferred_type = infer_type_from_json_schema(sub_schema)
                if inferred_type:
                    types.append(inferred_type)
            if types:
                if len(set(types)) == 1:
                    return types[0]
                if "string" in types:
                    return "string"
                return types[0]
    if "enum" in schema and isinstance(schema["enum"], list):
        if not schema["enum"]:
            return "string"
        enum_types = set()
        for value in schema["enum"]:
            if value is None:
                enum_types.add("null")
            elif isinstance(value, bool):
                enum_types.add("boolean")
            elif isinstance(value, int):
                enum_types.add("integer")
            elif isinstance(value, float):
                enum_types.add("number")
            elif isinstance(value, str):
                enum_types.add("string")
            elif isinstance(value, list):
                enum_types.add("array")
            elif isinstance(value, dict):
                enum_types.add("object")
        if len(enum_types) == 1:
            return enum_types.pop()
        return "string"
    if "allOf" in schema and isinstance(schema["allOf"], list):
        for sub_schema in schema["allOf"]:
            inferred_type = infer_type_from_json_schema(sub_schema)
            if inferred_type and inferred_type != "string":
                return inferred_type
        return "string"
    if "properties" in schema:
        return "object"
    if "items" in schema:
        return "array"
    return None

'''

filepath = sys.argv[1]
with open(filepath, 'r') as f:
    content = f.read()

if 'infer_type_from_json_schema' in content:
    print("Already patched")
    sys.exit(0)

marker = "def get_json_schema_constraint("
idx = content.find(marker)
if idx == -1:
    content += FUNC_CODE
else:
    content = content[:idx] + FUNC_CODE + content[idx:]

with open(filepath, 'w') as f:
    f.write(content)
print("Patched successfully")
