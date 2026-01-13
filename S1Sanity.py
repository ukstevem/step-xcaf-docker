import json, pathlib

p = pathlib.Path("out/xcaf_instances.json")
j = json.loads(p.read_text())

defs = j["definitions"]
occs = j["occurrences"]

print("defs:", len(defs))
print("occs:", len(occs))
print("leaf:", len(j["indexes"]["leaf_occ_ids"]))

# integrity checks
missing_defs = [oid for oid,o in occs.items() if o["ref_def"] not in defs]
print("missing ref_defs:", len(missing_defs))

bad_children = 0
for parent_def, kids in j["indexes"]["children_by_parent_def"].items():
    for oid in kids:
        if occs[oid]["parent_def"] != parent_def:
            bad_children += 1
print("bad_children:", bad_children)
