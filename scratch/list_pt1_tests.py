import re

path = "tests/test_point1_four_key_binding.py"
content = open(path, encoding="utf-8").read()
tests = re.findall(r"^def (test_\w+)", content, re.MULTILINE)
print(f"Total tests: {len(tests)}")
for i, t in enumerate(tests, 1):
    print(f"{i}. {t}")
