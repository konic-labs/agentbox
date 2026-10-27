"""Small coding tasks for single-GPU agentic RL: file creation, bug fixes, stubs.

    python examples/verl/make_coding_tasks.py data/tasks

writes data/tasks/train.jsonl (27 tasks) and data/tasks/eval.jsonl (9 held-out
tasks). Code tasks use protected tests, planted-file detection and
pytest_fraction partial credit, so GRPO groups get a graded signal. Before
training, Qwen3-0.6B (thinking on) solves nearly every file task but few of the
bug fixes and stubs; docs/integrations/verl.md has the measured numbers.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from agentbox import Task, VerifierSpec
from agentbox.tasks.schema import PYTEST_TAMPER_FILES

rng = random.Random(7)
WORDS = [
    "apple",
    "river",
    "stone",
    "cloud",
    "maple",
    "ember",
    "coral",
    "delta",
    "frost",
    "lunar",
    "olive",
    "pearl",
    "quartz",
    "sable",
    "tundra",
    "violet",
]


def pytest_task(
    task_id: str, desc: str, module: str, starter: str, tests: list[str], difficulty: str
) -> Task:
    name = module.removesuffix(".py")
    body = f"from {name} import *\n\n" + "\n\n".join(tests) + "\n"
    return Task(
        task_id=task_id,
        description=desc,
        starter_files={module: starter, f"test_{name}.py": body},
        verifier=VerifierSpec(
            type="pytest",
            command="python -m pytest -q",
            reward_mode="pytest_fraction",
            protected_paths=["test_*.py"],
            forbid_new_files=list(PYTEST_TAMPER_FILES),
            timeout_s=60,
        ),
        metadata={"difficulty": difficulty, "family": task_id.split("_")[0]},
    )


def file_tasks() -> list[Task]:
    out = []
    for i, word in enumerate(WORDS):
        fname = f"{rng.choice(['note', 'answer', 'result', 'output'])}_{i}"
        out.append(
            Task(
                task_id=f"file_{i:02d}",
                description=(
                    f"Create a file named {fname}.txt in the workspace whose entire content "
                    f"is the word {word}."
                ),
                verifier=VerifierSpec(
                    type="command", command=f'test "$(tr -d "[:space:]" < {fname}.txt)" = {word}'
                ),
                metadata={"difficulty": "easy", "family": "file"},
            )
        )
    return out


BUGS = [
    (
        "add",
        "def add(a, b):\n    return a - b\n",
        "add(a, b) returns the wrong value; it must return the sum.",
        [
            "def test_a():\n    assert add(2, 3) == 5",
            "def test_b():\n    assert add(-1, 1) == 0",
            "def test_c():\n    assert add(0, 0) == 0",
        ],
    ),
    (
        "multiply",
        "def multiply(a, b):\n    return a + b\n",
        "multiply(a, b) must return the product of a and b.",
        [
            "def test_a():\n    assert multiply(2, 3) == 6",
            "def test_b():\n    assert multiply(4, 0) == 0",
            "def test_c():\n    assert multiply(-2, 5) == -10",
        ],
    ),
    (
        "maximum",
        "def maximum(xs):\n    return min(xs)\n",
        "maximum(xs) must return the largest element of the list.",
        [
            "def test_a():\n    assert maximum([1, 5, 3]) == 5",
            "def test_b():\n    assert maximum([-4, -2]) == -2",
            "def test_c():\n    assert maximum([7]) == 7",
        ],
    ),
    (
        "is_even",
        "def is_even(n):\n    return n % 2 == 1\n",
        "is_even(n) must return True for even numbers.",
        [
            "def test_a():\n    assert is_even(4)",
            "def test_b():\n    assert not is_even(7)",
            "def test_c():\n    assert is_even(0)",
        ],
    ),
    (
        "absolute",
        "def absolute(n):\n    return n\n",
        "absolute(n) must return the absolute value of n.",
        [
            "def test_a():\n    assert absolute(-3) == 3",
            "def test_b():\n    assert absolute(4) == 4",
            "def test_c():\n    assert absolute(0) == 0",
        ],
    ),
    (
        "reverse",
        "def reverse(s):\n    return s\n",
        "reverse(s) must return the string reversed.",
        [
            "def test_a():\n    assert reverse('abc') == 'cba'",
            "def test_b():\n    assert reverse('') == ''",
            "def test_c():\n    assert reverse('racecar') == 'racecar'",
        ],
    ),
    (
        "shout",
        "def shout(s):\n    return s.lower()\n",
        "shout(s) must return the string in upper case.",
        [
            "def test_a():\n    assert shout('hi') == 'HI'",
            "def test_b():\n    assert shout('Ok') == 'OK'",
            "def test_c():\n    assert shout('') == ''",
        ],
    ),
    (
        "factorial",
        "def factorial(n):\n    result = 1\n    for i in range(1, n):\n"
        "        result *= i\n    return result\n",
        "factorial(n) is off by one; it must return n! (with factorial(0) == 1).",
        [
            "def test_a():\n    assert factorial(5) == 120",
            "def test_b():\n    assert factorial(1) == 1",
            "def test_c():\n    assert factorial(0) == 1",
        ],
    ),
    (
        "total",
        "def total(xs):\n    return sum(xs[1:])\n",
        "total(xs) skips the first element; it must return the sum of all elements.",
        [
            "def test_a():\n    assert total([1, 2, 3]) == 6",
            "def test_b():\n    assert total([]) == 0",
            "def test_c():\n    assert total([5]) == 5",
        ],
    ),
    (
        "to_fahrenheit",
        "def to_fahrenheit(c):\n    return c * 5 / 9 + 32\n",
        "to_fahrenheit(c) uses the wrong formula; "
        "it must convert Celsius to Fahrenheit (c * 9 / 5 + 32).",
        [
            "def test_a():\n    assert to_fahrenheit(0) == 32",
            "def test_b():\n    assert to_fahrenheit(100) == 212",
            "def test_c():\n    assert to_fahrenheit(-40) == -40",
        ],
    ),
    (
        "count_vowels",
        "def count_vowels(s):\n    return sum(1 for c in s if c not in 'aeiou')\n",
        "count_vowels(s) counts the wrong characters; it must count the vowels a, e, i, o, u.",
        [
            "def test_a():\n    assert count_vowels('hello') == 2",
            "def test_b():\n    assert count_vowels('xyz') == 0",
            "def test_c():\n    assert count_vowels('aeiou') == 5",
        ],
    ),
    (
        "last",
        "def last(xs):\n    return xs[0]\n",
        "last(xs) must return the last element of the list.",
        [
            "def test_a():\n    assert last([1, 2, 3]) == 3",
            "def test_b():\n    assert last(['x']) == 'x'",
            "def test_c():\n    assert last([4, 4, 9]) == 9",
        ],
    ),
]

STUBS = [
    (
        "clamp",
        "def clamp(x, lo, hi):\n    raise NotImplementedError\n",
        "Implement clamp(x, lo, hi): return lo if x < lo, hi if x > hi, otherwise x.",
        [
            "def test_a():\n    assert clamp(5, 0, 10) == 5",
            "def test_b():\n    assert clamp(-1, 0, 10) == 0",
            "def test_c():\n    assert clamp(11, 0, 10) == 10",
        ],
    ),
    (
        "is_prime",
        "def is_prime(n):\n    raise NotImplementedError\n",
        "Implement is_prime(n): True if n is a prime number, False otherwise (n < 2 is not prime).",
        [
            "def test_a():\n    assert is_prime(7)",
            "def test_b():\n    assert not is_prime(9)",
            "def test_c():\n    assert not is_prime(1)",
            "def test_d():\n    assert is_prime(2)",
        ],
    ),
    (
        "gcd",
        "def gcd(a, b):\n    raise NotImplementedError\n",
        "Implement gcd(a, b): the greatest common divisor of two positive integers.",
        [
            "def test_a():\n    assert gcd(12, 8) == 4",
            "def test_b():\n    assert gcd(7, 3) == 1",
            "def test_c():\n    assert gcd(10, 10) == 10",
        ],
    ),
    (
        "word_count",
        "def word_count(text):\n    raise NotImplementedError\n",
        "Implement word_count(text): the number of whitespace-separated words in text.",
        [
            "def test_a():\n    assert word_count('a b c') == 3",
            "def test_b():\n    assert word_count('') == 0",
            "def test_c():\n    assert word_count('  hi   there ') == 2",
        ],
    ),
    (
        "flatten",
        "def flatten(xs):\n    raise NotImplementedError\n",
        "Implement flatten(xs): turn a list of lists into one flat list (one level deep).",
        [
            "def test_a():\n    assert flatten([[1, 2], [3]]) == [1, 2, 3]",
            "def test_b():\n    assert flatten([]) == []",
            "def test_c():\n    assert flatten([[], [4]]) == [4]",
        ],
    ),
    (
        "average",
        "def average(xs):\n    raise NotImplementedError\n",
        "Implement average(xs): the arithmetic mean of a non-empty list of numbers.",
        [
            "def test_a():\n    assert average([1, 2, 3]) == 2",
            "def test_b():\n    assert average([5]) == 5",
            "def test_c():\n    assert average([1, 2]) == 1.5",
        ],
    ),
    (
        "count_char",
        "def count_char(s, c):\n    raise NotImplementedError\n",
        "Implement count_char(s, c): how many times the character c occurs in s.",
        [
            "def test_a():\n    assert count_char('banana', 'a') == 3",
            "def test_b():\n    assert count_char('', 'a') == 0",
            "def test_c():\n    assert count_char('xyz', 'q') == 0",
        ],
    ),
    (
        "squares",
        "def squares(n):\n    raise NotImplementedError\n",
        "Implement squares(n): the list [0, 1, 4, ...] of the first n square numbers.",
        [
            "def test_a():\n    assert squares(3) == [0, 1, 4]",
            "def test_b():\n    assert squares(0) == []",
            "def test_c():\n    assert squares(1) == [0]",
        ],
    ),
]


def code_tasks(
    specs: list[tuple[str, str, str, list[str]]], family: str, difficulty: str
) -> list[Task]:
    out = []
    for fn, starter, desc, tests in specs:
        module = f"{fn}_mod.py" if family == "stub" else f"{fn}.py"
        files_note = (
            f" The code is in {module}; the tests are in test_{module.removesuffix('.py')}.py."
        )
        out.append(
            pytest_task(
                f"{family}_{fn}",
                desc + files_note + " Make all tests pass.",
                module,
                starter,
                tests,
                difficulty,
            )
        )
    return out


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    tasks = file_tasks() + code_tasks(BUGS, "bug", "easy") + code_tasks(STUBS, "stub", "medium")
    rng.shuffle(tasks)
    # hold out a quarter of each family for evaluation
    by_family: dict[str, list[Task]] = {}
    for t in tasks:
        by_family.setdefault(t.metadata["family"], []).append(t)
    for fam in by_family.values():
        cut = max(1, len(fam) // 4)
        for i, t in enumerate(fam):
            t.split = "eval" if i < cut else "train"
    for split in ("train", "eval"):
        rows = [t for t in tasks if t.split == split]
        (out / f"{split}.jsonl").write_text(
            "".join(t.model_dump_json(exclude_none=True) + "\n" for t in rows)
        )
        print(
            split,
            len(rows),
            {f: sum(1 for t in rows if t.metadata["family"] == f) for f in by_family},
        )


if __name__ == "__main__":
    main()
