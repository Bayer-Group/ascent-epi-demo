"""Present so that ``tests`` is importable as a package.

Test modules import shared builders by their full path -- e.g.
``from tests.ascent.utils.concept_list_mother import DEFAULT_CONCEPTS``. That
resolves because pytest puts the directory holding the topmost ``conftest.py``
on ``sys.path``; delete this file and those imports fail at collection with
``No module named 'tests'``.

There are no fixtures here on purpose. Anything shared belongs in the
``conftest.py`` next to the tests that use it.
"""
