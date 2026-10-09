# Engineering

## Local checks

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m beam.examples.theory_alignment
python -m pip install -r requirements-docs.txt
python -m mkdocs build --strict
python scripts/check_site.py
```

The finite-state tests construct the complete three-agent transition kernel. Numerical tests compare the covariance gradient with finite differences of the exact constrained objective. Execution tests cover inactive agents, fixed selected edges, concurrent requests, checkpoints and integration masks.

## Continuous integration

Python 3.9 and 3.12 run the test suite and offline example. Documentation checks build the site in strict mode and inspect local links and assets. The quick-start page includes the tested Python example directly from its source file.

## Static publishing

The repository keeps `main` as its development and publishing branch. GitHub Pages receives an uploaded static artifact through GitHub Actions. The publishing job runs the Python checks, builds the documentation and verifies its links before deployment. Its deployment identity has Pages permissions; it does not write a generated branch.

## Update the documentation

Edit existing `.md` files in `docs/`. Add new pages to the `nav` section in `mkdocs.yml`. Preview changes locally:

```bash
python -m pip install -r requirements-docs.txt
python -m mkdocs serve
```

Open the local URL printed by MkDocs; the preview refreshes when Markdown changes. Commit and push to `main` to publish after the checks pass. The pencil icon on each page opens its Markdown source on GitHub.

## Reproducible execution

Set `random_seed` for the graph sampler. Version-2 checkpoints contain fitted offsets, GP moments, energies, completeness, Adam state and the private random-generator state. Restore into a graph with the same node order and execution protocol.

Keep each reference chain's utility and parameters fixed. Apply energy and GP updates between calibration batches. LLM evaluations belong to your chosen task protocol; the included regression suite uses deterministic callbacks.

[Algorithm mapping](models/paper-to-code.md) · [Latest CI runs](https://github.com/erwinmsmith/BEAM/actions)
