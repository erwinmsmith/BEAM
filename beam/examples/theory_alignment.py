"""Compatibility entry point for the Bayesian workflow example."""

import asyncio

from beam.examples.bayesian_workflow import build_example, main

if __name__ == "__main__":
    asyncio.run(main())
