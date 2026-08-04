"""Algorithm-specific next-event models.

Import implementations from their canonical modules:

- :mod:`src.models.markov`
- :mod:`src.models.gru`
- :mod:`src.models.lstm`

The package intentionally performs no eager algorithm imports.  In particular,
using the Markov module does not load either neural architecture.
"""
