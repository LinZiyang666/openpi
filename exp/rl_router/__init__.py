"""X14 online-RL router baseline: collection, training, and evaluation scripts.

The experiment answers the standing reviewer question "why retrieval instead of
a trained router?" by actually training the router — three MLP variants
(R_ts / R_tc / R_tsc) as an online-RL baseline against TIER's retrieval +
threshold routing, priced by an interaction-efficiency curve.

Mechanism lives in ``src/openpi`` (the ``mlp_router`` judge, the tri-state
interceptor dispatch, the conductor's accepted plumbing); this package holds
only experiment policy: batch assembly, the REINFORCE trainer, warm-start
fitting, the cost microbench, the lambda pilot, and the yaml emitters.
"""
