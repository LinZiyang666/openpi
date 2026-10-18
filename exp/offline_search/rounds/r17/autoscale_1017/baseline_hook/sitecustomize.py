import os
if os.environ.get('AUTOSCALE_BASELINE_ROOT'):
    from exp.offline_search.rounds.r17.autoscale_1017.baseline_sources import install
    install()
