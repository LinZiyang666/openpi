import os
if os.environ.get('AUTOSCALE_SOURCE_VARIANT'):
    from exp.offline_search.rounds.r17.autoscale_1017.followup_1018.source_overlay import install
    install()
