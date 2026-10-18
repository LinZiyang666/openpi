"""Generated release copies are test data, never additional test suites."""
collect_ignore_glob = ['tmp*']

pytest_plugins = ["exp.offline_search.rounds.r17.autoscale_1017.tests.explicit_candidate"]
