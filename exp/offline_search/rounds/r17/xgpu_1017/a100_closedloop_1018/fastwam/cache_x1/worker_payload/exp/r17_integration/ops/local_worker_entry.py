"""Process-local endpoint routing over unchanged native worker/client classes."""
from contextlib import contextmanager
import importlib
import os


@contextmanager
def loopback(canonical, destination):
    from openpi_client import websocket_client_policy as module
    original = module.WebsocketClientPolicy
    source_host,source_port = canonical.rsplit(':',1)
    destination_host,destination_port = destination.rsplit(':',1)
    class RoutedClient(original):
        def __init__(self,host=source_host,port=int(source_port),**kwargs):
            if (host,int(port)) != (source_host,int(source_port)):
                raise ValueError('local worker requested a different sealed policy endpoint')
            super().__init__(host=destination_host,port=int(destination_port),**kwargs)
    module.WebsocketClientPolicy = RoutedClient
    try:
        yield
    finally:
        module.WebsocketClientPolicy = original


def main(argv=None):
    target = os.environ['R17_LOCAL_WORKER_TARGET']
    if target not in ('exp.r17_integration.conductor.worker_entry',
                      'exp.r17_integration.ops.worker_lifetime','exp.r17_integration.ops.cache_study_worker'):
        raise ValueError('unknown native local worker target')
    with loopback(os.environ['R17_CANONICAL_POLICY_ENDPOINT'],os.environ['R17_LOCAL_POLICY_ENDPOINT']):
        return importlib.import_module(target).main(argv)


if __name__ == '__main__':
    main()
