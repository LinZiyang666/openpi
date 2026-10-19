# 4090-host R16 workers: the GR00T servers and the driver run on this machine, so
# 'ziyanglin.com' (its public name) resolves to loopback instead of a WAN hairpin.
import socket
_getaddrinfo = socket.getaddrinfo
def _loopback(host, *args, **kwargs):
    return _getaddrinfo('127.0.0.1' if host == 'ziyanglin.com' else host, *args, **kwargs)
socket.getaddrinfo = _loopback
