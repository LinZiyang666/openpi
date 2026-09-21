"""Tiny TCP listener on the server box: a client lane sends ``DONE k=<k>\\n`` when its k is finished;
this touches /tmp/nfe/kdone_<tag>_<k> so ladder_server.sh switches k at once instead of waiting for
the idle timeout. usage: python3 kdone_listener.py <port> <tag>"""
import pathlib
import re
import socket
import sys
import time

port, tag = int(sys.argv[1]), sys.argv[2]
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", port))
s.listen(8)
print(f"KDONE listening :{port} tag={tag}", flush=True)
while True:
    c, addr = s.accept()
    try:
        c.settimeout(5)
        data = c.recv(256).decode(errors="ignore")
    except Exception:
        data = ""
    finally:
        c.close()
    m = re.search(r"DONE k=(\d+)", data)
    if m:
        pathlib.Path(f"/tmp/nfe/kdone_{tag}_{m.group(1)}").touch()
        print(f"KDONE k={m.group(1)} from {addr[0]} {time.strftime('%H:%M:%S')}", flush=True)
    else:
        print(f"KDONE ignored {data!r} from {addr[0]}", flush=True)
