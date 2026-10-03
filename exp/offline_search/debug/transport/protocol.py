# Adapted from rounds/r06/p3_profiling/stream_protocol.py; offset ACK/durable receipt semantics retained.
"""P3 byte-offset protocol. Python 3.8, standard library only."""
import hashlib
import json
import re
import socket
import struct

MAX_HEADER = 256 * 1024
MAX_BODY = 256 * 1024


def identifier(value, arm=False):
    if not isinstance(value, str) or len(value) > 200 or not re.fullmatch(
            r"[A-Za-z0-9_]+" if arm else r"[A-Za-z0-9_-]+", value):
        raise ValueError("invalid arm/run identifier")
    return value


def attempt_key(uid, attempt):
    if not isinstance(uid, str) or len(uid) > 512 or not re.fullmatch(r"[A-Za-z0-9_:-]+", uid):
        raise ValueError("invalid task uid")
    if type(attempt) is not int or not 1 <= attempt <= 1000000:
        raise ValueError("invalid attempt")
    return hashlib.sha256(uid.encode()).hexdigest()[:24] + "_a" + str(attempt)


def filename(name):
    if not isinstance(name, str) or not re.fullmatch(
            r"episode\.json|events\.jsonl|controls_\d{4}\.npz|snap_(reset|final|\d{6})\.npz", name):
        raise ValueError("invalid debug filename")
    return name


def endpoint(value):
    host, port = value.rsplit(":", 1)
    if not re.fullmatch(r"[A-Za-z0-9.-]+", host) or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError("invalid stream endpoint")
    return host, int(port)


def pack(header, body=b""):
    raw = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > MAX_HEADER or len(body) > MAX_BODY:
        raise ValueError("oversized stream frame")
    return struct.pack("!II", len(raw), len(body)) + raw + body


def read_exact(read, size, eof=False):
    result = bytearray()
    while len(result) < size:
        value = read(size-len(result))
        if not value:
            if eof and not result:
                return None
            raise EOFError("truncated stream frame")
        result.extend(value)
    return bytes(result)


def receive(read):
    prefix = read_exact(read, 8, eof=True)
    if prefix is None:
        return None
    nh, nb = struct.unpack("!II", prefix)
    if nh > MAX_HEADER or nb > MAX_BODY or not nh:
        raise ValueError("oversized stream frame")
    header = json.loads(read_exact(read, nh))
    if not isinstance(header, dict):
        raise ValueError("header must be an object")
    return header, read_exact(read, nb)


def request(address, header, body=b"", timeout=3):
    with socket.create_connection(endpoint(address), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(pack(header, body))
        response = receive(sock.recv)
        if response is None or not response[0].get("ok"):
            raise RuntimeError("receiver refused request: " + str(response))
        return response[0]
