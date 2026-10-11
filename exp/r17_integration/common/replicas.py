"""Plan compatible replica processes; startup/ownership remains P6 ops."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ReplicaPlan:
    commands: tuple[tuple[str, ...], ...]
    public_port: int
    backend_ports: tuple[int, ...]
    metadata_identity: str

    def run_proxy(self):
        # Explicit call only, after operators attest every model endpoint.
        from openpi.serving.replica_proxy import run_proxy
        run_proxy(self.public_port, list(self.backend_ports))


def plan_replicas(command, *, public_port, backend_ports, model_spec):
    ports = (public_port, *backend_ports)
    if not backend_ports or len(set(ports)) != len(ports) or any(type(p) is not int or not 0 < p < 65536 for p in ports):
        raise ValueError("replica ports must be distinct valid explicit ports")
    if "--port" in command:
        raise ValueError("base command must omit --port; each replica gets its own")
    return ReplicaPlan(tuple(tuple(command) + ("--port",str(p)) for p in backend_ports),
                        public_port,tuple(backend_ports),model_spec.identity)
