"""Opt-in control extensions around the unchanged stock WebSocket handler.

Connection lifecycle, bundle selection, inference, timing, health and cleanup
remain stock. A socket facade consumes only R17 load/terminal-ack controls.
No global functions/classes are patched and no sealed source is edited.
"""
import asyncio
from contextvars import ContextVar
from pathlib import Path

from openpi.serving import websocket_policy_server as bundle_registry
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy as StockClient

get_current_cache_bundle = bundle_registry.get_current_cache_bundle
get_cache_bundle_ids = bundle_registry.get_cache_bundle_ids
_socket = ContextVar("r17_connection_socket", default=None)


class _ControlSocket:
    def __init__(self, socket, server):
        self.socket, self.server = socket, server
        self.policy = server._policy if not server._concurrent else None
        self.bound_bundle_id = None
        self.prepared_binding = None
        self.packer = msgpack_numpy.Packer()

    def __getattr__(self, name):
        return getattr(self.socket, name)

    async def send(self, value):
        decoded = msgpack_numpy.unpackb(value) if not isinstance(value, str) else None
        changed = False
        if isinstance(decoded, dict):
            meta = decoded.get('__hit_meta__', {})
            if meta.get('clock_unit') == 'segment_decision':
                # Frozen workers retain server_timing verbatim in per_step.
                # Their transport decision_seq remains the ack sequence.
                timing = decoded.setdefault('server_timing', {})
                timing.update({k: meta[k] for k in ('decision_index','delivery_index','env_step_index','segment_head','clock_unit','decision_seed','wire_explicit_seed') if k in meta})
                changed = True
        if self.policy is not None:
            if isinstance(decoded, dict) and decoded.get("__ack__") == "episode_end" and getattr(self, "last_execution_ack", None):
                decoded = dict(decoded, execution=self.last_execution_ack)
                changed = True
        if changed:
            value = self.packer.pack(decoded)
        return await self.socket.send(value)

    async def recv(self):
        while True:
            raw = await self.socket.recv()
            if isinstance(raw, str):
                return raw
            message = msgpack_numpy.unpackb(raw)
            control = message.get("__ctrl__") if isinstance(message, dict) else None
            # The stock handler's binder is synchronous. Prepare its stack in
            # a worker before handing it the control, so validation, library
            # fitting and adapter construction cannot block other sockets.
            target = None
            if self.server._concurrent and self.server._r17_factory:
                if control == "select_bundle":
                    candidate = message.get("bundle_id", "default")
                    with bundle_registry._bundle_lock:
                        known = candidate == "default" or candidate in bundle_registry._bundles
                    if known and (candidate == "default" or self.server._allow_dynamic_bundles):
                        target = candidate
                elif self.policy is None and control == "episode_start":
                    target = message.get("bundle_id", "default")
                elif self.policy is None and control is None:
                    target = "default"
            if target is not None and target != self.bound_bundle_id:
                try:
                    policy = await asyncio.to_thread(self.server._r17_factory, self.server._policy, target)
                    self.prepared_binding = target, policy
                except Exception as exc:
                    await self.send(self.packer.pack({"__ack__": "error", "msg": f"bundle bind failed: {exc}"}))
                    continue
            if control in ("episode_start", "select_bundle"):
                self.last_execution_ack = None
            if control == "load_cache_config":
                try:
                    response = await self.server._load_r17_bundle(message)
                except Exception as exc:
                    response = {"__ack__": "error", "msg": str(exc)}
                await self.send(self.packer.pack(response))
                continue
            if control in ("episode_end", "acknowledge_execution"):
                ack = message.get("__execution__")
                try:
                    if ack is not None:
                        if self.policy is None:
                            raise ValueError("execution acknowledgment has no bound connection")
                        if control == "episode_end":
                            ack = dict(ack, terminal=True)
                        self.last_execution_ack = await asyncio.to_thread(self.policy.acknowledge_execution, ack)
                    elif control == "acknowledge_execution":
                        raise ValueError("missing __execution__ acknowledgment")
                    elif self.policy is not None and getattr(self.policy, "requires_context", False) and self.policy._awaiting is not None:
                        raise ValueError("episode_end requires final __execution__ acknowledgment")
                except Exception as exc:
                    await self.send(self.packer.pack({"__ack__": "error", "msg": str(exc)}))
                    continue
                if control == "acknowledge_execution":
                    await self.send(self.packer.pack({"__ack__": control, "execution": self.last_execution_ack}))
                    continue
            return raw


class WebsocketPolicyServer(bundle_registry.WebsocketPolicyServer):
    def __init__(self, *args, bundle_validator=None, **kwargs):
        self._r17_validator = bundle_validator
        original_factory = kwargs.get("connection_policy_factory")
        self._r17_factory = original_factory
        if original_factory:
            def factory(base, bundle_id):
                connection = _socket.get()
                if connection is not None:
                    if connection.prepared_binding is None or connection.prepared_binding[0] != bundle_id:
                        raise ValueError("bundle was not prepared off the event loop")
                    _, policy = connection.prepared_binding
                    connection.prepared_binding = None
                    connection.policy = policy
                    connection.bound_bundle_id = bundle_id
                else:
                    policy = original_factory(base, bundle_id)
                return policy
            kwargs["connection_policy_factory"] = factory
        super().__init__(*args, **kwargs)

    async def _handler(self, websocket):
        connection = _ControlSocket(websocket, self)
        token = _socket.set(connection)
        try:
            await super()._handler(connection)
        finally:
            _socket.reset(token)

    async def _load_r17_bundle(self, message):
        import yaml
        from .config import config_from_dict, build_shared_storage
        from .recipe import bundle_identity
        if not self._allow_dynamic_bundles or not self._concurrent:
            raise ValueError("load_cache_config requires concurrent dynamic-bundle serving")
        path = message.get("yaml_path")
        content = message.get("yaml_content")
        if not path and not content:
            raise ValueError("missing yaml_path or yaml_content")
        def build():
            from openpi.cache.config import _substitute_env_vars
            raw = yaml.safe_load(_substitute_env_vars(content if content else Path(path).read_text())) or {}
            config = config_from_dict(raw)
            if config.adapter_binding is None:
                raise ValueError("R17 server requires an explicit adapter binding")
            if self._r17_validator:
                self._r17_validator(config, None)
            storage = build_shared_storage(config)
            if self._r17_validator:
                self._r17_validator(config, storage)
            return config, storage
        config, storage = await asyncio.to_thread(build)
        identity = bundle_identity(config)
        if message.get("bundle_identity") not in (None, identity):
            pass
        yaml_id = message.get("yaml_id") or None
        bundle_id = message.get("bundle_id") or yaml_id or "default"
        with bundle_registry._bundle_lock:
            previous = bundle_registry._bundles.get(bundle_id)
            if previous is not None and bundle_identity(previous.cache_config) != identity:
                raise ValueError("bundle id already names different content; publish a new generation id")
            bundle_registry._bundle_version += 1
            version = bundle_registry._bundle_version
            bundle = bundle_registry.CurrentCacheBundle(path or "<r17:inline>", config, storage, version, yaml_id)
            bundle_registry._bundles[bundle_id] = bundle
            bundle_registry._current_bundle = bundle
        return {"__ack__": "load_cache_config", "yaml_path": path or "<r17:inline>",
                "version": version, "bundle_id": bundle_id, "bundle_identity": identity}


class WebsocketClientPolicy(StockClient):
    def episode_end(self, success=False, *, execution=None):
        message = {"__ctrl__": "episode_end", "__success__": success}
        if execution is not None:
            message["__execution__"] = execution
        return self._send_ctrl(message, expected_ack="episode_end")

    def acknowledge_execution(self, execution):
        return self._send_ctrl({"__ctrl__": "acknowledge_execution", "__execution__": execution},
                               expected_ack="acknowledge_execution")
