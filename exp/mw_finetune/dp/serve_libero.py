"""Existing LIBERO websocket/msgpack policy interface, with current-frame DP."""
import argparse
import threading

from .policy import load_policy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="/data/mw_finetune/dp/checkpoints/libero")
    ap.add_argument("--text-encoder", default="/data/mw_finetune/dp/clip_text")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--port", type=int, default=23154)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--denoising-steps", type=int, default=100)
    ap.add_argument("--cuda-graph", action="store_true", help="Replay the denoiser after a required bit-exact startup check")
    a = ap.parse_args()
    policy = load_policy(a.checkpoint, a.device, domain="libero", denoising_steps=a.denoising_steps,
                         text_encoder=a.text_encoder, cuda_graph=a.cuda_graph)
    lock = threading.Lock()

    class Connection:
        def on_episode_start(self, **kwargs):
            pass

        def on_episode_end(self, **kwargs):
            pass

        def infer(self, obs):
            with lock:
                policy.reset()
                return policy.infer(obs)

    def factory(base, bundle_id="default"):
        if bundle_id != "default":
            raise ValueError("DP serves only default bundle")
        return Connection()

    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    server = WebsocketPolicyServer(Connection(), host=a.host, port=a.port,
        metadata={"model": "dp-libero", "checkpoint": policy.checkpoint,
                  "training_step": policy.metadata["global_step"], "pure_inference": True, "concurrent": True, **policy.get_metadata()},
        concurrent=True, connection_policy_factory=factory)
    server.serve_forever()


if __name__ == "__main__":
    main()
