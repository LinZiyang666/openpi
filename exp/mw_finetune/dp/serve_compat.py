"""Register DP in A's server; --cuda-graph enables verified denoiser replay."""
import argparse
import sys

from exp.mw_finetune import serve


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--cuda-graph", action="store_true")
    options, remaining = ap.parse_known_args()
    if options.cuda_graph:
        sys.argv[1:] = [*remaining, "--policy-option", "cuda_graph=true"]
    serve.MODEL_MODULES["dp"] = "exp.mw_finetune.dp.policy"
    serve.main()


if __name__ == "__main__":
    main()
