"""DP lane registration shim until A integrates REQUESTS/E_to_A_dp.md."""
from exp.mw_finetune import worker_launch


def main():
    worker_launch.PORTS["dp"] = 54
    worker_launch.main()


if __name__ == "__main__":
    main()
