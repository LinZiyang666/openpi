"""Neighbour mode/stage mass, demo support and unsupported successor tails."""
from . import common as C
from .kernels import catalog_map, kernel


def analyze(arm, args=None):
    """Profile frozen neighbour provenance; unknown catalogue mass stays explicit."""
    decisions, episodes = C.inputs(arm)
    catalog = catalog_map(arm)
    output = []
    for row in decisions.to_dict("records"):
        record = C.identity(row)
        rows, weights = C.field(row, "rows"), C.field(row, "weights")
        try:
            if rows is None or weights is None:
                raise C.Unavailable("live retrieval not available at this decision")
            record.update(kernel(rows, weights, catalog, C.field(row, "lib")), status="available",
                          logged_unsupported_mass=C.number(C.field(row, "unsupported_mass")))
        except C.Unavailable as error:
            record.update(status="unavailable", reason=str(error))
        output.append(record)
    n = sum(row["status"] == "available" for row in output)
    return {"status": "available" if n else "unavailable", "coverage": C.coverage(len(decisions), n),
            "episode_denominator": len(episodes), "notes": ["Tail mass means no next catalogue row; it is not an inferred failed action.",
              "Effective demos merges weights within each task/demo; row ESS is separate."],
            "tables": {"decisions": output, "by_stage": C.summarize(output, ["stage"],
              ["unknown_mass", "unknown_stage_mass", "unknown_catalog_mass", "effective_demos", "cross_stage_kernel", "cross_mode_kernel", "unsupported_next_mass"])}}


def main():
    C.cli("provenance", analyze)


if __name__ == "__main__":
    main()
