# C1 SA composition hook

`methods:CallController` subclasses R6 `CalibratedRescue`. CU uses `tilt=false,
placement="uniform"`; CT uses `tilt=true, placement="uniform"`. Both require the
R6 calibrated stall artifact for selected arms. There is no added policy-tail
extension. When tilt is false and the same placement/calibration/coins are used,
the original R6 query, blind and policy-tail paths execute unchanged.

After fitting a controller, call `controller.install_cache_extension(extension)`
before the first reset. C1 supplies a fitted, connection-local object:

```python
class CacheExtension:
    def reset(self, episode): ...
    def invalidate_anchor(self): ...
    def on_anchor(self, controller, q, result, *, call):
        # controller.base._anchor is original A's rows/weights/action/state.
        # A call clears extension eligibility. The policy tail stays R6's.
        ...
    def blind_step(self, controller, bq, base_step):
        # base_step(bq) is the ordinary R6/A blind path. Preserve the first
        # tail; at its expiry apply C1's structural/stage/state-valve rules.
        # Return BlindResult or LookReason with SF decision telemetry.
        ...
```

The ready bridge `install_follow_extension(component)` accepts C1's landed
`c1_follow.methods:FollowExtension` (plan/check/serve API) directly. It preserves
the first cache tail, applies the valve on extended commitments, clears plans
on MISS/reset/invalidation, and keeps policy tails outside the extension.
Use `FollowExtension(table, extend_blocks=0)` to prove composed disabled parity.

The wrapper checks R6's pending stall LOOK before dispatching to the extension.
Only fresh `query` anchors observe stall and update CT's deviation-entry latch.
Blind state-valve probes do not consume a lottery coin or update that latch.
The extension must keep its own cursor: do not change the saved policy-tail
gate or the base fitted metric/projections. `policy_tail_step` is inherited.
Adding the extension requires a separate SA budget replay and re-solve: CU/CT's
calibrations price the original 10-control cadence and full-camera looks.
The hook itself provides no SA arm with a mislabeled .30 budget.

CT calls C1 `StageTable.online(rows, weights)` and
`deviation(state, rows, weights)`; it reads `event_occupancy`,
`deviation_p75` and `deviation_occupancy` keyed by integer task ID.
StageTable owns scales, valid dimensions, reference retrieval and occupancy.
No stage segmentation or threshold is fitted in C3.

The event factor is `1 + event_mass*(1/h - 1)` for `h>0`, otherwise 1: the
kernel mean of each known member's 1/h event weight and unit interior/unknown
weight. The deviation factor is 1/h_dev at each fresh high entry, otherwise 1.
These factors multiply when both signals occur together. A return below/equal
p75 clears the latch; missing state preserves it. This composition rule uses no
task-name condition, event-mass threshold or zero-call interior.
