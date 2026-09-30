"""Connection-local bridge to C1's stateless FollowExtension component."""
from exp.offline_search.closed_loop.blind import BlindResult


class FollowCacheExtension:
    def __init__(self, component):
        self.component = component
        self.plan = None

    def reset(self, episode):
        self.plan = None

    def invalidate_anchor(self):
        self.plan = None

    def on_anchor(self, controller, q, result, *, call):
        self.plan = None if call or not self.component.extend_blocks else self.component.plan(controller.base._anchor)
        if self.plan is not None:
            result.extras.update(self.plan.extras)

    def blind_step(self, controller, bq, base_step):
        base, component, plan = controller.base, self.component, self.plan
        if plan is None or not plan.blocks:
            return base_step(bq)
        reason = base.lifecycle_reason(bq)
        if reason is not None:
            self.plan = None
            return reason
        anchor = base._anchor
        age = int(bq.step) - anchor['step']
        reason, extras = component.check(anchor, bq, age, plan)
        if reason is not None:
            base.last_blind_extras = extras
            return reason
        if age < component.table.reference_blocks:
            result = base_step(bq)
            base.last_blind_extras.update(extras)
            return result
        result = component.serve(anchor, age, base.act, base.cand_name, extras)
        anchor['last_step'] = int(bq.step)
        anchor['phase'] = component.table.advance(anchor['rows'], age)
        base.last_blind_extras = extras
        return BlindResult(result.action, result.rows, result.weights, result.library,
                           {**result.extras, **controller._carried_log()})
