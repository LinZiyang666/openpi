from .formal_a_analysis import choose
from .formal_a_queue import classify


def test_test_A_first_failure_two_successes_off_override_and_missing_init():
    order=('k50','k40','k30','k20','k10','koff')
    def data(n):
        return {('libero_10',str(t),i):dict(success=i<n) for t in range(10) for i in range(50)}
    pure=data(40);arms={level:data(50) for level in order}
    arms.update(k50=data(38),k40=data(37),koff=data(39))
    result=choose(arms,pure,order)
    assert result['complete'] and not result['no_knob']
    assert {r['selected'] for r in result['tasks'].values()}=={'k50'}
    assert all(r['first_failure']=='k40' and 'k30' in r['non_monotone_recoveries'] for r in result['tasks'].values())
    arms['koff']=data(40);result=choose(arms,pure,order)
    assert result['no_knob'] and not result['tuned'] and {r['selected'] for r in result['tasks'].values()}=={'koff'}
    arms.update(koff=data(39),k50=data(37));result=choose(arms,pure,order)
    assert {r['selected'] for r in result['tasks'].values()}=={'pure'} and result['data']==pure
    arms['k50']=data(38);arms['k50'].pop(('libero_10','0',49))
    result=choose(arms,pure,order)
    assert not result['complete'] and result['tasks']['0']['selected'] is None
    vla_order=('ir45','ir40','ir32','ir25','off')
    result=choose({level:data(39) for level in vla_order},pure,vla_order)
    assert result['grid_complete'] and {r['selected'] for r in result['tasks'].values()}=={'off'}


def test_off_dispatch_pruning_needs_complete_cell_and_reaches_pure_SR():
    def data(n):
        return {('libero_10',str(t),i):dict(success=i<n) for t in range(10) for i in range(50)}
    pure=data(40)
    assert classify(data(39),pure)['status']=='tune'
    assert classify(data(40),pure)['status']=='skip'
    assert classify(data(41),pure)['status']=='skip'
    partial=data(50);partial.pop(('libero_10','0',49))
    assert classify(partial,pure)['status']=='pending'
    assert classify({},pure)['status']=='pending'
