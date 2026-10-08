from .d2_screen import *
names=['frozen_atlas','consensus2_r0.1','consensus6_r0.1','consensus6_r0.15','exact2_r0.1','exact6_r0.15']
for p in sorted(D2.glob('r12_devlooks_*/*/screen.json')):
    if p.parent.parent.name not in VALID_ROOTS:continue
    x=json.loads(p.read_text());print(x['root'],x['cell'],x['size'])
    for a in x['results']:
        if a['config']['name'] not in names:continue
        m=a['all'];c=a['calibration'];v=a['evaluation']
        print(a['config']['name'],'H',m['H'],'save',round(m['saving_fixed_calls'],6),'miss/extra/grip',m['missed'],m['extra'],m['gripper_disagree_half'],
              'cal',c['H'],c['missed'],c['extra'],c['gripper_disagree_half'],'eval',v['H'],v['missed'],v['extra'],v['gripper_disagree_half'],
              'mse',m.get('shadow_mse_delta'))
