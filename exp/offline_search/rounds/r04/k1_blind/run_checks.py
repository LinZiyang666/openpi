from exp.offline_search.rounds.r04.k1_blind.checks import parity, replay
for key in ('pi05_l10','pi05_spatial','groot_l10','groot_spatial'):
    for scale in (50,500):
        if key=='pi05_l10' and scale==50: continue  # exact commands/results retained separately
        parity(key,scale)
        replay(key,scale)
