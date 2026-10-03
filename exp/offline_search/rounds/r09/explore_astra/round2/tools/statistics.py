"""Conservative paired non-inferiority; use ONLY coordinator-approved pairs."""
from scipy.stats import beta


def paired_ni(wins, losses, n, family=1, alpha=.05, margin=.02):
    if n<=0 or min(wins,losses)<0 or wins+losses>n or family<1:
        raise ValueError('invalid paired counts')
    tail=alpha/(2*family)
    lower_win=0. if wins==0 else float(beta.ppf(tail,wins,n-wins+1))
    upper_loss=1. if losses==n else float(beta.ppf(1-tail,losses+1,n-losses))
    lower=lower_win-upper_loss
    return dict(delta=(wins-losses)/n,lower=lower,passes=lower>-margin,
                n=n,wins=wins,losses=losses,family=family,alpha=alpha,margin=margin)
