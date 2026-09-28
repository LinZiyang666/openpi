"""Standalone plots of the measured fixed-observation curves."""
import json
import os
from pathlib import Path
from exp.offline_search.rounds.r05.q4_growth.common import OUT
R=OUT/'results'/'demo'
os.environ['MPLCONFIGDIR']=str(R/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    records=[r for s in ('l10','spatial') for r in json.loads((R/f'curve_{s}.json').read_text())['records']]
    lookup={(r['suite'],r['stream'],r['size'],r['variant']):r for r in records}
    sizes=[50,100,200,300,500]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True,layout='constrained')
    for row,suite in enumerate(('l10','spatial')):
        for col,(key,label) in enumerate((('action_rms','Action head RMS / current sigma'),('successor_rms','Observed successor displacement RMS'))):
            ax=axs[row,col]
            for stream,color in (('inf','#0072B2'),('cache','#D55E00')):
                for variant,style in (('refit','-'),('frozen50','--')):
                    yy=[lookup[suite,stream,n,variant][key] for n in sizes]
                    ax.plot(sizes[1:],yy[1:],style,color=color,marker='o' if variant=='refit' else 's',markersize=4,label=f'{stream} / {variant}')
                    ax.scatter([50],[yy[0]],color=color,marker='x',s=45)
            ax.set_title(f'libero_{"10" if suite=="l10" else "spatial"}');ax.set_ylabel(label)
            ax.set_xticks(sizes);ax.grid(alpha=.2);ax.axvspan(35,70,color='gray',alpha=.09)
            if row==1:ax.set_xlabel('Library episodes (50 = separate current collection)')
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Demo-data scaling: fixed recorded A-pool observations\n100–500 are nested B-pool banks; no SR estimate',fontsize=12)
    for ext in ('png','pdf'):fig.savefig(R/f'demo_curve_losses.{ext}')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,suite in zip(axs,('l10','spatial')):
        for stream,color in (('inf','#0072B2'),('cache','#D55E00')):
            for key,style in (('common_d1','-'),('common_d16','--')):
                yy=[lookup[suite,stream,n,'frozen50'][key] for n in sizes]
                ax.plot(sizes[1:],yy[1:],style,color=color,marker='o',markersize=4,label=f'{stream} / {key[7:].upper()}')
                ax.scatter([50],[yy[0]],color=color,marker='x',s=45)
        ax.set_title(suite);ax.set_ylabel('Distance in the same frozen-50 geometry');ax.set_xlabel('Library episodes')
        ax.set_xticks(sizes);ax.grid(alpha=.2);ax.axvspan(35,70,color='gray',alpha=.09)
    axs[0].legend(fontsize=8)
    fig.suptitle('Common-frame density (identical for both fit labels)\n50 is not part of the nested demo series',fontsize=12)
    for ext in ('png','pdf'):fig.savefig(R/f'demo_curve_density.{ext}')
    plt.close(fig)
    print('Wrote demo_curve_{losses,density}.{png,pdf}')

if __name__=='__main__':main()
