"""Render exported Cube arrays, with no regenerated or decorative simulation values."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def render(run_directory):
    root=Path(run_directory)
    m=json.loads((root/'manifest.json').read_text())
    with np.load(root/'cube-fields.npz') as data:
        lactate=data['lactate']; ph=data['pH']; biomass=data['composition'].sum(axis=3)
    fig=plt.figure(figsize=(14,6.5),facecolor='#081720')
    ax=fig.add_subplot(121,projection='3d',facecolor='#081720')
    n=lactate.shape[0]
    step=max(1,n//12)
    x,y,z=np.mgrid[0:n:step,0:n:step,0:n:step]
    vals=lactate[::step,::step,::step]
    scatter=ax.scatter(x.ravel(),y.ravel(),z.ravel(),c=vals.ravel(),cmap='YlOrRd',s=14,alpha=.65)
    ax.set(xlabel='x voxel: enamel to tissue',ylabel='y voxel: adjacent biofilm',zlabel='z voxel: deep to oral')
    ax.set_title('Lactate model field in the 3-D Cube',color='white',pad=20)
    ax.tick_params(colors='#b4cbd5')
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]: axis.label.set_color('#b4cbd5')
    cb=fig.colorbar(scatter,ax=ax,shrink=.55,pad=.1)
    cb.set_label('Legacy model units',color='#b4cbd5');cb.ax.tick_params(colors='#b4cbd5')
    ax2=fig.add_subplot(122,facecolor='#081720')
    image=ax2.imshow(ph[:,n//2,:].T,origin='lower',cmap='viridis',aspect='auto')
    ax2.set_title('pH slice at the middle y plane',color='white',pad=20)
    ax2.set_xlabel('x voxel: enamel to tissue',color='#b4cbd5')
    ax2.set_ylabel('z voxel: deep to oral',color='#b4cbd5')
    ax2.tick_params(colors='#b4cbd5')
    c=fig.colorbar(image,ax=ax2,shrink=.6);c.set_label('Modeled pH',color='#b4cbd5');c.ax.tick_params(colors='#b4cbd5')
    fig.suptitle('FASOLATI  /  BIOREACTOR CUBE FOUNDATION',color='white',fontsize=18,y=.98)
    fig.text(.5,.025,f"{m['voxels']:,} voxels  •  {m['steps']} steps  •  115 targets: 7 model proxies, 108 unimplemented\nHypothesis model. Physical clock unanchored. No sensor measurements or clinical prediction.",ha='center',color='#b4cbd5',fontsize=11)
    fig.subplots_adjust(top=.87,bottom=.2,wspace=.32)
    fig.savefig(root/'cube-preview.png',dpi=150,facecolor=fig.get_facecolor())
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_directory')
    render(p.parse_args().run_directory)

