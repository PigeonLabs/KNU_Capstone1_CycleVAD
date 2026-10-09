"""Check fixed-arm reproduction and all saved event/window decisions."""
import argparse
from pathlib import Path
import numpy as np
from cycle_vad.paper_calibration import read,load_bank,deserialize_thresholds,alarms,SCENES
ROOT=Path(__file__).resolve().parents[1]
def b0():
    paths=sorted((ROOT/'results/E15B0').glob('seed*/R*/fold*/run.json'));assert len(paths)==60
    events_count=0
    for p in paths:
        r=read(p);rel=p.parent.relative_to(ROOT/'results/E15B0');data=load_bank(ROOT,rel);th=deserialize_thresholds(r['thresholds'])
        old=read(ROOT/'results/E11B'/rel/'run.json');np.testing.assert_array_equal(th[:,1,0],np.array(old['window_thresholds'])[[3,1,0],6])
        saved={e['id']:e for e in read(ROOT/'results/E11B'/rel/'historical_events.json')}
        for e in read(p.with_name('events.json')):
            h=np.array(e['hit'],bool);d=np.array(e['delay']);o=saved[e['id']]
            assert (d[0,0] if d[0,0]>=0 else None)==o['delay_primary'][4]
            assert (d[0,1] if d[0,1]>=0 else None)==o['delay_primary'][6]
            assert np.all(~h[:,0]|h[:,4]);events_count+=1
        for n in read(p.with_name('normal.json')):
            o=n['overlap_q99'];f=np.array(n['grid_far'])[0]
            np.testing.assert_allclose(f[4],f[0]+o['P_only'],atol=1e-14)
            baseline=next(v for v in read(ROOT/'results/E11B'/rel/'normal.json') if v['id']==n['id'])
            np.testing.assert_array_equal(np.array(n['grid_far'])[:,0],np.array(baseline['grid_alarm_rates'])[[3,1,0],4])
            np.testing.assert_array_equal(np.array(n['grid_far'])[:,1],np.array(baseline['grid_alarm_rates'])[[3,1,0],6])
    assert events_count==198
    text=f'PASS E15B0: 60 units, {events_count} seed-specific historical events; original C/CP delays and normal grid alarms reproduced; OR preservation and normal cost identity verified.\n'
    (ROOT/'results/E15B0/validation.txt').write_text(text);print(text)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bc',action='store_true');a=p.parse_args();b0()
    if a.bc:raise RuntimeError('E15BC validation is pending')
