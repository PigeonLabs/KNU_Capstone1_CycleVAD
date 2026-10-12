import numpy as np
import pytest
from cycle_vad.stall import StallModel, StallState, motion, PrefixAlarm, adaptive_thresholds

@pytest.fixture(scope='module')
def fitted():
    rng=np.random.default_rng(42)
    def bank(prefix):
        return {f'{prefix}{j}':(np.cumsum(rng.normal(size=(300,8))*.04,axis=0)+1,np.arange(300)/75%1) for j in range(4)}
    return StallModel().fit(bank('fit'),bank('ref'),400)

@pytest.mark.parametrize('stride',[1,2])
@pytest.mark.parametrize('variant',['S0','S1','S2','S3'])
def test_stream_and_prefix(fitted,stride,variant):
    rng=np.random.default_rng(21);g=np.cumsum(rng.normal(size=(160,8))*.04,axis=0)+1
    g[30:100]=g[29];theta=np.arange(len(g))/100%1
    batch=fitted.score(g,theta,[variant],stride)
    state=StallState(fitted,stride,variant)
    rows=[state.step(v,t) for v,t in zip(g,theta)]
    np.testing.assert_allclose([r['raw'] for r in rows],batch['raw'][:,0],atol=1e-10)
    np.testing.assert_allclose([r['score'] for r in rows],batch['scores'][:,0],atol=1e-9)
    np.testing.assert_array_equal([r['valid'] for r in rows],batch['valid'])
    np.testing.assert_array_equal(fitted.score(g[:70],theta[:70],[variant],stride)['scores'],batch['scores'][:70])
    g[70:]=rng.normal(size=g[70:].shape)
    np.testing.assert_array_equal(fitted.score(g,theta,[variant],stride)['scores'][:70],batch['scores'][:70])
    state.reset();assert state.step(g[0],theta[0])['score']==0

def test_motion_warmup_and_freeze():
    g=np.tile([1.,2.,3.],(20,1));m=motion(g)
    assert np.isnan(m[:4,1]).all()
    np.testing.assert_allclose(m[4:],0,atol=1e-15)

@pytest.mark.parametrize('stride',[1,2])
@pytest.mark.parametrize('n',[1,12,48,49,100])
def test_adaptive_no_lookahead(stride,n):
    s=np.repeat(np.linspace(2,20,(n+stride-1)//stride),stride)[:n,None]
    fixed=np.array([[5.],[10.]])
    target=adaptive_thresholds(s,fixed,np.array([1.]),stride)
    for q in range(2):
        state=PrefixAlarm(fixed[q,0],1)
        rows=[state.step(float(v[0]),i%stride==0) for i,v in enumerate(s)]
        np.testing.assert_allclose([r['threshold'] for r in rows],target[:,q,0])
        assert all(not r['alarm'] or s[i,0]>fixed[q,0] for i,r in enumerate(rows))
        if n>48: assert rows[47]['threshold']==fixed[q,0] and rows[48]['threshold']>fixed[q,0]
    s2=s.copy();s2[50:]*=100
    np.testing.assert_array_equal(adaptive_thresholds(s2,fixed,np.array([1.]),stride),target)

def test_fit_rejects_overlap():
    x={'same':(np.ones((10,3)),np.zeros(10))}
    with pytest.raises(ValueError):StallModel().fit(x,x,100)
