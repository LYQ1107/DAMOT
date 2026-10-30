"""Check nontrivial statistic implementations against independent references."""
import numpy as np
from sklearn.metrics import silhouette_samples
from statistics_utils import cosine_silhouette,paired_summary,weights,ratio_replicates

rng=np.random.RandomState(4);X=rng.randn(79,12);X/=np.linalg.norm(X,axis=1,keepdims=True)
labels=np.array([0]*22+[1]*29+[2]*27+[3])
expected=silhouette_samples(X,labels,metric='cosine')
observed=cosine_silhouette(X,labels,chunk=17)
assert np.allclose(expected,observed,atol=1e-12),np.max(np.abs(expected-observed))
assert observed[-1]==0
print('PASS exact cosine silhouette vs sklearn including singleton',flush=True)
s=paired_summary(np.zeros(8));assert s['ci_low']==s['ci_high']==0 and s['wilcoxon_p_two_sided']==1
assert np.array_equal(weights(7),weights(7))
num=np.array([1.,6.,3.]);den=np.array([2.,12.,6.])
assert np.allclose(ratio_replicates(num,den,weights(3)),.5)
print('PASS paired bootstrap zero effects, reproducibility and ratio of sums',flush=True)
