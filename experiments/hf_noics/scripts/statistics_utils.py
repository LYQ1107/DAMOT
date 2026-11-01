"""Exact cosine silhouette and paired video bootstrap helpers."""
import numpy as np
from scipy.stats import wilcoxon,spearmanr

def weights(n,repeats=10000):
    assert n>0
    draws=np.random.RandomState(0).randint(0,n,size=(repeats,n))
    return np.array([np.bincount(r,minlength=n) for r in draws],dtype=float)

def interval(a):
    a=np.asarray(a);valid=a[np.isfinite(a)]
    assert len(valid)>0,'No finite bootstrap replicates'
    return float(np.percentile(valid,2.5)),float(np.percentile(valid,97.5)),len(valid)

def ratio_replicates(num,den,w):
    a=w@np.asarray(num,dtype=float);b=w@np.asarray(den,dtype=float)
    return np.divide(a,b,out=np.full_like(a,np.nan),where=b>0)

def paired_summary(difference,lower_is_better=False):
    d=np.asarray(difference,dtype=float);assert len(d)>0 and np.isfinite(d).all()
    lo,hi,reps=interval(weights(len(d))@d/len(d))
    if np.all(d==0):p=1.
    else:p=float(wilcoxon(d,zero_method='wilcox',alternative='two-sided').pvalue)
    return dict(videos=len(d),mean=float(d.mean()),median=float(np.median(d)),
                improved_fraction=float(np.mean(d<0 if lower_is_better else d>0)),ci_low=lo,ci_high=hi,
                bootstrap_valid_replicates=reps,wilcoxon_p_two_sided=p)

def correlation(x,y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return dict(rho=None,p=None)
    r,p=spearmanr(x,y);return dict(rho=float(r),p=float(p))

def cosine_silhouette(X,labels,chunk=2048):
    """Exactly equal to full cosine silhouette, without an N-by-N distance matrix.

    For normalized x, mean cosine distance to a cluster equals 1-x dot
    that cluster's unnormalized mean. Exclude self from the own-cluster mean.
    """
    X=np.asarray(X,dtype=float)
    assert np.allclose(np.linalg.norm(X,axis=1),1,atol=1e-6)
    _,ids=np.unique(labels,return_inverse=True);count=np.bincount(ids);k=len(count)
    assert 1<k<len(X)
    sums=np.zeros((k,X.shape[1]));np.add.at(sums,ids,X);means=sums/count[:,None]
    result=np.empty(len(X))
    for start in range(0,len(X),chunk):
        end=min(start+chunk,len(X));ii=ids[start:end];dot=X[start:end]@means.T
        own=dot[np.arange(end-start),ii].copy()
        a=np.divide(count[ii]*(1-own),count[ii]-1,out=np.zeros(end-start),where=count[ii]>1)
        dot[np.arange(end-start),ii]=-np.inf
        b=np.maximum(0,1-dot.max(axis=1));a=np.maximum(a,0)
        den=np.maximum(a,b);s=np.divide(b-a,den,out=np.zeros_like(a),where=den>0);s[count[ii]==1]=0
        result[start:end]=s
    return result
