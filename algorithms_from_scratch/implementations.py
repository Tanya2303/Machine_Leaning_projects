"""Small NumPy implementations for learning; use sklearn pipelines for production."""
import numpy as np

class GradientDescent:
    """Batch gradient descent for a differentiable scalar objective."""
    def __init__(self, learning_rate=0.01, n_iter=1000): self.lr=learning_rate; self.n_iter=n_iter
    def minimize(self, gradient, initial):
        x=np.asarray(initial,dtype=float).copy(); self.history_=[]
        for _ in range(self.n_iter):
            x -= self.lr*np.asarray(gradient(x)); self.history_.append(x.copy())
        self.solution_=x; return self

class LinearRegressionGD:
    def __init__(self, learning_rate=.01, n_iter=3000): self.lr=learning_rate; self.n_iter=n_iter
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,float).reshape(-1,1); self.mean_=X.mean(0); self.scale_=X.std(0); self.scale_[self.scale_==0]=1
        Z=(X-self.mean_)/self.scale_; A=np.c_[np.ones(len(Z)),Z]; w=np.zeros((A.shape[1],1)); self.loss_history_=[]
        for _ in range(self.n_iter):
            err=A@w-y; w -= self.lr*(A.T@err)/len(y); self.loss_history_.append(float(np.mean(err**2)/2))
        self.intercept_=float(w[0]); self.coef_=w[1:,0]; return self
    def predict(self,X): return self.intercept_+((np.asarray(X)-self.mean_)/self.scale_)@self.coef_

class LogisticRegressionGD:
    def __init__(self, learning_rate=.1, n_iter=3000, l2=0.0): self.lr=learning_rate; self.n_iter=n_iter; self.l2=l2
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,float).reshape(-1); self.classes_=np.unique(y)
        if len(self.classes_)!=2: raise ValueError('Binary targets only; encode as 0/1.')
        y=(y==self.classes_[1]).astype(float); self.mean_=X.mean(0); self.scale_=X.std(0); self.scale_[self.scale_==0]=1
        A=np.c_[np.ones(len(X)),(X-self.mean_)/self.scale_]; w=np.zeros(A.shape[1]); self.loss_history_=[]
        for _ in range(self.n_iter):
            p=self._sigmoid(A@w); w -= self.lr*(A.T@(p-y)/len(y)+self.l2*np.r_[0,w[1:]]/len(y))
            p=np.clip(self._sigmoid(A@w),1e-12,1-1e-12); self.loss_history_.append(float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))))
        self.intercept_=w[0]; self.coef_=w[1:]; return self
    @staticmethod
    def _sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))
    def predict_proba(self,X):
        p=self._sigmoid(self.intercept_+((np.asarray(X)-self.mean_)/self.scale_)@self.coef_); return np.c_[1-p,p]
    def predict(self,X): return self.classes_[(self.predict_proba(X)[:,1]>=.5).astype(int)]

class KNN:
    """Brute-force KNN classification or regression; scale features before use."""
    def __init__(self,k=5,task='classification'): self.k=k; self.task=task
    def fit(self,X,y): self.X_=np.asarray(X,float); self.y_=np.asarray(y); return self
    def predict(self,X):
        X=np.asarray(X,float); out=[]
        for x in X:
            ix=np.argsort(np.sum((self.X_-x)**2,axis=1))[:self.k]; vals=self.y_[ix]
            out.append(np.mean(vals.astype(float)) if self.task=='regression' else np.unique(vals,return_counts=True)[0][np.argmax(np.unique(vals,return_counts=True)[1])])
        return np.asarray(out)

class KMeans:
    def __init__(self,k=3,n_iter=100,random_state=42): self.k=k; self.n_iter=n_iter; self.seed=random_state
    def fit(self,X):
        X=np.asarray(X,float); rng=np.random.default_rng(self.seed); C=X[rng.choice(len(X),self.k,replace=False)].copy()
        for _ in range(self.n_iter):
            labels=np.argmin(((X[:,None,:]-C[None,:,:])**2).sum(2),axis=1)
            new=C.copy()
            for j in range(self.k):
                if np.any(labels==j): new[j]=X[labels==j].mean(0)
                else: new[j]=X[rng.integers(len(X))]
            if np.allclose(C,new): break
            C=new
        self.cluster_centers_=C; self.labels_=np.argmin(((X[:,None,:]-C[None,:,:])**2).sum(2),axis=1); return self
    def predict(self,X): return np.argmin(((np.asarray(X)[:,None,:]-self.cluster_centers_[None,:,:])**2).sum(2),axis=1)

class DecisionTreeClassifierScratch:
    """Readable numeric-feature CART classifier using Gini impurity."""
    def __init__(self,max_depth=4,min_samples_split=2): self.depth=max_depth; self.min_split=min_samples_split
    @staticmethod
    def _gini(y):
        if not len(y): return 0
        p=np.unique(y,return_counts=True)[1]/len(y); return 1-float(p@p)
    def _grow(self,X,y,d):
        labels,counts=np.unique(y,return_counts=True); node={'value':labels[np.argmax(counts)]}
        if d>=self.depth or len(y)<self.min_split or len(labels)==1: return node
        best=(0,None,None)
        parent=self._gini(y)
        for j in range(X.shape[1]):
            for t in np.unique(X[:,j])[:-1]:
                mask=X[:,j]<=t
                if not mask.any() or mask.all(): continue
                gain=parent-(mask.mean()*self._gini(y[mask])+(~mask).mean()*self._gini(y[~mask]))
                if gain>best[0]: best=(gain,j,t)
        if best[1] is None: return node
        mask=X[:,best[1]]<=best[2]; node.update(feature=best[1],threshold=best[2],left=self._grow(X[mask],y[mask],d+1),right=self._grow(X[~mask],y[~mask],d+1)); return node
    def fit(self,X,y): self.tree_=self._grow(np.asarray(X,float),np.asarray(y),0); return self
    def _one(self,x,n):
        while 'feature' in n: n=n['left'] if x[n['feature']]<=n['threshold'] else n['right']
        return n['value']
    def predict(self,X): return np.asarray([self._one(x,self.tree_) for x in np.asarray(X,float)])

class GaussianNaiveBayes:
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y); self.classes_=np.unique(y); self.prior_=[]; self.mean_=[]; self.var_=[]
        for c in self.classes_:
            a=X[y==c]; self.prior_.append(len(a)/len(X)); self.mean_.append(a.mean(0)); self.var_.append(a.var(0)+1e-9)
        return self
    def predict_log_proba(self,X):
        X=np.asarray(X,float); scores=[]
        for p,m,v in zip(self.prior_,self.mean_,self.var_): scores.append(np.log(p)-.5*np.sum(np.log(2*np.pi*v)+(X-m)**2/v,axis=1))
        s=np.array(scores).T; return s-np.logaddexp.reduce(s,axis=1,keepdims=True)
    def predict(self,X): return self.classes_[self.predict_log_proba(X).argmax(1)]

class PCA:
    def __init__(self,n_components): self.n_components=n_components
    def fit(self,X):
        X=np.asarray(X,float); self.mean_=X.mean(0); centered=X-self.mean_; _,s,Vt=np.linalg.svd(centered,full_matrices=False)
        self.components_=Vt[:self.n_components]; self.explained_variance_=s[:self.n_components]**2/max(1,len(X)-1)
        self.explained_variance_ratio_=self.explained_variance_/(s**2/max(1,len(X)-1)).sum(); return self
    def transform(self,X): return (np.asarray(X)-self.mean_)@self.components_.T
    def fit_transform(self,X): return self.fit(X).transform(X)
