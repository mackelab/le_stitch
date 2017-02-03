
import numpy as np

class ObservationScheme(object):

	def __init__(self, p, T, 
				sub_pops=None, 
				obs_pops=None, 
				obs_time=None,
				obs_idx=None,
				idx_grp=None,
				mask=None):

		self._sub_pops = (np.arange(p),) if sub_pops is None else self._argcheck_sub_pops(sub_pops)
		self._obs_pops = np.array((0,)) if obs_pops is None else self._argcheck_obs_pops(obs_pops)
		self._obs_time = np.array((T,)) if obs_time is None else self._argcheck_obs_time(obs_time)

		self._mask = self._argcheck_mask(mask)
		self._use_mask = False if mask is None else True # general: mask overrules scheme

		self._p = p
		self._T = T

		self.num_subpops = len(self._sub_pops)
		self.num_obstime = self._obs_time.size

		self.check_obs_scheme()
		self.comp_subpop_stats()

	@staticmethod
	def _argcheck_sub_pops(sub_pops):

		assert len(sub_pops) > 0

		for pop in sub_pops:
			assert isinstance(pop, np.ndarray)

		return sub_pops

	@staticmethod
	def _argcheck_obs_pops(obs_pops):

		return np.asarray(obs_pops)


	@staticmethod
	def _argcheck_obs_time(obs_time):

		assert(obs_time[0]!=0)

		return np.asarray(obs_time)

	def _argcheck_mask(self, mask):

		if mask is None:
			pass
		else:
			assert np.all(mask.shape == (self._T,self._p))
			mask = np.nan_to_num(mask).astype(dtype=bool)

		return mask

	def check_obs_scheme(self):
		" Checks the internal validity of provided observation schemes "

		# check sub_pops
		idx_union = np.sort(self._sub_pops[0])
		i = 1
		while idx_union.size < self._p and i < len(self._sub_pops):
			idx_union = np.union1d(idx_union, self._sub_pops[i]) 
			i += 1
		if idx_union.size != self._p or np.any(idx_union!=np.arange(self._p)):
			raise Exception(('all subpopulations together have to cover '
			'exactly all included observed varibles y_i in y.'
			'This is not the case. Change the difinition of '
			'subpopulations in variable sub_pops or reduce '
			'the number of observed variables p. '
			'The union of indices of all subpopulations is'),
			idx_union )

		# check obs_time
		if not self._obs_time[-1]==self._T:
			raise Exception(('Entries of obs_time give the respective ends of '
							'the periods of observation for any '
							'subpopulation. Hence the last entry of obs_time '
							'has to be the full recording length. The last '
							'entry of obs_time before is '), self._obs_time[-1])

		if np.any(np.diff(self._obs_time)<1):
			raise Exception(('lengths of observation have to be at least 1. '
							'Minimal observation time for a subpopulation: '),
							np.min(np.diff(self._obs_time)))

		# check obs_pops
		if not self._obs_time.size == self._obs_pops.size:
			raise Exception(('each entry of obs_pops gives the index of the '
							'subpopulation observed up to the respective '
							'time given in obs_time. Thus the sizes of the '
							'two arrays have to match. They do not. '
							'no. of subpop. switch points and no. of '
							'subpopulations ovserved up to switch points '
							'are '), (self._obs_time.size, self._obs_pops.size))

		idx_pops = np.sort(np.unique(self._obs_pops))
		if not np.min(idx_pops)==0:
			raise Exception(('first subpopulation has to have index 0, but '
							'is given the index '), np.min(idx_pops))
		elif not idx_pops.size == len(self._sub_pops):
			raise Exception(('number of specified subpopulations in variable '
							'sub_pops does not meet the number of '
							'subpopulations indexed in variable obs_pops. '
							'Delete subpopulations that are never observed, '
							'or change the observed subpopulations in '
							'variable obs_pops accordingly. The number of '
							'indexed subpopulations is '),
							len(self._sub_pops))
		elif not np.all(np.diff(idx_pops)==1):
			raise Exception(('subpopulation indices have to be consecutive '
							'integers from 0 to the total number of '
							'subpopulations. This is not the case. '
							'Given subpopulation indices are '),
							idx_pops)

		if not self._mask is None:
			assert np.all(self._mask.shape == (self._T,self._p))


	def comp_subpop_stats(self):
		"computes a collection of helpful index sets for the stitching context"

		sub_pops = self._sub_pops
		obs_pops = self._obs_pops

		if obs_pops is None:
		    obs_pops = tuple(range(self.num_subpops))
		self.obs_idx, self.idx_grp = self._get_obs_index_groups()
		self.overlaps, self.overlap_grp, self.idx_overlap = self._get_obs_index_overlaps()
		self.idx_time = []
		for i in range(len(self.idx_grp)):
			ts = [ range(self._obs_time[0]) ] if i in self.obs_idx[0] else []
			for t in range(1,len(self.obs_time)):
				if i in self.obs_idx[t]:
					ts.append( range(self._obs_time[t-1], self.obs_time[t]) )
			ts = np.hstack( [ np.asarray(ts_) for ts_ in ts] )
			self.idx_time.append( ts )

	def _get_obs_index_groups(self):

	    J = np.zeros((self._p, self.num_subpops))  
	    for i in range(self.num_subpops):   
	        if self._sub_pops[i].size > 0:  
	            J[self._sub_pops[i],i] = 1   

	    twoexp = np.power(2,np.arange(self.num_subpops)) 
	    hsh = np.sum(J*twoexp,1)                        

	    lbls = np.unique(hsh) 
	                                     
	    idx_grp = [] # list of arrays that define the index groups
	    for i in range(lbls.size):
	        idx_grp.append(np.where(hsh==lbls[i])[0])

	    obs_idx = [] # list of arrays giving the index groups observed at each
	                 # given time interval
	    for i in range(len(self._obs_pops)):
	        obs_idx.append([])
	        for j in np.unique(hsh[np.where(J[:,self._obs_pops[i]]==1)]):
	            obs_idx[i].append(np.where(lbls==j)[0][0])            

	    return np.asarray(obs_idx), np.asarray(idx_grp)

	def _get_obs_index_overlaps(self):
		num_idx_grps = len(self.idx_grp)

		idx_overlap = []
		idx = np.zeros(num_idx_grps, dtype=int)
		for j in range(num_idx_grps):
			idx_overlap.append([])
			for i in range(self.num_subpops):
				if np.any(np.intersect1d(self._sub_pops[i], self.idx_grp[j])):
					idx[j] += 1
					idx_overlap[j].append(i)
			idx_overlap[j] = np.array(idx_overlap[j])

		overlaps = [self.idx_grp[i] for i in np.where(idx>1)[0]]
		overlap_grp = [i for i in np.where(idx>1)[0]]
		idx_overlap = [idx_overlap[i] for i in np.where(idx>1)[0]]

		return overlaps, overlap_grp, idx_overlap	    

	def set_schedule(self, obs_pops, obs_time):
		self._obs_pops = self._argcheck_obs_pops(obs_pops)
		self._obs_time = self._argcheck_obs_time(obs_time)
		self.check_obs_scheme()

	def gen_mask_from_scheme(self):
		sub_pops,obs_pops,obs_time=self._sub_pops,self._obs_pops,self._obs_time
		self._mask = np.zeros((self._T,self._p),dtype=bool)
		self._mask[np.ix_(range(obs_time[0]), sub_pops[obs_pops[0]])] = True
		for i in range(1,len(obs_time)):
			ts = range(obs_time[i-1], obs_time[i])
			self._mask[np.ix_(ts,sub_pops[obs_pops[i]])] = True

	def gen_get_observed(self, use_mask=None):

		use_mask = self._use_mask if use_mask is None else use_mask

		if use_mask:
			assert np.all(self._mask.shape == (self._T, self._p))
			def get_observed(t):
				return np.where(self._mask[t,:])[0]
		else:
			def get_observed(t):
				i = self._obs_pops[np.digitize(t, self._obs_time)]
				return self._sub_pops[i]

		return get_observed


	def gen_get_idx_grp(self):

		def get_idx_grp(t):
			return self.obs_idx[np.searchsorted(self._obs_time,t,side='right')]

		return get_idx_grp


	def gen_get_coobs_intervals(self, lag_range):

		kl_ = np.max(lag_range) + 1
		def get_coobs_intervals(i,j,m):

			ts_i = np.zeros(self._T, dtype=bool)
			tsm = self.idx_time[i]+m
			cut_off = np.searchsorted(tsm[-kl_:],self._T-kl_+m,side='right')
			ts_i[tsm[:-kl_+cut_off]] = True

			ts_j = np.zeros(self._T, dtype=bool)
			ts_j[self.idx_time[j]] = True

			return (ts_i and ts_j)

		return get_coobs_intervals

	def comp_coocurrence_weights(self, lag_range, sso=False, idx_a=None, idx_b=None):

		kl_ = np.max(lag_range)+1

		if sso: # 'serial subset ovservations' allow to work with variable subsets

			assert not self._use_mask # sso assumes simple block-wise observations
			if not (idx_a is None and idx_b is None):
				print('ObservationScheme warning: ignoring arguments idx_a,idx_b if sso=True')

			# could make this compute much faster if we assumed long stretches of
			# time observing the same subpopulation, but the edge cases are tedious.
			# Below version is simple, but takes care of switching variable subsets.
			idx_grp,get_idx_grp = self.idx_grp, self.gen_get_idx_grp()
			obs_time = self._obs_time
			ng = len(idx_grp)
			W = [np.zeros((ng,ng), dtype=int) for m in lag_range]
			for m in range(len(lag_range)):            
				m_ = lag_range[m]
				for t in range(self._T-kl_):
					is_, js_ = get_idx_grp(t+m_), get_idx_grp(t)
					W[m][np.ix_(is_,js_)] += 1
				W[m] = 1./np.maximum(W[m]-1, 1)

		else: # we have to work in p x p observed space, but can subsample

			p = self._p
			idx_a = np.arange(p) if idx_a is None else np.sort(idx_a)
			idx_b = np.arange(p) if idx_b is None else np.sort(idx_b)
			pa, pb = len(idx_a), len(idx_b)

			get_observed = self.gen_get_observed()
			if pa<p:
				def get_obs_sub_idx_a(t):
					return np.in1d(idx_a, np.intersect1d(get_observed(t), idx_a))
			else:
				def get_obs_sub_idx_a(t):
					return get_observed(t)

			if pb<p:
				def get_obs_sub_idx_b(t):
					return np.in1d(idx_b, np.intersect1d(get_observed(t), idx_b))
			else:
				def get_obs_sub_idx_b(t):
					return get_observed(t)

			W = [np.zeros((pa,pb), dtype=int) for m in lag_range]
			for m in range(len(lag_range)):
				m_ = lag_range[m]
				for t in range(self._T-kl_):
					a, b = get_obs_sub_idx_a(t+m_), get_obs_sub_idx_b(t)
					W[m][np.ix_(a,b)] += 1
				W[m] = 1./np.maximum(W[m]-1, 1)

		return W



	@property
	def sub_pops(self):
		return self._sub_pops

	@sub_pops.setter
	def sub_pops(self, sub_pops):
		self._sub_pops = self._argcheck_sub_pops(sub_pops)
		self.num_pops = len(self._sub_pops)
		self.check_obs_scheme()
		self.comp_subpop_stats()

	@property
	def obs_pops(self):
		return self._obs_pops

	@obs_pops.setter
	def obs_pops(self, obs_pops):
		self._obs_pops = self._argcheck_obs_pops(obs_pops)

	@property
	def obs_time(self):
		return self._obs_time

	@obs_time.setter
	def obs_time(self, obs_time):
		self._obs_time = self._argcheck_obs_time(obs_time)

	@property
	def T(self):
		return self._T

	@T.setter
	def T(self,T):
		self._T = T
		self.check_obs_scheme()

	@property
	def mask(self):
		return self._mask

	@mask.setter
	def mask(self, mask):
		self._mask = self._argcheck_mask(mask)
		self._use_mask = True

	@property
	def use_mask(self):
		return self._use_mask

	@use_mask.setter
	def use_mask(self, use_mask):
		assert use_mask in (True, False)
		self._use_mask = use_mask


	@property
	def p(self):
		return self._p
