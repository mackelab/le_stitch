
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

		def co_observed(x, i):
			for idx in self.obs_idx:
				if x in idx and i in idx:
					return True
			return False        

		num_idx_grps, self.co_obs = len(self.idx_grp), []
		for i in range(num_idx_grps):    
			self.co_obs.append([self.idx_grp[x] for x in np.arange(len(self.idx_grp)) \
				if co_observed(x,i)])
			self.co_obs[i] = np.sort(np.hstack(self.co_obs[i]))
		    

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

	    return obs_idx, idx_grp

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

	@property
	def p(self):
		return self._p
