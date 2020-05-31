# SOLUTION TO PROBLEM 1
# -------------------------
# -------------------------
# 
# For each company, this source finds 
# -> frequent episodes
# -> cardinalities nX and nY
# -> frequency of each frequent episode
# -> confidence of each frequent episode
# -> recall of each frequent episode
# 
# -------------------------

import numpy as np
# read data
f = open('YahooFinance.data')
lines = f.readlines()
lines = lines[1:] # ignore header line 

# Parse dataset
sequence = [];
for line in lines:
	x = line.split('\t')[2].strip().split('=')
	y = (int(x[1]), x[0])
	sequence.append(y)

# find unique companies in the dataset
companies = []
for seq in sequence:
    companies.append(seq[1])
companies = np.unique(companies)
print('Total number of companies = %d' % (len(companies)))


# Get sequence of each company
events = {}
for comp in companies:
	events[comp] = []

for seq in sequence:
  events[seq[1]].append(seq[0])

# find episodes of all the companies
episodes = {}
for comp in companies:
	ev = events[comp] # contains increase, drop or const(0) values of 1255 days
	ep = [];
	for i in range(len(ev)-1):
		prev = ev[i] # check value of previous day
		curr = ev[i+1] # check value of current day

		# if values of two consecutive days are not 0, consider it an episode
		if (prev != 0 and curr != 0):
			epsd = [prev, curr]
			ep.append(epsd)
	episodes[comp] = np.array(ep)
	print('Episodes of', comp, ' = ', len(episodes[comp]))


# Loop through companies
for comp in companies:

	print('----------------------------')
	print('Company :', comp)
	print('----------------------------')
	ep = episodes[comp]
	my_array = ep
	dt = np.dtype((np.void, my_array.dtype.itemsize * my_array.shape[1]))
	b = np.ascontiguousarray(my_array).view(dt)
	unq, cnt = np.unique(b, return_counts=True)

	# unq contains unique episodes of company
	# cnt contains count of each unique episode of company
	unq = unq.view(my_array.dtype).reshape(-1, my_array.shape[1])

	# find episodes which occur atleast min_freq times i.e. atleast times
	ind = np.where(cnt >= 50)[0]
	freq_eps = unq[ind,:]

	L = len(ep) # total number of episodes for a company
	print('Number of frequent episodes of', comp, ':', freq_eps.shape[0])
	print('Frequent episodes of', comp, ':\n', freq_eps)
	i = 0
	for eps in freq_eps:
		eps_count = cnt[ind[i]]
		i = i+1
		x = eps[0]
		y = eps[1]
		indX = np.where(ep[:,0] == x)[0]
		nX = len(indX)
		indY = np.where(ep[:,0] == y)[0]
		nY = len(indY)
		# calculate frequency of frequent episode
		frequency = float(eps_count)/float(L)

		# calculate confidence
		confidence = float(eps_count)/float(nX)

		# calculate recall
		nYX = 0
		for e in ep:
			if (e[1] == eps[0] and e[0] == eps[1]):
				nYX = nYX+1
		recall = float(nYX)/float(nY)
		print('Frequent episode: [%d, %d], nX->Y = %d, nY->X = %d, nX = %d, nY = %d, frequency = %.3f, confidence = %.3f, recall = %.3f' % (eps[0], eps[1], eps_count, nYX, nX, nY, frequency, confidence, recall))

	print('____________________________')
	print('\n\n')



