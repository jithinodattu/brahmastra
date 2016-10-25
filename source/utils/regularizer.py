
def L1(params):
	return sum([w.sum() for w in params if w.name=='W'])

def L2_sqr(params):
	return sum([(w**2).sum() for w in params if w.name=='W'])