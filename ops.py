import numpy as np

def softmax(ary):
	ary_exp = np.exp(ary-np.max(ary, axis=-1).reshape(-1, 1))
	return ary_exp / np.sum(ary_exp, axis=-1).reshape(-1, 1)

class Tensor(np.ndarray):
	def __new__(cls, input_array, requires_grad=False):
		obj = np.asarray(input_array).view(cls)
		obj.grad = np.zeros(input_array.shape, dtype=np.float32)
		obj.requires_grad = requires_grad
		return obj

	def backward(self, grad):
		self.grad += grad

def matmul(t1, t2):
	v = np.matmul(t1, t2)
	v.requires_grad = True

	def backward(grad):
		g1 = None
		g2 = None
		if t1.requires_grad:
			g1 = grad.dot(t2.T)
			t1.backward(g1)
		if t2.requires_grad:
			g2 = t1.T.dot(grad)
			t2.backward(g2)

	v.backward = backward
	return v

def add(t1, t2):
	v = t1 + t2
	v.requires_grad = True

	def backward(grad):
		g1 = None
		g2 = None
		if t1.requires_grad:
			g1 = grad
			t1.backward(g1)
		if t2.requires_grad:
			# hack for broadcasting
			if t2.squeeze().ndim == 1:
				g2 = np.mean(grad, axis=0)
			else:
				g2 = grad
			t2.backward(g2)

	v.backward = backward
	return v

def relu(t1):
	v = np.maximum(t1, 0.0)
	v.requires_grad = True
	mask = (t1 > 0).astype(np.int32)

	def backward(grad):
		g1 = None
		if t1.requires_grad:
			g1 = grad * mask
			t1.backward(g1)

	v.backward = backward
	return v

def sigmoid(t1):
	v = 1 / (1 + np.exp(-t1))
	v.requires_grad = True

	def backward(grad):
		g1 = None
		if t1.requires_grad:
			g1 = v * (1 - v) * grad
			t1.backward(g1)

	v.backward = backward
	return v

def tanh(t1):
	v = np.tanh(t1)
	v.requires_grad = True

	def backward(grad):
		g1 = None
		if t1.requires_grad:
			g1 = (1 - v**2) * grad
			t1.backward(g1)

	v.backward = backward
	return v

def dropout(t1, drop_prob):
	keep_prob = 1.0 - drop_prob
	# scaled by 1/keep_prob
	mask = np.random.binomial(1, keep_prob, size=t1.shape) / keep_prob
	v = t1 * mask
	v.requires_grad = True

	def backward(grad):
		g1 = None
		if t1.requires_grad:
			g1 = mask * grad
			t1.backward(g1)

	v.backward = backward
	return v

def sparse_softmax_cross_entropy_with_logits(labels, logits):
	l_softmax = softmax(logits) + np.finfo(logits.dtype).eps
	nll = -np.mean(np.log(l_softmax[range(len(labels)), labels]))
	nll.requires_grad = True

	def backward():
		l_softmax[range(len(labels)), labels] -= 1
		logits.backward(l_softmax)
	nll.backward = backward
	return nll

def l2_loss(t1, rate):
	v = np.mean(t1**2)
	v.requires_grad = True

	def backward():
		g1 = 2*t1*rate
		t1.backward(g1)
	v.backward = backward
	return v

def l1_loss(t1, rate):
	v = np.mean(np.abs(t1))
	v.requires_grad = True

	mask = (t1 > 0).astype(np.int32)
	def backward():
		g1 = mask * 2 - 1
		t1.backward(g1)
	v.backward = backward
	return v

class GradientDescentOptimizer(object):
	"""docstring for GradientDescentOptimizer"""
	def __init__(self, var_list, lr):
		super(GradientDescentOptimizer, self).__init__()
		self.var_list = var_list
		self.lr = lr

	def zero_grad(self):
		for v in self.var_list:
			v.grad = np.zeros(v.shape, dtype=np.float32)

	def step(self):
		for v in self.var_list:
			v -= v.grad * self.lr

class MomentumOptimizer(GradientDescentOptimizer):
	"""docstring for MomentumOptimizer"""
	def __init__(self, var_list, lr, momentum=0.9, nesterov=False):
		super(MomentumOptimizer, self).__init__(var_list, lr)
		self.momentum = momentum
		self.v = None
		self.nesterov = nesterov

	def step(self):
		this_grad = [v.grad*self.lr for v in self.var_list]
		if self.v is None:
			new_v = this_grad
		else:
			new_v = [self.momentum * v + g for v, g in zip(self.v, this_grad)]
		self.v = new_v

		for var, v in zip(self.var_list, new_v):
			var -= v

		if self.nesterov:
			for var, v in zip(self.var_list, new_v):
				var -= self.momentum * v

class RMSPropOptimizer(GradientDescentOptimizer):
	"""docstring for RMSPropOptimizer"""
	def __init__(self, var_list, lr, decay_rate=0.9):
		super(RMSPropOptimizer, self).__init__(var_list, lr)
		self.decay_rate = decay_rate
		self.r = None

	def step(self):
		this_r = [v.grad**2 * (1-self.decay_rate) for v in self.var_list]
		if self.r is None:
			new_r = this_r
		else:
			new_r = [self.decay_rate * old_r + new_r for old_r, new_r in zip(self.r, this_r)]
		self.r = new_r

		for var, r in zip(self.var_list, new_r):
			var -= (self.lr / np.sqrt(1e-6 + r)) * var.grad


class AdamOptimizer(GradientDescentOptimizer):
	def __init__(self, var_list, lr, p1=0.9, p2=0.999):
		super(AdamOptimizer, self).__init__(var_list, lr)
		self.p1 = p1
		self.p2 = p2
		self.t = 1
		self.s = None
		self.r = None

	def step(self):
		this_s = [v.grad * (1-self.p1) for v in self.var_list]
		this_r = [v.grad**2 * (1-self.p2) for v in self.var_list]
		if self.s is None:
			new_s = this_s
		else:
			new_s = [self.p1 * old_s + new_s for old_s, new_s in zip(self.s, this_s)]
		if self.r is None:
			new_r = this_r
		else:
			new_r = [self.p2 * old_r + new_r for old_r, new_r in zip(self.r, this_r)]
		self.s = new_s
		self.r = new_r

		new_s = np.asarray(new_s) / (1-self.p1**self.t)
		new_r = np.asarray(new_r) / (1-self.p2**self.t)
		new_coef = [s/(np.sqrt(r) + 1e-8) for s, r in zip(new_s, new_r)]

		for var, coef in zip(self.var_list, new_coef):
			var -= self.lr * coef

		self.t += 1
