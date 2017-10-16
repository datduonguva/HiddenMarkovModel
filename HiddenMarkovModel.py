import numpy as np 
import random
import matplotlib.pyplot as plt
random.seed(0)
class HiddenMarkov():
	def __init__(self, n_hidden = 2, n_visible = 3):
		self.n_hidden = n_hidden #this is because the first and last states are non-emitting 
		self.n_visible = n_visible
		#probability of transition from state i to state j. Prohibited transitions have probability 0
		self.transition_prob_matrix = np.zeros((self.n_hidden+1, self.n_hidden+2))
		#State observation likelihood (n_hidden x n_visible)
		#The probability of hidden state i having visible state j
		self.state_obs_llh_matrix = np.zeros((self.n_hidden + 1, self.n_visible))
		self.learn = False
	#return both forward matrix and end state, probability
	def forward(self, obs_list):
		#initializa from starting state, which is a non-emitting state
		forward_maxtrix = np.zeros((self.n_hidden + 1, len(obs_list)+1))
		for j in range(1, self.n_hidden+1):
			forward_maxtrix[j,1] = self.transition_prob_matrix[0,j]*self.state_obs_llh_matrix[j, obs_list[0]]
		#calculate the probability matrix dynamically
		for t in range(2, len(obs_list)+1):
			for j in range(1, self.n_hidden+1):
				forward_maxtrix[j, t ] = 0;
				for i in range(1, self.n_hidden+1):
					forward_maxtrix[j, t] += forward_maxtrix[i, t-1]*self.transition_prob_matrix[i, j]
				forward_maxtrix[j, t] *= self.state_obs_llh_matrix[j, obs_list[t-1]]
		end_state_prob = 0
		#return the probability to move to the end state, which is also a non-emitting state
		for j in range(1, self.n_hidden+1):
			end_state_prob += forward_maxtrix[j, len(obs_list)]*self.transition_prob_matrix[j, self.n_hidden+1]
		return (forward_maxtrix, end_state_prob) 
	#this Viterbi algorithm decodes a visible list of states to corresponding hidden states
	#that maximizes the probability.
	def backward(self, obs_list):
		n_obs = len(obs_list)
		backward_matrix = np.zeros((self.n_hidden+1, n_obs+1))
		#initialization
		backward_matrix[1:self.n_hidden+1, n_obs] = self.transition_prob_matrix[1:self.n_hidden+1, self.n_hidden+1]
		#dynamically calculate all other element of the matrix:
		for t in range(n_obs-1, 0, -1):
			for j in range(1, self.n_hidden+1):
				for i in range(1, self.n_hidden+1):
					backward_matrix[j, t] += backward_matrix[i, t+1]*self.transition_prob_matrix[j, i]*self.state_obs_llh_matrix[i,obs_list[t]]

		begin_state_prob = 0
		for i in range(1, self.n_hidden+1):
			begin_state_prob += self.transition_prob_matrix[0, i]*backward_matrix[i, 1]\
				*self.state_obs_llh_matrix[i, obs_list[0]]
		return backward_matrix, begin_state_prob
	def Viterbi_Path(self, obs_list):
		viterbi = np.zeros((self.n_hidden+2, len(obs_list)+1))
		path = [0]
		#initializa from stating state
		max_prob = 0
		max_pos = 0
		for j in range(1, self.n_hidden+1):
			viterbi[j, 1] = self.transition_prob_matrix[0, j]*self.state_obs_llh_matrix[j, obs_list[0]]
			if viterbi[j, 1]> max_prob:
				max_prob = viterbi[j, 1]
				max_pos = j
		path.append(max_pos)
		#finding the path up
		for t in range(2, len(obs_list)+1):
			max_prob = 0
			max_pos = 0
			for j in range(1, self.n_hidden+1):
				viterbi[j, t] = max([viterbi[i, t-1]*self.transition_prob_matrix[i, j] for i in range(1, self.n_hidden+1)])
				viterbi[j, t] *= self.state_obs_llh_matrix[j,obs_list[t-1]]
				if viterbi[j, t]> max_prob:
					max_prob = viterbi[j, t]
					max_pos = j
			path.append(max_pos)
		return path
	def vectorize(self, obs_list):
		if not self.learn:
			self.learn =True
			self.min_value = min(obs_list)
			self.max_value = max(obs_list)
			self.bin_width = (self.max_value- self.min_value)/self.n_visible

		result = [int((obs_list[i]- self.min_value-np.exp(-30))/self.bin_width) for i in range(len(obs_list))]
		return result

	def reverse_trasform(self, vector):
		return [self.min_value + self.bin_width*(i+0.5) for i in vector]
	#give a list, predict what will come out next
	def predict(self, obs_list_input):
		obs_list = self.vectorize(obs_list_input)


		temp = []
		for j in range(self.n_visible):
			temp1 = obs_list[:]
			temp1.append(j)
			#print(temp1)
			dump, probability = self.forward(temp1)
			temp.append(probability)
			#print(probability)
		
		return np.argmax(np.array(temp))


	def fit(self, obs_list_input, n_iter = 150):
		obs_list = self.vectorize(obs_list_input)
		n_hidden = self.n_hidden
		n_visible = self.n_visible
		#probability of transition from state i to state j. Prohibited transitions have probability 0
		self.transition_prob_matrix = np.zeros((self.n_hidden+1, self.n_hidden+2))
		#State observation likelihood (n_hidden x n_visible)
		#The probability of hidden state i having visible state j
		self.state_obs_llh_matrix = np.zeros((self.n_hidden + 1, self.n_visible))

		#initialize the matrices:
		for i in range(n_hidden+1):
			if i == 0:
				for j in range(1, n_hidden+1):
					self.transition_prob_matrix[0, j] = 1.0/n_hidden+ (2*np.random.random()-1)/10.0/n_hidden
			else:
				for j in range(1, n_hidden+2):
					self.transition_prob_matrix[i, j] = 1.0/(n_hidden+1) + (2*np.random.random()-1)/10.0/n_hidden	
		self.state_obs_llh_matrix[1:, :] += 1/n_visible
		#print(self.transition_prob_matrix)

		#Iterations:
		arr = []
		for i_iter in range(n_iter):
			print("iteration: ", i_iter)
			forward_maxtrix, end_state_probability = self.forward(obs_list)
			backward_matrix, begin_state_prob = self.backward(obs_list)
			#print(end_state_probability)
			
			zeta = np.zeros((len(obs_list), self.n_hidden+1, self.n_hidden+1))
			for t in range(1, len(obs_list)):
				for i in range(1, self.n_hidden+1):
					for j in range(1, self.n_hidden+1):
						zeta[t, i, j] = forward_maxtrix[i, t]*self.transition_prob_matrix[i, j]
						zeta[t, i, j] *= backward_matrix[j, t+1]*self.state_obs_llh_matrix[j, obs_list[t]]/begin_state_prob
			gamma = np.zeros((len(obs_list)+1, self.n_hidden+1))
			for t in range(1, len(obs_list)+1):
				for j in range(1, self.n_hidden+1):
					gamma[t, j] = forward_maxtrix[j, t]*backward_matrix[j, t]/begin_state_prob

			for i in range(1, self.n_hidden+1):
				for j in range(1, self.n_hidden+1):
					self.transition_prob_matrix[i, j] = sum(zeta[:, i, j])/sum(sum(zeta[:, i, :]))
			
			for j in range(1, self.n_hidden+1):
				for k in range(self.n_visible):
					self.state_obs_llh_matrix[j, k] = sum([gamma[t, j] if obs_list[t-1] == k else 0 for t in range(1, len(obs_list)+1)])
					self.state_obs_llh_matrix[j, k] /= sum(gamma[:, j])
			
			arr.append([self.transition_prob_matrix[1,1], self.transition_prob_matrix[1, int(self.n_hidden/5)],
				self.transition_prob_matrix[2, int(2*self.n_hidden/5)], self.transition_prob_matrix[2, int(4*self.n_hidden/5)]])
		arr = np.array(arr)
		plt.figure()
		plt.title("Convergence of some parameters")
		plt.plot(arr[:, 0])
		plt.plot(arr[:, 1])
		plt.plot(arr[:, 2])
		plt.plot(arr[:, 3])
		plt.legend()

		

if __name__=='__main__':
	n_hidden = 12
	n_visible = 60
	hm = HiddenMarkov(n_hidden, n_visible)
	obs_list = np.genfromtxt('forex-eod.csv', delimiter=',')[:,2][:150]
	"""plt.figure()
	plt.plot(obs_list)
	plt.show()
"""
	#obs_list = hm.vectorize(obs_list)
	hm.fit(obs_list)
	#print(hm.transition_prob_matrix)
	given = []
	pred = []
	for i in range(len(obs_list)-25):
		given.append(obs_list[i+9])
		pred.append(hm.predict(obs_list[i: i+10]))
		print(i)
	pred = hm.reverse_trasform(pred)
	#print(hm.state_obs_llh_matrix[1])
	plt.figure()
	plt.title("HHM with "+ str(n_hidden) +" hidden states \n and " + str(n_visible)+ " visible states")
	plt.plot(given, label = 'data')
	plt.plot(pred, label = 'prediction')
	plt.legend()
	plt.show()