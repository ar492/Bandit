

import numpy as np
import matplotlib.pyplot as plt
import random
import math

k=10 # number of actions/arms
arms=[]
horizon=2500 # number of pulls
max_delay=10
pending_pulls=[]
global_time=1
fig, axs = plt.subplots(3)

class Pull:
	reward=0
	reach_time=0 # when does the reward reach the learner
	arm_index=0
	def __init__(self, r, t, i):
		self.reward=r
		self.reach_time=t
		self.arm_index=i

class Arm:
	avg_reward=0
	rewards=0
	pulls=0
	def __init__(self): # normal distribution
		self.mu=random.uniform(5, 6)
		self.sigma=1
		self.delay_mu=max_delay
	def reset(self):
		self.avg_reward=0
		self.rewards=0
		self.pulls=0
		self.reward_pulls=0 # number of pulls for which feedback has been received
	def update(self, i):
		self.rewards+=pending_pulls[i].reward
		self.reward_pulls+=1
		# self.pulls+=1 # not doing this because its already done in "pull" since it doesn't matter whether feedback is received yet
		self.reward_pulls+=1
		self.avg_reward=(self.rewards/self.pulls)
	def pull_reward(self):
		return np.random.normal(self.mu, self.sigma, 1)[0] # return a random value from the distribution
	def pull_delay(self):
		return round(np.random.normal(max_delay, 1, 1)[0]) # return a random value from the distribution
	#	return round(self.delay_mu)
	def pull(self, i):
		r=self.pull_reward()
		d=self.pull_delay()
		self.pulls+=1
		self.avg_reward=self.rewards/self.pulls
		pending_pulls.append(Pull(r, global_time+d, i))

def reset_everything():
	global global_time, pending_pulls, arms
	global_time=1
	pending_pulls.clear()
	for i in range(k):
		arms[i].reset()

def naiveUCB():
	global pending_pulls, global_time, arms # ensuring global references to these variables rather than local
	cumulative_rewards=[]
	r=0 # accumulated reward
	average_rewards=[]
	reset_everything()

	for i in range(horizon):
		index = np.argmax([ np.inf if a.rewards==0 else ((a.rewards/a.reward_pulls + math.sqrt(math.log(global_time)/a.reward_pulls))) for a in arms])
		arms[index].pull(index)
		#print("naive ucb pulled ", arms[index].mu)
		for j in range (len(pending_pulls)):
			if pending_pulls[j].reach_time==global_time:
				arms[pending_pulls[j].arm_index].update(j)
				r+=pending_pulls[j].reward
			#	print("adding ", pending_pulls[j].reward)
		pending_pulls=[x for x in pending_pulls if x.reach_time>global_time]
		average_rewards.append(r/(i+1))
		cumulative_rewards.append(r)
		global_time+=1
	pending_pulls.clear() # for the pulls that didn't return feedback within the horizon
	return [cumulative_rewards, average_rewards]

def heuristicUCB():
	global pending_pulls, global_time, arms # ensuring global references to these variables rather than local
	cumulative_rewards=[]
	r=0 # accumulated reward
	average_rewards=[]
	reset_everything()

	for i in range(horizon):
		index = np.argmax([ np.inf if a.rewards==0 else ((a.avg_reward + math.sqrt(math.log(global_time)/a.pulls))) for a in arms])
		arms[index].pull(index)
		for j in range (len(pending_pulls)):
			if pending_pulls[j].reach_time==global_time:
				arms[pending_pulls[j].arm_index].update(j)
				r+=pending_pulls[j].reward
		pending_pulls=[x for x in pending_pulls if x.reach_time>global_time]
		average_rewards.append(r/(i+1))
		cumulative_rewards.append(r)
		global_time+=1
	pending_pulls.clear() # for the pulls that didn't return feedback within the horizon
	return [cumulative_rewards, average_rewards]


def ourUCB(): # delta is the confidence level
	global pending_pulls, global_time, arms # ensuring global references to these variables rather than local
	cumulative_rewards=[]
	average_rewards=[]
	which_arm=[] # which_arm[i] is which arm is pulled at time i
	r=0 # accumulated reward
	m=0 # max delay noticed so far
	reset_everything()

	for i in range(horizon):
		sum_term=0
		for x in range(int(max(0, global_time-m)), global_time-1): # for the sum term
			sum_term+=math.sqrt(1/arms[which_arm[x]].pulls)
		index = np.argmax([ np.inf if a.rewards==0 else a.avg_reward + ((math.sqrt(math.log(global_time))) + sum_term) * math.sqrt(1/a.pulls) for a in arms])
		arms[index].pull(index)
		# print("our ucb pulled ", arms[index].mu)
		which_arm.append(index)
		for j in range (len(pending_pulls)):
			if pending_pulls[j].reach_time==global_time:
				arms[pending_pulls[j].arm_index].update(j)
				r+=pending_pulls[j].reward
				# m=max(m, arms[pending_pulls[j].arm_index].delay_mu)
		pending_pulls=[x for x in pending_pulls if x.reach_time>global_time]
		average_rewards.append(r/(i+1))
		cumulative_rewards.append(r)
		global_time+=1
	pending_pulls.clear() # for the pulls that didn't return feedback within the horizon
	return [cumulative_rewards, average_rewards]

def optimal_strategy(): # based on the max cumulative reward at the end
	global pending_pulls, global_time, arms # ensuring global references to these variables rather than local
	best_cumulative_rewards=[]
	reset_everything()
	
	index=0
	best_mu=0
	for i in range(k):
		if(arms[i].mu>best_mu):
			index=i
			best_mu=arms[i].mu 

	cumulative_rewards=[]
	r=0 # accumulated reward
	for i in range(horizon):
		arms[index].pull(index)
		for j in range (len(pending_pulls)): # check if there are any pending pulls for which the reward just arrived
			if pending_pulls[j].reach_time==global_time:
				arms[pending_pulls[j].arm_index].update(j)
				r+=pending_pulls[j].reward
		pending_pulls=[x for x in pending_pulls if x.reach_time>global_time]
		cumulative_rewards.append(r)
		global_time+=1
	reset_everything();
	return cumulative_rewards

for i in range(k):
	arms.append(Arm())


batches=500

# for naive UCB
cr_avg_naive=[] # average of cumulative rewards at each timestep
ar_avg_naive=[] # average of ( average reward thus far ) at each timestep
regret_naive=[]
# for our UCB
cr_avg_our=[]
ar_avg_our=[]
regret_our=[]
# for heuristic UCB
cr_avg_heu=[]
ar_avg_heu=[]
regret_heu=[]

for i in range(horizon):
	cr_avg_naive.append(0)
	ar_avg_naive.append(0)
	regret_naive.append(0)

	cr_avg_our.append(0)
	ar_avg_our.append(0)
	regret_our.append(0)

	cr_avg_heu.append(0)
	ar_avg_heu.append(0)
	regret_heu.append(0)

for i in range (batches):
	naive=naiveUCB()
	our=ourUCB()
	heu=heuristicUCB()
	
	cr_naive=naive[0]
	ar_naive=naive[1]

	cr_our=our[0]
	ar_our=our[1]

	cr_heu=heu[0]
	ar_heu=heu[1]

	best_cr=optimal_strategy()
	for j in range(horizon):
		# naive UCB
		cr_avg_naive[j]+=cr_naive[j]
		ar_avg_naive[j]+=ar_naive[j]
		regret_naive[j]+=(best_cr[j]-cr_naive[j])
		# our UCB
		cr_avg_our[j]+=cr_our[j]
		ar_avg_our[j]+=ar_our[j]
		regret_our[j]+=(best_cr[j]-cr_our[j])
		# heuristic UCB
		cr_avg_heu[j]+=cr_heu[j]
		ar_avg_heu[j]+=ar_heu[j]
		regret_heu[j]+=(best_cr[j]-cr_heu[j])

	global_time=1
	arms.clear()
	for v in range(k):
		arms.append(Arm())

	if (i%10==0):
		print("at batch ", i)


for i in range(horizon):
	cr_avg_naive[i]/=batches
	ar_avg_naive[i]/=batches
	regret_naive[i]/=batches

	cr_avg_our[i]/=batches
	ar_avg_our[i]/=batches
	regret_our[i]/=batches

	cr_avg_heu[i]/=batches
	ar_avg_heu[i]/=batches
	regret_heu[i]/=batches

axs[0].plot(cr_avg_naive, label="Naive UCB")
axs[0].plot(cr_avg_our, label="Heuristic UCB")
axs[0].plot(cr_avg_our, label="Our UCB")
axs[1].plot(ar_avg_naive)
axs[1].plot(ar_avg_heu)
axs[1].plot(ar_avg_our)

axs[0].set_xlabel('Time')
axs[0].set_ylabel('Cumulative Reward')
axs[1].set_ylabel('Running Average Reward')
axs[0].legend()

axs[2].plot(regret_naive)
axs[2].plot(regret_heu)
axs[2].plot(regret_our)
axs[2].set_ylabel('Regret')

plt.savefig('simulation2.pdf')
plt.show()
