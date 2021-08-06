

import numpy as np
import matplotlib.pyplot as plt
import random
import math

k=10 # number of actions/arms
arms=[]
horizon=500 # number of trials
max_delay=5
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
	trials=0
	def __init__(self): # normal distribution
		self.mu=random.uniform(5, 6)
		self.sigma=1
		self.delay_mu=random.uniform(0, max_delay)
	def reset(self):
		self.avg_reward=0
		self.rewards=0
		self.trials=0
	def update(self, i):
		self.rewards+=pending_pulls[i].reward
		# self.trials+=1 # not doing this because its already done in "pull" since it doesn't matter whether feedback is received yet
		self.avg_reward=(self.rewards/self.trials)
	def pull_reward(self):
		return np.random.normal(self.mu, self.sigma, 1)[0] # return a random value from the distribution
	def pull_delay(self):
		return round(self.delay_mu) 
	def pull(self, i):
		r=self.pull_reward()
		d=self.pull_delay()
		self.trials+=1
		self.avg_reward=self.rewards/self.trials
		pending_pulls.append(Pull(r, global_time+d, i))

def reset():
	global global_time, pending_pulls
	global_time=1
	pending_pulls.clear()
	for i in range(k):
		arms[i].reset()

def naiveUCB():
	global pending_pulls, global_time, arms # ensuring global references to these variables rather than local
	cumulative_rewards=[]
	r=0 # accumulated reward
	average_rewards=[]
	reset()

	for i in range(horizon):
		index = np.argmax([ np.inf if a.rewards==0 else ((a.avg_reward + math.sqrt(math.log(global_time)/a.avg_reward))) for a in arms])
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
	reset()

	for i in range(horizon):
		sum_term=0
		for x in range(int(max(0, global_time-m)), global_time-1): # for the sum term
			sum_term+=arms[which_arm[x]].trials
		index = np.argmax([ np.inf if a.rewards==0 else a.avg_reward + ((math.sqrt(math.log(global_time))) + sum_term) * math.sqrt(1/a.trials) for a in arms])
		arms[index].pull(index)
		which_arm.append(index)
		for j in range (len(pending_pulls)):
			if pending_pulls[j].reach_time==global_time:
				arms[pending_pulls[j].arm_index].update(j)
				r+=pending_pulls[j].reward
				m=max(m, arms[pending_pulls[j].arm_index].delay_mu)
		pending_pulls=[x for x in pending_pulls if x.reach_time>global_time]
		average_rewards.append(r/(i+1))
		cumulative_rewards.append(r)
		global_time+=1
	pending_pulls.clear() # for the pulls that didn't return feedback within the horizon
	return [cumulative_rewards, average_rewards]

def optimal_strategy(): # based on the max cumulative reward at the end
	global pending_pulls, global_time, arms # ensuring global references to these variables rather than local
	best_cumulative_rewards=[]
	reset()
	
	for index in range (k):
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
	
		reset();

		if(len(best_cumulative_rewards)==0 or cumulative_rewards[-1]>best_cumulative_rewards[-1]):
			best_cumulative_rewards=cumulative_rewards.copy()

	return best_cumulative_rewards


for i in range(k):
	arms.append(Arm())


batches=200

# for naive UCB
cr_avg_naive=[] # average of cumulative rewards at each timestep
ar_avg_naive=[] # average of ( average reward thus far ) at each timestep
regret_naive=[]
# for our UCB
cr_avg_our=[]
ar_avg_our=[]
regret_our=[]

for i in range(horizon):
	cr_avg_naive.append(0)
	ar_avg_naive.append(0)
	regret_naive.append(0)

	cr_avg_our.append(0)
	ar_avg_our.append(0)
	regret_our.append(0)

for i in range (batches):
	our=ourUCB()
	naive=naiveUCB()
	
	cr_naive=naive[0]
	ar_naive=naive[1]
	cr_our=our[0]
	ar_our=our[1]
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
		
	global_time=1
	arms.clear()
	for v in range(k):
		arms.append(Arm())

	if (i%50==0):
		print("at batch ", i)


for i in range(horizon):
	cr_avg_naive[i]/=batches
	ar_avg_naive[i]/=batches
	regret_naive[i]/=batches

	cr_avg_our[i]/=batches
	ar_avg_our[i]/=batches
	regret_our[i]/=batches


axs[0].plot(cr_avg_naive, label="Naive UCB")
axs[0].plot(cr_avg_our, label="Our UCB")
axs[1].plot(ar_avg_naive)
axs[1].plot(ar_avg_our)

axs[0].set_xlabel('Time')
axs[0].set_ylabel('Cumulative Reward')
axs[1].set_ylabel('Running Average Reward')
axs[0].legend()

axs[2].plot(regret_naive)
axs[2].plot(regret_our)
axs[2].set_ylabel('Regret')

plt.savefig('simulation.pdf') # doesn't work
plt.show()
