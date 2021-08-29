

import numpy as np
import matplotlib.pyplot as plt
import random
import math

k=10 # number of actions (arms)
arms=[]
horizon=1500 # number of pulls
delay=20
pending_pulls=[] # pulls for which feedback has not been recieved
global_time=1
fig, axs = plt.subplots()

class Pull:
	reward=0.1
	reach_time=0 # when does the reward reach the learner
	arm_index=0
	def __init__(self, r, t, i):
		self.reward=r
		self.reach_time=t
		self.arm_index=i

class Arm:
	rewards=0.1
	pulls=0
	reward_pulls=0
	def __init__(self): # normal distribution
		self.mu=random.uniform(0, 1)
		self.p=random.uniform(0, 1) # probability of 1. otherwise, 0
	def reset(self):
		self.rewards=0.1
		self.pulls=0 # all pulls, regardless of feedback
		self.reward_pulls=0 # number of pulls for which feedback has been received
	def update(self, i):
		self.rewards+=pending_pulls[i].reward
		self.reward_pulls+=1
	def pull_reward(self):
		return (random.uniform(0, 1)<self.p)
	def pull_delay(self):
		#return round(random.uniform(1, delay))
		return delay
	def pull(self, i):
		r=self.pull_reward()
		d=self.pull_delay()
		self.pulls+=1
		pending_pulls.append(Pull(r, global_time+d, i))

def reset_between_algorithms():
	global global_time, pending_pulls, arms
	global_time=1
	pending_pulls.clear()
	for i in range(k):
		arms[i].reset()

def naiveUCB():
	global pending_pulls, global_time, arms # ensuring global references to these variables rather than local
	cumulative_rewards=[]
	r=0 # accumulated reward
	reset_between_algorithms()
	for i in range(horizon):
		index = np.argmax([ np.inf if a.reward_pulls==0 else ((a.rewards/a.reward_pulls + math.sqrt(math.log(global_time)/a.reward_pulls))) for a in arms])
		arms[index].pull(index)
		for j in range (len(pending_pulls)):
			if pending_pulls[j].reach_time==global_time:
				arms[pending_pulls[j].arm_index].update(j)
				r+=pending_pulls[j].reward
		pending_pulls=[x for x in pending_pulls if x.reach_time>global_time]
		cumulative_rewards.append(r)
		global_time+=1
	pending_pulls.clear() # for the pulls that didn't return feedback within the horizon
	return cumulative_rewards

def heuristicUCB1():
	global pending_pulls, global_time, arms # ensuring global references to these variables rather than local
	cumulative_rewards=[]
	r=0 # accumulated reward
	reset_between_algorithms()

	for i in range(horizon):
		index = np.argmax([ np.inf if a.pulls==0 else ((a.rewards/a.pulls + math.sqrt(math.log(global_time)/a.pulls))) for a in arms])
		arms[index].pull(index)
		for j in range (len(pending_pulls)):
			if pending_pulls[j].reach_time==global_time:
				arms[pending_pulls[j].arm_index].update(j)
				r+=pending_pulls[j].reward
		pending_pulls=[x for x in pending_pulls if x.reach_time>global_time]
		cumulative_rewards.append(r)
		global_time+=1
	pending_pulls.clear() # for the pulls that didn't return feedback within the horizon
	return cumulative_rewards

def heuristicUCB2():
	global pending_pulls, global_time, arms # ensuring global references to these variables rather than local
	cumulative_rewards=[]
	r=0 # accumulated reward
	reset_between_algorithms()

	for i in range(horizon):
		index = np.argmax([ np.inf if a.reward_pulls==0 else ((a.rewards/a.reward_pulls + math.sqrt(math.log(global_time)/a.pulls))) for a in arms])
		arms[index].pull(index)
		for j in range (len(pending_pulls)):
			if pending_pulls[j].reach_time==global_time:
				arms[pending_pulls[j].arm_index].update(j)
				r+=pending_pulls[j].reward
		pending_pulls=[x for x in pending_pulls if x.reach_time>global_time]
		cumulative_rewards.append(r)
		global_time+=1
	pending_pulls.clear() # for the pulls that didn't return feedback within the horizon
	return cumulative_rewards


def ourUCB(): # delta is the confidence level
	global pending_pulls, global_time, arms # ensuring global references to these variables rather than local
	cumulative_rewards=[]
	which_arm=[] # which_arm[i] is which arm is pulled at time i
	r=0 # accumulated reward
	m=delay
	reset_between_algorithms()

	for i in range(horizon):
		#sum_term=0
		#for x in range(int(max(0, global_time-m)), global_time-1): # for the sum term
		#	sum_term+=math.sqrt(1/arms[which_arm[x]].pulls)
	#	for a in arms:
	#		if a.pulls!=0:
	#			print(a.rewards, a.pulls)
	#			print(math.log(global_time)/(a.rewards/a.pulls))
	#			print(math.sqrt(math.log(global_time)/(a.rewards/a.pulls)))
		index = np.argmax([ np.inf if a.pulls==0 else (a.rewards/a.pulls + (math.sqrt(math.log(global_time)/(a.rewards/a.pulls))) + m/(a.rewards/a.pulls)) for a in arms])
		arms[index].pull(index)
		which_arm.append(index)
		for j in range (len(pending_pulls)):
			if pending_pulls[j].reach_time==global_time:
				arms[pending_pulls[j].arm_index].update(j)
				r+=pending_pulls[j].reward
		pending_pulls=[x for x in pending_pulls if x.reach_time>global_time]
		cumulative_rewards.append(r)
		global_time+=1
	pending_pulls.clear() # for the pulls that didn't return feedback within the horizon
	#print(cumulative_rewards)
	return cumulative_rewards


def optimal_strategy(): # based on the max cumulative reward at the end
	global pending_pulls, global_time, arms # ensuring global references to these variables rather than local
	reset_between_algorithms()
	index=0
	best_p=0
	for i in range(k):
		if(arms[i].p>best_p):
			index=i
			best_p=arms[i].p
		#if(arms[i].mu>best_p):
		#	index=i
		#	best_p=arms[i].mu
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
	return cumulative_rewards

for i in range(k):
	arms.append(Arm())

batches=200

regret_naive=[]
regret_our=[]
regret_heu1=[]
regret_heu2=[]

for i in range(horizon):
	regret_naive.append(0)
	regret_our.append(0)
	regret_heu1.append(0)
	regret_heu2.append(0)


for i in range (batches):
	cr_naive=naiveUCB()
	cr_our=ourUCB()
	cr_heu1=heuristicUCB1()
	cr_heu2=heuristicUCB2()
	best_cr=optimal_strategy()
	for j in range(horizon):
		regret_naive[j]+=(best_cr[j]-cr_naive[j])
		regret_our[j]+=(best_cr[j]-cr_our[j])
		regret_heu1[j]+=(best_cr[j]-cr_heu1[j])
		regret_heu2[j]+=(best_cr[j]-cr_heu2[j])
	arms.clear()
	for v in range(k):
		arms.append(Arm())
	if (i%10==0):
		print("at batch ", i)

print(cr_our)
print(best_cr)

for i in range(horizon):
	regret_naive[i]/=batches
	regret_our[i]/=batches
	regret_heu1[i]/=batches
	regret_heu2[i]/=batches

axs.plot(regret_naive, label="Naive UCB")
axs.plot(regret_our, label="Our UCB")
axs.plot(regret_heu1, label="Heuristic UCB 1")
axs.plot(regret_heu2, label="Heuristic UCB 2")
axs.set_ylabel('Regret')
axs.set_xlabel('Time')
axs.legend(fontsize="large")

plt.savefig('simulation.pdf')
plt.show()
