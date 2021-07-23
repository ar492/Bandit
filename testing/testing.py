
import numpy as np
import matplotlib.pyplot as plt
import random
import math

k=10 # number of actions/arms
arms=[]
horizon=500 # number of trials
max_delay=horizon
pending_pulls=[]
global_time=1
fig, axs = plt.subplots(3)

class Pull:
	reward=0
	time=0 # when does the reward reach the learner
	arm_index=0
	def __init__(self, r, t, i):
		self.reward=r
		self.time=t
		self.arm_index=i

class avgR:
	reward=0
	add=0
	def __init__(self, r, a):
		self.reward=r
		self.add=a
class Arm:
	avg_reward=0
	rewards=0
	trials=0
	def __init__(self): # normal distribution
		self.mu=random.uniform(1, 2)
		self.sigma=1
		self.delay_mu=random.uniform(0, max_delay)
		self.delay_sigma=1 # what should this be
	def update(self, i):
		self.rewards+=pending_pulls[i].reward
		# self.trials+=1 # not doing this because i already did it in "pull"
		self.avg_reward=(self.rewards/self.trials)
	def pull_reward(self):
		return np.random.normal(self.mu, self.sigma, 1)[0] # return a random value from the distribution
	def pull_delay(self):
		return round(np.random.normal(self.delay_mu, self.delay_sigma, 1)[0]) # return a random value from the distribution
	def pull(self, i):
		r=self.pull_reward()
		d=self.pull_delay()
		self.trials+=1
		self.avg_reward=self.rewards/self.trials
		pending_pulls.append(Pull(r, global_time+d, i))

def UCB(delta, add): # delta is the confidence level
	global pending_pulls, global_time, arms # ensuring global references to these variables rather than local
	cumulative_rewards=[]
	r=0 # accumulated reward
	average_rewards=[]
	for i in range(horizon):
		index = np.argmax([a.avg_reward*(global_time-1) + math.sqrt(((2*math.log(1/delta))/(a.trials*(global_time-1)) if a.trials*(global_time-1) !=0 else np.inf))+add for a in arms])
		arms[index].pull(index)
		for j in range (len(pending_pulls)):
			if pending_pulls[j].time==global_time:
				arms[pending_pulls[j].arm_index].update(j)
				r+=pending_pulls[j].reward
		pending_pulls=[x for x in pending_pulls if x.time>global_time]
		average_rewards.append(r/(i+1))
		cumulative_rewards.append(r)
		global_time+=1
	pending_pulls.clear() # for the arms that didn't return feedback within the horizon
	return [cumulative_rewards, average_rewards]



for i in range(k):
	arms.append(Arm())


batches=1000

addrewards=[]
for add in range(10):
	# add=10*add
	print(str(add)+ " starting")
	cr_avg=[]
	ar_avg=[]
	for i in range(horizon):
		cr_avg.append(0)
		ar_avg.append(0);
	for i in range (batches):
		x=UCB(0.2, add)
		cr=x[0]
		ar=x[1]
		for j in range(horizon):
			cr_avg[j]+=cr[j]
			ar_avg[j]+=ar[j]
		global_time=1
		arms.clear()
		for v in range(k):
			arms.append(Arm())
		if (i%100==0):
			print("at batch ", i)
	for i in range(horizon):
		cr_avg[i]/=batches
		ar_avg[i]/=batches

	addrewards.append(avgR(ar_avg[-1], add))
	axs[0].plot(cr_avg, label="Delayed UCB: " + "$\delta$ = " + str(0.2) + ", add " + str(add) + " to variance ")
	axs[1].plot(ar_avg)


mx=0
bestadd=0
for i in range(len(addrewards)):
	if(addrewards[i].reward>mx):
		mx=addrewards[i].reward
		bestadd=addrewards[i].add

axs[2].plot([a.add for a in addrewards], [a.reward for a in addrewards])

print("best add is " + str(bestadd))

axs[0].set_xlabel('Time')
axs[0].set_ylabel('Cumulative Reward')
axs[1].set_ylabel('Running Average Reward')
axs[0].legend()
plt.savefig('simulation.pdf')
plt.show()
