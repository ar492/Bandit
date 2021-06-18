import numpy as np
import matplotlib.pyplot as plt
import random
import math

class Arm:
	rewards=0
	trials=0
	avg_reward=0
	def __init__(self): # normal distribution
		self.mu=random.uniform(1, 2)
		self.sigma=1
	def pull(self):
		return np.random.normal(self.mu, self.sigma, 1)[0] # return a random value from the distribution
	def update(self):
		r=self.pull()
		self.rewards+=r
		self.trials+=1
		self.avg_reward=(self.rewards/self.trials)
		return r

k=10 # number of actions/arms
arms=[]
size=10 # batch size
horizon=50000 # number of trials

fig, axs = plt.subplots(2)

def epsilon_greedy(epsilon):
	arms_copy=arms.copy() # save the original reward/talent initialization for future algorithms
	average_rewards=[]; cumulative_rewards=[];
	r=0 # accumulated reward
	for i in range(horizon):
		if random.uniform(0, 1)>epsilon: # probability 1-eps
			j = np.argmax([a.avg_reward for a in arms_copy]) #exploit
		else:  # probability eps
			j = np.random.choice(k) # explore
		r+=arms_copy[j].update()
		if((i+1)%size==0):
			average_rewards.append(r/(i+1))
			cumulative_rewards.append(r)
	axs[1].plot(average_rewards)
	axs[0].plot(cumulative_rewards, label="$\epsilon$ = " + str(epsilon) + " greedy")

def explore_then_commit(m): # explore for mk rounds (each action m times)
	arms_copy=arms.copy()
	r=0 # accumulated reward
	average_rewards=[]; cumulative_rewards=[];
	itr=0
	for i in range(m):
		for j in range (k):
			itr+=1
			r+=arms_copy[j].update()
			if itr%size==0:
				average_rewards.append(r/itr)
				cumulative_rewards.append(r)
	best = np.argmax([a.avg_reward for a in arms_copy])
	for i in range (horizon-m*k):
		itr+=1
		r+=arms_copy[best].pull()
		if itr%size==0:
			average_rewards.append(r/itr)
			cumulative_rewards.append(r)
	axs[0].plot(cumulative_rewards, label="ETC: m = " + str(m))
	axs[1].plot(average_rewards)

def UCB(delta): # delta is the confidence level
	# be optimistic about the environment; favor exploration of arms with high uncertainty
	# sublinear regret
	# UCB formula is from hoeffding's inequality
	arms_copy=arms.copy()
	time=1
	cumulative_rewards=[]
	r=0 # accumulated reward
	average_rewards=[]
	batch_size=0
	for i in range(horizon):
		j = np.argmax([a.avg_reward*(time-1) + math.sqrt(((2*math.log(1/delta))/(a.trials*(time-1)) if a.trials*(time-1) !=0 else np.inf)) for a in arms_copy])
		r+=arms_copy[j].update()
		batch_size+=1
		if batch_size%size==0:
			average_rewards.append(r/(i+1))
			cumulative_rewards.append(r)
		time+=1
	axs[0].plot(cumulative_rewards, label="UCB: " + "$\delta$ = " + str(delta))
	axs[1].plot(average_rewards)

for i in range(k):
	arms.append(Arm())

#explore_then_commit(10)
#explore_then_commit(30)
#for i in range (3):
#	explore_then_commit(horizon//(k+(i+1)*50))


#UCB(0.1)
#UCB(0.3)
#UCB(0.5)
#UCB(0.7)
#UCB(0.9)

axs[0].set_xlabel('Batch #')
axs[0].set_ylabel('Cumulative Reward')
axs[1].set_ylabel('Running Average Reward')
axs[1].set_xlabel('Batch #')
#axs[0].legend(["Epsilon Greedy", "Explore-then-commit (ETC)", "UCB"], loc ="upper left")
axs[0].legend()
plt.show()
