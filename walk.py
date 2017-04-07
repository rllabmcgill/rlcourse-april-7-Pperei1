import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import linear_model

'''
class that defines the chain

'''
class Walk:
	def __init__(self,startingPos):
		self.pos = startingPos
	
	'''
	updates the position on the chain and returns the transition reward

	'''
	def step(self,action):
		self.pos = self.pos + action
		if self.pos <= 0:
			self.pos = 0
			return 0
		elif self.pos >= 20:
			self.pos = 20
			return 50
		else:
			return -2
	
	#the valid moves in the chain
	def getValidMoves(self):
		return [1,2,3]
	
	'''
	performs the rollout alogrithm to estimate the 
	value of the state-action pair
	'''		
	def rollout(self,state,action,policy):
		self.pos = state
		Qvalue = self.step(action)
		for k in range(0,50):
			while(self.pos != 20):
				Qvalue = Qvalue + self.step(policy[self.pos])
		return Qvalue/(100.0)
	
	'''
	given a policy, plays a game with this policy and returns the cumulative reward
	'''

	def playGame(self,policy):
		reward = 0
		self.pos = 0
		while(self.pos != 20):
			reward = reward + self.step(policy[self.pos])
		return reward
	
class APIPlayer:
	def __init__(self,policy):
		self.policy = policy
	
	'''
	uses the rollout algorithm to generate positive and negative examples for every state
	the data is then aggregated and used to train an svm
	the svm is then used to determine which action has the best value at every state in order to generate a new policy
	'''

	def updatePolicy(self,walk):
		training = []
		label = []
		policy = self.policy
		for state in range(0,20):
			data = []
			for action in walk.getValidMoves():
				data.append([action,walk.rollout(state,action,self.policy)])
			bS,bA = self.bestAction(data)
			policy[state]=bA
			v = self.onehotencoding(state)
			v.append(bA)
			training.append(v)
			label.append(1)
			for a in walk.getValidMoves():
				if a != bA:
					v = self.onehotencoding(state)
					v.append(a)
					training.append(v)
					label.append(0)
		clf = svm.SVC()
		clf.fit(training,label)
		for state in range(0,20):
			for action in walk.getValidMoves():
				v = self.onehotencoding(state)
				v.append(action)
				pred = clf.predict([v])
				if pred[0] >= 0.5:
					policy[state] = action
					break
		self.policy = policy
	
	'''
	finds the action with the best value in a 2D array
	'''	
	def bestAction(self,data):
		bestAction = data[0][0]
		bestScore = data[0][1]
		for i in range(1,len(data)):
			if data[i][1] > bestScore:
				bestAction = data[i][0]
				bestScore = data[i][1]
		return bestScore,bestAction
	

	def onehotencoding(self,state):
		a = [0]*20
		a[state] = 1
		return a
for _ in range(0,10):
	rp = APIPlayer(np.random.randint(3,size=20)+1)
	w = Walk(0)
	cumReward = []
	for i in range(0,50):
		rp.updatePolicy(w)
		cumReward.append(w.playGame(rp.policy))
	plt.plot(cumReward)
plt.ylabel('cumulative reward')
plt.xlabel('iterative step')
plt.show()

