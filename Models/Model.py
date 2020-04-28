from abc import ABC, abstractmethod

class Model(ABC):

	@abstractmethod
	def f(self, state, control, params, t):
		pass
	@abstractmethod
	def getForwardIntegration(self, state, control, params, t):
		pass
	@abstractmethod
	def getForwardIntegration(self):
		pass
	@abstractmethod		
	def getForwardIntegrationLambda(self, state, control, params, t):
		pass
