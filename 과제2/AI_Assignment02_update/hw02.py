from util import manhattanDistance
from game import Directions
import random, util
import sys

from game import Agent

## Example Agent
class ReflexAgent(Agent):

  def Action(self, gameState):

    move_candidate = gameState.getLegalActions()

    scores = [self.reflex_agent_evaluationFunc(gameState, action) for action in move_candidate]
    bestScore = max(scores)
    Index = [index for index in range(len(scores)) if scores[index] == bestScore]
    get_index = random.choice(Index)

    return move_candidate[get_index]

  def reflex_agent_evaluationFunc(self, currentGameState, action):

    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()



def scoreEvalFunc(currentGameState):

  return currentGameState.getScore()

class AdversialSearchAgent(Agent):

  def __init__(self, getFunc ='scoreEvalFunc', depth ='2'):
    self.index = 0
    self.evaluationFunction = util.lookup(getFunc, globals())

    self.depth = int(depth)

######################################################################################

class MinimaxAgent(AdversialSearchAgent):
  """
    [문제 01] MiniMax의 Action을 구현하시오. (20점)
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """

  def Action(self, gameState):
    ####################### Write Your Code Here ################################
    def maximizer(gameState,depth,agentIndex=0):
      if gameState.isWin() or gameState.isLose() or depth == self.depth:
        return self.evaluationFunction(gameState),None
      
      v = -sys.maxsize; move = Directions.STOP
      move_candidate = gameState.getLegalActions(agentIndex)
      for a in move_candidate:
        if a == Directions.STOP:
          continue
        s = gameState.generateSuccessor(0,a)
        v2,_ = minimizer(s,depth,1)
        if v2 >= v:
          v = v2
          move = a
      
      return v,move

    def minimizer(gameState,depth,agentIndex):
      if gameState.isWin() or gameState.isLose() or depth == self.depth:
        return self.evaluationFunction(gameState),None
      
      v = sys.maxsize; move = Directions.STOP
      move_candidate = gameState.getLegalActions(agentIndex)
      
      for a in move_candidate:
        if a == Directions.STOP:
          continue
        s =gameState.generateSuccessor(agentIndex,a)
        if gameState.getNumAgents() == agentIndex+1:
          v2,_ = maximizer(s,depth+1)
        else:
          v2,_ = minimizer(s,depth,agentIndex+1)
        
        if v2 <= v:
          v = v2
          move = a

      return v,move

    move_candidate = gameState.getLegalActions()
    move_like_jagger = Directions.STOP
    v = -sys.maxsize
    value, move_like_jagger = maximizer(gameState,0)
#    print(value)  #************value 를 찍으면 맨 처음 나오는 값이 initial value 이다**************************
    return move_like_jagger
    
    ##############
    # ##############################################################


class AlphaBetaAgent(AdversialSearchAgent):
  """
    [문제 02] AlphaBeta의 Action을 구현하시오. (25점)
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """
  def Action(self, gameState):
    ####################### Write Your Code Here ################################

    def maximizer(gameState,depth,alpha,beta,agentIndex=0):
      if gameState.isWin() or gameState.isLose() or depth == self.depth:
        return self.evaluationFunction(gameState), None
      
      v = -sys.maxsize; move = Directions.STOP
      move_candidate = gameState.getLegalActions(agentIndex)
      for a in move_candidate:
        if a == Directions.STOP:
          continue
        s = gameState.generateSuccessor(0,a)
        v2,_ = minimizer(s,depth,1,alpha,beta)
        if v2 > v:
          v = v2
          move = a
          alpha = max(alpha,v)
        if v >= beta:
          return v, move

      return v,move

    def minimizer(gameState,depth,agentIndex,alpha,beta):
      if gameState.isWin() or gameState.isLose() or depth == self.depth:
        return self.evaluationFunction(gameState),None
      
      v = sys.maxsize; move = Directions.STOP
      move_candidate = gameState.getLegalActions(agentIndex)
      
      for a in move_candidate:
        if a == Directions.STOP:
          continue
        s =gameState.generateSuccessor(agentIndex,a)
        
        if gameState.getNumAgents() == agentIndex+1:
          v2,_ = maximizer(s,depth+1,alpha,beta)
        else:
          v2,_ = minimizer(s,depth,agentIndex+1,alpha,beta)
        
        if v2 < v:
          v = v2
          move = a
          beta = min(beta,v)
        
        if v <= alpha:
          return v,move
        
      return v,move

    move_candidate = gameState.getLegalActions()
    move_like_jagger = Directions.STOP
    v = -sys.maxsize
    alpha = -sys.maxsize
    beta = sys.maxsize
    value, move_like_jagger = maximizer(gameState,0,alpha,beta)
#    print(value) #*********************value 를 찍으면 맨 처음 나오는 값이 initial value 이다**************************
    return move_like_jagger

    ############################################################################



class ExpectimaxAgent(AdversialSearchAgent):
  """
    [문제 03] Expectimax의 Action을 구현하시오. (25점)
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """
  def Action(self, gameState):
    ####################### Write Your Code Here ################################
    def maximizer(gameState,depth,agentIndex=0):
      if gameState.isWin() or gameState.isLose() or depth == self.depth:
        return self.evaluationFunction(gameState)
      
      v = -sys.maxsize
      move_candidate = gameState.getLegalActions(agentIndex)
      for a in move_candidate:
        if a == Directions.STOP:
          continue
        s = gameState.generateSuccessor(0,a)
        v = max(v,expectimizer(s,depth,1))

      return v

    def minimizer(gameState,depth,agentIndex):
      if gameState.isWin() or gameState.isLose() or depth == self.depth:
        return self.evaluationFunction(gameState)
      
      v = sys.maxsize
      move_candidate = gameState.getLegalActions(agentIndex)
      
      for a in move_candidate:
        if a == Directions.STOP:
          continue
        s =gameState.generateSuccessor(agentIndex,a)
        if gameState.getNumAgents() == agentIndex+1:
          v = min(v, expectimizer(s,depth+1,0))
        else:
          v = min(v, minimizer(s,depth,agentIndex+1))
      return v

    def expectimizer(gameState,depth,agentIndex):
      if gameState.isWin() or gameState.isLose() or depth == self.depth:
        return self.evaluationFunction(gameState)
      
      v = 0
      move_candidate = gameState.getLegalActions(agentIndex)
      for a in move_candidate:
        if a == Directions.STOP:
          continue
        s =gameState.generateSuccessor(agentIndex,a)
        if agentIndex == 0:
          v += maximizer(s,depth+1) / len(move_candidate)
        else:
          v += minimizer(s,depth,agentIndex+1) / len(move_candidate)
      return v

    move_candidate = gameState.getLegalActions()
    move_like_jagger = Directions.STOP
    v = -sys.maxsize
    for a in move_candidate:
      s = gameState.generateSuccessor(0,a)
      move_value = expectimizer(s,0,1)
      
      if move_value > v:
        v = move_value
        move_like_jagger = a
    
    return move_like_jagger
    ############################################################################
