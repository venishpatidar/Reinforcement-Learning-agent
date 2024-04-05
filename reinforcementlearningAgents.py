import game
import random,util,math
from pacman import GameState
from featureExtractors import *

random.seed(10)

class Policy:
    def __init__(self) -> None:
        # Parameter theta 
        self.weights = util.Counter()
        self.grad_weights = util.Counter()

        # record log for a episode
        self.rewards = []
        self.saved_features_t = []
        self.saved_probs_t = []
        self.saved_action_t = []

    def softmax(self,x:list[float])->list[float]:
        """
            Apply softmax activation function and
            returns proability distribution
            input: [a, b, c, d]
            output: [pr(a), pr(b), pr(c), pr(d)]
        """
        denominator = sum([math.exp(element) for element in x]) + 1e-12
        return [math.exp(element)/denominator for element in x]

    
    def forward(self, x:list[dict])->list[float]:
        """
            Returns proability distribution of taking action a at t given state at t 
            and parameters w
            Pi(a_t | s_t, w) = Pr(a_t | s_t, w) = softmax(w_0 + (w_1 * x_1) + (w_2 * x_2) ...., w_0 + (w_1 * x1)... )
        """
        h_w_x = [1e-6]*len(x)
        for i, s_a in enumerate(x):
            for feature in s_a:
                if self.weights[feature]:
                    h_w_x[i] = h_w_x[i] + self.weights[feature] * x[i][feature]
                else:
                    self.weights[feature] = random.uniform(-10, 10)
                    h_w_x[i] = h_w_x[i] + self.weights[feature] * x[i][feature]

        pi = self.softmax(h_w_x)
        return pi

    def grad(self,t:int)->None:
        """
                         
            grad(log_pi) = feature i of Action A - Sum[ pi(s,b) * feature i of b] 
            ∇ π_w(s,a) = fi(s,a) −∑ π_w(s,b) * fi(s,b)
        """
        feature_t = self.saved_features_t[t]
        action_probs_t = self.saved_probs_t[t]
        selected_s_a_feature = feature_t[self.saved_action_t[t]]
        sum_probs = util.Counter()
        for feature_s_b, action_probs_b in zip(feature_t,action_probs_t):
            for feature in feature_s_b:
                sum_probs[feature] = sum_probs[feature] + action_probs_b * feature_s_b[feature]
        for feature in selected_s_a_feature:
            self.grad_weights[feature] = selected_s_a_feature[feature] - sum_probs[feature]


    def optimize(self,lr:float,factor:float)->None:
        """
            Gradient Asscent
        """
        for feature in self.weights:
            self.weights[feature] = self.weights[feature] + (lr * factor * self.grad_weights[feature])





class ReinforceAgent(game.Agent):
    """
    REINFORCE ALGORITHM:
    Policy: a differential policy pi(a|s, theta)
    While True:
        Generate an episode by taking action guided by the policy
        For each time step in an episode:
            G <- returns from step t
            polic_loss = gamma^t * G * log_action_probs 
            # Optimization step
            theta <- theta + learing_rate * gradient_policy_loss
    """

    def __init__(self, numTraining=100, epsilon=0.8, alpha=2e-12, gamma=0.9):
        print("REINFORCE Agent initialized")
        self.numTraining = numTraining
        # Parameters:
        self.gamma = gamma #Discount factor
        self.alpha = alpha #Learing rate
        self.epsilon = epsilon
        self.policy = Policy()
        self.feature_extractor = SimpleExtractor()

        # episode track
        self.episode = 0
        self.episode_reward=0

        # Episode score track
        self.last_score = 0
        self.episode_record = ()

        self.last_action = None

    def final(self, state:GameState):
        if state.isWin() or state.isLose():
            reward = state.getScore()-self.last_score
            self.episode_reward += reward
            self.policy.rewards.append(reward)


        # Updating the trainable parameters
        self.train()

        print(f"Episode {self.episode} finished score: {self.episode_reward}")

        # Reseting epsiode state
        self.episode+=1
        self.episode_reward=0
        self.last_score = 0
        self.last_action = None


    def observationFunction(self, state:GameState)->GameState:
        """
            This is where we ended up after our last action.
            The simulation somehow enusers to call this step
          * This function should return state:gameState
        """
        self.observationStep(state)
        return state
    
    def observationStep(self, state:GameState):
        """
            Recording an episode
            Recording rewards and states after the last action had placed
        """

        if self.last_action:
            # Reward occured for last action
            reward = state.getScore()-self.last_score
            self.episode_reward += reward
            self.policy.rewards.append(reward)

        self.last_score = state.getScore()
        

    def train(self):
        """
            G <- [reward_t + discount factor * G] (return from step t)
            G_t = R_t+1 + gammaR_t+2 + gamma^2R_t+3 + ... + gamma^(T-1)R_T
            policy_loss <- (discount factor ^ t) * G * action_log_prob
            theta <- theta + ( learning_rate * grad_policy_loss ) {gradient asscent}
        """
        returns = []
        R = 0
        # Normalizing and calculating the returns 
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            returns.append(R)
        returns = util.normalize(returns)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        for t, G in reversed(list(enumerate(returns))):
            policy_loss = ((self.gamma**t) * G )
            self.policy.grad(t)
            self.policy.optimize(self.alpha,policy_loss)

        # Reseting variables for new episode
        del self.policy.rewards[:]
        del self.policy.saved_action_t[:]
        del self.policy.saved_features_t[:]
        del self.policy.saved_probs_t[:]



    def getAction(self, state:GameState):
        legal_action = state.getLegalPacmanActions()
        features_s_a = [self.feature_extractor.getFeatures(state,action) for action in legal_action]
        
        action_probs = self.policy.forward(features_s_a)

        # Chossing action based on computed softmax action proabilites
        if self.numTraining<=self.episode:        
            action_index = action_probs.index(max(action_probs))
        else:
            action_index = random.choices(range(len(action_probs)), weights=action_probs)[0]
        
            # action_index = random.choices(
            #     [action_probs.index(max(action_probs)),random.choices(range(len(action_probs)), weights=action_probs)[0]],
            #     weights=[self.epsilon,1-self.epsilon]
            # )[0]


        action = legal_action[action_index]


        self.policy.saved_features_t.append(features_s_a)
        self.policy.saved_action_t.append(action_index)
        self.policy.saved_probs_t.append(action_probs)

        self.last_action = action
        return self.last_action




class Critic:
    def __init__(self) -> None:
        # Parameter omega 
        self.weights = util.Counter()
        self.grad_weights = util.Counter()

    def relu(self,x:float)->float:
        """
            Apply relu activation function 
            input: [-a, b, c, d]
            output: [0, b, c, d]
        """
        return max(0,x)
    
    def forward(self, x:dict)->float:
        """
            Returns value of state S at t
            and parameters w
            V(S_t, w) = relu(w_0 + (w_1 * x_1) + (w_2 * x_2) ....)
        """
        v_s_w = 1e-6
        for feature in x:
            if self.weights[feature]:
                v_s_w = v_s_w + self.weights[feature] * x[feature]
            else:
                self.weights[feature] = random.uniform(-10, 10)
                v_s_w = v_s_w + self.weights[feature] * x[feature]

        v = self.relu(v_s_w)
        return v

    def grad(self,last_state:dict)->None:
        for feature in last_state:
            # Relu gradient
            if self.weights[feature]*last_state[feature]>0:
                self.grad_weights[feature] = last_state[feature]
            else:
                self.grad_weights[feature] = 0

    def optimize(self,lr:float,factor:float)->None:
        for feature in self.weights:
            self.weights[feature] = self.weights[feature] + (lr * factor * self.grad_weights[feature])



class ActorCriticAgent(game.Agent):
    """
    ACTOR CRITIC ALGORITHM

    policy: a differentiable policy pi(a| s, theta)
    crtic: a differentiable value network v(S,w)

    For each episode:
    I <- 1
    While not end:
        A ~ Take action based on policy pi(a | s, theta)
        value = get value from critic network
        target = R + gamma * value(next_state) if next_state is terminal than = R
        delta = target - value

        critic_loss =  I * delta * value(state)
        policy_loss = I * delta * ln(pi(A | S, theta))

        update critic network weights
        update policy network weights
        
        I <- gamma * I
    """
    def __init__(self, numTraining=100, epsilon=0.8, alpha=1e-2, gamma=0.9):
        print("Actor Critic Agent initialized")

        self.numTraining = numTraining
        # Parameters:
        self.gamma = gamma #Discount factor
        self.alpha = alpha #Learing rate
        self.epsilon = epsilon
        self.policy = Policy()
        self.critic = Critic()
        self.feature_extractor = SimpleExtractor()

        self.I = 1
        # episode track
        self.episode = 0
        self.episode_reward=0

        # Episode score track
        self.last_score = 0
        self.episode_record = ()

        self.last_state = None
    
    def final(self, state:GameState):
        self.train(state)

        print(f"Episode {self.episode} finished score: {self.episode_reward}")
        self.episode+=1
        self.episode_reward=0
        self.last_score = 0
        self.last_state = None


    def observationFunction(self, state:GameState)->GameState:
        """
            This is where we ended up after our last action.
            The simulation somehow enusers to call this step
          * This function should return state:gameState
        """
        self.observationStep(state)
        return state
    
    def observationStep(self, state:GameState):
        if self.last_state:
            self.train(state)

    def train(self,state:GameState):

        R = state.getScore()-self.last_score
        self.episode_reward += R
        self.last_score = state.getScore()
        
        last_state_feature = self.feature_extractor.getFeatures(self.last_state)
        state_feature = self.feature_extractor.getFeatures(state)
    

        value = self.critic.forward(last_state_feature)
        if state.isWin() or state.isLose():
            target = R
        else:
            target = R + (self.gamma * self.critic.forward(state_feature))

        delta = target-value

        self.critic.grad(last_state_feature)
        self.critic.optimize(self.alpha,delta)

        self.policy.grad(0)
        self.policy.optimize(self.alpha,delta)

        self.I = self.gamma * self.I
        
        # Since the policy class is same as used in reinforce 
        # instead of storing in variable in this we are storing it in array
        # last element and deleting it just after use as actor crtic updates 
        # after every time step so no need to maintain record of these. 
        del self.policy.saved_action_t[:]
        del self.policy.saved_features_t[:]
        del self.policy.saved_probs_t[:]




    def getAction(self, state:GameState):
        """
            Taking action defined by proability distribution pi during training 
            and taking max action while training is done
        """
        legal_action = state.getLegalPacmanActions()
        features_s_a = [self.feature_extractor.getFeatures(state,action) for action in legal_action]
        
        action_probs = self.policy.forward(features_s_a)

        # Chossing action based on computed softmax action proabilites
        if self.numTraining<=self.episode:        
            action_index = action_probs.index(max(action_probs))
        else:
            action_index = random.choices(range(len(action_probs)), weights=action_probs)[0]
        

        action = legal_action[action_index]


        self.policy.saved_features_t.append(features_s_a)
        self.policy.saved_action_t.append(action_index)
        self.policy.saved_probs_t.append(action_probs)

        self.last_state = state
        return action


