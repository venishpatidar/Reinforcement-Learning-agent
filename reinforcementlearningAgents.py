import random,util,math
from game import Agent
from featureExtractors import *

# Global parameters
NUM_EPS_UPDATE = 100

class ReinforcementLearingAgent(Agent):
    def __init__(self, filename=None, agent=None, numTraining=100, epsilon=0.05, alpha=2e-4, gamma=0.9):
        self.numTraining = numTraining
        # Parameters:
        self.filename = filename
        self.gamma = gamma #Discount factor
        self.alpha = alpha #Learing rate
        self.epsilon = epsilon
        self.feature_extractor = SimpleExtractor()
        self.agent = agent
        # episode track
        self.episode = 0

        # Train result track over all training
        self.total_accumulated_reward = 0.0
        # Result accumulated over a window 
        self.window_accumulated_reward = 0.0
        
        self.last_score = 0.0
        self.last_action = None
        self.last_state = None

    def results(self):
        """
            Displays results after the end of every number of episode NUM_EPS_UPDATE
            Saves the results to file 
        """
        if self.episode % NUM_EPS_UPDATE == 0:
            print(f'{self.agent} Learning Status:')
            window_avg = self.window_accumulated_reward / float(NUM_EPS_UPDATE)
            # In training
            if self.episode <= self.numTraining:
                train_avg = self.total_accumulated_reward / float(self.episode)
                print('\tCompleted %d out of %d training episodes' % (self.episode,self.numTraining))
                print('\tAverage Rewards over all training: %.2f' % (train_avg))

                if self.filename:
                    with open('./results/'+ self.agent+'/train_avg_'+self.filename,'a') as f:
                        f.write(str(train_avg)+'\n')
                    with open('./results/'+self.agent+'/train_window_avg_'+self.filename,'a') as f:
                        f.write(str(window_avg)+'\n')

                # Reseting for test accumulation
                if self.episode==self.numTraining:
                    self.total_accumulated_reward=0.0
            # In testing
            else:
                test_avg = self.total_accumulated_reward / float(self.episode-self.numTraining)
                print('\tCompleted %d test episodes' % (self.episode - self.numTraining))
                print('\tAverage Rewards over testing: %.2f' % test_avg)
                if self.filename:
                    with open('./results/'+self.agent+'/test_avg_'+self.filename,'a') as f:
                        f.write(str(test_avg)+'\n')
                    with open('./results/'+self.agent+'/test_window_avg_'+self.filename,'a') as f:
                        f.write(str(window_avg)+'\n')

            print('\tAverage Rewards for last %d episodes: %.2f' % (NUM_EPS_UPDATE,window_avg))
            self.window_accumulated_reward = 0.0


class Policy:
    def __init__(self):
        # Parameter theta 
        self.weights = util.Counter()
        self.grad_weights = util.Counter()

        # record log for a episode
        self.rewards = []
        self.saved_features_t = []
        self.saved_probs_t = []
        self.saved_action_t = []

    def softmax(self,x):
        """
            Apply softmax activation function and
            returns proability distribution
            input: [a, b, c, d]
            output: [pr(a), pr(b), pr(c), pr(d)]
        """
        denominator = sum([math.exp(element) for element in x]) + 1e-12
        return [math.exp(element)/denominator for element in x]

    
    def forward(self, x):
        """
            Returns proability distribution of taking action a at t given state at t 
            and parameters w
            Pi(a_t | s_t, w) = Pr(a_t | s_t, w) = softmax(w_0 + (w_1 * x_1) + (w_2 * x_2) ...., w_0 + (w_1 * x1)... )
        """
        h_w_x = [1e-12]*len(x)
        for i, s_a in enumerate(x):
            for feature in s_a:
                h_w_x[i] = h_w_x[i] + self.weights[feature] * x[i][feature]
                if self.weights[feature]:
                    h_w_x[i] = h_w_x[i] + self.weights[feature] * x[i][feature]
                else:
                    self.weights[feature] = random.uniform(-10, 10)
                    h_w_x[i] = h_w_x[i] + 1e-12 * x[i][feature]


        pi = self.softmax(h_w_x)
        return pi

    def grad(self,t):
        """
            grad(log_pi) = feature i of Action A - Sum[ pi(s,b) * feature i of b] 
            ∇ π_w(s,a) = fi(s,a) − ∑ π_w(s,b) * fi(s,b)
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


    def optimize(self,lr,factor):
        """
            Gradient Asscent
        """
        for feature in self.weights:
            self.weights[feature] = self.weights[feature] + (lr * factor * self.grad_weights[feature])


class ReinforceAgent(ReinforcementLearingAgent):
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

    def __init__(self, **args):
        args['agent']="ReinforceAgent"
        ReinforcementLearingAgent.__init__(self, **args)

        print("REINFORCE Agent initialized")
        self.policy = Policy()
       

    def final(self, state):
        if state.isWin() or state.isLose():
            reward = state.getScore()-self.last_score
            self.total_accumulated_reward += reward
            self.window_accumulated_reward += reward
            self.policy.rewards.append(reward)


        # Updating the trainable parameters
        self.train()

        self.episode+=1
        self.results()
        
        # Important Donot change
        self.last_score = 0.0
        self.last_action = None

        if self.episode == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg,'-' * len(msg)))


    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation somehow enusers to call this step
          * This function should return state:gameState
        """
        self.observationStep(state)
        return state
    
    def observationStep(self, state):
        """
            Recording an episode
            Recording rewards and states after the last action had placed
        """

        if self.last_action:
            # Reward occured for last action
            reward = state.getScore()-self.last_score
            self.total_accumulated_reward += reward
            self.window_accumulated_reward += reward
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


    def getAction(self, state):
        legal_action = state.getLegalPacmanActions()
        features_s_a = [self.feature_extractor.getFeatures(state,action) for action in legal_action]
        
        action_probs = self.policy.forward(features_s_a)

        # Chossing action based on computed softmax action proabilites
        if self.numTraining<=self.episode:        
            action_index = action_probs.index(max(action_probs))
        else:
            # Epsilon greedy: with proability epsilon select random action
            action_index = random.choices(
                [action_probs.index(max(action_probs)),random.choices(range(len(action_probs)), weights=action_probs)[0]],
                weights=[1-self.epsilon,self.epsilon]
             )[0]

        action = legal_action[action_index]

        self.policy.saved_features_t.append(features_s_a)
        self.policy.saved_action_t.append(action_index)
        self.policy.saved_probs_t.append(action_probs)

        self.last_action = action
        return self.last_action

class Critic:
    def __init__(self):
        # Parameter omega 
        self.weights = util.Counter()
        self.grad_weights = util.Counter()

    def relu(self,x):
        """
            Apply relu activation function 
            input: [-a, b, c, d]
            output: [0, b, c, d]
        """
        return max(0,x)
    
    def forward(self, x):
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
                v_s_w = v_s_w + 1e-12 * x[feature]

        v = self.relu(v_s_w)
        return v

    def grad(self,last_state):
        for feature in last_state:
            # Relu gradient
            if self.weights[feature]*last_state[feature]>0:
                self.grad_weights[feature] = last_state[feature]
            else:
                self.grad_weights[feature] = 0

    def optimize(self,lr:float,factor:float)->None:
        for feature in self.weights:
            self.weights[feature] = self.weights[feature] + (lr * factor * self.grad_weights[feature])


class ActorCriticAgent(ReinforcementLearingAgent):
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
    def __init__(self, **args):
        args['agent']="ActorCriticAgent"
        ReinforcementLearingAgent.__init__(self, **args)

        print("Actor Critic Agent initialized")
        self.policy = Policy()
        self.critic = Critic()
        self.I = 1.0
    
    def final(self, state):
        self.train(state)
        self.episode+=1


        self.results()

        # Important Donot change
        self.last_score = 0.0
        self.last_state = None

        if self.episode == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg,'-' * len(msg)))


    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation somehow enusers to call this step
          * This function should return state:gameState
        """
        self.observationStep(state)
        return state
    
    def observationStep(self, state):
        if self.last_state:
            self.train(state)

    def train(self,state):

        R = state.getScore()-self.last_score
        self.total_accumulated_reward += R
        self.window_accumulated_reward += R
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


    def getAction(self, state):
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
            # Epsilon greedy: with proability epsilon select random action
            action_index = random.choices(
                [action_probs.index(max(action_probs)),random.choices(range(len(action_probs)), weights=action_probs)[0]],
                weights=[1-self.epsilon,self.epsilon]
             )[0]
            
        action = legal_action[action_index]

        self.policy.saved_features_t.append(features_s_a)
        self.policy.saved_action_t.append(action_index)
        self.policy.saved_probs_t.append(action_probs)

        self.last_state = state
        return action


