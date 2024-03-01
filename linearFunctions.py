from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import timeit
import time
import numpy as np
from state import State
from actions import ActionInterface 
import scipy
from sklearn.preprocessing import StandardScaler

'''An algorithim that learns how to choose 
whether to make paperclips, buy wire, or wait'''
def waitMakeBuySGSARSA(driver, interval, steps, stepsize, gamma, epsi):
    # intialize
    actions = 3
    actions_taken = np.zeros(actions)
    num_features = 8
    weights = np.zeros((actions, num_features))
    state = State(driver)
    action_interface = ActionInterface(driver)
    sa_features = state.getThreeActionVect()
    action = chooseAction(sa_features, weights, actions, epsi, actions_taken, 0)
    # get next action
    actions_taken[action] += 1
    # action = chooseAction(sa_features, weights, actions, epsi)
    its_per_sec = int(1/interval)
    print(its_per_sec)

    start = None
    for i in range(steps):
        if (i % 100) == 0:
            if start != None:
                end = time.time()
                print((end - start)/100, "seconds per iteration.\n")
            start = time.time()
        if (i % its_per_sec) == 0:
            print(weights)
        if (i % its_per_sec == 0) or (action != last_action):
            print("action", action)    
            print("expected return:", valueFunction(sa_features, action, weights))
        if action == 1:
            action_interface.makePaperClip()
        elif action == 2:
            action_interface.buyWire()
        # wait for transition
        time.sleep(interval)
        # get next state and reward
        last_features = sa_features
        reward = state.updateAndCalculateReward()
        sa_features = state.getThreeActionVect()
        # get next action
        last_action = action
        action = chooseAction(sa_features, weights, actions, epsi, actions_taken, i)
        actions_taken[action] += 1
        if (i % its_per_sec == 0) or (action != last_action):
            print("reward:", reward)
            print("next sate expected reward:", valueFunction(sa_features, action, weights))
        # update weights
        weights[last_action] += update(weights, reward, last_features, sa_features, last_action, action, stepsize, gamma)
    
    
    printWeights(weights)
    print(f"strategy was {100 * state.getAccumulatedReturn() / (steps - (steps // 1000))}% optimal")
    print("quitting!\n")
    return

def printWeights(weights):
    print("Final weights:\n")
    print(weights)

def valueFunction(state, action, weights):
    return np.dot(weights[action], state[action])

def gradFunction(state,action):
    return state[action]

def chooseAction(sa_features, weights, actions, epsi, actions_taken, step):
    # pick action if it hasnt been picked yet
    if np.any(actions_taken == 0):
        return np.where(actions_taken == 0)[0][0]
    

    values = np.zeros(actions)
    for action in range(actions):
        values[action] = valueFunction(sa_features, action, weights) + 0.1 * (np.log(step) / actions_taken[action])    
    return np.argmax(values)
    

    luck = np.random.rand()
    if luck > epsi:
        values = np.zeros(actions)
        for action in range(actions):
            values[action] = valueFunction(sa_features, action, weights) + (np.log(step) / actions_taken[action])    
        return np.argmax(values)
    else:
        return np.random.randint(0,actions)
    # values = np.matmul(sa_features, weights.T)
    # implied_prob = scipy.special.softmax(values)
    # cum_prob = np.cumsum(implied_prob)
    # luck = np.random.rand()
    # return np.searchsorted(cum_prob, luck)


def update(weights, reward, state, next_state, action, next_action, stepsize, gamma):
    expected_future = valueFunction(next_state, next_action, weights)
    expected = valueFunction(state, action, weights)
    grad  = gradFunction(state, action)
    return stepsize * (reward + (gamma * expected_future) - expected) * grad 
    