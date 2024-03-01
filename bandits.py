from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import timeit
import time
import numpy as np

'''
An eGreedyBanditAlgorithim for learning whether it is better to click or not click
'''
def eGreedyBanditLearn(driver, interval, iterations, epsi):
    # establish basic elements
    num_actions = 2
    actions_taken = np.zeros(num_actions)
    action_values = np.zeros(num_actions)
    make_clip_bttn = driver.find_element(By.ID, "btnMakePaperclip")
    clips_element = driver.find_element(By.ID, "clips")
    clips = int(clips_element.text.replace(",", ""))
    it_pers_sec = int(1 / interval)

    for i in range(iterations):
        if (i % it_pers_sec) == 0:
            print("Estimate actions values:", action_values)
        action = eGreedyChooseAction(action_values, num_actions, actions_taken, epsi)
        if (i % it_pers_sec) == 0:
            print("Chosen action:", action)
        prev_return = clips
        if action == 1:
            driver.execute_script("arguments[0].click();", make_clip_bttn)
        time.sleep(interval)
        clips = int(clips_element.text.replace(",", ""))
        reward = clips - prev_return
        actions_taken[action] += 1
        action_values[action] += 1/actions_taken[action] * (reward - action_values[action])

    # print results
    print("Estimate actions values:", action_values)
    print("Ideal action taken:", clips / iterations)

def eGreedyChooseAction(values, actions, actions_taken, epsi):
    # pick any action which has not been chosen
    if np.any(actions_taken == 0):
        return np.where(actions_taken == 0)[0][0]
    
    # otherwise pick epsilon greedy
    luck = np.random.rand()
    if luck > epsi:
        return np.argmax(values) 
    else:
        return np.random.randint(0, actions)
       
'''
An ucbBanditAlgorithim for learning whether it is better to click or not click
'''
def ucbBanditLearn(driver, interval, iterations, drift):
    # establish basic elements
    num_actions = 2
    actions_taken = np.zeros(num_actions)
    action_values = np.zeros(num_actions)
    make_clip_bttn = driver.find_element(By.ID, "btnMakePaperclip")
    clips_element = driver.find_element(By.ID, "clips")
    clips = int(clips_element.text.replace(",", ""))
    it_pers_sec = int(1 / interval)

    for i in range(iterations):
        if (i % it_pers_sec) == 0:
            print("Estimate actions values:", action_values)
        action = ucbChooseAction(action_values, actions_taken, i, drift)
        if (i % it_pers_sec) == 0:
            print("Chosen action:", action)
        prev_return = clips
        if action == 1:
            driver.execute_script("arguments[0].click();", make_clip_bttn)
        time.sleep(interval)
        clips = int(clips_element.text.replace(",", ""))
        reward = clips - prev_return
        actions_taken[action] += 1
        action_values[action] += 1/actions_taken[action] * (reward - action_values[action])

    # print results
    print("Estimate actions values:", action_values)
    print("Ideal action taken:", clips / iterations)

def ucbChooseAction(action_values, actions_taken, timestep, drift):
    # pick action if it hasnt been picked yet
    if np.any(actions_taken == 0):
        return np.where(actions_taken == 0)[0][0]
    
    # otherwise pick based on upper confidence bound
    ucb_value = action_values + (drift * np.sqrt (np.log(timestep) / actions_taken))
    return np.argmax(ucb_value)