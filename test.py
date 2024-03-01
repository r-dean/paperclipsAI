from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import timeit
import time
import numpy as np
from bandits import eGreedyBanditLearn, ucbBanditLearn
from state import State
from linearFunctions import waitMakeBuySGSARSA
import scipy

def interact(driver, element, n):
    for i in range(n):
        try:
            driver.execute_script("arguments[0].click();", element)
        except Exception as e:
            print("exception:", e)

def start_browser():
    # path to web driver executable
    driver_path = './geckodriver.exe'

    driver = webdriver.Firefox()

    # Open the website
    driver.get("https://www.decisionproblem.com/paperclips/index2.html")

    return driver

def quit(driver):
    driver.quit()

def restart(driver):
    driver.quit()
    return start_browser()


def main():
    print(scipy.special.softmax(np.zeros(3)))
    print(scipy.special.softmax(np.arange(4)))

    driver = start_browser()
    
    # state = State(driver)
    # print(state)
    
    waitMakeBuySGSARSA(driver, interval=0.1, steps=4000, stepsize=0.01, gamma=0.5, epsi = 0.001)

    # ucbBanditLearn(driver, 0.01, 1000, 1)

    # state.updateState()
    # print(state)

    # quit(driver)


if __name__ == "__main__":
    main()



