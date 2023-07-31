import random
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import itertools
from os import path

PLAYERS_PER_GAME = 3
POPULATION_SIZE = 1000
NUMBER_OF_GAMES = 3
NUMBER_OF_GAMES_REPEAT = 100
NUMBER_OF_EPOCHS = 500
SECOND_ORDER_EPOCH = 100
COIN_AMOUNTS = [1,3,5]
NUMBER_OF_TURNS = 3
SMOOTHING_WINDOW = 10
eps = 0.2
gamma = 0.9
alpha = 0.5
lSpeed = 0.5

class State():
    def __init__(self, turn, coins):
        self.transitions = {}
        self.turn = turn
        self.coins = coins
        self.k = getStateString(turn, coins)

    def addTransition(self, a, s):
        if a not in self.transitions: self.transitions[a] = [s]
        else: self.transitions[a].append(s)

class Agent():
    def __init__(self, order, p, epsilon, states):
        self.order = order
        self.eps = epsilon
        self.p = p
        self.k = getStateString(0, [0]*PLAYERS_PER_GAME)
        self.c = []
        self.a_predict = [[0 for _ in range(PLAYERS_PER_GAME)] for _ in range(order)]
        for o in range(1, order+1):
            self.c.append([0] + [int(o == 99)] * (PLAYERS_PER_GAME-1))
        self.initializeQs(states)
        self.coins = [0 for _ in range(PLAYERS_PER_GAME)]

    def initializeQs(self, states):
        self.qValues = []
        for _ in range(PLAYERS_PER_GAME):
            values = {}
            for k, s in states.items():
                v = {}
                for a in s.transitions.keys(): v[a] = a
                values[k] = v
            self.qValues.append(values)

    def getOtherStates(self, states, s):
        otherStates = []
        for i in range(1, PLAYERS_PER_GAME):
            otherStates.append(states[getStateString(s.turn, list(s.coins[i:])+list(s.coins[:i]))])
        return otherStates
    
    def getPhi(self, b, q, states, s):
        s_other = self.getOtherStates(states, s)
        permutations = [p for p in itertools.product(COIN_AMOUNTS, repeat=PLAYERS_PER_GAME-1)]
        phi = [0]*len(COIN_AMOUNTS)
        for p in permutations:
            chance = np.prod([b[j][COIN_AMOUNTS.index(x)] for j, x in enumerate(p)])
            for i, a in enumerate(COIN_AMOUNTS):
                k2_all, r_all = makeMoves([s]+s_other, [a]+list(p), s.turn)
                if k2_all[0] not in ['win','tie','loss']:
                    qs = q[k2_all[0]]
                    r_all[0] += max(qs, key=qs.get)
                phi[i] += chance * (r_all[0])
        return np.array(phi)
    
    def integratePhi(self, phi, phi2, c):
        return [(1-c) * phi[i] + c * phi2[i] for i in range(len(phi))]
    
    def getBelief(self, a):
        b = []
        for x in COIN_AMOUNTS:
            if x == a: b.append(1 - self.eps)
            else: b.append(self.eps / (len(COIN_AMOUNTS)-1))
        return b
    
    def transformQC(self, q, c, p, order):
        c2 = []
        q2 = q[p:]+q[:p]
        for o in range(1,order+1):
            c2.append(c[o-1][p:]+c[o-1][:p])
        return q2, c2
    
    def qToList(self, q, s):
        return [q[s.k][x] for x in COIN_AMOUNTS]

    def tomRec(self, states, order, s, q, c, show=False):
        phi = self.qToList(q[0], s)
        if order == 0: return phi, None
        else:
            otherStates = self.getOtherStates(states, states[self.k])
            a_predict = [[0 for _ in range(PLAYERS_PER_GAME)] for _ in range(order)]
            a_predict[0][0] = COIN_AMOUNTS[np.argmax(phi)]
            b = [[] for _ in range(PLAYERS_PER_GAME-1)]
            for o in range(1, order+1):
                for i, otherState in enumerate(otherStates):
                    q2, c2 = self.transformQC(q, c, i+1, o-1)
                    phi2, _ = self.tomRec(states, o-1, otherState, q2, c2)
                    a_predict[o-1][i+1] = COIN_AMOUNTS[np.argmax(phi2)]
                    b[i].append(self.getBelief(a_predict[o-1][i+1]))

            permutations = [p for p in itertools.product(range(order), repeat=PLAYERS_PER_GAME-1)]
            for o in range(1, order+1):
                for p in permutations[:]:
                    if max(p) == o-1:
                        permutations.remove(p)
                        chance = np.prod([c[x][i+1] for i, x in enumerate(p)])
                        b2 = [b[i][x] for i, x in enumerate(p)]
                        phi2 = self.getPhi(b2, q[0], states, s)
                        phi = self.integratePhi(phi, phi2, chance)
                if o < order: a_predict[o][0] = COIN_AMOUNTS[np.argmax(phi)]
            return phi, a_predict

    def tom(self, states, show=False):
        phi, a_predict = self.tomRec(states, self.order, states[self.k], self.qValues, self.c, show=show)
        self.a_predict = a_predict
        return phi
    
    def learn(self, actions):
        for i, a in enumerate(actions[self.p:] + actions[:self.p]):
            found = False
            for o in range(self.order):
                if a == self.a_predict[o][i]:
                    if not found: self.c[o][i] = lSpeed + (1 - lSpeed) * self.c[o][i]
                    found = True
                else: self.c[o][i] = (1 - lSpeed) * self.c[o][i]

    def selectAction(self, states):
        phi = self.tom(states)
        a = np.argmax(phi)
        if random.uniform(0.0, 1.0) > self.eps: return COIN_AMOUNTS[a]
        else: return random.choice([x for x in COIN_AMOUNTS if x != a])

def getStateString(turn, coins):
    otherMax = max(coins[1:])
    if (NUMBER_OF_TURNS - turn) * COIN_AMOUNTS[-1] + otherMax < coins[0]:
        return 'win'
    elif turn == NUMBER_OF_TURNS:
        return [['win','tie'][coins[0]==otherMax],'loss'][coins[0]<otherMax]
    else:
        return f"{turn}-{'-'.join(str(x) for x in coins)}"

def getStates():
    startState = State(0, [0]*PLAYERS_PER_GAME)
    stateString = getStateString(0, [0]*PLAYERS_PER_GAME)
    states = {stateString:startState}
    openStates = [stateString]
    depth = 0
    dC = 0
    d = 0
    permutations = [p for p in itertools.product(COIN_AMOUNTS, repeat=PLAYERS_PER_GAME)]
    while(openStates):
        stateString = openStates.pop(0)
        s = states[stateString]
        for p in permutations:
            coins = checkBridges(p)
            newCoins = np.add(s.coins, coins)
            state2String = getStateString(depth+1, newCoins)
            s.addTransition(p[0], state2String)

            if state2String not in ['win','tie','loss']:
                if state2String not in states:
                    s2 = State(depth+1, newCoins)
                    states[state2String] = s2
                    openStates.append(state2String)

        if d == dC:
            depth += 1
            dC = len(openStates)
            d = 0
        d += 1
        if depth == NUMBER_OF_TURNS: break
    return states

def initializeState(s, states):
    s.rewards = {}
    for a in s.transitions.keys():
        s.rewards[a] = {}
        for k2 in s.transitions[a]:
            if k2 == 'loss': s.rewards[a][k2] = -max(COIN_AMOUNTS)
            elif k2 == 'tie': s.rewards[a][k2] = 0
            elif k2 == 'win': s.rewards[a][k2] = max(COIN_AMOUNTS)
            else: s.rewards[a][k2] = states[k2].coins[0] - max(states[k2].coins[1:]) - (s.coins[0] - max(s.coins[1:]))

def checkBridges(choices):
    c = Counter(choices)
    coins = []
    for choice in choices:
        if c[choice] == 1: coins.append(choice)
        else: coins.append(0)
    return coins

def makeMoves(s, a, turn):
    s2 = []
    r = []
    coins = []
    for i in range(PLAYERS_PER_GAME):
        coins.append(s[i].coins[0])
    coins = [x+y for x, y in zip(checkBridges(a), coins)]

    for i in range(PLAYERS_PER_GAME):
        k = getStateString(turn+1, coins[i:]+coins[:i])
        s2.append(k)
        if hasattr(s[i], 'rewards'): r.append(s[i].rewards[a[i]][k])
        else: r.append('rand')
    return s2, r

def playGameWeighted(playerOrders):
    weights = [x+1 for x in playerOrders]
    winner = random.choices(playerOrders, weights=weights, k=1)[0]
    weights = [1 if x == winner else 0 for x in playerOrders]
    return random.choices(list(range(len(playerOrders))), weights=weights, k=1)[0]
    
def runEpoch(tomOrders, links, states, weights=None):
    newTomOrders = []
    for i in range(POPULATION_SIZE):
        playerOrders = [tomOrders[x] for x in links[i]]
        newTomOrders.append(competition(playerOrders, states, weights=weights))
    return newTomOrders

def competition(playerOrders, states, weights=None, show=True):
    if weights is None:
        wins = [0]*len(playerOrders)
        for i in range(NUMBER_OF_GAMES_REPEAT):
            winner = playGames(states, playerOrders)
            if winner != -1: wins[winner] += 1
        return playerOrders[np.argmax(wins)], wins
    else:
        return random.choices(sorted(playerOrders), weights=weights[repr(sorted(playerOrders))], k=1)[0]
        
def playGames(states, playerOrders, show=False):
    agents = []
    for i, o in enumerate(playerOrders):
        agents.append(Agent(o, i, eps, states))

    for i in range(NUMBER_OF_GAMES):
        winner, agents = playGame(states, agents, show=show)
        if i == NUMBER_OF_GAMES-1: return winner

def playGame(states, agents, show=False):
    for agent in agents:
        agent.k = getStateString(0, [0]*PLAYERS_PER_GAME)
    r = [0]*PLAYERS_PER_GAME

    for turn in range(NUMBER_OF_TURNS):
        s = [states[agent.k] for agent in agents]
        a = [agent.selectAction(states) for agent in agents]
        k, r = makeMoves(s, a, turn)

        win = None
        for i, agent in enumerate(agents):
            if k[i] in ['win','tie','loss']:
                for j, agent2 in enumerate(agents):
                    p = (PLAYERS_PER_GAME-(j-i))%PLAYERS_PER_GAME
                    agent2.qValues[p][agent.k][a[i]] += alpha * (r[i] - agent2.qValues[p][agent.k][a[i]])
                if k[i] == 'win': win = i
                if k[i] == 'tie': win = -1
            else:
                for j, agent2 in enumerate(agents):
                    p = (PLAYERS_PER_GAME-(j-i))%PLAYERS_PER_GAME
                    q2 = max(agent2.qValues[p][k[i]].values())
                    agent2.qValues[p][agent.k][a[i]] += alpha * (r[i] + gamma * q2 - agent2.qValues[p][agent.k][a[i]])
            
        for i, agent in enumerate(agents):
            agent.learn(a)
            if win is None:
                agent.k = k[i]

        if win is not None: return win, agents
        
def movingaverage(x, window_size=None):
    if window_size is None: window_size = SMOOTHING_WINDOW
    window = np.ones(int(window_size))/float(window_size)
    z = int(window_size/2)
    return np.concatenate(([np.nan]*z, np.convolve(x, window, 'same')[z:-z]))

def getWinrates(states, show=True):
    winrates = {}
    permutations = [p for p in itertools.product([0,1,2], repeat=PLAYERS_PER_GAME)]
    for p in permutations:
        config = sorted(p)
        if repr(config) not in winrates:
            if config.count(config[0]) == len(config): winrates[repr(config)] = [1]*PLAYERS_PER_GAME
            else:
                _, rates = competition(config, states, weights=None, show=show)
                winrates[repr(config)] = rates

                config.count(config[0])
    if show: print(f"rates[((1,3,5), {eps}, {NUMBER_OF_GAMES})] = {winrates}")
    return winrates

def populationSimTwice(states, winrates, firstPopulationHistory=None):
    plotSecond = firstPopulationHistory is not None
    tomOrders = [0]*POPULATION_SIZE
    tomOrders[0] = 1

    populationHistory = [Counter(tomOrders)]
    i = 0
    stable = 0
    while True:
        # print(f"epoch {i}")
        if i >= SECOND_ORDER_EPOCH and plotSecond: tomOrders[0] = 2
        else: tomOrders[0] = 1
        tomOrders = runEpoch(tomOrders, states, weights=winrates)
        populationHistory.append(Counter(tomOrders))
        if (i >= 100
            and abs(populationHistory[i][0] - populationHistory[i-100][0]) < POPULATION_SIZE/200
            and abs(populationHistory[i][1] - populationHistory[i-100][1]) < POPULATION_SIZE/200
            and abs(populationHistory[i][2] - populationHistory[i-100][2]) < POPULATION_SIZE/200
            and (not plotSecond or len(populationHistory) >= len(firstPopulationHistory))):
            stable += 1
            if stable > 10: break
        i += 1

    if plotSecond:
        firstZeroOrderHistory = [x[0] for x in firstPopulationHistory]
        firstFirstOrderHistory = [x[1] for x in firstPopulationHistory]
        diff = len(populationHistory) - len(firstPopulationHistory)
        if diff > 0:
            firstZeroOrderHistory.extend([firstZeroOrderHistory[-1]]*diff)
            firstFirstOrderHistory.extend([firstFirstOrderHistory[-1]]*diff)
        zeroOrderHistory = [x[0] for x in populationHistory]
        firstOrderHistory = [x[1] for x in populationHistory]
        secondOrderHistory = [x[2] for x in populationHistory]
        fig = plt.figure(figsize=(10, 5))
        plt.plot(movingaverage(firstZeroOrderHistory), label="ToM$_{0}$ (no ToM$_{2}$)", alpha=0.2, color='C0')
        plt.plot(movingaverage(firstFirstOrderHistory), label="ToM$_{1}$ (no ToM$_{2}$)", alpha=0.2, color='C1')
        plt.plot(movingaverage(zeroOrderHistory), label="ToM$_{0}$", color='C0')
        plt.plot(movingaverage(firstOrderHistory), label="ToM$_{1}$", color='C1')
        plt.plot(movingaverage(secondOrderHistory), label="ToM$_{2}$", color='C2')
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [2,0,3,1,4]
        plt.legend([handles[i] for i in order],[labels[i] for i in order], prop={'size': 11}, loc='center left')
        plt.xlabel("Epoch")
        plt.ylabel("Population")
        plt.axvline(x=SECOND_ORDER_EPOCH, linestyle ="--", color='black')
        fig.subplots_adjust(left=0.065, bottom=0.09, right=0.99, top=0.98, wspace=1, hspace=1)
        fig.savefig(f'eps{eps}_g{NUMBER_OF_GAMES}.png')
        return populationHistory[-1]
    else:
        return populationSimTwice(states, winrates, firstPopulationHistory=populationHistory)

def populationSim(states, winrates, wellMixed=False, save=False):
    tomOrders = [0]*POPULATION_SIZE
    tomOrders[0] = 1

    populationHistory = [Counter(tomOrders)]
    links = []
    if wellMixed:
        for x in range(POPULATION_SIZE):
            links.append([x] + random.sample(list(range(x))+list(range(x+1,POPULATION_SIZE)), k=PLAYERS_PER_GAME-1))
    else:
        for x in range(POPULATION_SIZE):
            lBound = int(x - (PLAYERS_PER_GAME-1)/2 + POPULATION_SIZE) % POPULATION_SIZE
            rBound = int(x + (PLAYERS_PER_GAME-1)/2 + 1) % POPULATION_SIZE
            if rBound > lBound: players = list(range(lBound,rBound))
            else: players = list(range(lBound,POPULATION_SIZE)) + list(range(0,rBound))
            links.append(players)
    i = 0
    while True:
        # print(f"epoch {i}")
        if i >= SECOND_ORDER_EPOCH:
            if not populationHistory[-1][2]: tomOrders[0] = 2
        elif not populationHistory[-1][1]: tomOrders[0] = 1
        tomOrders = runEpoch(tomOrders, links, states, weights=winrates)
        populationHistory.append(Counter(tomOrders))
        if (i >= max(NUMBER_OF_EPOCHS, SECOND_ORDER_EPOCH*2)
            and abs(np.mean([x[0] for x in populationHistory[int(i*0.8):int(i*0.85)]]) - np.mean([x[0] for x in populationHistory[int(i*0.95):]])) < POPULATION_SIZE/500
            and abs(np.mean([x[1] for x in populationHistory[int(i*0.8):int(i*0.85)]]) - np.mean([x[1] for x in populationHistory[int(i*0.95):]])) < POPULATION_SIZE/500
            and abs(np.mean([x[2] for x in populationHistory[int(i*0.8):int(i*0.85)]]) - np.mean([x[2] for x in populationHistory[int(i*0.95):]])) < POPULATION_SIZE/500):
            break
        i += 1

    zeroOrderHistory = [x[0] for x in populationHistory]
    firstOrderHistory = [x[1] for x in populationHistory]
    secondOrderHistory = [x[2] for x in populationHistory]
    fig = plt.figure(figsize=(10, 5))
    plt.plot(movingaverage(zeroOrderHistory), label="ToM$_{0}$", color='C0')
    plt.plot(movingaverage(firstOrderHistory), label="ToM$_{1}$", color='C1')
    plt.plot(movingaverage(secondOrderHistory), label="ToM$_{2}$", color='C2')
    plt.legend(prop={'size': 11}, loc='center left')
    plt.xlabel("Epoch")
    plt.ylabel("Population")
    plt.axvline(x=SECOND_ORDER_EPOCH, linestyle ="--", color='black')
    fig.subplots_adjust(left=0.065, bottom=0.09, right=0.99, top=0.98, wspace=1, hspace=1)
    mixed = ["lattice","mixed"][int(wellMixed)]
    if save: fig.savefig(f'{mixed}_eps{eps}_g{NUMBER_OF_GAMES}.png')
    else: plt.show()
    return populationHistory[-1]


def getOldWinrates():
    rates = {}
    rates[((1,3,5), 0.05, 1)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [6, 3, 83], '[0, 0, 2]': [3, 3, 87], '[0, 1, 1]': [75, 8, 9], '[0, 1, 2]': [81, 11, 3], '[0, 2, 2]': [80, 6, 7], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [11, 9, 7], '[1, 2, 2]': [8, 10, 10], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.1, 1)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [6, 14, 74], '[0, 0, 2]': [7, 6, 80], '[0, 1, 1]': [69, 12, 13], '[0, 1, 2]': [66, 11, 14], '[0, 2, 2]': [63, 20, 13], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [19, 13, 15], '[1, 2, 2]': [13, 18, 12], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.2, 1)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [17, 17, 40], '[0, 0, 2]': [11, 18, 45], '[0, 1, 1]': [43, 15, 18], '[0, 1, 2]': [35, 23, 15], '[0, 2, 2]': [39, 14, 21], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [17, 22, 15], '[1, 2, 2]': [20, 16, 16], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.3, 1)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [23, 20, 26], '[0, 0, 2]': [22, 22, 28], '[0, 1, 1]': [34, 21, 16], '[0, 1, 2]': [32, 24, 19], '[0, 2, 2]': [29, 21, 21], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [23, 19, 22], '[1, 2, 2]': [19, 20, 22], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.4, 1)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [24, 24, 19], '[0, 0, 2]': [29, 24, 15], '[0, 1, 1]': [20, 21, 26], '[0, 1, 2]': [19, 17, 26], '[0, 2, 2]': [20, 20, 22], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [28, 20, 18], '[1, 2, 2]': [24, 20, 24], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.05, 2)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [1, 1, 97], '[0, 0, 2]': [1, 3, 93], '[0, 1, 1]': [81, 4, 7], '[0, 1, 2]': [12, 5, 21], '[0, 2, 2]': [90, 1, 5], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [5, 9, 84], '[1, 2, 2]': [87, 7, 5], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.1, 2)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [3, 5, 84], '[0, 0, 2]': [4, 4, 86], '[0, 1, 1]': [65, 11, 6], '[0, 1, 2]': [20, 8, 26], '[0, 2, 2]': [72, 6, 9], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [19, 8, 67], '[1, 2, 2]': [70, 13, 5], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.2, 2)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [11, 11, 47], '[0, 0, 2]': [12, 11, 45], '[0, 1, 1]': [34, 17, 16], '[0, 1, 2]': [24, 14, 19], '[0, 2, 2]': [44, 11, 11], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [8, 20, 40], '[1, 2, 2]': [51, 10, 15], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.3, 2)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [11, 24, 29], '[0, 0, 2]': [27, 13, 30], '[0, 1, 1]': [23, 25, 24], '[0, 1, 2]': [18, 21, 22], '[0, 2, 2]': [24, 18, 24], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [20, 15, 30], '[1, 2, 2]': [28, 17, 24], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.4, 2)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [25, 19, 28], '[0, 0, 2]': [15, 21, 27], '[0, 1, 1]': [29, 15, 26], '[0, 1, 2]': [22, 19, 21], '[0, 2, 2]': [24, 22, 26], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [20, 19, 25], '[1, 2, 2]': [23, 20, 22], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.05, 3)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [1, 0, 99], '[0, 0, 2]': [0, 1, 97], '[0, 1, 1]': [74, 7, 11], '[0, 1, 2]': [66, 9, 18], '[0, 2, 2]': [78, 10, 10], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [3, 4, 93], '[1, 2, 2]': [73, 9, 14], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.1, 3)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [5, 3, 84], '[0, 0, 2]': [2, 5, 87], '[0, 1, 1]': [62, 13, 10], '[0, 1, 2]': [33, 22, 33], '[0, 2, 2]': [69, 9, 11], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [13, 11, 72], '[1, 2, 2]': [61, 15, 19], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.2, 3)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [7, 9, 50], '[0, 0, 2]': [5, 8, 63], '[0, 1, 1]': [36, 16, 15], '[0, 1, 2]': [26, 13, 31], '[0, 2, 2]': [42, 12, 12], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [11, 17, 43], '[1, 2, 2]': [30, 17, 25], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.3, 3)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [12, 19, 40], '[0, 0, 2]': [11, 13, 39], '[0, 1, 1]': [21, 27, 20], '[0, 1, 2]': [14, 32, 22], '[0, 2, 2]': [25, 21, 25], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [16, 17, 35], '[1, 2, 2]': [20, 25, 21], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.4, 3)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [13, 24, 32], '[0, 0, 2]': [21, 21, 29], '[0, 1, 1]': [26, 20, 25], '[0, 1, 2]': [27, 18, 27], '[0, 2, 2]': [31, 24, 15], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [16, 25, 25], '[1, 2, 2]': [33, 24, 13], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.05, 4)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [1, 2, 96], '[0, 0, 2]': [0, 0, 100], '[0, 1, 1]': [64, 18, 16], '[0, 1, 2]': [15, 15, 65], '[0, 2, 2]': [76, 10, 13], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [5, 4, 88], '[1, 2, 2]': [70, 14, 11], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.1, 4)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [2, 2, 91], '[0, 0, 2]': [1, 3, 91], '[0, 1, 1]': [50, 17, 17], '[0, 1, 2]': [18, 17, 58], '[0, 2, 2]': [54, 19, 18], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [4, 11, 75], '[1, 2, 2]': [55, 17, 13], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.2, 4)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [4, 8, 59], '[0, 0, 2]': [10, 6, 56], '[0, 1, 1]': [32, 19, 17], '[0, 1, 2]': [10, 21, 36], '[0, 2, 2]': [29, 23, 20], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [12, 15, 45], '[1, 2, 2]': [26, 19, 23], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.3, 4)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [19, 9, 46], '[0, 0, 2]': [17, 10, 39], '[0, 1, 1]': [26, 20, 24], '[0, 1, 2]': [18, 20, 31], '[0, 2, 2]': [17, 22, 32], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [19, 19, 32], '[1, 2, 2]': [21, 20, 23], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.4, 4)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [25, 12, 35], '[0, 0, 2]': [17, 25, 32], '[0, 1, 1]': [23, 27, 26], '[0, 1, 2]': [25, 17, 27], '[0, 2, 2]': [19, 23, 24], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [17, 22, 28], '[1, 2, 2]': [28, 20, 25], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.05, 5)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [1, 1, 97], '[0, 0, 2]': [0, 1, 97], '[0, 1, 1]': [61, 22, 11], '[0, 1, 2]': [8, 4, 83], '[0, 2, 2]': [65, 18, 13], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [7, 6, 83], '[1, 2, 2]': [78, 8, 10], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.1, 5)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [1, 3, 92], '[0, 0, 2]': [3, 7, 87], '[0, 1, 1]': [53, 14, 22], '[0, 1, 2]': [12, 10, 75], '[0, 2, 2]': [49, 26, 17], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [8, 11, 75], '[1, 2, 2]': [50, 15, 18], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.2, 5)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [11, 6, 54], '[0, 0, 2]': [10, 7, 60], '[0, 1, 1]': [34, 23, 15], '[0, 1, 2]': [14, 17, 33], '[0, 2, 2]': [19, 18, 23], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [16, 8, 42], '[1, 2, 2]': [35, 16, 19], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.3, 5)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [11, 9, 51], '[0, 0, 2]': [9, 13, 43], '[0, 1, 1]': [23, 24, 21], '[0, 1, 2]': [16, 17, 35], '[0, 2, 2]': [17, 22, 22], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [23, 20, 28], '[1, 2, 2]': [23, 21, 19], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.4, 5)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [19, 15, 31], '[0, 0, 2]': [16, 20, 33], '[0, 1, 1]': [29, 21, 20], '[0, 1, 2]': [26, 26, 22], '[0, 2, 2]': [25, 31, 18], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [17, 20, 29], '[1, 2, 2]': [26, 20, 20], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.05, 7)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [2, 3, 93], '[0, 0, 2]': [0, 0, 94], '[0, 1, 1]': [53, 23, 19], '[0, 1, 2]': [18, 10, 43], '[0, 2, 2]': [52, 22, 12], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [10, 5, 73], '[1, 2, 2]': [67, 15, 13], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.1, 7)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [6, 6, 85], '[0, 0, 2]': [3, 4, 81], '[0, 1, 1]': [36, 22, 29], '[0, 1, 2]': [19, 10, 48], '[0, 2, 2]': [39, 26, 17], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [9, 10, 66], '[1, 2, 2]': [37, 18, 32], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.2, 7)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [7, 5, 62], '[0, 0, 2]': [7, 11, 54], '[0, 1, 1]': [26, 30, 17], '[0, 1, 2]': [17, 14, 38], '[0, 2, 2]': [22, 17, 24], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [12, 12, 44], '[1, 2, 2]': [20, 23, 21], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.3, 7)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [15, 16, 37], '[0, 0, 2]': [9, 18, 44], '[0, 1, 1]': [23, 22, 21], '[0, 1, 2]': [15, 26, 32], '[0, 2, 2]': [18, 29, 20], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [20, 11, 39], '[1, 2, 2]': [24, 24, 19], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.4, 7)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [16, 19, 34], '[0, 0, 2]': [23, 16, 33], '[0, 1, 1]': [29, 26, 20], '[0, 1, 2]': [16, 27, 25], '[0, 2, 2]': [21, 24, 26], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [28, 15, 25], '[1, 2, 2]': [25, 28, 15], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.05, 10)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [8, 8, 83], '[0, 0, 2]': [10, 4, 82], '[0, 1, 1]': [59, 19, 16], '[0, 1, 2]': [38, 19, 34], '[0, 2, 2]': [42, 21, 22], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [8, 6, 72], '[1, 2, 2]': [60, 14, 17], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.1, 10)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [7, 6, 81], '[0, 0, 2]': [8, 11, 78], '[0, 1, 1]': [37, 26, 24], '[0, 1, 2]': [17, 11, 59], '[0, 2, 2]': [34, 19, 33], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [13, 19, 59], '[1, 2, 2]': [49, 19, 20], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.2, 10)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [6, 9, 55], '[0, 0, 2]': [14, 9, 45], '[0, 1, 1]': [29, 21, 19], '[0, 1, 2]': [19, 19, 30], '[0, 2, 2]': [13, 29, 26], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [16, 11, 41], '[1, 2, 2]': [23, 21, 24], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.3, 10)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [15, 13, 39], '[0, 0, 2]': [20, 15, 32], '[0, 1, 1]': [17, 21, 26], '[0, 1, 2]': [21, 18, 26], '[0, 2, 2]': [18, 33, 18], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [15, 22, 36], '[1, 2, 2]': [24, 27, 20], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.4, 10)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [17, 14, 40], '[0, 0, 2]': [20, 18, 35], '[0, 1, 1]': [19, 23, 29], '[0, 1, 2]': [23, 18, 32], '[0, 2, 2]': [25, 21, 21], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [23, 26, 18], '[1, 2, 2]': [26, 17, 25], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.05, 15)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [8, 13, 77], '[0, 0, 2]': [4, 9, 79], '[0, 1, 1]': [39, 29, 26], '[0, 1, 2]': [32, 23, 33], '[0, 2, 2]': [48, 22, 13], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [12, 11, 65], '[1, 2, 2]': [57, 14, 16], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.1, 15)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [13, 17, 68], '[0, 0, 2]': [10, 16, 67], '[0, 1, 1]': [31, 22, 34], '[0, 1, 2]': [29, 17, 36], '[0, 2, 2]': [31, 20, 36], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [12, 16, 53], '[1, 2, 2]': [34, 22, 28], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.2, 15)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [4, 9, 56], '[0, 0, 2]': [10, 12, 51], '[0, 1, 1]': [19, 21, 36], '[0, 1, 2]': [20, 17, 33], '[0, 2, 2]': [21, 18, 24], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [8, 22, 43], '[1, 2, 2]': [17, 27, 23], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.3, 15)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [23, 17, 35], '[0, 0, 2]': [17, 16, 35], '[0, 1, 1]': [18, 23, 27], '[0, 1, 2]': [23, 22, 29], '[0, 2, 2]': [22, 26, 21], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [27, 14, 25], '[1, 2, 2]': [28, 18, 21], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.4, 15)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [17, 21, 34], '[0, 0, 2]': [16, 22, 29], '[0, 1, 1]': [25, 16, 31], '[0, 1, 2]': [21, 28, 22], '[0, 2, 2]': [20, 31, 20], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [20, 23, 27], '[1, 2, 2]': [26, 21, 22], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.05, 20)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [9, 8, 80], '[0, 0, 2]': [8, 8, 81], '[0, 1, 1]': [28, 35, 26], '[0, 1, 2]': [31, 26, 31], '[0, 2, 2]': [33, 20, 30], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [12, 15, 60], '[1, 2, 2]': [29, 26, 26], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.1, 20)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [7, 14, 70], '[0, 0, 2]': [9, 12, 71], '[0, 1, 1]': [31, 28, 31], '[0, 1, 2]': [27, 24, 36], '[0, 2, 2]': [35, 28, 17], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [26, 9, 45], '[1, 2, 2]': [37, 21, 21], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.2, 20)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [11, 11, 49], '[0, 0, 2]': [16, 8, 48], '[0, 1, 1]': [25, 30, 16], '[0, 1, 2]': [17, 18, 34], '[0, 2, 2]': [26, 22, 21], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [16, 13, 37], '[1, 2, 2]': [25, 18, 21], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.3, 20)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [19, 12, 39], '[0, 0, 2]': [16, 12, 42], '[0, 1, 1]': [21, 29, 20], '[0, 1, 2]': [18, 13, 36], '[0, 2, 2]': [23, 21, 23], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [22, 12, 29], '[1, 2, 2]': [25, 24, 23], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.4, 20)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [19, 17, 36], '[0, 0, 2]': [14, 17, 41], '[0, 1, 1]': [20, 26, 22], '[0, 1, 2]': [22, 18, 28], '[0, 2, 2]': [22, 25, 21], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [20, 20, 28], '[1, 2, 2]': [22, 18, 27], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.05, 25)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [5, 10, 83], '[0, 0, 2]': [4, 10, 81], '[0, 1, 1]': [45, 23, 20], '[0, 1, 2]': [30, 16, 42], '[0, 2, 2]': [44, 23, 23], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [24, 22, 39], '[1, 2, 2]': [36, 19, 27], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.1, 25)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [9, 15, 73], '[0, 0, 2]': [13, 14, 69], '[0, 1, 1]': [34, 32, 24], '[0, 1, 2]': [34, 27, 25], '[0, 2, 2]': [30, 30, 25], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [25, 21, 45], '[1, 2, 2]': [34, 26, 25], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.2, 25)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [14, 16, 40], '[0, 0, 2]': [14, 10, 52], '[0, 1, 1]': [18, 22, 28], '[0, 1, 2]': [17, 31, 25], '[0, 2, 2]': [25, 22, 21], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [18, 21, 27], '[1, 2, 2]': [17, 14, 31], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.3, 25)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [13, 16, 40], '[0, 0, 2]': [21, 17, 33], '[0, 1, 1]': [24, 23, 21], '[0, 1, 2]': [25, 16, 25], '[0, 2, 2]': [21, 25, 18], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [18, 24, 25], '[1, 2, 2]': [25, 15, 22], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.4, 25)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [17, 20, 35], '[0, 0, 2]': [13, 18, 37], '[0, 1, 1]': [18, 27, 19], '[0, 1, 2]': [23, 22, 30], '[0, 2, 2]': [23, 26, 24], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [23, 19, 25], '[1, 2, 2]': [24, 17, 23], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.05, 50)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [22, 14, 61], '[0, 0, 2]': [17, 20, 61], '[0, 1, 1]': [47, 26, 19], '[0, 1, 2]': [42, 13, 32], '[0, 2, 2]': [39, 20, 29], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [21, 18, 49], '[1, 2, 2]': [38, 25, 24], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.1, 50)] = {'[0, 0, 0]': [2, 2, 2], '[0, 0, 1]': [15, 10, 72], '[0, 0, 2]': [13, 17, 63], '[0, 1, 1]': [34, 30, 25], '[0, 1, 2]': [31, 24, 30], '[0, 2, 2]': [32, 25, 27], '[1, 1, 1]': [2, 2, 2], '[1, 1, 2]': [35, 22, 31], '[1, 2, 2]': [38, 27, 20], '[2, 2, 2]': [2, 2, 2]}
    rates[((1,3,5), 0.2, 50)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [11, 10, 56], '[0, 0, 2]': [12, 10, 52], '[0, 1, 1]': [17, 31, 18], '[0, 1, 2]': [18, 24, 28], '[0, 2, 2]': [25, 21, 21], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [26, 10, 27], '[1, 2, 2]': [27, 16, 22], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.3, 50)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [14, 15, 40], '[0, 0, 2]': [14, 25, 34], '[0, 1, 1]': [19, 26, 25], '[0, 1, 2]': [18, 20, 30], '[0, 2, 2]': [18, 22, 26], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [20, 13, 35], '[1, 2, 2]': [21, 22, 24], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.4, 50)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [23, 18, 31], '[0, 0, 2]': [15, 18, 34], '[0, 1, 1]': [30, 27, 12], '[0, 1, 2]': [21, 21, 30], '[0, 2, 2]': [16, 24, 32], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [27, 25, 22], '[1, 2, 2]': [20, 29, 26], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.05, 100)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [24, 17, 32], '[0, 0, 2]': [27, 25, 23], '[0, 1, 1]': [30, 22, 18], '[0, 1, 2]': [27, 15, 29], '[0, 2, 2]': [30, 20, 20], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [26, 19, 28], '[1, 2, 2]': [28, 23, 18], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.1, 100)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [12, 13, 45], '[0, 0, 2]': [15, 13, 45], '[0, 1, 1]': [20, 26, 25], '[0, 1, 2]': [32, 11, 27], '[0, 2, 2]': [31, 21, 22], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [20, 19, 28], '[1, 2, 2]': [31, 20, 19], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.2, 100)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [5, 17, 48], '[0, 0, 2]': [17, 8, 40], '[0, 1, 1]': [26, 27, 23], '[0, 1, 2]': [24, 19, 29], '[0, 2, 2]': [25, 26, 23], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [23, 17, 28], '[1, 2, 2]': [31, 21, 21], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.3, 100)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [17, 14, 37], '[0, 0, 2]': [12, 15, 45], '[0, 1, 1]': [22, 23, 23], '[0, 1, 2]': [20, 16, 33], '[0, 2, 2]': [27, 18, 31], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [19, 26, 25], '[1, 2, 2]': [26, 23, 18], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.4, 100)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [15, 22, 33], '[0, 0, 2]': [19, 21, 33], '[0, 1, 1]': [25, 16, 29], '[0, 1, 2]': [16, 20, 36], '[0, 2, 2]': [26, 22, 23], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [14, 27, 28], '[1, 2, 2]': [22, 18, 32], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.05, 200)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [19, 27, 26], '[0, 0, 2]': [25, 21, 24], '[0, 1, 1]': [24, 20, 25], '[0, 1, 2]': [36, 11, 28], '[0, 2, 2]': [33, 25, 9], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [22, 20, 25], '[1, 2, 2]': [22, 23, 26], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.1, 200)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [10, 22, 39], '[0, 0, 2]': [17, 17, 40], '[0, 1, 1]': [25, 20, 26], '[0, 1, 2]': [31, 15, 28], '[0, 2, 2]': [27, 14, 28], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [14, 27, 24], '[1, 2, 2]': [27, 22, 19], '[2, 2, 2]': [1, 1, 1]}
    rates[((1,3,5), 0.2, 200)] = {'[0, 0, 0]': [1, 1, 1], '[0, 0, 1]': [15, 17, 39], '[0, 0, 2]': [14, 12, 45], '[0, 1, 1]': [21, 22, 23], '[0, 1, 2]': [19, 22, 30], '[0, 2, 2]': [29, 18, 20], '[1, 1, 1]': [1, 1, 1], '[1, 1, 2]': [17, 19, 39], '[1, 2, 2]': [18, 26, 26], '[2, 2, 2]': [1, 1, 1]}
    return rates
    
def allGames(states, wellMixed=False):
    counters = {}
    rates = getOldWinrates()
    for g in [2, 3, 4, 5, 7, 10, 15, 20, 25, 50, 100, 200]:
        for e in [0.05, 0.1, 0.2, 0.3, 0.4]:
            global eps
            eps = e
            global NUMBER_OF_GAMES
            NUMBER_OF_GAMES = g
            if ((1,3,5),e,g) in rates: winrates = rates[((1,3,5),e,g)]
            else: winrates = getWinrates(states, show=False)
            print(f"rates[((1,3,5), {eps}, {NUMBER_OF_GAMES})] = {winrates}")
            c = populationSim(states, winrates, wellMixed=wellMixed, save=True)
            counters[((1,3,5), e, g)] = c
    print(counters)

def main():
    states = getStates()
    print(f'All states found ({len(states)}).')

    for s in states.values(): initializeState(s, states)

    allGames(states, wellMixed=False)

if __name__ == "__main__":
    main()