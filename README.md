# Reinforcement-Learning-agent
The purpose of this project is to compare the performances of three different learning agents - Reinforcement Learning (Monte-Carlo policy based gradient), Actor-critic Agent, and Approximate Q-Learning.

## Running Pacman

In order to visualize the pacman running under one of the agents, simply run `python pacman.py` with the option to choose layout, agent type, number of runs, training iterations, etc.

|Option|Description|
|-|-|
| -h, --help | Show help message and exit |
| -n GAMES, --numGames=GAMES | The number of GAMES to play [Default: 1] |
|-l LAYOUT_FILE, --layout=LAYOUT_FILE|the LAYOUT_FILE from which to load the map layout<br>[Default: mediumClassic]|
|-p TYPE, --pacman=TYPE|the agent TYPE in the pacmanAgents module to use<br>[Default: KeyboardAgent]|
|-t, --textGraphics|Display output as text only|
|-q, --quietTextGraphics|Generate minimal output and no graphics|
|-g TYPE, --ghosts=TYPE|the ghost agent TYPE in the ghostAgents module to use<br>[Default: RandomGhost]|
|-k NUMGHOSTS, --numghosts=NUMGHOSTS|The maximum number of ghosts to use<br>[Default: 4]|
|-z ZOOM, --zoom=ZOOM|Zoom the size of the graphics window<br>[Default: 1.0]|
|-f, --fixRandomSeed|Fixes the random seed to always play the same game|
|-r, --recordActions|Writes game histories to a file (named by the time they were played)|
|--replay=GAMETOREPLAY|A recorded game file (pickle) to replay|
|-a AGENTARGS, --agentArgs=AGENTARGS|Comma separated values sent to agent. e.g. "opt1=val1,opt2,opt3=val3"|
|-x NUMTRAINING, --numTraining=NUMTRAINING|How many episodes are training (suppresses output)<br>[Default: 0]|
|--frameTime=FRAMETIME|Time to delay between frames; <0 means keyboard<br>[Default: 0.1]|
|-c, --catchExceptions|Turns on exception handling and timeouts during games|
|--timeout=TIMEOUT|Maximum length of time an agent can spend computing in a single game<br>[Default: 30]|

## Generating Layouts

The layouts may be randomly-generated using `test.sh` script, with the arguments `layout`, `run_number`, and `training_episodes`. The results will be stored in the `results` folder.

## Running Student's T-Test

The Student's T-Test may be run between each pair of agents, for each generated layout in the `results` folder.

The result of the T-Test will be stored in `t_test_results.json`.

## Running Normality Test

TODO: Insert instructions here
