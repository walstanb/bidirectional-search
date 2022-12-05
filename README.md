## Team 25 : The Q-Learners
Implements : Bidirectional search that is guaranteed to meet in the middle : [link](https://ojs.aaai.org/index.php/AAAI/article/view/10436/10295)

Team Members :
1. Agneet Chatterjee 
2. Kavya Sree Bachina 
3. Walstan Baptista 
4. Souradip Nath 

We implement the "Meet in Middle" (MM) algorithm on the Pacman search environemnt defined in “UC Berkeley Pacman AI Projects” developed by the DeNero, J.; Klein, D. Available: [link](http://ai.berkeley.edu/project_overview.html.)

We develop both the MM algorithm as well as its non-heuristic variant MM0. We implement it for two problem,

* The Position Search - To find the path from the start location to the location of the single food pellet in the pacman maze.
* The Corner Search - To find the path to eat 4 food pellets spread across the 4 corners of the pacman maze.

Instructions to run : 

## 1) Position Search Problem : 

```python pacman.py -l MAZE_NAME -p SearchAgent -a fn=mm,heuristic=HEURISTIC_NAME```

Replace MAZE_NAME with 1 amongst the 6 mazes we have created:

1. tinyMaze
2. smallMaze
3. mediumMaze
4. contoursMaze
5. customMaze
6. openMaze

**In order to run MM0:** Replace HEURISTIC_NAME with nullHeuristic

**In order to run MM with Manhattan Heuristic:** Replace HEURISTIC_NAME with manhattanHeuristic

**In order to run MM with any other Heuristic:** Define the heuristic function in [search.py](https://github.com/walstanb/bidirectional-search/blob/main/search/search.py), and replace HEURISTIC_NAME with the function name



## 2) Corner Search Problem : 

```python pacman.py -l MAZE_NAME -p SearchAgent -a fn=mm_corner,prob=CornersProblem```

Replace MAZE_NAME with 1 amongst the 6 mazes we have created : 

1. tinyCorners
2. mediumCorners
3. bigCorners
4. customTinyCorners
5. customMediumCorners
6. customBigCorners

The above code will run by default with a heuristic. To run MM0, uncomment these two lines : [1](https://github.com/walstanb/bidirectional-search/blob/main/search/search.py#L556) and [2](https://github.com/walstanb/bidirectional-search/blob/main/search/search.py#L656), and run the code again.
