import random
import sys
import util


def generateMaze(numRows=25, numCols=25, prob="PositionSearchProblem"):
    maze = [[" " for _ in range(numCols + 2)] for _ in range(numRows + 2)]

    # Get the boundary done
    for i in range(numRows + 2):
        maze[i][0] = "%"
        maze[i][-1] = "%"
    for j in range(numCols + 2):
        maze[0][j] = "%"
        maze[-1][j] = "%"

    # Place the pacman
    pacmanPositionX = 1
    pacmanPositionY = numCols

    # print(f"Pacman: ({pacmanPositionX}, {pacmanPositionY})")
    maze[pacmanPositionX][pacmanPositionY] = "P"

    # Place the food item
    if prob == "PositionSearchProblem":
        maze[numRows][1] = "."

    if prob == "CornersProblem":
        maze[numRows][1] = "."
        maze[numRows][numCols] = "."
        maze[1][numCols] = "."
        maze[1][1] = "."

    # Place the walls

    for row in range(2, numRows, 2):
        isWall = util.flipCoin(0.5)
        if isWall:
            wallLength = random.randint(1, numCols - 4)
            start = random.randint(1, numCols - wallLength)

            for col in range(start, start + wallLength):
                if maze[row][col] != "P" and maze[row][col] != ".":
                    maze[row][col] = "%"

    for col in range(2, numCols, 2):
        isWall = util.flipCoin(0.5)
        if isWall:
            wallLength = random.randint(1, numRows - 4)
            start = random.randint(1, numRows - wallLength)

            for row in range(start, start + wallLength):
                if maze[row][col] != "P" and maze[row][col] != ".":
                    maze[row][col] = "%"

    for row in maze:
        for col in row:
            print(col, end="")
        print()


if __name__ == "__main__":

    """
    Command Line Args must be in the form of
        python generateMaze.py  --> Generates a Position Search Problem Maze with dimension 25 x 25
        python generateMaze.py -r 10 -c 20 --> Generates a Position Search Problem Maze with dimension 10 x 20
        python generateMaze.py -r 10 -c 20 -p CornersProblem --> Generates a Corners Problem Maze with dimension 10 x 20
        python generateMaze.py  -p CornersProblem --> Generates a Corners Problem Maze with dimension 25 x 25
    """

    args = sys.argv

    if len(args) == 1:
        generateMaze()

    elif len(args) == 3:
        problem = args[-1]
        generateMaze(prob=problem)

    elif len(args) == 5:
        rows = int(args[2])
        cols = int(args[-1])
        generateMaze(numRows=rows, numCols=cols)

    elif len(args) == 7:
        rows = int(args[2])
        cols = int(args[4])
        problem = args[-1]
        generateMaze(numRows=rows, numCols=cols, prob=problem)

    else:
        print("Error: Incorrect Command")
