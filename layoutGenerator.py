import random, argparse
from collections import defaultdict

CONST_FOOD = "."
CONST_WALL = "%"
CONST_CAPSULE = "o"
CONST_GHOST = "G"
CONST_PACMAN = "P"
CONST_NEWLINE = "\n"

class MiniGrid:
    CONST_VERTICAL = 0
    CONST_HORIZONTAL = 1 

    CONST_LEFT = 2
    CONST_RIGHT = 3
    CONST_UP = 4
    CONST_DOWN = 5

    def __init__(self, r, c):
        if(r == 0):
            raise Exception("miniGrid row cannot be initialized as 0!")
        if(c == 0):
            raise Exception("miniGrid col cannot be initialized as 0!")

        self.__r = r
        self.__c = c
        self.__grid = []
        for _ in range(r):
            self.__grid.append([CONST_FOOD] * c)
        
        if((r == 1) or (c == 1)): 
            self.__createLinearGrid()
        else:
            self.__createGrid()
    
    def __createLinearGrid(self):
        
        if(self.__r == 1):
            pieceLen =  random.randint(1, self.__c)
            for c in range(pieceLen):
                self.__grid[0][c] = CONST_WALL
        else:
            pieceLen =  random.randint(1, self.__r)
            for r in range(pieceLen):
                self.__grid[r][0] = CONST_WALL

    def __createGrid(self):
        primaryPiece = random.choice([MiniGrid.CONST_VERTICAL, MiniGrid.CONST_HORIZONTAL])

        if(primaryPiece == MiniGrid.CONST_VERTICAL):
            primaryPieceStartCol = random.randrange(self.__c)
            # primaryPieceLen = random.randint(1, self.__r)
            primaryPieceLen = self.__r

            for r in range(primaryPieceLen):
                self.__grid[r][primaryPieceStartCol] = CONST_WALL

            secondaryPieceStartDirections = []
            if(primaryPieceStartCol != 0):
                secondaryPieceStartDirections.append(MiniGrid.CONST_LEFT)
            if(primaryPieceStartCol != (self.__c - 1)):
                secondaryPieceStartDirections.append(MiniGrid.CONST_RIGHT)

            secondaryPieceDirection = random.choice(secondaryPieceStartDirections)
            secondaryPieceStartRow = random.randrange(primaryPieceLen)
            secondaryPieceMaxLen = primaryPieceStartCol if (secondaryPieceDirection == MiniGrid.CONST_LEFT) else (self.__c - primaryPieceStartCol - 1)
            secondaryPieceLen = random.randrange(1, secondaryPieceMaxLen + 1)

            if(secondaryPieceDirection == MiniGrid.CONST_LEFT):
                for c in range(primaryPieceStartCol - secondaryPieceLen, primaryPieceStartCol):
                    self.__grid[secondaryPieceStartRow][c] = CONST_WALL
            else:
                for c in range(primaryPieceStartCol + 1, primaryPieceStartCol + secondaryPieceLen + 1):
                    self.__grid[secondaryPieceStartRow][c] = CONST_WALL

        else:
            primaryPieceStartRow = random.randrange(self.__r)
            # primaryPieceLen = random.randint(1, self.__c)
            primaryPieceLen = self.__c

            for c in range(primaryPieceLen):
                self.__grid[primaryPieceStartRow][c] = CONST_WALL

            secondaryPieceStartDirections = []
            if(primaryPieceStartRow != 0):
                secondaryPieceStartDirections.append(MiniGrid.CONST_UP)
            if(primaryPieceStartRow != (self.__r - 1)):
                secondaryPieceStartDirections.append(MiniGrid.CONST_DOWN)

            secondaryPieceDirection = random.choice(secondaryPieceStartDirections)
            secondaryPieceStartCol = random.randrange(primaryPieceLen)
            secondaryPieceMaxLen = primaryPieceStartRow if (secondaryPieceDirection == MiniGrid.CONST_UP) else (self.__r - primaryPieceStartRow - 1)
            secondaryPieceLen = random.randrange(1, secondaryPieceMaxLen + 1)

            if(secondaryPieceDirection == MiniGrid.CONST_UP):
                for r in range(primaryPieceStartRow - secondaryPieceLen, primaryPieceStartRow):
                    self.__grid[r][secondaryPieceStartCol] = CONST_WALL
            else:
                for r in range(primaryPieceStartRow + 1, primaryPieceStartRow + secondaryPieceLen + 1):
                    self.__grid[r][secondaryPieceStartCol] = CONST_WALL
    
    def getGrid(self):
        return self.__grid
    
    def __str__(self):
        s = ""
        for r in range(self.__r):
            s += str(self.__grid[r]) + CONST_NEWLINE
        return s.rstrip(CONST_NEWLINE)

class Grid:

    def __init__(self, gridDim, miniGridDim, capsuleCount):
        self.__capsuleCount = capsuleCount
        self.__gridDim = gridDim
        self.__miniGridDim = miniGridDim
        self.__halfGrid = []
        self.__fullGrid = []

        self.__createHalfGrid()
        self.__createFullGrid()
        self.__addCapsules()
        self.__createFullGridString()

    def __createHalfGrid(self):
        for r in range(self.__miniGridDim, self.__gridDim, self.__miniGridDim):
            row = []

            for c in range(self.__miniGridDim, self.__gridDim//2 + 1, self.__miniGridDim):
                row.append(MiniGrid(self.__miniGridDim, self.__miniGridDim))
        
            if(c != self.__gridDim//2):
                row.append(MiniGrid(self.__miniGridDim, (self.__gridDim//2) - c))
            
            self.__halfGrid.append(row)
        
        if(r != self.__gridDim):   
            remainingR = self.__gridDim - r
            row = []

            for c in range(self.__miniGridDim, self.__gridDim//2 + 1, self.__miniGridDim):
                row.append(MiniGrid(remainingR, self.__miniGridDim))
            
            if(c != self.__gridDim//2):
                row.append(MiniGrid(remainingR, (self.__gridDim//2) - c))
            
            self.__halfGrid.append(row)
    
    def printHalfGrid(self):
        for r in range(len(self.__halfGrid)):
            for c in range(len(self.__halfGrid[r])):
                print(str(r)+" "+str(c))
                print(self.__halfGrid[r][c])
                print()

    def __createFullGrid(self):
        spacerRow = None

        for r in range(len(self.__halfGrid)):
            #   Add left border wall
            fullGridRows = defaultdict(lambda: [CONST_WALL, CONST_FOOD])

            for c in range(len(self.__halfGrid[r])):
                miniGrid = self.__halfGrid[r][c].getGrid()
                
                for miniGridRow in range(len(miniGrid)):
                    #   Add space after each miniGrid row in finalGrid's row
                    fullGridRows[miniGridRow] += miniGrid[miniGridRow] + [CONST_FOOD]
            
            finalFullRows = []
            for i in range(max(fullGridRows.keys(), default= -1) + 1):
                firstHalfRow = fullGridRows[i]
                secondHalfRow = firstHalfRow[::-1][1:]
                finalFullRows.append(firstHalfRow + secondHalfRow)
            
            #   Add space between miniGrid rows in finalGrid's row
            spacerRow = spacerRow if(spacerRow != None) else ([CONST_WALL] + ([CONST_FOOD] *(len(finalFullRows[-1]) - 2)) + [CONST_WALL] )
            finalFullRows.append(spacerRow.copy())
            self.__fullGrid += finalFullRows
        
        # Add bottom border wall
        self.__fullGrid.append([CONST_WALL] * len(spacerRow))

        # Add top border wall
        self.__fullGrid.insert(0, spacerRow[:-5] + ([CONST_GHOST] * 4) + [CONST_WALL])
        self.__fullGrid.insert(0, [CONST_WALL] * len(spacerRow))

        self.__fullGrid[-2][1] = CONST_PACMAN

    def __addCapsules(self):
        foodPositions = []
        
        for r in range(0, len(self.__fullGrid) - 1):
            for c in range(1, len(self.__fullGrid[r]) - 1):
                if(self.__fullGrid[r][c] == CONST_FOOD):
                    foodPositions.append((r, c))

        for _ in range(self.__capsuleCount):
            x,y = random.choice(foodPositions)
            self.__fullGrid[x][y] = CONST_CAPSULE
            foodPositions.remove((x,y))

    def __createFullGridString(self):
        stringifiedRows = []
        for row in self.__fullGrid:
            stringifiedRows.append("".join(row))
        self.__fullGridString = "\n".join(stringifiedRows)

    def __str__(self):        
        return self.__fullGridString.replace("", " ")
    
    def writeToFile(self, pathToFile):
        f = open(pathToFile, "w")
        f.write(self.__fullGridString + CONST_NEWLINE)
        f.close()


# _____________Driver code_____________
# A sample command
# python gridMaker.py -g 20 -m 3 -c 3

parser = argparse.ArgumentParser()
parser.add_argument("-g", required=True, type=int, help="Size of the grid")
parser.add_argument("-m", required=True, type=int, help="Size of the miniGrid. miniGrids make up the main grid")
parser.add_argument("-c", type=int, help="Number of capsules to add to the grid", default=0)
args = parser.parse_args()

# Inputs    
    # Assumed grid is a square
    # Assumed gridDimension is even and length of square grid's side
gridDimension = args.g

    # miniGrid dimension
    # This parameter allows you to split up the bigger grid into smaller ones. 
    # The smaller this value, the more wall-dense the grid becomes
miniGridDimension = args.m

capsuleCount = args.c

# Validations
    # Ensures board symmetry
if((gridDimension % 2) != 0):
    raise Exception("maze dimension must be even!")
if((gridDimension//2) < miniGridDimension):
    raise Exception("miniGridDimensions must be <= halfGridDimension")

o = Grid(gridDimension, miniGridDimension, capsuleCount)
print(o)

#   Overwrite an existing layout to avoid extra overhead of adding layout to simulator's in-memory indices.
o.writeToFile("./capsuleClassic.lay")
