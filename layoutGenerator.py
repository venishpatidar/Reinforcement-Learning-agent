"""
layoutGenerator.py
------------------
This script is responsible for generating random symmetric pacman layouts.

Constraints/Assumptions
-----------------------
1) The script assumes generated layouts to be squares. Hence the gridDimension input (-g) only accepts
a single value
2) The final generated layout may not be a square. During generation the script adds rows and columns
to ensure there are no dead-ends. This may cause the final layout to have a few more columns than rows
or vice versa
3) Pacman always starts from the bottom left
4) A maximum of 4 ghosts may be present and they all start from the top right
5) We take in even gridDimensions to ensure we can divide the board evenly to ensure symmetry

Algorithm
---------
This script ensures symmetry of the final layout by generating only the left side of the layout. This
half grid's dimension would have 'g' rows and 'g/2' columns (Again, the final layout may have a few
more rows or columns as explained earlier). Once half the layout is generated, the right side is created
by simply mirroring the left. This ensures layout symmetry.

To generate half the layout, the script splits the layout (Referred to by the Grid class) into multiple 
smaller square grids (Referred to by the MiniGrid class). Hence joining all these MiniGrids yields
half the grid, which is then mirrored to get the full grid.

Each HalfGrid is responsible for generating a random wall. Hence the maximum dimension of any wall in the 
final layout is the maximum size of a MiniGrid. This is important to note as the MiniGrid dimension is passed 
in by the miniGridDimension (-m) parameter. Large -m values imply that we can have larger individual MiniGrids, 
but the total number of MiniGrids is less. Hence we can have larger contiguous wall structures, but fewer
walls in total. Such layouts appear to be more 'spaced out'. On the other hand lower -m values imply that 
we have smaller individual MiniGrids, but a higher number Minigrids in total. Hence individual walls may be
smaller, but we have more walls per unit space. So the layout appears to be more 'dense'

It is also important to note that not all MiniGrids would be of size m x m. If the halfGrid doesn't divide 
completely by the provided 'm', there may be some MiniGrids that are rectangular to accommodate the remainder
rows/columns.

Once all MiniGrids are generated, the script proceeds to join all of them together to get the final halfGrid.
While doing so it adds a row and column in between any 2 MiniGrids. This ensures that all non-wall regions in 
the layout are accessibile. Consider the following MiniGrids

   ------
   |####|
   ------
   ---    -------     
   |#|    |   # |
   |#|    |#### |
   |#|    |   # |
   ---    -------

Joining them all together would give,

    #####
    # X #
    #####
    #   #

The X cell becomes inaccessible. Hence adding empty rows and columns between any 2 MiniGrids avoids this.

With the halfGrid created, the script simply mirrors it to get the fullGrid. It then adds the top and bottom walls.
adds the Pacman and Ghosts to generate the final layout.


Wall Generation in a MiniGrid
-----------------------------
Pacman maze walls are generally rectangular or 'T' shaped. This script honors this during generation. To generate a 
wall, the script first generates 2 pieces, the primary and the secondary piece. Joining both together gives the
final wall.

First the script decides the orientation of the primary piece. It randomly chooses between VERTICAL and HORIZONTAL.
Assume VERTICAL is chosen. The primary piece length is set to the maximum allowable by the MiniGrid dimension. If
the MiniGrid is a x b, since the primaryPiece is VERTICAL, it will be set to have a length of b. The next decision
to make is where to place the primaryPiece. The script chooses an index from all available column indices in the 
MiniGrid (0 to b, both included), since the piece is VERTICAL.

The next step is to generate the secondaryPiece. Since the primary is VERTICAL, the secondary has to be HORIZONTAL.
The secondary piece can either grow leftwards or rightwards yielding the following possible walls,
     
#             #
##### or  #####
#             #

The script first randomly chooses from the possible growth directions (If the vertical piece is already on the
right border of the MiniGrid, left is the only possible option. It's a similar case if the vertical piece is already
on the left border).

With the direction selected, the script then randomly chooses the secondaryPiece's length from (0 to maxPosibleLength).
If 0 is chosen, we get a rectangular wall of only the VERTICAL primaryPiece. Else we get some T shaped wall. 
"""


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
            #   Fill all empty spaces with food initially. Walls will replace them later during generation
            self.__grid.append([CONST_FOOD] * c)
        
        if((r == 1) or (c == 1)): 
            self.__createLinearGrid()
        else:
            self.__createGrid()
    
    # If we have a MiniGrid with row or column as 1, there is only space for a primaryPiece.
    # Handle this case separately
    def __createLinearGrid(self):
        
        if(self.__r == 1):
            pieceLen =  random.randint(1, self.__c)
            for c in range(pieceLen):
                self.__grid[0][c] = CONST_WALL
        else:
            pieceLen =  random.randint(1, self.__r)
            for r in range(pieceLen):
                self.__grid[r][0] = CONST_WALL

    # Generate wall randomly for this MiniGrid
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

    # Split halfGrid into many MiniGrids
    # self.__halfGrid = [Rows of MiniGrids]
    def __createHalfGrid(self):
        for r in range(self.__miniGridDim, self.__gridDim, self.__miniGridDim):
            row = []

            for c in range(self.__miniGridDim, self.__gridDim//2 + 1, self.__miniGridDim):
                row.append(MiniGrid(self.__miniGridDim, self.__miniGridDim))
        
            # Handle remainder columns
            if(c != self.__gridDim//2):
                row.append(MiniGrid(self.__miniGridDim, (self.__gridDim//2) - c))
            
            self.__halfGrid.append(row)
        
        # Handle remainder rows
        if(r != self.__gridDim):   
            remainingR = self.__gridDim - r
            row = []

            for c in range(self.__miniGridDim, self.__gridDim//2 + 1, self.__miniGridDim):
                row.append(MiniGrid(remainingR, self.__miniGridDim))
            
            if(c != self.__gridDim//2):
                row.append(MiniGrid(remainingR, (self.__gridDim//2) - c))
            
            self.__halfGrid.append(row)
    
    # For debugging and visualization
    def printHalfGrid(self):
        for r in range(len(self.__halfGrid)):
            for c in range(len(self.__halfGrid[r])):
                print(str(r)+" "+str(c))
                print(self.__halfGrid[r][c])
                print()

    # Mirror the halfGrid and create the full grid
    # self.__fullGrid = [Rows of Characters such as % . P G o]
    def __createFullGrid(self):
        spacerRow = None

        # Convert each MiniGrid's rows (Yes, one MiniGrid corresponds to multiple rows) into fullGrid's rows
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
                secondHalfRow = firstHalfRow[::-1][1:] # Mirroring
                finalFullRows.append(firstHalfRow + secondHalfRow)
            
            #   Add space between miniGrid rows in finalGrid's row
            #   Spacer row is essentially a row full of food bordered by the left and right walls
            spacerRow = spacerRow if(spacerRow != None) else ([CONST_WALL] + ([CONST_FOOD] *(len(finalFullRows[-1]) - 2)) + [CONST_WALL] )
            finalFullRows.append(spacerRow.copy())
            self.__fullGrid += finalFullRows
        
        # Add bottom border wall
        self.__fullGrid.append([CONST_WALL] * len(spacerRow))
        
        # Add top border wall
        self.__fullGrid.insert(0, spacerRow.copy())
        self.__fullGrid.insert(0, [CONST_WALL] * len(spacerRow))

        # Add ghosts to center column sparating halfGrids
        midColumn = len(self.__fullGrid[0])//2
        midRow = len(self.__fullGrid)//2
        self.__fullGrid[midRow][midColumn] = CONST_GHOST
        self.__fullGrid[midRow - 1][midColumn] = CONST_GHOST
        self.__fullGrid[midRow + 1][midColumn] = CONST_GHOST
        self.__fullGrid[midRow - 2][midColumn] = CONST_GHOST

        # Add pacman in bottom left
        self.__fullGrid[-2][1] = CONST_PACMAN

    #   Add capsules randomly to free spaces available in the grid
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

if __name__ == "__main__":
    """
        USAGE:      python layoutGenerator.py <options>
        OPTIONS:    
                -g  Size of the grid, Assumed grid is a square,
                    gridDimension is even and length of square grid's side
                -m  This parameter allows you to split up the bigger grid into smaller ones,
                    the smaller this value, the more wall-dense the grid becomes
                -c  Number of capsules to add to the grid at random cells.
                -f  Save file name 

        EXAMPLE:    python layoutGenerator.py -g 20 -m 3 -c 3 -s randomClassic.lay
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", required=True, type=int, help="Size of the grid")
    parser.add_argument("-m", required=True, type=int, help="Size of the miniGrid. miniGrids make up the main grid")
    parser.add_argument("-c", type=int, help="Number of capsules to add to the grid", default=0)
    parser.add_argument("-f", type=str, help="Save file name")
    args = parser.parse_args()

    gridDimension = args.g
    miniGridDimension = args.m
    capsuleCount = args.c
    fileName = args.f

    # Ensures board symmetry
    if((gridDimension % 2) != 0):
        raise Exception("-g Grid dimension must be even!")
    if((gridDimension//2) < miniGridDimension):
        raise Exception("-m Mini Grid dimensions must be less than half grid dimension")
    if not fileName:
        raise Exception("-f Name of file needs to be specified")

    layoutGenerator = Grid(gridDimension, miniGridDimension, capsuleCount)
    layoutGenerator.writeToFile("./layouts/"+fileName)
