import random

f = open("hard.txt", "a")
f.write(str(500))
f.write("\n")

mat = [[0 for column in range(500)] for row in range(500)]
for i in range(500):
    for j in range(499):
        mat[i][j] = random.randint(10, 200)
        f.write(str(mat[i][j]) + ",")
    mat[i][499] = random.randint(10, 200)
    f.write(str(mat[i][j]))
    f.write("\n")

f.close()