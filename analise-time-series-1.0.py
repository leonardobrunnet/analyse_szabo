import os
import sys

def verifica(pergunta,n_files):
    print(pergunta)
    while True :
        value  = sys.stdin.readline().split()
        try:
            value = int(value[0])
            return value
        except ValueError:
            print("Please, enter a number in interval 0-%d"%(n_files-1))

path="data/"   #files of interest are in data directory just below
files_in_dir=os.listdir(path) 
n_files=len(files_in_dir) #number of files and dirs in path
print("\n -> Found %d files\n"%n_files)
for i in range(n_files):
    print("%d - %s"%(i,files_in_dir[i]))
pergunta="\n -> Choose the file by the number in the list"
index=verifica(pergunta,n_files)
print("\n -> Analyzing file: %s\n"%files_in_dir[index])
input_file=open(path+files_in_dir[index])

## Here we look for the number of system snapshots
snapshot_counter = 0
while 1 :
    line = input_file.readline()
    if not line:
        break #EOF
    line_splitted = line.split()
    if line_splitted[0] == "TIME:" :
        snapshot_counter += 1
print(" -> Found %d snaphots. Analyzing..."%snapshot_counter)
input_file.close()
### You may improve the lines above to choose the snapshots you want

input_file=open(path+files_in_dir[index])
global_matrix=list([] for i in range(snapshot_counter))
counter = -1
while 1 :
    line = input_file.readline()
    if not line:
        break #EOF
    line_splitted = line.split()
    if line_splitted[0] == "TIME:" :
        counter += 1
        line = input_file.readline()  #here you may take the snapshotparameters
    else:
        line = input_file.readline()  #this is X Y X_Virtual...
        line_splitted = list(map(float,line.split()))
        global_matrix[counter].append(line_splitted)

#print all data after reading
for j in range(snapshot_counter):
    print("\n -> Snapshot %d\n"%j)
    for i in global_matrix[j] :
        print(i)

        




    
