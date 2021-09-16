#This program enters the "data" directory reads file
#containing a series of snapshots of position and velocity of szabo particles.
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def init():
    path="data/"   #files of interest are in data directory just below
    files_in_dir=os.listdir(path) 
    n_files=len(files_in_dir) #number of files and dirs in path
    print("\n -> Found %d files\n"%n_files)
    for i in range(n_files):
        print("%d - %s"%(i,files_in_dir[i]))
    pergunta="\n -> Choose the file by the number in the list"
    index=verifica(pergunta,n_files)
    print("\n -> Analyzing file: %s\n"%files_in_dir[index])
    input_file_name=path+files_in_dir[index]
    return(path,input_file_name)





def verifica(pergunta,n_files):
    print(pergunta)
    while True :
        value  = sys.stdin.readline().split()
        try:
            value = int(value[0])
            return value
        except ValueError:
            print("Please, enter a number in interval 0-%d"%(n_files-1))

def snapshot_count(input_file_name):
    input_file=open(input_file_name)
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
    return snapshot_counter



def construct_global_matrix(input_file_name,snapshot_counter):
    input_file=open(input_file_name)
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
            line = input_file.readline()  #this is X Y X_Virtual...
        else:
            line_splitted = list(map(float,line.split()))
            global_matrix[counter].append(line_splitted)
    return global_matrix

def printall(global_matrix) :
    for j in range(snapshot_counter):
        print("\n -> Snapshot %d\n"%j)
        for i in global_matrix[j] :
            print(i)



def phi_calc(global_matrix, snapshot_counter):
    phi=[]
    for i in range(snapshot_counter):
        vx,vy,num=0,0,0
        for j in global_matrix[i] :
            vx+=j[4]
            vy+=j[5]
            num+=1
        phi.append(np.sqrt(vx**2 + vy**2)/num)
    x = np.arange(snapshot_counter)
    plt.scatter(x,phi)
    plt.show()
    return


#main program
path,input_file_name=init()

## Here we look for the number of system snapshots
snapshot_counter=snapshot_count(input_file_name)

#Reopen file, read all data and put it on global_matrix
global_matrix=construct_global_matrix(input_file_name,snapshot_counter)

#print all data after reading
#printall(global_matrix)
        
#calculo do phi do vicseck e gráfico por snapshot
phi_calc(global_matrix,snapshot_counter)

    
