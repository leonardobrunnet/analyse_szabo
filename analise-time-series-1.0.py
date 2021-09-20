#This program enters the "data" directory reads file
#containing a series of snapshots of position and velocity of szabo particles.
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
            size=int(line_splitted[20].split(".")[0])
            v0=float(line_splitted[23])
            snapshot_counter += 1
    print(" -> Found %d snaphots. Analyzing..."%snapshot_counter)
    input_file.close()
    ### You may improve the lines above to choose the snapshots you want
    return snapshot_counter,size,v0



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
    phi = np.array(phi)
    phimed = np.mean(phi)
    phistd = np.std(phi)
    print("Average phi=%.3f, standard dev=%.4f"%(phimed,phistd))
    plt.scatter(x,phi)
    phimed_list=[]
    list(phimed_list.append(phimed) for i in range(snapshot_counter))
    label="phi_av=%.4f"%phimed
    plt.plot(x,phimed_list,label=label)
    plt.xlabel("time")
    plt.ylabel("phi")
    plt.legend()
    plt.show()
    return

def density_distribution(nt,snapshot_counter,number_particles):
    #calculate average particle density along the space
    rho_av = np.zeros(nt)
    for i in range(snapshot_counter):
        for j in box:
            rho_av[j.index]+=len(j.mylist[i])

    rho_av/=(lbox**2*snapshot_counter)
    legend=("Average density = {0:.3f} ".format(number_particles/size**2))
    sns.distplot(rho_av, hist =True, bins=20, kde=True, color='#1F78B4', label=legend)
    plt.title("Density distribution")
    plt.legend()
    plt.show()
    return

def phi_time_average(nt,snapshot_counter):
    phi_av_time=np.zeros(nt)
    for i in range(snapshot_counter):
        for j in box:
            vx,vy=0,0
            for k in j.mylist[i]:
                vx+=part[k].vx[i]
                vy+=part[k].vy[i]
            phi_av_time[j.index]+=np.sqrt(vx**2+vy**2)/(v0*len(j.mylist[i]))

    phi_av_time/=snapshot_counter
    phi_av_time_space=np.sum(phi_av_time)/nt
    legend=(r"Average $\phi$ = {0:.3f} ".format(phi_av_time_space))
    sns.distplot(phi_av_time, hist=True, bins=20, kde=True, color='#1F78B4', label=legend)
    plt.title(r" $\phi$ distribution")
    plt.legend()
    plt.show()
    return




# def msd(global_matrix,snapshot_counter):
#     glob=np.array(global_matrix)
#     for i in range(snapshot_counter - 1 ):
#         for j in range(glob[i,:].size):
#             print((glob[i+1][2]-glob[i][2])**2+(glob[i+1][3]-glob[i][3])**2)
#     return

#Particle class definition
class particle:
    def __init__(self, index, x, y, vx, vy, theta ):
        self.index = index
        self.x=x   #these are np arrays for all snapshots of each particle
        self.y=y
        self.vx=vx
        self.vy=vy
        self.theta=theta
        self.Mybox=[]

    def mybox(self,lbox,nx): #each particle calculates the box it is in at the different snaphots
        aux=(np.array(list(map(int,self.x[:]/lbox)))+nx*np.array(list(map(int,self.y[:]/lbox))))
        self.Mybox=list(aux)
        return 
                
#Box class definition
class boite:
    def __init__(self,index,snapshot_counter):
        self.index = index
        self.mylist = list( [] for i in range(snapshot_counter))
        return

#main program
path,input_file_name=init()

## Here we look for the number of system snapshots and system size
snapshot_counter,size,v0=snapshot_count(input_file_name)

#define the box size used to analyse
lbox=10
nx=int(size/lbox)+1
nt=nx**2


#Reopen file, read all data and put it on global_matrix
global_matrix=construct_global_matrix(input_file_name,snapshot_counter)

#print all data after reading
#printall(global_matrix)
        
#calculus of vicsek order parameter and graphics at each snapshot
#phi_calc(global_matrix,snapshot_counter)

#msd calculus
#msd(global_matrix,snapshot_counter)


glob_array=np.array(global_matrix) #I suppose the number of particles will not vary, so I trasform to numpy array

#construct particle class and box class; each particle has arrays of x,y,vx,vy with values for all snapshots
snapshot_counter,number_particles,dyn_var=glob_array.shape
part=list(particle(i,glob_array[:,i,0],glob_array[:,i,1],glob_array[:,i,4],glob_array[:,i,5],glob_array[:,i,6]) for i in range(number_particles))


#define the boxes as an object list                                                       
box=list(boite(i,snapshot_counter) for i in range(nt))

#Discover the boxes each particle passes and put in an array
#map(lambda i:i.mybox(lbox,nx), part)
for i in part:
    i.mybox(lbox,nx)

#Construct the particle population on each box at each time by particle index
for i in range(snapshot_counter):
    for j in part :
        k=int(j.Mybox[i])
        if k > nt :
            print("No box for this particle! Check the coordinates and system size.")
            print(k,j.Mybox[i],nt)
            exit()
        else:
            box[k].mylist[i].append(j.index)
            
#calculate particle density distribution averaged over snapshots
#density_distribution(nt,snapshot_counter,number_particles)



#phi distribution averaged over snapshots
phi_time_average(nt,snapshot_counter)
