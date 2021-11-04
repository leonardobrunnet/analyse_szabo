#This program enters the "data" directory reads file
#containing a series of snapshots of position and velocity of szabo particles.
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def init():
    path="data/"   #files of interest are in data directory just below
    ensure_dir(path)
    files_in_dir=os.listdir(path) 
    n_files=len(files_in_dir) #number of files and dirs in path
    if n_files == 0 :
        print("Please, put data files in directory 'data' and restart the program.")
        exit()
    else:
        print("\n -> Found %d files\n"%n_files)
    for i in range(n_files):
        print("%d - %s"%(i,files_in_dir[i]))
    question1="\n -> Choose the file by the number in the list"
    index=verifica(question1,n_files)
    print("\n -> Analyzing file: %s\n"%files_in_dir[index])
    input_file_name=path+files_in_dir[index]
    question2="Available measures:\n1- Vicsek parameter\n2- Display system snapshot images\n3- Cluster distribution \n4- Density distribution\n5- Phi distribution\n You may choose anyany group of integers in the interval 1..5.\n E.g. 1 2 or 3 4 5\n"
    measure=verifica2(question2)
    return(path,input_file_name,measure)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def verifica(question,n_files):
    print(question)
    while True :
        value  = sys.stdin.readline().split()
        if int(value[0]) > n_files -1 :
            print("Please, enter a number in interval 0-%d"%(n_files-1))
        if not value:
            print("Please, enter a number in interval 0-%d"%(n_files-1))

        else:
            return int(value[0])

def verifica2(question):
    print(question)
    while True :
        value  = sys.stdin.readline().split()
        try:
            test = int(value[0])
            value=list(map(int,value))
            return value
        except ValueError:
            print("You have not chosen. Exiting...")
            
            
def verifica3(question):
    print(question)
    while True :
        value  = sys.stdin.readline().split()
        if not value :
            return float(1.1)
        else:
            return float(value[0])

def verifica4(question):
    print(question)
    while True :
        value  = sys.stdin.readline().split()
        if not value:
            print("Value considered by default lbox=10")
            return int(10)
        else:
            return int(value[0])
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
    if snapshot_counter == 0 :
        print("->Found nothing to analyse.\n Exiting...")
    else:
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
    fig=plt.gcf()
    plt.show()
    figname="average_phi.png"
    fig.savefig(figname)
    return

def images(global_matrix, snapshot_counter):
    print("Show system snapshots? y or n")
    value  = sys.stdin.readline().split()
    if value[0] == "y":
        for i in range(snapshot_counter):
            x,y=[],[]
            for j in global_matrix[i] :
                x.append(j[0])
                y.append(j[1])
            plt.scatter(x,y,s=0.1)
            plt.show()
    # list(phimed_list.append(phimed) for i in range(snapshot_counter))
    # label="phi_av=%.4f"%phimed
    # plt.plot(x,phimed_list,label=label)
    # plt.xlabel("time")
    # plt.ylabel("phi")
    # plt.legend()
    # plt.show()
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
    fig=plt.gcf()
    plt.show()
    figname="density_distribution.png"
    fig.savefig(figname)
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
    fig = plt.gcf()
    plt.show()
    figname="phi_distribution.png"
    fig.savefig(figname)
    return

def distmat_square(X):
        #Xsq = (X**2).sum(axis=-1)
        #cross = X.matmul(X.T)
        #D = Xsq[:,None]+Xsq[None,:]-2*cross
        #D = torch.where(D < 0.00001*torch.ones(1,device=device), 0.00001*torch.ones(1,device=device),D)
        #return torch.sum(bc_diff(X[:,None,:]-X[None,:,:])**2,axis=2)
        
        D = torch.sum((X[:,None,:]-X[None,:,:])**2,axis=2)
        return D
def neighbor_list(X,dviz):
    #Enter the particle's position, return a list of lists of neighbors
    D=torch.sqrt(distmat_square(X))
    #define zero and one tensors
    zero_tensor=torch.zeros(number_particles,number_particles,dtype=int)
    one_tensor = torch.ones(number_particles,number_particles,dtype=int)
    #put 1 when you find a neighbor
    viz=torch.where(D<dviz,one_tensor,zero_tensor)
    index=torch.tensor(np.arange(1,number_particles+1),dtype=int)
    #put the particle index relative to each 1.
    viz=viz[:,:]*index[None,:]
    #convert to list
    viz=viz.tolist()
    #eliminate the zeros
    part_list=[]
    for i,w in enumerate(viz):
        aux=[]
        for j in w:
            if j >0 :
                aux.append(j-1)
        part_list.append(aux)
    return part_list



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
        self.r=np.array([x,y])
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
    #a list for each snapshot
    def __init__(self,index,snapshot_counter):
        self.index = index
        self.mylist = list( [] for i in range(snapshot_counter))
        return

def part_and_box(size,snapshot_counter,number_particles,glob_array):
    part=list(particle(i,glob_array[:,i,0],glob_array[:,i,1],glob_array[:,i,4],glob_array[:,i,5],glob_array[:,i,6]) for i in range(number_particles))
    
    #define the box size used to analyse
    question4="Enter the box size to make averages. Default is 10."
    lbox = verifica4(question4)
    nx=int(size/lbox)+1
    nt=nx**2

    #define the boxes as an object list                                                       
    box=list(boite(i,snapshot_counter) for i in range(nt))

    #Discover the boxes each particle passes and put in an array
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
    
    return nt,lbox,box,part

def cluster_dist(glob_array,number_particles):
    question3="\nEnter the distance to consider two particles as neighbors - Default = 1.1\n"
    dviz=verifica3(question3)
    
    #Define torch tensor for particle positions in the last image
    X=torch.tensor([[glob_array[-1,i,0],glob_array[-1,i,1]] for i in range(number_particles)])
    #max dist to consider neighbor

    part_list=neighbor_list(X,dviz)

    for i in range(number_particles):
        for j in part_list[i]:
            if j != i:
                part_list[i]+=part_list[j]
                part_list[i]=list(set(part_list[i]))
                part_list[j]=[]
        #print(part_list[i])
    clust_dist=np.zeros(number_particles+1)
    for i in part_list :
        if not i:
            continue
        else:
            #print(i,len(i))
            clust_dist[len(i)]+=1
    xmin,xmax=0,number_particles+1
    for i,w in enumerate(clust_dist):
        if w > 0 :
            xmin = i
            break
    for i,w in enumerate(reversed(clust_dist)):
        if w > 0 :
            #print(i,w)
            xmax = len(clust_dist)-i
            break

    #print(xmin,xmax)
    ans="y"
    xlim=[xmin,xmax]
    histo_dens(clust_dist,xlim,ans)
    print("Reset x limits? y or n")
    ans = sys.stdin.readline().split()[0]
    
    while ans == "y" :
        print("Enter new min/max limits for x. E.g. 2 20. (min=1;max=%d)"%(number_particles+1))
        xlim=list(map(int,sys.stdin.readline().split()))
        histo_dens(clust_dist,xlim,ans)
        print("Reset x limit? y or n")
        ans = sys.stdin.readline().split()[0]
    histo_dens(clust_dist,xlim,ans)
    return


def histo_dens(clust_dist,xlim,ans):
#    fig, ax = plt.subplots()
    x=np.arange(xlim[0],xlim[1])
    plt.bar(x,clust_dist[xlim[0]:xlim[1]])
    gamma=0
    norm =0
    for i,v in enumerate(clust_dist):
        gamma+=i**2*v
        norm+=i*v
    gamma=np.sqrt(gamma)/norm
    #print(gamma,norm)
#    sns.distplot(clust_dist, hist=True, ax=ax,  kde=True, color='#1F78B4', bins=20)
#ax.set_xlim(1,xlim)
    plt.title("Cluster distribution - $\gamma$=%.3f"%gamma)
    #ax.set_xticks(range(1,32))
    if ans == "y" :
        plt.show()
    if ans == "n" :
        figname="cluster_distribution.png"
        plt.savefig(figname)

    #legend=("Cluster distribution = {0:.3f} ".format(number_particles/size**2))
    #sns.distplot(clust_dist, hist =True, bins=20, kde=True, color='#1F78B4', range=rangex)#, label=legend)
    #plt.legend()
    #plt.show()
    return
    
#main program
path,input_file_name,measure=init()

## Here we look for the number of system snapshots and system size
snapshot_counter,size,v0=snapshot_count(input_file_name)

#Reopen file, read all data and put it on global_matrix
global_matrix=construct_global_matrix(input_file_name,snapshot_counter)

#print all data after reading
#printall(global_matrix)

glob_array=np.array(global_matrix) #I suppose the number of particles will not vary, so I trasform to numpy array
snapshot_counter,number_particles,dyn_var=glob_array.shape


#calculus of vicsek order parameter and graphics at each snapshot
if 1 in measure :
    phi_calc(global_matrix,snapshot_counter)

#msd calculus
# if 2 in measure :
#     msd(global_matrix,snapshot_counter)

#show images
if 2 in measure :
    images(global_matrix, snapshot_counter)

#Cluster distribution
if 3 in measure :

    cluster_dist(glob_array,number_particles)


if 4 in measure or 5 in measure :

    #construct particle class and box class; each particle has arrays of x,y,vx,vy with values for all snapshots
    nt,lbox,box,part=part_and_box(size,snapshot_counter,number_particles,glob_array)

    #calculate particle density distribution averaged over snapshots
    if 4 in measure :
        density_distribution(nt,snapshot_counter,number_particles)
    #phi distribution averaged over snapshots
    if 5 in measure :
        phi_time_average(nt,snapshot_counter)



