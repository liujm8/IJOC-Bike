import numpy as np
from gurobipy import *
import networkx as nx
import csv

def distance_sphere(row,row1):
    (lat1,lng1,lat2,lng2)=(row[0],row[1],row1[0],row1[1])
    degrees_to_radians = math.pi/180.0
    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
    # theta = longitude
    theta1 = lng1*degrees_to_radians
    theta2 = lng2*degrees_to_radians
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))
    if cos>=1:
        arc=0
    else:
        arc = math.acos( cos )
    return arc*3960 #return mile

# Callback - use lazy constraints to eliminate sub-tours
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        print('yes start')
        selected_length=[0 for v in range(veh_n)]
        selected = [[] for v in range(veh_n)]
        # make a list of edges selected in the solution

        for v in range(veh_n):
            selected[v] += [(i,j) for i in range(-2,n) for j in range(-2,n) if i!=j and model.cbGetSolution(model._vars[v,i,j]) > 0.5]
            print('selected',v,selected[v])

        nodes=[[] for v in range(veh_n)]
        for v in range(veh_n):
            for (i,j) in selected[v]:
                if i not in nodes[v]:
                    nodes[v].append(i)
                if j not in nodes[v]:
                    nodes[v].append(j)
        for v in range(veh_n):
            selected_length[v]=len(nodes[v])

            # find the shortest cycle in the selected edge list
        for v in range(veh_n):
            if len(selected[v])>0:
                print('yes1',selected[v],selected_length[v])
                tours = subtour(selected[v],selected_length[v])
                print('yes2','tour',tours)
                print(selected_length[v])
                for tour in tours:
                    if -1 not in tour:
                        if len(tour) < selected_length[v]:
                            print(v,'I am adding SEC######################################################')
                            # add a subtour elimination constraint
                            expr = 0
                            for i in range(len(tour)-1):
                                expr += model._vars[v,tour[i], tour[i+1]]
                            expr += model._vars[v,tour[-1], tour[0]]
                                    # print 'expr',expr
                            model.cbLazy(expr <= len(tour)-1)


# Euclidean distance between two points

def distance(points, i, j):
    dx = points[i][0] - points[j][0]
    dy = points[i][1] - points[j][1]
    return math.sqrt(dx*dx + dy*dy)


# Given a list of edges, finds the shortest subtour

def subtour(edges,n1):
    # print 'n1',n1
    visited = {}
    for i,j in edges:
        if i not in visited:
            visited[i]=False
        if j not in visited:
            visited[j]=False
    cycles = []
    lengths = []
    selected = {}
    for x, y in edges:
        selected[x]=y
    # print 'subtour',selected
    while True:
        if visited[-1]==False:
            current=-1
        else:
            for nod in visited:
                if visited[nod]==False:
                    current=nod
                    break

        thiscycle = [current]
        while True:
            visited[current] = True
            neighbors = [selected[current]]
            # print 'neighbors',current,neighbors
            if visited[neighbors[0]] == True:
                break
            current = neighbors[0]
            thiscycle.append(current)
        print('thiscycle',thiscycle)
        cycles.append(thiscycle)
        lengths.append(len(thiscycle))
        if sum(lengths) == n1:
            break
    print('cycles',cycles)
    if len(cycles)>1:
        return cycles
    else:
        return cycles

problem_size=[]
time_cost=[]

c=5
veh_n=2
lamb=0.5

G=nx.Graph()
file_name='Case_15C.csv'
color_list=['black','blue','red','green','gray','lime','purple','pink','orange','plum','tomato','silver','brown','springgreen','yellowgreen']
station=np.loadtxt('../data/3-SmallCase/'+file_name,delimiter=',',skiprows=1)
i=0

for row in station:
    G.add_node(i)
    G.nodes[i]['id']=i
    G.nodes[i]['lat']=row[1]
    G.nodes[i]['lng']=row[2]
    G.nodes[i]['target']=row[3]
    G.nodes[i]['class']=row[c]
    G.nodes[i]['available']=row[8]
    G.nodes[i]['dock']=row[9]
    G.nodes[i]['next']=[]
    i += 1

unique_class=np.unique(station[:,c])
if 0 in unique_class:
    class_num=len(unique_class)-1
else:
    class_num=len(unique_class)

print(class_num, 'number of classes')
classes={}
complete_class=[]
for row in unique_class:
    if row!=0:
        classes[row]=[]
print(file_name+str(veh_n))
for row in G.nodes():
    if G.nodes[row]['class']!=0 and G.nodes[row]['target']!=0:
        classes[G.nodes[row]['class']].append(G.nodes[row])
        complete_class.append(row)
jj=1
ff=0
points={}
target={}
available={}
dock={}
mm=1000000

i=0
for rows in classes[23]:
    points[i]=([rows['lat'],rows['lng']])
    target[i]=rows['target']
    available[i]=rows['available']
    dock[i]=rows['dock']
    i+=1

depot=[40.716629, -73.982616]
n = len(points)
print(len(target), target)
target[-1]=0
target[-2]=0
points[-1]=depot
points[-2]=depot
# start=time.clock()
m = Model()
# Create variables
y={}
vars = {}
U={}
s={}
U[-1]=0
U[-2]=0
for i in range(-2,n):
    for v in range(veh_n):
        y[v,i] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="Y"+str(v)+str(i))
        U[i]= m.addVar(obj=0,vtype=GRB.CONTINUOUS,name="U"+str(i))
        s[v,i]=m.addVar(obj=0,vtype=GRB.CONTINUOUS,lb=-26.0,name="S"+str(v)+str(i))
        for j in range(-2,n):
            vars[v,i,j] = m.addVar(obj=0, vtype=GRB.BINARY,
                                         name='e'+str(i)+'_'+str(j))

m.update()

for v in range(v):
    for i in range(-2,n):
        m.addConstr(vars[v,i,i]==0)


expr=LinExpr()
for v in range(veh_n):
    for i in range(-2,n):
        for j in range(-2,n):
            if i!=j:
                expr.addTerms(distance_sphere(points[i], points[j]),vars[v,i,j])

for i in range(n):
    expr.addTerms(lamb,U[i])

m.setObjective(expr,GRB.MINIMIZE)

m.update()

###constrain U[i] (4)(5)
for i in range(n):
    m.addConstr(quicksum(s[v,i] for v in range(veh_n))-target[i], GRB.LESS_EQUAL,U[i])
    m.addConstr(target[i]-quicksum(s[v,i] for v in range(veh_n)),GRB.LESS_EQUAL,U[i])

###constrain flow, (6),(7)
for v in range(veh_n):
    for i in range(-2,n):
        for j in range(-2,n):
            # if i!=j:
            m.addConstr(y[v,i]+s[v,j]-y[v,j]+(1-vars[v,i,j])*mm,GRB.GREATER_EQUAL,0)
            m.addConstr(y[v,i]+s[v,j]-y[v,j]-(1-vars[v,i,j])*mm,GRB.LESS_EQUAL,0)

for v in range(veh_n):
    for i in range(n):
        m.addConstr(s[v, i] + mm * quicksum(vars[v, i, j] for j in range(n)), GRB.GREATER_EQUAL, 0)
        m.addConstr(s[v, i] - mm * quicksum(vars[v, i, j] for j in range(n)), GRB.LESS_EQUAL, 0)

###constrain vehicle capacity (8)
for i in range(-2,n):
    for v in range(veh_n):
        m.addConstr(y[v,i],GRB.LESS_EQUAL,25)
        m.addConstr(y[v,i],GRB.GREATER_EQUAL,0)

###constrain continuity (9)
# for i in range(n):
#     m.addConstr(quicksum(vars[v,i,j] for v in range(veh_n) for j in range(-2,n) if i!=j),GRB.LESS_EQUAL,1)
#     m.addConstr(quicksum(vars[v,j,i] for v in range(veh_n) for j in range(-2,n) if i != j),GRB.LESS_EQUAL,1)
###constrain 1 visit (10)
# for j in range(n):
#     m.addConstr(quicksum(vars[v,i,j] for v in range(veh_n) for i in range(-2,n))==1)

###constrain Inventory (11)
for i in range(n):
    for v in range(veh_n):
        print(G.nodes[i]['available'],G.nodes[i]['dock'])
        m.addConstr(s[v,i], GRB.LESS_EQUAL, available[i])
        m.addConstr(s[v,i],GRB.GREATER_EQUAL,(available[i]-dock[i]))


for v in range(veh_n):
    for i in range(n):
        m.addConstr(quicksum(vars[v,i,j] for j in range(-2,n) if j!=i and j!=-1)==quicksum(vars[v,j,i] for j in range(-2,n) if j!=i))

for v in range(veh_n):
    m.addConstr(quicksum(vars[v,-1,j] for j in range(n))==1)
    m.addConstr(quicksum(vars[v,j,-2] for j in range(n))==1)
    for i in range(n):
        m.addConstr(quicksum(vars[v,i,j] for j in range(n)), GRB.LESS_EQUAL,1)

    # m.addConstr(quicksum(vars[v,-2,j] for j in range(n))==0)
    m.addConstr(vars[v,-2,-1]==1)

# m.addConstr(quicksum(vars[0,i,j] for i in range(-2,n) for j in range(-2,n) if i!=j)==7)
ff+=1
m.update()


# Optimize model

m._vars = vars
m.params.LazyConstraints = 1
m.optimize(subtourelim)
# end=time.clock()
solution = m.getAttr('x', vars)
# time_cost.append(end-start)
problem_size.append(n)
selected = [(v,i,j) for i in range(-2,n) for j in range(-2,n) for v in range(veh_n) if i!=j and vars[v,i,j].x > 0.5]
# print('running time: %g' %(end-start_time))
print('')
print('Optimal cost: %g' % m.objVal)


#############################################Output Result
solution = m.getAttr('x', vars)
selected = [(v,i,j) for i in range(-2,n) for j in range(-2,n) for v in range(veh_n) if i!=j and solution[v,i,j] > 0.5]

# for row in selected:
#     print(row,s[row[0],row[1]],U[row[1]],y[row[0],row[1]])

writer=csv.writer(open('../results/SmallCase-result-MILP-multivisit/'+'result_'+str(veh_n)+'_'+file_name,'w',newline=''))

unsatisfied=lamb*sum([U[i].x for i in range(n)])
print('unsatisfied penalty:',unsatisfied)
print('Wall Clock time',m.Runtime)
writer.writerow(['obj:',m.objVal])
writer.writerow(['unsatisfied:',unsatisfied])
writer.writerow(['travel distance:',m.objVal-unsatisfied])
writer.writerow(['clock time',m.Runtime])

writer.writerow(['v','i','j','Laitude1','Longitude1','Latitude2','Longitude2','yvi','yvj','targetj','Ui','Uj','svi','svj'])
for row in selected:
    if row[1]>=0 and row[2]>=0:
        writer.writerow([row[0],row[1],row[2],points[row[1]][0],points[row[1]][1],points[row[2]][0],points[row[2]][1],y[row[0],row[1]].x,y[row[0],row[2]].x,target[row[2]],U[row[1]].x,U[row[2]].x,s[row[0],row[1]].x,s[row[0],row[2]].x])
    else:
        writer.writerow([row[0],row[1],row[2],points[row[1]][0],points[row[1]][1],points[row[2]][0],points[row[2]][1],y[row[0],row[1]].x,y[row[0],row[2]].x,target[row[2]],0,0,s[row[0],row[1]].x,s[row[0],row[2]].x])

