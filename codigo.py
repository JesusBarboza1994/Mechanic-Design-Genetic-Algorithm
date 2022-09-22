import random
import numpy as np
inicio1 = 1
inicio = 1
fin = 1
intermitencia =1
inicio2 = 1
inicion = 1
fin1 = 1
fin2= 1
finn = 1
(intermitencia1) = 1
(intermitencia2) = 1
(intermitencian) = 1
prop11=1
prop12=1
prop1j=1
prop21=1
prop22=1
prop2j=1
propi1=1
propi2=1
propij=1
valor1=1
valor2=1
valor31=1
valor32=1
valork =21
rangon=1
variable1 =1
variable2=1
variablen=1
ip1=1
ip2=1
ipi=1
op1=1
op2=2
opj=1


def parametrosIniciales():
    rango1 = [inicio1, fin1, variable1, (intermitencia1)]
    rango2 = [inicio2, fin2, variable2, (intermitencia2)]
    ...
    rangon = [inicion, finn, variablen, (intermitencian)]
    rangos =[]
    rangos.append(rango1)
    rangos.append(rango2)
    ...
    rangos.append(rangon)
    return rangos

def materiales():
    m = [["Material1", prop11, prop12, ..., prop1j],
        ["Material2", prop21, prop22, ..., prop2j],
        ...
        ["Materiali", propi1, propi2, ..., propij]]
     
    #i: Número de materiales a analizar
    #j: Número de propiedades de cada material 
    return m

def requerimientos():
    req = []
    req1 = valor1
    req2 = valor2
    req3 = [valor31, valor32]
    ...
    reqk = valork
    req.append(req1)
    req.append(req2)
    req.append(req3)
    ...
    req.append(reqk)

    return req




#Si es entero
random.randint(inicio,fin) 

#Si es decimal
random.uniform(inicio,fin) 

#Si es valor de una lista
random.choice(rangon) 

#Si es intermitente
random.randint(intermitencia*inicio, intermitencia*fin)/intermitencia


def aleatorio(parInicial):
  
    RandAl = random.randint(parInicial[0]*2,parInicial[1]*2)/2
    RandDext = random.randint(parInicial[2],parInicial[3])
    RandN = random.randint(8*parInicial[4],8*parInicial[5])/8
    rangoLo = random.randint(parInicial[6],parInicial[7])
    material = int(random.randint(0,len(materiales())-1))

    return RandAl,RandDext, RandN, rangoLo, material

def aleatorio():
    rand = []
    rangos = parametrosIniciales()
    for i in range(len(rangos)):
        if(rangos[i][2])==0:
            rand.append(random.randint(rangos[i][0],rangos[i][1]))
        elif(rangos[i][2])==1:
            rand.append(random.uniform(rangos[i][0],rangos[i][1]))
        elif(rangos[i][2])==2:
            rand.append(random.choice(rangos[i][0]))
        elif(rangos[i][2])==3:
            rand.append(random.randint(rangos[i][0]/rangos[i][3],
            rangos[i][1]/rangos[i][3])*rangos[i][3])
    
    rand.append(int(random.randint(0,len(materiales())-1)))
    return rand

def diseno(a,dext,nt,lo,i):
       #Declarar condiciones iniciales
    li = requerimientos()[0][0]
    fio = requerimientos()[0][1]
    ko = requerimientos()[1]   
    #ty = 1200
    #G = 78500
    #Fórmulas
    i = int(i)
    G = int(materiales()[i][4]*1000) #MPa  
    ty = materiales()[i][1]/(a**materiales()[i][2])
    dm = dext - a
    n = nt - 1.5
    k = G*a**4/(8*n*dm**3) #N/mm
    c = dm/a
    
    lm = lo - a
    p = lm/n
    xc = (p-a)*n
    fc = xc*k
    ans = parametrosIniciales()
    while fc < fio or lo < li:      
       # a = random.randint(ans[0]*2,ans[1]*2)/2
        dext = random.randint(ans[2],ans[3])
        nt = random.randint(8*ans[4],8*ans[5])/8
        lo = random.randint(ans[6],ans[7])
        dm = dext - a
        n = nt - 1.5
        c = dm/a
        k = G*a**4/(8*n*dm**3) #N/mm
        lm = lo - a
        p = lm/n
        xc = (p-a)*n
        fc = xc*k
    wc = (4*c-1)/(4*c-4)+0.615/c
    tc = 8*dm*fc/(pi*a**3)*wc
    lda = dm*pi*nt+200
    peso = lda*pi/4*a**2*materiales()[i][3]
    #peso = (a/12.7)**2*lda/1000

#Se debe poner restricciones por crear solidos que no se pueden fabricar (Ejem: Paso negativo)
    if p<=0:
        k = 0.001
    
    if p <= a:
        tc = 10000
    
    return k,ko,lo,li,fc,fio,ty,tc,peso,a,dext,nt



def diseno(solution):
    req = requerimientos()
    material = materiales()   
    parinit = parametrosIniciales()
    
    #Fórmulas paramétricas
    ...
    ...
    ...

    #Datos de salida
    output = [op1, op2, ..., opj]
    #j es el numero de outputs
    input = [ip1, ip2, ..., ipi]
    #i es el numero de inputs

    return  input, req, output

    material
    parinit
    solution
i=1
j=1

def fiti(inputs):
    datos = diseno(inputs)
    reqi = datos[1][i] #requerimiento asociado a funcion fitness i
    opj = datos[2][j] #output asociado al requerimiento i
    ei = abs((opj-reqi)/reqi) #error absoluto calculado

    return ei

def fit1(a,dext,nt,lo,i): #error en rigidez
    ans = diseno(a,dext,nt,lo,i)
    ko = ans[1]
    ek = abs((ans[0]-ko)/ko)

    return ek

def fit2(inputs):
    return 1
 
w1=1
w2=1
wi=1

def fitness(inputs):
    fitness = []
    fitness.append(fit1(inputs))
    fitness.append(fit2(inputs))
    ...
    fitness.append(fiti(inputs))
    
    w=[w1, w2, ..., wi]
    
    fit=1
    for i in range(len(fitness)):
        fit = fit*fitness[i]**w[i]
    
    return fit

def initialPopulation():
    solutions = []
    for s in range(pobinit): #Aca se debe definir extremos de cada variable
        solutions.append(aleatorio())
    solutions = np.array(solutions)
    return solutions

#-------------------------------------------------------------------


pobinit=1

#Filtro de soluciones  
def filterSolution(solutions):
    bool =[]
    for s in range(pobinit):
        filtro= diseno(solutions[s])              
        filtro = np.array(filtro)
        for i in range(len(solutions[s])):
            solutions[s][0] = filtro[0][i]
        
        for j in range(s):
            bool1 = all(solutions[s] == solutions[j])
            if bool1:
                bool.append([bool1,s])
                for i in range(len(aleatorio())-1):
                    solutions[s][i] = aleatorio()[i]
               
                filtro= diseno(solutions[s])
                for i in range(len(solutions[s])):
                    solutions[s][0] = filtro[0][i]
    return solutions

bestpob=1

#Selección
def selection(solutions):
    rankedsolutions = []
    for s in solutions:
        rankedsolutions.append(fitness(s),
                                  fit1(s), 
                                  fit2(s),
                                  ...,
                                  fiti(s),
                                       s)
                
    rankedsolutions.sort() 
    bestsolutions = rankedsolutions[:int(bestpob)]

    return bestsolutions

nval=1
e1=1
e2=1
ej=1

#Error máximo permitido
def errorPermitido():
    e = [e1,e2,...,ej]
    #j es el número de funciones 
    # fitness con error máximo permitido.
    return e


def stopCondition(bestsolutions):
    npbestsol = np.array(bestsolutions)
    e = errorPermitido()
    det = 1
    for i in range(len(e)):
        det = det*(max(npbestsol[:,i])<e[i])

    if det:
        print("Se interrumpio algoritmo: ")
        print(*bestsolutions, sep = "\n")
        
        return True

p_cruce=2

#Cruce
def crossOver(bestsolutions):
    f = i+1 #i es el numero de funciones fitness
    #se suma uno debido al espacio de la funcion fitness global
    crosSolution = []
    for i in range(int(bestpob/2)):
        a = bestsolutions[2*i][f:len(bestsolutions[2*i])] 
        b = bestsolutions[2*i+1][f:len(bestsolutions[2*i+1])] 
        separ = random.randint(1,len(a)-1)
        sepran = random.random()
    
        if p_cruce < sepran:
            x = []
            x.append(b[:separ])
            x[0] = x[0] + a[separ:]
            y = []
            y.append(a[:separ])
            y[0] =y[0] + b[separ:]
            crosSolution.append(x[0])
            crosSolution.append(y[0])
        else:
            crosSolution.append(a)
            crosSolution.append(b)

    return crosSolution

p_mutacion =1

#Mutación
def mutate(crosSolution):
    mutSolution = []
    for i in crosSolution:
        mutran = random.random()
        mut = random.randint(0,len(crosSolution[0])-1)
        
        if mutran > p_mutacion:           
            varmut = aleatorio()[mut]
            j=[]
            j = i[:mut]
            j = np.insert(j,len(j),varmut)
            j = np.concatenate([j,i[mut+1:]])
            mutSolution.append(j)
        else:
            mutSolution.append(i)

    return mutSolution


#Nueva generación
def newSolution(mutSolution, bestSolution):
    newGen = []
    f = i+1 #i es el numero de funciones fitness
    #se suma uno debido al espacio de la funcion fitness global
    for i in mutSolution:
        newGen.append(np.asarray(i))
    
    for i in bestSolution:
        newGen.append(np.asarray(i[5:10]))
    newGen = np.array(newGen)
    
    newGen2 = []
    n = 0
    for s in newGen:
        n = n+1
        newGen2.append((fitness(s),
                           fit1(s),
                           fit2(s),
                               ...,
                           fiti(s),
                                s))

    newGen2.sort()
    newGen2 = np.array(newGen2)
    solutions = newGen2[:pobinit,f:]  

    return solutions