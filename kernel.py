import numpy as np
import pylab as plt
import element as elAssemb
import copy
from scipy.sparse import coo_matrix
import scipy.sparse.linalg as slinalg #import norm
import scipy.linalg as scplinalg #import norm

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D

GLOBAL_numberOfDiagonalStrips = 7
GLOBAL_materialRatio = 8
GLOBAL_zeroPivot = 1e-8
GLOBAL_BC_TYPE = 2
GLOBAL_GC_MATRIX = "HFETI_REDUNDANT"  # "HFETI" "HFETI_REDUNDANT"  "ECONOMIC"
#GLOBAL_GC_MATRIX = "HFETI"

###############################################
DBG=2
GLOBAL_cnt  = 0

GLOBAL_nDOFsOneNode = 2


def callAssemblerLocalMatrix(E,mu,rho,xCoord, yCoord):

    if GLOBAL_nDOFsOneNode == 1:
        Ex = Ey = 1.0
        Ke = elAssemb.elem2d4nHeat(Ex,Ey,xCoord ,yCoord,[GLOBAL_materialRatio, GLOBAL_numberOfDiagonalStrips, GLOBAL_BC_TYPE])
    else:
        Ke = elAssemb.elem2d4n(E,mu,rho,xCoord ,yCoord,[GLOBAL_materialRatio, GLOBAL_numberOfDiagonalStrips, GLOBAL_BC_TYPE])

    return Ke

def mesh(inputs):

    exOnSub = inputs[0]
    eyOnSub = inputs[1]
    sx = inputs[2]
    sy = inputs[3]
    elemType = inputs[4]

    if elemType == '3nodes':
        print("ERROR: ASSEMBLER NOT IMPLEMENTED")
        return []
        isDoubled = 2
    elif elemType == '4nodes':
        isDoubled = 1
    else:
        return []

    exAll = exOnSub * sx
    eyAll = eyOnSub * sy

    hx = 1.0 / exAll
    hy = 1.0 / eyAll

    meanL = (1.0 + 1.0) * 0.5

    nxAll = exAll + 1
    nyAll = eyAll + 1

    nxOnSub = exOnSub + 1
    nyOnSub = eyOnSub + 1

    nnodsAll = nxAll * nyAll
    nelemAll = isDoubled * exAll * eyAll 

    coordinates = np.zeros((nnodsAll,2))
    cntElem = 0
    for iy in range(nyAll):
        for ix in range(nyAll):
            coordinates[cntElem,:] = np.array([ix * hx, iy*hy])
            cntElem += 1

    subId = np.zeros((nelemAll),dtype=int)
    elements = np.zeros((nelemAll,3 + (2-isDoubled)),dtype=int)
    cntElem = 0
    cntSubId = 0
    nodesGrid = np.reshape(np.arange(nxAll*nyAll),(nyAll,nxAll)).T
    elementsGrid = np.reshape(np.arange(exAll*eyAll),(eyAll,exAll)).T

    for _sy in range(sy):
        for _sx in range(sx):
            for _ey in range(eyOnSub):
                for _ex in range(exOnSub):
                    I = _sx * exOnSub + _ex
                    J = _sy * eyOnSub + _ey
                    P00 = I + 0 + (J + 0) * nxAll
                    P10 = I + 1 + (J + 0) * nxAll
                    P11 = I + 1 + (J + 1) * nxAll
                    P01 = I + 0 + (J + 1) * nxAll

                    if isDoubled == 2:
                        elements[cntElem] = np.array([P00,P10,P11])
                        subId[cntElem] = cntSubId
                        cntElem+=1

                        elements[cntElem] = np.array([P00,P11,P01])
                        subId[cntElem] = cntSubId
                        cntElem+=1
                    else:
                        elements[cntElem] = np.array([P00,P10,P11,P01])
                        cntElem+=1

            cntSubId +=1


    materialId = np.zeros(nelemAll,dtype=int)
    for iE in range(cntElem):
        indOneElem = elements[iE,:]
        coordsOneElem = coordinates[indOneElem,:]
        XT = np.mean(coordsOneElem[:,0])
        YT = np.mean(coordsOneElem[:,1])

        meanXY = XT + YT;

        indicator = np.sin(np.pi * meanXY  * GLOBAL_numberOfDiagonalStrips)/ (meanL);

    out = {}
    out['elements'] = elements
    out['nodes'] = coordinates
    out['nodesGrid'] = nodesGrid
    out['elementsGrid'] = elementsGrid
    out['subId'] = subId
    out['materialId'] = materialId

    return out

def getMappingVectors(elementsAll, subSetElements,whoCalled=""):

    subSetElements = np.ravel(subSetElements)
    subMatrixDOFs = np.zeros((0),dtype=int)

    for ieSubSet in subSetElements:
        oneElemNdsInd = elementsAll[ieSubSet,:]
        nPointsLoc = oneElemNdsInd.shape[0]

        oneElemDOFsInd = np.zeros(nPointsLoc*GLOBAL_nDOFsOneNode,dtype=int)
        cntdof = 0
        for idof in range(GLOBAL_nDOFsOneNode):
            for inod in range(nPointsLoc):
                oneElemDOFsInd[cntdof] = GLOBAL_nDOFsOneNode * oneElemNdsInd[inod] + idof
                cntdof +=1
        subMatrixDOFs = np.concatenate((subMatrixDOFs,oneElemDOFsInd))


    l2g = np.unique(subMatrixDOFs)

    dimKloc = l2g.shape[0]
    g2l = {}
    for i in range(dimKloc):
        g2l[l2g[i]] = i

    return l2g, g2l

def computeKernelFromElementsMatrix(nodesAll,elementsAll, subSetElements):#,materialId, ratio):

    E0 = 1.0
    mu = 0.3

    kernels = []

    l2g, g2l = getMappingVectors(elementsAll, subSetElements,"kernelFromElements")
    dimKloc = l2g.shape[0]
    Kloc = np.zeros((dimKloc,dimKloc))

    for ieSubSet in subSetElements:
        oneElemNdsInd = elementsAll[ieSubSet,:]
        nPointsLoc = oneElemNdsInd.shape[0]
        oneElemDOFsInd = np.zeros(nPointsLoc*GLOBAL_nDOFsOneNode,dtype=int)
        cntdof = 0
        for idof in range(GLOBAL_nDOFsOneNode):
            for inod in range(nPointsLoc):
                oneElemDOFsInd[cntdof] = GLOBAL_nDOFsOneNode * oneElemNdsInd[inod] + idof
                cntdof +=1

        xyLocal = nodesAll[oneElemNdsInd,:]
        Ke = callAssemblerLocalMatrix(E0 ,mu,1,xyLocal[:,0],xyLocal[:,1])


        for jLoc in range(Ke.shape[1]):
            jGlb = g2l[oneElemDOFsInd[jLoc]]
            for iLoc in range(Ke.shape[0]):
                iGlb = g2l[oneElemDOFsInd[iLoc]]
                Kloc[iGlb,jGlb] += Ke[iLoc,jLoc]

    kerKloc, rank, jumpEigVals = getKernelDenseMatrix(Kloc)



    return kerKloc, l2g, g2l, jumpEigVals

def getKernelDenseMatrix(Kloc_):
    global  GLOBAL_cnt
    Kloc = Kloc_.copy()

    dimKloc = Kloc.shape[0]
    diagonalKloc = np.sqrt(1./np.diag(Kloc))
    iDK = np.diag(diagonalKloc)
    U, s, V = np.linalg.svd(iDK.dot(Kloc.dot(iDK)))


    rank= dimKloc

    for dof in range(1,dimKloc):
        if (abs(s[dof]) / abs(s[dof-1]) < GLOBAL_zeroPivot):
            rank = dof 
            break
        else:
            rank = dimKloc


    if s.shape[0]>1 and rank < dimKloc:
        jumpEigVals = s[rank] / s[rank-1]
    else:
        jumpEigVals = 0
    if DBG>0:
        print(GLOBAL_cnt,":")
        print("dimKloc = ", dimKloc, ", rank = ",rank, ", defect = ",dimKloc-rank,end=", ")
        print("jump =  {:.2e}".format(jumpEigVals))

        for iss in s: print("{:.2e}".format(iss),end=" ")
        print("")
        GLOBAL_cnt +=1
        if jumpEigVals > 1e-10:
            print("!")


    kerKloc = iDK.dot(V[rank:,:].T)
    for iR in range(kerKloc.shape[1]):
        kerKloc[:,iR] /= np.linalg.norm(kerKloc[:,iR])



    if kerKloc.shape[1] > 0:
        normKR = np.linalg.norm(Kloc.dot(kerKloc))
        normK = np.linalg.norm(Kloc)
        normR = np.linalg.norm(kerKloc)

        rel_normKR = normKR / (normK * normR)

        if DBG>1: print("||K*R||= {:.2e}".format(rel_normKR))




    return kerKloc, rank, jumpEigVals



def dissection(elementsGrid, elementsAll,nodesAll):#,MaterialId, ratio):

    _ex = elementsGrid.shape[0]
    _ey = elementsGrid.shape[1]

    if _ex == 1  and _ey == 1:
        if DBG > 2: print("lowest level")
        kernelPack = computeKernelFromElementsMatrix(nodesAll,elementsAll, elementsGrid[0])#,MaterialId, ratio)
    else:

        if (_ex > _ey):
            half = int(_ex * 0.5)
            first = elementsGrid[:half,:]
            second  = elementsGrid[half:,:]
        else:
            half = int(_ey * 0.5)
            first = elementsGrid[:,:half]
            second  = elementsGrid[:,half:]

        kernelPack10 = dissection(first, elementsAll,nodesAll)#,MaterialId, ratio)
        kernelPack11 = dissection(second, elementsAll,nodesAll)#,MaterialId, ratio)


        R10 = kernelPack10[0]
        l2g10 = kernelPack10[1]
        g2l10 = kernelPack10[2]
        jumpEigVals01 = kernelPack10[3]

        R11 = kernelPack11[0]
        l2g11 = kernelPack11[1]
        g2l11 = kernelPack11[2]
        jumpEigVals11 = kernelPack11[3]

        interface = np.intersect1d(l2g10,l2g11)

        if GLOBAL_GC_MATRIX[:5] == "HFETI":

            if len(GLOBAL_GC_MATRIX) == 5:
                if R10.shape[1] > R11.shape[1]:
                    Grows = R10.shape[1]
                else:
                    Grows = R11.shape[1]

                B10T = np.zeros((R10.shape[0],Grows))
                B11T = np.zeros((R11.shape[0],Grows))

                if R10.shape[1] > R11.shape[1]:
                    for iG in interface:
                        B10T[g2l10[iG],:] = R10[g2l10[iG],:]
                        B11T[g2l11[iG],:] = -R10[g2l10[iG],:]
                else:
                    for iG in interface:
                        B10T[g2l10[iG],:] = R11[g2l10[iG],:]
                        B11T[g2l11[iG],:] = -R11[g2l10[iG],:]

            elif GLOBAL_GC_MATRIX[5:] == "_REDUNDANT":
                Grows01 = R10.shape[1] 
                Grows11 = R11.shape[1]
                Grows = Grows01 + Grows11 

                B10T = np.zeros((R10.shape[0],Grows))
                B11T = np.zeros((R11.shape[0],Grows))


                for iG in interface:
                    for jG in range(Grows01):
                        B10T[g2l10[iG],jG] = R10[g2l10[iG],jG]
                        B11T[g2l11[iG],jG] = -R10[g2l10[iG],jG]
                    for jG in range(Grows11):
                        B10T[g2l10[iG],Grows01+jG] = R11[g2l10[iG],jG]
                        B11T[g2l11[iG],Grows01+jG] = -R11[g2l10[iG],jG]




#################################################
            G10 = B10T.T.dot(R10)
            G11 = B11T.T.dot(R11)
            G = np.hstack((G10,G11))



            H_kerG, rankG, jumpEigValsG = getKernelDenseMatrix(G.T.dot(G))

#            q5, r5, p5 = scplinalg.qr(G, mode='economic', pivoting=True)
#            q5, r5, p5 = scplinalg.qr(G, pivoting=True)
#            print("q5\n",q5.round(4))
#            print("r5\n",r5.round(4))
#            print("p5\n",p5.round(4))
#            print("Grows: ",Grows)
#            print("G.shape: ",G.shape )
#            print("H.shape: ",H_kerG.shape )
#            print("H_ker:,\n",H_kerG)
#            print("R10\n",R10)
#            print("R11\n",R11)
#            print("G10\n",G10)
#            print("G11\n",G11)
#            print("G\n",G)
#            print("GtG\n",G.T.dot(G))


            # correction by H
            defect10 = R10.shape[1]
            defect11 = R11.shape[1]

            R10 = R10.dot(H_kerG[:defect10,:])
            R11 = R11.dot(H_kerG[defect10:,:])
            defect = H_kerG.shape[1]

        elif GLOBAL_GC_MATRIX == "ECONOMIC":
            RG10 = np.zeros((interface.shape[0],R10.shape[1]))
            RG11 = np.zeros((interface.shape[0],R11.shape[1]))
            cnt = 0
            for iG in interface:
                RG10[cnt,:] = R10[g2l10[iG],:]
                RG11[cnt,:] = R11[g2l11[iG],:]
                cnt += 1
            if R10.shape[1] > R11.shape[1]:
                A_RHS = RG10.T.dot(RG10)
                H_RHS = RG10.T.dot(RG11)
                H_kerG = np.linalg.solve(A_RHS,H_RHS)
                R10 = R10.dot(H_kerG)
            else:
                A_RHS = RG11.T.dot(RG11)
                H_RHS = RG11.T.dot(RG10)
                #print(A_RHS)
                H_kerG = np.linalg.solve(A_RHS,H_RHS)
                R11 = R11.dot(H_kerG)

            defect = H_RHS.shape[1]
            jumpEigValsG = 1e15
        else:
            print("GLOBAL_GC_MATRIX has no option: ", GLOBAL_BC_TYPE)
            return None

        l2g, g2l = getMappingVectors(elementsAll, elementsGrid,"dissection")

        kerKloc = np.zeros((l2g.shape[0],defect))



        for i in range(l2g10.shape[0]):
            kerKloc[g2l[l2g10[i]],:] = R10[i,:]

        for i in range(l2g11.shape[0]):
            kerKloc[g2l[l2g11[i]],:] = R11[i,:]


        kerKloc  = myQR(kerKloc)


        updateJumpEigVals = [jumpEigValsG]

        if type(jumpEigVals01) == list:
            updateJumpEigVals += jumpEigVals01
        else:
            updateJumpEigVals.append(jumpEigVals01)
        if type(jumpEigVals11) == list:
            updateJumpEigVals += jumpEigVals11
        else:
            updateJumpEigVals.append(jumpEigVals11)


        kernelPack = [kerKloc,l2g,g2l, updateJumpEigVals]


    return kernelPack


def myQR(a):

    m = a.shape[1]
    n = a.shape[0]

    G = np.zeros((n,m))

    for i in range(m):
        w = a[:,i]
        for j in range(i):
            coeff = np.dot(w,G[:,j])
            w -= G[:,j] * coeff
        coeff = np.linalg.norm(w)
        G[:,i] = w / coeff


    return G



def getMatrix(elementsAll,nodesAll):#, materialRatio, materialId):


    numberOfAllElements = elementsAll.shape[0]
    nDOFs = nodesAll.shape[0] * GLOBAL_nDOFsOneNode

    E0 = 1.0
    mu = 0.3

    Lx2D = Ly2D = 1.0
    meanL = 0.5 * (Lx2D + Ly2D);

    kernels = []

    nDOFsOneElement = GLOBAL_nDOFsOneNode * elementsAll.shape[1]
    dimIJV = nDOFsOneElement**2 * numberOfAllElements

    IK = np.zeros(dimIJV,dtype=int)
    JK = np.zeros(dimIJV,dtype=int)
    VK = np.zeros(dimIJV,dtype=float)
    cntK = 0

    for iE in range(numberOfAllElements):
        oneElemNdsInd = elementsAll[iE,:]
        nPointsLoc = oneElemNdsInd.shape[0]

        oneElemDOFsInd = np.zeros(nPointsLoc*GLOBAL_nDOFsOneNode,dtype=int)
        cntdof = 0
        for idof in range(GLOBAL_nDOFsOneNode):
            for inod in range(nPointsLoc):
                oneElemDOFsInd[cntdof] = GLOBAL_nDOFsOneNode * oneElemNdsInd[inod] + idof
                cntdof +=1

        xyLocal = nodesAll[oneElemNdsInd,:]
        X = xyLocal[:,0]
        Y = xyLocal[:,1]

        XT = np.mean(X)
        YT = np.mean(Y)






        Ke = callAssemblerLocalMatrix(E0,mu,1,X,Y)
        for jLoc in range(Ke.shape[1]):
            jGlb = oneElemDOFsInd[jLoc]
            for iLoc in range(Ke.shape[0]):
                iGlb = oneElemDOFsInd[iLoc]
                IK[cntK] = iGlb
                JK[cntK] = jGlb
                VK[cntK] = Ke[iLoc,jLoc]
                cntK+=1


    K = coo_matrix((VK, (IK,JK)), shape=(nDOFs,nDOFs)).tocsr()
    return K

def getAnalyticKernel(nodesAll):
    kerAnalytic = None
    if GLOBAL_nDOFsOneNode == 2:
        nDOFs = 2 * nodesAll.shape[0]
        if GLOBAL_BC_TYPE == 0:
            kerAnalytic = np.zeros((nDOFs,3))
            kerAnalytic[::2,0] = 1
            kerAnalytic[1::2,1] = 1
            kerAnalytic[::2,2] = -nodesAll[:,1]
            kerAnalytic[1::2,2] = nodesAll[:,0]
        elif GLOBAL_BC_TYPE == 1:
            kerAnalytic =  np.zeros((nDOFs,1))
            kerAnalytic[::2,0] = -nodesAll[:,1]-1
            kerAnalytic[1::2,0] = nodesAll[:,0] 
        elif GLOBAL_BC_TYPE == 2:
            kerAnalytic =  np.zeros((nDOFs,1))
            kerAnalytic[::2,0] = 1

    else:
        nDOFs = nodesAll.shape[0]
        kerAnalytic = np.ones((nDOFs,1))

    if kerAnalytic is not None:
        kerExactOrtho = myQR(kerAnalytic)
    else:
        kerExactOrtho = None

    return kerExactOrtho

def plotHeatKernel(nx,ny,hv_):


    xv,yv =np.meshgrid(np.linspace(0,1,nx),np.linspace(0,1,nx)) 
    hv = np.reshape(hv_,(nx,ny))
    # Simple plot of mountain and parametric curve
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    from matplotlib import cm
    #ax.plot_surface(xv, yv, hv, cmap=cm.coolwarm, rstride=1, cstride=1)
    ax.plot_wireframe(xv, yv, hv, rstride=2, cstride=2)
    plt.show()

def plotMesh(elements,nodes,displacement, matColoring=np.zeros(0), indx=0,ax=None):

    X0 = nodes[:,0].copy()
    Y0 = nodes[:,1].copy()
    X = nodes[:,0].copy()
    Y = nodes[:,1].copy()

    if matColoring.shape[0] == 0 :
        if GLOBAL_nDOFsOneNode == 2:
            Zcolor = np.sqrt(displacement[::2]**2 + displacement[::2]**2)
        else:
            Zcolor = displacement
    else:
        Zcolor = matColoring 

    maxD = np.max(np.abs(displacement))
    if maxD==0:maxD=1;

    if GLOBAL_nDOFsOneNode == 2:
        Lchar = np.max(nodes) - np.min(nodes)
        scale = 0.1 * Lchar / maxD
        X += displacement[::2] * scale
        Y += displacement[1::2] * scale

    triangles = np.vstack((elements[:,[0,1,2]],elements[:,[0,2,3]]))

    if ax is None:
        fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_title(str(indx))
    ax.set_aspect('equal')
#    ax.triplot(X0,Y0,triangles,'g:', linewidth = 0.1)
    ax.tricontourf(X, Y, triangles, Zcolor)
    ax.triplot(X,Y,triangles,'k-', linewidth = 0.1)
    #ax.colorbar()
    XSQ = np.array([0,1,1,0,0])
    YSQ = np.array([0,0,1,1,0])
    ax.plot(XSQ,YSQ,'k--', linewidth=0.3)

    #plt.show()

# N   = 0, 1, 2, ...
# e0  = 1, 2, 3, ... 



def makeAnalysis(N=5):

    e0 = 1

    ex = ey =  e0 * 2**N
    nx = ny = ex + 1
    eAll = ex * ey
    nAll = nx * ny

    meshParameters = [ex,ey,1,1,'4nodes']
    data = mesh(meshParameters)
    elementsAll = data['elements']
    nodesAll = data['nodes']
    nodesGrid  = data['nodesGrid']
    elementsGrid  = data['elementsGrid']
    MaterialId = data['materialId']


    FinalKernelPack = dissection(elementsGrid, elementsAll,nodesAll)#,MaterialId, materialRatio)
    kerK = FinalKernelPack[0]
    matK = getMatrix(elementsAll,nodesAll)#,materialRatio,MaterialId)

    kerKExact = getAnalyticKernel(nodesAll)
    allJumps = np.array(FinalKernelPack[-1])



    #whichColumn =-4 
    #
    if GLOBAL_nDOFsOneNode == 2:
        matColoring = np.zeros(nodesAll.shape[0])
        for iN in range(nodesAll.shape[0]):
            x = nodesAll[iN,0]
            y = nodesAll[iN,1]
            matColoring[iN] = elAssemb.materialCorrection(x,y,GLOBAL_materialRatio,GLOBAL_numberOfDiagonalStrips)
    data['matColoring']=matColoring
    data['kerK']=kerK
    data['kerKExact']=kerKExact
    return matK, data
#
#    if kerK.shape[1] > 0:
#        plotMesh(elementsAll,nodesAll,kerK[:,-1],matColoring)
#    else:
#        plotMesh(elementsAll,nodesAll,np.zeros(np.prod(nodesAll.shape)),matColoring)
#
#else:
#    if kerK.shape[1]>0:
#        plotHeatKernel(nx,ny,kerK[:,-1])

#np.savetxt("elemntsGrid.txt",elementsGrid,fmt='%i') 
#np.savetxt("nodesGrid.txt",nodesGrid,fmt='%i') 


#levels = 5
#Ksparse, data0 = makeAnalysis(levels)
#U,s,V = np.linalg.svd(Ksparse.toarray())
NN = 3
MM = 4
fig1, ax1 = plt.subplots(NN,MM)

for j in range(MM):
    for i in range(NN):
        indx = j * NN + i + 1
        plotMesh(data0['elements'],data0['nodes'],U[:,-indx],data0['matColoring'],indx,ax1[i,j])
fname = "/home/mar440/WorkSpace/theory/kernel/Figs/tmp_modes.eps"
plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
#plt.show()

print("dim(K) = " , Ksparse.shape[0])
print("defect(K) = " , data0["kerK"].shape[1])

delta = np.linalg.norm(Ksparse.dot(data0["kerK"]),2)
normK = slinalg.norm(Ksparse)
if data0["kerK"].shape[1]>0:
    normR = np.linalg.norm(data0["kerK"],2)
    print("|| K * R_numeric  || = " , delta / (normK * normR))

if data0["kerKExact"] is not None:
    delta1 = np.linalg.norm(Ksparse.dot(data0["kerKExact"]),2)
    normRa = np.linalg.norm(data0["kerKExact"],2)
    print("|| K * R_analytic || = " , delta1 / (normK * normRa))
    print("|| K || =              " , normK )
    print("|| Ra|| =              " , normRa)
    print("|| Rn|| =              " , normR)


