import numpy as np

GLOBAL_meanL = (1.0 + 1.0) * 0.5
GLOBAL_ApplyDirichletBC = True
GLOBAL_penalty = 1e3

def materialCorrection(_x,_y, ratio,numberOfDiagonalStrips):

    meanXY = GLOBAL_meanL*0.075 + _x + 0.1*_y
    out = np.sin(np.pi * meanXY  * numberOfDiagonalStrips)/ GLOBAL_meanL

    return pow(10,ratio * float(out>0))

def __shapeFun(r, s):
    N = np.zeros(4)
    dNr = np.zeros(4)
    dNs = np.zeros(4)
    N[0] = 0.25*(1.0 - r)*(1.0 - s);
    N[1] = 0.25*(1.0 + r)*(1.0 - s);
    N[2] = 0.25*(1.0 + r)*(1.0 + s);
    N[3] = 0.25*(1.0 - r)*(1.0 + s); 

    dNr[0] = 0.25*(-1.0)*(1.0 - s);
    dNr[1] = 0.25*( 1.0)*(1.0 - s);
    dNr[2] = 0.25*( 1.0)*(1.0 + s);
    dNr[3] = 0.25*(-1.0)*(1.0 + s);

    dNs[0] = 0.25*(1.0 - r)*(-1.0);
    dNs[1] = 0.25*(1.0 + r)*(-1.0);
    dNs[2] = 0.25*(1.0 + r)*( 1.0);
    dNs[3] = 0.25*(1.0 - r)*( 1.0);
    return N, dNr, dNs


def __GaussPoints_1D(nGP):
    if nGP == 2:
        w  = np.ones(2);
        r = np.array([-1,1]) * 0.5773502691896257
    elif nGP == 3:
        w = np.array([0.8888888888888888, 0.5555555555555556, 0.5555555555555556])
        r = np.array([ 0.0000000000000000, -0.7745966692414834, 0.7745966692414834])
    return r, w



def elem2d4n(E, mu, rho, xCoord, yCoord, package):# ratio = 8, numberOfDiagonalStrips = 6):

    #[GLOBAL_materialRatio, GLOBAL_numberOfDiagonalStrips, GLOBAL_BC_TYPE]   

    ratio = package[0]
    numberOfDiagonalStrips = package[1]
    BC_TYPE = package[2]

    nP = xCoord.shape[0];
    nDOFs = nP * 2

    m_numberOfGaussPoints = 2
    B = np.zeros((3, nDOFs))
    Ke = np.zeros((nDOFs, nDOFs))
    f = np.zeros(nDOFs)
    C = np.zeros((3,3))
    C[0, 0] = 1.0
    C[1, 1] = 1.0
    C[0, 1] = mu
    C[1, 0] = mu
    C[2, 2] = 1.0 - mu
    C *=  E / (1.0 - mu * mu)


    volF = np.array([0,-9.81])
    r_GP, weights = __GaussPoints_1D(m_numberOfGaussPoints)


    for i in range(m_numberOfGaussPoints):
        for j in range(m_numberOfGaussPoints):
            N, dNr, dNs = __shapeFun(r_GP[i], r_GP[j])
            x_rs = y_rs = dxdr = dxds = dydr = dyds = 0;

            x_rs = np.dot(N,xCoord)
            y_rs = np.dot(N,yCoord)

            dxdr = np.dot(dNr,xCoord)
            dydr = np.dot(dNr,yCoord)

            dxds = np.dot(dNs,xCoord)
            dyds = np.dot(dNs,yCoord)


            J =  np.array([[dxdr, dydr],[dxds,dyds]])

            detJ = np.linalg.det(J) 


            if detJ < 0: 
                print("negativ Jacobian") 
                return []

            dNrs = np.vstack((dNr,dNs))
            dNxy = np.linalg.solve(J,dNrs)

            for k in range(nP):
                B[0, k] = dNxy[0, k]
                B[1, k + nP] = dNxy[1, k]
                B[2, k] = dNxy[1, k]
                B[2, k + nP] = dNxy[0, k]

            w_ij_detJ = weights[i] * weights[j] * detJ;


            C_current = C.copy()

            indicator = materialCorrection(x_rs,y_rs,ratio, numberOfDiagonalStrips)
            C_current *= indicator


            Ke += B.T.dot(C_current.dot(B)) * w_ij_detJ;

    if GLOBAL_ApplyDirichletBC:



        if BC_TYPE  == 0:
            "floating ... no Dirichlet"
            logIndx = np.zeros(nP,dtype=bool)
            logIndy = np.zeros(nP,dtype=bool)
        elif BC_TYPE == 1:
            "XY_LEFT_TOP_CORNER"
            logIndx = (abs(xCoord) < 0.01) * (abs(yCoord - 1) < 0.01)
            logIndy = (abs(xCoord) < 0.01) * (abs(yCoord - 1) < 0.01)
        elif BC_TYPE == 2:
            "X_LEFT_EDGE"
            logIndx = ((abs(xCoord)<0.0001))
            logIndy = np.zeros(nP,dtype=bool)
        elif BC_TYPE == 3:
            "X_LEFT_EDGE_Y_BOTTOM_EDGE_"
            logIndx = (abs(yCoord)< ( 0.0001))**0 
            logIndy = ((abs(xCoord)<0.0001)**0  + (abs(xCoord-1)<0.0001)*0)




        if any(logIndx) or any(logIndy):

            indxs = np.zeros(0,dtype=int)
            if any(logIndx):
                dirX = np.where(logIndx)[0]
                indxs = np.concatenate((indxs,dirX))
            if any(logIndy):
                dirY = np.where(logIndy)[0]
                indxs = np.concatenate((indxs,nP + dirY))

            for ik in indxs:
                dKe = Ke[ik,ik]
                Ke[ik,:] = 0
                Ke[:,ik] = 0
                Ke[ik,ik] = GLOBAL_penalty * dKe
    return Ke

def elem2d4nHeat(Ex,Ey,xCoord ,yCoord, package=None):


    if package is not None:
        ratio = package[0]
        numberOfDiagonalStrips = package[1]
        BC_TYPE = package[2]

    nP = xCoord.shape[0];
    nDOFs = nP

    m_numberOfGaussPoints = 2
    Ke = np.zeros((nDOFs, nDOFs))
    f = np.zeros(nDOFs)
    C = np.diag(np.array([Ex,Ey]))


    r_GP, weights = __GaussPoints_1D(m_numberOfGaussPoints)


    for i in range(m_numberOfGaussPoints):
        for j in range(m_numberOfGaussPoints):
            N, dNr, dNs = __shapeFun(r_GP[i], r_GP[j])
            x_rs = y_rs = dxdr = dxds = dydr = dyds = 0;

            x_rs = np.dot(N,xCoord)
            y_rs = np.dot(N,yCoord)

            dxdr = np.dot(dNr,xCoord)
            dydr = np.dot(dNr,yCoord)

            dxds = np.dot(dNs,xCoord)
            dyds = np.dot(dNs,yCoord)


            J =  np.array([[dxdr, dydr],[dxds,dyds]])

            detJ = np.linalg.det(J) 


            if detJ < 0: 
                print("negativ Jacobian") 
                return []

            dNrs = np.vstack((dNr,dNs))
            dNxy = np.linalg.solve(J,dNrs)

            w_ij_detJ = weights[i] * weights[j] * detJ;


            C_current = C.copy()
            if package is not None:
                indicator = materialCorrection(x_rs,y_rs,ratio, numberOfDiagonalStrips)
                C_current *= indicator


            Ke += dNxy.T.dot(C_current.dot(dNxy)) * w_ij_detJ;

    if package is not None and GLOBAL_ApplyDirichletBC:

        if BC_TYPE  == 0:
            "floating ... no Dirichlet"
            logIndx = np.zeros(nP,dtype=bool)
        elif BC_TYPE == 1:
            "XY_LEFT_TOP_CORNER"
            logIndx = (abs(xCoord) < 0.01) * (abs(yCoord - 1) < 0.01)
        elif BC_TYPE == 2:
            "X_LEFT_EDGE_Y_BOTTOM_EDGE_"
            logIndx = ((abs(xCoord)<0.001)**0  + (abs(xCoord-1)<0.001))
        elif BC_TYPE == 3:
            "X_LEFT_EDGE_Y_BOTTOM_EDGE_"
            logIndx = ((abs(yCoord)< ( 0.0001))**0 + (abs(xCoord)<0.0001)**0 )


        if any(logIndx):
            indxs = np.zeros(0,dtype=int)
            if any(logIndx):
                indxs = np.where(logIndx)[0]
            for ik in indxs:
                dKe = Ke[ik,ik]
                Ke[ik,:] = 0
                Ke[:,ik] = 0
                Ke[ik,ik] = GLOBAL_penalty * dKe
    return Ke

if __name__ == "__main__":

    E = 1.1
    mu = 0.35
    rho = 1.4
    xCoord = np.array([0, 0.9, 0.95, 0.1])
    yCoord = np.array([0.01,0.02,0.99,1.001])
    Ke = elem2d4n(E, mu, rho, xCoord, yCoord)
    U,s,V = np.linalg.svd(Ke)
    print(s)
