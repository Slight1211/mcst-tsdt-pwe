"""Independent classical 3D periodic tetrahedral solid FE; no MCST/surface/fluid.

CAD circle is meshed with straight tetrahedron faces. P2 displacement; fourth
degree element quadrature integrates polynomial mass exactly on each tetrahedron.
"""
import numpy as np
import gmsh
from skfem import MeshTet,Basis,ElementVector,ElementTetP2,BilinearForm,asm
from skfem.helpers import sym_grad,ddot,div,dot
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh,norm as sparse_norm

def mesh_cell(size,h=.1,fraction=.3,uniform=False,output=None):
    gmsh.initialize();gmsh.option.setNumber('General.Terminal',0);gmsh.option.setNumber('General.NumThreads',1)
    try:
        gmsh.model.add('classical_periodic_cell');occ=gmsh.model.occ
        box=occ.addBox(-.5,-.5,-h/2,1,1,h)
        if not uniform:
            cylinder=occ.addCylinder(0,0,-h/2,0,0,h,np.sqrt(fraction/np.pi));occ.fragment([(3,box)],[(3,cylinder)])
        occ.synchronize()
        volumes=gmsh.model.getEntities(3);inclusion=None
        if not uniform:
            inclusion=min(volumes,key=lambda v:abs(occ.getMass(*v)-fraction*h))[1]
            if abs(occ.getMass(3,inclusion)-fraction*h)>1e-8:raise ValueError('CAD material volume mismatch')
        faces=gmsh.model.getEntities(2)
        for axis in [0,1]:
            selected=[]
            for target in [-.5,.5]:
                tags=[tag for dim,tag in faces if abs(gmsh.model.getBoundingBox(dim,tag)[axis]-target)<1e-6 and abs(gmsh.model.getBoundingBox(dim,tag)[axis+3]-target)<1e-6]
                if len(tags)!=1:raise ValueError('Unexpected periodic face partition')
                selected.append(tags[0])
            affine=np.eye(4);affine[axis,3]=1
            gmsh.model.mesh.setPeriodic(2,[selected[1]],[selected[0]],affine.ravel().tolist())
        gmsh.option.setNumber('Mesh.MeshSizeMin',size/3);gmsh.option.setNumber('Mesh.MeshSizeMax',size)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',40 if not uniform else 0)
        gmsh.model.mesh.generate(3)
        if output:gmsh.write(str(output))
        tags,coords,_=gmsh.model.mesh.getNodes();order=np.argsort(tags);tags=tags[order];points=np.asarray(coords).reshape(-1,3)[order]
        cells=[];phase=[]
        for _,tag in volumes:
            types,etags,enodes=gmsh.model.mesh.getElements(3,tag)
            for typ,elems,nodes in zip(types,etags,enodes):
                if typ!=4:raise ValueError('Expected linear geometry tetrahedra')
                cells.append(np.searchsorted(tags,np.asarray(nodes).reshape(-1,4)));phase.extend([int(tag==inclusion)]*len(elems))
        mesh=MeshTet(points.T,np.vstack(cells).T)
        return mesh,np.asarray(phase)
    finally:gmsh.finalize()

def assemble(mesh,phase,materials):
    basis=Basis(mesh,ElementVector(ElementTetP2()),intorder=4)
    E=np.array([materials[i][0] for i in phase]);nu=np.array([materials[i][1] for i in phase]);rho=np.array([materials[i][2] for i in phase])
    mu=E/(2*(1+nu));la=E*nu/((1+nu)*(1-2*nu))
    @BilinearForm
    def elastic(u,v,w):return 2*w.mu*ddot(sym_grad(u),sym_grad(v))+w.la*div(u)*div(v)
    @BilinearForm
    def inertia(u,v,w):return w.rho*dot(u,v)
    @BilinearForm
    def vertical(u,v,w):return w.rho*u[2]*v[2]
    K=asm(elastic,basis,mu=mu[:,None],la=la[:,None]);M=asm(inertia,basis,rho=rho[:,None]);Mz=asm(vertical,basis,rho=rho[:,None])
    return basis,K,M,Mz

def bloch_map(basis,k):
    loc=basis.doflocs.copy();shifts=np.zeros_like(loc)
    for axis in [0,1]:
        side=np.isclose(loc[axis],.5,atol=1e-9,rtol=0);loc[axis,side]-=1;shifts[axis,side]=1
    components=np.empty(basis.N,int)
    for j,ids in enumerate(basis.split_indices()):components[ids]=j
    groups={};columns=[];base_counts={}
    for j in range(basis.N):
        key=(int(components[j]),*np.round(loc[:,j],9));column=groups.setdefault(key,len(groups));columns.append(column)
        if np.all(shifts[:,j]==0):base_counts[column]=base_counts.get(column,0)+1
    if any(base_counts.get(j)!=1 for j in range(len(groups))):raise ValueError('Nonmatching periodic mesh/DOF coordinates')
    phase=np.exp(1j*(k[0]*shifts[0]+k[1]*shifts[1]))
    P=coo_matrix((phase,(np.arange(basis.N),columns)),shape=(basis.N,len(groups))).tocsr()
    return P

def solve_modes(basis,K,M,Mz,k,count=12):
    P=bloch_map(basis,k);Kr=P.conj().T@K@P;Mr=P.conj().T@M@P
    herm=max(sparse_norm(Kr-Kr.conj().T)/sparse_norm(Kr),sparse_norm(Mr-Mr.conj().T)/sparse_norm(Mr))
    if herm>1e-11:raise ValueError('Non-Hermitian assembled periodic solid')
    lam,v=eigsh(Kr,k=count,M=Mr,sigma=-1e-7,which='LM',tol=1e-10);order=np.argsort(lam);lam=lam[order];v=v[:,order]
    if np.min(lam)<-1e-7:raise ValueError('Negative squared frequency')
    u=P@v;vertical=np.real(np.sum(u.conj()*(Mz@u),axis=0)/np.sum(u.conj()*(M@u),axis=0))
    residual=np.linalg.norm(Kr@v-(Mr@v)*lam)/(sparse_norm(Kr)*np.linalg.norm(v)+sparse_norm(Mr)*np.linalg.norm(v*lam))
    return dict(lam=lam,u=u,vertical_fraction=vertical,hermitian_error=float(herm),residual=float(residual),full_dofs=int(basis.N),reduced_dofs=int(P.shape[1]))
