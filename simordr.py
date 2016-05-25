import numpy as np

""""
Usage:

xgap, ygap = calculate_gaps(curves_centered, gapfrac=0.1)
ubound, lbound = boundaries(curves_centered[i], int(xgap/2), 2*ygap)

# calculate simtable
simtable = calculate_simtable_par(curves_centered, xgap, ygap)

# generate ordering
sim_ordering = simple_similarity_ordering(simtable, curves_ordering)
# plot
fdplot(curves_centered, sim_ordering)

""""

def calculate_gaps(curves, gapfrac):
    """
    Given the percentage (gapfrac), calculate the distance in 
    the range and domain of the curves
    """
    yrange = np.max(curves) - np.min(curves)
    xrange = curves.shape[1]
    
    ygap = yrange*gapfrac
    xgap = xrange*gapfrac
    return (int(np.round(xgap)), ygap)

def boundaries(curve, xgap, ygap):
    """
    Generate upper and lower boundary for given curve
    """
    ubound = curve
    lbound = curve
    for i in range(1,xgap):
        lshift = np.concatenate((curve[i:], curve[-1]*np.ones(i)))
        rshift = np.concatenate((curve[0]*np.ones(i), curve[:-i]))

        ubound = np.max((ubound, curve+ygap, rshift+ygap, lshift+ygap), axis=0)
        lbound = np.min((lbound, curve-ygap, rshift-ygap, lshift-ygap), axis=0)
    return ubound, lbound

def similar(c1, c2, xgap, ygap): 
    "Binary similarity measure, 1 if similar, 0 if not"
    
    ubound, lbound = boundaries(c1, xgap, ygap)
    
    #if np.all(c2[xgap:-xgap]<=ubound[xgap:-xgap]) and \
    #   np.all(c2[xgap:-xgap]>=lbound[xgap:-xgap]):
    if np.all(c2<=ubound) and \
       np.all(c2>=lbound):
        return True
    else:
        return False
    
def calculate_simtable_par(curves, xgap, ygap, pool=multiprocessing.Pool()):
    n,_ = curves.shape
    simtable = np.zeros((n,n))
    for i in range(n):
        row = pool.starmap(similar, zip(repeat(curves[i]), 
                                        curves.tolist(), 
                                        repeat(xgap), 
                                        repeat(ygap)))
        simtable[i,:] = np.asarray(list(row))
    return simtable

def calculate_simtable(curves, xgap, ygap):
    n,_ = curves.shape
    simtable = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            simtable[i,j] = similar(curves[i], curves[j], xgap, ygap)
    return simtable
        
def simple_similarity_ordering(sim, curves, projection_power = 3, grouping_fraction = 0.3):    
    # simtable not necessarly simmmetrical. Make it so.
    sim[~(sim.transpose()==sim)]=0.5
    # list of indexes of curves from which the orderinf is created 
    c_set_ind = np.arange(curves.shape[0])
    # initialize empty list for the ordered indexes
    ordering = np.array([], dtype=np.int64)

    while len(c_set_ind)>=1:
        # order the simtable by how many things is it similar to
        tbl_ordr = sim.sum(axis=1).argsort()
        sim = sim[tbl_ordr]
        sim = sim.T[tbl_ordr].T
        c_set_ind = c_set_ind[tbl_ordr]

        # Apply matrix to a unit vector
        unit = np.zeros(len(c_set_ind))
        unit[-1]=1
        proj = unit
        for k in range(projection_power):
            proj = np.dot(sim, proj)

        # get dom of top grouping_fraction of range
        bound = proj.max() - (proj.max()-proj.min())*grouping_fraction
        top_sim = proj>=bound

        # next batch if indexes
        next_batch = np.where(top_sim)[0].tolist()
        ordering = np.concatenate((ordering, c_set_ind[next_batch]))
        
        # remove already selected rows and cols from sim table and index
        sim = sim[~top_sim]
        sim = sim.T[~top_sim].T
        c_set_ind = c_set_ind[~top_sim]

        return ordering