@numba.jit(nopython=True, nogil=True)
def build_grid_3d21_sensker( x, y, z, fntau, tau, nspread, sens_ker = None ):
    if sens_ker is None: #then no coil sensitivity map will be used, but sill sens_ker need to be 4d
        sens_ker = np.ones((2 * nspread + 1, 2 * nspread + 1, 2 * nspread + 1, 1),\
                   dtype = fntau.dtype)/((2 * nspread + 1)**3)
    nc    = sens_ker.shape[3] #currently fix the forth dim as the coil dim
    nf1   = fntau.shape[0]    #in addtion, the forth dim of fntau need to be 1 or None
    nf2   = fntau.shape[1]
    nf3   = fntau.shape[2]
    hx    = 2 * np.pi / nf1
    hy    = 2 * np.pi / nf2
    hz    = 2 * np.pi / nf3
    Em    = np.zeros((2 * nspread + 1, 2 * nspread + 1, 2 * nspread + 1)) #will reuse this exponential
    # make the forth dim be the coil dim, which need to be length = 1 for fntau
    if len(fntau.shape) > 3 and fntau.shape[3] > 1: #if dim > 3, and dim 4 (coil dim) is not 1
        # input dim [nf1, nf2, nf3, a ,b c] output [nf1, nf2, nf3, 1, a, b, c]
        fntau     = np.expand_dims(fntau, axis = 3)
    #zero pad the sens_ker, enforce the sens_ker equals to convlution kernel for simplicity
    if sens_ker.shape[0:3] != (2 * nspread + 1, 2 * nspread + 1, 2 * nspread + 1 ):
        sens_ker  = ut.pad_or_cut3d( sens_ker, 2 * nspread + 1, 2 * nspread + 1, 2 * nspread + 1 )
    #initial ftau as zeros for output data
    ftau          = np.zeros(fntau.shape, dtype = fntau.dtype)
    outftaushape, outsenskshape = ut.dim_match(fntau.shape, sens_ker.shape)#match the dim, by adding 1 in extra dim
    fntau         = fntau.reshape(outftaushape)# [nf1, nf2, nf3, 1, a, b, c] or [nf1, nf2, nf3, 1] if 3d data
    ftau          = ftau.reshape(outftaushape) # [nf1, nf2, nf3, 1, a, b, c] or [nf1, nf2, nf3, 1] if 3d data
    sens_ker      = sens_ker.reshape(outsenskshape) #[nk, nk, nk, nc, 1, 1, 1] or [nk, nk, nk, nc] nk = 2*nspread + 1
    #do gridding for each ksp data point
    for i in range(x.shape[0]):
        c  = np.multiply(np.zeros(fntau[0,0,0].shape, dtype = fntau.dtype), \
             np.zeros(sens_ker[0,0,0].shape, dtype = fntau.dtype)) #coefficient, saved temporarily
        xi = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
        yi = y[i] % (2 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
        zi = z[i] % (2 * np.pi) #z, shift the source point zj so that it lies in [0,2*pi]
        m1 = 1 + int(xi // hx) #index for the closest grid point
        m2 = 1 + int(yi // hy) #index for the closest grid point
        m3 = 1 + int(zi // hz) #index for the closest grid point
        #grid once
        for mm1 in range(-nspread, nspread): #mm index for all the spreading points
            for mm2 in range(-nspread,nspread):
                for mm3 in range(-nspread,nspread):
                    #griding with g(x,y) = exp(-(x^2 + y^2 + z^2) / 4*tau)
                    Em[mm1 + nspread, mm2 + nspread, mm3 + nspread] =\
                     np.exp(-0.25 * (\
                    (xi - hx * (m1 + mm1)) ** 2 + \
                    (yi - hy * (m2 + mm2)) ** 2 + \
                    (zi - hz * (m3 + mm3)) ** 2 ) / tau)
                    #added sens_ker this is covlution with expoential Em and sens_kernel
                    c += np.multiply(\
                         fntau[(m1 + mm1) % nf1, (m2 + mm2) % nf2, (m3 + mm3) % nf3]\
                             * Em[mm1 + nspread, mm2 + nspread, mm3 + nspread], \
                         sens_ker[mm1 + nspread, mm2 + nspread, mm3 + nspread])
        #grid again
        c = c/(nf1*nf2*nf3)
        for mm1 in range(-nspread, nspread): #mm index for all the spreading points
            for mm2 in range(-nspread,nspread):
                for mm3 in range(-nspread,nspread):
                    #griding with g(x,y) = exp(-(x^2 + y^2 + z^2) / 4*tau)
                    #convlute with Em and conj(sens_ker), then sum along coil dimension
                    ftau[(m1 + mm1) % nf1, (m2 + mm2) % nf2, (m3 + mm3) % nf3] \
                    += np.sum(np.multiply(c * Em[mm1 + nspread, mm2 + nspread, mm3 + nspread],\
                                np.conj(sens_ker[mm1 + nspread, mm2 + nspread, mm3 + nspread])), axis = 3)
    return ftau
    #type1, fast version with precompute of exponentials
@cuda.jit
def gaussker_array2_3d1_fast_cuda(x, y, z, c, hx, hy, hz, nf1, nf2, nf3, nt, nspread, tau, E3, real_ftau, imag_ftau ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    i     = cuda.grid(1)
    if i > c.shape[0]:
        return
  
    #if i0 > x.shape[0]:
    #    i = i0%x.shape[0]
    #else:
    #    i = i0

    #read x, y, z values
    xi    = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
    yi    = y[i] % (2 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
    zi    = z[i] % (2 * np.pi) #z, shift the source point zj so that it lies in [0,2*pi]
    mx    = 1 + int(xi // hx) #index for the closest grid point
    my    = 1 + int(yi // hy) #index for the closest grid point
    mz    = 1 + int(zi // hz) #index for the closest grid point
    xi    = (xi - hx * mx) #offsets from the closest grid point
    yi    = (yi - hy * my) #offsets from the closest grid point
    zi    = (zi - hz * mz) #offsets from the closest grid point
    # precompute E1, E2x, E2y, E2z
    E1    = exp(-0.25 * (xi ** 2 + yi ** 2 + zi ** 2) / tau)
    E2x   = exp((xi * np.pi) / (nf1 * tau))
    E2y   = exp((yi * np.pi) / (nf2 * tau))
    E2z   = exp((zi * np.pi) / (nf3 * tau))
    for t in range(nt):
        V0    = c[i,t] * E1
        #do the 3d griding here,
        #use the symmetry of E1, E2 and E3, e.g. 1/(E2mmz*E2z) = 1/(E2x**(mmx)*E2x) = E2x**(-mmx-1)
        E2mmx = 1#update with E2mmx *= E2x <-> E2mmx = E2x**(mmz) in the middle loop
        for mmx in range(nspread):#mm index for all the spreading points
            E2mmy = 1#update with E2mmy *= E2y <-> E2mmy = E2y**(mmy) in the middle loop
            for mmy in range(nspread):#mm index for all the spreading points
                E2mmz = 1#update with E2mmz *= E2z <-> E2mmz = E2z**(mmz) in the middle loop
                for mmz in range(nspread):#mm index for all the spreading points
                    #ftau[    (mx + mmx) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3] +=
                    tmpxpypzp = V0 * E2mmx       * E2mmy       * E2mmz        * E3[    mmx,     mmy,     mmz]
                    #ftau[    (mx + mmx) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3] +=
                    tmpxpynzp = V0 * E2mmx       / (E2mmy*E2y) * E2mmz        * E3[    mmx, mmy + 1,     mmz]
                    #ftau[(mx - mmx - 1) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3] +=
                    tmpxnypzp = V0 / (E2mmx*E2x) * E2mmy       * E2mmz        * E3[mmx + 1,     mmy,     mmz]
                    #ftau[(mx - mmx - 1) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3] +=
                    tmpxnynzp = V0 / (E2mmx*E2x) / (E2mmy*E2y) * E2mmz        * E3[mmx + 1, mmy + 1,     mmz]
                    #ftau[    (mx + mmx) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3] +=
                    tmpxpypzn = V0 * E2mmx       * E2mmy       / (E2mmz*E2z)  * E3[    mmx,     mmy, mmz + 1]
                    #ftau[    (mx + mmx) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3] +=
                    tmpxpynzn = V0 * E2mmx       / (E2mmy*E2y) / (E2mmz*E2z)  * E3[    mmx, mmy + 1, mmz + 1]
                    #ftau[(mx - mmx - 1) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3] +=
                    tmpxnypzn = V0 / (E2mmx*E2x) * E2mmy       / (E2mmz*E2z)  * E3[mmx + 1,     mmy, mmz + 1]
                    #ftau[(mx - mmx - 1) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3] +=
                    tmpxnynzn = V0 / (E2mmx*E2x) / (E2mmy*E2y) / (E2mmz*E2z)  * E3[mmx + 1, mmy + 1, mmz + 1]
                    #use atom sum here
                    cuda.atomic.add(real_ftau, (    (mx + mmx) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3, t), tmpxpypzp.real) #x  1, y  1, z  1
                    cuda.atomic.add(imag_ftau, (    (mx + mmx) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3, t), tmpxpypzp.imag)
                    cuda.atomic.add(real_ftau, (    (mx + mmx) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3, t), tmpxpynzp.real) #x  1, y -1, z  1
                    cuda.atomic.add(imag_ftau, (    (mx + mmx) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3, t), tmpxpynzp.imag)
                    cuda.atomic.add(real_ftau, ((mx - mmx - 1) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3, t), tmpxnypzp.real) #x -1, y  1, z  1
                    cuda.atomic.add(imag_ftau, ((mx - mmx - 1) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3, t), tmpxnypzp.imag)
                    cuda.atomic.add(real_ftau, ((mx - mmx - 1) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3, t), tmpxnynzp.real) #x -1, y -1, z  1
                    cuda.atomic.add(imag_ftau, ((mx - mmx - 1) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3, t), tmpxnynzp.imag)
                    cuda.atomic.add(real_ftau, (    (mx + mmx) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3, t), tmpxpypzn.real) #x  1, y  1, z  1
                    cuda.atomic.add(imag_ftau, (    (mx + mmx) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3, t), tmpxpypzn.imag)
                    cuda.atomic.add(real_ftau, (    (mx + mmx) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3, t), tmpxpynzn.real) #x  1, y -1, z  1
                    cuda.atomic.add(imag_ftau, (    (mx + mmx) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3, t), tmpxpynzn.imag)
                    cuda.atomic.add(real_ftau, ((mx - mmx - 1) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3, t), tmpxnypzn.real) #x -1, y  1, z  1
                    cuda.atomic.add(imag_ftau, ((mx - mmx - 1) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3, t), tmpxnypzn.imag)
                    cuda.atomic.add(real_ftau, ((mx - mmx - 1) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3, t), tmpxnynzn.real) #x -1, y -1, z  1
                    cuda.atomic.add(imag_ftau, ((mx - mmx - 1) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3, t), tmpxnynzn.imag)
                E2mmz *= E2z
            E2mmy *= E2y
        E2mmx *= E2x


#type1, fast version with precompute of exponentials
@cuda.jit
def gaussker_array0_3d1_fast_cuda(x, y, z, c, hx, hy, hz, nf1, nf2, nf3, nspread, tau, E3, real_ftau, imag_ftau ):
    """This kernel function for gauss grid 1d typ1, and it will be executed by a thread."""
    i0     = cuda.grid(1)
    if i0 > c.shape[0]:
        return
  
    if i0 > x.shape[0]:
        i = i0%x.shape[0]
    else:
        i = i0

    #read x, y, z values
    xi    = x[i] % (2 * np.pi) #x, shift the source point xj so that it lies in [0,2*pi]
    yi    = y[i] % (2 * np.pi) #y, shift the source point yj so that it lies in [0,2*pi]
    zi    = z[i] % (2 * np.pi) #z, shift the source point zj so that it lies in [0,2*pi]
    mx    = 1 + int(xi // hx) #index for the closest grid point
    my    = 1 + int(yi // hy) #index for the closest grid point
    mz    = 1 + int(zi // hz) #index for the closest grid point
    xi    = (xi - hx * mx) #offsets from the closest grid point
    yi    = (yi - hy * my) #offsets from the closest grid point
    zi    = (zi - hz * mz) #offsets from the closest grid point
    # precompute E1, E2x, E2y, E2z
    E1    = exp(-0.25 * (xi ** 2 + yi ** 2 + zi ** 2) / tau)
    E2x   = exp((xi * np.pi) / (nf1 * tau))
    E2y   = exp((yi * np.pi) / (nf2 * tau))
    E2z   = exp((zi * np.pi) / (nf3 * tau))
    V0    = c[i0] * E1
    #do the 3d griding here,
    #use the symmetry of E1, E2 and E3, e.g. 1/(E2mmz*E2z) = 1/(E2x**(mmx)*E2x) = E2x**(-mmx-1)
    E2mmx = 1#update with E2mmx *= E2x <-> E2mmx = E2x**(mmz) in the middle loop
    for mmx in range(nspread):#mm index for all the spreading points
        E2mmy = 1#update with E2mmy *= E2y <-> E2mmy = E2y**(mmy) in the middle loop
        for mmy in range(nspread):#mm index for all the spreading points
            E2mmz = 1#update with E2mmz *= E2z <-> E2mmz = E2z**(mmz) in the middle loop
            for mmz in range(nspread):#mm index for all the spreading points
                #ftau[    (mx + mmx) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3] +=
                tmpxpypzp = V0 * E2mmx       * E2mmy       * E2mmz        * E3[    mmx,     mmy,     mmz]
                #ftau[    (mx + mmx) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3] +=
                tmpxpynzp = V0 * E2mmx       / (E2mmy*E2y) * E2mmz        * E3[    mmx, mmy + 1,     mmz]
                #ftau[(mx - mmx - 1) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3] +=
                tmpxnypzp = V0 / (E2mmx*E2x) * E2mmy       * E2mmz        * E3[mmx + 1,     mmy,     mmz]
                #ftau[(mx - mmx - 1) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3] +=
                tmpxnynzp = V0 / (E2mmx*E2x) / (E2mmy*E2y) * E2mmz        * E3[mmx + 1, mmy + 1,     mmz]
                #ftau[    (mx + mmx) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3] +=
                tmpxpypzn = V0 * E2mmx       * E2mmy       / (E2mmz*E2z)  * E3[    mmx,     mmy, mmz + 1]
                #ftau[    (mx + mmx) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3] +=
                tmpxpynzn = V0 * E2mmx       / (E2mmy*E2y) / (E2mmz*E2z)  * E3[    mmx, mmy + 1, mmz + 1]
                #ftau[(mx - mmx - 1) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3] +=
                tmpxnypzn = V0 / (E2mmx*E2x) * E2mmy       / (E2mmz*E2z)  * E3[mmx + 1,     mmy, mmz + 1]
                #ftau[(mx - mmx - 1) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3] +=
                tmpxnynzn = V0 / (E2mmx*E2x) / (E2mmy*E2y) / (E2mmz*E2z)  * E3[mmx + 1, mmy + 1, mmz + 1]
                #use atom sum here
                cuda.atomic.add(real_ftau, (    (mx + mmx) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3, i0//x.shape[0]), tmpxpypzp.real) #x  1, y  1, z  1
                cuda.atomic.add(imag_ftau, (    (mx + mmx) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3, i0//x.shape[0]), tmpxpypzp.imag)
                cuda.atomic.add(real_ftau, (    (mx + mmx) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3, i0//x.shape[0]), tmpxpynzp.real) #x  1, y -1, z  1
                cuda.atomic.add(imag_ftau, (    (mx + mmx) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3, i0//x.shape[0]), tmpxpynzp.imag)
                cuda.atomic.add(real_ftau, ((mx - mmx - 1) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3, i0//x.shape[0]), tmpxnypzp.real) #x -1, y  1, z  1
                cuda.atomic.add(imag_ftau, ((mx - mmx - 1) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3, i0//x.shape[0]), tmpxnypzp.imag)
                cuda.atomic.add(real_ftau, ((mx - mmx - 1) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3, i0//x.shape[0]), tmpxnynzp.real) #x -1, y -1, z  1
                cuda.atomic.add(imag_ftau, ((mx - mmx - 1) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3, i0//x.shape[0]), tmpxnynzp.imag)
                cuda.atomic.add(real_ftau, (    (mx + mmx) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3, i0//x.shape[0]), tmpxpypzn.real) #x  1, y  1, z  1
                cuda.atomic.add(imag_ftau, (    (mx + mmx) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3, i0//x.shape[0]), tmpxpypzn.imag)
                cuda.atomic.add(real_ftau, (    (mx + mmx) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3, i0//x.shape[0]), tmpxpynzn.real) #x  1, y -1, z  1
                cuda.atomic.add(imag_ftau, (    (mx + mmx) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3, i0//x.shape[0]), tmpxpynzn.imag)
                cuda.atomic.add(real_ftau, ((mx - mmx - 1) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3, i0//x.shape[0]), tmpxnypzn.real) #x -1, y  1, z  1
                cuda.atomic.add(imag_ftau, ((mx - mmx - 1) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3, i0//x.shape[0]), tmpxnypzn.imag)
                cuda.atomic.add(real_ftau, ((mx - mmx - 1) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3, i0//x.shape[0]), tmpxnynzn.real) #x -1, y -1, z  1
                cuda.atomic.add(imag_ftau, ((mx - mmx - 1) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3, i0//x.shape[0]), tmpxnynzn.imag)
                E2mmz *= E2z
            E2mmy *= E2y
        E2mmx *= E2x

@numba.jit(nopython=True, nogil=True)
def build_grid_3d21_fast( x, y, z, dcf, ctmp, fntau, tau, nspread, E3 ):
    nf1 = fntau.shape[0]
    nf2 = fntau.shape[1]
    nf3 = fntau.shape[2]
    hx = 2 * np.pi / nf1
    hy = 2 * np.pi / nf2
    hz = 2 * np.pi / nf3
    ftau  = np.zeros(fntau.shape, dtype = fntau.dtype)

    #c = np.zeros(x.shape,dtype = fntau.dtype)
    # precompute some exponents
    for p in range(nspread + 1):
        for l in range(nspread + 1):
            for j in range(nspread + 1):
                E3[j,l,p] = np.exp(-((np.pi * j / nf1) ** 2 + (np.pi * l / nf2) ** 2 + (np.pi * p / nf3) ** 2)/ tau)
    # spread values onto ftau
    for i in range(x.shape[0]):
        ctmp = 0.0 * ctmp
        xi = x[i] % (2 * np.pi) #x
        yi = y[i] % (2 * np.pi) #y
        zi = z[i] % (2 * np.pi)
        mx = 1 + int(xi // hx) #index for the closest grid point
        my = 1 + int(yi // hy)
        mz = 1 + int(zi // hz)
        xi = (xi - hx * mx) #
        yi = (yi - hy * my)
        zi = (zi - hz * mz)
        E1 = np.exp(-0.25 * (xi ** 2 + yi ** 2 + zi ** 2) / tau) #missed zi ** 2 find@20170415
        E2x = np.exp((xi * np.pi) / (nf1 * tau))
        E2y = np.exp((yi * np.pi) / (nf2 * tau))
        E2z = np.exp((zi * np.pi) / (nf3 * tau))

        E2mmx = 1
        for mmx in range(nspread):
            E2mmy = 1
            for mmy in range(nspread):#use the symmetry of E1, E2 and E3, e.g. 1/(E2(mmx)*E2x) = E2(mx-mmx)
                E2mmz = 1
                for mmz in range(nspread):
                    ctmp += fntau[    (mx + mmx) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3] * E1 * E2mmx       * E2mmy       * E2mmz        * E3[    mmx,     mmy,     mmz]
                    ctmp += fntau[    (mx + mmx) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3] * E1 * E2mmx       / (E2mmy*E2y) * E2mmz        * E3[    mmx, mmy + 1,     mmz]
                    ctmp += fntau[(mx - mmx - 1) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3] * E1 / (E2mmx*E2x) * E2mmy       * E2mmz        * E3[mmx + 1,     mmy,     mmz]
                    ctmp += fntau[(mx - mmx - 1) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3] * E1 / (E2mmx*E2x) / (E2mmy*E2y) * E2mmz        * E3[mmx + 1, mmy + 1,     mmz]
                    ctmp += fntau[    (mx + mmx) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3] * E1 * E2mmx       * E2mmy       / (E2mmz*E2z)  * E3[    mmx,     mmy, mmz + 1]
                    ctmp += fntau[    (mx + mmx) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3] * E1 * E2mmx       / (E2mmy*E2y) / (E2mmz*E2z)  * E3[    mmx, mmy + 1, mmz + 1]
                    ctmp += fntau[(mx - mmx - 1) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3] * E1 / (E2mmx*E2x) * E2mmy       / (E2mmz*E2z)  * E3[mmx + 1,     mmy, mmz + 1]
                    ctmp += fntau[(mx - mmx - 1) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3] * E1 / (E2mmx*E2x) / (E2mmy*E2y) / (E2mmz*E2z)  * E3[mmx + 1, mmy + 1, mmz + 1]
                    E2mmz *= E2z
                E2mmy *= E2y
            E2mmx *= E2x
        
        ctmp = dcf[i] * ctmp/(nf1*nf2*nf3)

        E2mmx = 1
        for mmx in range(nspread):
            E2mmy = 1
            for mmy in range(nspread):#use the symmetry of E1, E2 and E3, e.g. 1/(E2(mmx)*E2x) = E2(mx-mmx)
                E2mmz = 1
                for mmz in range(nspread):
                    ftau[    (mx + mmx) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3] += E1 * ctmp * E2mmx       * E2mmy       * E2mmz        * E3[    mmx,     mmy,     mmz]
                    ftau[    (mx + mmx) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3] += E1 * ctmp * E2mmx       / (E2mmy*E2y) * E2mmz        * E3[    mmx, mmy + 1,     mmz]
                    ftau[(mx - mmx - 1) % nf1,     (my + mmy) % nf2,     (mz + mmz) % nf3] += E1 * ctmp / (E2mmx*E2x) * E2mmy       * E2mmz        * E3[mmx + 1,     mmy,     mmz]
                    ftau[(mx - mmx - 1) % nf1, (my - mmy - 1) % nf2,     (mz + mmz) % nf3] += E1 * ctmp / (E2mmx*E2x) / (E2mmy*E2y) * E2mmz        * E3[mmx + 1, mmy + 1,     mmz]
                    ftau[    (mx + mmx) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3] += E1 * ctmp * E2mmx       * E2mmy       / (E2mmz*E2z)  * E3[    mmx,     mmy, mmz + 1]
                    ftau[    (mx + mmx) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3] += E1 * ctmp * E2mmx       / (E2mmy*E2y) / (E2mmz*E2z)  * E3[    mmx, mmy + 1, mmz + 1]
                    ftau[(mx - mmx - 1) % nf1,     (my + mmy) % nf2, (mz - mmz - 1) % nf3] += E1 * ctmp / (E2mmx*E2x) * E2mmy       / (E2mmz*E2z)  * E3[mmx + 1,     mmy, mmz + 1]
                    ftau[(mx - mmx - 1) % nf1, (my - mmy - 1) % nf2, (mz - mmz - 1) % nf3] += E1 * ctmp / (E2mmx*E2x) / (E2mmy*E2y) / (E2mmz*E2z)  * E3[mmx + 1, mmy + 1, mmz + 1]
                    E2mmz *= E2z
                E2mmy *= E2y
            E2mmx *= E2x
    return ftau

@numba.njit(nopython=True, nogil=True, parallel = True )
def build_grid_array_3d21_fast( x, y, z, dcf, ctmp, fntau, tau, nspread, E3, nt):
    for t in numba.prange(nt):
        fntau[...,t]=build_grid_3d21_fast( x, y, z, dcf, 0.0, fntau[...,t], tau, nspread, E3)
    return fntau

def build_grid_3d21_fast_wrap( x, y, z, dcf, ctmp, fntau, tau, nspread, E3):
    if len(fntau.shape) is 3:
        build_grid_3d21_fast( x, y, z, dcf, ctmp, fntau, tau, nspread, E3 )

    elif len(fntau.shape) is 4:
        nt      = np.prod(fntau.shape[3:])
        build_grid_array_3d21_fast( x, y, z, dcf, ctmp, fntau, tau, nspread, E3, nt )
    elif len(fntau.shape) > 4:
        nt       = np.prod(fntau.shape[3:])
        tmp_dims = fntau.shape[3:]
        # flatten dims > 3
        fntau = np.reshape(fntau, fntau.shape[0:3] + (nt,))
        build_grid_array_3d21_fast( x, y, z, dcf, ctmp, fntau, tau, nspread, E3, nt )
        fntau = np.reshape(fntau, fntau.shape[0:3] + tmp_dims)    
    return fntau
