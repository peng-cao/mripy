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
