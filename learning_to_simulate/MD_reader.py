import pipes
import numpy as np
import io

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dumps', nargs='*', help='dumps')
    args = parser.parse_args()
    for fn in args.dumps:
        read_dump(fn, write=True)


def arr2str1d(m):
    return ' '.join(map(str, m))

def matrix2text(m_in, separator='\n'):
    try:
        m=np.array(m_in)
    except:
        # m is not a matrix, e.g. [[1,2], [3,4,5]]
        return separator.join([arr2str1d(x) for x in m_in])
    if len(m.shape) <= 1:
        return arr2str1d(m)
    else:
        return separator.join([arr2str1d(x) for x in m])


def read_dump(fn, write=True):
    # lines = open(fn, 'r').readlines()
    with open(fn) as f:
        lines = f.read().splitlines() 
    # t1 = pipes.Template()
    # t1.append("awk 'NR==4' ", '--')
    # t2 = pipes.Template()
    # t2.append("awk '(NR>=6) && (NR<9)' ", '--')
    # t_pbc = pipes.Template()
    # t_pbc.append("awk 'NR==5 {print $4,$5,$6}' ", '--')
    # tstep = np.loadtxt(t0.open(fn, 'r')).ravel()[0]
    # natom = int(np.loadtxt(t1.open(fn, 'r')).ravel()[0])
    # cell = np.diag(np.loadtxt(t2.open(fn, 'r'))[:,1])
    # pbc = t_pbc.open(fn, 'r').read().strip('\n')
    # natom = int(np.loadtxt(t1.open(fn, 'r')).ravel()[0])
    natom = int(lines[3])
    pbc = lines[4].replace("ITEM: BOX BOUNDS ","")
    cell = np.diag(np.loadtxt(io.StringIO('\n'.join(lines[5:8])))[:,1])
    if write:
        outf = open(f'MD_DATA.{fn}', 'w')
        outf.write(f'#\n#PBC= {pbc}\n')
    nl = natom+9
    lattices = []
    trajectories = []
    for t in range(len(lines)//nl):
        lines_this = lines[nl*t:nl*t+nl]
        tstep = int(lines_this[1])
        lattice_vec = np.loadtxt(io.StringIO('\n'.join(lines_this[5:8])))
        lattices.append(np.diag(lattice_vec[:,1]-lattice_vec[:,0]))
        dat = np.loadtxt(io.StringIO('\n'.join(lines_this[-natom:])))
        trajectories.append(dat)
        # print('debug', dat, lines_this)
        if write:
            outf.write(f'iter= {tstep}\n{matrix2text(cell)}\n{natom}\n')
            outf.write(matrix2text(dat[:,1:8]))
        ke = np.sum(dat[:,-2])
        pe = np.sum(dat[:,-1])
        temperature = np.mean(dat[:,-2])*2/3
        if write:
            outf.write(f'\nel-ion E= {pe} eV\nKIN E= {ke} eV\ntotal E= {ke+pe} eV\nTemp.= {temperature} K\n')
    if write:
        outf.close()
    else:
        return [s=='pp' for s in pbc.strip().split()], trajectories[0][:,1].astype(int)-1, np.array(lattices), np.array(trajectories)[:,:,2:5]
