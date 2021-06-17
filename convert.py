import numpy as np
import point
from tqdm import trange

def pointToNumpy(x, y, left, right, flag_1, flag_2, nbhs, conn, nx, ny, prim, prim_old, flux_res, q, dq, entropy, xpos_nbhs, xneg_nbhs, ypos_nbhs, yneg_nbhs, xpos_conn, xneg_conn, ypos_conn, yneg_conn, delta, min_dist):
    pres = np.float64
    point_dtype = np.dtype([('x', np.float64),
                            ('y', np.float64),
                            ('left', np.int32),
                            ('right', np.int32),
                            ('flag_1', np.int8),
                            ('flag_2', np.int8),
                            ('nbhs', np.int8),
                            ('conn', np.int32, (20,)),
                            ('nx', pres),
                            ('ny', pres),
                            ('prim', pres, (4,)),
                            ('prim_old', pres, (4,)),
                            ('flux_res', pres, (4,)),
                            ('q', pres, (4,)),
                            ('dq', pres, (2, 4)),
                            ('entropy', pres),
                            ('xpos_nbhs', np.int8),
                            ('xneg_nbhs', np.int8),
                            ('ypos_nbhs', np.int8),
                            ('yneg_nbhs', np.int8),
                            ('xpos_conn', np.int32, (20,)),
                            ('xneg_conn', np.int32, (20,)),
                            ('ypos_conn', np.int32, (20,)),
                            ('yneg_conn', np.int32, (20,)),
                            ('delta', pres),
                            ('min_dist', pres)], align=True)
    temp = np.zeros(1, dtype=point_dtype)
    temp['x'] = x
    temp['y'] = y
    temp['left'] = left
    temp['right'] = right
    temp['flag_1'] = flag_1
    temp['flag_2'] = flag_2
    temp['nbhs'] = nbhs
    conn = conn
    N = 20 - len(conn)
    conn = np.pad(conn, (0, N), 'constant')
    temp['conn'] = conn
    temp['nx'] = nx
    temp['ny'] = ny
    temp['prim'] = prim
    temp['prim_old'] = np.zeros((4,))
    temp['flux_res'] = flux_res
    temp['q'] = q
    temp['dq'] = dq
    temp['xpos_nbhs'] = xpos_nbhs
    temp['xneg_nbhs'] = xneg_nbhs
    temp['ypos_nbhs'] = ypos_nbhs
    temp['yneg_nbhs'] = yneg_nbhs
    xpos_conn = xpos_conn
    N = 20 - len(xpos_conn)
    xpos_conn = np.pad(xpos_conn, (0, N), 'constant')
    temp['xpos_conn'] = xpos_conn
    xneg_conn = xneg_conn
    N = 20 - len(xneg_conn)
    xneg_conn = np.pad(xneg_conn, (0, N), 'constant')
    temp['xneg_conn'] = xneg_conn
    ypos_conn = ypos_conn
    N = 20 - len(ypos_conn)
    ypos_conn = np.pad(ypos_conn, (0, N), 'constant')
    temp['ypos_conn'] = ypos_conn
    yneg_conn = yneg_conn
    N = 20 - len(yneg_conn)
    yneg_conn = np.pad(yneg_conn, (0, N), 'constant')
    temp['yneg_conn'] = yneg_conn
    temp['delta'] = delta
    temp['min_dist'] = min_dist
    return temp

def convert_globaldata_to_gpu_globaldata_new(globaldata):
    maxconn, maxxpconn, maxxnegconn, maxypconn, maxynegconn = 0,0,0,0,0
    for idx in trange(1, len(globaldata)):
        if globaldata[idx].nbhs > maxconn:
            maxconn = globaldata[idx].nbhs
        if globaldata[idx].xpos_nbhs > maxxpconn:
            maxxpconn = globaldata[idx].xpos_nbhs
        if globaldata[idx].xneg_nbhs > maxxnegconn:
            maxxnegconn = globaldata[idx].xneg_nbhs
        if globaldata[idx].ypos_nbhs > maxypconn:
            maxypconn = globaldata[idx].ypos_nbhs
        if globaldata[idx].yneg_nbhs > maxynegconn:
            maxynegconn = globaldata[idx].yneg_nbhs
    
    print("Numpy Dynamic Array Creation: {} {} {} {} {}".format(maxconn, maxxpconn, maxxnegconn, maxypconn, maxynegconn))

    x = np.zeros(len(globaldata), dtype=np.float64, order="F")
    y = np.zeros(len(globaldata), dtype=np.float64, order="F")
    left = np.zeros(len(globaldata), dtype=np.int32, order="F")
    right = np.zeros(len(globaldata), dtype=np.int32, order="F")
    flag_1 = np.zeros(len(globaldata), dtype=np.int8, order="F")
    flag_2 = np.zeros(len(globaldata), dtype=np.int8, order="F")
    nbhs = np.zeros(len(globaldata), dtype=np.int8, order="F")
    conn = np.zeros((len(globaldata), maxconn), dtype=np.int32, order="F")
    nx = np.zeros(len(globaldata), np.float64, order="F")
    ny = np.zeros(len(globaldata), np.float64, order="F")
    prim = np.zeros((len(globaldata), 4), np.float64, order="F")
    prim_old = np.zeros((len(globaldata), 4), np.float64, order="F")
    flux_res = np.zeros((len(globaldata), 4), np.float64, order="F")
    q = np.zeros((len(globaldata), 4), np.float64, order="F")
    dq = np.zeros((len(globaldata), 2, 4), np.float64, order="F")
    # entropy = np.zeros(len(globaldata), np.float64)
    xpos_nbhs = np.zeros(len(globaldata), np.int8, order="F")
    xneg_nbhs = np.zeros(len(globaldata), np.int8, order="F")
    ypos_nbhs = np.zeros(len(globaldata), np.int8, order="F")
    yneg_nbhs = np.zeros(len(globaldata), np.int8, order="F")
    xpos_conn = np.zeros((len(globaldata), maxxpconn), np.int8, order="F")
    xneg_conn = np.zeros((len(globaldata), maxxnegconn), np.int8, order="F")
    ypos_conn = np.zeros((len(globaldata), maxypconn), np.int8, order="F")
    yneg_conn = np.zeros((len(globaldata), maxynegconn), np.int8, order="F")
    delta = np.zeros(len(globaldata), np.float64, order="F")
    min_dist = np.zeros(len(globaldata), np.float64, order="F")
    maxminq = np.zeros((len(globaldata), 2, 4), np.float64, order="F")
    inner1 = np.zeros((len(globaldata), 4), np.float64, order="F")
    inner2 = np.zeros((len(globaldata), 4), np.float64, order="F")
    
    for idx in trange(1, len(globaldata)):
        x[idx] = globaldata[idx].x
        y[idx] = globaldata[idx].y
        left[idx] = globaldata[idx].left
        right[idx] = globaldata[idx].right
        flag_1[idx] = globaldata[idx].flag_1
        flag_2[idx] = globaldata[idx].flag_2
        nbhs[idx] = globaldata[idx].nbhs
        connt = globaldata[idx].conn
        for i in range(len(connt)):
            conn[idx][i] = connt[i]
        nx[idx] = globaldata[idx].nx
        ny[idx] = globaldata[idx].ny
        prim[idx] = globaldata[idx].prim
        prim_old[idx] = np.zeros((4,))
        flux_res[idx] = globaldata[idx].flux_res
        q[idx] = globaldata[idx].q
        dq[idx] = globaldata[idx].dq

        xpos_nbhs[idx] = globaldata[idx].xpos_nbhs
        xneg_nbhs[idx] = globaldata[idx].xneg_nbhs
        ypos_nbhs[idx] = globaldata[idx].ypos_nbhs
        yneg_nbhs[idx] = globaldata[idx].yneg_nbhs
        
        xpos_connt = globaldata[idx].xpos_conn
        for i in range(len(xpos_connt)):
            xpos_conn[idx][i] = connt.index(xpos_connt[i])
        xneg_connt = globaldata[idx].xneg_conn
        for i in range(len(xneg_connt)):
            xneg_conn[idx][i] = connt.index(xneg_connt[i])
        ypos_connt = globaldata[idx].ypos_conn
        for i in range(len(ypos_connt)):
            ypos_conn[idx][i] = connt.index(ypos_connt[i])
        yneg_connt = globaldata[idx].yneg_conn
        for i in range(len(yneg_connt)):
            yneg_conn[idx][i] = connt.index(yneg_connt[i])
        delta[idx] = globaldata[idx].delta
        min_dist[idx] = globaldata[idx].min_dist
    
    return x, y, left, right, flag_1, flag_2, nbhs, conn, nx, ny, prim, prim_old, flux_res, q, dq, xpos_nbhs, xneg_nbhs, ypos_nbhs, yneg_nbhs, xpos_conn, xneg_conn, ypos_conn, yneg_conn, delta, min_dist, maxminq, inner1, inner2

def convert_globaldata_to_gpu_globaldata(globaldata):
    pres = np.float64
    point_dtype = np.dtype([('x', np.float64),
                            ('y', np.float64),
                            ('left', np.int32),
                            ('right', np.int32),
                            ('flag_1', np.int8),
                            ('flag_2', np.int8),
                            ('nbhs', np.int8),
                            ('conn', np.int32, (20,)),
                            ('nx', pres),
                            ('ny', pres),
                            ('prim', pres, (4,)),
                            ('prim_old', pres, (4,)),
                            ('flux_res', pres, (4,)),
                            ('q', pres, (4,)),
                            ('dq', pres, (2, 4)),
                            ('entropy', pres),
                            ('xpos_nbhs', np.int8),
                            ('xneg_nbhs', np.int8),
                            ('ypos_nbhs', np.int8),
                            ('yneg_nbhs', np.int8),
                            ('xpos_conn', np.int32, (20,)),
                            ('xneg_conn', np.int32, (20,)),
                            ('ypos_conn', np.int32, (20,)),
                            ('yneg_conn', np.int32, (20,)),
                            ('delta', pres),
                            ('min_dist', pres),
                            ('maxminq', pres, (2, 4))], align=True)
    temp = np.zeros(len(globaldata), dtype=point_dtype)
    for idx in trange(1, len(globaldata)):
        temp[idx]['x'] = globaldata[idx].x
        temp[idx]['y'] = globaldata[idx].y
        temp[idx]['left'] = globaldata[idx].left
        temp[idx]['right'] = globaldata[idx].right
        temp[idx]['flag_1'] = globaldata[idx].flag_1
        temp[idx]['flag_2'] = globaldata[idx].flag_2
        temp[idx]['nbhs'] = globaldata[idx].nbhs
        conn = globaldata[idx].conn
        N = 20 - len(conn)
        conn = np.pad(conn, (0, N), 'constant')
        temp[idx]['conn'] = conn
        temp[idx]['nx'] = globaldata[idx].nx
        temp[idx]['ny'] = globaldata[idx].ny
        temp[idx]['prim'] = globaldata[idx].prim
        temp[idx]['prim_old'] = np.zeros((4,))
        temp[idx]['flux_res'] = globaldata[idx].flux_res
        temp[idx]['q'] = globaldata[idx].q
        temp[idx]['dq'] = globaldata[idx].dq
        temp[idx]['xpos_nbhs'] = globaldata[idx].xpos_nbhs
        temp[idx]['xneg_nbhs'] = globaldata[idx].xneg_nbhs
        temp[idx]['ypos_nbhs'] = globaldata[idx].ypos_nbhs
        temp[idx]['yneg_nbhs'] = globaldata[idx].yneg_nbhs
        xpos_conn = globaldata[idx].xpos_conn
        N = 20 - len(xpos_conn)
        xpos_conn = np.pad(xpos_conn, (0, N), 'constant')
        temp[idx]['xpos_conn'] = xpos_conn
        xneg_conn = globaldata[idx].xneg_conn
        N = 20 - len(xneg_conn)
        xneg_conn = np.pad(xneg_conn, (0, N), 'constant')
        temp[idx]['xneg_conn'] = xneg_conn
        ypos_conn = globaldata[idx].ypos_conn
        N = 20 - len(ypos_conn)
        ypos_conn = np.pad(ypos_conn, (0, N), 'constant')
        temp[idx]['ypos_conn'] = ypos_conn
        yneg_conn = globaldata[idx].yneg_conn
        N = 20 - len(yneg_conn)
        yneg_conn = np.pad(yneg_conn, (0, N), 'constant')
        temp[idx]['yneg_conn'] = yneg_conn
        temp[idx]['delta'] = globaldata[idx].delta
        temp[idx]['min_dist'] = globaldata[idx].min_dist
    return temp    

def convert_gpu_globaldata_to_globaldata(globaldata):
    globaldata_cpu = np.zeros(len(globaldata), dtype=object)
    for idx in trange(len(globaldata)):
        itm = globaldata[idx]
        conn = itm['conn'][:itm['nbhs']]
        xpos_conn = itm['xpos_conn'][:itm['xpos_nbhs']]
        xneg_conn = itm['xneg_conn'][:itm['xneg_nbhs']]
        ypos_conn = itm['ypos_conn'][:itm['ypos_nbhs']]
        yneg_conn = itm['yneg_conn'][:itm['yneg_nbhs']]
        temp = point.Point(idx, itm['x'], itm['y'], itm['left'], itm['right'], itm['flag_1'], itm['flag_2'], itm['nbhs'], conn, itm['nx'], itm['ny'], itm['prim'], itm['flux_res'], itm['q'], itm['dq'], None, itm['xpos_nbhs'], itm['xneg_nbhs'], itm['ypos_nbhs'], itm['yneg_nbhs'], xpos_conn, xneg_conn, ypos_conn, yneg_conn, itm['delta'], itm['min_dist'])
        globaldata_cpu[idx] = temp
    return globaldata_cpu