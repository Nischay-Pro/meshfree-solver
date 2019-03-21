import numpy as np
import point
from progress import printProgressBar

def convert_globaldata_to_gpu_globaldata(globaldata):
    point_dtype = np.dtype([('localID', np.int32),
                            ('x', np.float64),
                            ('y', np.float64),
                            ('left', np.int32),
                            ('right', np.int32),
                            ('flag_1', np.int8),
                            ('flag_2', np.int8),
                            ('nbhs', np.int8),
                            ('conn', np.int32, (20,)),
                            ('nx', np.float64),
                            ('ny', np.float64),
                            ('prim', np.float64, (4,)),
                            ('flux_res', np.float64, (4,)),
                            ('q', np.float64, (4,)),
                            ('dq', np.float64, (2, 4)),
                            ('entropy', np.float64),
                            ('xpos_nbhs', np.int8),
                            ('xneg_nbhs', np.int8),
                            ('ypos_nbhs', np.int8),
                            ('yneg_nbhs', np.int8),
                            ('xpos_conn', np.int32, (10,)),
                            ('xneg_conn', np.int32, (10,)),
                            ('ypos_conn', np.int32, (10,)),
                            ('yneg_conn', np.int32, (10,)),
                            ('delta', np.float64)], align=True)
    temp = np.zeros(len(globaldata), dtype=point_dtype)
    for idx in range(len(globaldata)):
        printProgressBar(
            idx, len(globaldata) - 1, prefix="Progress:", suffix="Complete", length=50
        )
        if idx > 0:
            temp[idx]['localID'] = globaldata[idx].localID
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
            temp[idx]['flux_res'] = globaldata[idx].flux_res
            temp[idx]['q'] = globaldata[idx].q
            temp[idx]['dq'] = globaldata[idx].dq
            temp[idx]['entropy'] = globaldata[idx].entropy
            temp[idx]['xpos_nbhs'] = globaldata[idx].xpos_nbhs
            temp[idx]['xneg_nbhs'] = globaldata[idx].xneg_nbhs
            temp[idx]['ypos_nbhs'] = globaldata[idx].ypos_nbhs
            temp[idx]['yneg_nbhs'] = globaldata[idx].yneg_nbhs
            xpos_conn = globaldata[idx].xpos_conn
            N = 10 - len(xpos_conn)
            xpos_conn = np.pad(xpos_conn, (0, N), 'constant')
            temp[idx]['xpos_conn'] = xpos_conn
            xneg_conn = globaldata[idx].xneg_conn
            N = 10 - len(xneg_conn)
            xneg_conn = np.pad(xneg_conn, (0, N), 'constant')
            temp[idx]['xneg_conn'] = xneg_conn
            ypos_conn = globaldata[idx].ypos_conn
            N = 10 - len(ypos_conn)
            ypos_conn = np.pad(ypos_conn, (0, N), 'constant')
            temp[idx]['ypos_conn'] = ypos_conn
            yneg_conn = globaldata[idx].yneg_conn
            N = 10 - len(yneg_conn)
            yneg_conn = np.pad(yneg_conn, (0, N), 'constant')
            temp[idx]['yneg_conn'] = yneg_conn
            temp[idx]['delta'] = globaldata[idx].delta
    return temp

def gpu_point(localID, x, y, left, right, flag_1, flag_2, nbhs, conn, nx, ny, prim, flux_res, q, dq, entropy, xpos_nbhs, xneg_nbhs, ypos_nbhs, yneg_nbhs, xpos_conn, xneg_conn, ypos_conn, yneg_conn, delta, globaldata):
    idx = localID
    globaldata[idx]['localID'] = localID
    globaldata[idx]['x'] = x
    globaldata[idx]['y'] = y
    globaldata[idx]['left'] = left
    globaldata[idx]['right'] = right
    globaldata[idx]['flag_1'] = flag_1
    globaldata[idx]['flag_2'] = flag_2
    globaldata[idx]['nbhs'] = nbhs
    conn = conn
    N = 20 - len(conn)
    conn = np.pad(conn, (0, N), 'constant')
    globaldata[idx]['conn'] = conn
    if nx != None:
        globaldata[idx]['nx'] = nx
    if ny != None:
        globaldata[idx]['ny'] = ny
    if prim != None:
        globaldata[idx]['prim'] = prim
    if flux_res != None:
        globaldata[idx]['flux_res'] = flux_res
    if q != None:
        globaldata[idx]['q'] = q
    if dq != None:
        globaldata[idx]['dq'] = dq
    if entropy != None:
        globaldata[idx]['entropy'] = entropy
    if xpos_nbhs != None:
        globaldata[idx]['xpos_nbhs'] = xpos_nbhs
        globaldata[idx]['xneg_nbhs'] = xneg_nbhs
        globaldata[idx]['ypos_nbhs'] = ypos_nbhs
        globaldata[idx]['yneg_nbhs'] = yneg_nbhs
        xpos_conn = xpos_conn
        N = 10 - len(xpos_conn)
        xpos_conn = np.pad(xpos_conn, (0, N), 'constant')
        globaldata[idx]['xpos_conn'] = xpos_conn
        xneg_conn = xneg_conn
        N = 10 - len(xneg_conn)
        xneg_conn = np.pad(xneg_conn, (0, N), 'constant')
        globaldata[idx]['xneg_conn'] = xneg_conn
        ypos_conn = ypos_conn
        N = 10 - len(ypos_conn)
        ypos_conn = np.pad(ypos_conn, (0, N), 'constant')
        globaldata[idx]['ypos_conn'] = ypos_conn
        yneg_conn = yneg_conn
        N = 10 - len(yneg_conn)
        yneg_conn = np.pad(yneg_conn, (0, N), 'constant')
        globaldata[idx]['yneg_conn'] = yneg_conn
    if delta != None:
        globaldata[idx]['delta'] = delta
    return globaldata
    

def convert_gpu_globaldata_to_globaldata(globaldata):
    globaldata_cpu = np.zeros(len(globaldata), dtype=object)
    for idx in range(len(globaldata)):
        itm = globaldata[idx]
        conn = itm['conn'][:itm['nbhs']]
        xpos_conn = itm['xpos_conn'][:itm['xpos_nbhs']]
        xneg_conn = itm['xneg_conn'][:itm['xneg_nbhs']]
        ypos_conn = itm['ypos_conn'][:itm['ypos_nbhs']]
        yneg_conn = itm['yneg_conn'][:itm['yneg_nbhs']]
        temp = point.Point(itm['localID'], itm['x'], itm['y'], itm['left'], itm['right'], itm['flag_1'], itm['flag_2'], itm['nbhs'], conn, itm['nx'], itm['ny'], itm['prim'], itm['flux_res'], itm['q'], itm['dq'], itm['entropy'], itm['xpos_nbhs'], itm['xneg_nbhs'], itm['ypos_nbhs'], itm['yneg_nbhs'], xpos_conn, xneg_conn, ypos_conn, yneg_conn, itm['delta'])
        globaldata_cpu[idx] = temp
    return globaldata_cpu

def initglobaldata(globaldata, length):
    point_dtype = np.dtype([('localID', np.int32),
                            ('x', np.float64),
                            ('y', np.float64),
                            ('left', np.int32),
                            ('right', np.int32),
                            ('flag_1', np.int8),
                            ('flag_2', np.int8),
                            ('nbhs', np.int8),
                            ('conn', np.int32, (20,)),
                            ('nx', np.float64),
                            ('ny', np.float64),
                            ('prim', np.float64, (4,)),
                            ('flux_res', np.float64, (4,)),
                            ('q', np.float64, (4,)),
                            ('dq', np.float64, (2, 4)),
                            ('entropy', np.float64),
                            ('xpos_nbhs', np.int8),
                            ('xneg_nbhs', np.int8),
                            ('ypos_nbhs', np.int8),
                            ('yneg_nbhs', np.int8),
                            ('xpos_conn', np.int32, (10,)),
                            ('xneg_conn', np.int32, (10,)),
                            ('ypos_conn', np.int32, (10,)),
                            ('yneg_conn', np.int32, (10,)),
                            ('delta', np.float64)], align=True)
    globaldata = np.zeros(length, dtype=point_dtype)
    return globaldata