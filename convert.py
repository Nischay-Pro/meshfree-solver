import numpy as np

def convert_globaldata_to_gpu_globaldata(globaldata):
    point_dtype = np.dtype([('localID', np.int32),
                            ('x', np.float64),
                            ('y', np.float64),
                            ('left', np.int32),
                            ('right', np.int32),
                            ('flag_1', np.int32),
                            ('flag_2', np.int32),
                            ('nbhs', np.int32),
                            ('conn', np.int32, (20,)),
                            ('nx', np.float64),
                            ('ny', np.float64),
                            ('prim', np.float64, (4,)),
                            ('flux_res', np.float64, (4,)),
                            ('q', np.float64, (4,)),
                            ('dq', np.float64, (2, 4)),
                            ('entropy', np.float64),
                            ('xpos_nbhs', np.int32),
                            ('xneg_nbhs', np.int32),
                            ('ypos_nbhs', np.int32),
                            ('yneg_nbhs', np.int32),
                            ('xpos_conn', np.float64, (20,)),
                            ('xneg_conn', np.float64, (20,)),
                            ('ypos_conn', np.float64, (20,)),
                            ('yneg_conn', np.float64, (20,)),
                            ('delta', np.float64)], align=True)
    temp = np.zeros(len(globaldata), dtype=point_dtype)
    for idx in range(len(globaldata)):
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
            N = 20 - len(conn)
            xpos_conn = np.pad(conn, (0, N), 'constant')
            temp[idx]['xpos_conn'] = xpos_conn
            xneg_conn = globaldata[idx].xneg_conn
            N = 20 - len(conn)
            xneg_conn = np.pad(conn, (0, N), 'constant')
            temp[idx]['xneg_conn'] = xneg_conn
            ypos_conn = globaldata[idx].ypos_conn
            N = 20 - len(conn)
            ypos_conn = np.pad(conn, (0, N), 'constant')
            temp[idx]['ypos_conn'] = ypos_conn
            yneg_conn = globaldata[idx].yneg_conn
            N = 20 - len(conn)
            yneg_conn = np.pad(conn, (0, N), 'constant')
            temp[idx]['yneg_conn'] = yneg_conn
            temp[idx]['delta'] = globaldata[idx].delta
    
    return temp    