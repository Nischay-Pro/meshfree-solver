import numpy as np
import point
import sys
try:
    from mpi4py import MPI
except:
    pass

def sync_ghost(globaldata_local, globaldata_ghost, globaldata_table, comm, foreign_communicators, sync_tags):
    communication_array = {}
    for itm in globaldata_ghost.keys():
        val = globaldata_ghost[itm]
        ghost_core = val.foreign_core
        if ghost_core in communication_array.keys():
            communication_array[ghost_core].append(val.globalID)
        else:
            communication_array[ghost_core] = [val.globalID]
    
    wait_queue = []
    keys_list = list(communication_array.keys())
    keys_list.sort()
    # print(foreign_communicators)
    if foreign_communicators != None:
        for keys in keys_list:
            comm.send(communication_array[keys], dest=keys)

        for i in foreign_communicators:
            data = comm.recv(source=i)
            result = {}
            for tag in sync_tags:
                for itm in data:
                    localID = globaldata_table[itm]
                    result[itm] = reducedStructure(globaldata_local[localID], tag)
                comm.send(result, dest=i, tag=tag)
    
    else:

        foreign_communicators = []
        for i in range(comm.Get_size()):
            if i != comm.Get_rank():
                req = None
                if i in communication_array.keys():
                    req = comm.isend(communication_array[i], dest=i)
                else:
                    req = comm.isend([None], dest=i)
                wait_queue.append(req)

        for itm in wait_queue:
            MPI.Request.Wait(itm)

        for i in range(comm.Get_size()):
            if i != comm.Get_rank():
                data = comm.recv(source=i)
                if data[0] == None:
                    pass
                else:
                    foreign_communicators.append(i)
                    result = {}
                    for tag in sync_tags:
                        for itm in data:
                            localID = globaldata_table[itm]
                            result[itm] = reducedStructure(globaldata_local[localID], tag)
                        comm.send(result, dest=i, tag=tag)
                    
        foreign_communicators.sort()

    comm.Barrier()

    for itm in keys_list:
        for tag in sync_tags:
            data = comm.recv(source=itm, tag=tag)
            for key in data.keys():
                localID = globaldata_table[key]
                response = data[key]
                if tag == 0:
                    globaldata_ghost[localID].dq = np.array(response)
                elif tag == 1:
                    globaldata_ghost[localID].minq = np.array(response)
                elif tag == 2:
                    globaldata_ghost[localID].maxq = np.array(response)
                elif tag == 3:
                    globaldata_ghost[localID].ds = response
                elif tag == 4:
                    globaldata_ghost[localID].prim = np.array(response)

    # print("Updated Successfully")

    return globaldata_ghost, foreign_communicators

def reducedStructure(point, tag):
    if tag == 0:
        dq = np.array(point.dq).tolist()
        return dq
    elif tag == 1:
        minq = np.array(point.minq).tolist()
        return minq
    elif tag == 2:
        maxq = np.array(point.maxq).tolist()
        return maxq
    elif tag == 3:
        ds = point.ds
        return ds
    elif tag == 4:
        prim = np.array(point.prim).tolist()
        return prim

# def convert_to_numpy(point_obj):
#     point_dtype = np.dtype([('localID', np.int32),
#                             ('x', np.float64),
#                             ('y', np.float64),
#                             ('left', np.int32),
#                             ('right', np.int32),
#                             ('flag_1', np.int32),
#                             ('flag_2', np.int32),
#                             ('nbhs', np.int32),
#                             ('conn', np.int32, (20,)),
#                             ('nx', np.float64),
#                             ('ny', np.float64),
#                             ('prim', np.float64, (4,)),
#                             ('flux_res', np.float64, (4,)),
#                             ('q', np.float64, (4,)),
#                             ('dq', np.float64, (2, 4)),
#                             ('entropy', np.float64),
#                             ('xpos_nbhs', np.int32),
#                             ('xneg_nbhs', np.int32),
#                             ('ypos_nbhs', np.int32),
#                             ('yneg_nbhs', np.int32),
#                             ('xpos_conn', np.int32, (20,)),
#                             ('xneg_conn', np.int32, (20,)),
#                             ('ypos_conn', np.int32, (20,)),
#                             ('yneg_conn', np.int32, (20,)),
#                             ('delta', np.float64),
#                             ('foreign', np.bool_),
#                             ('foreign_core', np.int32),
#                             ('globalID', np.int32)], align=True)
#     temp = np.zeros((1), dtype=point_dtype)
#     temp['localID'] = point_obj.localID
#     temp['x'] = point_obj.x
#     temp['y'] = point_obj.y
#     temp['left'] = point_obj.left
#     temp['right'] = point_obj.right
#     temp['flag_1'] = point_obj.flag_1
#     temp['flag_2'] = point_obj.flag_2
#     temp['nbhs'] = point_obj.nbhs
#     conn = point_obj.conn
#     N = 20 - len(conn)
#     conn = np.pad(conn, (0, N), 'constant')
#     temp['conn'] = conn
#     temp['nx'] = point_obj.nx
#     temp['ny'] = point_obj.ny
#     temp['prim'] = point_obj.prim
#     temp['flux_res'] = point_obj.flux_res
#     temp['q'] = point_obj.q
#     temp['dq'] = point_obj.dq
#     temp['entropy'] = point_obj.entropy
#     temp['xpos_nbhs'] = point_obj.xpos_nbhs
#     temp['xneg_nbhs'] = point_obj.xneg_nbhs
#     temp['ypos_nbhs'] = point_obj.ypos_nbhs
#     temp['yneg_nbhs'] = point_obj.yneg_nbhs
#     xpos_conn = point_obj.xpos_conn
#     N = 20 - len(xpos_conn)
#     xpos_conn = np.pad(xpos_conn, (0, N), 'constant')
#     temp['xpos_conn'] = xpos_conn
#     xneg_conn = point_obj.xneg_conn
#     N = 20 - len(xneg_conn)
#     xneg_conn = np.pad(xneg_conn, (0, N), 'constant')
#     temp['xneg_conn'] = xneg_conn
#     ypos_conn = point_obj.ypos_conn
#     N = 20 - len(ypos_conn)
#     ypos_conn = np.pad(ypos_conn, (0, N), 'constant')
#     temp['ypos_conn'] = ypos_conn
#     yneg_conn = point_obj.yneg_conn
#     N = 20 - len(yneg_conn)
#     yneg_conn = np.pad(yneg_conn, (0, N), 'constant')
#     temp['yneg_conn'] = yneg_conn
#     temp['delta'] = point_obj.delta
#     temp['foreign'] = point_obj.foreign
#     temp['foreign_core'] = point_obj.foreign_core
#     temp['globalID'] = point_obj.globalID
#     return temp

# def convert_from_numpy(point_obj):
#         itm = point_obj
#         conn = itm['conn'][:itm['nbhs']]
#         xpos_conn = itm['xpos_conn'][:itm['xpos_nbhs']]
#         xneg_conn = itm['xneg_conn'][:itm['xneg_nbhs']]
#         ypos_conn = itm['ypos_conn'][:itm['ypos_nbhs']]
#         yneg_conn = itm['yneg_conn'][:itm['yneg_nbhs']]
#         temp = point.Point(itm['localID'], itm['x'], itm['y'], itm['left'], itm['right'], itm['flag_1'], itm['flag_2'], itm['nbhs'], conn, itm['nx'], itm['ny'], itm['prim'], itm['flux_res'], itm['q'], itm['dq'], itm['entropy'], itm['xpos_nbhs'], itm['xneg_nbhs'], itm['ypos_nbhs'], itm['yneg_nbhs'], xpos_conn, xneg_conn, ypos_conn, yneg_conn, itm['delta'], itm['foreign'], itm['foreign_core'], itm['globalID'])
#         return temp