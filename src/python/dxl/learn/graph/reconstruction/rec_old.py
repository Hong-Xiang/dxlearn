# def make_recon_local(global_graph, local_graphs):
#   for g in local_graphs:
#     g.copy_to_global(global_graph)
#     g.recon_local()
#   global_graph.x_update_by_merge()

# def get_my_local_graph(local_graphs: Iterable[LocalGraph]):
#   host = ThisHost.host()
#   for g in local_graphs:
#     if g.graph_info.host == host:
#       return g
#   raise KeyError("No local graph for {}.{} found".format(
#       host.job, host.task_index))

# def run_step(master_op, worker_ops, global_graph, local_graphs):
#   if ThisHost.is_master():
#     _op = master_op
#   else:
#     _op = get_my_local_graph(local_graphs)
#   ThisSession.run(_op)

# def run_init_ops(master_op, worker_ops, global_graph: GlobalGraph,
#                  local_graphs: Iterable[LocalGraph]):
#   if ThisHost.is_master():
#     ThisSession.run(master_op)
#     print_tensor(global_graph.tensor(global_graph.KEYS.TENSOR.X), 'x:global')
#   else:
#     logger.info('Pre intialization.')
#     print_tensor(
#         global_graph.tensor(global_graph.KEYS.TENSOR.X),
#         'x:global direct fetch')
#     tid = ThisHost.host().task_index
#     ThisSession.run(worker_ops[tid])
#     lg = get_my_local_graph(local_graphs)
#     TK = lg.KEYS.TENSOR
#     ptensor(lg.tensor(TK.X), 'x:local')
#     # ptensor(lg.tensor(TK.SYSTEM_MATRIX), 'x:local')
#   logger.info('Intialization DONE. ==============================')

# def run_recon_step(master_op, worker_ops, global_graph, local_graphs):
#   if ThisHost.is_master():
#     logger.info('PRE RECON')
#     print_tensor(global_graph.tensor('x'))
#     logger.info('START RECON')
#     ThisSession.run(master_op)
#     logger.info('POST RECON')
#     ptensor(global_graph.tensor('x'), 'x:global')
#   else:
#     logger.info('PRE RECON')
#     lg = get_my_local_graph(local_graphs)
#     TK = lg.KEYS.TENSOR
#     print_tensor(lg.tensor(TK.X), 'x:local')
#     logger.info('POST RECON')
#     ThisSession.run(worker_ops[ThisHost.host().task_index])
#     # ptensor(lg.tensor(TK.X_UPDATE), 'x:update')
#     print_tensor(lg.tensor(TK.X_RESULT), 'x:result')
#     print_tensor(lg.tensor(TK.X_GLOBAL_BUFFER), 'x:global_buffer')
#     # ThisSession.run(ThisHost.host().task_index)

# def run_merge_step(master_op, worker_ops, global_graph, local_graphs):
#   if ThisHost.is_master():
#     ThisSession.run(master_op)

# def full_step_run(m_op,
#                   w_ops,
#                   global_graph,
#                   local_graphs,
#                   nb_iter=0,
#                   verbose=0):
#   if verbose > 0:
#     print('PRE RECON {}'.format(nb_iter))
#     lg = None
#     if ThisHost.is_master():
#       TK = global_graph.KEYS.TENSOR
#       ptensor(global_graph.tensor(TK.X), 'x:global')
#     else:
#       lg = get_my_local_graph(local_graphs)
#       TK = lg.KEYS.TENSOR
#       ptensor(lg.tensor(TK.X), 'x:local')
#   print('START RECON {}'.format(nb_iter))
#   if ThisHost.is_master():
#     ThisSession.run(m_op)
#   else:
#     ThisSession.run(w_ops[ThisHost.host().task_index])
#   if verbose > 0:
#     print('POST RECON {}'.format(nb_iter))
#     if ThisHost.is_master():
#       TK = global_graph.KEYS.TENSOR
#       ptensor(global_graph.tensor(TK.X), 'x:global')
#     else:
#       lg = get_my_local_graph(local_graphs)
#       TK = lg.KEYS.TENSOR
#       ptensor(lg.tensor(TK.X), 'x:local')

# def main(job, task):
#   hosts, hmi = dist_init(job, task)
#   global_graph = init_global(hmi)
#   local_graphs = init_local(global_graph, hosts)

#   m_op_init, w_ops_init = global_graph.init_op(local_graphs)
#   make_recon_local(global_graph, local_graphs)
#   m_op_rec, w_ops_rec = global_graph.recon_step(local_graphs, hosts)
#   m_op, w_ops = global_graph.merge_step(m_op_rec, w_ops_rec, hosts)
#   # global_tensors = {'x': x_t, 'y': y_t, 'sm': sm_t, 'em': e_t}
#   # g2l_init, update_global, x_g2l, x_l, y_l, sm_l, em_l = recon_init(
#   #     x_t, y_t, sm_t, e_t, hosts, x_init)
#   # gop, l_ops = recon_step(update_global, x_g2l, x_l, y_l, sm_l, em_l, hosts)
#   # init_op = make_init_op(g2l_init, hosts)

#   make_distribute_session()

#   tf.summary.FileWriter('./graph', ThisSession.session().graph)
#   print('|DEBUG| Make Graph done.')

#   init_run(m_op_init, w_ops_init, global_graph, local_graphs)
#   ptensor(global_graph.tensor(global_graph.KEYS.TENSOR.X))

#   # time.sleep(5)
#   # recon_run(m_op_rec, w_ops_rec, global_graph, local_graphs)
#   start_time = time.time()
#   for i in range(5):
#     full_step_run(m_op, w_ops, global_graph, local_graphs, i)
#     end_time = time.time()
#     delta_time = end_time - start_time
#     msg = "the step running time is:{}".format(delta_time / (i + 1))
#     print(msg)
#     if ThisHost.is_master():
#       res = global_graph.tensor(global_graph.KEYS.TENSOR.X).run()
#       np.save('./gpu_all/recon_{}.npy'.format(i), res)
#   ptensor(global_graph.tensor(global_graph.KEYS.TENSOR.X))
#   # full_step_run(m_op, w_ops, global_graph, local_graphs, 1)
#   # full_step_run(m_op, w_ops, global_graph, local_graphs, 2)
#   # if ThisHost.is_master():
#   #     ThisSession.run(gop)
#   # else:
#   #     ThisSession.run(l_ops[ThisHost.host().task_index])
#   if ThisHost.is_master():
#     # res = global_graph.tensor(global_graph.KEYS.TENSOR.X).run()
#     # np.save('recon.npy', res)
#     pass
#   # print('|DEBUG| JOIN!')
#   # Server.join()
#   print('DONE!')
#   end_time = time.time()
#   delta_time = end_time - start_time
#   msg = "the total running time is:{}".format(delta_time)
#   print(msg)
#   if ThisHost.is_master():
#     with open('time_cost.txt', 'w') as fout:
#       print(msg, file=fout)
# import imageio

# img = np.load('recon.npy')
# img = img.reshape([150, 150, 150])
# imgslice = img[75,:,:]
# imageio.imwrite('recon.png', imgslice)

# y_ts, sm_ts = split(y_t, sm_t)
# make_distribute_session
# x_init.run()
# res = ThisSession.run(x_t.data)
# x_ns = []
# for i in range(NB_WORKERS):

#     x_ns.append(recon(x_t, y_ts[i], sm_ts[i],  e_t, i))

# x_n = sm(x_ns)
# x_update = x_t.assign(x_n)
# for i in range(100):
#     x_update.run()
# res = x_t.run()
# print(res)
# np.save('recon.npy', res)