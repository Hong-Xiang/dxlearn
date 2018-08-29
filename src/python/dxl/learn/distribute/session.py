from ..core import SessionBase
__all__ = ['DistributeSession', 'MonitoredSession', 'make_distribute_session']


class DistributeSession(SessionBase):
    def __init__(self, name='depsession', target=None, **kw):
        super().__init__(name=name, **kw)
        self.target = target

    def _create_session(self):
        return tf.Session(self.target, config=self.get_session_config())


class MonitoredSession(DistributeSession):
    class KEYS:
        class CONFIG(SessionBase.KEYS.CONFIG):
            CHECKPOINT_DIR = 'checkpoint_dir'

    def __init__(self, name='depsession', target=None, checkpoint_dir='./save/'):
        super().__init__(name=name, target=target)
        self.checkpoint_dir = checkpoint_dir

    def _create_session(self):
        from .distribute import ThisHost, Master
        master = Master.master_host().job_name
        if ThisHost.is_master():
            creator = tf.train.ChiefSessionCreator(
                master=self.target,
                config=self.get_session_config(),
                checkpoint_dir=self.checkpoint_dir)
        else:
            creator = tf.train.WorkerSessionCreator(
                master=self.target, config=self.get_session_config())
        return tf.train.MonitoredSession(session_creator=creator, )

    def _post_session_created(self):
        pass


def make_distribute_session(session_name='depsession', target=None):
    if target is None:
        target = Server.server().target
    ThisSession.set_session(SessionMonitored(session_name, target))
    return ThisSession.session()

from .api import distribute

@distribute.register(SessionBase)
def _(session, cluster, host):
    pass

# class TensorFlowDistributedSession(TensorFlowSession):
#     def init(self):
#         from dxl.learn.distribute import ThisHost
#         if ThisHost.is_master:
#             self.depsession.run(tf.global_variables_initializer())
#         self.depsession.run(tf.local_variables_initializer())