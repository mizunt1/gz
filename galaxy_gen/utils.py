import inspect
from torch.utils.tensorboard import SummaryWriter
import os
import shutil


def setup_run_directory(path):
    if os.path.exists(path):
        # if this path already exists, then probably we deleted a sacred run and
        # don't care about this tensorboard anymore.
        # in this case we can clean it up and replace it now.
        shutil.rmtree(path)
        return
    os.makedirs(path)
    return path.as_posix()


class DummyWriter:
    """
    A class which mocks a SummaryWriter, but does nothing when any of it's methods are called.
    This is mostly useful for being able to do debugging runs without clogging up my tensorboard log dirs
    """

    def __init__(self, logdir):
        sw = SummaryWriter(log_dir=logdir)

        def do_nothing(*args, **kwargs):
            pass

        for name, _ in inspect.getmembers(sw, predicate=inspect.ismethod):
            setattr(self, name, do_nothing)
        return


class DummyWriter:
    """
    A class which mocks a SummaryWriter, but does nothing when any of it's methods are called.
    This is mostly useful for being able to do debugging runs without clogging up my tensorboard log dirs
    """

    def __init__(self, logdir="/dev/null"):
        sw = SummaryWriter(log_dir=logdir)

        def do_nothing(*args, **kwargs):
            pass

        for name, _ in inspect.getmembers(sw, predicate=inspect.ismethod):
            setattr(self, name, do_nothing)
        return
