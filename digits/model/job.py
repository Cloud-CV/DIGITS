# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

#from digits.job import Job, PretrainedJob
from . import tasks

from digits.utils import override

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

class ModelJob(Job):
    """
    A Job that creates a neural network model
    """

    def __init__(self, dataset_id, **kwargs):
        """
        Arguments:
        dataset_id -- the job_id of the DatasetJob that this ModelJob depends on
        """
        super(ModelJob, self).__init__(**kwargs)
        self.pickver_job_dataset = PICKLE_VERSION

        self.dataset_id = dataset_id
        self.load_dataset()

    def __getstate__(self):
        state = super(ModelJob, self).__getstate__()
        if 'dataset' in state:
            del state['dataset']
        return state

    def __setstate__(self, state):
        super(ModelJob, self).__setstate__(state)
        self.dataset = None

    @override
    def json_dict(self, verbose=False):
        d = super(ModelJob, self).json_dict(verbose)

        if verbose:
            d.update({
                'snapshots': [s[1] for s in self.train_task().snapshots],
                })
        return d

    def load_dataset(self):
        from digits.webapp import scheduler
        job = scheduler.get_job(self.dataset_id)
        assert job is not None, 'Cannot find dataset'
        self.dataset = job
        for task in self.tasks:
            task.dataset = job

    def train_task(self):
        """Return the first TrainTask for this job"""
        return [t for t in self.tasks if isinstance(t, tasks.TrainTask)][0]

    def download_files(self):
        """
        Returns a list of tuples: [(path, filename)...]
        These files get added to an archive when this job is downloaded
        """
        return NotImplementedError()

class PretrainedModelJob(PretrainedJob):
    """
    A Job that loads a neural network pretrained model.
    """

    def __init__(self, **kwargs):
        super(PretrainedModelJob, self).__init__(**kwargs)

    def __getstate__(self):
        state = super(PretrainedModelJob, self).__getstate__()
        return state

    def __setstate__(self, state):
        super(PretrainedModelJob, self).__setstate__(state)

    def download_files(self):
        """
        Returns a list of tuples: [(path, filename)...]
        These files get added to an archive when this job is downloaded
        """
        return NotImplementedError()

    @override
    def json_dict(self, verbose=False):
        d = super(PretrainedModelJob, self).json_dict(verbose)

        if verbose:
            d.update({
                'snapshots': [s[1] for s in self.load_model_task().snapshots],
                })
        return d

    def load_model_task(self):
        """
        Return the first LoadModelTask for this job.
        """
        return [t for t in self.tasks if isinstance(t, tasks.LoadModelTask)][0]
