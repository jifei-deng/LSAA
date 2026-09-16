import os

import six
from gym import error, logger
from gym.utils import closer
from gym.wrappers import Monitor as MO

from ma_gym.wrappers.monitoring import stats_recorder

FILE_PREFIX = 'openaigym'
MANIFEST_PREFIX = FILE_PREFIX + '.manifest'


class Monitor(MO):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_agents = self.env.n_agents

    def _start(self, directory, video_callable=None, force=False, resume=False,
               write_upon_reset=False, uid=None, mode=None):

        if self.env.spec is None:
            logger.warn(
                "Trying to monitor an environment which has no 'spec' set. "
                "This usually means you did not create it via 'gym.make', and is recommended only for advanced users.")
            env_id = '(unknown)'
        else:
            env_id = self.env.spec.id

        if not os.path.exists(directory):
            logger.info('Creating monitor directory %s', directory)
            if six.PY3:
                os.makedirs(directory, exist_ok=True)
            else:
                os.makedirs(directory)

        if video_callable is None:
            video_callable = capped_cubic_video_schedule
        elif video_callable == False:
            video_callable = disable_videos
        elif not callable(video_callable):
            raise error.Error('You must provide a function, None, or False for video_callable, not {}: {}'.format(
                type(video_callable), video_callable))
        self.video_callable = video_callable


        if force:
            clear_monitor_files(directory)
        elif not resume:
            training_manifests = detect_training_manifests(directory)
            if len(training_manifests) > 0:
                raise error.Error('''Trying to write to monitor directory {} with existing monitor files: {}.
                                    You should use a unique directory for each training run, or use 'force=True'
                                     to automatically clear previous monitor files.'''
                                  .format(directory, ', '.join(training_manifests[:5])))

        self._monitor_id = monitor_closer.register(self)

        self.enabled = True
        self.directory = os.path.abspath(directory)


        self.file_prefix = FILE_PREFIX
        self.file_infix = '{}.{}'.format(self._monitor_id, uid if uid else os.getpid())

        self.stats_recorder = stats_recorder.StatsRecorder(directory, '{}.episode_batch.{}'.format(self.file_prefix,
                                                                                                   self.file_infix),
                                                           autoreset=self.env_semantics_autoreset, env_id=env_id)

        if not os.path.exists(directory): os.mkdir(directory)
        self.write_upon_reset = write_upon_reset

        if mode is not None:
            self._set_mode(mode)


def detect_training_manifests(training_dir, files=None):
    if files is None:
        files = os.listdir(training_dir)
    return [os.path.join(training_dir, f) for f in files if f.startswith(MANIFEST_PREFIX + '.')]


def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith(FILE_PREFIX + '.')]


def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return

    logger.info('Clearing %d monitor files from previous run (because force=True was provided)', len(files))
    for file in files:
        os.unlink(file)


def capped_cubic_video_schedule(episode_id):
    if episode_id < 1000:
        return int(round(episode_id ** (1. / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0


def disable_videos(episode_id):
    return False


monitor_closer = closer.Closer()
