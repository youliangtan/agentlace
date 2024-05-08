import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.rlds.rlds_base import DatasetConfig, build_info
from tensorflow_datasets.core import SequentialWriter, Version, dataset_info


class RLDSWriter:
    def __init__(
        self,
        dataset_name: str,
        data_spec: tf.TypeSpec,
        data_directory: str,
        version: str,
        *,
        max_episodes_per_file: int = 1000,
    ):
        data_features = tf.nest.map_structure(
            lambda x: tfds.features.Tensor(shape=x.shape, dtype=x.dtype), data_spec
        )
        ds_config = DatasetConfig(
            name=dataset_name,
            observation_info=tfds.features.FeaturesDict(data_features["observation"]),
            action_info=data_features["action"],
        )
        ds_identity = dataset_info.DatasetIdentity(
            name=ds_config.name,
            version=Version(version),
            data_dir=data_directory,
            module_name="",
        )
        self._ds_info = build_info(ds_config, ds_identity)

        self._sequential_writer = SequentialWriter(
            self._ds_info, max_episodes_per_file, overwrite=False
        )
        self._sequential_writer.initialize_splits(["train"], fail_if_exists=False)
        self._episode = []

    def __call__(self, data):
        self._episode.append(data)
        if (
            data.get("is_last", False)
            or data.get("is_terminal", False)
            or data.get("end_of_trajectory", False)
        ):
            self._write_episode()

    def _write_episode(self):
        episode = tf.nest.map_structure(lambda x: x._numpy(), self._episode)
        self._sequential_writer.add_examples({"train": [{"steps": episode}]})
        self._episode = []

    def close(self):
        if len(self._episode) > 0:
            self._episode[-1]["is_last"] = tf.constant(True, dtype=tf.bool)
            self._write_episode()

        self._sequential_writer.close_all()
