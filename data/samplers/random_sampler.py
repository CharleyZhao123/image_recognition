import copy
import random
import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - batch_size (int): number of examples in a batch.
    - num_instances (int): number of instances per identity in a batch.
    """
    def __init__(self, data_source, batch_size, num_instances):
        super(RandomIdentitySampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_baggage_ids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, _, _, baggage_id, _) in enumerate(self.data_source):
            self.index_dic[baggage_id].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for bid in self.pids:
            idxs = copy.deepcopy(self.index_dic[bid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs,
                                        size=self.num_instances,
                                        replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[bid].append(batch_idxs)
                    batch_idxs = []

        avai_baggage_ids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_baggage_ids) >= self.num_baggage_ids_per_batch:
            selected_baggage_ids = random.sample(
                avai_baggage_ids, self.num_baggage_ids_per_batch)
            for bid in selected_baggage_ids:
                batch_idxs = batch_idxs_dict[bid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[bid]) == 0:
                    avai_baggage_ids.remove(bid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


if __name__ == '__main__':
    from data.datasets import MultiViewBaggage

    mvb_dataset = MultiViewBaggage()
    triplet_sampler = RandomIdentitySampler(mvb_dataset.train, 64, 4)
