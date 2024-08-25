r""""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
data from an iterable-style or map-style dataset. This logic is shared in both
single- and multi-processing data loading.
"""
import redis
import random
import queue

class _BaseDatasetFetcher(object):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index, worker_id):
        raise NotImplementedError()


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_IterableDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)
        self.ended = False

    def fetch(self, possibly_batched_index, worker_id):
        if self.ended:
            raise StopIteration

        if self.auto_collation:
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    self.ended = True
                    break
            if len(data) == 0 or (self.drop_last and len(data) < len(possibly_batched_index)):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_MapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)

    def fetch(self, possibly_batched_index, worker_id=None):
        if self.auto_collation:
            data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)


class _QuiverFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last, sampler):
        super(_QuiverFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        self.sampler = sampler

    def fetch(self, possibly_batched_index, worker_id=None): #implements the substitutable hits and co-operative miss handling
        if self.auto_collation:
            data = []
            used_index_list=[]
            for idx in possibly_batched_index:
                sample = self.dataset.try_get_from_cache(idx)
                if sample != None:
                    used_index_list.append(idx)
                    data.append(sample)
                if len(data)>=256:
                    unused_indexes = []
                    for i in possibly_batched_index:
                        if i not in used_index_list:
                            unused_indexes.append(i)

                    self.sampler.refill_unused(unused_indexes)
                    return self.collate_fn(data)
            unused_indexes = []
            for i in possibly_batched_index:
                if i not in used_index_list:
                    unused_indexes.append(i)
            #assert(len(unused_indexes) == (256*10)-len(data))

            #print("unused indices ",len(unused_indexes))
            if len(data) < 256:
                for idx in unused_indexes:
                    used_index_list.append(idx)
                    sample = self.dataset.get_from_disk(idx)
                    data.append(sample)
                    if len(data) >= 256:
                        unused_indexes = []
                        for i in possibly_batched_index:
                            if i not in used_index_list:
                                unused_indexes.append(i)

                        self.sampler.refill_unused(unused_indexes)
                        return self.collate_fn(data)

                #data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)

class _BRMapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last, br_factor):
        super(_BRMapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        self.br_factor = br_factor
        #print('connecting to redis raw in fetch.py', self.dataset.sample_host,self.dataset.sample_port)
        #print('connecting to redis tensors in fetch.py', self.dataset.tensor_host,self.dataset.tensor_port)
        try:
            #self.redis_cache = redis.Redis(host=self.dataset.sample_host, port=self.dataset.sample_port)
            #self.redis_cache_2 = redis.Redis(host=self.dataset.tensor_host, port=self.dataset.tensor_port)
            self.seen_samples = redis.Redis(port=int(dataset.job_sample_tracker_port))
        except:
            print("[ERROR] PROBLEM in fetch.py")
        #print(self.br_factor)

        #self.missing_batches = multiprocessing.Queue()
        #thread = threading.Thread(target=self._fetch_missing, args=(self.missing_batches, ))
        #thread.daemon = True
        #thread.start()

    def _fetch_missing(self, possibly_batched_index_queue):
        while 1:
            try:
                indexes = possibly_batched_index_queue.get()
            except queue.Empty:
                continue
            if self.auto_collation:
                data = [self.dataset[idx] for idx in indexes]
            else:
                data = self.dataset[indexes]

    def fetch(self, possibly_batched_index, worker_id=None):
        replacement_candidates = set()
        sample_cache_ = redis.Redis(host=self.dataset.sample_host, port=self.dataset.sample_port)
        tensor_cache_ = redis.Redis(host=self.dataset.tensor_host, port=self.dataset.tensor_port)
        decoded_cache_ = redis.Redis(host=self.dataset.decoded_host, port=self.dataset.decoded_port)

        for idx in possibly_batched_index:
            if not sample_cache_.exists(idx) and not tensor_cache_.exists(idx) and not decoded_cache_.exists(idx):
            #if not self.redis_cache.exists(idx):
                replacement_candidates.add(idx)


        swap_len = len(replacement_candidates)
        #if swap_len > self.br_factor*256:
        #    swap_len = int(self.br_factor*256)
        #all_keys = self.redis_cache.keys('*')
        repl_samples=[]
        if (500 + sample_cache_.dbsize() + tensor_cache_.dbsize() + decoded_cache_.dbsize()) > self.seen_samples.dbsize():
            num_retries = 0
            while len(repl_samples) < swap_len and num_retries < 5:
                num_retries+=1
                keys_pp = tensor_cache_.scan(cursor=random.randint(0, 10000), match='*', count=swap_len)
                # keys_pp = [key for key in keys_pp[1] if re.match(r'^\d+$', str(key))]
                #keys_pp = [key for key in keys_pp[1] if isinstance(key, int)]
                #print("HEREEEEE", keys_pp[1])
                for key in keys_pp[1]:
                    if not self.seen_samples.exists(key):
                        repl_samples.append(key)
                if len(repl_samples) < swap_len:
                    keys_pp = decoded_cache_.scan(cursor=random.randint(0, 10000), match='*',
                                                count=swap_len - len(repl_samples))
                    # keys_pp = [key for key in keys_pp[1] if re.match(r'^\d+$', str(key))]
                    #keys_pp = [key for key in keys_pp[1] if isinstance(key, int)]
                    for key in keys_pp[1]:
                        if not self.seen_samples.exists(key):
                            repl_samples.append(key)
                if len(repl_samples) < swap_len:
                    keys_pp = sample_cache_.scan(cursor=random.randint(0, 10000), match='*',
                                                count=swap_len - len(repl_samples))
                    # keys_pp = [key for key in keys_pp[1] if re.match(r'^\d+$', str(key))]
                    #keys_pp = [key for key in keys_pp[1] if isinstance(key, int)]
                    for key in keys_pp[1]:
                        if not self.seen_samples.exists(key):
                            repl_samples.append(key)
                if len(repl_samples)>=swap_len:
                    break
            samples = repl_samples[:swap_len]
            new_possibly_batched_index = []
            for i in range(len(possibly_batched_index)):
                if possibly_batched_index[i] not in replacement_candidates:
                    new_possibly_batched_index.append(int(possibly_batched_index[i]))
            new_possibly_batched_index += samples
            if len(new_possibly_batched_index) < 256:
                new_possibly_batched_index += list(replacement_candidates)[:256 - len(new_possibly_batched_index)]

            if self.auto_collation:
                data = [self.dataset[int(idx)] for idx in new_possibly_batched_index]
            else:
                data = self.dataset[new_possibly_batched_index]
            sample_cache_.close()
            tensor_cache_.close()
            decoded_cache_.close()
            return self.collate_fn(data)
        else:
            new_possibly_batched_index = possibly_batched_index

            if self.auto_collation:
                data = [self.dataset[int(idx)] for idx in new_possibly_batched_index]
            else:
                data = self.dataset[new_possibly_batched_index]
            sample_cache_.close()
            tensor_cache_.close()
            decoded_cache_.close()
            return self.collate_fn(data)

    def fetch_(self, possibly_batched_index, worker_id):
        replacement_candidates = set()
        for idx in possibly_batched_index:
            if not self.redis_cache.exists(idx) and not self.redis_cache_2.exists(idx):
            #if not self.redis_cache.exists(idx):
                replacement_candidates.add(idx)
        swap_len = len(replacement_candidates)
        if swap_len > self.br_factor*256:
            swap_len = int(self.br_factor*256)
        all_keys = self.redis_cache.keys('*')
        #all_keys_2 = self.redis_cache_2.keys('*')
        #all_keys += all_keys_2
        seen_samples = set(self.seen_samples.keys('*'))
        replacable_samples = []
        for key in all_keys:
            if key not in seen_samples:
                replacable_samples.append(key)
            if len(replacable_samples) == swap_len:
                break
        #replacable_samples = [key for key in all_keys if key not in seen_samples]
        samples = replacable_samples#[:swap_len]

        #print(self.dataset.get_seen_samples())
        #print(len(replacable_samples))
        new_possibly_batched_index = []
        for i in range(len(possibly_batched_index)):
            if possibly_batched_index[i] not in replacement_candidates:
                new_possibly_batched_index.append(int(possibly_batched_index[i]))
        new_possibly_batched_index += samples
        #self.missing_batches.put(list(replacement_candidates))
        #print("factor:", len(replacable_samples)/len(possibly_batched_index))
        #thread = threading.Thread(target=self._fetch_missing, args=(possibly_batched_index,))
        #thread.start()

        #replacable_samples = []
        #print("-----------")
        #print(len(possibly_batched_index))
        #print(len(new_possibly_batched_index))
        #print("-----------")
        if len(new_possibly_batched_index) < 256:
            new_possibly_batched_index = possibly_batched_index

        if self.auto_collation:
            data = [self.dataset[int(idx)] for idx in new_possibly_batched_index]
        else:
            data = self.dataset[new_possibly_batched_index]
        return self.collate_fn(data)
