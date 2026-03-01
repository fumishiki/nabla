use std::collections::VecDeque;
use std::sync::Arc;

pub trait Dataset {
    type Item;
    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> Self::Item;
}

pub trait Batcher<I> {
    type Output;
    fn batch(&self, items: Vec<I>) -> Self::Output;
}

#[derive(Clone, Copy)]
pub struct VecBatcher;

impl<I> Batcher<I> for VecBatcher {
    type Output = Vec<I>;
    fn batch(&self, items: Vec<I>) -> Vec<I> { items }
}

fn shuffle_indices(mut idx: Vec<usize>, seed: u64) -> Vec<usize> {
    let mut s = if seed == 0 { 0xA5A5_F00D_CAFE_BEEFu64 } else { seed };
    for i in (1..idx.len()).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let j = (s as usize) % (i + 1);
        idx.swap(i, j);
    }
    idx
}

#[derive(Clone, Copy)]
pub enum Sampler {
    Sequential,
    Shuffle { seed: u64 },
    ShuffleEpoch { seed: u64, epoch: u64 },
}

impl Sampler {
    fn indices(&self, len: usize) -> Vec<usize> {
        let idx: Vec<usize> = (0..len).collect();
        match self {
            Sampler::Sequential => idx,
            Sampler::Shuffle { seed } => shuffle_indices(idx, *seed),
            Sampler::ShuffleEpoch { seed, epoch } => shuffle_indices(idx, seed ^ epoch),
        }
    }
}

pub struct DataLoader<D, B> {
    dataset: Arc<D>,
    batcher: B,
    batch_size: usize,
    sampler: Sampler,
    drop_last: bool,
    repeat: bool,
    prefetch: usize,
}

impl<D, B> DataLoader<D, B>
where
    D: Dataset + Send + Sync + 'static,
    B: Batcher<D::Item> + Clone,
{
    #[must_use]
    pub fn new(dataset: D, batcher: B, batch_size: usize) -> Self {
        Self {
            dataset: Arc::new(dataset),
            batcher,
            batch_size,
            sampler: Sampler::Sequential,
            drop_last: false,
            repeat: false,
            prefetch: 0,
        }
    }

    #[must_use]
    pub fn sampler(mut self, sampler: Sampler) -> Self {
        self.sampler = sampler;
        self
    }

    #[must_use]
    pub fn shuffle_seed(mut self, seed: u64) -> Self {
        self.sampler = Sampler::Shuffle { seed };
        self
    }

    #[must_use]
    pub fn shuffle_seed_epoch(mut self, seed: u64, epoch: u64) -> Self {
        self.sampler = Sampler::ShuffleEpoch { seed, epoch };
        self
    }

    #[must_use]
    pub fn set_epoch(mut self, epoch: u64) -> Self {
        self.sampler = match self.sampler {
            Sampler::Shuffle { seed } => Sampler::ShuffleEpoch { seed, epoch },
            Sampler::ShuffleEpoch { seed, .. } => Sampler::ShuffleEpoch { seed, epoch },
            other => other,
        };
        self
    }

    #[must_use]
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    #[must_use]
    pub fn repeat(mut self, repeat: bool) -> Self {
        self.repeat = repeat;
        self
    }

    #[must_use]
    pub fn prefetch(mut self, prefetch: usize) -> Self {
        self.prefetch = prefetch;
        self
    }

    #[must_use]
    pub fn iter(&self) -> DataLoaderIter<D, B> {
        DataLoaderIter::new(
            self.dataset.clone(),
            self.batcher.clone(),
            self.batch_size,
            self.sampler,
            self.drop_last,
            self.repeat,
            self.prefetch,
        )
    }
}

pub struct DataLoaderIter<D, B>
where
    D: Dataset,
    B: Batcher<D::Item>,
{
    dataset: Arc<D>,
    batcher: B,
    batch_size: usize,
    sampler: Sampler,
    drop_last: bool,
    repeat: bool,
    prefetch: usize,
    indices: Vec<usize>,
    pos: usize,
    buffer: VecDeque<B::Output>,
    epoch: u64,
}

impl<D, B> DataLoaderIter<D, B>
where
    D: Dataset + Send + Sync + 'static,
    B: Batcher<D::Item> + Clone,
{
    fn new(
        dataset: Arc<D>,
        batcher: B,
        batch_size: usize,
        sampler: Sampler,
        drop_last: bool,
        repeat: bool,
        prefetch: usize,
    ) -> Self {
        let indices = sampler.indices(dataset.len());
        let epoch = match sampler {
            Sampler::ShuffleEpoch { epoch, .. } => epoch,
            _ => 0,
        };
        Self {
            dataset,
            batcher,
            batch_size: batch_size.max(1),
            sampler,
            drop_last,
            repeat,
            prefetch,
            indices,
            pos: 0,
            buffer: VecDeque::new(),
            epoch,
        }
    }

    fn refill_buffer(&mut self) {
        while self.buffer.len() < self.prefetch {
            if let Some(batch) = self.next_batch() {
                self.buffer.push_back(batch);
            } else {
                break;
            }
        }
    }

    fn next_batch(&mut self) -> Option<B::Output> {
        if self.pos >= self.indices.len() {
            if self.repeat {
                self.epoch += 1;
                self.indices = match self.sampler {
                    Sampler::ShuffleEpoch { seed, .. } => {
                        Sampler::ShuffleEpoch { seed, epoch: self.epoch }.indices(self.dataset.len())
                    }
                    _ => self.sampler.indices(self.dataset.len()),
                };
                self.pos = 0;
            } else {
                return None;
            }
        }
        let end = (self.pos + self.batch_size).min(self.indices.len());
        let count = end - self.pos;
        if count == 0 {
            return None;
        }
        if self.drop_last && count < self.batch_size {
            self.pos = self.indices.len();
            return None;
        }
        let mut items = Vec::with_capacity(count);
        for i in self.pos..end {
            let idx = self.indices[i];
            items.push(self.dataset.get(idx));
        }
        self.pos = end;
        Some(self.batcher.batch(items))
    }
}

impl<D, B> Iterator for DataLoaderIter<D, B>
where
    D: Dataset + Send + Sync + 'static,
    B: Batcher<D::Item> + Clone,
{
    type Item = B::Output;

    fn next(&mut self) -> Option<Self::Item> {
        if self.prefetch > 0 {
            if let Some(batch) = self.buffer.pop_front() {
                self.refill_buffer();
                return Some(batch);
            }
            self.refill_buffer();
            return self.buffer.pop_front();
        }
        self.next_batch()
    }
}

pub struct Subset<D> {
    dataset: Arc<D>,
    indices: Vec<usize>,
}

impl<D> Subset<D> {
    #[must_use]
    pub fn new(dataset: Arc<D>, indices: Vec<usize>) -> Self { Self { dataset, indices } }

    #[must_use]
    pub fn indices(&self) -> &[usize] { &self.indices }
}

impl<D: Dataset> Dataset for Subset<D> {
    type Item = D::Item;

    fn len(&self) -> usize { self.indices.len() }

    fn get(&self, idx: usize) -> Self::Item { self.dataset.get(self.indices[idx]) }
}

pub fn split_dataset<D: Dataset>(
    dataset: D,
    train_ratio: f64,
    val_ratio: f64,
    seed: u64,
) -> Result<(Subset<D>, Subset<D>, Subset<D>), String> {
    if train_ratio < 0.0 || val_ratio < 0.0 {
        return Err("nabla-train: ratios must be non-negative".to_owned());
    }
    if train_ratio + val_ratio > 1.0 + 1e-12 {
        return Err("nabla-train: train+val ratios exceed 1.0".to_owned());
    }
    let total = dataset.len();
    let train_len = ((total as f64) * train_ratio).floor() as usize;
    let val_len = ((total as f64) * val_ratio).floor() as usize;
    if train_len + val_len > total {
        return Err("nabla-train: split sizes exceed dataset length".to_owned());
    }
    let test_len = total - train_len - val_len;

    let mut indices: Vec<usize> = (0..total).collect();
    let mut s = if seed == 0 { 0xA5A5_F00D_CAFE_BEEFu64 } else { seed };
    for i in (1..indices.len()).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let j = (s as usize) % (i + 1);
        indices.swap(i, j);
    }

    let train_idx = indices[..train_len].to_vec();
    let val_idx = indices[train_len..train_len + val_len].to_vec();
    let test_idx = indices[train_len + val_len..train_len + val_len + test_len].to_vec();

    let shared = Arc::new(dataset);
    Ok((
        Subset::new(shared.clone(), train_idx),
        Subset::new(shared.clone(), val_idx),
        Subset::new(shared, test_idx),
    ))
}
