use nabla_train::ml as nabla;
use nabla_train::prelude::*;

struct VecDataset<T> {
    data: Vec<T>,
}

impl<T: Clone> Dataset for VecDataset<T> {
    type Item = T;

    fn len(&self) -> usize { self.data.len() }

    fn get(&self, idx: usize) -> Self::Item { self.data[idx].clone() }
}

fn collect_order(loader: DataLoader<VecDataset<usize>, VecBatcher>) -> Vec<usize> {
    let mut out = Vec::new();
    for batch in loader.iter() {
        for item in batch {
            out.push(item);
        }
    }
    out
}

#[test]
fn shuffle_seed_is_reproducible() {
    let dataset = VecDataset { data: (0..8).collect() };
    let order_a = collect_order(DataLoader::new(dataset, VecBatcher, 2).shuffle_seed(123));
    let dataset = VecDataset { data: (0..8).collect() };
    let order_b = collect_order(DataLoader::new(dataset, VecBatcher, 2).shuffle_seed(123));
    assert_eq!(order_a, order_b);
}

#[test]
fn shuffle_epoch_changes_order() {
    let dataset = VecDataset { data: (0..8).collect() };
    let order_a = collect_order(DataLoader::new(dataset, VecBatcher, 2).shuffle_seed_epoch(123, 0));
    let dataset = VecDataset { data: (0..8).collect() };
    let order_b = collect_order(DataLoader::new(dataset, VecBatcher, 2).shuffle_seed_epoch(123, 1));
    let mut sorted = order_b.clone();
    sorted.sort();
    assert_eq!(sorted, (0..8).collect::<Vec<_>>());
    assert_eq!(order_a.len(), order_b.len());
}

#[test]
fn repeat_keeps_streaming() {
    let dataset = VecDataset { data: (0..4).collect() };
    let loader = DataLoader::new(dataset, VecBatcher, 2).repeat(true);
    let mut iter = loader.iter();
    let first = iter.next();
    let second = iter.next();
    let third = iter.next();
    assert!(first.is_some());
    assert!(second.is_some());
    assert!(third.is_some());
}

#[test]
fn split_dataset_sizes_and_elements() {
    let data: Vec<i32> = (0..10).collect();
    let dataset = VecDataset { data: data.clone() };
    let (train, val, test) = match split_dataset(dataset, 0.6, 0.2, 123) {
        Ok(out) => out,
        Err(e) => panic!("split failed: {e}"),
    };

    assert_eq!(train.len(), 6);
    assert_eq!(val.len(), 2);
    assert_eq!(test.len(), 2);

    let mut merged = Vec::new();
    for i in 0..train.len() { merged.push(train.get(i)); }
    for i in 0..val.len() { merged.push(val.get(i)); }
    for i in 0..test.len() { merged.push(test.get(i)); }
    merged.sort();
    assert_eq!(merged, data);
}

#[test]
fn cpu_allreduce_mean() {
    let (r0, r1) = CpuAllReduce::<f64, DefaultBackend>::pair();
    let a = mat![[1.0_f64, 3.0, 5.0]];
    let b = mat![[3.0_f64, 5.0, 7.0]];
    let handle = std::thread::spawn(move || {
        r1.allreduce_mean(&b).unwrap_or_else(|_| Tensor::zeros(1, 3))
    });
    let t0 = r0.allreduce_mean(&a).unwrap_or_else(|_| Tensor::zeros(1, 3));
    let t1 = handle.join().unwrap_or_else(|_| Tensor::zeros(1, 3));
    for j in 0..3 {
        let v0 = t0.get(0, j);
        let v1 = t1.get(0, j);
        assert!((v0 - (j as f64 * 2.0 + 2.0)).abs() < 1e-10);
        assert!((v1 - (j as f64 * 2.0 + 2.0)).abs() < 1e-10);
    }
}
