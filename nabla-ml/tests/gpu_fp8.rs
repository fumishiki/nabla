#[cfg(feature = "cuda")]
mod cuda_fp8 {
    use nabla::backend::Cuda;
    use nabla::prelude::*;
    use nabla::scalar::{Fp8E4M3, Fp8E5M2};

    fn make_base(rows: usize, cols: usize) -> Tensor<f32, Cuda> {
        let data: Vec<f32> = (0..rows * cols).map(|i| (i + 1) as f32 * 0.1).collect();
        Tensor::from_vec(data, rows, cols)
    }

    fn make<T: Scalar>(rows: usize, cols: usize) -> Tensor<T, Cuda> {
        make_base(rows, cols).cast::<T>()
    }

    fn make_from<T: Scalar>(vals: &[f32], rows: usize, cols: usize) -> Tensor<T, Cuda> {
        Tensor::from_vec(vals.to_vec(), rows, cols).cast::<T>()
    }

    fn assert_shape<T: Scalar>(t: &Tensor<T, Cuda>, shape: (usize, usize)) {
        assert_eq!(t.shape(), shape);
    }

    fn run_ops<T: Scalar>() {
        let a = make::<T>(2, 3);
        let b = make::<T>(2, 3);

        let _ = Tensor::<T, Cuda>::zeros(2, 3);
        let _ = Tensor::<T, Cuda>::ones(2, 3);
        let _ = Tensor::<T, Cuda>::fill(2, 3, T::from_f64(2.0));
        let eye = Tensor::<T, Cuda>::identity(3);
        let _ = Tensor::<T, Cuda>::arange(T::zero(), T::one(), 4);
        let _ = Tensor::<T, Cuda>::linspace(T::zero(), T::one(), 4);
        let _ = Tensor::<T, Cuda>::rand(2, 3, 42);
        let _ = Tensor::<T, Cuda>::randn(2, 3, 42);
        let _ = Tensor::<T, Cuda>::empty(2, 3);
        let _ = a.clone();
        let _ = a.contiguous();
        assert_shape(&eye, (3, 3));

        let y1 = make::<T>(1, 5).conv1d(&make::<T>(1, 3), None, 1, 1, 5, 1, 3, 1, 0, 1, 1);
        let y2 = make::<T>(1, 9).conv2d(
            &make::<T>(1, 4),
            None,
            1,
            1,
            3,
            3,
            1,
            2,
            2,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        );
        let y3 = make::<T>(1, 8).conv3d(
            &Tensor::<T, Cuda>::fill(1, 8, T::one()),
            None,
            1,
            1,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            (1, 1, 1),
            (0, 0, 0),
            (1, 1, 1),
            1,
        );
        let yt = make::<T>(1, 4).conv_transpose2d(
            &make::<T>(1, 4),
            None,
            1,
            1,
            2,
            2,
            1,
            2,
            2,
            (1, 1),
            (0, 0),
            (0, 0),
        );
        assert_shape(&y1, (1, 3));
        assert_shape(&y2, (1, 4));
        assert_shape(&y3, (1, 1));
        assert_shape(&yt, (1, 9));

        let pmax = make::<T>(1, 16).max_pool2d(4, 4, 2, 2, (2, 2), (0, 0));
        let pavg = make::<T>(1, 4).avg_pool2d(2, 2, 2, 2, (2, 2), (0, 0));
        let padp = make::<T>(1, 16).adaptive_avg_pool2d(4, 4, 2, 2);
        let p1d = make::<T>(1, 5).max_pool1d(5, 2, 1, 0);
        assert_shape(&pmax, (1, 4));
        assert_shape(&pavg, (1, 1));
        assert_shape(&padp, (1, 4));
        assert_shape(&p1d, (1, 4));

        let ln = a.layer_norm(1, T::from_f64(1e-5));
        let rn = a.rms_norm(1, &Tensor::fill(1, 3, T::one()), T::from_f64(1e-5));
        let bn = a.batch_norm(
            &Tensor::fill(1, 3, T::zero()),
            &Tensor::fill(1, 3, T::one()),
            &Tensor::fill(1, 3, T::one()),
            &Tensor::fill(1, 3, T::zero()),
            T::from_f64(1e-5),
        );
        let gn = a.group_norm(
            1,
            &Tensor::fill(1, 3, T::one()),
            &Tensor::fill(1, 3, T::zero()),
            T::from_f64(1e-5),
        );
        assert_shape(&ln, (2, 3));
        assert_shape(&rn, (2, 3));
        assert_shape(&bn, (2, 3));
        assert_shape(&gn, (2, 3));

        let _ = a.relu();
        let _ = a.gelu();
        let _ = a.sigmoid();
        let _ = a.softmax(1);
        let _ = a.silu();
        let _ = a.mish();
        let _ = a.leaky_relu(T::from_f64(0.01));
        let _ = a.elu(T::from_f64(1.0));
        let _ = a.hardswish();
        let _ = a.log_softmax(1);

        let ce_targets = Tensor::fill(2, 3, T::from_f64(1.0 / 3.0));
        let _ = a.log_softmax(1).cross_entropy_loss(&ce_targets);
        let _ = a.mse_loss(&b);
        let _ = a.l1_loss(&b);
        let _ = a.smooth_l1_loss(&b, T::from_f64(1.0));
        let _ = a.bce_with_logits(&b);
        let nll_idx = make_from::<T>(&[0.0, 1.0], 2, 1);
        let _ = a.log_softmax(1).nll_loss(&nll_idx);
        let q = a.softmax(1);
        let _ = a.log_softmax(1).kl_div(&q);
        let _ =
            Tensor::cosine_embedding_loss(&make::<T>(1, 2), &make::<T>(1, 2), T::one(), T::zero());

        let emb_w = make::<T>(3, 2);
        let emb_i = make_from::<T>(&[2.0, 0.0, 1.0], 1, 3);
        let emb = Tensor::embedding(&emb_i, &emb_w);
        assert_shape(&emb, (3, 2));

        let qv = make::<T>(2, 2);
        let kv = make::<T>(2, 2);
        let vv = make::<T>(2, 2);
        let sdpa = Tensor::scaled_dot_product_attention(&qv, &kv, &vv, None);
        assert_shape(&sdpa, (2, 2));

        let qh = Tensor::fill(4, 8, T::one());
        let kh = Tensor::fill(4, 8, T::one());
        let vh = Tensor::fill(4, 8, T::one());
        let mha = Tensor::multi_head_attention(&qh, &kh, &vh, 2, None);
        assert_shape(&mha, (4, 8));

        let reshaped = a.reshape(3, 2);
        let permuted = a.permute(&[1, 0]);
        let cat = Tensor::cat(&[&a, &b], 0);
        let stacked = Tensor::stack(&[&a, &b], 0);
        let squeezed = a.reshape(1, 6).squeeze(0);
        let flat = a.flatten();
        let chunks = a.chunk(2, 1);
        let pad = a.pad([1, 1, 1, 1], T::zero());
        let idx = make_from::<T>(&[0.0, 1.0, 1.0, 0.0], 2, 2);
        let gather = a.gather(1, &idx);
        let scatter_src = make::<T>(2, 2);
        let scatter = a.scatter(1, &idx, &scatter_src);
        let sel_idx = make_from::<T>(&[2.0, 0.0], 1, 2);
        let select = a.index_select(1, &sel_idx);
        let mask = make_from::<T>(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 2, 3);
        let masked = a.masked_fill(&mask, T::zero());
        let wh = a.where_cond(&mask, &b);
        let triu = a.triu(0);
        let tril = a.tril(0);
        let roll = a.roll(1, 1);
        let flip = a.flip(1);
        let (gx, gy) = Tensor::meshgrid(
            &Tensor::arange(T::zero(), T::one(), 3),
            &Tensor::arange(T::zero(), T::one(), 2),
        );
        let (topk_vals, topk_idx) = a.topk(2, 1);
        let (sort_vals, sort_idx) = a.sort(1, false);
        assert_shape(&reshaped, (3, 2));
        assert_shape(&permuted, (3, 2));
        assert_shape(&cat, (4, 3));
        assert_eq!(stacked.shape_vec(), &[2, 2, 3]);
        assert_shape(&squeezed, (1, 6));
        assert_shape(&flat, (1, 6));
        assert_eq!(chunks.len(), 2);
        assert_shape(&chunks[0], (2, 2));
        assert_shape(&pad, (4, 5));
        assert_shape(&gather, (2, 2));
        assert_shape(&scatter, (2, 3));
        assert_shape(&select, (2, 2));
        assert_shape(&masked, (2, 3));
        assert_shape(&wh, (2, 3));
        assert_shape(&triu, (2, 3));
        assert_shape(&tril, (2, 3));
        assert_shape(&roll, (2, 3));
        assert_shape(&flip, (2, 3));
        assert_shape(&gx, (2, 3));
        assert_shape(&gy, (2, 3));
        assert_shape(&topk_vals, (2, 2));
        assert_shape(&topk_idx, (2, 2));
        assert_shape(&sort_vals, (2, 3));
        assert_shape(&sort_idx, (2, 3));

        let bmm_a = make::<T>(4, 2);
        let bmm_b = make::<T>(4, 2);
        let bmm = bmm_a.bmm(&bmm_b, 2, 2, 2, 2);
        assert_shape(&bmm, (4, 2));

        let addmm_c = make::<T>(2, 2);
        let addmm_a = make::<T>(2, 2);
        let addmm_b = make::<T>(2, 2);
        let addmm = addmm_c.addmm(&addmm_a, &addmm_b, T::one(), T::one());
        assert_shape(&addmm, (2, 2));

        let badd_c = make::<T>(4, 2);
        let badd_a = make::<T>(4, 2);
        let badd_b = make::<T>(4, 2);
        let badd = badd_c.baddbmm(&badd_a, &badd_b, 2, 2, 2, 2, T::one(), T::one());
        assert_shape(&badd, (4, 2));

        let _ = a.sum();
        let _ = a.max();
        let _ = a.min();
        let _ = a.mean();
        let _ = a.var();
        let _ = a.std();
        let _ = a.argmax();
        let _ = a.argmin();
        let _ = a.cumsum(1);
        let _ = a.cumprod(1);
        let _ = a.prod();
        let _ = a.norm();
        let _ = a.count_nonzero();

        #[cfg(feature = "cpu")]
        {
            let drop = a.dropout(0.25, false, 123);
            let up_near = make::<T>(1, 4).interpolate_nearest(2, 2, 4, 4);
            let up_bi = make::<T>(1, 4).interpolate_bilinear(2, 2, 4, 4);
            assert_shape(&drop, (2, 3));
            assert_shape(&up_near, (1, 16));
            assert_shape(&up_bi, (1, 16));
        }
    }

    #[test]
    fn gpu_fp8_e4m3_ops() {
        run_ops::<Fp8E4M3>();
    }

    #[test]
    fn gpu_fp8_e5m2_ops() {
        run_ops::<Fp8E5M2>();
    }
}
