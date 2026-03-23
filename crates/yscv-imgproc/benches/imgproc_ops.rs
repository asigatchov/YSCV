use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use yscv_imgproc::{
    ImageU8, box_blur_3x3_u8, dilate_3x3_u8, erode_3x3_u8, grayscale_u8, resize_bilinear_u8,
    resize_nearest_u8, sobel_3x3_magnitude_u8,
};
use yscv_imgproc::{
    box_blur_3x3, closing_3x3, dilate_3x3, erode_3x3, flip_horizontal, flip_vertical,
    morph_gradient_3x3, normalize, opening_3x3, resize_nearest, rgb_to_grayscale, rotate90_cw,
    sobel_3x3_gradients, sobel_3x3_magnitude,
};
use yscv_tensor::Tensor;

fn rgb_image(width: usize, height: usize, seed: f32) -> Tensor {
    let mut data = Vec::with_capacity(width * height * 3);
    for idx in 0..(width * height * 3) {
        data.push(((idx % 251) as f32 * 0.0041 + seed).fract());
    }
    Tensor::from_vec(vec![height, width, 3], data).expect("valid image")
}

fn bench_imgproc_ops(c: &mut Criterion) {
    let rgb_640_480 = rgb_image(640, 480, 0.17);
    let rgb_320_240 = rgb_image(320, 240, 0.42);

    let mut group = c.benchmark_group("imgproc_ops");
    group.sample_size(30);
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));

    group.bench_function("rgb_to_grayscale_640x480", |b| {
        b.iter(|| {
            let out = rgb_to_grayscale(black_box(&rgb_640_480)).expect("grayscale");
            black_box(out);
        });
    });

    group.bench_function("resize_nearest_320x240_to_640x480", |b| {
        b.iter(|| {
            let out = resize_nearest(black_box(&rgb_320_240), 480, 640).expect("resize");
            black_box(out);
        });
    });

    group.bench_function("normalize_640x480_rgb", |b| {
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
        b.iter_batched(
            || rgb_640_480.clone(),
            |input| {
                let out = normalize(black_box(&input), black_box(&mean), black_box(&std))
                    .expect("normalize");
                black_box(out);
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("box_blur_3x3_640x480_rgb", |b| {
        b.iter(|| {
            let out = box_blur_3x3(black_box(&rgb_640_480)).expect("blur");
            black_box(out);
        });
    });

    group.bench_function("dilate_3x3_640x480_rgb", |b| {
        b.iter(|| {
            let out = dilate_3x3(black_box(&rgb_640_480)).expect("dilate");
            black_box(out);
        });
    });

    group.bench_function("erode_3x3_640x480_rgb", |b| {
        b.iter(|| {
            let out = erode_3x3(black_box(&rgb_640_480)).expect("erode");
            black_box(out);
        });
    });

    group.bench_function("opening_3x3_640x480_rgb", |b| {
        b.iter(|| {
            let out = opening_3x3(black_box(&rgb_640_480)).expect("opening");
            black_box(out);
        });
    });

    group.bench_function("closing_3x3_640x480_rgb", |b| {
        b.iter(|| {
            let out = closing_3x3(black_box(&rgb_640_480)).expect("closing");
            black_box(out);
        });
    });

    group.bench_function("morph_gradient_3x3_640x480_rgb", |b| {
        b.iter(|| {
            let out = morph_gradient_3x3(black_box(&rgb_640_480)).expect("morph gradient");
            black_box(out);
        });
    });

    group.bench_function("sobel_3x3_gradients_640x480_rgb", |b| {
        b.iter(|| {
            let out = sobel_3x3_gradients(black_box(&rgb_640_480)).expect("sobel gradients");
            black_box(out);
        });
    });

    group.bench_function("sobel_3x3_magnitude_640x480_rgb", |b| {
        b.iter(|| {
            let out = sobel_3x3_magnitude(black_box(&rgb_640_480)).expect("sobel magnitude");
            black_box(out);
        });
    });

    group.bench_function("flip_horizontal_640x480_rgb", |b| {
        b.iter(|| {
            let out = flip_horizontal(black_box(&rgb_640_480)).expect("flip horizontal");
            black_box(out);
        });
    });

    group.bench_function("flip_vertical_640x480_rgb", |b| {
        b.iter(|| {
            let out = flip_vertical(black_box(&rgb_640_480)).expect("flip vertical");
            black_box(out);
        });
    });

    group.bench_function("rotate90_cw_640x480_rgb", |b| {
        b.iter(|| {
            let out = rotate90_cw(black_box(&rgb_640_480)).expect("rotate90 cw");
            black_box(out);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// u8 image helpers
// ---------------------------------------------------------------------------

fn gray_u8_image(width: usize, height: usize) -> ImageU8 {
    let data: Vec<u8> = (0..width * height).map(|i| (i % 251) as u8).collect();
    ImageU8::new(data, height, width, 1).expect("valid u8 image")
}

fn rgb_u8_image(width: usize, height: usize) -> ImageU8 {
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 251) as u8).collect();
    ImageU8::new(data, height, width, 3).expect("valid u8 image")
}

// ---------------------------------------------------------------------------
// u8 benchmark group — separate from f32 to avoid OOM
// ---------------------------------------------------------------------------

fn bench_imgproc_u8_ops(c: &mut Criterion) {
    let rgb_640_480 = rgb_u8_image(640, 480);
    let gray_640_480 = gray_u8_image(640, 480);
    let gray_320_240 = gray_u8_image(320, 240);

    let mut group = c.benchmark_group("imgproc_u8_ops");
    group.sample_size(30);
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));

    group.bench_function("grayscale_u8_640x480", |b| {
        b.iter(|| {
            let out = grayscale_u8(black_box(&rgb_640_480)).expect("grayscale u8");
            black_box(out);
        });
    });

    group.bench_function("resize_nearest_u8_320x240_to_640x480", |b| {
        b.iter(|| {
            let out =
                resize_nearest_u8(black_box(&gray_320_240), 480, 640).expect("resize nearest u8");
            black_box(out);
        });
    });

    group.bench_function("resize_bilinear_u8_320x240_to_640x480", |b| {
        b.iter(|| {
            let out =
                resize_bilinear_u8(black_box(&gray_320_240), 480, 640).expect("resize bilinear u8");
            black_box(out);
        });
    });

    group.bench_function("dilate_3x3_u8_640x480_gray", |b| {
        b.iter(|| {
            let out = dilate_3x3_u8(black_box(&gray_640_480)).expect("dilate u8");
            black_box(out);
        });
    });

    group.bench_function("erode_3x3_u8_640x480_gray", |b| {
        b.iter(|| {
            let out = erode_3x3_u8(black_box(&gray_640_480)).expect("erode u8");
            black_box(out);
        });
    });

    group.bench_function("box_blur_3x3_u8_640x480_gray", |b| {
        b.iter(|| {
            let out = box_blur_3x3_u8(black_box(&gray_640_480)).expect("box blur u8");
            black_box(out);
        });
    });

    group.bench_function("sobel_3x3_u8_640x480_gray", |b| {
        b.iter(|| {
            let out = sobel_3x3_magnitude_u8(black_box(&gray_640_480)).expect("sobel u8");
            black_box(out);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_imgproc_ops, bench_imgproc_u8_ops);
criterion_main!(benches);
