use crate::{Device, Tensor};

#[test]
fn test_device_default_cpu() {
    let t = Tensor::zeros(vec![2, 3]).unwrap();
    assert_eq!(t.device(), Device::Cpu);
}

#[test]
fn test_device_scalar_default_cpu() {
    let t = Tensor::scalar(1.0);
    assert_eq!(t.device(), Device::Cpu);
}

#[test]
fn test_device_from_vec_default_cpu() {
    let t = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    assert_eq!(t.device(), Device::Cpu);
}

#[test]
fn test_device_to_device_gpu() {
    let t = Tensor::ones(vec![2, 2]).unwrap();
    let t_gpu = t.to_device(Device::Gpu(0));
    assert_eq!(t_gpu.device(), Device::Gpu(0));
    assert_eq!(t_gpu.data(), t.data());
    assert_eq!(t_gpu.shape(), t.shape());
}

#[test]
fn test_device_to_device_roundtrip() {
    let t = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let t_gpu = t.to_device(Device::Gpu(0));
    let t_cpu = t_gpu.to_device(Device::Cpu);
    assert_eq!(t_cpu.device(), Device::Cpu);
    assert_eq!(t_cpu.data(), t.data());
}

#[test]
fn test_device_enum_equality() {
    assert_eq!(Device::Cpu, Device::Cpu);
    assert_eq!(Device::Gpu(0), Device::Gpu(0));
    assert_ne!(Device::Cpu, Device::Gpu(0));
    assert_ne!(Device::Gpu(0), Device::Gpu(1));
}

#[test]
fn test_device_preserved_by_reshape() {
    let t = Tensor::ones(vec![2, 3]).unwrap().to_device(Device::Gpu(0));
    let t2 = t.reshape(vec![3, 2]).unwrap();
    assert_eq!(t2.device(), Device::Gpu(0));
}

#[test]
fn test_device_preserved_by_to_dtype() {
    let t = Tensor::ones(vec![2]).unwrap().to_device(Device::Gpu(1));
    let t2 = t.to_dtype(crate::DType::F16);
    assert_eq!(t2.device(), Device::Gpu(1));
}

#[test]
fn test_device_default_is_cpu() {
    assert_eq!(Device::default(), Device::Cpu);
}
