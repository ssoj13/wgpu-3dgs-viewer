use glam::*;

use crate::core::{self, BufferWrapper, FixedSizeBufferWrapper};

/// The crop-bounds buffer.
///
/// Holds an axis-aligned bounding box (in *original Gaussian space* — the same space as the
/// positions in [`crate::core::GaussiansBuffer`], i.e. BEFORE the model transform is applied) used
/// by the preprocess compute pass to cull Gaussians whose center lies outside the box.
///
/// Why this exists: it turns interactive cropping (e.g. dragging a crop gizmo) into a cheap
/// per-frame uniform write instead of a full CPU re-filter + GPU buffer rebuild. When `enabled` is
/// `0` the pass keeps every Gaussian, so a freshly created buffer is a no-op until updated.
#[derive(Debug, Clone)]
pub struct CropBoundsBuffer(wgpu::Buffer);

impl CropBoundsBuffer {
    /// Create a new crop-bounds buffer (cropping disabled until [`Self::update`]).
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Crop Bounds Buffer"),
            size: std::mem::size_of::<CropBoundsPod>() as u64,
            usage: Self::DEFAULT_USAGES,
            mapped_at_creation: false,
        });

        Self(buffer)
    }

    /// Update the crop bounds. `enabled = false` disables cropping (all Gaussians kept); `min`/`max`
    /// are in original Gaussian space (pre model transform).
    pub fn update(&self, queue: &wgpu::Queue, min: Vec3, max: Vec3, enabled: bool) {
        self.update_with_pod(queue, &CropBoundsPod::new(min, max, enabled));
    }

    /// Update the crop-bounds buffer with [`CropBoundsPod`].
    pub fn update_with_pod(&self, queue: &wgpu::Queue, pod: &CropBoundsPod) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(pod));
    }
}

impl BufferWrapper for CropBoundsBuffer {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

impl From<CropBoundsBuffer> for wgpu::Buffer {
    fn from(wrapper: CropBoundsBuffer) -> Self {
        wrapper.0
    }
}

impl TryFrom<wgpu::Buffer> for CropBoundsBuffer {
    type Error = core::FixedSizeBufferWrapperError;

    fn try_from(buffer: wgpu::Buffer) -> Result<Self, Self::Error> {
        Self::verify_buffer_size(&buffer).map(|()| Self(buffer))
    }
}

impl FixedSizeBufferWrapper for CropBoundsBuffer {
    type Pod = CropBoundsPod;
}

/// The POD representation of [`CropBoundsBuffer`].
///
/// Field order matches the `CropBounds` uniform in `preprocess.wesl`. std140 aligns each
/// `vec3<f32>` to 16 bytes, so the scalar `enabled` / `_padding` deliberately occupy the trailing
/// 4 bytes of each `vec3`'s 16-byte slot (total size 32 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CropBoundsPod {
    /// AABB minimum corner (original Gaussian space).
    pub min: Vec3,
    /// Non-zero = cropping active; `0` = keep all Gaussians.
    pub enabled: u32,
    /// AABB maximum corner (original Gaussian space).
    pub max: Vec3,
    pub _padding: u32,
}

impl CropBoundsPod {
    /// Create a new crop-bounds POD.
    pub fn new(min: Vec3, max: Vec3, enabled: bool) -> Self {
        Self {
            min,
            enabled: enabled as u32,
            max,
            _padding: 0,
        }
    }
}
