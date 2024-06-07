use wgpu::{
    vertex_attr_array, Buffer, BufferDescriptor, BufferUsages, Device, VertexAttribute,
    VertexBufferLayout,
};

pub const VERTICES: [f32; 12] = [
    -0.8, -0.8, //
    0.8, -0.8, //
    0.8, 0.8, //
    -0.8, -0.8, //
    0.8, 0.8, //
    -0.8, 0.8,
];

pub fn create_vertex_buffer(device: &Device, size: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: None,
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        size,
        mapped_at_creation: false,
    })
}

const VERTEX_ATTRIBUTES: &[VertexAttribute] = &vertex_attr_array![0 => Float32x2];

pub fn get_vertex_buffer_layout<'a>() -> VertexBufferLayout<'a> {
    VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: VERTEX_ATTRIBUTES,
    }
}
