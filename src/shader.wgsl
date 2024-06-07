struct VertexIn {
    @location(0) pos: vec2<f32>, 
    @builtin(instance_index) i: u32,
}

struct VertexOut {
    @builtin(position) vertex_p: vec4<f32>,
    @location(0) cell_hue: vec2<f32>,
}

@group(0) @binding(0) var<uniform> grid_size: vec2<f32>;
@group(0) @binding(1) var<storage> cellState: array<f32>;

@vertex
fn vertexMain(in: VertexIn) -> VertexOut {
    let cell_offset_x = f32(in.i) % grid_size.x;
    let cell_offset_y = floor(f32(in.i) / grid_size.y);
    let cell_offset = vec2<f32>(cell_offset_x, cell_offset_y) * 2 / grid_size;
    let grid_pos = (in.pos * cellState[in.i] + 1) / grid_size - 1 + cell_offset;

    var out: VertexOut;
    out.vertex_p = vec4<f32>(grid_pos, 0.0, 1.0);
    out.cell_hue = cell_offset / 2;
    return out;
}

@fragment
fn fragmentMain(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.cell_hue, 1 - in.cell_hue.x, 1.0);
}
