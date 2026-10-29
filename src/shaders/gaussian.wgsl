struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) center: vec2<f32>,
    @location(1) @interpolate(flat) conic: vec3<f32>,
    @location(2) @interpolate(flat) color: vec4<f32>,
};

struct Splat {
    pos: u32,
    size: u32,
    conic_opacity: array<u32,2>,
    color: array<u32,2>,
};

@group(0) @binding(0)
var<storage, read> splats : array<Splat>;
@group(0) @binding(1)
var<storage, read> sort_indices : array<u32>;
@group(0) @binding(2)
var<uniform> camera: CameraUniforms;

const QUAD = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    // Render the 2D splat on quad utlizing indirect draw call (instance count from process step) in sorted order.
    let splat = splats[sort_indices[instance_index]];
    let center = unpack2x16float(splat.pos);
    let size = unpack2x16float(splat.size);
    let co0 = unpack2x16float(splat.conic_opacity[0]);
    let co1 = unpack2x16float(splat.conic_opacity[1]);
    let c0 = unpack2x16float(splat.color[0]);
    let c1 = unpack2x16float(splat.color[1]);

    // vertex shader: reconstruct 2D quad vertices (NDC) from splat data, send conic and color information to fragment shader
    var out: VertexOutput;
    out.position = vec4<f32>(center + QUAD[vertex_index] * size, 0.0, 1.0);
    out.center = vec2<f32>(center.x * 0.5 + 0.5, 0.5 - center.y * 0.5) * camera.viewport;
    out.conic = vec3<f32>(co0.x, co0.y, co1.x);
    out.color = vec4<f32>(c0.x, c0.y, c1.x, co1.y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // To determine whether a point is inside a splat, you need to compute the offset from the fragment position to the center of the quad in pixel space, with the x-coordinate reversed.
    let d = in.center - in.position.xy;
    // fragment shader: using conic matrix see "Centered matrix equation" and CUDA implementation to determine whether point is inside splat. The opacity should decade exponentially as it distant from center.
    let power = -0.5 * (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y) - in.conic.y * d.x * d.y;
    if (power > 0.0) {
        discard;
    }
    let alpha = min(0.99, in.color.a * exp(power));
    if (alpha < 1.0 / 255.0) {
        discard;
    }
    return vec4<f32>(in.color.rgb, alpha);
}
