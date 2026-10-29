const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortWorkgroupSize: u32;  // must match the sorter's histogram workgroup size, independent of workgroupSize
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
    frustum_culling: u32,  // 0 = off, 1 = on
    num_active: u32,       // only the first num_active gaussians are processed
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    pos: u32,
    size: u32,
    conic_opacity: array<u32,2>,
    color: array<u32,2>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> render_settings: RenderSettings;

@group(1) @binding(0)
var<storage, read> gaussians : array<Gaussian>;
@group(1) @binding(1)
var<storage, read> sh_buffer : array<u32>;
@group(1) @binding(2)
var<storage, read_write> splats : array<Splat>;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let base = splat_idx * 48u + c_idx * 3u;
    return vec3<f32>(read_f16(base), read_f16(base + 1u), read_f16(base + 2u));
}

fn read_f16(half_idx: u32) -> f32 {
    let pair = unpack2x16float(sh_buffer[half_idx / 2u]);
    return select(pair.x, pair.y, (half_idx & 1u) == 1u);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&gaussians) || idx >= render_settings.num_active) {
        return;
    }

    let g = gaussians[idx];
    let a = unpack2x16float(g.pos_opacity[0]);
    let b = unpack2x16float(g.pos_opacity[1]);
    let pos = vec4<f32>(a.x, a.y, b.x, 1.0);

    // Implement view frustum culling to remove non-visible Gaussians (make bounding box to be slightly larger to keep the edge gaussians)
    let view_pos = camera.view * pos;
    if (view_pos.z < 0.2) {
        return;
    }
    let clip = camera.proj * view_pos;
    let ndc = clip.xy / clip.w;
    if (render_settings.frustum_culling != 0u && any(abs(ndc) > vec2<f32>(1.2))) {
        return;
    }

    // `opacity`: use sigmoid function to bring back maximum opacity
    let opacity = 1.0 / (1.0 + exp(-b.y));

    // Compute 3D covariance based on rotation and scale, also user inputted gaussian multipler.
    let r01 = unpack2x16float(g.rot[0]);
    let r23 = unpack2x16float(g.rot[1]);
    let q = normalize(vec4<f32>(r01.x, r01.y, r23.x, r23.y));
    let qw = q.x; let qx = q.y; let qy = q.z; let qz = q.w;
    let R = transpose(mat3x3<f32>(
        1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy),
        2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx),
        2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy),
    ));

    // `scale` (log space): use exponential function to bring back actual scale
    let s01 = unpack2x16float(g.scale[0]);
    let s23 = unpack2x16float(g.scale[1]);
    let scale = exp(vec3<f32>(s01.x, s01.y, s23.x)) * render_settings.gaussian_scaling;
    let S = mat3x3<f32>(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z,
    );
    let M = R * S;
    let cov3d = M * transpose(M);

    // Compute 2D conic, maximum radius, and maximum quad size in NDC
    let focal = camera.focal;
    let tan_fov = 0.5 * camera.viewport / focal;
    let lim = 1.3 * tan_fov;
    let tz = view_pos.z;
    let txy = clamp(view_pos.xy / tz, -lim, lim) * tz;

    let J = mat3x3<f32>(
        focal.x / tz, 0.0, 0.0,
        0.0, focal.y / tz, 0.0,
        -focal.x * txy.x / (tz * tz), -focal.y * txy.y / (tz * tz), 0.0,
    );
    let W = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let T = J * W;
    let cov2d_full = T * cov3d * transpose(T);
    let cov2d = vec3<f32>(cov2d_full[0][0] + 0.3, cov2d_full[0][1], cov2d_full[1][1] + 0.3);

    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det <= 0.0) {
        return;
    }
    let conic = vec3<f32>(cov2d.z, -cov2d.y, cov2d.x) / det;

    let mid = 0.5 * (cov2d.x + cov2d.z);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3.0 * sqrt(lambda1));
    let size = 2.0 * vec2<f32>(radius) / camera.viewport;

    // Using spherical harmonics coeffiecients to evaluate the color of the gaussian from particular view direction
    let cam_pos = camera.view_inv[3].xyz;
    let dir = normalize(pos.xyz - cam_pos);
    let color = computeColorFromSH(dir, idx, u32(render_settings.sh_deg));

    // Store essential 2D gaussian data for later rasteriation pipeline
    let store_idx = atomicAdd(&sort_infos.keys_size, 1u);
    splats[store_idx] = Splat(
        pack2x16float(ndc),
        pack2x16float(size),
        array<u32,2>(pack2x16float(conic.xy), pack2x16float(vec2<f32>(conic.z, opacity))),
        array<u32,2>(pack2x16float(color.rg), pack2x16float(vec2<f32>(color.b, 0.0))),
    );

    // Add key_size, indices, and depth to sorter.
    sort_depths[store_idx] = ~bitcast<u32>(view_pos.z);
    sort_indices[store_idx] = store_idx;

    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    let keys_per_dispatch = sortWorkgroupSize * sortKeyPerThread;
    if (store_idx % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}