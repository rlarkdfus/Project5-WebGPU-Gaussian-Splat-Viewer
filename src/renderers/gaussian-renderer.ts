import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  set_gaussian_multiplier: (value: number) => void,
  set_frustum_culling: (enabled: boolean) => void,
  set_workgroup_size: (size: number) => void,
  set_num_active: (count: number) => void,
  workgroup_sizes: number[],
}

// candidate preprocess workgroup sizes; ones above the device limit are skipped
const c_preprocess_wg_sizes = [16, 32, 64, 128, 256, 512, 1024];

const c_size_splat = 6 * 4; // must match struct Splat in the shaders

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);
  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);
  const dispatch_reset_data = new Uint32Array([0, 1, 1]);

  const splat_buffer = createBuffer(
    device, 'splats', pc.num_points * c_size_splat,
    GPUBufferUsage.STORAGE,
  );

  // DrawIndirect: vertex_count, instance_count, first_vertex, first_instance
  // instance_count is overwritten each frame with the visible splat count
  const indirect_draw_buffer = createBuffer(
    device, 'gaussian indirect draw', 4 * 4,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    new Uint32Array([6, pc.num_points, 0, 0]),
  );

  // RenderSettings: gaussian_scaling (f32), sh_deg (f32), frustum_culling (u32), num_active (u32)
  const render_settings = new ArrayBuffer(4 * 4);
  const render_settings_f32 = new Float32Array(render_settings);
  const render_settings_u32 = new Uint32Array(render_settings);
  render_settings_f32[0] = 1.0;
  render_settings_f32[1] = pc.sh_deg;
  render_settings_u32[2] = 1;
  render_settings_u32[3] = pc.num_points;
  const render_settings_buffer = createBuffer(
    device, 'render settings', render_settings.byteLength,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    render_settings,
  );
  const write_render_settings = () => device.queue.writeBuffer(render_settings_buffer, 0, render_settings);

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  // one pipeline per workgroup size, since @workgroup_size is fixed at pipeline creation.
  // with layout 'auto', bind groups are only compatible with the pipeline they came from.
  const preprocess_module = device.createShaderModule({ code: preprocessWGSL });
  const workgroup_sizes = c_preprocess_wg_sizes.filter(
    (size) => size <= device.limits.maxComputeInvocationsPerWorkgroup
      && size <= device.limits.maxComputeWorkgroupSizeX,
  );

  const preprocess_variants = new Map(workgroup_sizes.map((size) => {
    const pipeline = device.createComputePipeline({
      label: `preprocess wg${size}`,
      layout: 'auto',
      compute: {
        module: preprocess_module,
        entryPoint: 'preprocess',
        constants: {
          workgroupSize: size,
          sortWorkgroupSize: C.histogram_wg_size,
          sortKeyPerThread: c_histogram_block_rows,
        },
      },
    });

    const camera_bind_group = device.createBindGroup({
      label: `preprocess camera wg${size}`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: camera_buffer } },
        { binding: 1, resource: { buffer: render_settings_buffer } },
      ],
    });

    const gaussian_bind_group = device.createBindGroup({
      label: `preprocess gaussians wg${size}`,
      layout: pipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
        { binding: 1, resource: { buffer: pc.sh_buffer } },
        { binding: 2, resource: { buffer: splat_buffer } },
      ],
    });

    const sort_bind_group = device.createBindGroup({
      label: `sort wg${size}`,
      layout: pipeline.getBindGroupLayout(2),
      entries: [
        { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
        { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
        { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
        { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
      ],
    });

    return [size, { pipeline, bind_groups: [camera_bind_group, gaussian_bind_group, sort_bind_group] }];
  }));

  let current_wg_size = workgroup_sizes.includes(C.histogram_wg_size)
    ? C.histogram_wg_size
    : workgroup_sizes[workgroup_sizes.length - 1];


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const render_shader = device.createShaderModule({ code: renderWGSL });
  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian render',
    layout: 'auto',
    vertex: {
      module: render_shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{
        format: presentation_format,
        // splats are drawn back to front, so standard "over" blending
        blend: {
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
      }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  const render_bind_group = device.createBindGroup({
    label: 'gaussian render',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 2, resource: { buffer: camera_buffer } },
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const preprocess = (encoder: GPUCommandEncoder) => {
    // clear the visible count and sort dispatch size; the preprocess shader rebuilds them
    device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
    device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, dispatch_reset_data);

    const variant = preprocess_variants.get(current_wg_size)!;
    const pass = encoder.beginComputePass({ label: 'preprocess' });
    pass.setPipeline(variant.pipeline);
    variant.bind_groups.forEach((bind_group, i) => pass.setBindGroup(i, bind_group));
    pass.dispatchWorkgroups(Math.ceil(render_settings_u32[3] / current_wg_size));
    pass.end();
  };

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_draw_buffer, 4, 4);

    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [
        {
          view: texture_view,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, render_bind_group);
    pass.drawIndirect(indirect_draw_buffer, 0);
    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      preprocess(encoder);
      sorter.sort(encoder);
      render(encoder, texture_view);
    },
    set_gaussian_multiplier: (value: number) => {
      render_settings_f32[0] = value;
      write_render_settings();
    },
    set_frustum_culling: (enabled: boolean) => {
      render_settings_u32[2] = enabled ? 1 : 0;
      write_render_settings();
    },
    set_workgroup_size: (size: number) => {
      if (preprocess_variants.has(size)) current_wg_size = size;
    },
    set_num_active: (count: number) => {
      render_settings_u32[3] = Math.max(1, Math.min(pc.num_points, Math.floor(count)));
      write_render_settings();
    },
    workgroup_sizes,
    camera_buffer,
  };
}
