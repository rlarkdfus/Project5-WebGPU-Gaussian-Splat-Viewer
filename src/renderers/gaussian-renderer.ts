import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  set_gaussian_multiplier: (value: number) => void,
}

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

  // RenderSettings: gaussian_scaling, sh_deg
  const render_settings = new Float32Array([1.0, pc.sh_deg]);
  const render_settings_buffer = createBuffer(
    device, 'render settings', render_settings.byteLength,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    render_settings,
  );

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const camera_bind_group = device.createBindGroup({
    label: 'preprocess camera',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: render_settings_buffer } },
    ],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'preprocess gaussians',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: pc.sh_buffer } },
      { binding: 2, resource: { buffer: splat_buffer } },
    ],
  });

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });


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
      // after 4 radix passes the sorted result is back in ping_pong[0]
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

    const pass = encoder.beginComputePass({ label: 'preprocess' });
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, gaussian_bind_group);
    pass.setBindGroup(2, sort_bind_group);
    pass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size));
    pass.end();
  };

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    // instance_count = number of visible splats (sort_infos.keys_size)
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
      render_settings[0] = value;
      device.queue.writeBuffer(render_settings_buffer, 0, render_settings);
    },
    camera_buffer,
  };
}
