/*
    GPU pass timing using the 'timestamp-query' feature.

    Each labelled pass gets a begin/end timestamp pair. After a frame is submitted the
    timestamps are copied to a mappable buffer and read back asynchronously, and the
    per-pass durations are averaged over a window of frames.

    Chrome quantizes timestamps to 100us unless launched with --enable-webgpu-developer-features.
*/

export interface GpuTimer {
  // timestamp writes for one pass; works for both compute and render passes
  writes: (label: string) => GPUComputePassTimestampWrites & GPURenderPassTimestampWrites,
  // timestamp writes split across several passes: begin on the first, end on the last
  writes_begin: (label: string) => GPUComputePassTimestampWrites,
  writes_end: (label: string) => GPUComputePassTimestampWrites,
  // record the query resolve + copy; call once at the end of the frame's encoder
  resolve: (encoder: GPUCommandEncoder) => void,
  // start reading back this frame's timestamps; call after queue.submit
  read: () => void,
  // averaged milliseconds per label, plus 'total' (first begin to last end); 0 if the pass did not run
  results: Record<string, number>,
}

const c_window = 30;       // frames averaged per published result
const c_readback_count = 4; // frames that may be in flight at once

export function create_gpu_timer(device: GPUDevice, labels: string[]): GpuTimer | null {
  if (!device.features.has('timestamp-query')) return null;

  const query_count = labels.length * 2;
  const byte_size = query_count * 8;

  const query_set = device.createQuerySet({ label: 'gpu timer', type: 'timestamp', count: query_count });
  const resolve_buffer = device.createBuffer({
    label: 'gpu timer resolve',
    size: byte_size,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const free_readbacks: GPUBuffer[] = Array.from({ length: c_readback_count }, (_, i) => device.createBuffer({
    label: `gpu timer readback ${i}`,
    size: byte_size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  }));

  const index_of = (label: string) => {
    const i = labels.indexOf(label);
    if (i < 0) throw new Error(`unknown gpu timer label '${label}'`);
    return i;
  };

  // which labels had timestamps written in the frame being recorded
  let used = new Set<number>();
  let pending: { buffer: GPUBuffer, used: Set<number> } | null = null;

  const results: Record<string, number> = { total: 0 };
  labels.forEach((label) => results[label] = 0);
  const sums = new Array(labels.length).fill(0);
  const counts = new Array(labels.length).fill(0);
  let total_sum = 0;
  let samples = 0;

  const accumulate = (ts: BigUint64Array, frame_used: Set<number>) => {
    let first = -1n;
    let last = -1n;
    for (const i of frame_used) {
      const begin = ts[2 * i];
      const end = ts[2 * i + 1];
      if (end < begin) return; // bogus sample (e.g. timestamp reset), drop the whole frame
      sums[i] += Number(end - begin) / 1e6;
      counts[i]++;
      if (first < 0n || begin < first) first = begin;
      if (end > last) last = end;
    }
    total_sum += Number(last - first) / 1e6;
    samples++;

    if (samples >= c_window) {
      labels.forEach((label, i) => {
        results[label] = counts[i] > 0 ? sums[i] / counts[i] : 0;
        sums[i] = 0;
        counts[i] = 0;
      });
      results.total = total_sum / samples;
      total_sum = 0;
      samples = 0;
    }
  };

  return {
    writes: (label) => {
      const i = index_of(label);
      used.add(i);
      return { querySet: query_set, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 };
    },
    writes_begin: (label) => {
      const i = index_of(label);
      used.add(i);
      return { querySet: query_set, beginningOfPassWriteIndex: 2 * i };
    },
    writes_end: (label) => {
      const i = index_of(label);
      used.add(i);
      return { querySet: query_set, endOfPassWriteIndex: 2 * i + 1 };
    },
    resolve: (encoder) => {
      const buffer = free_readbacks.pop();
      // skip this frame if nothing was timed or every readback buffer is still being mapped
      if (used.size === 0 || !buffer) {
        if (buffer) free_readbacks.push(buffer);
        used = new Set();
        return;
      }
      encoder.resolveQuerySet(query_set, 0, query_count, resolve_buffer, 0);
      encoder.copyBufferToBuffer(resolve_buffer, 0, buffer, 0, byte_size);
      pending = { buffer, used };
      used = new Set();
    },
    read: () => {
      if (!pending) return;
      const { buffer, used: frame_used } = pending;
      pending = null;
      buffer.mapAsync(GPUMapMode.READ).then(() => {
        accumulate(new BigUint64Array(buffer.getMappedRange().slice(0)), frame_used);
        buffer.unmap();
        free_readbacks.push(buffer);
      }).catch(() => {
        // device lost or buffer destroyed; just drop the sample
      });
    },
    results,
  };
}
