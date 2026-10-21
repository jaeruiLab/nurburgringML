// Replay decoding: turns a recorded .i16.gz file back into car positions.
//
// File layout (written by record.py): gzip of little-endian int16 values in three "planes":
//   [x of car 0, x of car 1, ...] [y of car 0, ...] [heading of car 0, ...]
// Car i has end_frames[i] + 1 frames. x and y are second differences in units of `quant`
// meters (add up twice to get positions); heading is a first difference of
// round(angle / pi * 32767) that wraps around like a 16-bit integer.

export async function gunzip(buffer) {
  if (typeof DecompressionStream === "undefined") {
    throw new Error("This browser cannot unpack the replay files (needs Chrome 80+, Safari 16.4+ or Firefox 113+).");
  }
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

export function decodePlanes(values, endFrames, quant, headingScale) {
  // `values` is an Int16Array (typed arrays use the machine's byte order, which is
  // little-endian on every phone and computer that runs a browser).
  const total = values.length / 3;
  if (!Number.isInteger(total)) throw new Error("replay data is not a whole number of frames");
  const cars = [];
  let o = 0;
  for (const end of endFrames) {
    const n = end + 1;
    const x = new Float32Array(n), y = new Float32Array(n), h = new Float32Array(n);
    let vx = 0, vy = 0, px = 0, py = 0, ph = 0;
    for (let k = 0; k < n; k++) {
      vx += values[o + k];
      vy += values[total + o + k];
      px += vx;
      py += vy;
      ph = ((ph + values[2 * total + o + k] + 32768) & 0xffff) - 32768;
      x[k] = px * quant;
      y[k] = py * quant;
      h[k] = ph / headingScale;
    }
    cars.push({ n, x, y, h });
    o += n;
  }
  if (o !== total) throw new Error("replay data does not match its index");
  return cars;
}

export async function loadReplay(url, meta, manifest) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not load ${url} (${res.status})`);
  const buf = await res.arrayBuffer();
  // Some servers unzip .gz files on the fly; only gunzip if the data still starts with the gzip signature.
  const head = new Uint8Array(buf, 0, Math.min(2, buf.byteLength));
  const raw = head[0] === 0x1f && head[1] === 0x8b ? await gunzip(buf) : buf;
  return decodePlanes(new Int16Array(raw), meta.end_frames, manifest.quant, manifest.heading_scale);
}
