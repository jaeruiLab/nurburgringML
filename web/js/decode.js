// Replay decoding: turns a recorded .i16.gz file back into car positions.
//
// File layout (written by record.py): gzip of little-endian int16 values in three "planes":
//   x: [car 0 frames..., car 1 frames..., ...]   second differences, units of `quant` meters
//   y: the same for y
//   h: heading keyframes, one every `heading_stride` frames plus each car's last frame,
//      first differences of round(angle * heading_steps / 2 pi)
// Car i has end_frames[i] + 1 frames. Positions add up twice; headings add up once and are
// interpolated between keyframes.

export async function gunzip(buffer) {
  if (typeof DecompressionStream === "undefined") {
    throw new Error("This browser cannot unpack the replay files (needs Chrome 80+, Safari 16.4+ or Firefox 113+).");
  }
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

export function headingKeyframes(n, stride) {
  const k = Math.floor((n - 1) / stride) + 1;                  // frames 0, stride, 2*stride, ...
  return (n - 1) % stride ? k + 1 : k;                          // ... plus the last frame
}

export function decodePlanes(values, endFrames, quant, headingStride, headingSteps) {
  // `values` is an Int16Array (typed arrays use the machine's byte order, which is
  // little-endian on every phone and computer that runs a browser).
  let total = 0, totalH = 0;
  for (const end of endFrames) { total += end + 1; totalH += headingKeyframes(end + 1, headingStride); }
  if (values.length !== 2 * total + totalH) throw new Error("replay data does not match its index");
  const hScale = (2 * Math.PI) / headingSteps;
  const cars = [];
  let o = 0, oh = 2 * total;
  for (const end of endFrames) {
    const n = end + 1;
    const x = new Float32Array(n), y = new Float32Array(n), h = new Float32Array(n);
    let vx = 0, vy = 0, px = 0, py = 0;
    for (let k = 0; k < n; k++) {
      vx += values[o + k];
      vy += values[total + o + k];
      px += vx;
      py += vy;
      x[k] = px * quant;
      y[k] = py * quant;
    }
    // Heading: add up the keyframe differences, then fill the frames in between linearly.
    let hq = 0, prevFrame = 0, prevAngle = 0;
    const nk = headingKeyframes(n, headingStride);
    for (let j = 0; j < nk; j++) {
      hq += values[oh + j];
      const frame = j === nk - 1 ? n - 1 : j * headingStride;
      const angle = hq * hScale;
      if (j === 0) h[0] = angle;
      else for (let f = prevFrame + 1; f <= frame; f++) h[f] = prevAngle + (angle - prevAngle) * (f - prevFrame) / (frame - prevFrame);
      prevFrame = frame; prevAngle = angle;
    }
    cars.push({ n, x, y, h });
    o += n;
    oh += nk;
  }
  return cars;
}

export async function loadReplay(base, meta, manifest) {
  const [res, carsRes] = await Promise.all([fetch(base + meta.file), fetch(base + meta.cars)]);
  if (!res.ok) throw new Error(`could not load ${meta.file} (${res.status})`);
  if (!carsRes.ok) throw new Error(`could not load ${meta.cars} (${carsRes.status})`);
  const buf = await res.arrayBuffer();
  // Some servers unzip .gz files on the fly; only gunzip if the data still starts with the gzip signature.
  const head = new Uint8Array(buf, 0, Math.min(2, buf.byteLength));
  const raw = head[0] === 0x1f && head[1] === 0x8b ? await gunzip(buf) : buf;
  const info = await carsRes.json();                             // per-car results
  const cars = decodePlanes(new Int16Array(raw), info.end_frames, manifest.quant,
                            manifest.heading_stride, manifest.heading_steps);
  return { cars, info };
}
