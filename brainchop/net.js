
const mindgrab = (() => {
const getTensorBuffer = (safetensorBuffer, tensorMetadata) => {
  return safetensorBuffer.subarray(...tensorMetadata.data_offsets);
};

const getTensorMetadata = (safetensorBuffer) => {
    const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
    return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}]));
};

const createEmptyBuf = (device, size) => {
    return device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
};

const createUniformBuf = (device, size) => {
  return device.createBuffer({size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST})
}

const createInfinityUniformBuf = (device) => {
  const size = 4;
  const buf = device.createBuffer({
    mappedAtCreation: true,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  new Float32Array(buf.getMappedRange())[0] = Infinity;
  buf.unmap();
  return buf;
};

const createWeightBuf = (device, size, data) => {
  const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
  new Uint8Array(buf.getMappedRange()).set(data); buf.unmap();
  return buf;
};

const addComputePass = (device, commandEncoder, pipeline, layout, infinityUniformBuf, bufs, workgroup) => {
  const bindGroup = device.createBindGroup({
    layout: layout,
    entries: [
      { binding: 0, resource: { buffer: infinityUniformBuf } },
      ...bufs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } }))
    ]
  });

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroup);
  passEncoder.end();
};

const r_5_256_32_4_8_16_3_4_3_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 256 */
  var gidx2 = i32(gindex.z); /* 5 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var precast0 = (gidx0&3);
  var cast0 = bitcast<u32>(precast0);
  var precast1 = gidx1;
  var precast2 = (bitcast<u32>(precast1)<<16u);
  var cast1 = bitcast<i32>(precast2);
  var precast3 = (gidx0>>2);
  var precast4 = (bitcast<u32>(precast3)<<11u);
  var cast2 = bitcast<i32>(precast4);
  var precast5 = (cast0<<6u);
  var cast3 = bitcast<i32>(precast5);
  var precast6 = lidx0;
  var precast7 = (bitcast<u32>(precast6)<<8u);
  var cast4 = bitcast<i32>(precast7);
  var precast8 = lidx1;
  var precast9 = (bitcast<u32>(precast8)<<2u);
  var cast5 = bitcast<i32>(precast9);
  var precast10 = (cast0<<4u);
  var alu0 = (lidx1+bitcast<i32>(precast10));
  var alu1 = (alu0<60);
  var alu2 = ((alu0<4)!=true);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  for (var ridx0 = 0; ridx0 < 3; ridx0++) {
    var precast11 = ridx0;
    var cast6 = bitcast<u32>(precast11);
    var alu3 = ((gidx2*81)+(ridx0*9));
    var val0 = data2[alu3];
    var val1 = data2[(alu3+1)];
    var val2 = data2[(alu3+2)];
    var val3 = data2[(alu3+3)];
    var val4 = data2[(alu3+4)];
    var val5 = data2[(alu3+5)];
    var val6 = data2[(alu3+6)];
    var val7 = data2[(alu3+7)];
    var val8 = data2[(alu3+8)];
    var val9 = data2[(alu3+27)];
    var val10 = data2[(alu3+28)];
    var val11 = data2[(alu3+29)];
    var val12 = data2[(alu3+30)];
    var val13 = data2[(alu3+31)];
    var val14 = data2[(alu3+32)];
    var val15 = data2[(alu3+33)];
    var val16 = data2[(alu3+34)];
    var val17 = data2[(alu3+35)];
    var val18 = data2[(alu3+54)];
    var val19 = data2[(alu3+55)];
    var val20 = data2[(alu3+56)];
    var val21 = data2[(alu3+57)];
    var val22 = data2[(alu3+58)];
    var val23 = data2[(alu3+59)];
    var val24 = data2[(alu3+60)];
    var val25 = data2[(alu3+61)];
    var val26 = data2[(alu3+62)];
    var precast12 = (cast6<<4u);
    var alu4 = (gidx1+bitcast<i32>(precast12));
    var precast13 = (cast6<<20u);
    var alu5 = (cast1+bitcast<i32>(precast13)+cast4+cast2+cast5+cast3);
    var alu6 = (((alu4<16)!=true)&(alu4<272));
    var val27 = select(0.0f, data1[(alu5+-1048576)], alu6);
    var val28 = select(0.0f, data1[(alu5+-1048575)], alu6);
    var val29 = select(0.0f, data1[(alu5+-1048574)], alu6);
    var val30 = select(0.0f, data1[(alu5+-1048573)], alu6);
    var alu7 = (alu6&(gidx0<120));
    var val31 = select(0.0f, data1[(alu5+-1044480)], alu7);
    var val32 = select(0.0f, data1[(alu5+-1044479)], alu7);
    var val33 = select(0.0f, data1[(alu5+-1044478)], alu7);
    var val34 = select(0.0f, data1[(alu5+-1044477)], alu7);
    var alu8 = (alu6&alu1);
    var val35 = select(0.0f, data1[(alu5+-1048560)], alu8);
    var val36 = select(0.0f, data1[(alu5+-1048559)], alu8);
    var val37 = select(0.0f, data1[(alu5+-1048558)], alu8);
    var val38 = select(0.0f, data1[(alu5+-1048557)], alu8);
    var alu9 = (alu6&((gidx0<8)!=true));
    var val39 = select(0.0f, data1[(alu5+-1052672)], alu9);
    var val40 = select(0.0f, data1[(alu5+-1052671)], alu9);
    var val41 = select(0.0f, data1[(alu5+-1052670)], alu9);
    var val42 = select(0.0f, data1[(alu5+-1052669)], alu9);
    var alu10 = (alu6&alu2);
    var val43 = select(0.0f, data1[(alu5+-1048592)], alu10);
    var val44 = select(0.0f, data1[(alu5+-1048591)], alu10);
    var val45 = select(0.0f, data1[(alu5+-1048590)], alu10);
    var val46 = select(0.0f, data1[(alu5+-1048589)], alu10);
    var alu11 = (alu7&alu1);
    var val47 = select(0.0f, data1[(alu5+-1044464)], alu11);
    var val48 = select(0.0f, data1[(alu5+-1044463)], alu11);
    var val49 = select(0.0f, data1[(alu5+-1044462)], alu11);
    var val50 = select(0.0f, data1[(alu5+-1044461)], alu11);
    var alu12 = (alu7&alu2);
    var val51 = select(0.0f, data1[(alu5+-1044496)], alu12);
    var val52 = select(0.0f, data1[(alu5+-1044495)], alu12);
    var val53 = select(0.0f, data1[(alu5+-1044494)], alu12);
    var val54 = select(0.0f, data1[(alu5+-1044493)], alu12);
    var alu13 = (alu9&alu1);
    var val55 = select(0.0f, data1[(alu5+-1052656)], alu13);
    var val56 = select(0.0f, data1[(alu5+-1052655)], alu13);
    var val57 = select(0.0f, data1[(alu5+-1052654)], alu13);
    var val58 = select(0.0f, data1[(alu5+-1052653)], alu13);
    var alu14 = (alu9&alu2);
    var val59 = select(0.0f, data1[(alu5+-1052688)], alu14);
    var val60 = select(0.0f, data1[(alu5+-1052687)], alu14);
    var val61 = select(0.0f, data1[(alu5+-1052686)], alu14);
    var val62 = select(0.0f, data1[(alu5+-1052685)], alu14);
    acc0 = (acc0+(val59*val0)+(val43*val3)+(val51*val6)+(val39*val1)+(val27*val4)+(val31*val7)+(val55*val2)+(val35*val5)+(val47*val8));
    acc1 = (acc1+(val59*val9)+(val43*val12)+(val51*val15)+(val39*val10)+(val27*val13)+(val31*val16)+(val55*val11)+(val35*val14)+(val47*val17));
    acc2 = (acc2+(val59*val18)+(val43*val21)+(val51*val24)+(val39*val19)+(val27*val22)+(val31*val25)+(val55*val20)+(val35*val23)+(val47*val26));
    acc3 = (acc3+(val60*val0)+(val44*val3)+(val52*val6)+(val40*val1)+(val28*val4)+(val32*val7)+(val56*val2)+(val36*val5)+(val48*val8));
    acc4 = (acc4+(val60*val9)+(val44*val12)+(val52*val15)+(val40*val10)+(val28*val13)+(val32*val16)+(val56*val11)+(val36*val14)+(val48*val17));
    acc5 = (acc5+(val60*val18)+(val44*val21)+(val52*val24)+(val40*val19)+(val28*val22)+(val32*val25)+(val56*val20)+(val36*val23)+(val48*val26));
    acc6 = (acc6+(val61*val0)+(val45*val3)+(val53*val6)+(val41*val1)+(val29*val4)+(val33*val7)+(val57*val2)+(val37*val5)+(val49*val8));
    acc7 = (acc7+(val61*val9)+(val45*val12)+(val53*val15)+(val41*val10)+(val29*val13)+(val33*val16)+(val57*val11)+(val37*val14)+(val49*val17));
    acc8 = (acc8+(val61*val18)+(val45*val21)+(val53*val24)+(val41*val19)+(val29*val22)+(val33*val25)+(val57*val20)+(val37*val23)+(val49*val26));
    acc9 = (acc9+(val62*val0)+(val46*val3)+(val54*val6)+(val42*val1)+(val30*val4)+(val34*val7)+(val58*val2)+(val38*val5)+(val50*val8));
    acc10 = (acc10+(val62*val9)+(val46*val12)+(val54*val15)+(val42*val10)+(val30*val13)+(val34*val16)+(val58*val11)+(val38*val14)+(val50*val17));
    acc11 = (acc11+(val62*val18)+(val46*val21)+(val54*val24)+(val42*val19)+(val30*val22)+(val34*val25)+(val58*val20)+(val38*val23)+(val50*val26));
  }
  var alu28 = (cast1+(gidx2*50331648)+cast2+cast3+cast4+cast5);
  data0[alu28] = acc0;
  data0[(alu28+1)] = acc3;
  data0[(alu28+2)] = acc6;
  data0[(alu28+3)] = acc9;
  data0[(alu28+16777216)] = acc1;
  data0[(alu28+16777217)] = acc4;
  data0[(alu28+16777218)] = acc7;
  data0[(alu28+16777219)] = acc10;
  data0[(alu28+33554432)] = acc2;
  data0[(alu28+33554433)] = acc5;
  data0[(alu28+33554434)] = acc8;
  data0[(alu28+33554435)] = acc11;
}`;

const r_120_32_16384_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 120 */
  var lidx0 = i32(lindex.x); /* 32 */
  var precast0 = gidx0;
  var cast0 = bitcast<u32>(precast0);
  var precast1 = lidx0;
  var precast2 = (cast0<<21u);
  var precast3 = (bitcast<u32>(precast1)<<16u);
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 16384; ridx0++) {
    var precast4 = ridx0;
    var precast5 = (bitcast<u32>(precast4)<<2u);
    var alu0 = (bitcast<i32>(precast2)+bitcast<i32>(precast3)+bitcast<i32>(precast5));
    var val0 = data1[alu0];
    var val1 = data1[(alu0+1)];
    var val2 = data1[(alu0+2)];
    var val3 = data1[(alu0+3)];
    acc0 = (acc0+val0+val1+val2+val3);
  }
  var precast6 = (cast0<<5u);
  data0[(lidx0+bitcast<i32>(precast6))] = acc0;
}`;

const r_15_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 256>;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 15 */
  var lidx0 = i32(lindex.x); /* 256 */
  var precast0 = gidx0;
  var precast1 = (bitcast<u32>(precast0)<<8u);
  var val0 = data1[(lidx0+bitcast<i32>(precast1))];
  temp0[lidx0] = val0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc0 = 0.0f;
    for (var ridx0 = 0; ridx0 < 256; ridx0++) {
      var val1 = temp0[ridx0];
      acc0 = (acc0+val1);
    }
    data0[gidx0] = (acc0*5.960464477539063e-08f);
  }
}`;

const r_5_4_3_16_16384_4_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(3,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 5 */
  var lidx0 = i32(lindex.x); /* 3 */
  var lidx1 = i32(lindex.y); /* 16 */
  var precast0 = gidx0;
  var cast0 = bitcast<u32>(precast0);
  var precast1 = lidx0;
  var cast1 = bitcast<u32>(precast1);
  var precast2 = lidx1;
  var cast2 = bitcast<u32>(precast2);
  var val0 = data2[(lidx0+(gidx1*3))];
  var precast3 = (cast0<<22u);
  var precast4 = (cast1<<24u);
  var precast5 = (cast2<<18u);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 16384; ridx0++) {
    var precast6 = ridx0;
    var precast7 = (bitcast<u32>(precast6)<<2u);
    var alu0 = (bitcast<i32>(precast3)+(gidx1*50331648)+bitcast<i32>(precast4)+bitcast<i32>(precast5)+bitcast<i32>(precast7));
    var val1 = data1[alu0];
    var val2 = data1[(alu0+1)];
    var val3 = data1[(alu0+2)];
    var val4 = data1[(alu0+3)];
    var val5 = data1[(alu0+65536)];
    var val6 = data1[(alu0+65537)];
    var val7 = data1[(alu0+65538)];
    var val8 = data1[(alu0+65539)];
    var val9 = data1[(alu0+131072)];
    var val10 = data1[(alu0+131073)];
    var val11 = data1[(alu0+131074)];
    var val12 = data1[(alu0+131075)];
    var val13 = data1[(alu0+196608)];
    var val14 = data1[(alu0+196609)];
    var val15 = data1[(alu0+196610)];
    var val16 = data1[(alu0+196611)];
    var alu1 = (val1-val0);
    var alu2 = (val2-val0);
    var alu3 = (val3-val0);
    var alu4 = (val4-val0);
    var alu5 = (val5-val0);
    var alu6 = (val6-val0);
    var alu7 = (val7-val0);
    var alu8 = (val8-val0);
    var alu9 = (val9-val0);
    var alu10 = (val10-val0);
    var alu11 = (val11-val0);
    var alu12 = (val12-val0);
    var alu13 = (val13-val0);
    var alu14 = (val14-val0);
    var alu15 = (val15-val0);
    var alu16 = (val16-val0);
    acc0 = (acc0+(alu1*alu1)+(alu2*alu2)+(alu3*alu3)+(alu4*alu4));
    acc1 = (acc1+(alu5*alu5)+(alu6*alu6)+(alu7*alu7)+(alu8*alu8));
    acc2 = (acc2+(alu9*alu9)+(alu10*alu10)+(alu11*alu11)+(alu12*alu12));
    acc3 = (acc3+(alu13*alu13)+(alu14*alu14)+(alu15*alu15)+(alu16*alu16));
  }
  var precast8 = (cast0<<6u);
  var precast9 = (cast1<<8u);
  var precast10 = (cast2<<2u);
  var alu22 = (bitcast<i32>(precast8)+(gidx1*768)+bitcast<i32>(precast9)+bitcast<i32>(precast10));
  data0[alu22] = acc0;
  data0[(alu22+1)] = acc1;
  data0[(alu22+2)] = acc2;
  data0[(alu22+3)] = acc3;
}`;

const r_15_256n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 256>;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 15 */
  var lidx0 = i32(lindex.x); /* 256 */
  var precast0 = gidx0;
  var precast1 = (bitcast<u32>(precast0)<<8u);
  var val0 = data1[(lidx0+bitcast<i32>(precast1))];
  temp0[lidx0] = val0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc0 = 0.0f;
    for (var ridx0 = 0; ridx0 < 256; ridx0++) {
      var val1 = temp0[ridx0];
      acc0 = (acc0+val1);
    }
    data0[gidx0] = (1/sqrt(((acc0*5.960464477539063e-08f)+1e-05f)));
  }
}`;

const E_5_262144_3_16_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(3,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32768 */
  var gidx1 = i32(gindex.y); /* 40 */
  var lidx0 = i32(lindex.x); /* 3 */
  var lidx1 = i32(lindex.y); /* 16 */
  var precast0 = gidx0;
  var precast1 = lidx0;
  var precast2 = lidx1;
  var alu0 = (lidx0+(gidx1*3));
  var val0 = data2[alu0];
  var val1 = data3[alu0];
  var precast3 = (bitcast<u32>(precast0)<<6u);
  var precast4 = (bitcast<u32>(precast1)<<24u);
  var precast5 = (bitcast<u32>(precast2)<<2u);
  var alu1 = (bitcast<i32>(precast3)+(gidx1*50331648)+bitcast<i32>(precast4)+bitcast<i32>(precast5));
  var val2 = data1[alu1];
  var alu2 = (alu1+1);
  var val3 = data1[alu2];
  var alu3 = (alu1+2);
  var val4 = data1[alu3];
  var alu4 = (alu1+3);
  var val5 = data1[alu4];
  var alu5 = ((val2-val0)*val1);
  data0[alu1] = (0.5f*alu5*2.0f*(1/(1.0f+exp2(((alu5+(0.044715f*alu5*alu5*alu5))*-2.302208198144325f)))));
  var alu7 = ((val3-val0)*val1);
  data0[alu2] = (0.5f*alu7*2.0f*(1/(1.0f+exp2(((alu7+(0.044715f*alu7*alu7*alu7))*-2.302208198144325f)))));
  var alu9 = ((val4-val0)*val1);
  data0[alu3] = (0.5f*alu9*2.0f*(1/(1.0f+exp2(((alu9+(0.044715f*alu9*alu9*alu9))*-2.302208198144325f)))));
  var alu11 = ((val5-val0)*val1);
  data0[alu4] = (0.5f*alu11*2.0f*(1/(1.0f+exp2(((alu11+(0.044715f*alu11*alu11*alu11))*-2.302208198144325f)))));
}`;

const r_5_256_32_4_8_16_15_3_4_3_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 256 */
  var gidx2 = i32(gindex.z); /* 5 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var precast0 = (gidx0&3);
  var cast0 = bitcast<u32>(precast0);
  var precast1 = gidx1;
  var precast2 = (bitcast<u32>(precast1)<<16u);
  var cast1 = bitcast<i32>(precast2);
  var precast3 = (gidx0>>2);
  var precast4 = (bitcast<u32>(precast3)<<11u);
  var cast2 = bitcast<i32>(precast4);
  var precast5 = (cast0<<6u);
  var cast3 = bitcast<i32>(precast5);
  var precast6 = lidx0;
  var precast7 = (bitcast<u32>(precast6)<<8u);
  var cast4 = bitcast<i32>(precast7);
  var precast8 = lidx1;
  var precast9 = (bitcast<u32>(precast8)<<2u);
  var cast5 = bitcast<i32>(precast9);
  var precast10 = (cast0<<4u);
  var alu0 = (lidx1+bitcast<i32>(precast10));
  var alu1 = (alu0<62);
  var alu2 = ((alu0<2)!=true);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  for (var ridx0 = 0; ridx0 < 15; ridx0++) {
    var precast11 = ridx0;
    var precast12 = (bitcast<u32>(precast11)<<24u);
    for (var ridx1 = 0; ridx1 < 3; ridx1++) {
      var precast13 = ridx1;
      var cast6 = bitcast<u32>(precast13);
      var precast14 = (cast6<<3u);
      var alu3 = (gidx1+bitcast<i32>(precast14));
      var alu4 = (((alu3<8)!=true)&(alu3<264));
      var alu5 = (alu4&((gidx0<4)!=true));
      var alu6 = (alu4&(gidx0<124));
      var precast15 = (cast6<<19u);
      var alu7 = (alu4&alu1);
      var alu8 = (alu4&alu2);
      var alu9 = (alu6&alu1);
      var alu10 = (alu6&alu2);
      var alu11 = (alu5&alu1);
      var alu12 = (alu5&alu2);
      var alu13 = (cast1+bitcast<i32>(precast15)+bitcast<i32>(precast12)+cast4+cast2+cast5+cast3);
      var val0 = select(0.0f, data1[(alu13+-526344)], alu12);
      var val1 = select(0.0f, data1[(alu13+-526343)], alu12);
      var val2 = select(0.0f, data1[(alu13+-526342)], alu12);
      var val3 = select(0.0f, data1[(alu13+-526341)], alu12);
      var val4 = select(0.0f, data1[(alu13+-526336)], alu5);
      var val5 = select(0.0f, data1[(alu13+-526335)], alu5);
      var val6 = select(0.0f, data1[(alu13+-526334)], alu5);
      var val7 = select(0.0f, data1[(alu13+-526333)], alu5);
      var val8 = select(0.0f, data1[(alu13+-526328)], alu11);
      var val9 = select(0.0f, data1[(alu13+-526327)], alu11);
      var val10 = select(0.0f, data1[(alu13+-526326)], alu11);
      var val11 = select(0.0f, data1[(alu13+-526325)], alu11);
      var val12 = select(0.0f, data1[(alu13+-524296)], alu8);
      var val13 = select(0.0f, data1[(alu13+-524295)], alu8);
      var val14 = select(0.0f, data1[(alu13+-524294)], alu8);
      var val15 = select(0.0f, data1[(alu13+-524293)], alu8);
      var val16 = select(0.0f, data1[(alu13+-524288)], alu4);
      var val17 = select(0.0f, data1[(alu13+-524287)], alu4);
      var val18 = select(0.0f, data1[(alu13+-524286)], alu4);
      var val19 = select(0.0f, data1[(alu13+-524285)], alu4);
      var val20 = select(0.0f, data1[(alu13+-524280)], alu7);
      var val21 = select(0.0f, data1[(alu13+-524279)], alu7);
      var val22 = select(0.0f, data1[(alu13+-524278)], alu7);
      var val23 = select(0.0f, data1[(alu13+-524277)], alu7);
      var val24 = select(0.0f, data1[(alu13+-522248)], alu10);
      var val25 = select(0.0f, data1[(alu13+-522247)], alu10);
      var val26 = select(0.0f, data1[(alu13+-522246)], alu10);
      var val27 = select(0.0f, data1[(alu13+-522245)], alu10);
      var val28 = select(0.0f, data1[(alu13+-522240)], alu6);
      var val29 = select(0.0f, data1[(alu13+-522239)], alu6);
      var val30 = select(0.0f, data1[(alu13+-522238)], alu6);
      var val31 = select(0.0f, data1[(alu13+-522237)], alu6);
      var val32 = select(0.0f, data1[(alu13+-522232)], alu9);
      var val33 = select(0.0f, data1[(alu13+-522231)], alu9);
      var val34 = select(0.0f, data1[(alu13+-522230)], alu9);
      var val35 = select(0.0f, data1[(alu13+-522229)], alu9);
      var alu14 = ((gidx2*1215)+(ridx0*27)+(ridx1*9));
      var val36 = data2[alu14];
      var val37 = data2[(alu14+1)];
      var val38 = data2[(alu14+2)];
      var val39 = data2[(alu14+3)];
      var val40 = data2[(alu14+4)];
      var val41 = data2[(alu14+5)];
      var val42 = data2[(alu14+6)];
      var val43 = data2[(alu14+7)];
      var val44 = data2[(alu14+8)];
      var val45 = data2[(alu14+405)];
      var val46 = data2[(alu14+406)];
      var val47 = data2[(alu14+407)];
      var val48 = data2[(alu14+408)];
      var val49 = data2[(alu14+409)];
      var val50 = data2[(alu14+410)];
      var val51 = data2[(alu14+411)];
      var val52 = data2[(alu14+412)];
      var val53 = data2[(alu14+413)];
      var val54 = data2[(alu14+810)];
      var val55 = data2[(alu14+811)];
      var val56 = data2[(alu14+812)];
      var val57 = data2[(alu14+813)];
      var val58 = data2[(alu14+814)];
      var val59 = data2[(alu14+815)];
      var val60 = data2[(alu14+816)];
      var val61 = data2[(alu14+817)];
      var val62 = data2[(alu14+818)];
      acc0 = (acc0+(val0*val36)+(val12*val39)+(val24*val42)+(val4*val37)+(val16*val40)+(val28*val43)+(val8*val38)+(val20*val41)+(val32*val44));
      acc1 = (acc1+(val0*val45)+(val12*val48)+(val24*val51)+(val4*val46)+(val16*val49)+(val28*val52)+(val8*val47)+(val20*val50)+(val32*val53));
      acc2 = (acc2+(val0*val54)+(val12*val57)+(val24*val60)+(val4*val55)+(val16*val58)+(val28*val61)+(val8*val56)+(val20*val59)+(val32*val62));
      acc3 = (acc3+(val1*val36)+(val13*val39)+(val25*val42)+(val5*val37)+(val17*val40)+(val29*val43)+(val9*val38)+(val21*val41)+(val33*val44));
      acc4 = (acc4+(val1*val45)+(val13*val48)+(val25*val51)+(val5*val46)+(val17*val49)+(val29*val52)+(val9*val47)+(val21*val50)+(val33*val53));
      acc5 = (acc5+(val1*val54)+(val13*val57)+(val25*val60)+(val5*val55)+(val17*val58)+(val29*val61)+(val9*val56)+(val21*val59)+(val33*val62));
      acc6 = (acc6+(val2*val36)+(val14*val39)+(val26*val42)+(val6*val37)+(val18*val40)+(val30*val43)+(val10*val38)+(val22*val41)+(val34*val44));
      acc7 = (acc7+(val2*val45)+(val14*val48)+(val26*val51)+(val6*val46)+(val18*val49)+(val30*val52)+(val10*val47)+(val22*val50)+(val34*val53));
      acc8 = (acc8+(val2*val54)+(val14*val57)+(val26*val60)+(val6*val55)+(val18*val58)+(val30*val61)+(val10*val56)+(val22*val59)+(val34*val62));
      acc9 = (acc9+(val3*val36)+(val15*val39)+(val27*val42)+(val7*val37)+(val19*val40)+(val31*val43)+(val11*val38)+(val23*val41)+(val35*val44));
      acc10 = (acc10+(val3*val45)+(val15*val48)+(val27*val51)+(val7*val46)+(val19*val49)+(val31*val52)+(val11*val47)+(val23*val50)+(val35*val53));
      acc11 = (acc11+(val3*val54)+(val15*val57)+(val27*val60)+(val7*val55)+(val19*val58)+(val31*val61)+(val11*val56)+(val23*val59)+(val35*val62));
    }
  }
  var alu29 = (cast1+(gidx2*50331648)+cast2+cast3+cast4+cast5);
  data0[alu29] = acc0;
  data0[(alu29+1)] = acc3;
  data0[(alu29+2)] = acc6;
  data0[(alu29+3)] = acc9;
  data0[(alu29+16777216)] = acc1;
  data0[(alu29+16777217)] = acc4;
  data0[(alu29+16777218)] = acc7;
  data0[(alu29+16777219)] = acc10;
  data0[(alu29+33554432)] = acc2;
  data0[(alu29+33554433)] = acc5;
  data0[(alu29+33554434)] = acc8;
  data0[(alu29+33554435)] = acc11;
}`;

const r_5_256_32_4_8_16_15_3_4_3_3_3n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 256 */
  var gidx2 = i32(gindex.z); /* 5 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var precast0 = (gidx0>>2);
  var cast0 = bitcast<u32>(precast0);
  var precast1 = (gidx0&3);
  var cast1 = bitcast<u32>(precast1);
  var precast2 = gidx1;
  var precast3 = (bitcast<u32>(precast2)<<16u);
  var cast2 = bitcast<i32>(precast3);
  var precast4 = (cast0<<11u);
  var cast3 = bitcast<i32>(precast4);
  var precast5 = (cast1<<6u);
  var cast4 = bitcast<i32>(precast5);
  var precast6 = lidx0;
  var precast7 = (bitcast<u32>(precast6)<<8u);
  var cast5 = bitcast<i32>(precast7);
  var precast8 = lidx1;
  var precast9 = (bitcast<u32>(precast8)<<2u);
  var cast6 = bitcast<i32>(precast9);
  var precast10 = (cast0<<3u);
  var alu0 = (lidx0+bitcast<i32>(precast10));
  var precast11 = (cast1<<4u);
  var alu1 = (lidx1+bitcast<i32>(precast11));
  var alu2 = (alu1<63);
  var alu3 = ((alu1<1)!=true);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  for (var ridx0 = 0; ridx0 < 15; ridx0++) {
    var precast12 = ridx0;
    var precast13 = (bitcast<u32>(precast12)<<24u);
    for (var ridx1 = 0; ridx1 < 3; ridx1++) {
      var precast14 = ridx1;
      var cast7 = bitcast<u32>(precast14);
      var precast15 = (cast7<<2u);
      var alu4 = (gidx1+bitcast<i32>(precast15));
      var alu5 = (((alu4<4)!=true)&(alu4<260));
      var alu6 = (alu5&((alu0<4)!=true));
      var alu7 = (alu5&(alu0<252));
      var precast16 = (cast7<<18u);
      var alu8 = (alu5&alu2);
      var alu9 = (alu5&alu3);
      var alu10 = (alu7&alu2);
      var alu11 = (alu7&alu3);
      var alu12 = (alu6&alu2);
      var alu13 = (alu6&alu3);
      var alu14 = (cast2+bitcast<i32>(precast16)+bitcast<i32>(precast13)+cast5+cast3+cast6+cast4);
      var val0 = select(0.0f, data1[(alu14+-263172)], alu13);
      var val1 = select(0.0f, data1[(alu14+-263171)], alu13);
      var val2 = select(0.0f, data1[(alu14+-263170)], alu13);
      var val3 = select(0.0f, data1[(alu14+-263169)], alu13);
      var val4 = select(0.0f, data1[(alu14+-263168)], alu6);
      var val5 = select(0.0f, data1[(alu14+-263167)], alu6);
      var val6 = select(0.0f, data1[(alu14+-263166)], alu6);
      var val7 = select(0.0f, data1[(alu14+-263165)], alu6);
      var val8 = select(0.0f, data1[(alu14+-263164)], alu12);
      var val9 = select(0.0f, data1[(alu14+-263163)], alu12);
      var val10 = select(0.0f, data1[(alu14+-263162)], alu12);
      var val11 = select(0.0f, data1[(alu14+-263161)], alu12);
      var val12 = select(0.0f, data1[(alu14+-262148)], alu9);
      var val13 = select(0.0f, data1[(alu14+-262147)], alu9);
      var val14 = select(0.0f, data1[(alu14+-262146)], alu9);
      var val15 = select(0.0f, data1[(alu14+-262145)], alu9);
      var val16 = select(0.0f, data1[(alu14+-262144)], alu5);
      var val17 = select(0.0f, data1[(alu14+-262143)], alu5);
      var val18 = select(0.0f, data1[(alu14+-262142)], alu5);
      var val19 = select(0.0f, data1[(alu14+-262141)], alu5);
      var val20 = select(0.0f, data1[(alu14+-262140)], alu8);
      var val21 = select(0.0f, data1[(alu14+-262139)], alu8);
      var val22 = select(0.0f, data1[(alu14+-262138)], alu8);
      var val23 = select(0.0f, data1[(alu14+-262137)], alu8);
      var val24 = select(0.0f, data1[(alu14+-261124)], alu11);
      var val25 = select(0.0f, data1[(alu14+-261123)], alu11);
      var val26 = select(0.0f, data1[(alu14+-261122)], alu11);
      var val27 = select(0.0f, data1[(alu14+-261121)], alu11);
      var val28 = select(0.0f, data1[(alu14+-261120)], alu7);
      var val29 = select(0.0f, data1[(alu14+-261119)], alu7);
      var val30 = select(0.0f, data1[(alu14+-261118)], alu7);
      var val31 = select(0.0f, data1[(alu14+-261117)], alu7);
      var val32 = select(0.0f, data1[(alu14+-261116)], alu10);
      var val33 = select(0.0f, data1[(alu14+-261115)], alu10);
      var val34 = select(0.0f, data1[(alu14+-261114)], alu10);
      var val35 = select(0.0f, data1[(alu14+-261113)], alu10);
      var alu15 = ((gidx2*1215)+(ridx0*27)+(ridx1*9));
      var val36 = data2[alu15];
      var val37 = data2[(alu15+1)];
      var val38 = data2[(alu15+2)];
      var val39 = data2[(alu15+3)];
      var val40 = data2[(alu15+4)];
      var val41 = data2[(alu15+5)];
      var val42 = data2[(alu15+6)];
      var val43 = data2[(alu15+7)];
      var val44 = data2[(alu15+8)];
      var val45 = data2[(alu15+405)];
      var val46 = data2[(alu15+406)];
      var val47 = data2[(alu15+407)];
      var val48 = data2[(alu15+408)];
      var val49 = data2[(alu15+409)];
      var val50 = data2[(alu15+410)];
      var val51 = data2[(alu15+411)];
      var val52 = data2[(alu15+412)];
      var val53 = data2[(alu15+413)];
      var val54 = data2[(alu15+810)];
      var val55 = data2[(alu15+811)];
      var val56 = data2[(alu15+812)];
      var val57 = data2[(alu15+813)];
      var val58 = data2[(alu15+814)];
      var val59 = data2[(alu15+815)];
      var val60 = data2[(alu15+816)];
      var val61 = data2[(alu15+817)];
      var val62 = data2[(alu15+818)];
      acc0 = (acc0+(val0*val36)+(val12*val39)+(val24*val42)+(val4*val37)+(val16*val40)+(val28*val43)+(val8*val38)+(val20*val41)+(val32*val44));
      acc1 = (acc1+(val0*val45)+(val12*val48)+(val24*val51)+(val4*val46)+(val16*val49)+(val28*val52)+(val8*val47)+(val20*val50)+(val32*val53));
      acc2 = (acc2+(val0*val54)+(val12*val57)+(val24*val60)+(val4*val55)+(val16*val58)+(val28*val61)+(val8*val56)+(val20*val59)+(val32*val62));
      acc3 = (acc3+(val1*val36)+(val13*val39)+(val25*val42)+(val5*val37)+(val17*val40)+(val29*val43)+(val9*val38)+(val21*val41)+(val33*val44));
      acc4 = (acc4+(val1*val45)+(val13*val48)+(val25*val51)+(val5*val46)+(val17*val49)+(val29*val52)+(val9*val47)+(val21*val50)+(val33*val53));
      acc5 = (acc5+(val1*val54)+(val13*val57)+(val25*val60)+(val5*val55)+(val17*val58)+(val29*val61)+(val9*val56)+(val21*val59)+(val33*val62));
      acc6 = (acc6+(val2*val36)+(val14*val39)+(val26*val42)+(val6*val37)+(val18*val40)+(val30*val43)+(val10*val38)+(val22*val41)+(val34*val44));
      acc7 = (acc7+(val2*val45)+(val14*val48)+(val26*val51)+(val6*val46)+(val18*val49)+(val30*val52)+(val10*val47)+(val22*val50)+(val34*val53));
      acc8 = (acc8+(val2*val54)+(val14*val57)+(val26*val60)+(val6*val55)+(val18*val58)+(val30*val61)+(val10*val56)+(val22*val59)+(val34*val62));
      acc9 = (acc9+(val3*val36)+(val15*val39)+(val27*val42)+(val7*val37)+(val19*val40)+(val31*val43)+(val11*val38)+(val23*val41)+(val35*val44));
      acc10 = (acc10+(val3*val45)+(val15*val48)+(val27*val51)+(val7*val46)+(val19*val49)+(val31*val52)+(val11*val47)+(val23*val50)+(val35*val53));
      acc11 = (acc11+(val3*val54)+(val15*val57)+(val27*val60)+(val7*val55)+(val19*val58)+(val31*val61)+(val11*val56)+(val23*val59)+(val35*val62));
    }
  }
  var alu30 = (cast2+(gidx2*50331648)+cast3+cast4+cast5+cast6);
  data0[alu30] = acc0;
  data0[(alu30+1)] = acc3;
  data0[(alu30+2)] = acc6;
  data0[(alu30+3)] = acc9;
  data0[(alu30+16777216)] = acc1;
  data0[(alu30+16777217)] = acc4;
  data0[(alu30+16777218)] = acc7;
  data0[(alu30+16777219)] = acc10;
  data0[(alu30+33554432)] = acc2;
  data0[(alu30+33554433)] = acc5;
  data0[(alu30+33554434)] = acc8;
  data0[(alu30+33554435)] = acc11;
}`;

const r_5_256_32_4_8_16_15_3_4_3_3_3n2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 256 */
  var gidx2 = i32(gindex.z); /* 5 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var precast0 = (gidx0>>2);
  var cast0 = bitcast<u32>(precast0);
  var precast1 = (gidx0&3);
  var cast1 = bitcast<u32>(precast1);
  var precast2 = lidx1;
  var cast2 = bitcast<u32>(precast2);
  var precast3 = gidx1;
  var precast4 = (bitcast<u32>(precast3)<<16u);
  var cast3 = bitcast<i32>(precast4);
  var precast5 = (cast0<<11u);
  var cast4 = bitcast<i32>(precast5);
  var precast6 = (cast1<<6u);
  var cast5 = bitcast<i32>(precast6);
  var precast7 = lidx0;
  var precast8 = (bitcast<u32>(precast7)<<8u);
  var cast6 = bitcast<i32>(precast8);
  var precast9 = (cast2<<2u);
  var cast7 = bitcast<i32>(precast9);
  var alu0 = (cast7+cast5);
  var precast10 = (cast0<<3u);
  var alu1 = (lidx0+bitcast<i32>(precast10));
  var alu2 = (alu0<251);
  var alu3 = ((alu0<1)!=true);
  var precast11 = (cast2<<1u);
  var precast12 = (cast1<<4u);
  var alu4 = ((lidx1+bitcast<i32>(precast12))<63);
  var precast13 = (cast1<<5u);
  var alu5 = (((bitcast<i32>(precast11)+bitcast<i32>(precast13))<1)!=true);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  for (var ridx0 = 0; ridx0 < 15; ridx0++) {
    var precast14 = ridx0;
    var precast15 = (bitcast<u32>(precast14)<<24u);
    for (var ridx1 = 0; ridx1 < 3; ridx1++) {
      var precast16 = ridx1;
      var cast8 = bitcast<u32>(precast16);
      var precast17 = (cast8<<1u);
      var alu6 = (gidx1+bitcast<i32>(precast17));
      var alu7 = (((alu6<2)!=true)&(alu6<258));
      var alu8 = (alu7&((alu1<2)!=true));
      var alu9 = (alu7&(alu1<254));
      var precast18 = (cast8<<17u);
      var alu10 = (cast3+bitcast<i32>(precast18)+bitcast<i32>(precast15)+cast6+cast4+alu0);
      var val0 = select(0.0f, data1[(alu10+-131586)], (alu8&alu5));
      var val1 = select(0.0f, data1[(alu10+-131585)], (alu8&alu3));
      var val2 = select(0.0f, data1[(alu10+-131584)], alu8);
      var val3 = select(0.0f, data1[(alu10+-131583)], alu8);
      var val4 = select(0.0f, data1[(alu10+-131582)], alu8);
      var val5 = select(0.0f, data1[(alu10+-131581)], alu8);
      var val6 = select(0.0f, data1[(alu10+-131580)], (alu8&alu4));
      var val7 = select(0.0f, data1[(alu10+-131579)], (alu8&alu2));
      var val8 = select(0.0f, data1[(alu10+-131074)], (alu7&alu5));
      var val9 = select(0.0f, data1[(alu10+-131073)], (alu7&alu3));
      var val10 = select(0.0f, data1[(alu10+-131072)], alu7);
      var val11 = select(0.0f, data1[(alu10+-131071)], alu7);
      var val12 = select(0.0f, data1[(alu10+-131070)], alu7);
      var val13 = select(0.0f, data1[(alu10+-131069)], alu7);
      var val14 = select(0.0f, data1[(alu10+-131068)], (alu7&alu4));
      var val15 = select(0.0f, data1[(alu10+-131067)], (alu7&alu2));
      var val16 = select(0.0f, data1[(alu10+-130562)], (alu9&alu5));
      var val17 = select(0.0f, data1[(alu10+-130561)], (alu9&alu3));
      var val18 = select(0.0f, data1[(alu10+-130560)], alu9);
      var val19 = select(0.0f, data1[(alu10+-130559)], alu9);
      var val20 = select(0.0f, data1[(alu10+-130558)], alu9);
      var val21 = select(0.0f, data1[(alu10+-130557)], alu9);
      var val22 = select(0.0f, data1[(alu10+-130556)], (alu9&alu4));
      var val23 = select(0.0f, data1[(alu10+-130555)], (alu9&alu2));
      var alu11 = ((gidx2*1215)+(ridx0*27)+(ridx1*9));
      var val24 = data2[alu11];
      var val25 = data2[(alu11+1)];
      var val26 = data2[(alu11+2)];
      var val27 = data2[(alu11+3)];
      var val28 = data2[(alu11+4)];
      var val29 = data2[(alu11+5)];
      var val30 = data2[(alu11+6)];
      var val31 = data2[(alu11+7)];
      var val32 = data2[(alu11+8)];
      var val33 = data2[(alu11+405)];
      var val34 = data2[(alu11+406)];
      var val35 = data2[(alu11+407)];
      var val36 = data2[(alu11+408)];
      var val37 = data2[(alu11+409)];
      var val38 = data2[(alu11+410)];
      var val39 = data2[(alu11+411)];
      var val40 = data2[(alu11+412)];
      var val41 = data2[(alu11+413)];
      var val42 = data2[(alu11+810)];
      var val43 = data2[(alu11+811)];
      var val44 = data2[(alu11+812)];
      var val45 = data2[(alu11+813)];
      var val46 = data2[(alu11+814)];
      var val47 = data2[(alu11+815)];
      var val48 = data2[(alu11+816)];
      var val49 = data2[(alu11+817)];
      var val50 = data2[(alu11+818)];
      acc0 = (acc0+(val0*val24)+(val8*val27)+(val16*val30)+(val2*val25)+(val10*val28)+(val18*val31)+(val4*val26)+(val12*val29)+(val20*val32));
      acc1 = (acc1+(val0*val33)+(val8*val36)+(val16*val39)+(val2*val34)+(val10*val37)+(val18*val40)+(val4*val35)+(val12*val38)+(val20*val41));
      acc2 = (acc2+(val0*val42)+(val8*val45)+(val16*val48)+(val2*val43)+(val10*val46)+(val18*val49)+(val4*val44)+(val12*val47)+(val20*val50));
      acc3 = (acc3+(val1*val24)+(val9*val27)+(val17*val30)+(val3*val25)+(val11*val28)+(val19*val31)+(val5*val26)+(val13*val29)+(val21*val32));
      acc4 = (acc4+(val1*val33)+(val9*val36)+(val17*val39)+(val3*val34)+(val11*val37)+(val19*val40)+(val5*val35)+(val13*val38)+(val21*val41));
      acc5 = (acc5+(val1*val42)+(val9*val45)+(val17*val48)+(val3*val43)+(val11*val46)+(val19*val49)+(val5*val44)+(val13*val47)+(val21*val50));
      acc6 = (acc6+(val2*val24)+(val10*val27)+(val18*val30)+(val4*val25)+(val12*val28)+(val20*val31)+(val6*val26)+(val14*val29)+(val22*val32));
      acc7 = (acc7+(val2*val33)+(val10*val36)+(val18*val39)+(val4*val34)+(val12*val37)+(val20*val40)+(val6*val35)+(val14*val38)+(val22*val41));
      acc8 = (acc8+(val2*val42)+(val10*val45)+(val18*val48)+(val4*val43)+(val12*val46)+(val20*val49)+(val6*val44)+(val14*val47)+(val22*val50));
      acc9 = (acc9+(val3*val24)+(val11*val27)+(val19*val30)+(val5*val25)+(val13*val28)+(val21*val31)+(val7*val26)+(val15*val29)+(val23*val32));
      acc10 = (acc10+(val3*val33)+(val11*val36)+(val19*val39)+(val5*val34)+(val13*val37)+(val21*val40)+(val7*val35)+(val15*val38)+(val23*val41));
      acc11 = (acc11+(val3*val42)+(val11*val45)+(val19*val48)+(val5*val43)+(val13*val46)+(val21*val49)+(val7*val44)+(val15*val47)+(val23*val50));
    }
  }
  var alu26 = (cast3+(gidx2*50331648)+cast4+cast5+cast6+cast7);
  data0[alu26] = acc0;
  data0[(alu26+1)] = acc3;
  data0[(alu26+2)] = acc6;
  data0[(alu26+3)] = acc9;
  data0[(alu26+16777216)] = acc1;
  data0[(alu26+16777217)] = acc4;
  data0[(alu26+16777218)] = acc7;
  data0[(alu26+16777219)] = acc10;
  data0[(alu26+33554432)] = acc2;
  data0[(alu26+33554433)] = acc5;
  data0[(alu26+33554434)] = acc8;
  data0[(alu26+33554435)] = acc11;
}`;

const r_5_256_32_4_8_16_15_3_4_3_3_3n3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 256 */
  var gidx2 = i32(gindex.z); /* 5 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var precast0 = (gidx0>>2);
  var cast0 = bitcast<u32>(precast0);
  var precast1 = (gidx0&3);
  var cast1 = bitcast<u32>(precast1);
  var precast2 = gidx1;
  var precast3 = (bitcast<u32>(precast2)<<16u);
  var cast2 = bitcast<i32>(precast3);
  var precast4 = (cast0<<11u);
  var cast3 = bitcast<i32>(precast4);
  var precast5 = (cast1<<6u);
  var cast4 = bitcast<i32>(precast5);
  var precast6 = lidx0;
  var precast7 = (bitcast<u32>(precast6)<<8u);
  var cast5 = bitcast<i32>(precast7);
  var precast8 = lidx1;
  var precast9 = (bitcast<u32>(precast8)<<2u);
  var cast6 = bitcast<i32>(precast9);
  var alu0 = (cast6+cast4);
  var precast10 = (cast0<<3u);
  var alu1 = (lidx0+bitcast<i32>(precast10));
  var alu2 = ((alu0<1)!=true);
  var precast11 = (cast1<<4u);
  var alu3 = ((lidx1+bitcast<i32>(precast11))<63);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  for (var ridx0 = 0; ridx0 < 15; ridx0++) {
    var precast12 = ridx0;
    var precast13 = (bitcast<u32>(precast12)<<24u);
    for (var ridx1 = 0; ridx1 < 3; ridx1++) {
      var alu4 = (gidx1+ridx1);
      var alu5 = (((alu4<1)!=true)&(alu4<257));
      var alu6 = (alu5&((alu1<1)!=true));
      var alu7 = (alu5&(alu1<255));
      var precast14 = ridx1;
      var precast15 = (bitcast<u32>(precast14)<<16u);
      var alu8 = (cast2+bitcast<i32>(precast15)+bitcast<i32>(precast13)+cast5+cast3+alu0);
      var val0 = select(0.0f, data1[(alu8+-65793)], (alu6&alu2));
      var val1 = select(0.0f, data1[(alu8+-65792)], alu6);
      var val2 = select(0.0f, data1[(alu8+-65791)], alu6);
      var val3 = select(0.0f, data1[(alu8+-65790)], alu6);
      var val4 = select(0.0f, data1[(alu8+-65789)], alu6);
      var val5 = select(0.0f, data1[(alu8+-65788)], (alu6&alu3));
      var val6 = select(0.0f, data1[(alu8+-65537)], (alu5&alu2));
      var val7 = select(0.0f, data1[(alu8+-65536)], alu5);
      var val8 = select(0.0f, data1[(alu8+-65535)], alu5);
      var val9 = select(0.0f, data1[(alu8+-65534)], alu5);
      var val10 = select(0.0f, data1[(alu8+-65533)], alu5);
      var val11 = select(0.0f, data1[(alu8+-65532)], (alu5&alu3));
      var val12 = select(0.0f, data1[(alu8+-65281)], (alu7&alu2));
      var val13 = select(0.0f, data1[(alu8+-65280)], alu7);
      var val14 = select(0.0f, data1[(alu8+-65279)], alu7);
      var val15 = select(0.0f, data1[(alu8+-65278)], alu7);
      var val16 = select(0.0f, data1[(alu8+-65277)], alu7);
      var val17 = select(0.0f, data1[(alu8+-65276)], (alu7&alu3));
      var alu9 = ((gidx2*1215)+(ridx0*27)+(ridx1*9));
      var val18 = data2[alu9];
      var val19 = data2[(alu9+1)];
      var val20 = data2[(alu9+2)];
      var val21 = data2[(alu9+3)];
      var val22 = data2[(alu9+4)];
      var val23 = data2[(alu9+5)];
      var val24 = data2[(alu9+6)];
      var val25 = data2[(alu9+7)];
      var val26 = data2[(alu9+8)];
      var val27 = data2[(alu9+405)];
      var val28 = data2[(alu9+406)];
      var val29 = data2[(alu9+407)];
      var val30 = data2[(alu9+408)];
      var val31 = data2[(alu9+409)];
      var val32 = data2[(alu9+410)];
      var val33 = data2[(alu9+411)];
      var val34 = data2[(alu9+412)];
      var val35 = data2[(alu9+413)];
      var val36 = data2[(alu9+810)];
      var val37 = data2[(alu9+811)];
      var val38 = data2[(alu9+812)];
      var val39 = data2[(alu9+813)];
      var val40 = data2[(alu9+814)];
      var val41 = data2[(alu9+815)];
      var val42 = data2[(alu9+816)];
      var val43 = data2[(alu9+817)];
      var val44 = data2[(alu9+818)];
      acc0 = (acc0+(val0*val18)+(val6*val21)+(val12*val24)+(val1*val19)+(val7*val22)+(val13*val25)+(val2*val20)+(val8*val23)+(val14*val26));
      acc1 = (acc1+(val0*val27)+(val6*val30)+(val12*val33)+(val1*val28)+(val7*val31)+(val13*val34)+(val2*val29)+(val8*val32)+(val14*val35));
      acc2 = (acc2+(val0*val36)+(val6*val39)+(val12*val42)+(val1*val37)+(val7*val40)+(val13*val43)+(val2*val38)+(val8*val41)+(val14*val44));
      acc3 = (acc3+(val1*val18)+(val7*val21)+(val13*val24)+(val2*val19)+(val8*val22)+(val14*val25)+(val3*val20)+(val9*val23)+(val15*val26));
      acc4 = (acc4+(val1*val27)+(val7*val30)+(val13*val33)+(val2*val28)+(val8*val31)+(val14*val34)+(val3*val29)+(val9*val32)+(val15*val35));
      acc5 = (acc5+(val1*val36)+(val7*val39)+(val13*val42)+(val2*val37)+(val8*val40)+(val14*val43)+(val3*val38)+(val9*val41)+(val15*val44));
      acc6 = (acc6+(val2*val18)+(val8*val21)+(val14*val24)+(val3*val19)+(val9*val22)+(val15*val25)+(val4*val20)+(val10*val23)+(val16*val26));
      acc7 = (acc7+(val2*val27)+(val8*val30)+(val14*val33)+(val3*val28)+(val9*val31)+(val15*val34)+(val4*val29)+(val10*val32)+(val16*val35));
      acc8 = (acc8+(val2*val36)+(val8*val39)+(val14*val42)+(val3*val37)+(val9*val40)+(val15*val43)+(val4*val38)+(val10*val41)+(val16*val44));
      acc9 = (acc9+(val3*val18)+(val9*val21)+(val15*val24)+(val4*val19)+(val10*val22)+(val16*val25)+(val5*val20)+(val11*val23)+(val17*val26));
      acc10 = (acc10+(val3*val27)+(val9*val30)+(val15*val33)+(val4*val28)+(val10*val31)+(val16*val34)+(val5*val29)+(val11*val32)+(val17*val35));
      acc11 = (acc11+(val3*val36)+(val9*val39)+(val15*val42)+(val4*val37)+(val10*val40)+(val16*val43)+(val5*val38)+(val11*val41)+(val17*val44));
    }
  }
  var alu24 = (cast2+(gidx2*50331648)+cast3+cast4+cast5+cast6);
  data0[alu24] = acc0;
  data0[(alu24+1)] = acc3;
  data0[(alu24+2)] = acc6;
  data0[(alu24+3)] = acc9;
  data0[(alu24+16777216)] = acc1;
  data0[(alu24+16777217)] = acc4;
  data0[(alu24+16777218)] = acc7;
  data0[(alu24+16777219)] = acc10;
  data0[(alu24+33554432)] = acc2;
  data0[(alu24+33554433)] = acc5;
  data0[(alu24+33554434)] = acc8;
  data0[(alu24+33554435)] = acc11;
}`;

const r_5_256_32_4_8_16_15_3_4_3_3_3n4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 256 */
  var gidx2 = i32(gindex.z); /* 5 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var precast0 = (gidx0&3);
  var cast0 = bitcast<u32>(precast0);
  var precast1 = gidx1;
  var precast2 = (bitcast<u32>(precast1)<<16u);
  var cast1 = bitcast<i32>(precast2);
  var precast3 = (gidx0>>2);
  var precast4 = (bitcast<u32>(precast3)<<11u);
  var cast2 = bitcast<i32>(precast4);
  var precast5 = (cast0<<6u);
  var cast3 = bitcast<i32>(precast5);
  var precast6 = lidx0;
  var precast7 = (bitcast<u32>(precast6)<<8u);
  var cast4 = bitcast<i32>(precast7);
  var precast8 = lidx1;
  var precast9 = (bitcast<u32>(precast8)<<2u);
  var cast5 = bitcast<i32>(precast9);
  var precast10 = (cast0<<4u);
  var alu0 = (lidx1+bitcast<i32>(precast10));
  var alu1 = (alu0<60);
  var alu2 = ((alu0<4)!=true);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  for (var ridx0 = 0; ridx0 < 15; ridx0++) {
    var precast11 = ridx0;
    var precast12 = (bitcast<u32>(precast11)<<24u);
    for (var ridx1 = 0; ridx1 < 3; ridx1++) {
      var precast13 = ridx1;
      var cast6 = bitcast<u32>(precast13);
      var precast14 = (cast6<<4u);
      var alu3 = (gidx1+bitcast<i32>(precast14));
      var alu4 = (((alu3<16)!=true)&(alu3<272));
      var alu5 = (alu4&((gidx0<8)!=true));
      var alu6 = (alu4&(gidx0<120));
      var precast15 = (cast6<<20u);
      var alu7 = (alu4&alu1);
      var alu8 = (alu4&alu2);
      var alu9 = (alu6&alu1);
      var alu10 = (alu6&alu2);
      var alu11 = (alu5&alu1);
      var alu12 = (alu5&alu2);
      var alu13 = (cast1+bitcast<i32>(precast15)+bitcast<i32>(precast12)+cast4+cast2+cast5+cast3);
      var val0 = select(0.0f, data1[(alu13+-1052688)], alu12);
      var val1 = select(0.0f, data1[(alu13+-1052687)], alu12);
      var val2 = select(0.0f, data1[(alu13+-1052686)], alu12);
      var val3 = select(0.0f, data1[(alu13+-1052685)], alu12);
      var val4 = select(0.0f, data1[(alu13+-1052672)], alu5);
      var val5 = select(0.0f, data1[(alu13+-1052671)], alu5);
      var val6 = select(0.0f, data1[(alu13+-1052670)], alu5);
      var val7 = select(0.0f, data1[(alu13+-1052669)], alu5);
      var val8 = select(0.0f, data1[(alu13+-1052656)], alu11);
      var val9 = select(0.0f, data1[(alu13+-1052655)], alu11);
      var val10 = select(0.0f, data1[(alu13+-1052654)], alu11);
      var val11 = select(0.0f, data1[(alu13+-1052653)], alu11);
      var val12 = select(0.0f, data1[(alu13+-1048592)], alu8);
      var val13 = select(0.0f, data1[(alu13+-1048591)], alu8);
      var val14 = select(0.0f, data1[(alu13+-1048590)], alu8);
      var val15 = select(0.0f, data1[(alu13+-1048589)], alu8);
      var val16 = select(0.0f, data1[(alu13+-1048576)], alu4);
      var val17 = select(0.0f, data1[(alu13+-1048575)], alu4);
      var val18 = select(0.0f, data1[(alu13+-1048574)], alu4);
      var val19 = select(0.0f, data1[(alu13+-1048573)], alu4);
      var val20 = select(0.0f, data1[(alu13+-1048560)], alu7);
      var val21 = select(0.0f, data1[(alu13+-1048559)], alu7);
      var val22 = select(0.0f, data1[(alu13+-1048558)], alu7);
      var val23 = select(0.0f, data1[(alu13+-1048557)], alu7);
      var val24 = select(0.0f, data1[(alu13+-1044496)], alu10);
      var val25 = select(0.0f, data1[(alu13+-1044495)], alu10);
      var val26 = select(0.0f, data1[(alu13+-1044494)], alu10);
      var val27 = select(0.0f, data1[(alu13+-1044493)], alu10);
      var val28 = select(0.0f, data1[(alu13+-1044480)], alu6);
      var val29 = select(0.0f, data1[(alu13+-1044479)], alu6);
      var val30 = select(0.0f, data1[(alu13+-1044478)], alu6);
      var val31 = select(0.0f, data1[(alu13+-1044477)], alu6);
      var val32 = select(0.0f, data1[(alu13+-1044464)], alu9);
      var val33 = select(0.0f, data1[(alu13+-1044463)], alu9);
      var val34 = select(0.0f, data1[(alu13+-1044462)], alu9);
      var val35 = select(0.0f, data1[(alu13+-1044461)], alu9);
      var alu14 = ((gidx2*1215)+(ridx0*27)+(ridx1*9));
      var val36 = data2[alu14];
      var val37 = data2[(alu14+1)];
      var val38 = data2[(alu14+2)];
      var val39 = data2[(alu14+3)];
      var val40 = data2[(alu14+4)];
      var val41 = data2[(alu14+5)];
      var val42 = data2[(alu14+6)];
      var val43 = data2[(alu14+7)];
      var val44 = data2[(alu14+8)];
      var val45 = data2[(alu14+405)];
      var val46 = data2[(alu14+406)];
      var val47 = data2[(alu14+407)];
      var val48 = data2[(alu14+408)];
      var val49 = data2[(alu14+409)];
      var val50 = data2[(alu14+410)];
      var val51 = data2[(alu14+411)];
      var val52 = data2[(alu14+412)];
      var val53 = data2[(alu14+413)];
      var val54 = data2[(alu14+810)];
      var val55 = data2[(alu14+811)];
      var val56 = data2[(alu14+812)];
      var val57 = data2[(alu14+813)];
      var val58 = data2[(alu14+814)];
      var val59 = data2[(alu14+815)];
      var val60 = data2[(alu14+816)];
      var val61 = data2[(alu14+817)];
      var val62 = data2[(alu14+818)];
      acc0 = (acc0+(val0*val36)+(val12*val39)+(val24*val42)+(val4*val37)+(val16*val40)+(val28*val43)+(val8*val38)+(val20*val41)+(val32*val44));
      acc1 = (acc1+(val0*val45)+(val12*val48)+(val24*val51)+(val4*val46)+(val16*val49)+(val28*val52)+(val8*val47)+(val20*val50)+(val32*val53));
      acc2 = (acc2+(val0*val54)+(val12*val57)+(val24*val60)+(val4*val55)+(val16*val58)+(val28*val61)+(val8*val56)+(val20*val59)+(val32*val62));
      acc3 = (acc3+(val1*val36)+(val13*val39)+(val25*val42)+(val5*val37)+(val17*val40)+(val29*val43)+(val9*val38)+(val21*val41)+(val33*val44));
      acc4 = (acc4+(val1*val45)+(val13*val48)+(val25*val51)+(val5*val46)+(val17*val49)+(val29*val52)+(val9*val47)+(val21*val50)+(val33*val53));
      acc5 = (acc5+(val1*val54)+(val13*val57)+(val25*val60)+(val5*val55)+(val17*val58)+(val29*val61)+(val9*val56)+(val21*val59)+(val33*val62));
      acc6 = (acc6+(val2*val36)+(val14*val39)+(val26*val42)+(val6*val37)+(val18*val40)+(val30*val43)+(val10*val38)+(val22*val41)+(val34*val44));
      acc7 = (acc7+(val2*val45)+(val14*val48)+(val26*val51)+(val6*val46)+(val18*val49)+(val30*val52)+(val10*val47)+(val22*val50)+(val34*val53));
      acc8 = (acc8+(val2*val54)+(val14*val57)+(val26*val60)+(val6*val55)+(val18*val58)+(val30*val61)+(val10*val56)+(val22*val59)+(val34*val62));
      acc9 = (acc9+(val3*val36)+(val15*val39)+(val27*val42)+(val7*val37)+(val19*val40)+(val31*val43)+(val11*val38)+(val23*val41)+(val35*val44));
      acc10 = (acc10+(val3*val45)+(val15*val48)+(val27*val51)+(val7*val46)+(val19*val49)+(val31*val52)+(val11*val47)+(val23*val50)+(val35*val53));
      acc11 = (acc11+(val3*val54)+(val15*val57)+(val27*val60)+(val7*val55)+(val19*val58)+(val31*val61)+(val11*val56)+(val23*val59)+(val35*val62));
    }
  }
  var alu29 = (cast1+(gidx2*50331648)+cast2+cast3+cast4+cast5);
  data0[alu29] = acc0;
  data0[(alu29+1)] = acc3;
  data0[(alu29+2)] = acc6;
  data0[(alu29+3)] = acc9;
  data0[(alu29+16777216)] = acc1;
  data0[(alu29+16777217)] = acc4;
  data0[(alu29+16777218)] = acc7;
  data0[(alu29+16777219)] = acc10;
  data0[(alu29+33554432)] = acc2;
  data0[(alu29+33554433)] = acc5;
  data0[(alu29+33554434)] = acc8;
  data0[(alu29+33554435)] = acc11;
}`;

const r_262144_2_16_4_15 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(2,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32768 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var precast0 = gidx0;
  var precast1 = gidx1;
  var precast2 = lidx1;
  var alu0 = (lidx0*15);
  var val0 = data2[alu0];
  var val1 = data2[(alu0+1)];
  var val2 = data2[(alu0+2)];
  var val3 = data2[(alu0+3)];
  var val4 = data2[(alu0+4)];
  var val5 = data2[(alu0+5)];
  var val6 = data2[(alu0+6)];
  var val7 = data2[(alu0+7)];
  var val8 = data2[(alu0+8)];
  var val9 = data2[(alu0+9)];
  var val10 = data2[(alu0+10)];
  var val11 = data2[(alu0+11)];
  var val12 = data2[(alu0+12)];
  var val13 = data2[(alu0+13)];
  var val14 = data2[(alu0+14)];
  var precast3 = (bitcast<u32>(precast0)<<9u);
  var precast4 = (bitcast<u32>(precast1)<<6u);
  var alu1 = (bitcast<i32>(precast3)+bitcast<i32>(precast4));
  var precast5 = (bitcast<u32>(precast2)<<2u);
  var cast0 = bitcast<i32>(precast5);
  var alu2 = (alu1+cast0);
  var val15 = data1[alu2];
  var val16 = data1[(alu2+1)];
  var val17 = data1[(alu2+2)];
  var val18 = data1[(alu2+3)];
  var val19 = data1[(alu2+16777216)];
  var val20 = data1[(alu2+16777217)];
  var val21 = data1[(alu2+16777218)];
  var val22 = data1[(alu2+16777219)];
  var val23 = data1[(alu2+33554432)];
  var val24 = data1[(alu2+33554433)];
  var val25 = data1[(alu2+33554434)];
  var val26 = data1[(alu2+33554435)];
  var val27 = data1[(alu2+50331648)];
  var val28 = data1[(alu2+50331649)];
  var val29 = data1[(alu2+50331650)];
  var val30 = data1[(alu2+50331651)];
  var val31 = data1[(alu2+67108864)];
  var val32 = data1[(alu2+67108865)];
  var val33 = data1[(alu2+67108866)];
  var val34 = data1[(alu2+67108867)];
  var val35 = data1[(alu2+83886080)];
  var val36 = data1[(alu2+83886081)];
  var val37 = data1[(alu2+83886082)];
  var val38 = data1[(alu2+83886083)];
  var val39 = data1[(alu2+100663296)];
  var val40 = data1[(alu2+100663297)];
  var val41 = data1[(alu2+100663298)];
  var val42 = data1[(alu2+100663299)];
  var val43 = data1[(alu2+117440512)];
  var val44 = data1[(alu2+117440513)];
  var val45 = data1[(alu2+117440514)];
  var val46 = data1[(alu2+117440515)];
  var val47 = data1[(alu2+134217728)];
  var val48 = data1[(alu2+134217729)];
  var val49 = data1[(alu2+134217730)];
  var val50 = data1[(alu2+134217731)];
  var val51 = data1[(alu2+150994944)];
  var val52 = data1[(alu2+150994945)];
  var val53 = data1[(alu2+150994946)];
  var val54 = data1[(alu2+150994947)];
  var val55 = data1[(alu2+167772160)];
  var val56 = data1[(alu2+167772161)];
  var val57 = data1[(alu2+167772162)];
  var val58 = data1[(alu2+167772163)];
  var val59 = data1[(alu2+184549376)];
  var val60 = data1[(alu2+184549377)];
  var val61 = data1[(alu2+184549378)];
  var val62 = data1[(alu2+184549379)];
  var val63 = data1[(alu2+201326592)];
  var val64 = data1[(alu2+201326593)];
  var val65 = data1[(alu2+201326594)];
  var val66 = data1[(alu2+201326595)];
  var val67 = data1[(alu2+218103808)];
  var val68 = data1[(alu2+218103809)];
  var val69 = data1[(alu2+218103810)];
  var val70 = data1[(alu2+218103811)];
  var val71 = data1[(alu2+234881024)];
  var val72 = data1[(alu2+234881025)];
  var val73 = data1[(alu2+234881026)];
  var val74 = data1[(alu2+234881027)];
  var precast6 = lidx0;
  var precast7 = (bitcast<u32>(precast6)<<24u);
  var alu3 = (alu1+bitcast<i32>(precast7)+cast0);
  data0[alu3] = ((val15*val0)+(val19*val1)+(val23*val2)+(val27*val3)+(val31*val4)+(val35*val5)+(val39*val6)+(val43*val7)+(val47*val8)+(val51*val9)+(val55*val10)+(val59*val11)+(val63*val12)+(val67*val13)+(val71*val14));
  data0[(alu3+1)] = ((val16*val0)+(val20*val1)+(val24*val2)+(val28*val3)+(val32*val4)+(val36*val5)+(val40*val6)+(val44*val7)+(val48*val8)+(val52*val9)+(val56*val10)+(val60*val11)+(val64*val12)+(val68*val13)+(val72*val14));
  data0[(alu3+2)] = ((val17*val0)+(val21*val1)+(val25*val2)+(val29*val3)+(val33*val4)+(val37*val5)+(val41*val6)+(val45*val7)+(val49*val8)+(val53*val9)+(val57*val10)+(val61*val11)+(val65*val12)+(val69*val13)+(val73*val14));
  data0[(alu3+3)] = ((val18*val0)+(val22*val1)+(val26*val2)+(val30*val3)+(val34*val4)+(val38*val5)+(val42*val6)+(val46*val7)+(val50*val8)+(val54*val9)+(val58*val10)+(val62*val11)+(val66*val12)+(val70*val13)+(val74*val14));
}`;

const setupNet = async (device, safetensor) => {
    const metadata = getTensorMetadata(safetensor);
    const infinityBuf = createInfinityUniformBuf(device);

    const layouts=[device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]})]

    const buf_0 = createEmptyBuf(device, 1006632960);;
    const input0 = createEmptyBuf(device, 67108864);;
    const buf_1 = createWeightBuf(device, 1620, getTensorBuffer(safetensor, metadata['model.0.weight']));
    const buf_2 = createEmptyBuf(device, 15360);;
    const buf_3 = createEmptyBuf(device, 60);;
    const buf_4 = createEmptyBuf(device, 60);;
    const buf_5 = createEmptyBuf(device, 1006632960);;
    const buf_6 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.3.weight']));
    const buf_7 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.6.weight']));
    const buf_8 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.9.weight']));
    const buf_9 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.12.weight']));
    const buf_10 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.15.weight']));
    const buf_11 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.18.weight']));
    const buf_12 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.21.weight']));
    const buf_13 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.24.weight']));
    const buf_14 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.27.weight']));
    const buf_15 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.30.weight']));
    const buf_16 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.33.weight']));
    const buf_17 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.36.weight']));
    const buf_18 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.39.weight']));
    const buf_19 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.42.weight']));
    const buf_20 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.45.weight']));
    const buf_21 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.48.weight']));
    const buf_22 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.51.weight']));
    const buf_23 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.54.weight']));
    const buf_24 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.57.weight']));
    const buf_25 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.60.weight']));
    const buf_26 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.63.weight']));
    const buf_27 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.66.weight']));
    const buf_28 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.69.weight']));
    const buf_29 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.72.weight']));
    const output0 = createEmptyBuf(device, 134217728);;
    const buf_30 = createWeightBuf(device, 120, getTensorBuffer(safetensor, metadata['model.75.weight']));

    const gpuWriteBuffer0 = device.createBuffer({size:input0.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });

    const gpuReadBuffer0 = device.createBuffer({size:output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const kernels = [r_5_256_32_4_8_16_3_4_3_3_3, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n1, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n2, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n3, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n4, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n1, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n2, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n3, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n4, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n1, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n2, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n3, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n4, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n1, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n2, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n3, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n4, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n1, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n2, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_5_256_32_4_8_16_15_3_4_3_3_3n3, r_120_32_16384_4, r_15_256, r_5_4_3_16_16384_4_4, r_15_256n1, E_5_262144_3_16_4, r_262144_2_16_4_15];
    const pipelines = await Promise.all(kernels.map(async (name, i) => {
      return await device.createComputePipelineAsync({
          layout: device.createPipelineLayout({
              bindGroupLayouts: [layouts[i]],
          }),
          compute: {
              module: device.createShaderModule({
                  code: name,
              }),
              entryPoint: "main",
          },
      });
  }))

    return async (_input0) => {
        const commandEncoder = device.createCommandEncoder();
        await gpuWriteBuffer0.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer0.getMappedRange()).set(_input0);
        gpuWriteBuffer0.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer0, 0, input0, 0, gpuWriteBuffer0.size);
        addComputePass(device, commandEncoder, pipelines[0], layouts[0], infinityBuf, [buf_0, input0, buf_1], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[1], layouts[1], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[2], layouts[2], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[3], layouts[3], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[4], layouts[4], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[5], layouts[5], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[6], layouts[6], infinityBuf, [buf_0, buf_5, buf_6], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[7], layouts[7], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[8], layouts[8], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[9], layouts[9], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[10], layouts[10], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[11], layouts[11], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[12], layouts[12], infinityBuf, [buf_0, buf_5, buf_7], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[13], layouts[13], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[14], layouts[14], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[15], layouts[15], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[16], layouts[16], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[17], layouts[17], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[18], layouts[18], infinityBuf, [buf_0, buf_5, buf_8], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[19], layouts[19], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[20], layouts[20], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[21], layouts[21], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[22], layouts[22], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[23], layouts[23], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[24], layouts[24], infinityBuf, [buf_0, buf_5, buf_9], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[25], layouts[25], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[26], layouts[26], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[27], layouts[27], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[28], layouts[28], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[29], layouts[29], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[30], layouts[30], infinityBuf, [buf_0, buf_5, buf_10], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[31], layouts[31], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[32], layouts[32], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[33], layouts[33], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[34], layouts[34], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[35], layouts[35], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[36], layouts[36], infinityBuf, [buf_0, buf_5, buf_11], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[37], layouts[37], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[38], layouts[38], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[39], layouts[39], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[40], layouts[40], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[41], layouts[41], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[42], layouts[42], infinityBuf, [buf_0, buf_5, buf_12], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[43], layouts[43], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[44], layouts[44], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[45], layouts[45], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[46], layouts[46], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[47], layouts[47], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[48], layouts[48], infinityBuf, [buf_0, buf_5, buf_13], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[49], layouts[49], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[50], layouts[50], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[51], layouts[51], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[52], layouts[52], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[53], layouts[53], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[54], layouts[54], infinityBuf, [buf_0, buf_5, buf_14], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[55], layouts[55], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[56], layouts[56], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[57], layouts[57], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[58], layouts[58], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[59], layouts[59], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[60], layouts[60], infinityBuf, [buf_0, buf_5, buf_15], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[61], layouts[61], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[62], layouts[62], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[63], layouts[63], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[64], layouts[64], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[65], layouts[65], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[66], layouts[66], infinityBuf, [buf_0, buf_5, buf_16], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[67], layouts[67], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[68], layouts[68], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[69], layouts[69], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[70], layouts[70], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[71], layouts[71], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[72], layouts[72], infinityBuf, [buf_0, buf_5, buf_17], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[73], layouts[73], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[74], layouts[74], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[75], layouts[75], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[76], layouts[76], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[77], layouts[77], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[78], layouts[78], infinityBuf, [buf_0, buf_5, buf_18], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[79], layouts[79], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[80], layouts[80], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[81], layouts[81], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[82], layouts[82], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[83], layouts[83], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[84], layouts[84], infinityBuf, [buf_0, buf_5, buf_19], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[85], layouts[85], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[86], layouts[86], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[87], layouts[87], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[88], layouts[88], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[89], layouts[89], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[90], layouts[90], infinityBuf, [buf_0, buf_5, buf_20], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[91], layouts[91], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[92], layouts[92], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[93], layouts[93], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[94], layouts[94], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[95], layouts[95], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[96], layouts[96], infinityBuf, [buf_0, buf_5, buf_21], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[97], layouts[97], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[98], layouts[98], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[99], layouts[99], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[100], layouts[100], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[101], layouts[101], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[102], layouts[102], infinityBuf, [buf_0, buf_5, buf_22], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[103], layouts[103], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[104], layouts[104], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[105], layouts[105], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[106], layouts[106], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[107], layouts[107], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[108], layouts[108], infinityBuf, [buf_0, buf_5, buf_23], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[109], layouts[109], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[110], layouts[110], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[111], layouts[111], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[112], layouts[112], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[113], layouts[113], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[114], layouts[114], infinityBuf, [buf_0, buf_5, buf_24], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[115], layouts[115], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[116], layouts[116], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[117], layouts[117], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[118], layouts[118], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[119], layouts[119], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[120], layouts[120], infinityBuf, [buf_0, buf_5, buf_25], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[121], layouts[121], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[122], layouts[122], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[123], layouts[123], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[124], layouts[124], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[125], layouts[125], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[126], layouts[126], infinityBuf, [buf_0, buf_5, buf_26], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[127], layouts[127], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[128], layouts[128], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[129], layouts[129], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[130], layouts[130], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[131], layouts[131], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[132], layouts[132], infinityBuf, [buf_0, buf_5, buf_27], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[133], layouts[133], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[134], layouts[134], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[135], layouts[135], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[136], layouts[136], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[137], layouts[137], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[138], layouts[138], infinityBuf, [buf_0, buf_5, buf_28], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[139], layouts[139], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[140], layouts[140], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[141], layouts[141], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[142], layouts[142], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[143], layouts[143], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[144], layouts[144], infinityBuf, [buf_0, buf_5, buf_29], [128, 256, 5]);
        addComputePass(device, commandEncoder, pipelines[145], layouts[145], infinityBuf, [buf_2, buf_0], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[146], layouts[146], infinityBuf, [buf_3, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[147], layouts[147], infinityBuf, [buf_2, buf_0, buf_3], [4, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[148], layouts[148], infinityBuf, [buf_4, buf_2], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[149], layouts[149], infinityBuf, [buf_5, buf_0, buf_3, buf_4], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[150], layouts[150], infinityBuf, [output0, buf_5, buf_30], [32768, 8, 1]);
        commandEncoder.copyBufferToBuffer(output0, 0, gpuReadBuffer0, 0, output0.size);
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        await gpuReadBuffer0.mapAsync(GPUMapMode.READ);
        const resultBuffer0 = new Float32Array(gpuReadBuffer0.size/4);
        resultBuffer0.set(new Float32Array(gpuReadBuffer0.getMappedRange()));
        gpuReadBuffer0.unmap();
        return [resultBuffer0];
    }
}
const load = async (device, weight_path) => { return await fetch(weight_path).then(x => x.arrayBuffer()).then(x => setupNet(device, new Uint8Array(x))); }
return { load, setupNet };
})();
export default mindgrab;
