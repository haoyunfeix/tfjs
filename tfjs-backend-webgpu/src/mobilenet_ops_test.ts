/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs-core';
import {describeWebGPU} from './test_util';
import {env} from '@tensorflow/tfjs-core/dist/environment';

let data:any = [];
let RET:any = [];
describeWebGPU('Ops mobilenet', () => {
  jasmine.DEFAULT_TIMEOUT_INTERVAL = 200000;
  // Performs `trials` trials, of `reps` repetitions each. At the end of each
  // trial, endTrial() is run (and included in the benchmark time). This
  // allows the cost of endTrial() to be amortized across the many iterations.
  // This is needed in particular because WebGPU readbacks are asynchronous
  // and therefore always incur latency. (Plus, in Chrome right now, readbacks
  // are very inefficient, making the problem way worse.) Readbacks could be
  // avoided by using fences, but we don't have a common abstraction over
  // WebGL and WebGPU fences at the moment.
  async function time(
      doRep: (r: number) => tf.Tensor[] | tf.Tensor,
      endTrial?: () => Promise<void>, disposeAfterEachTrial = false,
      trials = 20, reps = 20, description:string = 'default') {
    const times = [];

    let toDispose: tf.Tensor[] = [];
    const dispose = () => {
      for (const t of toDispose) {
        t.dispose();
      }
      toDispose = [];
    };

    const trial = async () => {
      let result;
      for (let r = 0; r < reps; ++r) {
        result = doRep(r);

        toDispose = toDispose.concat(Array.isArray(result) ? result : [result]);
      }

      if (endTrial != null) {
        await endTrial();
      } else {
        await (Array.isArray(result) ? result[0] : result).data();
      }
    };

    // Warm-up. Specifically, this pre-allocates enough memory for an entire
    // trial, ensuring that no allocations happen when timing a trial (if the
    // backend reuses allocations).
    await trial();
    dispose();

    for (let t = 0; t < trials; ++t) {
      const start = tf.util.now();
      await trial();
      times.push(tf.util.now() - start);
      if (disposeAfterEachTrial) {
        dispose();
      }
    }

    const mean = times.reduce((a, b) => a + b, 0) / trials;
    const min = Math.min(...times);
    const fmt = (n: number) => n.toFixed(3);
    console.log(`${fmt(mean / reps)}`);
    RET.push(fmt(mean / reps));
    //console.log(`Mean time: ${fmt(mean)} ms -> ${fmt(mean / reps)} / rep`);
    //console.log(`Min time: ${fmt(min)} ms -> ${fmt(min / reps)} / rep`);
    let jsonData:any = {};
    jsonData['name'] = description;
    jsonData['mean'] = fmt(mean / reps);
    jsonData['min'] = fmt(min / reps);
    jsonData['numTrials'] = trials;
    jsonData['backend'] = env().getFlags()['WEBGPU_CPU_FORWARD'] !== undefined ? 'webgpu' : 'webgl';
    data.push(jsonData);
  }
  const nRep = 50;
  const nTrail = 10;

  function testPad(xShape: Array<number>, pShape: Array<[number, number]>) {
  xit(`pad xShape:${xShape} pShape:${pShape}`, async () => {
    const doTest = async (xShape: Array<number>, pShape: Array<[number, number]>) => {
      const arr = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const a = arr.map((x) => tf.randomNormal(xShape));
      const b:Array<[number, number]> = pShape;
      const c = 0;
      await time(
        (r) => {
          res[r] = tf.pad(a[r], b, c)
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep,`pad xShape:${xShape} pShape:${pShape}`);
      a.forEach(t => t.dispose());
    };
    await doTest(xShape, pShape);
  });
  }
    testPad([1, 15, 15, 256], [[0, 0], [1, 1], [1, 1], [0, 0]]);
    testPad([1, 29, 29, 128], [[0, 0], [1, 1], [1, 1], [0, 0]]);
    testPad([1, 57, 57, 64], [[0, 0], [1, 1], [1, 1], [0, 0]]);
    testPad([1, 113, 113, 64], [[0, 0], [1, 1], [1, 1], [0, 0]]);
    testPad([1, 225, 225, 3], [[0, 0], [3, 3], [3, 3], [0, 0]]);
    testPad([500, 600, 3], [[50, 50], [0, 0], [0, 0]]);
  
  function testConv2d(xShape: [number, number, number, number], fShape: [number, number, number, number],
	       stride: [number, number], pad: 'valid'|'same'|number, format: 'NHWC'|'NCHW') {
  it(`conv2d xShape:${xShape} fShape:${fShape} stride:${stride} pad:${pad} format:${format}`, async () => {
	  jasmine.DEFAULT_TIMEOUT_INTERVAL = 100000;
    const doTest = async (
      xShape: [number, number, number, number], fShape: [number, number, number, number],
      stride: [number, number], pad: 'valid'|'same'|number, format: 'NHWC'|'NCHW') => {
      const arrX = new Array(nRep).fill(0);
      const arrF = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const x = arrX.map((x) => tf.randomNormal<tf.Rank.R4>(xShape));
      const f = arrF.map((x) => tf.randomNormal<tf.Rank.R4>(fShape));
      await time(
        (r) => {
          res[r] = tf.conv2d(x[r], f[r], stride, pad, format);
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep, `conv2d xShape:${xShape} fShape:${fShape} stride:${stride} pad:${pad} format:${format}`);
      x.forEach(t => t.dispose());
      f.forEach(t => t.dispose());
    };
    await doTest(xShape,fShape,stride,pad,format);
  });
  }
  testConv2d([1,513,513,3],[3,3,3,24],[2,2],'same', 'NHWC');
  testConv2d([1,257,257,24],[1,1,24,48],[1,1],'same', 'NHWC');
  testConv2d([1,129,129,48],[1,1,48,96],[1,1],'same', 'NHWC');
  testConv2d([1,129,129,96],[1,1,96,96],[1,1],'same', 'NHWC');
  testConv2d([1,65,65,96],[1,1,96,192],[1,1],'same', 'NHWC');
  testConv2d([1,65,65,192],[1,1,192,192],[1,1],'same', 'NHWC');
  testConv2d([1,33,33,192],[1,1,192,384],[1,1],'same', 'NHWC');
  testConv2d([1,33,33,384],[1,1,384,384],[1,1],'same', 'NHWC');
  testConv2d([1,33,33,384],[1,1,384,32],[1,1],'same', 'NHWC');
  testConv2d([1,33,33,384],[1,1,384,17],[1,1],'same', 'NHWC');
  testConv2d([1,33,33,384],[1,1,384,34],[1,1],'same', 'NHWC');

  function testDepthConv2d(xShape: [number, number, number, number], fShape: [number, number, number, number],
	       stride: [number, number], pad: 'valid'|'same'|number, format: 'NHWC'|'NCHW') {
  fit(`depthConv2d xShape:${xShape} fShape:${fShape} stride:${stride} pad:${pad} format:${format}`, async () => {
	  jasmine.DEFAULT_TIMEOUT_INTERVAL = 100000;
    const doTest = async (
      xShape: [number, number, number, number], fShape: [number, number, number, number],
      stride: [number, number], pad: 'valid'|'same'|number, format: 'NHWC'|'NCHW') => {
      const arrX = new Array(nRep).fill(0);
      const arrF = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const x = arrX.map((x) => tf.randomNormal<tf.Rank.R4>(xShape));
      const f = arrF.map((x) => tf.randomNormal<tf.Rank.R4>(fShape));
      await time(
        (r) => {
          res[r] = tf.depthwiseConv2d(x[r], f[r], stride, pad, format);
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep, `conv2d xShape:${xShape} fShape:${fShape} stride:${stride} pad:${pad} format:${format}`);
      x.forEach(t => t.dispose());
      f.forEach(t => t.dispose());
    };
    await doTest(xShape,fShape,stride,pad,format);
  });
  }
  testDepthConv2d([1,257,257,24],[3,3,24,1],[1,1],'same', 'NHWC');
  testDepthConv2d([1,257,257,48],[3,3,48,1],[2,2],'same', 'NHWC');
  testDepthConv2d([1,129,129,96],[3,3,96,1],[1,1],'same', 'NHWC');
  testDepthConv2d([1,129,129,96],[3,3,96,1],[2,2],'same', 'NHWC');
  testDepthConv2d([1,65,65,192],[3,3,192,1],[1,1],'same', 'NHWC');
  testDepthConv2d([1,65,65,192],[3,3,192,1],[2,2],'same', 'NHWC');
  testDepthConv2d([1,33,33,384],[3,3,384,1],[1,1],'same', 'NHWC');

  function testClip(xShape: Array<number>) {
  it(`clip xShape:${xShape}`, async () => {
    const doTest = async (xShape: Array<number>) => {
      const arr = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const a = arr.map((x) => tf.randomNormal(xShape));
      await time(
        (r) => {
          res[r] = tf.clipByValue(a[r], 0, 6);
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep,`relu xShape:${xShape}`);
      a.forEach(t => t.dispose());
    };
    await doTest(xShape);
  });
  }
    testClip([1,257,257,24]);
    testClip([1,257,257,48]);
    testClip([1,129,129,24]);
    testClip([1,129,129,96]);
    testClip([1,65,65,96]);
    testClip([1,65,65,192]);
    testClip([1,33,33,192]);
    testClip([1,33,33,384]);

  function testRelu(xShape: Array<number>) {
  it(`relu xShape:${xShape}`, async () => {
    const doTest = async (xShape: Array<number>) => {
      const arr = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const a = arr.map((x) => tf.randomNormal(xShape));
      await time(
        (r) => {
          res[r] = tf.relu(a[r]);
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep,`relu xShape:${xShape}`);
      a.forEach(t => t.dispose());
    };
    await doTest(xShape);
  });
  }
    testRelu([1,15,15,128]);
    testRelu([1,15,15,256]);
    testRelu([1,15,15,512]);
    testRelu([1,15,15,1024]);
    testRelu([1,29,29,64]);
    testRelu([1,29,29,128]);
    testRelu([1,29,29,256]);
    testRelu([1,29,29,512]);
    testRelu([1,57,57,64]);
    testRelu([1,57,57,256]);
    testRelu([1,8,8,256]);
    testRelu([1,8,8,512]);
    testRelu([1,8,8,1024]);
    testRelu([1,8,8,2048]);
    testRelu([1,113,113,64]);


  function testAdd(aShape: Array<number>, bShape: Array<number>) {
  it(`add aShape:${aShape} bShape:${bShape}`, async () => {
    const doTest = async (aShape: Array<number>, bShape: Array<number>) => {
      const arrA = new Array(nRep).fill(0);
      const arrB = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const a = arrA.map((x) => tf.randomNormal(aShape));
      const b = arrB.map((x) => tf.randomNormal(bShape));
      await time(
        (r) => {
          res[r] = tf.add(a[r], b[r]);
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep, `add aShape:${aShape} bShape:${bShape}`);
      a.forEach(t => t.dispose());
      b.forEach(t => t.dispose());
    };
    await doTest(aShape, bShape);
  });
  }
    testAdd([1,257,257,24],[24]);
    testAdd([1,257,257,48],[48]);
    testAdd([1,129,129,48],[48]);
    testAdd([1,129,129,96],[96]);
    testAdd([1,65,65,96],[96]);
    testAdd([1,65,65,192],[192]);
    testAdd([1,33,33,192],[192]);
    testAdd([1,33,33,384],[384]);
    testAdd([1,33,33,32],[32]);
    testAdd([1,33,33,17],[17]);
    testAdd([1,33,33,34],[34]);
    testAdd([17,2],[17,2]);

  function testMaxPool(xShape: [number, number, number, number], filter: [number, number],
	     stride: [number, number], pad: 'valid'|'same'|number) {
    it(`maxPool xShape:${xShape} filter:${filter}`, async () => {
    const doTest = async (
      xShape: [number, number, number, number], filter: [number, number],
      stride: [number, number], pad: 'valid'|'same'|number) => {
      const arrX = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const x = arrX.map((x) => tf.randomNormal<tf.Rank.R4>(xShape));
      await time(
        (r) => {
          res[r] = tf.maxPool(x[r], filter, stride, pad);
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep, `maxPool xShape:${xShape} filter:${filter}`);
      x.forEach(t => t.dispose());
    };
    await doTest(xShape, filter, stride, pad);
  });
  }
    testMaxPool([1,15,15,1024], [1, 1], [2, 2], 'same');
    testMaxPool([1,29,29,512], [1, 1], [2, 2], 'same');
    testMaxPool([1,57,57,256], [1, 1], [2, 2], 'same');
    testMaxPool([1,115,115,64], [3, 3], [2, 2], 'valid');
  function download(content:any, fileName:any, contentType:any) {
    return new Promise(resolve => {
      const jsonData = JSON.stringify(content);
      const a = document.createElement("a");
      const file = new Blob([jsonData], {type: contentType});
      a.href = URL.createObjectURL(file);
      a.download = fileName;
      a.click();
      setTimeout(() => {
        resolve('done');
      }, 5000);
    });
  }
  afterAll(async function() {
    console.log(RET);
    if(env().getFlags()['WEBGPU_CPU_FORWARD'] !== undefined) {
      await download(data, 'json.json', 'application/json');
    }
  }, 20000)
});
