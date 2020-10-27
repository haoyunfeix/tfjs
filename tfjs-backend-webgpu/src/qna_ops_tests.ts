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
let printable:string = '\n';
describeWebGPU('Ops qna', () => {
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
    printable += (fmt(mean / reps)) + '\n';
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
  const nTrail = 50;

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
    testAdd([1,1,384,384],[1,4,384,384]);
    testAdd([1,4,384],[1,4,384]);
    testAdd([1,4,384,384],[1,1,384,384]);
    testAdd([1,384,128],[1,384,128]);
    testAdd([1,384,128],[128]);
    testAdd([1,384,512],[1,384,512]);
    testAdd([128],[1,384,128]);
    testAdd([384,2],[2]);
    testAdd([384,128],[128]);
    testAdd([384,512],[512]);
    testAdd([512],[1,384,512]);

  function testAddN(aShape: Array<number>, bShape: Array<number>) {
  it(`add aShape:${aShape} bShape:${bShape}`, async () => {
    const doTest = async (aShape: Array<number>, bShape: Array<number>) => {
      const arrA = new Array(nRep).fill(0);
      const arrB = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const a = arrA.map((x) => tf.randomNormal(aShape));
      const b = arrB.map((x) => tf.randomNormal(bShape));
      await time(
        (r) => {
          res[r] = tf.addN([a[r], b[r]]);
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
    testAddN([1,384,128],[1,384,128]);

  function testMatMul(aShape: Array<number>, bShape: Array<number>,
    transposeA: boolean, transposeB: boolean) {
  it(`add aShape:${aShape} bShape:${bShape}`, async () => {
    const doTest = async (aShape: Array<number>, bShape: Array<number>) => {
      const arrA = new Array(nRep).fill(0);
      const arrB = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const a = arrA.map((x) => tf.randomNormal(aShape));
      const b = arrB.map((x) => tf.randomNormal(bShape));
      await time(
        (r) => {
          res[r] = tf.matMul(a[r], b[r],transposeA,transposeB);
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
    testMatMul([1,384,128],[1,128,128],false,false);
    testMatMul([1,384,128],[1,128,512],false,false);
    testMatMul([1,384,384],[1,384,512],false,false);
    testMatMul([1,384,512],[1,2,512],false,true);
    testMatMul([1,384,512],[1,512,128],false,false);
    testMatMul([4,384,32],[4,384,32],false,true);
    testMatMul([4,384,384],[4,384,32],false,false);
  
  function testExp(xShape: Array<number>) {
  it(`Exp xShape:${xShape}`, async () => {
    const doTest = async (xShape: Array<number>) => {
      const arr = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const a = arr.map((x) => tf.randomNormal(xShape));
      await time(
        (r) => {
          res[r] = tf.exp(a[r]);
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep,`Exp xShape:${xShape}`);
      a.forEach(t => t.dispose());
    };
    await doTest(xShape);
  });
  }
    testExp([1,4,384,384]);

  function testLog(xShape: Array<number>) {
  it(`Exp xShape:${xShape}`, async () => {
    const doTest = async (xShape: Array<number>) => {
      const arr = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const a = arr.map((x) => tf.randomNormal(xShape));
      await time(
        (r) => {
          res[r] = tf.log(a[r]);
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep,`Exp xShape:${xShape}`);
      a.forEach(t => t.dispose());
    };
    await doTest(xShape);
  });
  }
    testLog([1,4,384]);

function testMax(aShape: Array<number>, axis: number) {
  it(`Max aShape:${aShape}`, async () => {
    const doTest = async (aShape: Array<number>, axis: number) => {
      const arrA = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const a = arrA.map((x) => tf.randomNormal(aShape));
      await time(
        (r) => {
          res[r] = tf.max(a[r], axis);
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep, `Max aShape:${aShape}`);
      a.forEach(t => t.dispose());
    };
    await doTest(aShape, axis);
  });
  }
    testMax([1,4,384,384],3);

  function testMultiply(aShape: Array<number>, bShape: Array<number>) {
  it(`add aShape:${aShape} bShape:${bShape}`, async () => {
    const doTest = async (aShape: Array<number>, bShape: Array<number>) => {
      const arrA = new Array(nRep).fill(0);
      const arrB = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const a = arrA.map((x) => tf.randomNormal(aShape));
      const b = arrB.map((x) => tf.randomNormal(bShape));
      await time(
        (r) => {
          res[r] = tf.mul(a[r], b[r]);
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
    testMultiply([1,1,384],[1,384,1]);
    testMultiply([1,384,128],[128]);
    testMultiply([128],[1,384,128]);
    testMultiply([512],[1,384,512]);

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
    testRelu([1,384,512]);

  function testSubtract(aShape: Array<number>, bShape: Array<number>) {
  it(`add aShape:${aShape} bShape:${bShape}`, async () => {
    const doTest = async (aShape: Array<number>, bShape: Array<number>) => {
      const arrA = new Array(nRep).fill(0);
      const arrB = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const a = arrA.map((x) => tf.randomNormal(aShape));
      const b = arrB.map((x) => tf.randomNormal(bShape));
      await time(
        (r) => {
          res[r] = tf.sub(a[r], b[r]);
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
    testSubtract([1,4,384,384],[1,4,384,1]);
  
  function testSum(aShape: Array<number>, axis: number) {
  it(`sum aShape:${aShape}`, async () => {
    const doTest = async (aShape: Array<number>, axis: number) => {
      const arrA = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const a = arrA.map((x) => tf.randomNormal(aShape));
      await time(
        (r) => {
          res[r] = tf.sum(a[r], axis);
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep, `add aShape:${aShape}`);
      a.forEach(t => t.dispose());
    };
    await doTest(aShape, axis);
  });
  }
    testSum([1,4,384,384], 3);


  function testTranspose(aShape: Array<number>, perm: number[]) {
  it(`transpose aShape:${aShape}`, async () => {
    const doTest = async (aShape: Array<number>, perm: number[]) => {
      const arrA = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const a = arrA.map((x) => tf.randomNormal(aShape));
      await time(
        (r) => {
          res[r] = tf.transpose(a[r], perm);
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep, `add aShape:${aShape}`);
      a.forEach(t => t.dispose());
    };
    await doTest(aShape, perm);
  });
  }
    testTranspose([1,4,384,32],[0,2,1,3]);
    testTranspose([1,384,32],[2,0,1]);
    testTranspose([1,384,4,32],[0,2,1,3]);

  function testSoftMax(aShape: Array<number>, perm: number) {
  it(`transpose aShape:${aShape}`, async () => {
    const doTest = async (aShape: Array<number>, perm: number) => {
      const arrA = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const a = arrA.map((x) => tf.randomNormal(aShape));
      await time(
        (r) => {
          res[r] = tf.softmax(a[r], perm);
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep, `add aShape:${aShape}`);
      a.forEach(t => t.dispose());
    };
    await doTest(aShape, perm);
  });
  }
    testSoftMax([1,4,384,384],3);

  function testGather(aShape: Array<number>, bShape: Array<number>, perm: number) {
  fit(`transpose aShape:${aShape}`, async () => {
    const doTest = async (aShape: Array<number>, bShape: Array<number>, perm: number) => {
      const arrA = new Array(nRep).fill(0);
      const arrB = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const a = arrA.map((x) => tf.randomNormal(aShape));
      const b = arrB.map((x) => tf.randomNormal(bShape,null,null,'int32'));
      await time(
        (r) => {
          res[r] = tf.gather(a[r], b[r],perm);
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep, `add aShape:${aShape}`);
      a.forEach(t => t.dispose());
      b.forEach(t => t.dispose());
    };
    await doTest(aShape, bShape, perm);
  });
  }
    testGather([2,512],[384],0);
    testGather([30522,128],[384],0);

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
      console.log(printable);
    if(env().getFlags()['WEBGPU_CPU_FORWARD'] !== undefined) {
      await download(data, 'json.json', 'application/json');
    }
  }, 20000)
});
