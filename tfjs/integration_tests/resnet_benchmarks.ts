/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import * as tfc from '@tensorflow/tfjs-core';
//import * as tfl from '@tensorflow/tfjs-layers';
import * as tfconv from '@tensorflow/tfjs-converter';
//import * as tfwebgpu from '../../tfjs-backend-webgpu/src/index';
//import * as tfwebgpu from '@tensorflow/tfjs-backend-webgpu';
import * as tfwebgpu from '@tensorflow/tfjs-backend-webgpu';

import {BenchmarkModelTest} from './types';
import * as util from './util';

const RESNET_MODEL_PATH =
    // tslint:disable-next-line:max-line-length
    // 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json';
    'https://storage.googleapis.com/tfjs-models/savedmodel/posenet/resnet50/quant2/model-stride32.json';
    //'./model-stride32.json';


export class ResNet50GPUBenchmark implements BenchmarkModelTest {
  //private model: tfl.LayersModel;
  private graphModel: tfconv.GraphModel;

  async loadModel() {
    this.graphModel = await tfconv.loadGraphModel(RESNET_MODEL_PATH);
    //const resnet = new ResNet(this.graphModel, outputStride);
    // this.model = await tfl.loadLayersModel(RESNET_MODEL_PATH);
  }

  async run(size: number): Promise<number> {
    console.log(tfwebgpu.webgpu.WebGPUBackend);
    console.log(tfc.getBackend());
    await tfc.ready();
    tfc.setBackend('webgpu');
    await tfc.ready();
    console.log(tfc.getBackend());

    const zeros = tfc.zeros([1, 224, 224, 3]);

    const benchmark = () => this.graphModel.predict(zeros) as tfc.Tensor[];

    const time = await util.benchmark(benchmark);

    zeros.dispose();

    return time;
  }
}
