/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {KernelFunc, Sum, SumAttrs, SumInputs, sumOutType, TensorInfo} from '@tensorflow/tfjs-core';
import {backend_util, KernelConfig} from '@tensorflow/tfjs-core';
import {util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {reduce} from '../kernel_utils/reduce';
import {reshape} from './Reshape';

export function sum(
    args: {inputs: SumInputs, backend: WebGPUBackend, attrs: SumAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {axis} = attrs;
  const webgpuBackend = backend;
  let xShape = x.shape;
  const xRank = xShape.length;

  const origAxes = util.parseAxisParam(axis, xShape);
  let axes = origAxes;
  backend_util.assertAxesAreInnerMostDims('sum', axes, xRank);
  const [outShape, reduceShape] =
      backend_util.computeOutAndReduceShapes(xShape, axes);
  const reduceSize = util.sizeFromShape(reduceShape);
  const a2D = reshape({inputs: {x}, attrs: {shape: [-1, reduceSize]},
      backend: webgpuBackend});
  const outputDType = sumOutType(x.dtype);
  const a2DReduce = reduce(a2D, outputDType, 'sum', webgpuBackend);
  return reshape({inputs: {x: a2DReduce}, attrs: {shape: outShape},
      backend: webgpuBackend});
}

export const sumConfig: KernelConfig = {
  kernelName: Sum,
  backendName: 'webgpu',
  kernelFunc: sum as {} as KernelFunc
};
