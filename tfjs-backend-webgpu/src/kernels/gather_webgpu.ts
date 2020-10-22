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


import {computeDispatch, flatDispatchLayout} from '../webgpu_util';
import {WebGPUProgram} from './webgpu_program';
import {getCoordsDataType} from '../shader_preprocessor';

export class GatherProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames: string[] = ['A', 'indices'];
  workPerThread = 1;
  workGroupSize: [number, number, number] = [128, 1, 1];
  rank: number;

  constructor(aShape: number[], indicesLength: number, axis: number) {
    this.outputShape = aShape.slice();
    this.outputShape[axis] = indicesLength;
    this.outputShape = this.outputShape;
    this.rank = this.outputShape.length;
    const dtype = getCoordsDataType(this.rank);
    const sourceCoords = getSourceCoords(aShape, axis);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);

    this.userCode = `
      void main() {
        int index = int(gl_GlobalInvocationID.x);
        ${dtype} resRC = getOutputCoords();
        setOutput(index, getA(${sourceCoords}));
      }
    `;
  }
}

function getSourceCoords(aShape: number[], axis: number): string {
  const rank = aShape.length;
  if (rank > 4) {
    throw Error(`Gather for rank ${rank} is not yet supported`);
  }
  if (rank === 1) {
    return `int(getIndices(resRC))`;
  }

  const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];

  const sourceCoords = [];
  for (let i = 0; i < aShape.length; i++) {
    if (i === axis) {
      sourceCoords.push(`int(getIndices(${currentCoords[i]}))`);
    } else {
      sourceCoords.push(`${currentCoords[i]}`);
    }
  }
  return sourceCoords.join();
}
