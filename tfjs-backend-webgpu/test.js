let list = [
  [64, 1024],
  [64, 2048],
  [64, 2048],
  [64, 512],
  [64, 17],
  [64, 32],
  [64, 34],
  [64, 512],
  [225, 512],
  [225, 1024],
  [225, 256],
  [225, 1024],
  [225, 256],
  [841, 256],
  [841, 512],
  [841, 128],
  [841, 512],
  [841, 128],
  [3249, 64],
  [3249, 256],
  [3249, 64],
];

for (let i in list) {
let workPerThread = {
  x: 1,
  y: 1
};
let workGroupSize = {
  x: 8,
  y: 8 
};
let tileInner = 16;

let physicalThreads = 56;
let SLM = 48;
let tileAOuter = workPerThread.y * workGroupSize.y;
let tileBOuter = workPerThread.x * workGroupSize.x;

let groupUseMemory = 32 * (tileAOuter + tileBOuter) * tileInner / 1024 / 8;
let memoryAvailableSet = Math.floor(SLM/groupUseMemory);

let groupUseThreads = workGroupSize.x * workGroupSize.y / 16;
let threadAvilableSet = Math.floor(physicalThreads / groupUseThreads);

let dispatch = Math.ceil(list[i][0] * list[i][1] / workPerThread.x / workPerThread.y / workGroupSize.x / workGroupSize.y);
//console.log('hahah' + dispatch);
let executionSet = Math.min(memoryAvailableSet, threadAvilableSet, dispatch);
//console.log(memoryAvailableSet);
//console.log(threadAvilableSet);
//console.log(executionSet);
let threadOccupancy = executionSet * groupUseThreads / physicalThreads;
let memoryOccupancy = executionSet * groupUseMemory / SLM;
console.log((threadOccupancy * 100).toFixed(2) + '%, ' + (memoryOccupancy * 100).toFixed(2) + '%');
}
