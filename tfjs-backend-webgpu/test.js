let list = [
  [64, 1024],
  //[64, 2048],
  //[64, 2048],
  //[64, 512],
  //[64, 17],
  //[64, 32],
  //[64, 34],
  //[64, 512],
  //[225, 512],
  //[225, 1024],
  //[225, 256],
  //[225, 1024],
  //[225, 256],
  //[841, 256],
  //[841, 512],
  //[841, 128],
  //[841, 512],
  //[841, 128],
  //[3249, 64],
  //[3249, 256],
  //[3249, 64],
];

function getGcd(a, b) {
    let max = Math.max(a, b);
    let min = Math.min(a, b);
    if (max % min === 0) {
        return min;
    } else {
        return getGcd(max % min, min);
    }
}

function getLcm(a, b) {
    return a * b / getGcd(a, b);
}

for (let i in list) {
  let workPerThread = {
    x: 4,
    y: 4
  };
  let workGroupSize = {
    x: 16,
    y: 16 
  };

  let tileInner = 16;

  let j = 0;
  let threadOccu = [];
  let memoryOccu = [];
  let optArray = [];
  for(workPerThread.x = 1; workPerThread.x <= 4; workPerThread.x++) {
    for(workPerThread.y = 1; workPerThread.y <= 4; workPerThread.y++) {
      for(workGroupSize.x = 1; workGroupSize.x <= 16; workGroupSize.x++) {
        for(workGroupSize.y = 1; workGroupSize.y <= 16; workGroupSize.y++) {
          for(tileInner = getLcm(workGroupSize.x, workGroupSize.y);
              tileInner <= 4 * getLcm(workGroupSize.x, workGroupSize.y); 
              tileInner += getLcm(workGroupSize.x, workGroupSize.y)) {
            let physicalThreads = 56;
            let SLM = 32;
            let tileAOuter = workPerThread.y * workGroupSize.y;
            let tileBOuter = workPerThread.x * workGroupSize.x;
  
            let groupUseMemory = 32 * (tileAOuter + tileBOuter) * tileInner / 1024 / 8;
            let memoryAvailableSet = Math.floor(SLM/groupUseMemory);
  
            let groupUseThreads = workGroupSize.x * workGroupSize.y / 16;
            let threadAvilableSet = Math.floor(physicalThreads / groupUseThreads);
  
            let dispatch = Math.ceil(list[i][0] * list[i][1] / workPerThread.x / workPerThread.y / workGroupSize.x / workGroupSize.y);
            let executionSet = Math.min(memoryAvailableSet, threadAvilableSet, dispatch);
            let threadOccupancy = executionSet * groupUseThreads / physicalThreads;
            let memoryOccupancy = executionSet * groupUseMemory / SLM;
            //console.log((threadOccupancy * 100).toFixed(2) + '%, ' + (memoryOccupancy * 100).toFixed(2) + '%');
            //threadOccu.push(threadOccupancy);
            //memoryOccu.push(memoryOccupancy);
            if (threadOccupancy <= 0.1 || memoryOccupancy <= 0.1) continue;
            threadOccu.push((threadOccupancy * 100).toFixed(2) + '%');
            memoryOccu.push((memoryOccupancy * 100).toFixed(2) + '%');
            optArray.push(`${workGroupSize.x}, ${workGroupSize.y}, ${workPerThread.x}, ${workPerThread.y}, ${tileInner}`);
            j++;
          }
        }
      }
    }
  }
  var fs = require('fs');
  var file = fs.createWriteStream('array.txt');
  file.on('error', function(err) { /* error handling */ });
  for (let k = 0; k<j; k++){
    file.write(`${threadOccu[k]}, ${memoryOccu[k]}, ${optArray[k]}`+ '\n');
    //console.log(`${threadOccu[k]}, ${memoryOccu[k]}, ${optArray[k]}`);
  }
  console.log(j);

  file.end();
  //console.log(Math.max(...threadOccu));
}
