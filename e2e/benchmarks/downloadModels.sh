if [ ! -d "savedmodel/posenet/resnet50/float" ]
then
  mkdir -p savedmodel/posenet/resnet50/float
  cd savedmodel/posenet/resnet50/float
  curl -O https://storage.googleapis.com/tfjs-models/savedmodel/posenet/resnet50/float/model-stride32.json
  for i in {1..23}
  do
    curl -O https://storage.googleapis.com/tfjs-models/savedmodel/posenet/resnet50/float/group1-shard"$i"of23.bin
  done
  cd -
fi

if [ ! -d "savemodel/posenet/mobilenet/quant2/075" ]
then
  mkdir -p savedmodel/posenet/mobilenet/quant2/075
  cd savedmodel/posenet/mobilenet/quant2/075
  curl -O https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/quant2/075/model-stride16.json
  curl -O https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/quant2/075/group1-shard1of1.bin
  cd -
fi
