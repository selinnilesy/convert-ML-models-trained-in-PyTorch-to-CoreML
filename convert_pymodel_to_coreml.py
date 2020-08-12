import torch
import sys
import numpy as np
from model import Net
import coremltools as ct
from coremltools.models.neural_network import quantization_utils

model_in = sys.argv[1]
label_count = sys.argv[2]

model = Net(output_label_count=int(label_count))  
model.load_state_dict(torch.load(model_in))       

model.cpu()     # convert model to cpu
model.eval()    # switch to eval mode       
   
random_input = torch.rand(1, 1, 98, 40)
traced_model = torch.jit.trace(model, random_input, check_trace=False)

print("converting pymodl to coreml model")
converted_model = ct.convert(
    traced_model,       # convert using Unified Conversion API
    inputs=[ct.TensorType(shape=random_input.shape)]
)
print("convertion is completed saving to disk f{}")

# allowed values of nbits = 16, 8, 7, 6, ...., 1
quantized_model = quantization_utils.quantize_weights(converted_model, 8)
converted_model.save(model_in.replace(".pymodel", "")+".mlmodel")
quantized_model.save(model_in.replace(".pymodel", "_quantized")+".mlmodel")
