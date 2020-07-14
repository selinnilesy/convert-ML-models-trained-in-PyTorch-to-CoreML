import torch
import sys
import numpy as np
from model import Net
import coremltools as ct

model_in = sys.argv[1]
label_count = sys.argv[2]

model = Net(output_label_count=int(label_count))
model.load_state_dict(torch.load(model_in))

model.cpu()     # convert model to cpu
model.eval()    # switch to eval mode

random_input = torch.rand(1, 1, 98, 40)
traced_model = torch.jit.trace(model, random_input)

print("converting pymodl to coreml model")
converted_model = ct.convert(
    traced_model,       # convert using Unified Conversion API
    inputs=[ct.TensorType(shape=random_input.shape)]
)
print("convertion is completed saving to disk f{}")

converted_model.save(model_in.replace(".pymodel", "")+".mlmodel")
