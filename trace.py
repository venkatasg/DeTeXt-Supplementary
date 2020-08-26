from torchvision.models import mobilenet_v2
import torch
import coremltools as ct
from torchvision import transforms

with open('class_names.txt') as f:
    class_labels = f.read().splitlines()

# Initialise model
mnet = mobilenet_v2(pretrained=False, progress=False,
                     num_classes=len(class_labels))
mnet.load_state_dict(torch.load('mobilenet.bin',
                      map_location=torch.device('cpu')))

# Add a softmax on top - why the fuck didn't I think of this
model = torch.nn.Sequential(mnet, torch.nn.Softmax(dim=1))
model.eval()

rand = torch.rand(1,3,200,300)
traced_model = torch.jit.trace(model, rand)

ctmodel = ct.convert(traced_model,
    inputs=[ct.ImageType(name="drawing", shape=rand.shape, bias=[-1,-1,-1],
                         scale=1/127)],
    classifier_config = ct.ClassifierConfig(class_labels))

spec = ctmodel.get_spec()
# Rename the output dictionary to something sensible
ct.utils.rename_feature(spec, '649', 'classLabelProbs')
ctmodel = ct.models.MLModel(spec)

# Set feature descriptions (these show up as comments in XCode)
ctmodel.input_description["drawing"] = "Input drawing to be classified"
ctmodel.output_description["classLabel"] = "Most likely symbol"
ctmodel.output_description["classLabelProbs"] = "Probability scores for each symbol"

# Set model author name
ctmodel.author = "Venkata S Govindarajan"

# Set the license of the model
ctmodel.license = "MIT License"

# Set a short description for the Xcode UI
ctmodel.short_description = "Detects the most likely LaTeX mathematical symbol \
                           corresponding to a drawing."

# Set a version for the model
ctmodel.version = "0.95"

# Save model
ctmodel.save("deTeX.mlmodel")