# Import needed libraries
from qs_kpa import KeyPointAnalysis

# Create a KeyPointAnalysis model
# Set from_pretrained=True in order to download the pretrained model
encoder = KeyPointAnalysis(from_pretrained=True) 

# Model configuration
print(encoder)

# Preparing data (a tuple of (topic, statement, stance) or a list of tuples)
inputs = [
    (
        "Assisted suicide should be a criminal offence",
        "a cure or treatment may be discovered shortly after having ended someone's life unnecessarily.",
        1,
    ),
    (
        "Assisted suicide should be a criminal offence",
        "Assisted suicide should not be allowed because many times people can still get better",
        1,
    ),
    ("Assisted suicide should be a criminal offence", "Assisted suicide is akin to killing someone", 1),
]

# Go and embedd everything
output = encoder.encode(inputs, convert_to_numpy=True)