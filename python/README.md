# Python Examples

This directory contains basic python examples to control your radio. Most of the required modules to run these examples are pre installed on the radio and installed with the provided AirStack conda environment; however, the detect_and_repeat.py should be run from the airstack-infer conda environment only because cupy is not pre installed.

Deepwave provides written tutorials [here](https://docs.deepwave.ai/Tutorials/) for a more detailed walk through of these examples and more.

## Provided Examples

* hello_world.py: A basic example of how to interact with the radio drivers, stream signal data from the radio to the Tegra, and create a one-time plot of the wireless spectrum

* transmit_tone.py: Creates an infinity repeatable tone and transmits it out of the radio

* airt_record.py: Interact with the radio drivers, record N samples, save to a file and plot the saved signal

* detect_and_repeat.py: Create a GPU power detector using CuPy, start a transmit task that sends any signal data array that is passed in to the AIR-T's RF transmitter, start the transceiver and continuously receive signal data, and repeat detected signals by sending them to the transmit task. 
    * **NOTE**: should be run using airstack-infer conda environment