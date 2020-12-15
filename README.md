# ImageFilterRL
## Installing
Project requires `python 3.8`. Other versions will not work.

To install my project, please create a virtualenv and activate it.
Then, run `pip install -r requirements.txt` to install project dependencies.
This project uses my image filtering library `libimg` as well as a RL library I've built atop pytorch, called `librl`.
Lastly, run `pip install -e .` to build my project.

## Training
To train a model, run the program `python tools/run.py --log-dir "some-directory"`.
The log directory must not be an already existing folder.
This will begin the training routine. 
Accuracy statistics will be logged to the console, as well as rewards and filter configurations.
Training takes ~1 day on a 32 core machine. 
You may adjust the number of adaptation steps, epochs, and timesteps to speed training time, but this will drastically degrade classifier performance since  requires many thousands of samples to learn efficiently.

Information to generate graphs is recorded in the log directory.
Graphing functioanlity is not implemented, since my focus was on speeding up training time rather than visualization.
