# How to create a simple Processor

If the chosen environment doesn't have a simple processor to run it can be added this way:

By importing the processor `JupyterDemoProcessor.eops`
or
By creating a new processor, you can follow the screenshot present here and add the following `Dockerfile` and `workflow.sh` in the sources tab. Also input and output must correspond to the ones in the screenshot or can be changed by modifying the `workflow.sh` script.

This is possibly one of the simplest processor that can be run, it is simply concatenating two input strings and produces an output text file.

![](1_descriptor.png)
![](2_sources.png)
![](3_input.png)
![](4_output.png)
![](5_build.png)