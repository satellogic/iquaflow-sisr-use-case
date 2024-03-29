# iq-sisr-use-case

The Single Image Super Resolution (SISR) use case is build to compare the image quality between different SiSR solutions. A SiSR algorithm inputs one frame and outputs an image with greater resolution.
These are the methods that are being compared in the use case:


1. Fast Super-Resolution Convolutional Neural Network (FSRCNN)
2. Local Implicit Image Function (LIIF)
3. Multi-scale Residual Network (MSRN)
4. Enhanced Super-Resolution Generative Adversarial Network (ESRGAN)

A use case in IQF usally involves wrapping a training within mlflow framework. In this case we estimate quality on the solutions offered by the different Dataset Modifiers which are the SISR algorithms. Similarity metrics against the Ground Truth are then compared.

____________________________________________________________________________________________________


## To reproduce the experiments:

1. `git clone git@publicgitlab.satellogic.com:iqf/iq-sisr-use-case.git`
2. `cd iq-sisr-use-case`
3. Then build the docker image with `make build`.(\*\*\*) This will also download required datasets and weights.
4. In order to execute the experiments:
    - `make dockershell` (\*)
    - Inside the docker terminal execute `python ./iqf-usecase.py`
5. Start the mlflow server by doing `make mlflow` (\*)
6. Notebook examples can be launched and executed by `make notebookshell NB_PORT=[your_port]"` (\**)
7. To access the notebook from your browser in your local machine you can do:
    - If the executions are launched in a server, make a tunnel from your local machine. `ssh -N -f -L localhost:[your_port]:localhost:[your_port] [remote_user]@[remote_ip]`  Otherwise skip this step.
    - Then, in your browser, access: `localhost:[your_port]/?token=sisr`


____________________________________________________________________________________________________

## Notes

   - The results of the IQF experiment can be seen in the MLflow user interface.
   - For more information please check the IQF_expriment.ipynb or IQF_experiment.py.
   - There are also examples of dataset Sanity check and Stats in SateAirportsStats.ipynb
   - The default ports are `8888` for the notebookshell, `5000` for the mlflow and `9197` for the dockershell
   - (*)
        Additional optional arguments can be added. The dataset location is:
        >`DS_VOLUME=[path_to_your_dataset]`
   - To change the default port for the mlflow service:
     >`MLF_PORT=[your_port]`
   - (**)
        To change the default port for the notebook: 
        >`NB_PORT=[your_port]`
   - A terminal can also be launched by `make dockershell` with optional arguments such as (*)
   - (***)
        Depending on the version of your cuda drivers and your hardware you might need to change the version of pytorch which is in the Dockerfile where it says:
        >`pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html`.
   - (***)
        The dataset is downloaded with all the results of executing the dataset modifiers already generated. This allows the user to freely skip the `.execute` as well as the `apply_metric_per_run` which __take long time__. Optionally, you can remove the pre-executed records folder (`./mlruns `) for a fresh start.
